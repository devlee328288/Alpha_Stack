"""수집 대장 테스트.

**무엇을 지키려는 테스트인가.** 이 표가 답해야 하는 질문은 하나다 —
*"이 대상을 지금 또 받아야 하는가."* 그 답이 틀리면 두 방향으로 손해가 난다.

    너무 자주 받는다  →  휴장일마다 영원히 같은 호출을 태운다 (한도가 마른다)
    안 받는다        →  진짜 실패가 조용히 묻힌다 (구멍 난 채로 학습에 들어간다)

그래서 여기서 잠그는 것은 저장 기능이 아니라 **상태의 구별**이다.

    수용 기준
    - `0건`(휴장일)과 `미수집`이 다르게 취급된다 ← 이게 핵심이다
    - 한도 소진은 실패로 세지 않는다 — 재시도 횟수를 먹지 않는다
    - 실패는 정해진 횟수까지만 다시 시도한다
    - 성공 시각은 나중에 실패해도 지워지지 않는다
    - **적재와 대장이 같은 트랜잭션에서 함께 되돌아간다** ← 어긋남을 막는 유일한 수단

실제 파일 DB(tmp_path)로 돈다 — 트랜잭션 원자성은 메모리 DB 로는 증명이 약하다.
"""

from __future__ import annotations

import sqlite3

import pytest

from ingest.store import collect_log as cl

# ── 도구 ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def db(tmp_path):
    """빈 DB 경로 하나. 첫 호출에서 마이그레이션이 돈다."""
    return tmp_path / "t.db"


def 연결(db) -> sqlite3.Connection:
    conn = sqlite3.connect(db, timeout=60, isolation_level=None)
    conn.row_factory = sqlite3.Row
    return conn


# ── 0건과 미수집의 구별 ──────────────────────────────────────────────────────

def test_안_받은_것은_받아야_한다(db):
    assert cl.entry("krx", "20260826", db_path=db) is None
    assert cl.should_collect("krx", "20260826", db_path=db) is True


def test_0건은_미수집과_다르다(db):
    """휴장일이다. 다시 물어봐야 아무것도 없으므로 받지 않는다."""
    cl.mark_empty("krx", "20260815", note="광복절", db_path=db)

    행 = cl.entry("krx", "20260815", db_path=db)
    assert 행 is not None                      # 안 받은 것과 구별된다
    assert 행["status"] == cl.EMPTY
    assert 행["rows"] == 0
    assert cl.should_collect("krx", "20260815", db_path=db) is False


def test_받은_것은_다시_받지_않는다(db):
    cl.mark_ok("krx", "20260826", rows=1961, db_path=db)

    assert cl.should_collect("krx", "20260826", db_path=db) is False
    assert cl.entry("krx", "20260826", db_path=db)["rows"] == 1961


def test_제공하지_않는_구간은_다시_묻지_않는다(db):
    """KRX 는 2010-01-04 이전 지수를 오류가 아니라 0행으로 준다 — 휴장일과 섞이면 안 된다."""
    cl.mark_out_of_range("krx_index", "20091230", note="제공 시작일 이전", db_path=db)

    assert cl.should_collect("krx_index", "20091230", db_path=db) is False
    assert cl.entry("krx_index", "20091230", db_path=db)["status"] == cl.OUT_OF_RANGE


def test_최근_0건은_다시_열어_준다(db):
    """장 마감 전에 받은 0건을 확정 휴장일로 굳히면 그날 자료를 영영 놓친다."""
    cl.mark_empty("krx", "20260826", db_path=db)

    assert cl.should_collect("krx", "20260826", db_path=db) is False
    assert cl.should_collect("krx", "20260826", empty_recheck_days=7, db_path=db) is True


# ── 실패와 한도 소진의 구별 ──────────────────────────────────────────────────

def test_실패는_정해진_횟수까지만_다시_받는다(db):
    for _ in range(cl.DEFAULT_MAX_ATTEMPTS - 1):
        cl.mark_error("krx", "20260826", note="타임아웃", db_path=db)
        assert cl.should_collect("krx", "20260826", db_path=db) is True

    cl.mark_error("krx", "20260826", note="타임아웃", db_path=db)

    행 = cl.entry("krx", "20260826", db_path=db)
    assert 행["attempts"] == cl.DEFAULT_MAX_ATTEMPTS
    assert cl.should_collect("krx", "20260826", db_path=db) is False


def test_한도_소진은_실패로_세지_않는다(db):
    """예산이 마른 것은 대상의 잘못이 아니다. 세 번 마르면 멀쩡한 날짜를 포기하게 된다."""
    for _ in range(cl.DEFAULT_MAX_ATTEMPTS + 2):
        cl.mark_quota_exhausted("krx", "20260826", db_path=db)

    행 = cl.entry("krx", "20260826", db_path=db)
    assert 행["attempts"] == 0                  # 재시도 횟수를 먹지 않았다
    assert cl.should_collect("krx", "20260826", db_path=db) is True


def test_한도_소진이_실패_이력을_지우지_않는다(db):
    """실패 2회 뒤 예산이 마르면, 예산이 풀린 뒤에도 남은 재시도는 1회여야 한다."""
    cl.mark_error("krx", "20260826", note="1", db_path=db)
    cl.mark_error("krx", "20260826", note="2", db_path=db)
    cl.mark_quota_exhausted("krx", "20260826", db_path=db)

    assert cl.entry("krx", "20260826", db_path=db)["attempts"] == 2


def test_성공하면_실패_이력이_사라진다(db):
    cl.mark_error("krx", "20260826", note="타임아웃", db_path=db)
    cl.mark_error("krx", "20260826", note="타임아웃", db_path=db)
    cl.mark_ok("krx", "20260826", rows=10, db_path=db)

    assert cl.entry("krx", "20260826", db_path=db)["attempts"] == 0


def test_성공_시각은_나중에_실패해도_남는다(db):
    """얼마나 오래 막혀 있는지 알려면 마지막 성공 시각을 잃으면 안 된다."""
    cl.mark_ok("krx", "20260826", rows=10, db_path=db)
    성공시각 = cl.entry("krx", "20260826", db_path=db)["last_success_at"]
    assert 성공시각

    cl.mark_error("krx", "20260826", note="그 뒤에 실패", db_path=db)

    행 = cl.entry("krx", "20260826", db_path=db)
    assert 행["status"] == cl.ERROR
    assert 행["last_success_at"] == 성공시각     # 덮이지 않았다
    assert 행["last_attempted_at"] >= 성공시각


# ── 적재와 같은 트랜잭션 ─────────────────────────────────────────────────────

def test_같은_트랜잭션에서_함께_되돌아간다(db):
    """대장만 따로 커밋하면 *"저장은 됐는데 대장에는 없는"* 어긋난 상태가 남는다."""
    cl._ensure_schema(db)
    conn = 연결(db)
    try:
        conn.execute("CREATE TABLE 적재 (k TEXT PRIMARY KEY)")

        conn.execute("BEGIN IMMEDIATE")
        conn.execute("INSERT INTO 적재 VALUES ('20260826')")
        cl.mark_ok("krx", "20260826", rows=1, conn=conn)
        conn.execute("ROLLBACK")

        assert conn.execute("SELECT COUNT(*) FROM 적재").fetchone()[0] == 0
    finally:
        conn.close()

    # 적재가 되돌아갔으면 대장도 되돌아가야 한다 — 안 그러면 영영 다시 안 받는다
    assert cl.entry("krx", "20260826", db_path=db) is None


def test_같은_트랜잭션에서_함께_커밋된다(db):
    cl._ensure_schema(db)
    conn = 연결(db)
    try:
        conn.execute("BEGIN IMMEDIATE")
        cl.mark_ok("krx", "20260826", rows=7, conn=conn)
        conn.execute("COMMIT")
    finally:
        conn.close()

    assert cl.entry("krx", "20260826", db_path=db)["rows"] == 7


# ── 방어 ────────────────────────────────────────────────────────────────────

def test_모르는_상태값은_거부한다(db):
    """표의 CHECK 가 터지기 전에 여기서 잡아야 어디가 틀렸는지 알 수 있다."""
    with pytest.raises(cl.UnknownStatus) as 오류:
        cl.record("krx", "20260826", "성공", db_path=db)

    assert "쓸 수 있는 값" in str(오류.value)     # 무엇을 해야 하는지까지 알려 준다


def test_상태값_목록이_표의_제약과_같다(db):
    """한쪽만 고치면 INSERT 가 런타임에 터진다. 목록이 갈라지지 않게 잠근다."""
    cl._ensure_schema(db)
    conn = 연결(db)
    try:
        for status in cl.STATUSES:
            conn.execute(
                "INSERT OR REPLACE INTO collect_log "
                "(source, target, status, last_attempted_at) VALUES (?,?,?,'now')",
                ("s", status, status),
            )
    finally:
        conn.close()


# ── 목록 · 요약 ─────────────────────────────────────────────────────────────

def test_받아야_할_것만_순서를_지켜_돌려준다(db):
    cl.mark_ok("krx", "20260824", rows=5, db_path=db)
    cl.mark_empty("krx", "20260825", db_path=db)
    cl.mark_error("krx", "20260826", note="타임아웃", db_path=db)

    남은것 = cl.pending("krx", ["20260824", "20260825", "20260826", "20260827"], db_path=db)

    # 받은 것·0건은 빠지고, 실패한 것과 처음 보는 것만 남는다. 순서는 준 그대로다.
    assert 남은것 == ["20260826", "20260827"]


def test_재시도를_다_쓴_것은_목록에서_빠진다(db):
    for _ in range(cl.DEFAULT_MAX_ATTEMPTS):
        cl.mark_error("krx", "20260826", note="타임아웃", db_path=db)

    assert cl.pending("krx", ["20260826"], db_path=db) == []
    # 대신 사람이 볼 목록에 올라온다
    막힌것 = cl.stuck("krx", db_path=db)
    assert [row["target"] for row in 막힌것] == ["20260826"]


def test_요약이_상태별로_센다(db):
    cl.mark_ok("krx", "20260824", rows=5, db_path=db)
    cl.mark_ok("krx", "20260825", rows=3, db_path=db)
    cl.mark_empty("krx", "20260815", db_path=db)
    for _ in range(cl.DEFAULT_MAX_ATTEMPTS):
        cl.mark_error("krx", "20260826", note="타임아웃", db_path=db)

    요약 = cl.summary(db_path=db)["krx"]

    assert 요약[cl.OK] == 2
    assert 요약[cl.EMPTY] == 1
    assert 요약[cl.ERROR] == 1
    assert 요약["rows"] == 8
    assert 요약["targets"] == 4
    assert 요약["stuck"] == 1                    # 사람이 봐야 할 건수
    assert 요약["last_success_at"]


def test_출처가_섞이지_않는다(db):
    cl.mark_ok("krx", "20260826", rows=5, db_path=db)
    cl.mark_ok("dart", "20260826", rows=3, db_path=db)

    요약 = cl.summary(db_path=db)

    assert 요약["krx"]["rows"] == 5
    assert 요약["dart"]["rows"] == 3
    # 같은 target 이라도 출처가 다르면 다른 줄이다
    assert cl.pending("naver", ["20260826"], db_path=db) == ["20260826"]


def test_리포트를_떨군다(db, tmp_path):
    cl.mark_ok("krx", "20260826", rows=5, db_path=db)

    경로 = cl.write_report(tmp_path / "collect_status.json", db_path=db)

    import json
    내용 = json.loads(경로.read_text(encoding="utf-8"))
    assert 내용["sources"]["krx"]["rows"] == 5
    assert 내용["generated_at_kst"]


# ── 옛 대장 넘겨받기 ─────────────────────────────────────────────────────────

def _옛표_만들기(db):
    """마이그레이션 전 저장소가 쓰던 두 표를 흉내 낸다."""
    cl._ensure_schema(db)
    conn = 연결(db)
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS fetch_log "
                     "(bas_dd TEXT PRIMARY KEY, rows INTEGER, fetched_at TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS index_fetch_log "
                     "(bas_dd TEXT, market TEXT, rows INTEGER, fetched_at TEXT,"
                     " PRIMARY KEY (bas_dd, market))")
        conn.executemany("INSERT INTO fetch_log VALUES (?,?,?)", [
            ("20260824", 2700, "2026-08-24T18:00:00"),
            ("20260815", 0, "2026-08-15T18:00:00"),
        ])
        conn.executemany("INSERT INTO index_fetch_log VALUES (?,?,?,?)", [
            ("20260824", "KOSPI", 51, "2026-08-24T18:00:00"),
        ])
    finally:
        conn.close()


def test_옛_이력을_넘겨받는다(db):
    """옮기지 않으면 이미 받은 4,343 거래일이 미수집으로 보여 16년치를 다시 받는다."""
    _옛표_만들기(db)

    옮긴수 = cl.import_legacy(db_path=db)

    assert 옮긴수 == {"krx_stock": 2, "krx_index": 1}
    assert cl.should_collect("krx_stock", "20260824", db_path=db) is False
    assert cl.entry("krx_stock", "20260824", db_path=db)["rows"] == 2700
    # 0행은 실패가 아니라 휴장일로 넘어와야 한다
    assert cl.entry("krx_stock", "20260815", db_path=db)["status"] == cl.EMPTY
    # 지수는 시장이 대상 이름에 들어간다
    assert cl.entry("krx_index", "KOSPI/20260824", db_path=db)["rows"] == 51


def test_넘겨받기를_두_번_해도_안전하다(db):
    _옛표_만들기(db)
    cl.import_legacy(db_path=db)
    cl.import_legacy(db_path=db)

    assert cl.summary("krx_stock", db_path=db)["krx_stock"]["targets"] == 2


def test_넘겨받기가_새_상태를_덮지_않는다(db):
    """방금 고친 실패가 옛 값으로 되살아나면 안 된다."""
    _옛표_만들기(db)
    cl.mark_error("krx_stock", "20260824", note="다시 받아야 한다", db_path=db)

    cl.import_legacy(db_path=db)

    assert cl.entry("krx_stock", "20260824", db_path=db)["status"] == cl.ERROR


def test_옛_표가_없어도_넘어간다(db):
    """새로 clone 한 사람의 DB 에는 옛 표가 아예 없다."""
    assert cl.import_legacy(db_path=db) == {"krx_stock": 0, "krx_index": 0}
