"""SQLite 스키마 마이그레이션 테스트.

**무엇을 지키려는 테스트인가.** 이 러너는 **이미 900만 행이 든 1.65GB DB** 위에서 돈다.
중간에 실패했을 때 *"표는 바뀌었는데 버전은 그대로"* 인 반쪽 상태가 남으면, 다음 실행이
같은 문장을 또 돌려 조용히 어긋난다. 그래서 여기서 잠그는 것은 기능이 아니라 **원자성**이다.

    수용 기준
    - 밀린 것만 순서대로 적용하고, 두 번 돌려도 안전하다
    - 이 러너가 생기기 전에 만들어진 DB(user_version=0 인데 표는 이미 있음)에도 안전하다
    - **실패하면 스키마와 버전이 함께 되돌아간다** ← 이게 핵심이다
    - autocommit 이 아닌 연결은 조용히 넘어가지 않고 거부한다

실제 파일 DB(tmp_path)로 돈다 — 원자성은 메모리 DB 로는 증명이 약하다.
"""

from __future__ import annotations

import sqlite3

import pytest

from ingest.store import migrations as mig

# ── 도구 ───────────────────────────────────────────────────────────────────

def 연결(tmp_path, 이름: str = "t.db") -> sqlite3.Connection:
    """마이그레이션용 연결 하나."""
    return mig.connect_for_migration(tmp_path / 이름)


def 표목록(conn: sqlite3.Connection) -> set:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {r[0] for r in rows}


# ── 기본 동작 ───────────────────────────────────────────────────────────────

def test_빈_DB_를_최신으로_올린다(tmp_path):
    conn = 연결(tmp_path)
    try:
        assert mig.user_version(conn) == 0

        적용수 = mig.migrate(conn)

        assert 적용수 == mig.LATEST_VERSION
        assert mig.user_version(conn) == mig.LATEST_VERSION
        # 수집 대장과 호출 예산 표가 실제로 생겼는가
        assert {"collect_log", "call_budget"} <= 표목록(conn)
    finally:
        conn.close()


def test_두_번_돌려도_안전하다(tmp_path):
    """배치가 매번 부를 것이므로 재실행이 공짜여야 한다."""
    conn = 연결(tmp_path)
    try:
        mig.migrate(conn)
        표_1회차 = 표목록(conn)

        적용수 = mig.migrate(conn)          # 두 번째

        assert 적용수 == 0, "이미 최신인데 무언가를 또 적용했다"
        assert mig.user_version(conn) == mig.LATEST_VERSION
        assert 표목록(conn) == 표_1회차
    finally:
        conn.close()


def test_이미_표가_있는_옛_DB_도_안전하다(tmp_path):
    """이 러너가 생기기 전 DB 는 user_version=0 인데 기본 표는 갖고 있다.

    새 DB 와 구별되지 않으므로, 마이그레이션 문장은 전부 여러 번 돌려도 같은 결과여야 한다.
    """
    conn = 연결(tmp_path)
    try:
        # 옛 DB 흉내 — krx_store.init_db() 가 만드는 표를 미리 넣고 버전은 0 으로 둔다
        conn.execute("CREATE TABLE daily_price (bas_dd TEXT, code TEXT, "
                     "PRIMARY KEY (bas_dd, code))")
        conn.execute("INSERT INTO daily_price VALUES ('20260826', '005930')")
        # collect_log 가 이미 있는 상황까지 흉내 낸다 (중단 후 재실행)
        conn.execute("CREATE TABLE IF NOT EXISTS collect_log ("
                     "source TEXT NOT NULL, target TEXT NOT NULL, status TEXT NOT NULL, "
                     "rows INTEGER NOT NULL DEFAULT 0, last_success_at TEXT, "
                     "last_attempted_at TEXT NOT NULL, cursor TEXT, note TEXT, "
                     "PRIMARY KEY (source, target))")
        assert mig.user_version(conn) == 0

        mig.migrate(conn)

        assert mig.user_version(conn) == mig.LATEST_VERSION
        # 기존 자료가 살아 있는가 — 마이그레이션이 행을 건드리면 안 된다
        assert conn.execute("SELECT COUNT(*) FROM daily_price").fetchone()[0] == 1
    finally:
        conn.close()


# ── 원자성 — 이 파일의 핵심 ──────────────────────────────────────────────────

def test_실패하면_스키마와_버전이_함께_되돌아간다(tmp_path, monkeypatch):
    """규약 ①③ 을 잠근다.

    두 번째 문장에서 터지는 마이그레이션을 넣고, 첫 문장이 만든 표까지 사라지는지 본다.
    `PRAGMA user_version` 이 트랜잭션과 함께 되돌아가는지도 여기서 확인한다 —
    *"헤더 값이라 트랜잭션 밖"* 이라는 통설이 맞다면 버전만 올라간 채 남을 것이다.
    """
    monkeypatch.setattr(mig, "MIGRATIONS", (
        ("v1: 일부러 실패한다", (
            "CREATE TABLE IF NOT EXISTS 먼저_만든_표 (a INTEGER)",
            "이건 SQL 이 아니다",                       # ← 여기서 터진다
        )),
    ))
    monkeypatch.setattr(mig, "LATEST_VERSION", 1)

    conn = 연결(tmp_path)
    try:
        with pytest.raises(mig.MigrationError) as 오류:
            mig.migrate(conn)

        # 무엇을 해야 하는지까지 알려 주는가 (막다른 길로 만들지 않는다)
        assert "할 일" in str(오류.value)

        assert mig.user_version(conn) == 0, "실패했는데 버전이 올라갔다 — 반쪽 상태다"
        assert "먼저_만든_표" not in 표목록(conn), "실패했는데 표가 남았다 — 롤백이 안 됐다"
    finally:
        conn.close()


def test_실패한_마이그레이션_뒤의_것은_적용되지_않는다(tmp_path, monkeypatch):
    """v1 이 실패했는데 v2 가 얹히면 스키마가 뒤엉킨다."""
    monkeypatch.setattr(mig, "MIGRATIONS", (
        ("v1: 실패", ("이건 SQL 이 아니다",)),
        ("v2: 여기까지 오면 안 된다", ("CREATE TABLE IF NOT EXISTS 오면_안_되는_표 (a INT)",)),
    ))
    monkeypatch.setattr(mig, "LATEST_VERSION", 2)

    conn = 연결(tmp_path)
    try:
        with pytest.raises(mig.MigrationError):
            mig.migrate(conn)

        assert mig.user_version(conn) == 0
        assert "오면_안_되는_표" not in 표목록(conn)
    finally:
        conn.close()


# ── 가드 ───────────────────────────────────────────────────────────────────

def test_autocommit_이_아니면_거부한다(tmp_path):
    """규약 ① — 기본 모드로 열면 BEGIN 이 안 먹어 원자성이 깨진다.

    조용히 넘어가면 *"롤백되겠지"* 라고 믿게 되므로, 시작 전에 막는다.
    """
    conn = sqlite3.connect(tmp_path / "t.db")      # 기본 isolation_level
    try:
        with pytest.raises(mig.MigrationError) as 오류:
            mig.migrate(conn)
        assert "autocommit" in str(오류.value)
        assert "할 일" in str(오류.value)
    finally:
        conn.close()


def test_DB_가_코드보다_최신이면_거부한다(tmp_path):
    """git pull 을 안 한 팀원이 옛 코드로 새 DB 를 열면, 모르는 표를 쓰다 조용히 틀린다."""
    conn = 연결(tmp_path)
    try:
        conn.execute(f"PRAGMA user_version={mig.LATEST_VERSION + 5}")

        with pytest.raises(mig.MigrationError) as 오류:
            mig.migrate(conn)
        assert "git pull" in str(오류.value)
    finally:
        conn.close()


# ── 스키마가 실제로 규칙을 강제하는가 ────────────────────────────────────────

def test_collect_log_는_모르는_상태값을_거부한다(tmp_path):
    """0건(empty)·한도소진(quota_exhausted)·범위밖(out_of_range)을 실패와 섞지 않기 위한 어휘다.

    "받아 봤더니 없었다"·"한도에 닿아 아껴 멈췄다"·"출처가 제공하지 않는 구간이다" 는
    전부 실패가 아니다. 오타로 'OK' 나 'done' 이 들어가면 수집 현황 화면과 품질 검사가
    이것들을 조용히 세지 못하고 넘어간다.
    """
    conn = 연결(tmp_path)
    try:
        mig.migrate(conn)

        for 상태 in ("ok", "empty", "error", "quota_exhausted", "out_of_range"):
            conn.execute(
                "INSERT INTO collect_log (source, target, status, last_attempted_at) "
                "VALUES (?,?,?,?)", ("krx", f"t-{상태}", 상태, "2026-08-26T13:00:00+09:00"))

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO collect_log (source, target, status, last_attempted_at) "
                "VALUES (?,?,?,?)", ("krx", "bad", "done", "2026-08-26T13:00:00+09:00"))
    finally:
        conn.close()


def test_call_budget_은_출처와_날짜로_한_행이다(tmp_path):
    """`kst_date` 가 PK 에 있어야 자정 리셋 로직 없이도 날짜가 바뀌면 0 부터 센다."""
    conn = 연결(tmp_path)
    try:
        mig.migrate(conn)
        conn.execute("INSERT INTO call_budget (source, kst_date, used, daily_limit) "
                     "VALUES ('krx','2026-08-26', 10, 10000)")
        # 날짜가 다르면 다른 행이다 — 리셋이 저절로 된다
        conn.execute("INSERT INTO call_budget (source, kst_date, used, daily_limit) "
                     "VALUES ('krx','2026-08-27', 0, 10000)")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO call_budget (source, kst_date, used, daily_limit) "
                         "VALUES ('krx','2026-08-26', 999, 10000)")
    finally:
        conn.close()


# ── ADD COLUMN 도구 — "이 행을 언제부터 알 수 있었나" 칸을 얹을 때 쓴다 ──────

def test_add_column_sql_은_이미_있는_칸에_None_을_준다(tmp_path):
    """`ADD COLUMN` 은 `IF NOT EXISTS` 가 없어 두 번째 실행에서 터진다. 미리 걸러야 한다."""
    conn = 연결(tmp_path)
    try:
        conn.execute("CREATE TABLE t (a INTEGER)")

        문장 = mig.add_column_sql(conn, "t", "known_at", "TEXT")
        assert 문장 == "ALTER TABLE t ADD COLUMN known_at TEXT"
        conn.execute(문장)

        assert mig.add_column_sql(conn, "t", "known_at", "TEXT") is None
    finally:
        conn.close()


def test_표_이름에_이상한_것이_오면_거부한다(tmp_path):
    """PRAGMA 는 바인딩을 못 받아 이름을 문자열로 끼워 넣는다 — 주입 경로를 막는다."""
    conn = 연결(tmp_path)
    try:
        with pytest.raises(mig.MigrationError):
            mig.column_names(conn, "t; DROP TABLE daily_price")
    finally:
        conn.close()
