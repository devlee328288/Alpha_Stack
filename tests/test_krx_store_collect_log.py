"""종목 시세 저장과 수집 대장의 배선 테스트.

**무엇을 지키려는 테스트인가.** 시세를 매일 받고 있는데도 화면의 "마지막 성공 시각"이
2026-08-26 에 얼어붙어 있었다. `_save` 가 옛 표(`fetch_log`)에만 쓰고 화면이 읽는
수집 대장(`collect_log`)에는 쓰지 않았기 때문이다. **예외도 경고도 없었다** — 수집은
멀쩡히 돌고 화면만 조용히 거짓말을 했다.

    수용 기준
    - 저장이 대장을 **시장별로** 갱신한다 (KRX 호출이 시장마다 따로다)
    - 0행인 시장은 `empty` — 휴장일을 실패로 적으면 매번 같은 호출을 태운다
    - 적재와 대장이 **같은 트랜잭션**에서 함께 되돌아간다
    - 받다가 실패한 시장이 대장에 남는다 (예외가 올라가면 어느 시장인지 알 수 없다)
    - 한도 소진은 실패로 세지 않는다
    - 옛 날짜 전용 줄(`20260826`)이 시장별(`KOSPI/20260826`)로 다시 깔린다

진짜 파일 DB(tmp_path)로 돈다 — 트랜잭션 원자성은 메모리 DB 로는 증명이 약하다.
"""

from __future__ import annotations

import sqlite3

import pytest

from ingest.clients import krx_data as api
from ingest.store import collect_log

# ── 도구 ───────────────────────────────────────────────────────────────────


@pytest.fixture
def 임시저장소(tmp_path, monkeypatch):
    """진짜 DB(1.5GB)를 건드리지 않도록 임시 파일로 갈아 끼운다."""
    from ingest.store import krx_store

    db = tmp_path / "test_store.db"
    monkeypatch.setattr(krx_store, "DB_PATH", db)
    krx_store.init_db()
    return krx_store


def _행(code: str, market: str, close: int = 70000) -> dict:
    """`_save` 가 받는 정규화된 한 종목 행."""
    return {"code": code, "name": f"종목{code}", "market": market, "sector": "제조",
            "open": close, "high": close, "low": close, "close": close,
            "change": 0, "change_rate": 0.0,
            "volume": 1000, "value": close * 1000,
            "market_cap": close * 100000, "listed_shares": 100000}


def _옛_대장_깔기(db, rows):
    """마이그레이션 전 저장소가 쓰던 `fetch_log` 와 날짜 전용 대장 줄을 흉내 낸다."""
    conn = sqlite3.connect(db, timeout=60, isolation_level=None)
    try:
        conn.executemany(
            "INSERT OR REPLACE INTO fetch_log (bas_dd, rows, fetched_at) VALUES (?,?,?)",
            rows,
        )
    finally:
        conn.close()
    # `import_legacy` 가 옛날에 남긴 모양 — 시장 없이 날짜만.
    for bas_dd, count, _ in rows:
        status = collect_log.OK if count else collect_log.EMPTY
        collect_log.record("krx_stock", bas_dd, status, rows=count,
                           note="fetch_log 에서 넘겨받음", db_path=db)


# ==============================================================
# 1. 저장이 대장을 갱신한다
# ==============================================================
def test_저장이_대장을_시장별로_갱신한다(임시저장소):
    """이게 없어서 화면의 마지막 성공 시각이 얼어붙어 있었다."""
    store = 임시저장소
    db = store.DB_PATH

    store._save("20260827", [_행("005930", "KOSPI"), _행("000660", "KOSPI"),
                             _행("035720", "KOSDAQ")])

    코스피 = collect_log.entry("krx_stock", "KOSPI/20260827", db_path=db)
    코스닥 = collect_log.entry("krx_stock", "KOSDAQ/20260827", db_path=db)
    assert 코스피["status"] == collect_log.OK
    assert 코스피["rows"] == 2
    assert 코스닥["rows"] == 1
    # 시각이 실제로 찍혀야 한다 — 이 값이 화면의 "마지막 성공 시각"이다
    assert 코스피["last_success_at"]


def test_대상_이름이_지수_쪽과_같은_규칙이다(임시저장소):
    """한 표 안에 `20260827` 과 `KOSPI/20260827` 이 섞이면 같은 날짜가 두 벌이 된다."""
    store = 임시저장소
    db = store.DB_PATH

    store._save("20260827", [_행("005930", "KOSPI"), _행("035720", "KOSDAQ")])

    assert collect_log.entry("krx_stock", "20260827", db_path=db) is None, \
        "날짜만 있는 옛 형식으로 쓰면 지수 쪽과 규칙이 갈린다"
    assert collect_log.entry("krx_stock", "KOSPI/20260827", db_path=db) is not None
    assert collect_log.entry("krx_stock", "KOSDAQ/20260827", db_path=db) is not None
    assert collect_log.summary("krx_stock", db_path=db)["krx_stock"]["targets"] == 2


def test_행이_없는_시장은_휴장으로_남는다(임시저장소):
    """`MARKETS` 를 기준으로 돌지 않으면 0행인 시장이 대장에서 통째로 빠진다.

    빠지면 그 시장은 영영 미수집으로 보여 매번 호출을 태운다.
    """
    store = 임시저장소
    db = store.DB_PATH

    store._save("20260827", [_행("005930", "KOSPI")])   # 코스닥은 한 행도 없다

    코스닥 = collect_log.entry("krx_stock", "KOSDAQ/20260827", db_path=db)
    assert 코스닥 is not None, "0행인 시장이 대장에서 빠졌다"
    assert 코스닥["status"] == collect_log.EMPTY
    assert 코스닥["rows"] == 0
    # 휴장일은 영원히 0행이다 — 다시 묻지 않아야 한다
    assert collect_log.should_collect("krx_stock", "KOSDAQ/20260827", db_path=db) is False


def test_휴장일은_두_시장_다_휴장이다(임시저장소):
    store = 임시저장소
    db = store.DB_PATH

    store._save("20260815", [])

    for market in ("KOSPI", "KOSDAQ"):
        행 = collect_log.entry("krx_stock", f"{market}/20260815", db_path=db)
        assert 행["status"] == collect_log.EMPTY


def test_저장이_되돌아가면_대장도_되돌아간다(임시저장소, monkeypatch):
    """적재만 롤백되고 대장이 남으면 그 날짜를 영영 다시 안 받는다."""
    store = 임시저장소
    # ⚠️ 경로를 **미리** 잡아 둔다. `monkeypatch.undo()` 는 fixture 가 건 DB_PATH
    #    교체까지 함께 되돌려, 그 뒤 검증이 진짜 DB 를 읽는다.
    db = store.DB_PATH

    원래 = collect_log.mark_ok

    def 쓰고_터뜨린다(*args, **kwargs):
        원래(*args, **kwargs)
        raise RuntimeError("적재 도중 죽었다")

    monkeypatch.setattr(collect_log, "mark_ok", 쓰고_터뜨린다)
    with pytest.raises(RuntimeError):
        store._save("20260827", [_행("005930", "KOSPI")])

    monkeypatch.setattr(collect_log, "mark_ok", 원래)
    assert collect_log.entry("krx_stock", "KOSPI/20260827", db_path=db) is None
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM daily_price").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM fetch_log").fetchone()[0] == 0


# ==============================================================
# 2. 실패를 남긴다
# ==============================================================
def test_받다_실패한_시장이_대장에_남는다(임시저장소, monkeypatch):
    """예외가 올라가고 나면 어느 시장이 왜 실패했는지 알 수 없다."""
    store = 임시저장소
    db = store.DB_PATH

    def 코스닥만_터진다(bas_dd, market):
        if market == "KOSDAQ":
            raise RuntimeError("연결이 끊겼다")
        return [_행("005930", "KOSPI")]

    monkeypatch.setattr(api, "fetch_snapshot", 코스닥만_터진다)
    with pytest.raises(RuntimeError):
        store.fetch_date("20260827")

    행 = collect_log.entry("krx_stock", "KOSDAQ/20260827", db_path=db)
    assert 행["status"] == collect_log.ERROR
    assert "연결이 끊겼다" in 행["note"]
    # 앞 시장은 저장이 통째로 롤백됐으므로 성공으로 남으면 안 된다
    assert collect_log.entry("krx_stock", "KOSPI/20260827", db_path=db) is None


def test_한도_소진은_실패로_세지_않는다(임시저장소, monkeypatch):
    """한도가 세 번 마르는 동안 멀쩡한 날짜가 영영 버려지면 안 된다."""
    store = 임시저장소
    db = store.DB_PATH

    def 한도소진(bas_dd, market):
        raise api.KrxQuotaExhausted("하루 10,000회를 다 썼다")

    monkeypatch.setattr(api, "fetch_snapshot", 한도소진)
    with pytest.raises(api.KrxQuotaExhausted):
        store.fetch_date("20260827")

    행 = collect_log.entry("krx_stock", "KOSPI/20260827", db_path=db)
    assert 행["status"] == collect_log.QUOTA_EXHAUSTED
    assert 행["attempts"] == 0, "예산이 마른 것은 이 날짜의 잘못이 아니다"
    # 내일 다시 받아야 한다
    assert collect_log.should_collect("krx_stock", "KOSPI/20260827", db_path=db) is True


# ==============================================================
# 3. 옛 날짜 전용 줄을 시장별로 다시 깐다
# ==============================================================
def test_옛_날짜_전용_줄을_시장별로_바꾼다(임시저장소):
    """옛 `fetch_log` 에는 시장 칸이 없어 대장이 `20260826` 으로 남아 있었다."""
    store = 임시저장소
    db = store.DB_PATH

    store._save("20260826", [_행("005930", "KOSPI"), _행("000660", "KOSPI"),
                             _행("035720", "KOSDAQ")])
    # 저장이 남긴 시장별 줄을 지우고, 옛 모양(날짜만)으로 되돌려 놓는다
    with sqlite3.connect(db) as conn:
        conn.execute("DELETE FROM collect_log WHERE source='krx_stock'")
    _옛_대장_깔기(db, [("20260826", 3, "2026-08-26T11:37:48")])

    결과 = store.rebuild_collect_log()

    assert 결과["removed"] == 1
    assert collect_log.entry("krx_stock", "20260826", db_path=db) is None
    # 건수는 `daily_price` 를 실제로 세어서 나온다 — 지어내지 않는다
    assert collect_log.entry("krx_stock", "KOSPI/20260826", db_path=db)["rows"] == 2
    assert collect_log.entry("krx_stock", "KOSDAQ/20260826", db_path=db)["rows"] == 1
    # 시각은 옛 대장의 것을 그대로 물려받는다
    assert collect_log.entry("krx_stock", "KOSPI/20260826",
                             db_path=db)["last_success_at"] == "2026-08-26T11:37:48"


def test_옛_휴장일도_두_시장으로_퍼진다(임시저장소):
    store = 임시저장소
    db = store.DB_PATH
    _옛_대장_깔기(db, [("20260815", 0, "2026-08-15T18:00:00")])

    store.rebuild_collect_log()

    for market in ("KOSPI", "KOSDAQ"):
        행 = collect_log.entry("krx_stock", f"{market}/20260815", db_path=db)
        assert 행["status"] == collect_log.EMPTY
        assert 행["rows"] == 0


def test_다시_깔기가_새_상태를_덮지_않는다(임시저장소):
    """방금 고친 실패가 옛 값으로 되살아나면 안 된다."""
    store = 임시저장소
    db = store.DB_PATH

    store._save("20260826", [_행("005930", "KOSPI"), _행("035720", "KOSDAQ")])
    _옛_대장_깔기(db, [("20260826", 2, "2026-08-26T11:37:48")])
    collect_log.mark_error("krx_stock", "KOSPI/20260826",
                           note="다시 받아야 한다", db_path=db)

    store.rebuild_collect_log()

    assert collect_log.entry("krx_stock", "KOSPI/20260826",
                             db_path=db)["status"] == collect_log.ERROR


def test_다시_깔기를_두_번_해도_안전하다(임시저장소):
    store = 임시저장소
    db = store.DB_PATH

    store._save("20260826", [_행("005930", "KOSPI"), _행("035720", "KOSDAQ")])
    _옛_대장_깔기(db, [("20260826", 2, "2026-08-26T11:37:48")])

    첫번째 = store.rebuild_collect_log()
    두번째 = store.rebuild_collect_log()

    assert 두번째["removed"] == 0, "지울 날짜 전용 줄이 더 없어야 한다"
    assert 첫번째["after"] == 두번째["after"]
    assert 두번째["after"] == 2


def test_옛_대장이_비어도_넘어간다(임시저장소):
    """새로 clone 한 사람의 DB 에는 받은 이력이 아예 없다."""
    store = 임시저장소

    결과 = store.rebuild_collect_log()

    assert 결과 == {"before": 0, "removed": 0, "built": 0, "after": 0}
