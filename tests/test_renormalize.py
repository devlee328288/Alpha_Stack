"""재정규화 저장 경로 테스트

**왜 이 테스트가 필요한가.** 재정규화는 *"같은 원문을 다시 읽는 일"* 이지 *"자료를 새로
받는 일"* 이 아니다. 그런데 수집용 저장 함수를 그대로 재사용하면 **수집 시각이 오늘로
덮인다.** 그 시각은 *"우리가 이 사실을 언제부터 알 수 있었나"* 의 근거이고, 오늘로
옮겨지면 **미래참조 방지의 바닥이 무너진다.**

그리고 **에러는 나지 않는다.** 행 값은 그대로고 시각만 움직이므로, 자료를 눈으로 봐도
멀쩡해 보인다. 실제로 이 저장소에서 지수 재정규화가 그렇게 돌고 있었다.
"""

from __future__ import annotations

import sqlite3

import pytest

from ingest.store import collect_log, krx_index, krx_store


@pytest.fixture()
def 임시저장소(tmp_path, monkeypatch):
    """진짜 DB 를 건드리지 않도록 임시 파일로 갈아 끼운다."""
    db = tmp_path / "t.db"
    monkeypatch.setattr(krx_store, "DB_PATH", db)
    monkeypatch.setattr(krx_index, "DB_PATH", db)
    krx_store.init_db()
    krx_index.init_db()
    return db


def _지수행(이름: str = "코스피 200", close: float = 100.0) -> dict:
    return {"index_name": 이름, "index_class": "KOSPI",
            "open": close, "high": close, "low": close, "close": close,
            "change": 0.0, "change_rate": 0.0,
            "volume": 1, "value": 1, "market_cap": 1}


def _종목행(code: str = "005930", market: str = "KOSPI", close: int = 70000) -> dict:
    return {"code": code, "name": "삼성전자", "market": market, "sector": "전기전자",
            "open": close, "high": close, "low": close, "close": close,
            "change": 0, "change_rate": 0.0,
            "volume": 1, "value": 1, "market_cap": 1, "listed_shares": 1}


#: 우연히 같아질 수 없는 옛 시각. 초 단위 해상도라 "방금 쓴 값" 과 겹칠 수 있어
#: 일부러 눈에 띄는 과거를 심어 두고 그대로인지 본다.
옛시각 = "2020-01-02T09:00:00"


def _시각을_옛날로(db, 표: str, where: str, 값: tuple) -> None:
    """수집 시각을 옛날로 심는다. 재정규화가 이걸 덮으면 테스트가 잡는다."""
    with sqlite3.connect(db) as conn:
        conn.execute(f"UPDATE {표} SET fetched_at=? WHERE {where}", (옛시각, *값))


def _시각(db, 표: str, where: str, 값: tuple) -> str:
    with sqlite3.connect(db) as conn:
        row = conn.execute(f"SELECT fetched_at FROM {표} WHERE {where}", 값).fetchone()
    return row[0] if row else ""


# ── 지수 ────────────────────────────────────────────────────────────────────

def test_지수_재정규화는_수집_시각을_건드리지_않는다(임시저장소):
    """🔴 이게 이 파일이 존재하는 이유다.

    예전 판은 `_save` 를 그대로 불러서 `index_fetch_log.fetched_at` 과 수집 대장의
    `last_success_at` 을 **오늘로 덮었다.** 행 값은 그대로라 눈으로는 안 보인다.
    """
    krx_index._save("20260821", "KOSPI", [_지수행()])
    _시각을_옛날로(임시저장소, "index_fetch_log", "bas_dd=? AND market=?",
                   ("20260821", "KOSPI"))
    with sqlite3.connect(임시저장소) as conn:
        conn.execute("UPDATE collect_log SET last_success_at=? WHERE source=?",
                     (옛시각, "krx_index"))
    수집시각 = 대장시각 = 옛시각

    # 값이 달라진 재정규화 결과를 다시 쓴다
    krx_index.save_renormalized("20260821", "KOSPI", [_지수행(close=111.0)])

    assert _시각(임시저장소, "index_fetch_log", "bas_dd=? AND market=?",
                 ("20260821", "KOSPI")) == 수집시각, "수집 시각이 덮였다"
    assert collect_log.entry("krx_index", "KOSPI/20260821",
                             db_path=임시저장소)["last_success_at"] == 대장시각


def test_지수_재정규화는_값을_실제로_갱신한다(임시저장소):
    """시각을 안 건드리는 것과 값을 안 고치는 것은 다르다."""
    krx_index._save("20260821", "KOSPI", [_지수행(close=100.0)])

    krx_index.save_renormalized("20260821", "KOSPI", [_지수행(close=111.0)])

    assert krx_index.series("코스피 200")[0]["close"] == 111.0


# ── 종목 ────────────────────────────────────────────────────────────────────

def test_종목_재정규화는_수집_이력을_건드리지_않는다(임시저장소):
    """🔴 `_save` 를 재사용하면 `fetch_log.rows` 가 줄어드는데 아무도 모른다.

    `_save(bas_dd, items)` 는 시장 구분 없이 받아 `rows` 를 통째로 덮는다.
    재정규화는 원문을 **시장별로** 쥐고 있으므로 KOSPI 원문만으로 부르면 두 시장
    합계였던 값이 한 시장 값으로 줄어든다. 그런데 `fetched_dates()` 는 `rows > 0` 만
    보므로 **재수집이 일어나지 않아 대장이 거짓이 된 채로 남는다.**
    """
    krx_store._save("20260825", [_종목행("005930", "KOSPI"),
                                 _종목행("035720", "KOSDAQ")])
    _시각을_옛날로(임시저장소, "fetch_log", "bas_dd=?", ("20260825",))
    수집시각 = 옛시각
    with sqlite3.connect(임시저장소) as conn:
        원래건수 = conn.execute("SELECT rows FROM fetch_log WHERE bas_dd=?",
                                ("20260825",)).fetchone()[0]

    # KOSPI 원문만으로 재정규화한다 — 실제 상황이 이렇다
    krx_store.save_renormalized("20260825", "KOSPI", [_종목행("005930", "KOSPI", 88000)])

    with sqlite3.connect(임시저장소) as conn:
        지금건수 = conn.execute("SELECT rows FROM fetch_log WHERE bas_dd=?",
                                ("20260825",)).fetchone()[0]
    assert 지금건수 == 원래건수 == 2, "fetch_log.rows 가 한 시장 값으로 줄었다"
    assert _시각(임시저장소, "fetch_log", "bas_dd=?", ("20260825",)) == 수집시각


def test_종목_재정규화는_값을_실제로_갱신한다(임시저장소):
    krx_store._save("20260825", [_종목행("005930", "KOSPI", 70000)])

    krx_store.save_renormalized("20260825", "KOSPI", [_종목행("005930", "KOSPI", 88000)])

    assert krx_store.rows_for("20260825", "KOSPI")["005930"]["close"] == 88000


def test_지금_표를_읽을_때_시장_조건을_지킨다(임시저장소):
    """🔴 시장 조건을 빼면 KOSPI 원문을 재정규화하면서 KOSDAQ 행까지 읽는다.

    지수는 지수명이 전역에서 유일해 이 실수가 드러나지 않는다. **종목은 조용히 틀린다.**
    """
    krx_store._save("20260825", [_종목행("005930", "KOSPI"),
                                 _종목행("035720", "KOSDAQ")])

    assert set(krx_store.rows_for("20260825", "KOSPI")) == {"005930"}
    assert set(krx_store.rows_for("20260825", "KOSDAQ")) == {"035720"}


# ── 잘못 부르면 조용히 넘어가지 않는다 ──────────────────────────────────────

def test_다른_시장_행이_섞이면_쓰기_전에_멈춘다(임시저장소):
    """어긋난 채로 쓰면 KOSDAQ 행이 KOSPI 이름으로 굳어 버린다."""
    with pytest.raises(ValueError, match="다른 시장"):
        krx_store.save_renormalized("20260825", "KOSPI",
                                    [_종목행("035720", "KOSDAQ")])


def test_수집_대상이_아닌_시장은_거부한다(임시저장소):
    """KONEX 는 MARKET_APIS 에는 있지만 우리 수집 대상이 아니다."""
    with pytest.raises(ValueError, match="수집 대상 시장이 아니다"):
        krx_store.save_renormalized("20260825", "KONEX", [_종목행("123456", "KONEX")])


def test_행이_0개면_아무것도_쓰지_않는다(임시저장소):
    """행이 0개인 원문이 실재한다(17B). 0건을 '처리했다' 로 세면 '다 됐다' 로 보인다."""
    krx_store._save("20260825", [_종목행("005930", "KOSPI")])
    _시각을_옛날로(임시저장소, "fetch_log", "bas_dd=?", ("20260825",))
    수집시각 = 옛시각

    assert krx_store.save_renormalized("20260825", "KOSPI", []) == 0
    assert krx_index.save_renormalized("20260821", "KOSPI", []) == 0

    # 기존 행도 시각도 그대로다
    assert set(krx_store.rows_for("20260825", "KOSPI")) == {"005930"}
    assert _시각(임시저장소, "fetch_log", "bas_dd=?", ("20260825",)) == 수집시각
