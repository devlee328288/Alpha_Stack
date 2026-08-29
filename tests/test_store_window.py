"""저장소 `window()` 의 상한 테스트 — 없으면 `as_of` 로 감싸도 미래가 딸려 나온다.

**왜 이 테스트가 필요한가.** `window()` 에는 하한만 있었다. "최근 N거래일" 을 고르고
`bas_dd >= floor` 만 걸었으므로, 2020년 폴드를 학습하면서 불러도 **오늘까지가 함께**
나왔다. `supply/` 가 그 결과를 다시 자르지 않는 한 미래가 그대로 섞이고,
**예외는 나지 않는다 — 성능만 좋아진다.**

상한을 거는 데 드는 비용은 없다. `bas_dd` 가 기본키의 첫 칸이라 범위 조건이 그대로
인덱스를 탄다.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from ingest.store import krx_store


def _평일들(n: int) -> list:
    out, day = [], date(2024, 1, 2)
    while len(out) < n:
        if day.weekday() < 5:
            out.append(day.strftime("%Y%m%d"))
        day += timedelta(days=1)
    return out


DAYS = _평일들(20)
CODES = ("000001", "000002")


@pytest.fixture()
def 저장소(tmp_path, monkeypatch):
    db = tmp_path / "window.db"
    monkeypatch.setattr(krx_store, "DB_PATH", db)
    krx_store.init_db()
    with krx_store.connect() as conn:
        conn.executemany(
            "INSERT INTO daily_price (bas_dd, code, name, market, open, high, low, "
            "close, change, change_rate, volume, value, market_cap, listed_shares) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [(d, c, f"종목{c}", "KOSPI", 1000, 1010, 990, 1000, 0, 0.0,
              5000, 5_000_000, 100_000, 100) for d in DAYS for c in CODES],
        )
    return db


def test_상한이_없으면_전부_나온다(저장소):
    """지금까지의 동작. 이게 문제였다."""
    rows = krx_store.window(days=60)

    assert {r["bas_dd"] for r in rows} == set(DAYS)


def test_상한을_주면_그_뒤가_안_나온다(저장소):
    """🔴 이 한 줄이 없어서 as_of 로 감싼 조회에도 미래가 딸려 왔다."""
    rows = krx_store.window(days=60, end=DAYS[9])

    보인날 = {r["bas_dd"] for r in rows}
    assert 보인날 == set(DAYS[:10])
    assert max(보인날) == DAYS[9]


def test_상한_이하에서_최근_N일을_고른다(저장소):
    """상한을 걸고도 `days` 는 살아 있어야 한다 — 둘은 다른 손잡이다."""
    rows = krx_store.window(days=3, end=DAYS[9])

    assert {r["bas_dd"] for r in rows} == set(DAYS[7:10])


def test_상한이_자료보다_이르면_빈_결과다(저장소):
    rows = krx_store.window(days=60, end="19990101")

    assert rows == []


def test_거래일_목록도_상한을_지킨다(저장소):
    """`available_dates` 가 미래를 담으면 그 목록으로 고른 하한부터 이미 틀린다."""
    days = krx_store.available_dates(limit=60, end=DAYS[4])

    assert max(days) == DAYS[4]
    assert len(days) == 5


def test_종목_시계열에_하한을_걸_수_있다(저장소):
    """`price_series` 가 `start` 를 그대로 넘긴다."""
    rows = krx_store.series(CODES[0], days=None, start=DAYS[15])

    assert [r["date"].replace("-", "") for r in rows] == DAYS[15:]


def test_days_가_None_이면_개수를_자르지_않는다(저장소):
    """라벨과 정리매매 판정은 종목 이력의 양 끝에 의존한다 — 자르면 틀린다."""
    rows = krx_store.series(CODES[0], days=None)

    assert len(rows) == len(DAYS)
