"""상장 구간 — 코드를 다시 받은 회사의 행에 앞 회사의 기록이 붙지 않게 하는 판정.

실측 2026-09-11 · 주권종류·업종 붙이기는 *"그 날까지 알게 된 가장 최근 기록"* 을 얼마나
오래됐든 붙여서, 코드를 다시 받은 두 종목에서 뒤 회사의 행에 앞 회사의 기록이 붙었다.

    036220  인포피아 → 오상헬스케어      주권종류 1행 · 업종  74행   개발구간
    101970  우양에이치씨 → 우양에이치씨  주권종류 1행 · 업종 187행   홀드아웃

붙은 값이 새 회사의 나중 값과 우연히 같아 값 대조로도, 같은 함수를 부르는 판정기로도 안
보였다. 붙이는 두 함수의 시험은 각자의 파일(`test_export_judgment_columns.py` ·
`test_supply_sector.py`)에 있고, 여기서는 그 둘이 기대는 판정 셋을 잰다.

  ① `series_restart_days` 는 `flag_series` 가 신규상장으로 켜는 재시작 자리와 **같은 날**이다
  ② `segment_numbers` 는 구간 시작일 **당일부터** 새 구간이다
  ③ `listing_days_by_code` 는 인덱스를 따라가지 않는다 — 따라가면 923만 행에서 74초다

망을 타지 않는다. DB 는 임시 경로.
"""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from common import corporate_actions as ca
from supply.listing_segment import restart_days_from_db, segment_numbers

#: 시험 달력. 20240312 는 `005930` 등만 거래해서 `036220` 의 2016 → 2024 사이가 거래일 공백이 된다.
CAL = ("20160502", "20160503", "20160504", "20240312", "20240313", "20240314")

#: 종목별 시세 날짜. 다섯 모양을 한 DB 에 둔다.
PRICES = {
    "036220": CAL[:3] + CAL[4:],   # 코드 재사용 — 공백 뒤에 새 상장일
    "005930": CAL,                 # 평범한 종목
    "035720": CAL,                 # 시장 이전 — 상장일은 바뀌지만 공백이 없다
    "111110": (CAL[0], CAL[4]),    # 공백만 있고 상장일은 그대로 — 우리가 모르는 자료 구멍
    "222220": CAL[4:],             # 신규상장 — 상장일 앞에 행이 없다
}

#: `(bas_dd, code, list_dd)`. 기본정보는 시세가 있는 날마다 있다.
LISTINGS = [
    *[(d, "036220", "20070605") for d in CAL[:3]],
    *[(d, "036220", "20240313") for d in CAL[4:]],
    *[(d, "005930", "19750611") for d in CAL],
    *[(d, "035720", "19991109") for d in CAL[:4]],
    *[(d, "035720", "20240313") for d in CAL[4:]],
    *[(d, "111110", "20100104") for d in (CAL[0], CAL[4])],
    *[(d, "222220", "20240313") for d in CAL[4:]],
]


@pytest.fixture
def db(tmp_path):
    """`daily_price` · `stock_base_info` 의 판정에 쓰는 칸과, 실제 DB 에 있는 인덱스 둘."""
    경로 = tmp_path / "segment.db"
    conn = sqlite3.connect(경로)
    conn.executescript(
        """
        CREATE TABLE daily_price (bas_dd TEXT NOT NULL, code TEXT NOT NULL);
        CREATE INDEX idx_code_date ON daily_price(code, bas_dd);
        CREATE TABLE stock_base_info (bas_dd TEXT NOT NULL, code TEXT NOT NULL, list_dd TEXT);
        CREATE INDEX idx_base_info_code ON stock_base_info(code, bas_dd);
        """
    )
    conn.executemany("INSERT INTO daily_price VALUES (?, ?)",
                     [(d, code) for code, days in PRICES.items() for d in days])
    conn.executemany("INSERT INTO stock_base_info VALUES (?, ?, ?)", LISTINGS)
    conn.commit()
    yield conn, 경로
    conn.close()


# ══════════════════════════════════════════════════════════════════════════
# ① 구간 시작 — 코드를 다시 받은 자리만
# ══════════════════════════════════════════════════════════════════════════
def test_코드를_다시_받은_자리만_구간_시작이다(db):
    """시장 이전(공백 없음) · 자료 구멍(상장일 그대로) · 신규상장(앞 행 없음)은 아니다."""
    conn, _ = db
    assert ca.series_restart_days(conn) == {"036220": ("20240313",)}


def test_구간_시작은_flag_series_가_신규상장으로_켜는_재시작_자리와_같다(db):
    """반출본의 `is_first_listing` 과 갈라지면 안 된다. 이웃 행을 전부 판정한 답과 대조한다.

    `series_restart_days` 는 상장일마다 이웃 한 쌍만 본다. 그래도 답이 같아야 한다.
    """
    conn, _ = db
    index, last = ca.market_calendar_index(conn)
    listing = ca.listing_days_by_code(conn)
    전수 = {}
    for code, days in PRICES.items():
        rows = [{"bas_dd": d, "open": 1, "high": 1, "low": 1, "volume": 1,
                 "listed_shares": 1} for d in days]
        flags = ca.flag_series(rows, calendar_index=index, market_last_index=last,
                               still_listed=True, collect_start=CAL[0],
                               listing_days=listing.get(code, ()))
        재시작 = tuple(r["bas_dd"] for i, (r, f) in enumerate(zip(rows, flags, strict=True))
                    if i > 0 and f.first_listing)
        if 재시작:
            전수[code] = 재시작

    assert 전수 == {"036220": ("20240313",)}, "대조가 비어 있으면 같다는 말이 의미가 없다"
    assert ca.series_restart_days(conn) == 전수


def test_상장일_표를_넘기면_그_표로_판정한다(db):
    """반출은 `MarketContext` 의 표를 넘긴다. 넘긴 표를 두고 DB 에서 다시 만들면 안 된다."""
    conn, _ = db
    index, _ = ca.market_calendar_index(conn)
    새상장일을_모르는_표 = {"036220": ("20070605",)}
    assert ca.series_restart_days(conn, listing_days=새상장일을_모르는_표,
                                  calendar_index=index) == {}


def test_경로만_주면_DB_에서_같은_답을_만든다(db):
    _, 경로 = db
    assert restart_days_from_db(경로) == {"036220": ("20240313",)}


# ══════════════════════════════════════════════════════════════════════════
# ② 구간번호 — 그 날짜 이하의 구간 시작일 수
# ══════════════════════════════════════════════════════════════════════════
def test_구간번호는_시작일_당일부터_새_구간이다():
    codes = pd.Series(["036220", "036220", "036220", "005930"])
    days = pd.Series(["20160504", "20240312", "20240313", "20240313"])
    got = segment_numbers(codes, days, {"036220": ("20240313",)})
    assert got.tolist() == [0, 0, 1, 0]      # 재사용이 없는 종목은 같은 날이어도 0 이다


def test_구간_시작이_둘이면_세_구간이고_순서가_섞여_와도_같다():
    codes = pd.Series(["999990"] * 4)
    days = pd.Series(["20191231", "20200102", "20240312", "20240313"])
    got = segment_numbers(codes, days, {"999990": ("20240313", "20200102")})
    assert got.tolist() == [0, 1, 1, 2]


def test_재사용이_없으면_전부_0_이고_빈_표도_된다():
    assert segment_numbers(pd.Series(["005930"]), pd.Series(["20240313"]), {}).tolist() == [0]
    빈 = pd.Series([], dtype=str)
    assert segment_numbers(빈, 빈, {"036220": ("20240313",)}).tolist() == []


# ══════════════════════════════════════════════════════════════════════════
# ③ 상장일 표 — 인덱스를 따라가면 74초
# ══════════════════════════════════════════════════════════════════════════
def test_상장일_표는_인덱스를_따라가지_않는다(db):
    """🔴 `ORDER BY code` 를 인덱스로 풀면 `list_dd` 를 읽으러 행마다 표를 다시 찾는다.

    실측 2026-09-11 · 923만 행: 인덱스를 따라감 74.38초 · 표 순서대로 9.49초 · 결과 같음.
    대조군을 같은 DB 에서 만든다 — 힌트를 빼면 SQLite 가 그 인덱스를 고른다.
    """
    conn, _ = db
    잡힌: list = []
    conn.set_trace_callback(잡힌.append)
    listing = ca.listing_days_by_code(conn)
    conn.set_trace_callback(None)
    sql = next(s for s in 잡힌 if "stock_base_info" in s)

    def 계획(query: str) -> str:
        return " | ".join(str(r[-1]) for r in conn.execute("EXPLAIN QUERY PLAN " + query))

    assert "USING INDEX" not in 계획(sql)
    assert "USING INDEX idx_base_info_code" in 계획(sql.replace(" NOT INDEXED", "")), \
        "대조: 힌트가 없으면 인덱스를 따라간다"
    assert listing["036220"] == ("20070605", "20240313")
