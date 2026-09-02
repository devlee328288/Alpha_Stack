"""수정주가 적재 테스트 — 조용히 틀리는 자리만 골라서

액면분할이 조정되지 않은 `close` 로 수익률을 계산하면 삼성전자 2018-05-04 가 -98% 로
읽힌다(#51). 그 조정을 `adj_*` 네 칸에 채우는데, 여기서 틀리는 방식이 하나같이
**예외를 내지 않는다.**

| 틀리는 방식 | 어떻게 드러나나 |
|---|---|
| 재적재가 `adj_*` 를 지운다 | 행 수는 그대로다. 조회할 때까지 모른다 |
| 정지일 `0` 을 가격으로 싣는다 | 그 날 수익률만 -100% 가 된다 |
| O/H/L 을 따로 반올림한다 | `adj_high < adj_close` 가 되고 고저 폭 피처가 음수를 뱉는다 |
| 배율을 부동소수로 누적한다 | 오래된 구간일수록 조금씩 어긋난다 |

전부 "값이 이상해질 뿐 멈추지 않는" 종류라 테스트로 못 박는다.
"""

from __future__ import annotations

import sqlite3
from fractions import Fraction

import pytest

from ingest.clients import fdr_data
from ingest.store import adj_price, krx_store, migrations

# ── 시세 원문 만들기 ────────────────────────────────────────────────────────

def _row(bas_dd, open_, high, low, close, change, shares, volume=1000):
    return {"bas_dd": bas_dd, "open": open_, "high": high, "low": low,
            "close": close, "change": change, "volume": volume,
            "listed_shares": shares}


#: 삼성전자 2018-05-04 액면분할 50:1 을 그대로 옮긴 표본.
#: 분할 직전 3일이 거래정지(시·고·저가 0)인 것까지 실제 원문과 같다 — KRX 가 주권 교체
#: 때문에 반드시 정지시키므로, 정지일을 빼면 분할이 한 건도 안 남는다.
분할표본 = [
    _row("20180426", 2521000, 2608000, 2520000, 2607000, 87000, 128386494),
    _row("20180427", 2669000, 2682000, 2622000, 2650000, 43000, 128386494),
    _row("20180430", 0, 0, 0, 2650000, 0, 128386494, volume=0),
    _row("20180502", 0, 0, 0, 2650000, 0, 128386494, volume=0),
    _row("20180503", 0, 0, 0, 2650000, 0, 128386494, volume=0),
    _row("20180504", 53000, 53900, 51800, 51900, -1100, 6419324700),
    _row("20180508", 52600, 53200, 51900, 52600, 700, 6419324700),
]


def _fdr(day, open_, high, low, close):
    return {day: {"adj_open": open_, "adj_high": high,
                  "adj_low": low, "adj_close": close}}


# ── 배율 ────────────────────────────────────────────────────────────────────

def test_분할_전_구간이_분할비율만큼_눌린다():
    """앵커가 분할 **뒤**에 있으면 그 앞은 50 으로 나뉘어야 한다."""
    adjusted = _fdr("20180508", 52600.0, 53200.0, 51900.0, 52600.0)
    scales = adj_price.scale_series(분할표본, {"20180508": 52600.0})

    # 앵커일은 배율 1 (원가격이 이미 분할 후 가격이다)
    assert scales[-1] == Fraction(1)
    # 분할 전날은 1/50
    assert scales[1] == Fraction(1, 50)

    rows = adj_price.build_rows(분할표본, adjusted)
    by_day = {r[-1]: r for r in rows}
    assert by_day["20180427"][3] == pytest.approx(2650000 / 50)   # 53,000


def test_분할일_수익률이_98퍼센트_폭락에서_실제값으로_펴진다():
    """이게 #51 의 본체다. 원가격으로는 -98.04%, 수정가격으로는 -2.08% 여야 한다."""
    adjusted = _fdr("20180508", 52600.0, 53200.0, 51900.0, 52600.0)
    by_day = {r[-1]: r for r in adj_price.build_rows(분할표본, adjusted)}

    원가격 = 51900 / 2650000 - 1
    assert 원가격 == pytest.approx(-0.9804, abs=0.001)

    수정 = by_day["20180504"][3] / by_day["20180503"][3] - 1
    assert 수정 == pytest.approx(-0.0208, abs=0.001)      # KRX change_rate 와 같다


def test_배율을_유리수로_옮겨_반올림이_누적되지_않는다():
    """1/50 을 float 으로 4,000번 곱하면 어긋난다. `Fraction` 이라 정확해야 한다."""
    scales = adj_price.scale_series(분할표본, {"20180508": 52600.0})
    assert all(isinstance(s, Fraction) for s in scales)
    assert scales[0] * 50 == 1                     # 정확히 1 — 근사가 아니다


# ── 정지일 ──────────────────────────────────────────────────────────────────

def test_정지일_시고저가는_0_이_아니라_비운다():
    """정지행은 `open=high=low=0` 이다. 0 × 배율 = 0 을 실으면 '그 날 0원' 이 된다."""
    adjusted = _fdr("20180508", 52600.0, 53200.0, 51900.0, 52600.0)
    by_day = {r[-1]: r for r in adj_price.build_rows(분할표본, adjusted)}

    정지일 = by_day["20180430"]
    assert 정지일[0] is None and 정지일[1] is None and 정지일[2] is None
    # 종가만은 남는다 — 직전 값을 물고 있고, 그게 재개일 등락률의 기준이 된다
    assert 정지일[3] == pytest.approx(2650000 / 50)


def test_FDR_정지일_0_을_None_으로_바꾼다():
    """FDR 도 정지일에 시·고·저가를 0 으로 준다. 그대로 실으면 안 된다."""
    import pandas as pd

    frame = pd.DataFrame(
        {"Open": [53380, 0], "High": [53639, 0], "Low": [52440, 0],
         "Close": [53000, 53000], "Volume": [606216, 0]},
        index=pd.to_datetime(["2018-04-27", "2018-04-30"]),
    )
    out = fdr_data._normalize(frame)
    assert out["20180427"]["adj_open"] == 53380.0
    assert out["20180430"]["adj_open"] is None
    assert out["20180430"]["adj_close"] == 53000.0      # 종가는 살아 있다


# ── 고저 관계 ───────────────────────────────────────────────────────────────

def test_FDR_이_고가를_종가보다_낮게_줘도_고저관계가_안_깨진다():
    """🔴 실측으로 드러난 함정이다.

    FDR 은 네 칸을 각각 따로 반올림한다. 원문에서 `close == high` 인 날에도 수정값은
    `adj_high(27,999) < adj_close(28,000)` 으로 뒤집힌다 — 표본 3종에서 68행 나왔다.
    `true_range`·`parkinson_20` 이 여기서 음수를 뱉는다.

    배율 하나를 넷에 똑같이 곱하면 원문의 관계가 그대로 보존된다.
    """
    rows = [_row("20150127", 1375000, 1400000, 1374000, 1400000, 25000, 147299337)]
    # FDR 이 고가만 1원 낮게 준 상황을 그대로 만든다
    adjusted = _fdr("20150127", 27500.0, 27999.0, 27480.0, 28000.0)

    built = adj_price.build_rows(rows, adjusted)[0]
    adj_open, adj_high, adj_low, adj_close = built[:4]

    assert adj_high >= adj_close, "고가가 종가보다 낮으면 고저 폭 피처가 음수가 된다"
    assert adj_low <= adj_close
    # 원문에서 close == high 였으므로 수정값도 같아야 한다
    assert adj_high == pytest.approx(adj_close)
    # 뒤집히지 않던 칸은 FDR 값과 그대로 일치한다
    assert adj_open == pytest.approx(27500.0)
    assert adj_low == pytest.approx(27480.0)


# ── 출처 표시 ───────────────────────────────────────────────────────────────

def test_출처를_날짜로_유추하지_않고_행마다_남긴다():
    """FDR 이 준 날은 `fdr`, 우리가 이어 붙인 날은 `chain`."""
    adjusted = _fdr("20180508", 52600.0, 53200.0, 51900.0, 52600.0)
    by_day = {r[-1]: r for r in adj_price.build_rows(분할표본, adjusted)}
    assert by_day["20180508"][4] == fdr_data.SOURCE_FDR
    assert by_day["20180427"][4] == adj_price.SOURCE_CHAIN


def test_FDR_이_한_날도_없으면_마지막_행_기준으로_이어_붙인다():
    """2014년 이전에 사라진 종목이 그렇다. 분할은 여전히 펴져야 한다."""
    rows = adj_price.build_rows(분할표본, {})
    by_day = {r[-1]: r for r in rows}

    assert all(r[4] == adj_price.SOURCE_CHAIN for r in rows)
    수정 = by_day["20180504"][3] / by_day["20180503"][3] - 1
    assert 수정 == pytest.approx(-0.0208, abs=0.001)     # 분할은 그래도 펴진다


# ── 🔴 재적재가 수정주가를 지우지 않는가 ────────────────────────────────────

def _migrated(tmp_path) -> sqlite3.Connection:
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db, isolation_level=None)
    conn.executescript(krx_store.SCHEMA)
    migrations.migrate(conn)
    conn.row_factory = sqlite3.Row
    return conn


def test_시세를_다시_받아도_수정주가가_지워지지_않는다(tmp_path):
    """🔴 이 테스트가 이 파일에서 제일 중요하다.

    예전 `INSERT OR REPLACE` 는 이름과 달리 **행을 지우고 새로 넣는다.** 그래서 문장에
    안 적힌 `adj_*` 다섯 칸이 조용히 NULL 이 됐다. 하루만 다시 받아도 그 날 3천 종목의
    수정주가가 통째로 사라지는데 **행 수는 그대로**라 어떤 검증에도 안 걸린다.
    """
    conn = _migrated(tmp_path)
    원문 = ("20240102", "005930", "삼성전자", "KOSPI", None,
            79600, 79800, 78200, 79600, 1000, 1.27, 100, 200, 300, 5969782550)
    conn.execute(krx_store.UPSERT_SQL, 원문)
    conn.execute("UPDATE daily_price SET adj_open=1, adj_high=2, adj_low=3, "
                 "adj_close=4, adj_source='fdr' WHERE bas_dd=? AND code=?",
                 ("20240102", "005930"))

    # 같은 날짜를 다시 받는다 — 종가만 바뀐 상황
    다시 = 원문[:8] + (79700,) + 원문[9:]
    conn.execute(krx_store.UPSERT_SQL, 다시)

    row = conn.execute("SELECT * FROM daily_price").fetchone()
    assert row["close"] == 79700, "원가격은 새 값으로 갱신돼야 한다"
    assert row["adj_close"] == 4, "🔴 수정주가가 지워졌다 — INSERT OR REPLACE 로 돌아갔나"
    assert row["adj_source"] == "fdr"
    assert conn.execute("SELECT COUNT(*) FROM daily_price").fetchone()[0] == 1
    conn.close()


def test_수정주가_저장이_원가격을_건드리지_않는다(tmp_path):
    conn = _migrated(tmp_path)
    conn.execute(krx_store.UPSERT_SQL,
                 ("20240102", "005930", "삼성전자", "KOSPI", None,
                  79600, 79800, 78200, 79600, 1000, 1.27, 100, 200, 300, 5969782550))
    adj_price.save(conn, "005930", [(1.0, 2.0, 3.0, 4.0, "chain", "20240102")])

    row = conn.execute("SELECT * FROM daily_price").fetchone()
    assert (row["open"], row["high"], row["low"], row["close"]) == (79600, 79800, 78200, 79600)
    assert row["adj_close"] == 4.0
    conn.close()


# ── 거래일 달력 ─────────────────────────────────────────────────────────────

def test_달력은_실제로_받은_날만_담는다(tmp_path):
    """휴장일을 계산으로 맞히지 않는다. 우리가 받은 날이 거래일이다.

    아래 표본에서 20240103 은 **행이 없다** — 달력에도 없어야 한다.
    """
    conn = _migrated(tmp_path)
    for day, market in (("20240102", "KOSPI"), ("20240102", "KOSDAQ"),
                        ("20240104", "KOSPI")):
        conn.execute(krx_store.UPSERT_SQL,
                     (day, f"C{market}{day}", None, market, None,
                      1, 1, 1, 1, 0, 0.0, 1, 1, 1, 1))
    adj_price.rebuild_calendar(conn)

    전체 = [r[0] for r in conn.execute(
        "SELECT bas_dd FROM trading_calendar WHERE market='ALL' ORDER BY bas_dd")]
    assert 전체 == ["20240102", "20240104"]
    assert "20240103" not in 전체

    시장별 = {(r[0], r[1]): r[2] for r in conn.execute(
        "SELECT bas_dd, market, stock_count FROM trading_calendar")}
    assert 시장별[("20240102", "ALL")] == 2          # 두 시장 합
    assert 시장별[("20240102", "KOSPI")] == 1
    assert 시장별[("20240104", "KOSPI")] == 1
    # 20240104 은 코스피만 열렸다. 코스닥 줄을 만들면 "그 날 코스닥도 거래일" 이 된다.
    assert ("20240104", "KOSDAQ") not in 시장별, "한쪽만 열린 날을 양쪽으로 세면 안 된다"
    conn.close()


def test_달력을_다시_깔아도_낡은_줄이_남지_않는다(tmp_path):
    conn = _migrated(tmp_path)
    conn.execute(krx_store.UPSERT_SQL,
                 ("20240102", "005930", None, "KOSPI", None,
                  1, 1, 1, 1, 0, 0.0, 1, 1, 1, 1))
    adj_price.rebuild_calendar(conn)
    conn.execute("DELETE FROM daily_price")
    conn.execute(krx_store.UPSERT_SQL,
                 ("20240104", "005930", None, "KOSPI", None,
                  1, 1, 1, 1, 0, 0.0, 1, 1, 1, 1))
    adj_price.rebuild_calendar(conn)

    남은 = [r[0] for r in conn.execute(
        "SELECT DISTINCT bas_dd FROM trading_calendar")]
    assert 남은 == ["20240104"], "지운 날이 달력에 남아 있으면 조용히 틀린다"
    conn.close()
