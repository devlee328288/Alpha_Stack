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


# ==================================================
# 🔴 FDR 이 **안 편** 자본변동 — 감자 (2026-09-04)
# ==================================================
#
# FDR 은 액면분할은 조정하는데 **감자는 조정하지 않는다.** 그러면 자본변동 양쪽에
# FDR 값이 다 있으므로 `scale_series` ③④(빈 자리 채우기)가 손대지 않고, 그 불연속이
# 그대로 남는다. 전 종목 대조에서 주식수가 2배 이상 변한 2,048자리 중 **17자리**가
# 그랬다 (`scripts/verify_base_info.py` §9).
#
# ⚠️ 액면가로 판정하면 안 보인다 — 분할·병합은 액면가가 같이 바뀌지만 감자는 그대로다.

#: 아센디오(012170) 2025-03-06 감자를 원문 그대로 옮긴 표본 (10.36:1).
#: 🔴 **앞 나흘이 거래정지**(zero-OHLC)다 — 감자도 주권 교체 때문에 KRX 가 정지시킨다.
#:    그래서 계수가 기준가가 아니라 **상장주식수 배율**에서 나온다.
감자표본 = [
    _row("20250304", 0, 0, 0, 230, 0, 103569488, volume=0),
    _row("20250305", 0, 0, 0, 230, 0, 103569488, volume=0),
    _row("20250306", 2000, 2600, 1700, 2065, 65, 10356948, volume=2146719),
    _row("20250307", 1981, 2015, 1704, 1704, -361, 10356948, volume=427848),
]

#: 그 감자의 계수 — 주식수 배율의 역수. 딱 10 이 아니라 10.0000008 이다.
감자계수 = Fraction(103569488, 10356948)


def test_FDR_이_감자를_안_폈으면_앞_구간_전체를_편다():
    """FDR 이 양쪽에 **원가격 그대로** 준 경우 — 우리가 편다.

    감자 앞(230원)과 뒤(2,065원)에 FDR 값이 다 있고 둘 다 조정이 안 돼 있다.
    그대로 두면 그 하루가 `+798%` 로 읽힌다.
    """
    fdr = {"20250304": 230.0, "20250305": 230.0,
           "20250306": 2065.0, "20250307": 1704.0}
    scales = adj_price.scale_series(감자표본, fdr)
    assert scales[0] == 감자계수          # 감자 앞은 주식수 배율만큼 올린다
    assert scales[1] == 감자계수
    assert scales[2] == Fraction(1)       # 감자 당일부터는 FDR 값 그대로
    assert scales[3] == Fraction(1)


def test_감자를_편_뒤_그날_수익률이_KRX_등락률에_가까워진다():
    """편 결과가 옳은지는 **바깥 값**(KRX 등락률)으로 잰다.

    ⚠️ 정확히 일치하지는 않는다 — 계수를 주식수 배율(10.0000008)에서 얻는데 KRX 기준가는
       2,000원(= 전일종가의 8.696배)이라 둘이 다르다. 감자 비율과 기준가 비율은 같지
       않다. 그래도 원가격의 `+798%` 와는 견줄 수 없이 가깝다.
    """
    fdr = {"20250304": 230.0, "20250305": 230.0,
           "20250306": 2065.0, "20250307": 1704.0}
    by_day = {r[-1]: r for r in adj_price.build_rows(감자표본, {
        d: {"adj_close": v} for d, v in fdr.items()})}
    변화 = (by_day["20250306"][3] / by_day["20250305"][3] - 1) * 100
    원가격 = (2065 / 230 - 1) * 100
    assert abs(원가격 - 797.8) < 0.1          # 안 펴면 이렇게 읽힌다
    assert abs(변화 - (-10.2)) < 0.5          # 펴면 하루 등락 크기로 내려온다


def test_이미_편_자본변동은_다시_펴지_않는다():
    """FDR 이 제대로 조정한 자리(액면분할)는 건드리면 안 된다 — 이중 조정이 된다."""
    fdr = {"20180427": 53000.0, "20180504": 51900.0, "20180508": 52600.0}
    scales = adj_price.scale_series(분할표본, fdr)
    # 20180427 원종가 2,650,000 → FDR 53,000 이므로 배율은 1/50 이고 그대로여야 한다.
    assert scales[1] == Fraction(53000) / Fraction(2650000)
    assert scales[5] == Fraction(1)


def test_자본변동이_두_번이면_가장_오래된_구간은_곱해서_눌린다():
    """🔴 손으로 경계를 잡으면 틀리는 자리.

    2026-09-04 에 스크래치 스크립트로 17자리를 보정했을 때, 아센디오(012170)처럼 감자가
    **두 번**인 종목에서 두 번째를 처리하며 **첫 감자 당일 하루를 빠뜨렸다.** 그 하루만
    배율이 어긋나 다음날 대비 `+312.6%` 가 됐고, 극단 검사에 딱 한 행으로 남았다.

    규칙 하나로 앞에서부터 전파하면 그 실수가 나올 수 없다 — 경계를 손으로 잡지 않는다.
    """
    표본 = [
        _row("20250304", 0, 0, 0, 100, 0, 1000000, volume=0),
        _row("20250305", 0, 0, 0, 100, 0, 1000000, volume=0),
        _row("20250306", 1000, 1000, 1000, 1000, 0, 100000),        # 10:1 감자
        _row("20260827", 0, 0, 0, 1000, 0, 100000, volume=0),
        _row("20260828", 5000, 5000, 5000, 5000, 0, 20000),         # 5:1 감자
    ]
    fdr = {r["bas_dd"]: float(r["close"]) for r in 표본}   # FDR 이 둘 다 안 폈다
    scales = adj_price.scale_series(표본, fdr)
    assert scales[0] == Fraction(50)      # 10 × 5 — 가장 오래된 구간
    assert scales[1] == Fraction(50)
    assert scales[2] == Fraction(5)       # 🔴 첫 감자 당일도 두 번째 감자를 받는다
    assert scales[3] == Fraction(5)
    assert scales[4] == Fraction(1)


def test_편_행에는_ca_fix_가_붙는다():
    """`adj_source` 로 "우리가 고쳤다" 를 남긴다 — 안 남기면 FDR 과 왜 다른지 못 답한다."""
    fdr = {"20250304": 230.0, "20250305": 230.0,
           "20250306": 2065.0, "20250307": 1704.0}
    by_day = {r[-1]: r for r in adj_price.build_rows(감자표본, {
        d: {"adj_close": v} for d, v in fdr.items()})}
    assert by_day["20250305"][4] == fdr_data.SOURCE_FDR + adj_price.SOURCE_CA_FIX
    assert by_day["20250306"][4] == fdr_data.SOURCE_FDR      # 감자 당일은 안 고쳤다


def test_권리락_같은_잔_조정은_건드리지_않는다():
    """×0.9~0.99 는 FDR 이 편다. 여기서 손대면 멀쩡한 값을 망친다."""
    표본 = [
        _row("20241227", 76600, 76800, 76000, 76600, 100, 1000000),
        # 주식배당 권리락 — 기준가 73,300 (전일종가의 0.957배). 주식수는 그대로.
        _row("20241230", 73300, 74000, 73000, 73900, 600, 1000000),
    ]
    fdr = {"20241227": 76600.0, "20241230": 73900.0}
    scales = adj_price.scale_series(표본, fdr)
    assert scales[0] == Fraction(1)      # 안 고쳤다
    assert scales[1] == Fraction(1)


def test_FDR_이_한쪽만_아는_자리는_전파가_맡는다():
    """③이 이미 편 자리를 ⑤가 다시 펴면 이중 조정이다."""
    fdr = {"20250306": 2065.0, "20250307": 1704.0}     # 감자 앞은 FDR 이 모른다
    scales = adj_price.scale_series(감자표본, fdr)
    assert scales[0] == 감자계수      # ③이 factor 로 전파한 값 — ⑤가 손대지 않았다
    assert scales[2] == Fraction(1)


def test_가격이_안_따라간_주식수_변동은_조정이_아니다():
    """🔴 오검출을 막는 관문 ①. 실제로 한 번 걸렸다.

    태영건설(009410) 2024-07-22 은 출자전환이라 상장주식수가 **24.9배** 늘었는데 KRX
    등락률이 `0.00%` 다 — 가격이 연속이다. 재개일이라 계수를 주식수 배율에서 얻는 바람에
    `factor` 가 0.0402 로 나오지만, 원문 종가가 그만큼 안 튀었으므로 조정 사건이 아니다.

    이 관문이 없으면 그 종목의 과거 전체가 **1/24.9** 로 눌린다.
    """
    표본 = [
        _row("20240719", 0, 0, 0, 1000, 0, 1000000, volume=0),
        _row("20240722", 1000, 1010, 990, 1000, 0, 24900000),   # 가격 그대로
    ]
    fdr = {"20240719": 1000.0, "20240722": 1000.0}
    scales = adj_price.scale_series(표본, fdr)
    assert scales[0] == Fraction(1)      # 손대지 않았다
    assert scales[1] == Fraction(1)


def test_인적분할처럼_가격도_함께_쪼개진_자리는_건드리지_않는다():
    """🔴 오검출을 막는 관문 ①·②. 태영건설우(009415) 2020-09-22 로 확인한 자리.

    주식수가 절반이 됐지만 인적분할이라 **가격도 절반**이 됐다(32,500 → 기준가 16,250).
    FDR 은 이 자리를 조정하지 않는 것이 맞고, 실제로 그 하루 수익률이 KRX 등락률
    `-29.85%` 와 일치한다. 우리가 펴면 `-82%` 가 되어 틀린다.
    """
    표본 = [
        _row("20200921", 0, 0, 0, 32500, 0, 2557480, volume=0),
        _row("20200922", 16250, 16250, 11400, 11400, -4850, 1302142, volume=1166712),
    ]
    # FDR 이 준 실제 수정종가 — 이 자리를 **이미 제대로 폈다**(뒤가 앞의 2.0033배).
    fdr = {"20200921": 32554.85, "20200922": 22838.48}
    scales = adj_price.scale_series(표본, fdr)
    # FDR 이 심은 값 그대로여야 한다 — ⑤가 손댔다면 여기에 1.9641 이 곱해진다.
    assert scales[0] == Fraction(32554.85).limit_denominator(10 ** 12) / Fraction(32500)
    by_day = {r[-1]: r for r in adj_price.build_rows(표본, {
        d: {"adj_close": v} for d, v in fdr.items()})}
    변화 = (by_day["20200922"][3] / by_day["20200921"][3] - 1) * 100
    assert abs(변화 - (-29.85)) < 0.5      # 바깥 값(KRX 등락률)과 맞는다
