"""F(§3.3) — `features/volatility.py` 손계산 테스트 + shift 검사.

`test_features_indicators.py`·`test_features_volume.py` 와 같은 이유로 손계산을 쓴다.

**high/low를 대칭 오프셋으로 두지 않은 이유.** `close`에 ±고정폭(예: ±0.3)을 그대로
씌우면, 종가 등락폭이 매일 똑같은(±1) 이 테스트 계열에서는 True Range가 매일 똑같은
값으로 나와 `max(H-L, |H-PC|, |L-PC|)` 세 갈래 중 어느 게 실제로 이겼는지 테스트가
증명하지 못한다(직접 계산해보고 발견함). 그래서 종목마다 다른 폭을 손으로 골랐다.

    t:      0     1     2     3     4     5     6     7     8     9
    close: 10    11    12    11    10     9    10    11    12    13
    high:  10.2  11.6  12.3  11.8  10.1   9.4  10.7  11.3  12.6  13.2
    low:    9.6  10.4  11.5  10.5   9.3   8.6   9.5  10.6  11.4  12.3

윈도는 다른 지표 테스트와 맞춰 3으로 둔다(ATR·Parkinson 모두 window=3).
"""

from __future__ import annotations

import math

import pytest

from features import volatility

CLOSE = [10, 11, 12, 11, 10, 9, 10, 11, 12, 13]
HIGH = [10.2, 11.6, 12.3, 11.8, 10.1, 9.4, 10.7, 11.3, 12.6, 13.2]
LOW = [9.6, 10.4, 11.5, 10.5, 9.3, 8.6, 9.5, 10.6, 11.4, 12.3]


def _assert_allclose(actual, expected):
    """`expected` 는 `None`(손계산) 또는 `nan`(shift 검사용) 둘 다 "값 없음"으로 받는다."""
    assert len(actual) == len(expected)
    for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
        if e is None or (isinstance(e, float) and math.isnan(e)):
            assert math.isnan(a), f"t={i}: nan 이어야 하는데 {a}"
        else:
            assert a == pytest.approx(e, rel=1e-9), f"t={i}: {a} != {e}"


# ── true_range ───────────────────────────────────────────────────────────

def test_true_range_10행_손계산():
    """t=0 은 전일 종가가 없어 High-Low 만 쓴다. t≥1 은 세 값 중 최댓값.

        t=0: 10.2 - 9.6 = 0.6
        t=1: H-L=1.2, |H-PC|=|11.6-10|=1.6, |L-PC|=|10.4-10|=0.4 → max=1.6
        t=3: H-L=1.3, |H-PC|=|11.8-12|=0.2, |L-PC|=|10.5-12|=1.5 → max=1.5
             (하락일엔 저가-전일종가 쪽이 이긴다 — 세 갈래가 실제로 갈린다)
    """
    expected = [0.6, 1.6, 1.3, 1.5, 1.7, 1.4, 1.7, 1.3, 1.6, 1.2]
    _assert_allclose(volatility.true_range(HIGH, LOW, CLOSE), expected)


def test_true_range_shift_검사():
    changed_high = HIGH[:-1] + [HIGH[-1] + 10_000.0]
    changed_low = LOW[:-1] + [LOW[-1] + 10_000.0]
    before = volatility.true_range(HIGH, LOW, CLOSE)
    after = volatility.true_range(changed_high, changed_low, CLOSE)
    _assert_allclose(after[:-1], list(before[:-1]))


def test_true_range_길이가_다르면_에러():
    with pytest.raises(ValueError):
        volatility.true_range(HIGH, LOW[:-1], CLOSE)


# ── atr ──────────────────────────────────────────────────────────────────

def test_atr_10행_손계산():
    """window=3, Wilder 평활(alpha=1/3). `true_range` 결과를 재귀식으로 평활한다
    (RSI 의 평균 상승폭/하락폭과 같은 평활 방식).

        TR: [0.6, 1.6, 1.3, 1.5, 1.7, 1.4, 1.7, 1.3, 1.6, 1.2]
        y0=0.6, y1=(1/3)*1.6+(2/3)*0.6=14/15(≈0.933, count=2 → 아직 nan)
        y2=(1/3)*1.3+(2/3)*(14/15)=19/18(count=3, 여기부터 유효) ≈ 1.0556
    """
    expected = [
        None, None,
        19 / 18, 65 / 54, 1109 / 810, 1676 / 1215, 2167 / 1458,
        31147 / 21870, 48643 / 32805, 136652 / 98415,
    ]
    _assert_allclose(volatility.atr(HIGH, LOW, CLOSE, window=3), expected)


def test_atr_shift_검사():
    changed_high = HIGH[:-1] + [HIGH[-1] + 10_000.0]
    changed_low = LOW[:-1] + [LOW[-1] + 10_000.0]
    before = volatility.atr(HIGH, LOW, CLOSE, window=3)
    after = volatility.atr(changed_high, changed_low, CLOSE, window=3)
    _assert_allclose(after[:-1], list(before[:-1]))


# ── historical_volatility ────────────────────────────────────────────────

def test_historical_volatility_10행_손계산():
    """window=3, ddof=1(표본표준편차), annualize=False.

    로그수익률(길이 9)의 롤링 표본표준편차 — 앞에 `nan` 한 칸을 더 붙여 원계열
    길이(10)에 맞춘다. `r_t = ln(close_t/close_{t-1})`.

        t=3: std(r_1, r_2, r_3)
             r_1=ln(11/10), r_2=ln(12/11), r_3=ln(11/12) 세 개의 표본표준편차(ddof=1)
    """
    expected = [
        None, None, None,
        0.10295139557290642, 0.10295139557290639, 0.009188491613517327,
        0.11886483241253758, 0.11886483241253758, 0.009188491613517416,
        0.0076433869614922715,
    ]
    _assert_allclose(volatility.historical_volatility(CLOSE, window=3, ddof=1), expected)


def test_historical_volatility_shift_검사():
    changed = CLOSE[:-1] + [CLOSE[-1] + 10_000.0]
    before = volatility.historical_volatility(CLOSE, window=3, ddof=1)
    after = volatility.historical_volatility(changed, window=3, ddof=1)
    _assert_allclose(after[:-1], list(before[:-1]))


def test_historical_volatility_annualize():
    """`annualize=True` 는 `sqrt(252)` 를 곱하기만 한다 — 배수 관계만 확인한다."""
    plain = volatility.historical_volatility(CLOSE, window=3, ddof=1, annualize=False)
    annualized = volatility.historical_volatility(CLOSE, window=3, ddof=1, annualize=True)
    factor = math.sqrt(252)
    for p, a in zip(plain, annualized, strict=True):
        if math.isnan(p):
            assert math.isnan(a)
        else:
            assert a == pytest.approx(p * factor, rel=1e-9)


# ── parkinson_volatility ─────────────────────────────────────────────────

def test_parkinson_volatility_10행_손계산():
    """window=3, annualize=False.

        σ²_t = (1/(4·ln2·3)) · Σ ln(H_i/L_i)²  (최근 3일)
    """
    expected = [
        None, None,
        0.04918579419607273, 0.06012804823669605, 0.05477646898357613,
        0.05837703161390575, 0.05891516570227121, 0.056070956958227496,
        0.0582840474574153, 0.047910807678074936,
    ]
    _assert_allclose(volatility.parkinson_volatility(HIGH, LOW, window=3), expected)


def test_parkinson_volatility_shift_검사():
    changed_high = HIGH[:-1] + [HIGH[-1] + 10_000.0]
    changed_low = LOW[:-1] + [LOW[-1] + 10_000.0]
    before = volatility.parkinson_volatility(HIGH, LOW, window=3)
    after = volatility.parkinson_volatility(changed_high, changed_low, window=3)
    _assert_allclose(after[:-1], list(before[:-1]))


# ── 길이 계약 ────────────────────────────────────────────────────────────

def test_모든_지표는_입력과_같은_길이를_돌려준다():
    n = len(CLOSE)
    assert len(volatility.true_range(HIGH, LOW, CLOSE)) == n
    assert len(volatility.atr(HIGH, LOW, CLOSE, window=3)) == n
    assert len(volatility.historical_volatility(CLOSE, window=3)) == n
    assert len(volatility.parkinson_volatility(HIGH, LOW, window=3)) == n
