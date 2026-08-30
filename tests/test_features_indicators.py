"""F(§3.3) — `features/indicators.py` 손계산 테스트 + shift 검사.

**왜 손계산인가.** 지표 계산은 pandas/numpy 라이브러리 함수를 갖다 쓰는 코드라
"돌아간다"는 거의 항상 참이다. 그런데 방향(부호), 평활 상수, 표본/모집단
표준편차(`ddof`) 하나만 틀려도 값은 그럴듯하게 나오면서 조용히 틀린다 — RSI가
거꾸로 나와도 0~100 범위 안에 있으면 눈으로는 안 잡힌다. 그래서 라이브러리를
다시 부르는 대신, 정의(재귀식)를 손으로 따라가 나온 숫자와 직접 비교한다.

**왜 shift 검사가 따로 있는가.** look-ahead 는 예외를 안 낸다 — 성능만 조용히
올린다(`timeseries/transform.py`, `evaluation/` 와 같은 문제의식). 각 지표 함수가
"t 시점까지 자료만 쓴다"는 것을, 미래 값을 바꿔치기해도 과거 출력이 그대로인지로
직접 확인한다.

10행 가격 계열은 손으로 추적하기 쉽게 작은 정수로 고른다:

    t:      0   1   2   3   4   5   6   7   8   9
    price: 10  11  12  11  10   9  10  11  12  13

윈도/기간도 10행 안에서 유효값이 여러 개 나오도록 작게 잡는다(SMA/RSI/볼린저는
window=3, EMA는 span=3, MACD는 fast=2·slow=3·signal=2) — 실제 서비스 기본값
(RSI 14, MACD 12/26/9 등)은 10행으로는 전부 `nan` 이라 손계산 검증이 안 된다.
기본값 자체가 맞는지는 별도로, 알려진 값(pandas `ewm`/`rolling` 기준 재귀식)과
직접 비교해 `features/indicators.py` 를 작성했다(구현 시 대조 완료).
"""

from __future__ import annotations

import math

import pytest

from features import indicators

PRICES = [10, 11, 12, 11, 10, 9, 10, 11, 12, 13]


def _assert_allclose(actual, expected):
    """`expected` 는 `None`(손계산) 또는 `nan`(다른 호출 결과 재사용, shift 검사용)
    둘 다 "값 없음"으로 받아들인다."""
    assert len(actual) == len(expected)
    for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
        if e is None or (isinstance(e, float) and math.isnan(e)):
            assert math.isnan(a), f"t={i}: nan 이어야 하는데 {a}"
        else:
            assert a == pytest.approx(e, rel=1e-9), f"t={i}: {a} != {e}"


# ── SMA ──────────────────────────────────────────────────────────────────

def test_sma_10행_손계산():
    """window=3, t 시점 포함 최근 3개 평균. 앞 2행은 계산 불가 → nan.

        t=2: (10+11+12)/3 = 11
        t=3: (11+12+11)/3 = 34/3
        t=9: (11+12+13)/3 = 12
    """
    expected = [None, None, 11.0, 34 / 3, 11.0, 10.0, 29 / 3, 10.0, 11.0, 12.0]
    _assert_allclose(indicators.sma(PRICES, 3), expected)


def test_sma_shift_검사():
    """t=9(마지막 행)을 바꿔도 t=0..8 은 그대로여야 한다 — window=3 은 자기 자신
    포함 과거 3개만 보므로, 미래(t=9)를 아무리 바꿔도 그 이전 행은 볼 수 없다."""
    changed = PRICES[:-1] + [PRICES[-1] + 10_000.0]
    before = indicators.sma(PRICES, 3)
    after = indicators.sma(changed, 3)
    _assert_allclose(after[:-1], list(before[:-1]))


# ── EMA ──────────────────────────────────────────────────────────────────

def test_ema_10행_손계산():
    """span=3 → alpha=2/(3+1)=0.5. 재귀식 y_t = 0.5*P_t + 0.5*y_{t-1}, y_0=P_0.

        y0=10, y1=0.5*11+0.5*10=10.5, y2=0.5*12+0.5*10.5=11.25 (여기부터 유효, count=3)
        y3=0.5*11+0.5*11.25=11.125
    min_periods=span=3 이므로 y0·y1 은 바깥으로 nan.
    """
    expected = [None, None, 11.25, 11.125, 10.5625, 9.78125,
                9.890625, 10.4453125, 11.22265625, 12.111328125]
    _assert_allclose(indicators.ema(PRICES, 3), expected)


def test_ema_shift_검사():
    """EMA 는 재귀식이라 이론상 앞자리 오염에 더 취약하다 — 미래를 바꿔도
    과거 출력이 그대로인지가 특히 중요하다."""
    changed = PRICES[:-1] + [PRICES[-1] + 10_000.0]
    before = indicators.ema(PRICES, 3)
    after = indicators.ema(changed, 3)
    _assert_allclose(after[:-1], list(before[:-1]))


# ── RSI ──────────────────────────────────────────────────────────────────

def test_rsi_10행_손계산():
    """window=3, Wilder 평활(alpha=1/3). 델타(t=1..9)의 상승폭/하락폭을 각각
    지수평활한 뒤 RSI = 100 * avg_gain / (avg_gain + avg_loss) 로 잰다
    (avg_gain+avg_loss 는 매 스텝 1로 보존되므로 100*avg_gain 과 같다).

        델타:  +1 +1 -1 -1 -1 +1 +1 +1 +1  (t=1..9)
        gain 만 평활(alpha=1/3, 시드=gain[0]=1):
          t=1: 1, t=2: 1, t=3: 2/3(count=3, 유효 시작) → RSI=200/3
          t=4: 4/9 → RSI=400/9
          t=9: 1883/2187 → RSI=188300/2187
    t=0,1,2 는 계산할 델타/표본이 모자라 nan.
    """
    expected = [None, None, None,
                200 / 3, 400 / 9, 800 / 27, 4300 / 81, 16700 / 243,
                57700 / 729, 188300 / 2187]
    _assert_allclose(indicators.rsi(PRICES, 3), expected)


def test_rsi_shift_검사():
    changed = PRICES[:-1] + [PRICES[-1] + 10_000.0]
    before = indicators.rsi(PRICES, 3)
    after = indicators.rsi(changed, 3)
    _assert_allclose(after[:-1], list(before[:-1]))


# ── MACD ─────────────────────────────────────────────────────────────────
# 실제 서비스 기본값(12/26/9)은 10행으로 전부 nan 이라, 여기서는 손으로
# 따라갈 수 있는 fast=2·slow=3·signal=2 로 정의(재귀식)를 직접 검증한다.

def test_macd_10행_손계산():
    macd_line = [None, None, 0.30555555555555556, 0.06018518518518518,
                 -0.16743827160493827, -0.31622942386831276, -0.06895147462277092,
                 0.16191200845907636, 0.31308525281969213, 0.40058570927323073]
    signal_line = [None, None, None, 0.1419753086419753, -0.06430041152263374,
                   -0.2322530864197531, -0.12338534522176497, 0.06681289056546258,
                   0.2309944654016156, 0.344055294649359]
    hist = [None, None, None, -0.08179012345679013, -0.10313786008230452,
            -0.08397633744855967, 0.05443387059899406, 0.09509911789361378,
            0.08209078741807652, 0.0565304146238717]

    out = indicators.macd(PRICES, fast=2, slow=3, signal=2)
    _assert_allclose(out["macd"], macd_line)
    _assert_allclose(out["signal"], signal_line)
    _assert_allclose(out["hist"], hist)


def test_macd_shift_검사():
    changed = PRICES[:-1] + [PRICES[-1] + 10_000.0]
    before = indicators.macd(PRICES, fast=2, slow=3, signal=2)
    after = indicators.macd(changed, fast=2, slow=3, signal=2)
    for key in ("macd", "signal", "hist"):
        _assert_allclose(after[key][:-1], list(before[key][:-1]))


# ── 볼린저밴드 ────────────────────────────────────────────────────────────

def test_bollinger_10행_손계산():
    """window=3, num_std=2, ddof=1(표본표준편차).

        t=2: 표본 {10,11,12}, mid=11, std=1 → upper=13, lower=9, bw=4/11
        t=3: 표본 {11,12,11}, mid=34/3, std=sqrt(1/3)=0.5773502691896257
    """
    mid = [None, None, 11.0, 34 / 3, 11.0, 10.0, 29 / 3, 10.0, 11.0, 12.0]
    upper = [None, None, 13.0, 12.488033871712584, 13.0, 12.0,
             10.821367205045917, 12.0, 13.0, 14.0]
    lower = [None, None, 9.0, 10.178632794954083, 9.0, 8.0,
             8.511966128287416, 8.0, 9.0, 10.0]
    bandwidth = [None, None, 4 / 11, 0.20377068324339714, 4 / 11, 0.4,
                 0.23890355966467255, 0.4, 4 / 11, 1 / 3]

    out = indicators.bollinger_bands(PRICES, window=3, num_std=2.0)
    _assert_allclose(out["mid"], mid)
    _assert_allclose(out["upper"], upper)
    _assert_allclose(out["lower"], lower)
    _assert_allclose(out["bandwidth"], bandwidth)


def test_bollinger_shift_검사():
    changed = PRICES[:-1] + [PRICES[-1] + 10_000.0]
    before = indicators.bollinger_bands(PRICES, window=3, num_std=2.0)
    after = indicators.bollinger_bands(changed, window=3, num_std=2.0)
    for key in ("mid", "upper", "lower", "bandwidth"):
        _assert_allclose(after[key][:-1], list(before[key][:-1]))


# ── 길이 계약 ────────────────────────────────────────────────────────────

def test_모든_지표는_입력과_같은_길이를_돌려준다():
    """`dataset.py` 가 여러 지표를 그대로 열로 이어 붙이려면 길이가 같아야 한다."""
    n = len(PRICES)
    assert len(indicators.sma(PRICES, 3)) == n
    assert len(indicators.ema(PRICES, 3)) == n
    assert len(indicators.rsi(PRICES, 3)) == n
    macd_out = indicators.macd(PRICES, fast=2, slow=3, signal=2)
    assert all(len(v) == n for v in macd_out.values())
    bb_out = indicators.bollinger_bands(PRICES, window=3, num_std=2.0)
    assert all(len(v) == n for v in bb_out.values())
