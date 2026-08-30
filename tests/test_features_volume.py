"""F(§3.3) — `features/volume.py` 손계산 테스트 + shift 검사.

`test_features_indicators.py` 와 같은 이유로 손계산을 쓴다 — 방향(부호)·평활 상수 하나만
틀려도 값이 그럴듯하게 나오면서 조용히 틀린다.

10행 가격·거래량 계열도 손으로 추적하기 쉽게 작은 정수로 고른다:

    t:      0    1    2    3    4    5    6    7    8    9
    price: 10   11   12   11   10    9   10   11   12   13
    volume:100  200  300  150  100   50  150  200  250  300

가격 계열은 `test_features_indicators.py` 와 똑같다(±1 등락 패턴) — 두 테스트를
나란히 놓고 봐도 헷갈리지 않게 하려는 의도. 윈도는 SMA/볼린저와 맞춰 3으로 둔다.
"""

from __future__ import annotations

import math

import pytest

from features import volume

PRICES = [10, 11, 12, 11, 10, 9, 10, 11, 12, 13]
VOLUMES = [100, 200, 300, 150, 100, 50, 150, 200, 250, 300]


def _assert_allclose(actual, expected):
    """`expected` 는 `None`(손계산) 또는 `nan`(shift 검사용) 둘 다 "값 없음"으로 받는다."""
    assert len(actual) == len(expected)
    for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
        if e is None or (isinstance(e, float) and math.isnan(e)):
            assert math.isnan(a), f"t={i}: nan 이어야 하는데 {a}"
        else:
            assert a == pytest.approx(e, rel=1e-9), f"t={i}: {a} != {e}"


# ── volume_sma ───────────────────────────────────────────────────────────

def test_volume_sma_10행_손계산():
    """window=3, t 시점 포함 최근 3개 거래량 평균.

        t=2: (100+200+300)/3 = 200
        t=3: (200+300+150)/3 = 650/3
        t=9: (200+250+300)/3 = 250
    """
    expected = [None, None, 200.0, 650 / 3, 550 / 3, 100.0, 100.0, 400 / 3, 200.0, 250.0]
    _assert_allclose(volume.volume_sma(VOLUMES, 3), expected)


def test_volume_sma_shift_검사():
    changed = VOLUMES[:-1] + [VOLUMES[-1] + 1_000_000.0]
    before = volume.volume_sma(VOLUMES, 3)
    after = volume.volume_sma(changed, 3)
    _assert_allclose(after[:-1], list(before[:-1]))


# ── volume_ratio ─────────────────────────────────────────────────────────

def test_volume_ratio_10행_손계산():
    """당일 거래량 / volume_sma(같은 t). 분모가 0 이하가 될 일이 없는 계열이라
    나눗셈만 확인한다.

        t=2: 300 / 200 = 1.5
        t=3: 150 / (650/3) = 9/13
    """
    expected = [None, None, 1.5, 9 / 13, 6 / 11, 0.5, 1.5, 1.5, 1.25, 1.2]
    _assert_allclose(volume.volume_ratio(VOLUMES, 3), expected)


def test_volume_ratio_shift_검사():
    changed = VOLUMES[:-1] + [VOLUMES[-1] + 1_000_000.0]
    before = volume.volume_ratio(VOLUMES, 3)
    after = volume.volume_ratio(changed, 3)
    _assert_allclose(after[:-1], list(before[:-1]))


# ── obv ──────────────────────────────────────────────────────────────────

def test_obv_10행_손계산():
    """종가가 오르면 그날 거래량을 더하고, 내리면 뺀다. t=0 은 전일이 없어
    기준점 0.

        델타: +1 +1 -1 -1 -1 +1 +1 +1 +1  (t=1..9, indicators 테스트와 같다)
        t=0: 0
        t=1: 0 + vol[1](200) = 200      (10→11, 상승)
        t=2: 200 + vol[2](300) = 500    (11→12, 상승)
        t=3: 500 - vol[3](150) = 350    (12→11, 하락)
        t=9: ... = 1100                 (12→13, 상승)
    """
    expected = [0.0, 200.0, 500.0, 350.0, 250.0, 200.0, 350.0, 550.0, 800.0, 1100.0]
    _assert_allclose(volume.obv(PRICES, VOLUMES), expected)


def test_obv_shift_검사():
    """마지막 행(가격+거래량)을 통째로 바꿔도 그 이전 누적값은 그대로여야 한다 —
    OBV 는 누적합이라 shift 취약점이 특히 잘 드러나는 지표다."""
    changed_prices = PRICES[:-1] + [PRICES[-1] + 10_000.0]
    changed_volumes = VOLUMES[:-1] + [VOLUMES[-1] + 1_000_000.0]
    before = volume.obv(PRICES, VOLUMES)
    after = volume.obv(changed_prices, changed_volumes)
    _assert_allclose(after[:-1], list(before[:-1]))


def test_obv_길이가_다르면_에러():
    with pytest.raises(ValueError):
        volume.obv(PRICES, VOLUMES[:-1])


# ── vwap ─────────────────────────────────────────────────────────────────

def test_vwap_10행_손계산():
    """window=3, Σ(가격×거래량)/Σ거래량.

        t=2: (10*100+11*200+12*300)/(100+200+300) = 6800/600 = 34/3
        t=3: (11*200+12*300+11*150)/(200+300+150) = 7450/650 = 149/13
    """
    expected = [None, None, 34 / 3, 149 / 13, 125 / 11, 31 / 3, 59 / 6, 83 / 8, 67 / 6, 182 / 15]
    _assert_allclose(volume.vwap(PRICES, VOLUMES, 3), expected)


def test_vwap_shift_검사():
    changed_prices = PRICES[:-1] + [PRICES[-1] + 10_000.0]
    changed_volumes = VOLUMES[:-1] + [VOLUMES[-1] + 1_000_000.0]
    before = volume.vwap(PRICES, VOLUMES, 3)
    after = volume.vwap(changed_prices, changed_volumes, 3)
    _assert_allclose(after[:-1], list(before[:-1]))


def test_vwap_길이가_다르면_에러():
    with pytest.raises(ValueError):
        volume.vwap(PRICES, VOLUMES[:-1], 3)


# ── volume_roc ───────────────────────────────────────────────────────────

def test_volume_roc_10행_손계산():
    """window=3, (V_t - V_{t-3}) / V_{t-3} * 100.

        t=3: (150-100)/100*100 = 50
        t=5: (50-300)/300*100 = -250/3
    """
    expected = [None, None, None, 50.0, -50.0, -250 / 3, 0.0, 100.0, 400.0, 100.0]
    _assert_allclose(volume.volume_roc(VOLUMES, 3), expected)


def test_volume_roc_shift_검사():
    changed = VOLUMES[:-1] + [VOLUMES[-1] + 1_000_000.0]
    before = volume.volume_roc(VOLUMES, 3)
    after = volume.volume_roc(changed, 3)
    _assert_allclose(after[:-1], list(before[:-1]))


# ── 길이 계약 ────────────────────────────────────────────────────────────

def test_모든_지표는_입력과_같은_길이를_돌려준다():
    n = len(PRICES)
    assert len(volume.volume_sma(VOLUMES, 3)) == n
    assert len(volume.volume_ratio(VOLUMES, 3)) == n
    assert len(volume.obv(PRICES, VOLUMES)) == n
    assert len(volume.vwap(PRICES, VOLUMES, 3)) == n
    assert len(volume.volume_roc(VOLUMES, 3)) == n
