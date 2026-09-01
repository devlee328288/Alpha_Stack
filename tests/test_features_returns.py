"""F(§3.3) — `features/returns.py` 손계산 테스트 + shift 검사.

`test_features_indicators.py`·`test_features_volume.py` 와 같은 이유로 손계산을 쓴다.
가격 계열도 그 둘과 똑같다(±1 등락 패턴) — 나란히 놓고 봐도 헷갈리지 않게 하려는
의도다. 윈도는 SMA/볼린저와 맞춰 3으로 둔다.

    t:      0    1    2    3    4    5    6    7    8    9
    price: 10   11   12   11   10    9   10   11   12   13
"""

from __future__ import annotations

import math

import pytest

from features import returns

PRICES = [10, 11, 12, 11, 10, 9, 10, 11, 12, 13]


def _assert_allclose(actual, expected):
    """`expected` 는 `None`(손계산) 또는 `nan`(shift 검사용) 둘 다 "값 없음"으로 받는다."""
    assert len(actual) == len(expected)
    for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
        if e is None or (isinstance(e, float) and math.isnan(e)):
            assert math.isnan(a), f"t={i}: nan 이어야 하는데 {a}"
        else:
            assert a == pytest.approx(e, rel=1e-9), f"t={i}: {a} != {e}"


# ── n_day_return ─────────────────────────────────────────────────────────

def test_n_day_return_10행_손계산():
    """window=3, ret_t = price_t / price_{t-3} - 1.

        t=3: 11/10 - 1 = 0.1
        t=5: 9/12 - 1 = -0.25   (12일 만에 3 빠짐 — 하락도 정직하게 음수로 잡힌다)
        t=8: 12/9 - 1 = 1/3
    """
    expected = [None, None, None, 0.1, -1 / 11, -0.25, -1 / 11, 0.1, 1 / 3, 0.3]
    _assert_allclose(returns.n_day_return(PRICES, 3), expected)


def test_n_day_return_shift_검사():
    """마지막 값이 미래에서 바뀌어도, 그보다 앞선 행은 값이 안 바뀐다(look-ahead 없음)."""
    changed = PRICES[:-1] + [PRICES[-1] + 10_000.0]
    before = returns.n_day_return(PRICES, 3)
    after = returns.n_day_return(changed, 3)
    _assert_allclose(after[:-1], list(before[:-1]))


def test_n_day_return_빈_입력은_빈_배열():
    """다른 함수들과 같은 규약 — 빈 입력엔 빈 배열 (#17 과 같은 종류의 회귀 방지)."""
    assert len(returns.n_day_return([], 5)) == 0


def test_n_day_return_길이가_창보다_짧으면_전부_nan():
    short = PRICES[:2]  # window=3 인데 2행뿐 — 비교할 과거가 아예 없다
    result = returns.n_day_return(short, 3)
    assert len(result) == 2
    assert math.isnan(result[0]) and math.isnan(result[1])


# ── 길이 계약 ────────────────────────────────────────────────────────────

def test_n_day_return은_입력과_같은_길이를_돌려준다():
    assert len(returns.n_day_return(PRICES, 3)) == len(PRICES)
