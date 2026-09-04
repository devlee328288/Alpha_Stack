"""F(§3.3) — `features/dynamic_threshold.py` 손계산 테스트 + shift 검사.

`historical_volatility`에 상수(`k · √horizon`)를 곱하기만 하는 얇은 함수라, 기대값은
`test_features_volatility.py::test_historical_volatility_10행_손계산`이 이미 손으로
검증해 둔 변동성 계열을 그대로 가져와 상수만 곱해서 만든다 — 구현을 다시 불러
기대값을 만들면 순환검증이 되므로, `volatility.py`가 이미 독립적으로 검증해 둔
숫자를 상수 취급해 재사용한다.

    close: 10 11 12 11 10 9 10 11 12 13      (test_features_volatility.py 와 동일 계열)
    window=3, ddof=1 일 때 historical_volatility:
        [None, None, None,
         0.10295139557290642, 0.10295139557290639, 0.009188491613517327,
         0.11886483241253758, 0.11886483241253758, 0.009188491613517416,
         0.0076433869614922715]
"""

from __future__ import annotations

import math

import pytest

from features import dynamic_threshold

CLOSE = [10, 11, 12, 11, 10, 9, 10, 11, 12, 13]

#: test_features_volatility.py 가 이미 손계산으로 검증해 둔 값 — 여기서는 상수로만 쓴다.
_HIST_VOL_W3 = [
    None, None, None,
    0.10295139557290642, 0.10295139557290639, 0.009188491613517327,
    0.11886483241253758, 0.11886483241253758, 0.009188491613517416,
    0.0076433869614922715,
]


def _assert_allclose(actual, expected):
    assert len(actual) == len(expected)
    for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
        if e is None or (isinstance(e, float) and math.isnan(e)):
            assert math.isnan(a), f"t={i}: nan 이어야 하는데 {a}"
        else:
            assert a == pytest.approx(e, rel=1e-9), f"t={i}: {a} != {e}"


def test_dynamic_threshold_10행_손계산():
    """window=3, k=0.40(기본값), horizon=5(기본값).

        threshold_t = 0.40 · historical_volatility(t) · √5

    `_HIST_VOL_W3`(검증된 상수)에 `0.40 * sqrt(5)`를 손으로 곱해 기대값을 만든다 —
    이 함수 구현으로 되짚어 만들지 않는다.

        factor = 0.40 * sqrt(5) = 0.894427190999916
        t=3: 0.10295139557290642 * factor = 0.09208252755179587
    """
    factor = 0.40 * math.sqrt(5)
    expected = [None if v is None else v * factor for v in _HIST_VOL_W3]
    _assert_allclose(dynamic_threshold.dynamic_threshold(CLOSE, window=3), expected)


def test_dynamic_threshold_기본값이_ADR_0002_배율과_같다():
    """모듈 상수 `DEFAULT_K`·`DEFAULT_HORIZON`이 함수 기본값과 정확히 같은지 —
    ADR-AS-0002가 실측해 둔 배율(k=0.40)·지평(5거래일)에서 어긋나지 않았는지 확인한다.
    """
    assert dynamic_threshold.DEFAULT_K == pytest.approx(0.40)
    assert dynamic_threshold.DEFAULT_HORIZON == 5


def test_dynamic_threshold_k와_horizon은_순수_배수다():
    """`k`·`horizon`을 바꾸면 결과가 그 배수만큼만 움직여야 한다 — 변동성 계산 자체는
    안 건드리고 상수만 곱하는 얇은 함수라는 것의 직접 증거.
    """
    base = dynamic_threshold.dynamic_threshold(CLOSE, window=3, k=0.40, horizon=5)
    doubled_k = dynamic_threshold.dynamic_threshold(CLOSE, window=3, k=0.80, horizon=5)
    for b, d in zip(base, doubled_k, strict=True):
        if math.isnan(b):
            assert math.isnan(d)
        else:
            assert d == pytest.approx(b * 2.0, rel=1e-9)

    quad_horizon = dynamic_threshold.dynamic_threshold(CLOSE, window=3, k=0.40, horizon=20)
    factor = math.sqrt(20 / 5)
    for b, q in zip(base, quad_horizon, strict=True):
        if math.isnan(b):
            assert math.isnan(q)
        else:
            assert q == pytest.approx(b * factor, rel=1e-9)


def test_dynamic_threshold_shift_검사():
    """t+1 이후 값을 바꿔도 그 이전 값(t 행)은 안 변한다 — look-ahead 없음을
    `historical_volatility`에서 물려받았는지 확인한다.
    """
    changed = CLOSE[:-1] + [CLOSE[-1] + 10_000.0]
    before = dynamic_threshold.dynamic_threshold(CLOSE, window=3)
    after = dynamic_threshold.dynamic_threshold(changed, window=3)
    _assert_allclose(after[:-1], list(before[:-1]))


def test_dynamic_threshold_빈_입력():
    assert len(dynamic_threshold.dynamic_threshold([], window=3)) == 0
