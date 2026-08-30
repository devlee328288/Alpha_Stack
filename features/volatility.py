"""변동성 지표 — True Range · ATR · 역사적 변동성 · Parkinson 변동성
(피처 엔지니어링 계층)

`features/` 계약(기능명세 §3.3)은 `indicators.py`와 같다 — 자세한 설명은 그쪽 모듈
docstring 참고, 여기서는 요약만 적는다.

    t 행은 t 시점까지의 자료만 쓴다 (look-ahead 금지)
    ------------------------------------------------
    rolling 은 과거 → 현재 방향으로만 누적한다(`center=False`). `shift(-1)` 에
    해당하는 어떤 연산도 하지 않는다.

이 모듈은 `features/dynamic_threshold.py`(변동성 기반 유동 임계값)가 그대로
갖다 쓰는 재료다 — 여기서 만드는 변동성 계열의 정확도가 곧 그 임계값의 정확도다.

⚠️ `indicators.py`·`volume.py`와 마찬가지로 numpy 로 직접 구현하고(pandas 아님),
`_to_array`·Wilder 평활 로직도 이 파일 안에 로컬로 둔다 — 다른 `features/` 모듈을
import 하지 않는다(모듈 간 결합을 최소로 둔다).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _to_array(values: Sequence) -> np.ndarray:
    """목록을 실수 배열로. `None` 은 `nan` 이 된다 (계산에서 자연히 전파된다)."""
    return np.asarray([np.nan if v is None else float(v) for v in values], dtype=float)


def _wilder_smooth(x: np.ndarray, window: int) -> np.ndarray:
    """Wilder 지수평활(alpha=1/window, adjust=False). `indicators.py::_ewm_mean`
    (alpha 를 인자로 받는 버전)과 정의가 같다 — RSI 의 평균 상승폭/하락폭 평활과
    같은 재귀식이다.

        y_0 = x_0,   y_t = alpha * x_t + (1 - alpha) * y_{t-1}

    `min_periods=window` 미만인 앞자리는 재귀 계산이 이미 진행됐어도 바깥으로는
    `nan` 으로 가린다.
    """
    alpha = 1.0 / max(1, int(window))
    min_periods = max(1, int(window))
    n = x.size
    out = np.full(n, np.nan)

    start = 0
    while start < n and not np.isfinite(x[start]):
        start += 1
    if start >= n:
        return out

    prev = x[start]
    count = 1
    out[start] = prev if count >= min_periods else np.nan

    for i in range(start + 1, n):
        xi = x[i]
        if np.isfinite(xi):
            prev = alpha * xi + (1.0 - alpha) * prev
            count += 1
        out[i] = prev if count >= min_periods else np.nan

    return out


def true_range(high: Sequence, low: Sequence, close: Sequence) -> np.ndarray:
    """True Range — 그날의 실질 변동폭.

        TR_t = max(High_t − Low_t, |High_t − Close_{t-1}|, |Low_t − Close_{t-1}|)

    전일 종가와 갭이 난 날은 단순 고가-저가보다 변동폭이 크게 잡힌다(갭도 변동성).
    첫 행(t=0)은 전일 종가가 없으므로 `High_0 − Low_0` 만 쓴다 — look-ahead 가
    아니라 "그 시점에 없는 정보를 요구하지 않는다"는 원칙일 뿐이다.
    """
    h = _to_array(high)
    lo = _to_array(low)
    c = _to_array(close)
    if not (h.size == lo.size == c.size):
        raise ValueError("high·low·close 의 길이가 다르다")

    n = h.size
    out = np.full(n, np.nan)
    if n == 0:
        return out

    out[0] = h[0] - lo[0]
    if n == 1:
        return out

    prev_close = c[:-1]
    hl = h[1:] - lo[1:]
    hc = np.abs(h[1:] - prev_close)
    lc = np.abs(lo[1:] - prev_close)
    out[1:] = np.maximum(hl, np.maximum(hc, lc))
    return out


def atr(high: Sequence, low: Sequence, close: Sequence, window: int = 14) -> np.ndarray:
    """ATR(Average True Range) — `true_range` 를 Wilder 평활(alpha=1/window)한 것.

    `indicators.rsi` 가 평균 상승폭/하락폭에 쓴 것과 같은 평활 방식이다 — 원조
    정의(Wilder, 1978)와 대부분의 차트 서비스가 이 방식을 쓴다.
    """
    tr = true_range(high, low, close)
    return _wilder_smooth(tr, window)


def historical_volatility(
    prices: Sequence,
    window: int = 20,
    ddof: int = 1,
    annualize: bool = False,
    periods_per_year: int = 252,
) -> np.ndarray:
    """역사적(실현) 변동성 — 로그수익률의 롤링 표본표준편차.

        r_t = ln(P_t / P_{t-1})
        vol_t = std(r_{t-window+1 .. t})   (표본표준편차, ddof=1 기본)

    `annualize=True` 면 `sqrt(periods_per_year)` 를 곱해 연율화한다(일별 데이터
    기준 거래일수 252가 관례값). 로그수익률은 원계열보다 한 칸 짧으므로, 앞에
    `nan` 한 칸을 더 붙여 원계열과 길이를 맞춘다(`indicators.rsi` 가 델타 배열
    앞에 `nan` 을 되붙이는 것과 같은 이유).
    """
    px = _to_array(prices)
    n = px.size
    window = max(1, int(window))

    if n < 2:
        return np.full(n, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        safe = np.where(px > 0, px, np.nan)
        log_returns = np.log(safe[1:] / safe[:-1])   # 길이 n-1

    m = log_returns.size
    vol = np.full(m, np.nan)
    if window > 1 and m >= window:
        for i in range(window - 1, m):
            vol[i] = np.std(log_returns[i - window + 1: i + 1], ddof=ddof)

    if annualize:
        vol = vol * np.sqrt(periods_per_year)

    return np.concatenate(([np.nan], vol))


def parkinson_volatility(
    high: Sequence,
    low: Sequence,
    window: int = 20,
    annualize: bool = False,
    periods_per_year: int = 252,
) -> np.ndarray:
    """Parkinson 변동성 — 고가·저가 범위 기반 추정량. 종가만 쓰는 것보다
    표본 효율이 좋다(하루 안의 움직임 정보를 버리지 않는다).

        σ²_t = (1 / (4·ln2·window)) · Σ ln(High_i / Low_i)²   (최근 window 일)

    High == Low(거래정지 등)인 행은 `ln(1)=0` 이라 자연히 변동성 0으로 잡힌다 —
    별도 예외처리가 필요 없다.
    """
    h = _to_array(high)
    lo = _to_array(low)
    if h.size != lo.size:
        raise ValueError("high·low 의 길이가 다르다")

    window = max(1, int(window))
    n = h.size
    out = np.full(n, np.nan)
    if n < window:
        return out

    with np.errstate(divide="ignore", invalid="ignore"):
        safe_high = np.where(h > 0, h, np.nan)
        safe_low = np.where(lo > 0, lo, np.nan)
        sq_log_range = np.log(safe_high / safe_low) ** 2

    cumulative = np.concatenate(([0.0], np.nancumsum(sq_log_range)))
    sums = cumulative[window:] - cumulative[:-window]
    variance = sums / (4.0 * np.log(2.0) * window)

    with np.errstate(invalid="ignore"):
        vol = np.sqrt(variance)
    if annualize:
        vol = vol * np.sqrt(periods_per_year)

    out[window - 1:] = vol
    return out
