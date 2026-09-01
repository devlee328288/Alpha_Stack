"""기술적 지표 — 이동평균 · RSI · MACD · 볼린저밴드 (피처 엔지니어링 계층)

`features/` 계약(기능명세 §3.3): 입력은 `supply.index_series(as_of=...)` 가 낸
시세 표(종가 계열), 출력은 지표가 붙은 `dataset.parquet` 이다.

    t 행은 t 시점까지의 자료만 쓴다 (look-ahead 금지)
    ------------------------------------------------
    `evaluation/metrics.py`, `timeseries/transform.py` 와 같은 원칙이다.
    여기서 어기면 **에러 없이 성능만 조용히 부풀어 오른다** — 그래서 모든 함수는
    과거 → 현재 방향으로만 누적되는 재귀식·롤링 합만 쓰고, 미래를 앞으로 당기는
    연산(`shift(-1)` 에 해당하는 것)은 절대 하지 않는다.

    수용 기준(기능명세 §3.3): 지표마다 10행짜리 손계산 테스트 + shift 검사
    (t+1 이후 값을 넣었다 빼도 t 시점 이전 출력이 변하지 않아야 한다).

이 모듈은 `timeseries/transform.py` 처럼 **순수 계산 함수**만 모아 둔다 — DB·설정을
모르고, 가격 계열(numpy 배열로 변환 가능한 어떤 시퀀스든, 인덱스는 거래일 오름차순)만
받아 지표 계열을 돌려준다. ⚠️ pandas 가 아니라 numpy 로 직접 구현한다
(`timeseries/transform.py` 와 스타일을 맞춘다 — `rolling`/`ewm` 같은 pandas 메서드에
기대지 않고, 누적합·재귀식을 직접 쓴다).

모든 함수는 **입력과 같은 길이**의 배열을 돌려준다 — 계산할 수 없는 앞쪽 자리는
`np.nan` 으로 채운다. `dataset.py` 가 여러 지표를 같은 길이의 열로 그대로 이어 붙일
수 있어야 하기 때문이다(`timeseries/transform.py::moving_average` 가 앞자리를
`None` 으로 채우는 것과 같은 이유 — 다만 여기서는 수치 계산에 바로 쓰도록 `nan`).

`features/dataset.py`(예정)가 이 함수들을 모아 표를 조립한다.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def _to_array(values: Sequence) -> np.ndarray:
    """목록을 실수 배열로. `None` 은 `nan` 이 된다 (계산에서 자연히 전파된다).

    `timeseries/transform.py::to_array` 와 같은 계약이다 — 이 모듈은 `timeseries/`
    를 import 하지 않고(계층 분리), 필요한 최소 변환만 로컬로 둔다.
    """
    return np.asarray([np.nan if v is None else float(v) for v in values], dtype=float)


def sma(prices: Sequence, window: int) -> np.ndarray:
    """단순이동평균(SMA). 앞쪽 `window-1` 자리는 계산할 수 없으므로 `nan` 이다.

    `timeseries/transform.py::moving_average` 와 같은 누적합 트릭 — 반복문보다
    빠르고 부동소수 오차도 작다.
    """
    x = _to_array(prices)
    window = max(1, int(window))
    n = x.size
    out = np.full(n, np.nan)
    if n < window:
        return out

    cumulative = np.concatenate(([0.0], np.nancumsum(x)))
    sums = cumulative[window:] - cumulative[:-window]

    # 창 안에 결측이 하나라도 있으면 그 창은 통째로 nan — "모르는 값이 섞인 평균은
    # 평균이 아니다" (#18). 여기서 안 막으면 nancumsum 이 결측을 0으로 세고도
    # window로 나눠서, 조용히 낮게 부풀린(왜곡된) 값을 낸다.
    nan_cumulative = np.concatenate(([0], np.cumsum(np.isnan(x))))
    nan_counts = nan_cumulative[window:] - nan_cumulative[:-window]

    out[window - 1:] = np.where(nan_counts > 0, np.nan, sums / window)
    return out


def _ewm_mean(x: np.ndarray, alpha: float, min_periods: int) -> np.ndarray:
    """지수가중평균 재귀식 (pandas `ewm(adjust=False)` 와 같은 정의).

        y_0 = x_0,   y_t = alpha * x_t + (1 - alpha) * y_{t-1}

    `adjust=False` 를 쓰는 이유 — 초기 구간을 뒤에 오는 값까지 함께 정규화해
    되짚어 계산하는 방식(`adjust=True`, pandas 기본값)이 아니라, 그 시점까지 관측치만
    으로 갱신되는 표준 재귀식을 원한다. `min_periods` 미만인 앞자리는 재귀 계산은
    이미 진행됐어도(다음 값들이 그 위에서 이어져야 하므로) 바깥으로는 `nan` 으로 가린다.
    """
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


def ema(prices: Sequence, span: int) -> np.ndarray:
    """지수이동평균(EMA). SMA 와 달리 `min_periods` 이후로는 앞쪽도 값이 채워진다.

        alpha = 2 / (span + 1)
    """
    x = _to_array(prices)
    span = max(1, int(span))
    alpha = 2.0 / (span + 1.0)
    return _ewm_mean(x, alpha=alpha, min_periods=span)


def rsi(prices: Sequence, window: int = 14) -> np.ndarray:
    """상대강도지수(RSI, Wilder 방식). 0~100 범위, 앞쪽 `window` 자리는 `nan`.

    평균 상승폭·평균 하락폭에 **Wilder 의 지수평활**(`alpha = 1/window`)을 쓴다.
    단순 SMA 로 평균을 내는 구현도 흔하지만, 원조 정의(Wilder, 1978)와 대부분의
    차트 서비스가 이 방식을 쓰므로 손계산 테스트 기대값과 맞추기 쉽다.
    """
    x = _to_array(prices)
    window = max(1, int(window))

    if x.size == 0:
        # 다른 13개 함수와 같은 규약 — 빈 입력엔 빈 배열. `supply`가 수집 시작일 이전
        # 조회를 빈 표로 돌려주므로(예외가 아니라 정상값), 여기서 길이가 어긋나면
        # 안 터지고 조용히 한 칸 밀린 값이 붙는다 (#17).
        return np.array([], dtype=float)

    delta = np.diff(x)  # 길이 x.size-1, 하루 전 대비 변화 — 과거만 본다 (0번째 행은 델타 없음)
    gain = np.where(delta > 0.0, delta, 0.0)
    loss = np.where(delta < 0.0, -delta, 0.0)

    # Wilder 평활: alpha = 1/window, adjust=False → 재귀식과 동일
    avg_gain = _ewm_mean(gain, alpha=1.0 / window, min_periods=window)
    avg_loss = _ewm_mean(loss, alpha=1.0 / window, min_periods=window)

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        result = 100.0 - (100.0 / (1.0 + rs))

    # 평균 하락폭이 0 인 구간(계속 상승만 함)은 RSI 정의상 100
    result = np.where(avg_loss == 0.0, 100.0, result)
    # avg_gain 이 nan 인 자리(min_periods 미만)는 위 where 가 100 으로 덮어썼을 수 있으니 되돌린다
    result = np.where(np.isnan(avg_gain), np.nan, result)

    # delta 는 원계열보다 한 칸 짧다 — 0번째 행(전일 없음) 자리를 nan 으로 되돌려 붙인다
    return np.concatenate(([np.nan], result))


def macd(
    prices: Sequence,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Dict[str, np.ndarray]:
    """MACD — `macd`(단기EMA-장기EMA) · `signal`(macd 의 EMA) · `hist`(둘의 차).

    셋 다 EMA 로만 구성되므로 재귀식이 과거만 본다 — 별도의 shift 방어가 필요 없다.
    `timeseries/transform.py::summarize` 처럼 이름 붙은 배열들을 `Dict` 로 돌려준다
    (`dataset.py` 가 `macd_line`, `macd_signal`, `macd_hist` 열로 그대로 얹는다).
    """
    fast_ema = ema(prices, fast)
    slow_ema = ema(prices, slow)
    macd_line = fast_ema - slow_ema
    signal_line = _ewm_mean(macd_line, alpha=2.0 / (max(1, int(signal)) + 1.0),
                             min_periods=signal)
    histogram = macd_line - signal_line

    return {
        "macd": macd_line,
        "signal": signal_line,
        "hist": histogram,
    }


def bollinger_bands(
    prices: Sequence, window: int = 20, num_std: float = 2.0
) -> Dict[str, np.ndarray]:
    """볼린저밴드 — `mid`(SMA) · `upper` · `lower` · `bandwidth`((upper-lower)/mid).

    표준편차는 표본표준편차(`ddof=1`)를 쓴다 — 우리가 가진 `window`개 관측치는
    모집단이 아니라 표본이다(`timeseries/transform.py::summarize` 와 같은 이유).
    """
    x = _to_array(prices)
    window = max(1, int(window))
    n = x.size

    mid = sma(x, window)

    std = np.full(n, np.nan)
    if n >= window and window > 1:
        for i in range(window - 1, n):
            std[i] = np.std(x[i - window + 1: i + 1], ddof=1)
    # window == 1 이면 표본표준편차가 정의되지 않는다(자유도 0) → nan 그대로 둔다

    with np.errstate(divide="ignore", invalid="ignore"):
        upper = mid + num_std * std
        lower = mid - num_std * std
        bandwidth = (upper - lower) / mid

    return {
        "mid": mid,
        "upper": upper,
        "lower": lower,
        "bandwidth": bandwidth,
    }


def percent_b(prices: Sequence, window: int = 20, num_std: float = 2.0) -> np.ndarray:
    """%B — 볼린저밴드 **안에서 가격의 위치**. `bandwidth`(밴드의 폭)와는 다른 것을 잰다.

        %B_t = (close_t - lower_t) / (upper_t - lower_t)

    1에 가까우면 상단선 근접(과열·상승 강도), 0에 가까우면 하단선 근접(과매도·하락
    강도). 급등락으로 밴드를 뚫고 나가면 0~1 범위를 벗어날 수 있다 — 그것도 정보다
    (밴드 폭 대비 얼마나 세게 뚫었는지).
    """
    x = _to_array(prices)
    bands = bollinger_bands(x, window=window, num_std=num_std)
    with np.errstate(divide="ignore", invalid="ignore"):
        width = bands["upper"] - bands["lower"]
        result = np.where(width > 0, (x - bands["lower"]) / width, np.nan)
    return result


def sma_gap(prices: Sequence, short: int, long: int) -> np.ndarray:
    """단기·장기 단순이동평균의 상대적 차이 — `(sma_short / sma_long) - 1`.

    이동평균 값 자체(레벨)는 종목의 가격 수준에 종속적이라 그대로 피처로 쓰면
    안 되지만(종목마다 스케일이 다르다), 두 이동평균의 **비율**은 가격 수준과
    무관한 추세 신호다. 양수면 단기 평균이 장기 평균 위(상승 추세), 음수면 아래
    (하락 추세).
    """
    x = _to_array(prices)
    sma_short = sma(x, short)
    sma_long = sma(x, long)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(sma_long != 0, sma_short / sma_long - 1.0, np.nan)
    return result


def macd_hist_ratio(
    prices: Sequence,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> np.ndarray:
    """MACD 히스토그램을 종가로 정규화한 값 — `hist / close`.

    `macd().hist` 는 가격 단위(원)라 종목마다·시기마다 스케일이 다르다. 종가로
    나누면 그 스케일이 지워져서 종목·시점 간에도 비교 가능한 비율이 된다.
    """
    x = _to_array(prices)
    hist = macd(x, fast=fast, slow=slow, signal=signal)["hist"]
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(x != 0, hist / x, np.nan)
    return result
