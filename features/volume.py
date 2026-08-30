"""거래량 파생지표 — OBV · 거래량 이동평균 · 거래량 비율 · VWAP · 거래량 변화율
(피처 엔지니어링 계층)

`features/` 계약(기능명세 §3.3)은 `indicators.py`와 같다 — 자세한 설명은 그쪽 모듈
docstring 참고, 여기서는 요약만 적는다.

    t 행은 t 시점까지의 자료만 쓴다 (look-ahead 금지)
    ------------------------------------------------
    rolling 은 과거 → 현재 방향으로만 누적한다(`center=False`). `shift(-1)` 에
    해당하는 어떤 연산도 하지 않는다.

    수용 기준(기능명세 §3.3): 지표마다 10행짜리 손계산 테스트 + shift 검사.

이 모듈도 `indicators.py`와 같은 스타일이다 — numpy 로 직접 구현하고(pandas 아님),
DB·설정을 모르는 순수 계산 함수만 모은다. 모든 함수는 입력과 같은 길이의 배열을
돌려주고, 계산할 수 없는 앞자리는 `np.nan` 으로 채운다.

⚠️ `indicators.py::_to_array`를 import 하지 않고 여기서도 똑같이 로컬로 둔다 —
`indicators.py`가 `timeseries/transform.py`를 import 하지 않은 것과 같은 이유
(계층/모듈 간 결합을 최소로 둔다).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _to_array(values: Sequence) -> np.ndarray:
    """목록을 실수 배열로. `None` 은 `nan` 이 된다 (계산에서 자연히 전파된다)."""
    return np.asarray([np.nan if v is None else float(v) for v in values], dtype=float)


def volume_sma(volumes: Sequence, window: int) -> np.ndarray:
    """거래량 단순이동평균. `indicators.sma` 와 같은 누적합 트릭 — 대상만 거래량이다."""
    x = _to_array(volumes)
    window = max(1, int(window))
    n = x.size
    out = np.full(n, np.nan)
    if n < window:
        return out

    cumulative = np.concatenate(([0.0], np.nancumsum(x)))
    sums = cumulative[window:] - cumulative[:-window]
    out[window - 1:] = sums / window
    return out


def volume_ratio(volumes: Sequence, window: int = 20) -> np.ndarray:
    """당일 거래량 / 최근 `window`일 평균 거래량. 평소보다 몇 배 거래됐는지를 잰다.

    분모의 `window`평균은 t 시점**까지**(t 포함) 계산한 `volume_sma` 를 그대로 쓴다 —
    t 시점 자기 자신은 분모에 들어가지만 t+1 이후 자료는 전혀 안 들어가므로
    look-ahead 가 아니다.
    """
    x = _to_array(volumes)
    avg = volume_sma(x, window)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(avg > 0, x / avg, np.nan)
    return ratio


def obv(prices: Sequence, volumes: Sequence) -> np.ndarray:
    """OBV(On-Balance Volume) — 종가가 오른 날은 거래량을 더하고, 내린 날은 뺀다.

    누적값이라 앞자리부터 값이 나온다. 첫 유효 행은 전일이 없어 비교할 방향이
    없으므로 누적 기준점 0으로 둔다(관례). 가격·거래량이 `nan` 인 행은 그 행의
    출력만 `nan` 으로 가리고, 누적 자체는 **직전 유효 종가** 기준으로 이어간다 —
    구멍 하나 때문에 그 뒤 모든 값이 통째로 `nan` 이 되는 것을 막는다.
    """
    px = _to_array(prices)
    vol = _to_array(volumes)
    if px.size != vol.size:
        raise ValueError("prices 와 volumes 의 길이가 다르다")

    n = px.size
    out = np.full(n, np.nan)

    total = 0.0
    prev_price = None
    for i in range(n):
        if not np.isfinite(px[i]):
            out[i] = np.nan
            continue
        if prev_price is not None and np.isfinite(vol[i]):
            if px[i] > prev_price:
                total += vol[i]
            elif px[i] < prev_price:
                total -= vol[i]
        out[i] = total
        prev_price = px[i]
    return out


def vwap(prices: Sequence, volumes: Sequence, window: int) -> np.ndarray:
    """롤링 거래량가중평균가격(VWAP) — 최근 `window`일의 Σ(가격×거래량) / Σ거래량.

    하루 안에서만 쓰는 "장중 누적 VWAP" 정의는 여기서 안 쓴다 — 일봉 여러 날에 걸친
    지표라 반드시 롤링 창을 둔다. `indicators.sma` 와 같은 누적합 트릭을
    (가격×거래량)과 거래량 각각에 적용한다.
    """
    px = _to_array(prices)
    vol = _to_array(volumes)
    if px.size != vol.size:
        raise ValueError("prices 와 volumes 의 길이가 다르다")

    window = max(1, int(window))
    n = px.size
    out = np.full(n, np.nan)
    if n < window:
        return out

    pv = px * vol
    cum_pv = np.concatenate(([0.0], np.nancumsum(pv)))
    cum_vol = np.concatenate(([0.0], np.nancumsum(vol)))
    sum_pv = cum_pv[window:] - cum_pv[:-window]
    sum_vol = cum_vol[window:] - cum_vol[:-window]

    with np.errstate(divide="ignore", invalid="ignore"):
        out[window - 1:] = np.where(sum_vol > 0, sum_pv / sum_vol, np.nan)
    return out


def volume_roc(volumes: Sequence, window: int = 5) -> np.ndarray:
    """거래량 변화율(Rate of Change, %) — `window`일 전 대비 현재 거래량 변화.

        ROC_t = (V_t − V_{t-window}) / V_{t-window} × 100

    앞쪽 `window`행은 비교할 과거가 없어 `nan`.
    """
    x = _to_array(volumes)
    window = max(1, int(window))
    n = x.size
    out = np.full(n, np.nan)
    if n <= window:
        return out

    with np.errstate(divide="ignore", invalid="ignore"):
        prev = x[:-window]
        curr = x[window:]
        out[window:] = np.where(prev != 0, (curr - prev) / prev * 100.0, np.nan)
    return out
