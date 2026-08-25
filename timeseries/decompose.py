"""분해 — 추세 · 계절 · 잔차 (시계열 엔진) — 명세서 §4.1 · 05-4

한 계열을 세 겹으로 나눈다.

    가법(additive)  y_t = 추세_t + 계절_t + 잔차_t
    승법(multiplicative)  y_t = 추세_t × 계절_t × 잔차_t

**왜 나누는가.** "이번 달 주가가 올랐다" 는 말은 세 가지 중 무엇인지 알아야 뜻이 생긴다.
추세가 올라간 것인지, 원래 이맘때 오르는 것인지, 아니면 설명 안 되는 사건이 있었던 것인지.
잔차만 따로 보면 **평소와 다른 날**이 눈에 띈다.

중심이동평균으로 추세를 뽑는 이유
------------------------------
계절 주기가 5일이면, 연속한 5일을 평균 내면 계절 성분이 서로 상쇄된다(합이 0이므로).
남는 것이 추세다. 그런데 주기가 **짝수**면 평균의 중심이 반 칸 어긋난다. 그래서
양 끝에 절반 가중치를 준 2×m 이동평균을 쓴다.

    m 이 홀수:  (y_{t−k} + … + y_t + … + y_{t+k}) / m         k = (m−1)/2
    m 이 짝수:  (½y_{t−m/2} + y_{t−m/2+1} + … + ½y_{t+m/2}) / m

앞뒤 끝은 창을 채울 수 없어 `None` 이다. **0 이나 첫 값으로 채우지 않는다** —
없는 추세를 만들어 내면 잔차가 그만큼 거짓으로 커진다.

⚠️ 주가에 '계절'이 있는가
----------------------
일봉 주가의 요일 효과(주기 5)는 있더라도 아주 약하다. 이 모듈이 계절 성분을 내놓는다고
해서 그것이 유의하다는 뜻은 아니다. **계절 성분의 진폭이 잔차 표준편차보다 작으면**
사실상 잡음을 계절이라고 부르는 것이므로, `seasonal_strength` 로 그 세기를 함께 낸다.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from timeseries import transform

# 일봉 기본 주기 — 한 주(거래일 5일). 월간 리듬을 보려면 21 을 넘긴다.
DEFAULT_PERIOD = 5


def centered_moving_average(values: Sequence, period: int) -> List[Optional[float]]:
    """중심이동평균 (머리말의 식). 앞뒤 끝은 `None`."""
    x = transform.to_array(values)
    m = max(2, int(period))
    n = x.size
    if n < m:
        return [None] * int(n)

    out: List[Optional[float]] = [None] * n
    half = m // 2

    if m % 2 == 1:
        for t in range(half, n - half):
            window = x[t - half:t + half + 1]
            out[t] = float(np.mean(window)) if np.all(np.isfinite(window)) else None
    else:
        # 2×m 이동평균 — 양 끝에 ½ 가중치
        for t in range(half, n - half):
            window = x[t - half:t + half + 1]
            if not np.all(np.isfinite(window)):
                continue
            weights = np.ones(m + 1)
            weights[0] = weights[-1] = 0.5
            out[t] = float(np.dot(window, weights) / m)
    return out


def classical(values: Sequence, period: int = DEFAULT_PERIOD,
              model: str = "additive") -> Dict:
    """고전적 분해 → `{trend, seasonal, resid}` (명세서 §4.1)

    절차
      1. 추세 = 중심이동평균
      2. 추세 제거 = y − 추세 (가법) 또는 y ÷ 추세 (승법)
      3. 계절 = 같은 주기 위치끼리 평균 → **평균이 0(가법) 또는 1(승법) 이 되게 보정**
      4. 잔차 = 남은 것

    3번의 보정을 빼먹으면 계절 성분에 추세의 일부가 섞여 들어가 두 성분이 겹친다.
    """
    x = transform.to_array(values)
    n = x.size
    m = max(2, int(period))
    multiplicative = model == "multiplicative"

    if n < 2 * m:
        return {
            "available": False,
            "reason": f"표본이 {n}개로 주기 {m} 의 두 배(={2 * m})에 못 미칩니다. "
                      "분해하려면 최소 두 주기가 필요합니다.",
            "period": m, "model": model,
            "trend": [], "seasonal": [], "resid": [], "observed": [],
        }

    if multiplicative and np.any(x[np.isfinite(x)] <= 0):
        return {
            "available": False,
            "reason": "승법 분해는 0 이하 값을 다룰 수 없습니다. 가법 분해를 쓰세요.",
            "period": m, "model": model,
            "trend": [], "seasonal": [], "resid": [], "observed": [],
        }

    # 1 — 추세
    trend = centered_moving_average(x, m)
    trend_array = np.array([np.nan if v is None else v for v in trend], dtype=float)

    # 2 — 추세 제거
    with np.errstate(divide="ignore", invalid="ignore"):
        detrended = (x / trend_array) if multiplicative else (x - trend_array)

    # 3 — 주기 위치별 평균 → 계절 지수
    seasonal_index = np.full(m, np.nan)
    for position in range(m):
        bucket = detrended[position::m]
        bucket = bucket[np.isfinite(bucket)]
        if bucket.size:
            seasonal_index[position] = float(np.mean(bucket))

    # 값을 못 구한 위치는 중립값으로 둔다 (가법 0 · 승법 1)
    neutral = 1.0 if multiplicative else 0.0
    seasonal_index = np.where(np.isfinite(seasonal_index), seasonal_index, neutral)

    # 보정 — 한 주기 안의 계절 효과가 서로 상쇄되게 만든다
    if multiplicative:
        scale = float(np.mean(seasonal_index))
        seasonal_index = seasonal_index / scale if scale else seasonal_index
    else:
        seasonal_index = seasonal_index - float(np.mean(seasonal_index))

    seasonal = np.array([seasonal_index[t % m] for t in range(n)], dtype=float)

    # 4 — 잔차
    with np.errstate(divide="ignore", invalid="ignore"):
        resid = (x / (trend_array * seasonal)) if multiplicative else (x - trend_array - seasonal)

    # 계절이 실제로 의미 있는 크기인지 (머리말 ⚠️)
    finite_resid = resid[np.isfinite(resid)]
    resid_sd = float(np.std(finite_resid, ddof=1)) if finite_resid.size > 1 else 0.0
    seasonal_amplitude = float(np.max(seasonal_index) - np.min(seasonal_index))
    strength = (seasonal_amplitude / resid_sd) if resid_sd > 0 else None

    def as_list(array: np.ndarray) -> List[Optional[float]]:
        return [None if not np.isfinite(v) else float(v) for v in array]

    return {
        "available": True,
        "period": m,
        "model": model,
        "observed": as_list(x),
        "trend": as_list(trend_array),
        "seasonal": as_list(seasonal),
        "resid": as_list(resid),
        "seasonal_index": [float(v) for v in seasonal_index],
        "seasonal_amplitude": seasonal_amplitude,
        "resid_sd": resid_sd,
        "seasonal_strength": strength,
        "strength_text": (
            "계절 성분을 판단할 수 없습니다." if strength is None else
            f"계절 진폭이 잔차 표준편차의 {strength:.2f}배입니다 — "
            + ("뚜렷한 계절성입니다." if strength >= 1.0 else
               "잔차보다 작아 계절성이라 부르기 어렵습니다 (잡음일 가능성이 큽니다).")
        ),
        "trend_points": int(np.sum(np.isfinite(trend_array))),
        "limitation": (
            f"중심이동평균이라 앞뒤 각 {m // 2}개 구간은 추세를 낼 수 없어 비어 있습니다. "
            "일봉 주가의 요일 효과는 대체로 약하므로 계절 성분의 세기를 함께 보십시오."
        ),
    }
