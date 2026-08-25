"""분포 함수 — scipy 없이 (시계열 엔진 공용) — 명세서 §4

시계열 엔진은 `numpy` + `pandas` 만 쓴다 (기획서 D3). 그런데 검정과 신뢰구간에는
분포 함수가 필요하다. `scipy.stats` 가 해 주던 일을 여기서 최소한으로 대신한다.

| 필요한 곳 | 함수 | 어떻게 |
|---|---|---|
| 신뢰구간 · 상승확률 | `norm_cdf` · `norm_pdf` | `math.erf` — **표준 라이브러리**다 |
| 신뢰구간 폭 (z값) | `norm_ppf` | 위 CDF 를 이분법으로 뒤집는다 — 정확하고 상수가 없다 |
| 시나리오 조건부 기대값 | `truncated_mean` | 정규분포 절단 평균 (역 밀스비) |
| 잔차 백색잡음 검정 | `chi2_sf` | **Wilson–Hilferty 근사** — 아래 한계 참고 |

`math` 는 파이썬 표준 라이브러리이므로 "외부 의존성 금지" 에 걸리지 않는다.
`scipy`·`statsmodels` 를 쓰지 않겠다는 뜻은 **통계 절차를 남의 블랙박스에 맡기지 않겠다**는
것이지, 오차함수 하나를 손으로 다시 짜겠다는 뜻이 아니다.

⚠️ `chi2_sf` 는 근사다
--------------------
카이제곱 상측확률은 불완전감마함수라 닫힌 식이 없다. Wilson–Hilferty 변환

    z = ((Q/k)^(1/3) − (1 − 2/(9k))) / sqrt(2/(9k))    →    p ≈ 1 − Φ(z)

을 쓴다. 자유도 k ≥ 3 에서 오차가 0.001 미만이고, Ljung-Box 는 보통 자유도가
10 이상이라 실무상 충분하다. **자유도가 작을수록 나빠지므로** 그 사실을 쓰는 쪽(`models.ljung_box`)이
리포트에 실어 내보낸다. 있는 척하지 않는다.
"""

from __future__ import annotations

import math
from typing import Optional

# 표준정규분포의 확률밀도 앞에 붙는 상수 1/√(2π)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def norm_pdf(z: float) -> float:
    """표준정규분포의 확률밀도 φ(z)."""
    return _INV_SQRT_2PI * math.exp(-0.5 * z * z)


def norm_cdf(z: float) -> float:
    """표준정규분포의 누적확률 Φ(z) = P(Z ≤ z).

    `math.erf` 로 정확히 계산한다 — 근사가 아니다.
    Φ(z) = ½(1 + erf(z/√2))
    """
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def norm_sf(z: float) -> float:
    """상측확률 P(Z > z). 꼬리에서 `1 - cdf` 는 유효숫자가 날아가므로 대칭성을 쓴다."""
    return norm_cdf(-z)


def norm_ppf(p: float) -> float:
    """Φ 의 역함수 — `p` 분위수에 해당하는 z. (95% 신뢰구간이면 `norm_ppf(0.975)` ≈ 1.95996)

    유리함수 근사 계수를 외우는 대신 **이분법으로 CDF 를 뒤집는다.**
    Φ 는 단조증가이므로 반드시 수렴하고, 100회면 배정밀도 한계까지 좁혀진다.
    상수를 안 쓰니 옮겨 적다 틀릴 여지도 없다.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"분위수는 0과 1 사이여야 합니다: {p}")

    low, high = -40.0, 40.0            # Φ(-40) 과 Φ(40) 은 배정밀도에서 0 과 1 이다
    for _ in range(200):
        mid = (low + high) / 2.0
        if norm_cdf(mid) < p:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def truncated_mean(mu: float, sigma: float,
                   lower: Optional[float] = None,
                   upper: Optional[float] = None) -> float:
    """정규분포 N(mu, sigma²) 를 `[lower, upper]` 로 자른 뒤의 **조건부 기대값**.

    시나리오 목표가에 쓴다. "저항선을 넘는다면 얼마쯤에서 자리를 잡을까" 는
    분포 전체의 평균이 아니라 **그 영역 안에서의 평균**이어야 맞다.

        E[Y | a < Y < b] = mu + sigma · (φ(α) − φ(β)) / (Φ(β) − Φ(α))

    자른 구간의 확률이 사실상 0이면(=일어날 수 없는 시나리오) 가장 가까운 경계를 돌려준다.
    0으로 나누어 `nan` 을 내보내는 것보다 낫다.
    """
    if sigma <= 0:
        return mu

    alpha = (lower - mu) / sigma if lower is not None else -math.inf
    beta = (upper - mu) / sigma if upper is not None else math.inf

    cdf_lo = norm_cdf(alpha) if alpha != -math.inf else 0.0
    cdf_hi = norm_cdf(beta) if beta != math.inf else 1.0
    mass = cdf_hi - cdf_lo

    if mass < 1e-12:
        # 확률이 없는 구간 — 경계값으로 답한다
        if lower is not None and upper is not None:
            return (lower + upper) / 2.0
        return lower if lower is not None else (upper if upper is not None else mu)

    pdf_lo = norm_pdf(alpha) if alpha != -math.inf else 0.0
    pdf_hi = norm_pdf(beta) if beta != math.inf else 0.0
    return mu + sigma * (pdf_lo - pdf_hi) / mass


def chi2_sf(stat: float, df: int) -> float:
    """카이제곱 상측확률 P(χ²_df > stat) — **Wilson–Hilferty 근사** (머리말 ⚠️ 참고).

    자유도 3 이상에서 오차 0.001 미만. Ljung-Box 검정에 쓴다.
    """
    if df <= 0 or stat < 0:
        return float("nan")
    if stat == 0:
        return 1.0

    # (χ²/k)^(1/3) 은 평균 1−2/(9k) · 분산 2/(9k) 의 정규분포에 가깝다
    ratio = stat / df
    mean = 1.0 - 2.0 / (9.0 * df)
    sd = math.sqrt(2.0 / (9.0 * df))
    z = (ratio ** (1.0 / 3.0) - mean) / sd
    return norm_sf(z)
