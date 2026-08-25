"""정상성 검정 — ADF (시계열 엔진) — 명세서 §4.1 · §4.2 · 05-10

"이 계열을 그대로 모델에 넣어도 되는가, 아니면 차분해야 하는가" 를 가린다.

ADF(Augmented Dickey-Fuller) 가 보는 것
--------------------------------------
    Δy_t = α + γ·y_{t−1} + Σ_{i=1}^{p} δ_i·Δy_{t−i} + ε_t

γ 가 0 이면 y_{t−1} 이 오늘의 변화를 전혀 설명하지 못한다는 뜻 —
**어제 얼마였든 오늘 변화와 무관하다**, 곧 확률보행(단위근)이다.
γ 가 음수로 뚜렷하면 어제가 높았을 때 오늘 내려오는 힘이 있다는 뜻 — 평균회귀, 곧 정상이다.

    귀무가설 H₀: γ = 0  (단위근이 있다 = 비정상)
    검정통계량 τ = γ̂ / SE(γ̂)

**τ 는 t분포를 따르지 않는다.** 회귀변수가 비정상이라 분포가 왼쪽으로 밀려 있다.
그래서 t표가 아니라 Dickey-Fuller 전용 분포표를 봐야 하고, 그 표가 아래 상수다.

⚠️ 임계값은 하드코딩된 근사다 (명세서 §4.2)
------------------------------------------
`scipy` 없이 구현하므로 MacKinnon 응답표면 계수를 상수로 둔다. **p값은 내지 않는다** —
임계값 세 개로 구간만 말한다. 있지도 않은 정밀도를 소수점으로 흉내 내지 않기 위해서다.
이 사실은 `limitation` 에 실려 리포트까지 그대로 나간다.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import numpy as np

from timeseries import transform

# MacKinnon(1994·2010) 응답표면 계수 — **상수항만 포함(constant only, 추세 없음)** 인 경우.
#
#     임계값(T) = β∞ + β₁/T + β₂/T² + β₃/T³
#
# β∞ 가 표본이 무한할 때의 값이고(명세서 §4.2 의 −3.43 · −2.86 · −2.57 이 이것이다),
# 뒤 항들이 **표본 크기 보정**이다. 표본이 작을수록 임계값이 더 아래로 내려간다 —
# 즉 정상이라고 말하기가 더 어려워진다. 250일짜리 계열에서 보정을 빼면
# 5% 임계값이 −2.86 대신 −2.87 쯤으로, 아슬아슬한 판정이 뒤집힐 수 있다.
#
# 추세항이 있는 모형(constant + trend)은 계수가 다르다. 여기서는 쓰지 않는다 —
# 주가에 결정적 시간추세를 가정하면 "영원히 우상향" 을 모형에 새겨 넣는 셈이 된다.
MACKINNON_CONSTANT = {
    "1%":  (-3.43035, -6.5393, -16.786, -79.433),
    "5%":  (-2.86154, -2.8903,  -4.234, -40.040),
    "10%": (-2.56677, -1.5384,  -2.809,   0.000),
}

# 임계값이 근사라는 사실을 늘 함께 내보낸다 (리포트 `limitation` 으로 그대로 간다)
LIMITATION = (
    "ADF 임계값은 MacKinnon(1994) 응답표면 계수를 하드코딩한 **근사값**입니다 "
    "(상수항만 포함, 표본 크기 보정 적용). scipy 를 쓰지 않으므로 정확한 p값은 내지 않고 "
    "1%·5%·10% 임계값과의 비교로만 판정합니다."
)


def critical_values(n: int) -> Dict[str, float]:
    """표본 크기 `n` 에 맞춘 임계값 세 개."""
    size = max(int(n), 10)               # 표본이 10 미만이면 보정항이 발산한다
    out = {}
    for level, (b0, b1, b2, b3) in MACKINNON_CONSTANT.items():
        out[level] = b0 + b1 / size + b2 / size ** 2 + b3 / size ** 3
    return out


def _default_maxlag(n: int) -> int:
    """Schwert(1989) 상한 — ⌊12·(n/100)^(1/4)⌋.

    시차를 너무 적게 넣으면 잔차에 자기상관이 남아 검정이 과잉기각하고,
    너무 많이 넣으면 검정력이 떨어진다. 이 식이 그 절충으로 널리 쓰인다.
    """
    return max(0, int(math.floor(12.0 * (max(n, 1) / 100.0) ** 0.25)))


def _fit(y: np.ndarray, lags: int) -> Optional[Dict]:
    """시차 `lags` 짜리 ADF 회귀 한 번. 설계행렬이 특이하면 `None`.

    Δy_t 를 [1, y_{t−1}, Δy_{t−1} … Δy_{t−lags}] 에 최소제곱으로 맞춘다.
    """
    dy = np.diff(y)
    n = dy.size
    start = lags                          # 시차 항을 만들 수 있는 첫 자리
    rows = n - start
    if rows <= lags + 2:                  # 자유도가 남지 않는다
        return None

    target = dy[start:]
    columns = [np.ones(rows), y[start:-1]]              # 상수항 · y_{t−1}
    for i in range(1, lags + 1):
        columns.append(dy[start - i:n - i])             # Δy_{t−i}
    design = np.column_stack(columns)

    try:
        beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    except np.linalg.LinAlgError:
        return None

    resid = target - design @ beta
    dof = rows - design.shape[1]
    if dof <= 0:
        return None

    sigma2 = float(resid @ resid) / dof
    try:
        # (XᵀX)⁻¹ 의 대각 두 번째 원소가 γ̂ 의 분산 배수다
        xtx_inv = np.linalg.inv(design.T @ design)
    except np.linalg.LinAlgError:
        return None

    se_gamma = math.sqrt(max(sigma2 * float(xtx_inv[1, 1]), 0.0))
    if se_gamma <= 0:
        return None

    ssr = float(resid @ resid)
    k = design.shape[1]
    # 시차 선택용 AIC. 같은 표본(같은 `rows`)에서 비교해야 의미가 있으므로,
    # 아래 `adf` 는 최대 시차 기준으로 표본을 고정해 두고 이 함수를 부른다.
    aic = rows * math.log(ssr / rows) + 2.0 * k if ssr > 0 else -math.inf

    return {
        "gamma": float(beta[1]),
        "se": se_gamma,
        "stat": float(beta[1]) / se_gamma,
        "lags": lags,
        "nobs": int(rows),
        "aic": aic,
    }


def adf(values: Sequence, maxlag: Optional[int] = None,
        autolag: bool = True) -> Dict:
    """ADF 검정. → `{stat, crit_1/5/10, lag, verdict, …}` (명세서 §4.1)

    `autolag=True` 면 시차를 0…maxlag 로 바꿔 가며 AIC 가 가장 작은 것을 고른다.
    이때 **모든 후보가 같은 표본을 보도록** 최대 시차 기준으로 앞을 잘라 둔다.
    표본이 다르면 AIC 를 비교할 수 없다 (짧은 표본이 무조건 유리해진다).
    """
    x = transform.to_array(values)
    x = x[np.isfinite(x)]
    n = x.size

    if n < 20:
        return {
            "available": False,
            "stat": None, "lag": None, "nobs": int(n),
            "crit_1": None, "crit_5": None, "crit_10": None,
            "verdict": "insufficient",
            "verdict_text": "표본이 20개 미만이라 검정할 수 없습니다.",
            "stationary": None,
            "limitation": LIMITATION,
        }

    upper = _default_maxlag(n) if maxlag is None else max(0, int(maxlag))
    upper = min(upper, max(0, (n - 5) // 2))       # 자유도가 남는 선까지만

    # 시차가 클수록 앞자리를 더 많이 버리게 된다. 그대로 두면 후보마다 표본 길이가 달라져
    # AIC 를 비교할 수 없다(짧은 표본이 무조건 유리해진다). 그래서 시차가 작은 후보는
    # 앞을 (upper − lag) 만큼 **더** 버려서 모든 후보가 정확히 같은 관측치를 보게 맞춘다.
    candidates = []
    for lag in range(0, upper + 1) if autolag else [upper]:
        fitted = _fit(x[upper - lag:], lag)
        if fitted:
            candidates.append(fitted)

    if not candidates:
        return {
            "available": False,
            "stat": None, "lag": None, "nobs": int(n),
            "crit_1": None, "crit_5": None, "crit_10": None,
            "verdict": "insufficient",
            "verdict_text": "회귀를 세울 수 없습니다 (표본이 짧거나 값이 모두 같습니다).",
            "stationary": None,
            "limitation": LIMITATION,
        }

    best = min(candidates, key=lambda c: c["aic"])
    crit = critical_values(best["nobs"])
    stat = best["stat"]

    # 임계값보다 **작아야**(더 음수여야) 귀무가설(단위근)을 기각한다
    if stat < crit["1%"]:
        level, verdict = "1%", "stationary"
        text = "1% 유의수준에서 단위근을 기각합니다 — 정상 계열입니다."
    elif stat < crit["5%"]:
        level, verdict = "5%", "stationary"
        text = "5% 유의수준에서 단위근을 기각합니다 — 정상 계열로 봅니다."
    elif stat < crit["10%"]:
        level, verdict = "10%", "weak"
        text = "10% 유의수준에서만 기각됩니다 — 정상성이 약합니다. 차분을 함께 검토하세요."
    else:
        level, verdict = None, "non-stationary"
        text = "단위근을 기각하지 못했습니다 — 비정상 계열입니다. 차분이 필요합니다."

    return {
        "available": True,
        "stat": stat,
        "gamma": best["gamma"],
        "se": best["se"],
        "lag": best["lags"],
        "nobs": best["nobs"],
        "maxlag": upper,
        "crit_1": crit["1%"],
        "crit_5": crit["5%"],
        "crit_10": crit["10%"],
        "reject_at": level,
        "verdict": verdict,
        "verdict_text": text,
        # 10% 에서만 걸리는 경우는 "정상" 으로 단정하지 않는다
        "stationary": verdict == "stationary",
        "autolag": "aic" if autolag else "fixed",
        "limitation": LIMITATION,
    }


def suggest_d(values: Sequence, max_d: int = 2) -> Dict:
    """정상이 될 때까지 차분하며 필요한 **차분 차수 d** 를 정한다 (ARIMA 의 d).

    d 를 2 로 제한한다. 그 이상 차분하면 잡음만 증폭되고, 실제 자료에서 3차 차분이
    필요한 경우는 사실상 없다. 2차까지 해도 비정상이면 그 사실을 그대로 알린다
    (억지로 정상인 척 만들지 않는다).
    """
    steps = []
    current = transform.to_array(values)
    current = current[np.isfinite(current)]

    for d in range(0, max(0, int(max_d)) + 1):
        result = adf(current)
        steps.append({
            "d": d,
            "stat": result.get("stat"),
            "crit_5": result.get("crit_5"),
            "verdict": result.get("verdict"),
            "nobs": result.get("nobs"),
        })
        if result.get("stationary"):
            return {
                "d": d,
                "reason": (f"원계열이 이미 정상입니다 ({result['verdict_text']})" if d == 0
                           else f"{d}차 차분 후 정상이 되었습니다 ({result['verdict_text']})"),
                "steps": steps,
                "adf": result,
                "limitation": LIMITATION,
            }
        current = np.diff(current)
        if current.size < 20:
            break

    return {
        "d": max_d,
        "reason": (f"{max_d}차 차분까지 해도 정상성을 확보하지 못했습니다. "
                   f"{max_d}차를 쓰되 예측을 신뢰구간과 함께 읽어야 합니다."),
        "steps": steps,
        "adf": adf(transform.difference(values, max_d)),
        "limitation": LIMITATION,
    }
