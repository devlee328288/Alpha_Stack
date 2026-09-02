"""상관도 — ACF · PACF (시계열 엔진) — 명세서 §4.1 · 05-7 · 05-8

"오늘 값이 며칠 전 값과 얼마나 닮았는가" 를 재는 두 자다. ARIMA 차수 (p, q) 를 고르는
근거가 여기서 나온다.

| | 무엇을 재나 | 어떤 차수를 읽나 |
|---|---|---|
| ACF  (자기상관) | k일 전과의 상관 — **중간 경로를 포함해서** | MA 차수 q — k=q 뒤로 뚝 끊긴다 |
| PACF (편자기상관) | k일 전과의 상관 — **중간 경로를 걷어내고** | AR 차수 p — k=p 뒤로 뚝 끊긴다 |

둘의 차이가 핵심이다. 어제와 그제가 닮았고 오늘과 어제가 닮았으면, 오늘과 그제는
**어제를 거쳐** 저절로 닮아 보인다. ACF 는 그 간접 경로까지 세지만 PACF 는 걷어낸다.
그래서 "진짜로 2일 전이 오늘에 직접 영향을 주는가" 는 PACF 로만 알 수 있다.

신뢰띠를 왜 그리는가
------------------
상관계수는 **무작위 잡음에서도 0 이 아니게** 나온다. 표본이 250개면 우연히 ±0.12 쯤은
늘 나온다. 띠 안쪽은 "0과 구별되지 않는다" 는 뜻이고, 그걸 그리지 않으면 잡음을
신호로 읽게 된다.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np

from timeseries import transform
from timeseries.numerics import norm_ppf


def _clean(values: Sequence) -> np.ndarray:
    """유한한 값만 남긴다. 상관 계산에 `nan` 이 하나만 섞여도 전부 `nan` 이 된다."""
    x = transform.to_array(values)
    return x[np.isfinite(x)]


def acf(values: Sequence, nlags: int = 40, alpha: float = 0.05) -> Dict:
    """자기상관함수와 **Bartlett 신뢰띠**.

        r_k = Σ_{t=k+1}^{n} (x_t − x̄)(x_{t−k} − x̄) / Σ_{t=1}^{n} (x_t − x̄)²

    분모를 `n − k` 가 아니라 **`n` 으로 고정**한다(편향추정량). 이렇게 해야 자기상관행렬이
    항상 양정치가 되어 뒤의 Durbin-Levinson 재귀가 발산하지 않는다. 표준 관행이다.

    Bartlett 공식 — 시차 k 의 표준오차는 **그 앞 시차들의 상관까지 반영**해서 커진다.

        SE(r_k) = sqrt( (1 + 2·Σ_{j=1}^{k−1} r_j²) / n )

    앞쪽에 강한 상관이 있으면 뒤쪽 띠가 넓어진다. 모든 시차에 ±1.96/√n 를 똑같이 그리면
    뒤쪽에서 없는 유의성을 만들어 낸다.
    """
    x = _clean(values)
    n = x.size
    nlags = max(1, min(int(nlags), max(1, n - 1)))

    if n < 3:
        return {"values": [], "lags": [], "conf_upper": [], "conf_lower": [],
                "n": int(n), "method": "bartlett", "alpha": alpha,
                "note": "표본이 너무 짧아 자기상관을 낼 수 없습니다 (3개 미만)."}

    centered = x - x.mean()
    denominator = float(np.dot(centered, centered))
    if denominator <= 0:
        return {"values": [], "lags": [], "conf_upper": [], "conf_lower": [],
                "n": int(n), "method": "bartlett", "alpha": alpha,
                "note": "모든 값이 같아 분산이 0입니다 — 자기상관이 정의되지 않습니다."}

    values_out: List[float] = []
    for k in range(1, nlags + 1):
        numerator = float(np.dot(centered[k:], centered[:-k]))
        values_out.append(numerator / denominator)

    z = norm_ppf(1.0 - alpha / 2.0)
    upper: List[float] = []
    running = 0.0                       # Σ r_j² (앞 시차들의 제곱합)
    for r in values_out:
        se = math.sqrt((1.0 + 2.0 * running) / n)
        upper.append(z * se)
        running += r * r                # 다음 시차의 띠에 반영된다

    return {
        "lags": list(range(1, nlags + 1)),
        "values": values_out,
        "conf_upper": upper,
        "conf_lower": [-u for u in upper],
        "n": int(n),
        "method": "bartlett",
        "alpha": alpha,
        "significant_lags": [
            k
            for k, r, u in zip(
                range(1, nlags + 1),
                values_out,
                upper,
                strict=False,
            )
            if abs(r) > u
        ],
        "note": "신뢰띠는 Bartlett 공식이라 시차가 커질수록 넓어집니다. "
                "띠 안쪽 값은 0과 구별되지 않습니다.",
    }


def pacf(values: Sequence, nlags: int = 40, alpha: float = 0.05) -> Dict:
    """편자기상관함수 — **Durbin-Levinson 재귀**.

    시차마다 회귀를 새로 돌리지 않고, 앞 단계 결과에서 다음 단계를 만들어 낸다.

        φ_11 = r_1
        φ_kk = ( r_k − Σ_{j=1}^{k−1} φ_{k−1,j}·r_{k−j} ) / ( 1 − Σ_{j=1}^{k−1} φ_{k−1,j}·r_j )
        φ_kj = φ_{k−1,j} − φ_kk·φ_{k−1,k−j}      (j = 1 … k−1)

    O(k²) 로 끝난다. 시차마다 최소제곱을 푸는 방식(O(k⁴))보다 훨씬 싸고,
    무엇보다 **행렬을 뒤집지 않아** 수치적으로 안정적이다.

    신뢰띠는 ACF 와 달리 **모든 시차에 ±z/√n 로 일정**하다.
    귀무가설이 "AR(k−1) 이 참" 이면 φ_kk 의 분산이 시차와 무관하게 1/n 이기 때문이다.
    """
    x = _clean(values)
    n = x.size
    nlags = max(1, min(int(nlags), max(1, n // 2)))     # 시차가 n/2 를 넘으면 추정이 무의미하다

    base = acf(values, nlags=nlags, alpha=alpha)
    r = base["values"]
    if not r:
        return {"values": [], "lags": [], "conf_upper": [], "conf_lower": [],
                "n": int(n), "method": "durbin-levinson", "alpha": alpha,
                "note": base.get("note", "")}

    phi_kk: List[float] = []
    previous: List[float] = []           # 직전 단계의 φ_{k−1,·}

    for k in range(1, len(r) + 1):
        if k == 1:
            new_kk = r[0]
        else:
            numerator = r[k - 1] - sum(previous[j] * r[k - 2 - j] for j in range(k - 1))
            denominator = 1.0 - sum(previous[j] * r[j] for j in range(k - 1))
            # 분모가 0 에 붙으면 계열이 완전 결정적(예: 상수·완전 주기)이라는 뜻이다.
            # 억지로 나누면 발산하므로 거기서 멈춘다.
            if abs(denominator) < 1e-12:
                break
            new_kk = numerator / denominator

        current = [previous[j] - new_kk * previous[k - 2 - j] for j in range(k - 1)]
        current.append(new_kk)
        previous = current
        phi_kk.append(new_kk)

    z = norm_ppf(1.0 - alpha / 2.0)
    band = z / math.sqrt(n)
    lags = list(range(1, len(phi_kk) + 1))

    return {
        "lags": lags,
        "values": phi_kk,
        "conf_upper": [band] * len(phi_kk),
        "conf_lower": [-band] * len(phi_kk),
        "n": int(n),
        "method": "durbin-levinson",
        "alpha": alpha,
        "significant_lags": [k for k, v in zip(lags, phi_kk, strict=False) if abs(v) > band],
        "note": "신뢰띠는 ±z/√n 로 시차와 무관하게 일정합니다. "
                "띠를 벗어난 마지막 시차가 AR 차수 p 의 후보입니다.",
    }


def suggest_order(values: Sequence, nlags: int = 20) -> Dict:
    """ACF·PACF 의 **눈으로 읽는 규칙**을 코드로 옮긴 것 (05-8 표).

    | ACF | PACF | 읽는 모형 |
    |---|---|---|
    | 서서히 감소 | 시차 p 뒤 끊김 | AR(p) |
    | 시차 q 뒤 끊김 | 서서히 감소 | MA(q) |
    | 둘 다 서서히 감소 | | ARMA — 격자탐색으로 정한다 |

    **이건 제안일 뿐 결정이 아니다.** 최종 차수는 `models.select_order` 의 AIC 격자탐색이
    정한다. 여기서는 "왜 그 차수가 나왔는지" 를 사람에게 설명하기 위한 근거를 만든다.
    """
    a = acf(values, nlags=nlags)
    p = pacf(values, nlags=nlags)

    def last_significant(result: Dict) -> int:
        """띠를 벗어난 **연속 구간의 마지막 시차**. 띄엄띄엄 튀는 것은 우연으로 본다."""
        lags = result.get("significant_lags") or []
        if not lags:
            return 0
        cutoff = 0
        for k in result.get("lags", []):
            if k in lags:
                cutoff = k
            else:
                break                    # 처음 끊기는 자리에서 멈춘다
        return cutoff

    p_hint = last_significant(p)
    q_hint = last_significant(a)

    if p_hint and not q_hint:
        shape = f"PACF 가 시차 {p_hint} 뒤 끊기고 ACF 는 서서히 줄어듭니다 → AR({p_hint}) 형태"
    elif q_hint and not p_hint:
        shape = f"ACF 가 시차 {q_hint} 뒤 끊기고 PACF 는 서서히 줄어듭니다 → MA({q_hint}) 형태"
    elif p_hint and q_hint:
        shape = (f"ACF·PACF 가 각각 시차 {q_hint}·{p_hint} 까지 유의합니다 → "
                 "ARMA 혼합 형태로 보이며 격자탐색으로 정합니다")
    else:
        shape = ("ACF·PACF 모두 신뢰띠 안입니다 → 자기상관이 거의 없습니다 "
                 "(백색잡음에 가까워 확률보행을 이기기 어렵습니다)")

    return {
        "p_hint": p_hint,
        "q_hint": q_hint,
        "reading": shape,
        "acf": a,
        "pacf": p,
    }
