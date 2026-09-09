"""예측 — 점예측 · 신뢰구간 · 상승확률 · 시나리오 (시계열 엔진) — 명세서 §4.1 · §4.4

`models` 가 맞춘 모형으로 앞날을 낸다. 명세서 §4.4 의 **3단 산출**이 여기서 만들어진다.

    statistical   점예측 + 95% 신뢰구간            ← ψ-weight
    probability   상승확률 + 백테스트              ← 정규분포 + `backtest`
    scenarios     Bear / Base / Bull + 트리거      ← 아래 "시나리오 규칙" 참고

왜 점예측만 내면 안 되는가
------------------------
"20일 뒤 78,400원" 은 **틀릴 것이 확실한** 숫자다. 의미가 있는 것은 폭이다.
"78,400원 ± 9,200원(95%)" 이라고 해야 독자가 위험을 가늠할 수 있다.
그래서 이 모듈은 점예측만 돌려주는 함수를 두지 않고, 늘 구간을 함께 낸다.

시나리오 규칙 (M3 확정 — U4 선행 결정)
------------------------------------
명세서 §4.4 는 시나리오에 `trigger` 와 `invalidation` 을 요구하지만 **무엇을 근거로
만들지는 정하지 않았다.** M3 에서 다음과 같이 못박는다.

    목표가 · 확률 → **통계** (예측분포에서 나온다)
    트리거 · 무효화 → **기술적 수준** (차트에서 눈으로 확인할 수 있어야 한다)

왜 이렇게 갈랐는가 — 예측분포는 정규분포라 "점예측 ± 0.5σ" 같은 σ 배수로 영역을 나누면
확률이 **언제나 30.9% / 38.2% / 30.9%** 로 고정된다. 종목이 무엇이든 같은 숫자가 나오니
정보가 없다. 그래서 영역의 경계를 **최근 고점·저점(저항·지지)** 이라는 실제 가격에 두고,
그 선을 넘을 확률을 예측분포에서 읽는다. 이러면 확률이 종목마다·기간마다 달라진다.

    Bull  y_h > 저항(최근 N일 고점)      확률 = 1 − Φ((저항 − 점예측)/σ_h)
    Bear  y_h < 지지(최근 N일 저점)      확률 = Φ((지지 − 점예측)/σ_h)
    Base  그 사이                        확률 = 1 − Bull − Bear

목표가는 각 영역의 **조건부 기대값**이다 (절단정규 평균). "저항을 넘는다면 어디쯤" 이
분포 전체의 평균이 아니라 그 영역 안의 평균이어야 맞기 때문이다.

트리거와 무효화는 08 투자분석 강의의 관측 가능한 규칙을 쓴다 — 20일 이동평균 돌파,
거래대금 20일 평균 대비 배수, 지지·저항선 종가 이탈. **차트를 보고 참·거짓을 판정할 수
있어야** 가설이 된다. "투자심리가 개선되면" 같은 것은 규칙이 아니다.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from timeseries import transform
from timeseries.numerics import norm_cdf, norm_ppf, norm_sf, truncated_mean

# 시나리오의 지지·저항을 읽는 기간 (거래일). 분기 남짓이라 최근 국면을 대표한다.
SCENARIO_LOOKBACK = 60

# 트리거에 쓰는 이동평균·거래대금 창
TRIGGER_MA = 20
VOLUME_SURGE_MULTIPLE = 1.5


# ==================================================
# 1. ψ-weight — 신뢰구간의 뼈대
# ==================================================
def psi_weights(phi: Sequence[float], theta: Sequence[float], horizon: int) -> np.ndarray:
    """ARMA 를 무한차 MA 로 펼쳤을 때의 계수 ψ₀ … ψ_{h−1}.

        ψ₀ = 1
        ψ_j = θ_j + Σ_{i=1}^{min(j,p)} φ_i · ψ_{j−i}       (j > q 면 θ_j = 0)

    **왜 필요한가.** h일 뒤 예측오차는 그 사이에 들어올 충격들의 합이다.

        e_h = ε_{T+h} + ψ₁ε_{T+h−1} + … + ψ_{h−1}ε_{T+1}

    충격끼리 독립이므로 분산이 ψ 제곱합으로 깔끔하게 떨어진다.
    이것이 없으면 "h가 커질수록 구간이 넓어진다" 를 정량화할 수 없다.
    """
    phi = np.asarray(phi, dtype=float)
    theta = np.asarray(theta, dtype=float)
    h = max(1, int(horizon))

    psi = np.zeros(h, dtype=float)
    psi[0] = 1.0
    for j in range(1, h):
        value = theta[j - 1] if j - 1 < theta.size else 0.0
        for i in range(1, min(j, phi.size) + 1):
            value += phi[i - 1] * psi[j - i]
        psi[j] = value
    return psi


def _integrated_psi(psi: np.ndarray, d: int) -> np.ndarray:
    """차분을 되돌린 계열의 ψ. 누적합을 `d` 번 취한다.

    (1−B)^d 로 나누는 것이 시간영역에서는 누적합이다. d=1 이면 ψ*_j = ψ₀+…+ψ_j 다.
    **이걸 빼먹으면** 차분 모형의 구간이 h 에 따라 넓어지지 않아 터무니없이 좁게 나온다.
    """
    out = psi.copy()
    for _ in range(max(0, int(d))):
        out = np.cumsum(out)
    return out


# ==================================================
# 2. 점예측
# ==================================================
def point(model: Dict, horizon: int = 20) -> np.ndarray:
    """h일 점예측 — **원계열 단위(가격)** 로 돌려준다 (명세서 §4.1).

    차분된 축에서 재귀로 앞날을 만든 뒤 역차분한다.

        ŵ_{T+h} = c + Σ φ_i·ŵ_{T+h−i} + Σ θ_j·ε̂_{T+h−j}

    아직 안 온 충격은 기대값이 0 이므로 ε̂_{T+h−j} = 0 (h > j) 으로 둔다.
    "미래 충격을 0 으로 본다" 는 것이 점예측이 h가 커질수록 평평해지는 이유다.
    """
    h = max(1, int(horizon))
    if not model:
        return np.empty(0)

    if model["kind"] == "random_walk":
        # 표류항만큼 일정하게 나아간다
        last = float(model["anchors"][0])
        drift = float(model.get("drift") or 0.0)
        return np.array([last + drift * (i + 1) for i in range(h)], dtype=float)

    phi = np.asarray(model.get("phi") if model.get("phi") is not None else [], dtype=float)
    theta = np.asarray(model.get("theta") if model.get("theta") is not None else [], dtype=float)
    const = float(model.get("const") or 0.0)
    history = np.asarray(model.get("history"), dtype=float)
    resid = model.get("resid_aligned")
    resid = np.asarray(resid, dtype=float) if resid is not None else np.zeros(history.size)

    predictions: List[float] = []
    for step in range(1, h + 1):
        value = const
        for i in range(1, phi.size + 1):
            # step−i ≤ 0 이면 아직 관측 구간 — 실제 값을 쓴다
            value += phi[i - 1] * (predictions[step - i - 1] if step - i >= 1
                                   else float(history[step - i - 1]))
        for j in range(1, theta.size + 1):
            if step - j >= 1:
                continue                     # 미래 충격 → 기대값 0
            past = resid[step - j - 1]
            value += theta[j - 1] * (0.0 if not np.isfinite(past) else float(past))
        predictions.append(value)

    forecast = np.asarray(predictions, dtype=float)
    anchors = model.get("anchors") or []
    return transform.integrate(forecast, anchors) if anchors else forecast


# ==================================================
# 3. 신뢰구간
# ==================================================
def interval(model: Dict, horizon: int = 20, alpha: float = 0.05) -> Dict:
    """ψ-weight 로 신뢰구간을 낸다 (명세서 §4.1).

        Var(e_h) = σ² · Σ_{j=0}^{h−1} (ψ*_j)²         구간 = 점예측 ± z_{α/2}·√Var

    ⚠️ 이 구간은 **모수를 정확히 안다고 가정**한 값이다. 계수 추정오차와 차수 선택의
    불확실성은 들어 있지 않아 **실제보다 좁다.** 그 사실을 `limitation` 으로 함께 낸다.
    """
    h = max(1, int(horizon))
    center = point(model, h)
    if center.size == 0 or not model:
        return {"point": [], "lower": [], "upper": [], "se": [], "alpha": alpha}

    sigma2 = float(model.get("sigma2") or 0.0)
    d = int(model.get("d") or 0)

    if model["kind"] == "random_walk":
        # 확률보행은 ψ ≡ 1 이라 분산이 h 에 정비례한다 (√h 법칙)
        psi_star = np.ones(h)
    else:
        # ⚠️ `model.get("phi") or []` 로 쓰면 안 된다. phi 는 numpy 배열이라
        #    `or` 가 진리값을 물어보고 원소가 둘 이상이면 ValueError 가 난다.
        phi = model.get("phi")
        theta = model.get("theta")
        psi_star = _integrated_psi(
            psi_weights(phi if phi is not None else [],
                        theta if theta is not None else [], h), d)

    variance = sigma2 * np.cumsum(psi_star ** 2)
    se = np.sqrt(np.maximum(variance, 0.0))
    z = norm_ppf(1.0 - alpha / 2.0)

    return {
        "point": [float(v) for v in center],
        "lower": [float(c - z * s) for c, s in zip(center, se, strict=False)],
        "upper": [float(c + z * s) for c, s in zip(center, se, strict=False)],
        "se": [float(s) for s in se],
        "alpha": alpha,
        "z": z,
        "limitation": (
            "신뢰구간은 모수를 정확히 안다고 가정한 값입니다. 계수 추정오차와 차수 선택의 "
            "불확실성이 빠져 있어 **실제보다 좁습니다.** 구간 밖으로 나갈 확률은 "
            f"명목 {alpha:.0%} 보다 큽니다."
        ),
    }


# ==================================================
# 4. 상승확률
# ==================================================
def direction_prob(model: Dict, horizon: int = 20) -> Dict:
    """h일 뒤 **현재보다 높을 확률** (명세서 §4.1).

        P(y_{T+h} > y_T) = Φ( (점예측_h − y_T) / σ_h )

    예측오차가 정규분포라는 가정 위에 선다. 주가 수익률은 실제로는 꼬리가 두꺼워
    **극단적인 움직임의 확률이 이 계산보다 크다.** 그래서 55% 같은 값을 "이길 확률" 로
    읽으면 안 되고, "이 모형이 보는 방향의 기울기" 정도로 읽어야 한다.
    """
    band = interval(model, horizon)
    if not band.get("point"):
        return {"up_prob": None, "horizon": horizon, "available": False}

    levels = np.asarray(model.get("levels") if model.get("levels") is not None else [],
                        dtype=float)
    current = float(levels[-1]) if levels.size else float(model["anchors"][0])

    last_point = band["point"][-1]
    last_se = band["se"][-1]
    if last_se <= 0:
        prob = 1.0 if last_point > current else 0.0
    else:
        prob = norm_sf((current - last_point) / last_se)

    # 하루 단위로도 낸다 — "20일 뒤" 보다 "매일" 이 직관적인 독자가 있다
    daily = [
        float(norm_sf((current - p) / s)) if s > 0 else (1.0 if p > current else 0.0)
        for p, s in zip(band["point"], band["se"], strict=False)
    ]

    return {
        "available": True,
        "up_prob": float(prob),
        "current": current,
        "horizon": int(horizon),
        "path": daily,
        "limitation": ("예측오차가 정규분포라는 가정 위의 값입니다. 실제 주가 수익률은 "
                       "꼬리가 두꺼워 극단적 움직임의 확률이 이보다 큽니다."),
    }


# ==================================================
# 5. 시나리오 3단 (머리말의 규칙)
# ==================================================
def _technical_levels(prices: Sequence, values: Optional[Sequence] = None,
                      lookback: int = SCENARIO_LOOKBACK) -> Dict:
    """시나리오의 경계와 트리거가 볼 **기술적 수준**을 뽑는다."""
    px = transform.to_array(prices)
    px = px[np.isfinite(px)]
    window = px[-lookback:] if px.size > lookback else px

    ma = transform.moving_average(px, TRIGGER_MA)
    ma_last = next((v for v in reversed(ma) if v is not None), None)

    value_avg = None
    if values is not None:
        vv = transform.to_array(values)
        vv = vv[np.isfinite(vv)]
        if vv.size:
            tail = vv[-TRIGGER_MA:] if vv.size > TRIGGER_MA else vv
            value_avg = float(np.mean(tail))

    return {
        "current": float(px[-1]) if px.size else None,
        "resistance": float(np.max(window)) if window.size else None,
        "support": float(np.min(window)) if window.size else None,
        "ma20": ma_last,
        "value_avg20": value_avg,
        "lookback": int(min(lookback, px.size)),
    }


def scenarios(model: Dict, horizon: int = 20,
              prices: Optional[Sequence] = None,
              values: Optional[Sequence] = None,
              lookback: int = SCENARIO_LOOKBACK) -> Dict:
    """Bear / Base / Bull 3단 + 트리거 + 무효화 (명세서 §4.4).

    규칙은 이 모듈 머리말에 적어 두었다. 요약하면
    **경계는 지지·저항(기술적) · 확률과 목표가는 예측분포(통계)** 다.
    """
    band = interval(model, horizon)
    if not band.get("point"):
        return {"available": False, "scenarios": [],
                "reason": "예측을 낼 수 없어 시나리오를 만들지 않았습니다."}

    levels = prices if prices is not None else model.get("levels")
    tech = _technical_levels(levels if levels is not None else [], values, lookback)

    mu = band["point"][-1]
    sigma = band["se"][-1]
    current = tech["current"]
    resistance = tech["resistance"]
    support = tech["support"]

    if sigma <= 0 or current is None or resistance is None or support is None:
        return {"available": False, "scenarios": [], "levels": tech,
                "reason": "지지·저항 또는 예측 표준오차를 구하지 못했습니다."}

    # ── 확률: 기술적 경계를 예측분포로 읽는다 ──
    bull_prob = norm_sf((resistance - mu) / sigma)
    bear_prob = norm_cdf((support - mu) / sigma)
    base_prob = max(0.0, 1.0 - bull_prob - bear_prob)

    # ── 목표가: 각 영역의 조건부 기대값 (절단정규 평균) ──
    bull_target = truncated_mean(mu, sigma, lower=resistance)
    bear_target = truncated_mean(mu, sigma, upper=support)
    base_target = truncated_mean(mu, sigma, lower=support, upper=resistance)

    def pct(target: float) -> float:
        return (target / current - 1.0) * 100.0 if current else 0.0

    ma20 = tech["ma20"]
    ma_text = f"{ma20:,.0f}" if ma20 else "20일선"
    value_text = (
        f"거래대금이 20일 평균({tech['value_avg20']:,.0f})의 "
        f"{VOLUME_SURGE_MULTIPLE}배 이상"
        if tech["value_avg20"]
        else f"거래대금이 20일 평균의 {VOLUME_SURGE_MULTIPLE}배 이상"
    )

    items = [
        {
            "name": "Bull",
            "label": "상승",
            "target": round(bull_target),
            "target_pct": round(pct(bull_target), 2),
            "prob": round(float(bull_prob), 4),
            "boundary": round(resistance),
            "boundary_reason": f"최근 {tech['lookback']}거래일 고점(저항)",
            "trigger": (f"종가가 저항 {resistance:,.0f}원을 상향 돌파하고, "
                        f"같은 날 {value_text}일 것"),
            "invalidation": (f"종가가 20일선({ma_text})을 하향 이탈하면 상승 가설을 폐기한다"
                             if ma20 else "종가가 20일선을 하향 이탈하면 상승 가설을 폐기한다"),
        },
        {
            "name": "Base",
            "label": "횡보",
            "target": round(base_target),
            "target_pct": round(pct(base_target), 2),
            "prob": round(float(base_prob), 4),
            "boundary": None,
            "boundary_reason": f"지지 {support:,.0f} ~ 저항 {resistance:,.0f} 사이",
            "trigger": (f"종가가 {support:,.0f}~{resistance:,.0f}원 구간 안에 머물고 "
                        f"20일선({ma_text}) 근처에서 등락할 것"),
            "invalidation": "어느 쪽이든 구간을 종가로 이탈하면 상승·하락 시나리오로 넘어간다",
        },
        {
            "name": "Bear",
            "label": "하락",
            "target": round(bear_target),
            "target_pct": round(pct(bear_target), 2),
            "prob": round(float(bear_prob), 4),
            "boundary": round(support),
            "boundary_reason": f"최근 {tech['lookback']}거래일 저점(지지)",
            "trigger": (f"종가가 지지 {support:,.0f}원을 하향 이탈하고 "
                        f"20일선({ma_text}) 아래에 머무를 것"),
            "invalidation": (f"종가가 저항 {resistance:,.0f}원을 회복하면 하락 가설을 폐기한다"),
        },
    ]

    return {
        "available": True,
        "horizon": int(horizon),
        "current": current,
        "point": mu,
        "sigma": sigma,
        "levels": tech,
        "scenarios": items,
        "rule": ("경계는 최근 고점·저점(기술적), 확률과 목표가는 예측분포(통계)에서 냅니다. "
                 "σ 배수로 영역을 나누면 확률이 언제나 31/38/31%로 고정돼 정보가 없기 때문입니다."),
        "limitation": ("확률은 예측오차가 정규분포라는 가정 위의 값이고, 신뢰구간과 마찬가지로 "
                       "계수 추정오차가 빠져 있습니다. 트리거는 관측 가능한 규칙이지만 "
                       "미래를 보장하지 않습니다."),
    }
