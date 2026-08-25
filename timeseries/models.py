"""모형 적합 — AR · ARIMA · 확률보행 (시계열 엔진) — 명세서 §4.1 · §4.3 · 05-8 · 05-9

MLE(`scipy.optimize`)를 못 쓰므로 **최소제곱 두 번**으로 ARIMA 를 맞춘다 (Hannan-Rissanen).
정확도는 MLE 보다 낮다. 그래서 **항상 확률보행과 백테스트로 겨루고, 못 이기면 그대로 적는다**
(기획서 D6). 이 모듈이 `random_walk` 를 함께 두는 이유가 그것이다.

Hannan-Rissanen 이 왜 두 단계인가 (명세서 §4.3)
---------------------------------------------
MA 항 θ·ε_{t−1} 의 ε 은 **관측되지 않는다.** 회귀를 하려면 설명변수가 있어야 하는데
그 자리에 넣을 값이 없다. 그래서 먼저 대신할 값을 만든다.

    1단계  고차 AR(k) 를 맞춘다 (k ≈ ln(n)²).  MA 는 무한차 AR 로 근사되므로
           차수를 충분히 높이면 그 잔차 ê 가 진짜 ε 에 가까워진다
    2단계  w_t 를 [w_{t−1..t−p}, ê_{t−1..t−q}] 에 최소제곱 회귀 → (φ, θ)
    3단계  2단계 잔차가 백색잡음인지 Ljung-Box 로 확인. 아니면 차수를 다시 찾는다

1단계 잔차는 진짜 ε 이 아니라 **추정치**다. 그래서 2단계 계수에 추가 오차가 실린다.
MLE 처럼 둘을 동시에 최적화하지 않기 때문이다. 이 한계는 리포트에 그대로 나간다.

모형 딕셔너리 규약
----------------
적합 결과는 `forecast` 가 그대로 받아 쓸 수 있게 다음을 담는다.

    kind        "ar" · "arima" · "random_walk"
    p, d, q     차수
    const       상수항
    phi, theta  AR·MA 계수 (numpy 배열)
    sigma2      잔차 분산 — 신뢰구간 폭의 출발점
    resid       2단계 잔차
    history     차분된 계열 전체 (예측 재귀의 초기값)
    anchors     역차분용 앵커 (`transform.anchors_for`)
    stable      AR 다항식 근이 단위원 밖인가 — 아니면 예측이 발산한다
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from timeseries import stationarity, transform
from timeseries.numerics import chi2_sf

# Ljung-Box 에서 볼 시차 수. 일봉이라 한 달(≈20거래일)치를 본다.
LJUNG_BOX_LAGS = 20

# 격자탐색 기본 상한. 일봉 주가에서 차수가 3을 넘는 모형이 이기는 일은 드물고,
# 후보를 늘릴수록 표본에만 맞는 모형(과적합)을 고를 위험이 커진다.
DEFAULT_MAX_P = 3
DEFAULT_MAX_Q = 3


# ==================================================
# 0. 공통 도구
# ==================================================
def _ols(design: np.ndarray, target: np.ndarray) -> Optional[np.ndarray]:
    """최소제곱 한 번. 설계행렬이 특이하면 `None`."""
    try:
        beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    except np.linalg.LinAlgError:
        return None
    return beta if np.all(np.isfinite(beta)) else None


def _is_stable(phi: Sequence[float]) -> bool:
    """AR 다항식 1 − φ₁z − … − φ_p z^p 의 근이 **모두 단위원 밖**인가.

    하나라도 안쪽이면 예측이 시간에 따라 발산한다 — 20일 뒤 목표가가 수십만 원이 되는 식이다.
    Hannan-Rissanen 은 최소제곱이라 이 조건을 강제하지 않으므로 여기서 직접 확인하고,
    불안정하면 `select_order` 가 그 후보를 떨어뜨린다.
    """
    phi = np.asarray(phi, dtype=float)
    if phi.size == 0:
        return True
    # numpy 의 계수 순서는 높은 차수부터다: [−φ_p, …, −φ₁, 1]
    coefficients = np.concatenate((-phi[::-1], [1.0]))
    try:
        roots = np.roots(coefficients)
    except (np.linalg.LinAlgError, ValueError):
        return False
    return bool(np.all(np.abs(roots) > 1.0 + 1e-8))


def _information_criteria(ssr: float, nobs: int, n_params: int) -> Dict:
    """AIC · BIC. **같은 표본(nobs)에서 잰 것끼리만 비교할 수 있다.**

        AIC = n·ln(SSR/n) + 2k        BIC = n·ln(SSR/n) + k·ln(n)

    `k` 는 추정한 모수 수 + 1(분산)이다. 상수가 하나 더 붙고 덜 붙고는 후보를 견줄 때
    똑같이 상쇄되므로 순위에 영향이 없다.
    """
    if nobs <= 0 or ssr <= 0:
        return {"aic": math.inf, "bic": math.inf, "sigma2": 0.0, "nobs": int(nobs)}
    sigma2 = ssr / nobs
    k = n_params + 1
    return {
        "aic": nobs * math.log(sigma2) + 2.0 * k,
        "bic": nobs * math.log(sigma2) + k * math.log(nobs),
        "sigma2": sigma2,
        "nobs": int(nobs),
    }


def ljung_box(resid: Sequence, lags: int = LJUNG_BOX_LAGS,
              n_params: int = 0) -> Dict:
    """Ljung-Box 검정 — **잔차에 아직 자기상관이 남아 있는가** (명세서 §4.3 3단계).

        Q = n(n+2) · Σ_{k=1}^{h} r_k² / (n − k)      ~  χ²(h − 모수 수)

    귀무가설은 "잔차가 백색잡음이다". 기각되면(p < 0.05) **모형이 아직 못 걷어낸 구조가
    남아 있다**는 뜻이라 차수를 다시 찾아야 한다.

    ⚠️ p값은 `numerics.chi2_sf` 의 Wilson–Hilferty 근사다. 자유도가 작으면 오차가 커진다.
    """
    x = np.asarray(resid, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    lags = max(1, min(int(lags), max(1, n // 4)))       # 시차가 n/4 를 넘으면 추정이 불안정하다

    if n < 8:
        return {"available": False, "stat": None, "p_value": None, "lags": lags,
                "df": None, "white_noise": None,
                "note": "표본이 짧아 Ljung-Box 를 낼 수 없습니다."}

    centered = x - x.mean()
    denominator = float(centered @ centered)
    if denominator <= 0:
        return {"available": False, "stat": None, "p_value": None, "lags": lags,
                "df": None, "white_noise": None,
                "note": "잔차 분산이 0입니다."}

    stat = 0.0
    for k in range(1, lags + 1):
        r_k = float(centered[k:] @ centered[:-k]) / denominator
        stat += r_k * r_k / (n - k)
    stat *= n * (n + 2)

    df = max(1, lags - int(n_params))
    p_value = chi2_sf(stat, df)

    return {
        "available": True,
        "stat": stat,
        "p_value": p_value,
        "lags": lags,
        "df": df,
        # p ≥ 0.05 면 "백색잡음이 아니라고 말할 근거가 없다" — 채택이 아니라 미기각이다
        "white_noise": bool(p_value >= 0.05),
        "note": ("잔차에서 자기상관을 찾지 못했습니다 (p ≥ 0.05) — 모형이 구조를 걷어냈습니다."
                 if p_value >= 0.05 else
                 f"잔차에 자기상관이 남아 있습니다 (p={p_value:.4f} < 0.05) — 차수를 다시 볼 필요가 있습니다."),
        "limitation": "p값은 Wilson–Hilferty 근사입니다 (자유도가 작을수록 오차가 큽니다).",
    }


# ==================================================
# 1. AR — 최소제곱 한 번
# ==================================================
def fit_ar(values: Sequence, p: int = 1, sample_start: Optional[int] = None) -> Optional[Dict]:
    """AR(p) 를 최소제곱으로 맞춘다. → 모형 딕셔너리 (명세서 §4.1)

        y_t = c + φ₁y_{t−1} + … + φ_p y_{t−p} + ε_t

    `sample_start` 를 주면 그 자리부터의 관측만 쓴다. 여러 차수를 AIC 로 견줄 때
    **모든 후보가 같은 표본을 보게** 맞추는 데 쓴다 (안 맞추면 차수가 낮을수록 유리해진다).
    """
    y = transform.to_array(values)
    y = y[np.isfinite(y)]
    n = y.size
    p = max(0, int(p))
    start = max(p, int(sample_start) if sample_start is not None else p)

    rows = n - start
    if rows <= p + 2:
        return None

    target = y[start:]
    columns = [np.ones(rows)]
    for i in range(1, p + 1):
        columns.append(y[start - i:n - i])
    design = np.column_stack(columns)

    beta = _ols(design, target)
    if beta is None:
        return None

    resid = target - design @ beta
    ssr = float(resid @ resid)
    info = _information_criteria(ssr, rows, p + 1)
    phi = beta[1:]

    return {
        "kind": "ar",
        "p": p, "d": 0, "q": 0,
        "label": f"AR({p})",
        "const": float(beta[0]),
        "phi": phi,
        "theta": np.empty(0),
        "sigma2": info["sigma2"],
        "resid": resid,
        "history": y,
        "anchors": [],
        "nobs": info["nobs"],
        "aic": info["aic"],
        "bic": info["bic"],
        "stable": _is_stable(phi),
        "ljung_box": ljung_box(resid, n_params=p),
    }


# ==================================================
# 2. ARIMA — Hannan-Rissanen 2단계
# ==================================================
def _stage1_order(n: int) -> int:
    """1단계 AR 차수 k ≈ ln(n)² (명세서 §4.3).

    표본이 250이면 k ≈ 30 이다. MA 를 근사하려면 충분히 높아야 하지만, 너무 높으면
    2단계에 쓸 관측이 남지 않는다. 그래서 표본의 1/5 을 넘지 않게 막는다.
    """
    if n < 10:
        return 1
    k = int(round(math.log(n) ** 2))
    return max(1, min(k, n // 5))


def fit_arima(values: Sequence, p: int = 1, d: int = 1, q: int = 1,
              sample_start: Optional[int] = None) -> Optional[Dict]:
    """ARIMA(p,d,q) 를 **Hannan-Rissanen 2단계 최소제곱**으로 맞춘다 (명세서 §4.3).

    `values` 는 **원계열(가격)** 이다. 차분은 이 함수가 안에서 한다 —
    부르는 쪽이 차분해서 넘기면 역차분용 앵커를 잃어버려 예측을 가격으로 되돌릴 수 없다.
    """
    y = transform.to_array(values)
    y = y[np.isfinite(y)]
    p, d, q = max(0, int(p)), max(0, int(d)), max(0, int(q))

    if q == 0:
        # MA 항이 없으면 2단계가 필요 없다. 차분한 계열에 AR 을 맞추고 앵커만 붙인다.
        fitted = fit_ar(transform.difference(y, d), p, sample_start=sample_start)
        if fitted is None:
            return None
        fitted.update({
            "kind": "arima", "d": d, "q": 0,
            "label": f"ARIMA({p},{d},0)",
            "anchors": transform.anchors_for(y, d),
            "levels": y,
            "stage1_order": 0,
        })
        return fitted

    w = transform.difference(y, d) if d else y
    n = w.size
    if n < 30:
        return None

    # ── 1단계: 고차 AR(k) 로 ε 을 만든다 ──
    k = _stage1_order(n)
    stage1 = fit_ar(w, k)
    if stage1 is None:
        return None

    # 잔차를 원래 시간 축에 맞춰 놓는다. AR(k) 잔차는 t = k … n−1 자리의 값이라
    # 앞 k칸을 비워 두어야 아래에서 시차를 맞출 때 어긋나지 않는다.
    eps = np.full(n, np.nan)
    eps[k:] = stage1["resid"]

    # ── 2단계: w_t ~ [1, w_{t−1..t−p}, ê_{t−1..t−q}] ──
    # ê 는 t ≥ k 부터 있으므로 ê_{t−q} 가 존재하려면 t ≥ k + q 여야 한다.
    minimum_start = max(k + q, p)
    start = max(minimum_start, int(sample_start) if sample_start is not None else 0)
    rows = n - start
    if rows <= p + q + 2:
        return None

    target = w[start:]
    columns = [np.ones(rows)]
    for i in range(1, p + 1):
        columns.append(w[start - i:n - i])
    for j in range(1, q + 1):
        columns.append(eps[start - j:n - j])
    design = np.column_stack(columns)

    if not np.all(np.isfinite(design)):
        return None

    beta = _ols(design, target)
    if beta is None:
        return None

    resid = target - design @ beta
    ssr = float(resid @ resid)
    info = _information_criteria(ssr, rows, p + q + 1)
    phi = beta[1:1 + p]
    theta = beta[1 + p:1 + p + q]

    # MA 항을 재귀로 예측하려면 ε̂_{T}, ε̂_{T−1}, … 이 **w 와 같은 시간축 위에** 있어야 한다.
    # 2단계 잔차는 t = start … n−1 자리의 값이므로 앞을 비운 배열에 얹어 둔다.
    # (`resid` 만 넘기면 부르는 쪽이 시차를 하나씩 밀어 쓰기 십상이다)
    resid_aligned = np.full(n, np.nan)
    resid_aligned[start:] = resid

    return {
        "kind": "arima",
        "p": p, "d": d, "q": q,
        "label": f"ARIMA({p},{d},{q})",
        "const": float(beta[0]),
        "phi": phi,
        "theta": theta,
        "sigma2": info["sigma2"],
        "resid": resid,
        "history": w,                  # 차분된 계열 — 예측 재귀의 초기값
        "resid_aligned": resid_aligned,  # w 와 같은 시간축의 2단계 잔차 — MA 항 재귀에 쓴다
        "stage1_resid": eps,           # 1단계 AR 잔차 (진단용)
        "resid_start": start,
        "anchors": transform.anchors_for(y, d),
        "levels": y,
        "nobs": info["nobs"],
        "aic": info["aic"],
        "bic": info["bic"],
        "stable": _is_stable(phi),
        "stage1_order": k,
        "ljung_box": ljung_box(resid, n_params=p + q),
        "method": "hannan-rissanen",
        "limitation": (
            "Hannan-Rissanen 2단계 최소제곱입니다. MA 항의 설명변수로 1단계 AR "
            f"({k}차) 의 **잔차 추정치**를 쓰므로 최대우도(MLE)보다 효율이 낮습니다. "
            "그래서 항상 확률보행과 백테스트로 비교합니다."
        ),
    }


# ==================================================
# 3. 확률보행 — 이겨야 할 상대
# ==================================================
def random_walk(values: Sequence, drift: bool = True) -> Optional[Dict]:
    """확률보행 벤치마크 (05-6). `drift=True` 면 표류항을 넣는다.

        y_t = y_{t−1} + μ + ε_t      (μ = 과거 일간 변화의 평균)

    **이것이 기준선이다.** "내일 가격은 오늘 가격" 이라는, 아무 모형도 아닌 예측이다.
    ARIMA 가 이걸 못 이기면 그 ARIMA 는 쓸모가 없다. 효율시장가설이 참에 가까울수록
    이 기준선은 이기기 어렵다 — 못 이기는 것이 이상한 일이 아니며, 그 사실을 감추지 않는다.
    """
    y = transform.to_array(values)
    y = y[np.isfinite(y)]
    if y.size < 3:
        return None

    steps = np.diff(y)
    mu = float(np.mean(steps)) if drift else 0.0
    resid = steps - mu
    ssr = float(resid @ resid)
    info = _information_criteria(ssr, resid.size, 1 if drift else 0)

    return {
        "kind": "random_walk",
        "p": 0, "d": 1, "q": 0,
        "label": "확률보행" + (" (표류 포함)" if drift else ""),
        "const": mu,
        "phi": np.empty(0),
        "theta": np.empty(0),
        "sigma2": info["sigma2"],
        "resid": resid,
        "history": steps,
        "anchors": [float(y[-1])],
        "levels": y,
        "nobs": info["nobs"],
        "aic": info["aic"],
        "bic": info["bic"],
        "stable": True,
        "drift": mu,
        "ljung_box": ljung_box(resid, n_params=1 if drift else 0),
    }


# ==================================================
# 4. 차수 선택 — AIC 격자탐색
# ==================================================
def select_order(values: Sequence, max_p: int = DEFAULT_MAX_P, max_q: int = DEFAULT_MAX_Q,
                 d: Optional[int] = None) -> Dict:
    """(p, d, q) 를 정한다 (명세서 §4.1).

    d 는 ADF 로, (p, q) 는 **AIC 격자탐색**으로 정한다.

    지키는 것 두 가지
      1. **같은 표본** — 후보마다 쓰는 관측이 다르면 AIC 를 비교할 수 없다.
         가장 많이 잘라 내는 후보(p=max_p, q=max_q)에 맞춰 전부 앞을 잘라 둔다.
      2. **불안정한 후보는 버린다** — AR 근이 단위원 안이면 예측이 발산한다.
         AIC 가 아무리 낮아도 쓸 수 없는 모형이다.
    """
    y = transform.to_array(values)
    y = y[np.isfinite(y)]

    d_info = stationarity.suggest_d(y) if d is None else {
        "d": max(0, int(d)), "reason": f"호출한 쪽이 d={d} 로 지정했습니다.", "steps": []}
    d_final = d_info["d"]

    w = transform.difference(y, d_final) if d_final else y
    if w.size < 40:
        return {"available": False, "p": 0, "d": d_final, "q": 0,
                "reason": f"차분 후 표본이 {w.size}개뿐이라 차수를 탐색할 수 없습니다 (40개 필요).",
                "d_reason": d_info.get("reason"), "candidates": []}

    # 모든 후보가 같은 관측을 보도록 시작점을 고정한다 (위 1번)
    k = _stage1_order(w.size)
    fixed_start = max(k + max_q, max_p)

    candidates: List[Dict] = []
    for p in range(0, max_p + 1):
        for q in range(0, max_q + 1):
            if p == 0 and q == 0:
                continue                    # 상수만 있는 모형 — 확률보행이 그 자리를 맡는다
            fitted = fit_arima(y, p=p, d=d_final, q=q, sample_start=fixed_start)
            if fitted is None:
                continue
            candidates.append({
                "p": p, "q": q,
                "aic": fitted["aic"], "bic": fitted["bic"],
                "nobs": fitted["nobs"],
                "stable": fitted["stable"],
                "white_noise": (fitted["ljung_box"] or {}).get("white_noise"),
            })

    usable = [c for c in candidates if c["stable"] and math.isfinite(c["aic"])]
    if not usable:
        # 전부 불안정하면 가장 단순한 모형으로 물러난다. 발산하는 예측을 내보내느니
        # 설명력이 낮더라도 안전한 쪽이 낫다.
        return {
            "available": False, "p": 1, "d": d_final, "q": 0,
            "reason": "안정적인 후보를 찾지 못했습니다 (AR 근이 단위원 안). ARIMA(1,d,0) 으로 물러납니다.",
            "d_reason": d_info.get("reason"),
            "candidates": sorted(candidates, key=lambda c: c["aic"])[:10],
        }

    best = min(usable, key=lambda c: c["aic"])
    ranked = sorted(usable, key=lambda c: c["aic"])

    # 잔차가 백색잡음이 아닌 최적 후보는 그 사실을 밝힌다 (명세서 §4.3 3단계).
    # 차수를 억지로 바꾸지는 않는다 — AIC 가 고른 것을 뒤집으려면 근거가 더 필요하다.
    residual_note = ""
    if best.get("white_noise") is False:
        alternative = next((c for c in ranked if c.get("white_noise")), None)
        if alternative:
            residual_note = (
                f" AIC 최적 ARIMA({best['p']},{d_final},{best['q']}) 는 잔차에 자기상관이 남아 "
                f"ARIMA({alternative['p']},{d_final},{alternative['q']}) 로 바꿉니다 (백색잡음 통과).")
            best = alternative
        else:
            residual_note = " 모든 후보의 잔차에 자기상관이 남아 있습니다 — 예측을 넓게 읽어야 합니다."

    return {
        "available": True,
        "p": best["p"], "d": d_final, "q": best["q"],
        "label": f"ARIMA({best['p']},{d_final},{best['q']})",
        "aic": best["aic"], "bic": best["bic"],
        "nobs": best["nobs"],
        "reason": (f"AIC 격자탐색 {len(candidates)}개 후보 중 최소 "
                   f"(AIC {best['aic']:.1f}, 관측 {best['nobs']}개 고정)." + residual_note),
        "d_reason": d_info.get("reason"),
        "d_steps": d_info.get("steps", []),
        "sample_start": fixed_start,
        "candidates": ranked[:10],
        "rejected_unstable": [c for c in candidates if not c["stable"]],
    }


def fit_best(values: Sequence, max_p: int = DEFAULT_MAX_P,
             max_q: int = DEFAULT_MAX_Q) -> Dict:
    """차수를 고르고 그 차수로 적합까지 마친다. → `{model, benchmark, order}`

    `benchmark`(확률보행)를 **항상 함께** 돌려준다. 기획서 D6 의 "못 이기면 그대로 적는다"를
    지키려면 비교 대상이 늘 옆에 있어야 하기 때문이다.
    """
    order = select_order(values, max_p=max_p, max_q=max_q)
    model = fit_arima(values, p=order["p"], d=order["d"], q=order["q"])

    if model is None:
        # 마지막 물러설 자리 — 1차 차분 AR(1)
        model = fit_arima(values, p=1, d=1, q=0)

    return {
        "order": order,
        "model": model,
        "benchmark": random_walk(values),
    }
