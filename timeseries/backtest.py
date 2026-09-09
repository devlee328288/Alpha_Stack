"""백테스트 — walk-forward 검증 (시계열 엔진) — 명세서 §4.1 · 04강

**적합이 잘 됐다는 것과 예측이 맞는다는 것은 다른 말이다.** AIC 가 낮아도 앞날을 못 맞출
수 있다. 그래서 과거의 어느 시점에 서서 그 다음을 맞혀 보게 하고, 실제와 견준다.

왜 무작위 분할(K-fold)을 쓰면 안 되는가
------------------------------------
시계열을 섞어서 나누면 **미래로 과거를 예측**하게 된다. 8월 자료로 학습해 3월을 맞히는
셈이라, 실제로는 알 수 없었던 정보가 학습에 새어 든다(look-ahead bias). 성능이 실제보다
훨씬 좋게 나오고, 실전에 올리면 그 성능이 사라진다.

그래서 **확장창(expanding window)** 을 쓴다. 시점을 앞으로 밀면서 그때까지의 자료만으로
매번 다시 적합하고, 그 다음 h일을 맞힌다.

    학습 [1 … 120] → 예측 [121 … 125]     실제와 비교
    학습 [1 … 130] → 예측 [131 … 135]     …
    학습 [1 … 140] → 예측 [141 … 145]     …

확률보행과 반드시 함께 잰다 (기획서 D6)
------------------------------------
ARIMA 의 RMSE 가 얼마인지는 그 자체로 의미가 없다. **아무것도 안 하는 예측**(내일 가격은
오늘 가격)보다 나은지가 전부다. 못 이기면 리포트에 그대로 적는다. 주가는 효율시장에
가까워서 못 이기는 것이 정상에 가깝다 — 그 사실을 감추면 리포트가 거짓말이 된다.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from timeseries import forecast, models, transform

# 첫 학습 구간의 최소 길이. 이보다 짧으면 ARIMA 차수 추정이 의미가 없다.
MIN_TRAIN = 120

# 기본 폴드 수. 늘릴수록 추정이 안정되지만 매번 다시 적합하므로 그만큼 느려진다.
DEFAULT_FOLDS = 12


def _metrics(predicted: Sequence[float], actual: Sequence[float],
             origin: Sequence[float]) -> Dict:
    """RMSE · MAE · 방향 적중률.

    `origin` 은 각 예측의 **출발점**(학습 구간 마지막 종가)이다. 방향 적중은
    "출발점 대비 오를 것인가" 를 맞혔는지로 잰다 — 값의 오차와 별개로 쓸모가 있다.
    (수치는 조금 틀려도 방향이 맞으면 매매 판단에는 도움이 된다)
    """
    p = np.asarray(predicted, dtype=float)
    a = np.asarray(actual, dtype=float)
    o = np.asarray(origin, dtype=float)

    mask = np.isfinite(p) & np.isfinite(a) & np.isfinite(o)
    p, a, o = p[mask], a[mask], o[mask]
    if p.size == 0:
        return {"n": 0, "rmse": None, "mae": None, "hit_rate": None, "mape": None}

    errors = p - a
    # 방향이 같은지 — 실제가 보합(정확히 같음)인 날은 판정에서 뺀다
    moved = a != o
    hits = np.sign(p - o)[moved] == np.sign(a - o)[moved]

    with np.errstate(divide="ignore", invalid="ignore"):
        ape = np.abs(errors / np.where(a != 0, a, np.nan))

    return {
        "n": int(p.size),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mae": float(np.mean(np.abs(errors))),
        "mape": float(np.nanmean(ape) * 100.0) if np.any(np.isfinite(ape)) else None,
        "hit_rate": float(np.mean(hits)) if hits.size else None,
        "direction_n": int(hits.size),
    }


def walk_forward(prices: Sequence, order: Optional[Tuple[int, int, int]] = None,
                 horizon: int = 5, folds: int = DEFAULT_FOLDS,
                 min_train: int = MIN_TRAIN) -> Dict:
    """확장창 walk-forward 검증 → `{rmse, mae, hit_rate, vs_rw}` (명세서 §4.1)

    `order` 를 주지 않으면 **첫 학습 구간에서 한 번** 차수를 고르고 이후 폴드에 그대로 쓴다.
    폴드마다 차수를 다시 고르면 정직하지만 12배 느리고, 무엇보다 "어떤 모형을 평가한
    것인지" 가 흐려진다. 차수 선택도 검증하고 싶다면 `order=None` 이 아니라 바깥에서
    구간을 둘로 나눠 돌리는 편이 낫다.
    """
    y = transform.to_array(prices)
    y = y[np.isfinite(y)]
    n = y.size
    horizon = max(1, int(horizon))
    folds = max(1, int(folds))

    if n < min_train + horizon + 1:
        return {
            "available": False,
            "reason": (f"표본이 {n}개로 백테스트에 부족합니다 "
                       f"(최소 학습 {min_train} + 예측 {horizon} 필요)."),
            "folds": 0, "rmse": None, "mae": None, "hit_rate": None, "vs_rw": None,
        }

    # 폴드를 균등하게 배치한다. 자료가 짧으면 폴드 수를 줄여서라도 돌린다.
    room = n - min_train - horizon
    if room < folds:
        folds = max(1, room + 1)
    step = max(1, room // folds) if folds > 1 else 1

    # ⚠️ 폴드 간격이 예측기간보다 좁으면 **평가 구간이 서로 겹친다.**
    #
    # 실측 — 배포본 축약본(150행)에서 min_train=120·horizon=10·folds=12 로 돌리면
    # step 이 1이 되어 학습집합이 120~132 로 거의 같아진다. 12회 시행처럼 보이지만
    # 사실상 한 번의 시행이고, 그래서 적중률이 15.8%(로컬 282행에서는 62.7%)까지 튀었다.
    # 숫자를 지우지는 않는다 — 그 숫자가 **독립 시행 12회의 결과가 아니라는 사실**을 밝힌다.
    overlapping = step < horizon
    independent_folds = max(1, room // horizon) if horizon else folds

    model_pred: List[float] = []
    rw_pred: List[float] = []
    actuals: List[float] = []
    origins: List[float] = []
    fold_rows: List[Dict] = []

    # 차수는 첫 학습 구간에서 정한다 (미래를 보지 않는다)
    if order is None:
        chosen = models.select_order(y[:min_train])
        order = (chosen["p"], chosen["d"], chosen["q"])
        order_reason = chosen.get("reason", "")
    else:
        order_reason = "호출한 쪽이 차수를 지정했습니다."

    p, d, q = order
    failures = 0

    for i in range(folds):
        cut = min_train + i * step
        if cut + horizon > n:
            break

        train = y[:cut]
        truth = y[cut:cut + horizon]
        origin = float(train[-1])

        fitted = models.fit_arima(train, p=p, d=d, q=q)
        benchmark = models.random_walk(train)
        if fitted is None or benchmark is None:
            failures += 1
            continue

        forecast_model = forecast.point(fitted, horizon)
        forecast_rw = forecast.point(benchmark, horizon)
        if forecast_model.size < horizon or forecast_rw.size < horizon:
            failures += 1
            continue

        model_pred.extend(float(v) for v in forecast_model[:horizon])
        rw_pred.extend(float(v) for v in forecast_rw[:horizon])
        actuals.extend(float(v) for v in truth)
        origins.extend([origin] * len(truth))

        fold_rows.append({
            "fold": i + 1,
            "train_end": int(cut),
            "train_size": int(cut),
            "origin": origin,
            "predicted": float(forecast_model[horizon - 1]),
            "actual": float(truth[-1]),
            "rw_predicted": float(forecast_rw[horizon - 1]),
        })

    if not fold_rows:
        return {
            "available": False,
            "reason": f"모든 폴드에서 적합에 실패했습니다 (실패 {failures}회).",
            "folds": 0, "rmse": None, "mae": None, "hit_rate": None, "vs_rw": None,
        }

    model_metrics = _metrics(model_pred, actuals, origins)
    rw_metrics = _metrics(rw_pred, actuals, origins)

    # 확률보행 대비 RMSE 개선률. 양수면 이긴 것이다.
    vs_rw = None
    if model_metrics["rmse"] is not None and rw_metrics["rmse"]:
        vs_rw = (rw_metrics["rmse"] - model_metrics["rmse"]) / rw_metrics["rmse"] * 100.0

    beats = bool(vs_rw is not None and vs_rw > 0)

    return {
        "available": True,
        "order": {"p": p, "d": d, "q": q},
        "order_reason": order_reason,
        "label": f"ARIMA({p},{d},{q})",
        "folds": len(fold_rows),
        "horizon": horizon,
        "min_train": int(min_train),
        "step": int(step),
        "overlapping": bool(overlapping),
        "independent_folds": int(min(independent_folds, len(fold_rows))),
        "points": model_metrics["n"],
        "rmse": model_metrics["rmse"],
        "mae": model_metrics["mae"],
        "mape": model_metrics["mape"],
        "hit_rate": model_metrics["hit_rate"],
        "random_walk": rw_metrics,
        "vs_rw": vs_rw,
        "vs_random_walk": (f"{vs_rw:+.1f}%" if vs_rw is not None else None),
        "beats_random_walk": beats,
        "failed_folds": failures,
        # 기획서 D6 — 못 이기면 그대로 적는다. 이 문장이 리포트에 그대로 실린다.
        "verdict": (
            f"ARIMA({p},{d},{q}) 가 확률보행보다 RMSE 를 {vs_rw:.1f}% 줄였습니다."
            if beats else
            f"ARIMA({p},{d},{q}) 가 확률보행을 이기지 못했습니다 "
            f"(RMSE {model_metrics['rmse']:,.1f} vs 확률보행 {rw_metrics['rmse']:,.1f}). "
            "이 종목·기간에서는 시계열 모형의 예측력이 '내일도 오늘과 같다' 보다 낫지 않습니다."
        ) if vs_rw is not None else "확률보행과 비교할 수 없었습니다.",
        "folds_detail": fold_rows,
        "limitation": (
            "차수는 첫 학습 구간에서 한 번 고르고 전 폴드에 같은 차수를 썼습니다. "
            "적중률은 출발점(학습 마지막 종가) 대비 방향이며, 보합인 날은 판정에서 뺐습니다."
            + (f" ⚠️ 표본이 {n}행뿐이라 폴드 간격이 {step}일로 좁아져 "
               f"**{len(fold_rows)}개 폴드의 평가 구간이 서로 겹칩니다**(예측기간 {horizon}일). "
               "학습 집합도 거의 같아 사실상 독립 시행은 "
               f"{min(independent_folds, len(fold_rows))}회 수준입니다 — "
               "적중률과 RMSE 를 '측정값'이 아니라 **한 구간의 결과**로 읽어야 합니다. "
               "긴 시계열(로컬 원본 캐시)에서 다시 재면 크게 달라질 수 있습니다."
               if overlapping else "")
        ),
    }
