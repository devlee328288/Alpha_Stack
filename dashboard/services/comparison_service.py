# dashboard/services/comparison_service.py
"""4모델 결과 비교 + best 선택. 정렬 기준은 노트북 05.모델비교와 동일."""

from __future__ import annotations

import numpy as np
import pandas as pd

COMPARISON_COLUMNS = {
    "model": "MODEL",
    "accuracy": "ACC",
    "macro_f1": "MACRO F1",
    "down_recall": "DOWN RECALL",
    "core_harmonic_mean": "HARMONIC",
    "balanced_accuracy": "BAL ACC",
    "majority_accuracy": "MAJORITY",
    "delta_sharpe_net_median": "ΔSHARPE",
    "all_cash_folds": "CASH FOLDS",
}


def build_comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for model_name, r in results.items():
        s = r["summary"]
        rows.append(
            {
                "model": s.get("model", model_name),
                "accuracy": float(s["accuracy"]),
                "macro_f1": float(s["macro_f1"]),
                "down_recall": float(s["down_recall"]),
                "core_harmonic_mean": float(s["core_harmonic_mean"]),
                "balanced_accuracy": float(s["balanced_accuracy"]),
                "majority_accuracy": float(s["majority_accuracy"]),
                "delta_sharpe_net_median": float(s["delta_sharpe_net_median"]),
                "all_cash_folds": int(s["all_cash_folds"]),
            }
        )
    df = (
        pd.DataFrame(rows)
        .sort_values("core_harmonic_mean", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    return df


def best_model(results: dict[str, dict]) -> str | None:
    if not results:
        return None
    return str(build_comparison_table(results).iloc[0]["model"])


def build_summary_kpis(results: dict[str, dict]) -> dict:
    if not results:
        return {}
    best = best_model(results)
    best_summary = results[best]["summary"]
    return {
        "best_model": best,
        "best_harmonic": float(best_summary["core_harmonic_mean"]),
        "best_accuracy": float(best_summary["accuracy"]),
        "best_macro_f1": float(best_summary["macro_f1"]),
        "best_down_recall": float(best_summary["down_recall"]),
        "best_delta_sharpe": float(best_summary["delta_sharpe_net_median"]),
        "oos_rows": int(best_summary["oos_rows"]),
        "n_models": len(results),
    }


def build_cross_table(baseline: dict, models: dict) -> dict:
    """
    Baseline + 모든 모델을 두 개의 표로.
    - strategy: Sharpe / CAGR / MDD / Win Rate / ΔSharpe
    - classification: ACC / Macro F1 / Bal Acc / Down Recall / Majority
    """
    perf = baseline.get("perf_metrics", {})
    cls = baseline.get("cls_metrics", {})
    thr = baseline.get("threshold", 0.01)
    b_name = f"6-PARAM {thr*100:.0f}%"

    strategy_rows = [
        {
            "SOURCE": "BASELINE",
            "MODEL": b_name,
            "SHARPE": perf.get("sharpe"),
            "CAGR": perf.get("cagr"),
            "MDD": perf.get("mdd"),
            "WIN RATE": perf.get("win_rate"),
            "ΔSHARPE": None,
        }
    ]
    cls_rows = [
        {
            "SOURCE": "BASELINE",
            "MODEL": b_name,
            "ACC": None,
            "MACRO F1": cls.get("f1_macro"),
            "BAL ACC": cls.get("balanced_acc"),
            "DOWN REC": None,
            "MAJORITY": None,
        }
    ]

    for name, r in models.items():
        s = r["summary"]
        strategy_rows.append(
            {
                "SOURCE": "MODEL",
                "MODEL": name,
                "SHARPE": None,
                "CAGR": None,
                "MDD": None,
                "WIN RATE": None,
                "ΔSHARPE": s.get("delta_sharpe_net_median"),
            }
        )
        cls_rows.append(
            {
                "SOURCE": "MODEL",
                "MODEL": name,
                "ACC": s.get("accuracy"),
                "MACRO F1": s.get("macro_f1"),
                "BAL ACC": s.get("balanced_accuracy"),
                "DOWN REC": s.get("down_recall"),
                "MAJORITY": s.get("majority_accuracy"),
            }
        )

    return {
        "strategy": pd.DataFrame(strategy_rows),
        "classification": pd.DataFrame(cls_rows),
    }


# ═══════════════════════════════════════════════════════════
# 문제 2: 기준선 비교 (fwd_return / adaptive)
# ═══════════════════════════════════════════════════════════


def build_baseline_comparison(
    baseline_adaptive: dict | None = None,
    fwd_labels: list | None = None,
) -> pd.DataFrame:
    """
    두 기준선 (fwd_return / adaptive)의 라벨 분포 비교.

    Parameters
    ----------
    baseline_adaptive : run_baseline 결과 dict (fold_details, cls_metrics 포함)
    fwd_labels : fwd_return 라벨 배열 ({0,1,2} 또는 NaN)

    Returns
    -------
    DataFrame with rows = 기준선, columns = [SOURCE, UP %, NEUTRAL %, DOWN %, N]
    """
    rows = []

    # fwd_return (상수 밴드 ±1%)
    if fwd_labels is not None:
        arr = np.asarray(fwd_labels, dtype=float)
        valid = np.isfinite(arr)
        if valid.sum() > 0:
            v = arr[valid]
            rows.append(
                {
                    "SOURCE": "fwd_return ±1%",
                    "UP %": float((v == 2).mean()),
                    "NEUTRAL %": float((v == 1).mean()),
                    "DOWN %": float((v == 0).mean()),
                    "N": int(valid.sum()),
                }
            )

    # adaptive (per-fold baseline 판정)
    if baseline_adaptive is not None:
        _cls = baseline_adaptive.get("cls_metrics", {})
        rows.append(
            {
                "SOURCE": "Adaptive 6-param",
                "UP %": float(_cls.get("ratio_up", 0) or 0),
                "NEUTRAL %": float(_cls.get("ratio_neutral", 0) or 0),
                "DOWN %": float(_cls.get("ratio_down", 0) or 0),
                "N": int(baseline_adaptive.get("total_folds", 0)) * 63,
            }
        )

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════
# 라벨 소스별 best 성능 pivot
# ═══════════════════════════════════════════════════════════


def build_metric_pivot(slots: dict) -> pd.DataFrame:
    """
    슬롯별 ML best 성능 표.
    slots: {"fwd_return": results_dict, "adaptive": results_dict}
    """
    rows = []
    name_map = {
        "fwd_return": "fwd_return ±1%",
        "adaptive": "Adaptive 6-param",
    }
    for src, results in slots.items():
        if not results:
            continue
        k = build_summary_kpis(results)
        rows.append(
            {
                "LABEL SOURCE": name_map.get(src, src),
                "BEST MODEL": k["best_model"],
                "ACC": k["best_accuracy"],
                "MACRO F1": k["best_macro_f1"],
                "DOWN REC": k["best_down_recall"],
                "HARMONIC": k["best_harmonic"],
                "ΔSHARPE": k["best_delta_sharpe"],
            }
        )
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════
# REPRODUCTION · ML이 Baseline 판정을 얼마나 재현?
# ═══════════════════════════════════════════════════════════


def compute_reproduction(
    model_result: dict,
    baseline_labels: list,
    baseline_dates: list,
) -> dict:
    """
    ML OOS 예측 vs Baseline 판정 일치도.

    ML 예측: {-1, 0, 1}
    Baseline 라벨: {0, 1, 2}

    bas_dd 로 시점 매칭 → 같은 시점끼리 비교.

    Returns
    -------
    dict: {
        agreement, kappa, n_matched,
        per_class: {"DOWN": {...}, "NEUTRAL": {...}, "UP": {...}},
        confusion: 3x3 list,  # rows=Baseline, cols=ML
        classes: ["DOWN", "NEUTRAL", "UP"],
    } or {"error": "..."}
    """
    from sklearn.metrics import cohen_kappa_score, confusion_matrix

    oos = pd.DataFrame(model_result.get("oos_predictions", []))
    if oos.empty:
        return {"error": "oos_predictions 없음"}

    if "bas_dd" not in oos.columns or "predicted" not in oos.columns:
        return {"error": "oos_predictions 에 bas_dd/predicted 없음"}

    if not baseline_labels or not baseline_dates:
        return {"error": "baseline_labels / dates 없음"}

    # baseline Series (dates index)
    bl_ser = pd.Series(
        np.asarray(baseline_labels, dtype="float64"),
        index=pd.to_datetime(pd.Series(baseline_dates), errors="coerce"),
    )
    bl_ser = bl_ser[~bl_ser.index.isna()]

    # oos bas_dd → datetime
    oos = oos.copy()
    oos["bas_dd"] = pd.to_datetime(oos["bas_dd"], errors="coerce")
    oos = oos.dropna(subset=["bas_dd"])

    # 매칭
    matched_ml = []
    matched_bl = []
    for _, row in oos.iterrows():
        d = row["bas_dd"]
        if d not in bl_ser.index:
            continue
        bl_label = bl_ser.loc[d]
        if pd.isna(bl_label):
            continue
        ml_pred = int(row["predicted"])  # -1, 0, 1
        bl_int = int(bl_label)  # 0, 1, 2
        matched_ml.append(ml_pred + 1)  # 변환: -1→0, 0→1, 1→2
        matched_bl.append(bl_int)

    n = len(matched_ml)
    if n < 10:
        return {"error": f"매칭된 표본 부족: {n}개"}

    ml_arr = np.array(matched_ml, dtype=int)
    bl_arr = np.array(matched_bl, dtype=int)

    # 전체 agreement
    agreement = float((ml_arr == bl_arr).mean())

    # Cohen's Kappa
    try:
        kappa = float(cohen_kappa_score(bl_arr, ml_arr))
    except Exception:
        kappa = float("nan")

    # Per-class agreement (Baseline 기준)
    names = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}
    per_class = {}
    for cls in (0, 1, 2):
        mask = bl_arr == cls
        cnt = int(mask.sum())
        if cnt > 0:
            per_class[names[cls]] = {
                "n": cnt,
                "agreement": float((ml_arr[mask] == cls).mean()),
            }
        else:
            per_class[names[cls]] = {"n": 0, "agreement": None}

    # Confusion (rows=Baseline, cols=ML)
    cm = confusion_matrix(bl_arr, ml_arr, labels=[0, 1, 2])

    return {
        "agreement": agreement,
        "kappa": kappa,
        "n_matched": n,
        "per_class": per_class,
        "confusion": cm.tolist(),
        "classes": ["DOWN", "NEUTRAL", "UP"],
    }
