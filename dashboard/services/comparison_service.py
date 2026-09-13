# dashboard/services/comparison_service.py
"""4모델 결과 비교 + best 선택. 정렬 기준은 노트북 05.모델비교와 동일."""
from __future__ import annotations

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
        rows.append({
            "model": s.get("model", model_name),
            "accuracy": float(s["accuracy"]),
            "macro_f1": float(s["macro_f1"]),
            "down_recall": float(s["down_recall"]),
            "core_harmonic_mean": float(s["core_harmonic_mean"]),
            "balanced_accuracy": float(s["balanced_accuracy"]),
            "majority_accuracy": float(s["majority_accuracy"]),
            "delta_sharpe_net_median": float(s["delta_sharpe_net_median"]),
            "all_cash_folds": int(s["all_cash_folds"]),
        })
    df = pd.DataFrame(rows).sort_values(
        "core_harmonic_mean", ascending=False, kind="stable"
    ).reset_index(drop=True)
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
    주의: baseline Sharpe는 pooled, 모델은 fold-median ΔSharpe — 정의가 다름.
    """
    perf = baseline.get("perf_metrics", {})
    cls = baseline.get("cls_metrics", {})
    thr = baseline.get("threshold", 0.01)
    b_name = f"6-PARAM {thr*100:.0f}%"

    strategy_rows = [{
        "SOURCE": "BASELINE", "MODEL": b_name,
        "SHARPE": perf.get("sharpe"),
        "CAGR": perf.get("cagr"),
        "MDD": perf.get("mdd"),
        "WIN RATE": perf.get("win_rate"),
        "ΔSHARPE": None,
    }]
    cls_rows = [{
        "SOURCE": "BASELINE", "MODEL": b_name,
        "ACC": None,
        "MACRO F1": cls.get("f1_macro"),
        "BAL ACC": cls.get("balanced_acc"),
        "DOWN REC": None,
        "MAJORITY": None,
    }]

    for name, r in models.items():
        s = r["summary"]
        strategy_rows.append({
            "SOURCE": "MODEL", "MODEL": name,
            "SHARPE": None, "CAGR": None, "MDD": None, "WIN RATE": None,
            "ΔSHARPE": s.get("delta_sharpe_net_median"),
        })
        cls_rows.append({
            "SOURCE": "MODEL", "MODEL": name,
            "ACC": s.get("accuracy"),
            "MACRO F1": s.get("macro_f1"),
            "BAL ACC": s.get("balanced_accuracy"),
            "DOWN REC": s.get("down_recall"),
            "MAJORITY": s.get("majority_accuracy"),
        })

    return {
        "strategy": pd.DataFrame(strategy_rows),
        "classification": pd.DataFrame(cls_rows),
    }