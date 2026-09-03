"""HF 전 종목 시장 내부 피처를 네 모델의 12폴드 OOS에서 한 번 평가한다."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from features.market_breadth import (  # noqa: E402
    COMBINED_MARKET_FEATURE_COLUMNS,
    MARKET_FEATURE_COLUMNS,
    build_combined_market_dataset,
    build_market_feature_dataset,
)
from models.experiment import (  # noqa: E402
    LABEL_HORIZON,
    MIN_TRAIN_SIZE,
    N_FOLDS,
    VALID_SIZE,
    classification_metrics,
    evaluate_nested_class_weights,
)
from supply.hf_model_data import REPO_ID, load_hf_market_prices  # noqa: E402

OUTPUTS = {
    "market_only": ROOT / "reports" / "market_feature_experiment.json",
    "combined": ROOT / "reports" / "combined_market_feature_experiment.json",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=tuple(OUTPUTS), default="combined")
    args = parser.parse_args(argv)
    snapshot = load_hf_market_prices()
    if args.mode == "market_only":
        dataset = build_market_feature_dataset(snapshot.index_frame, snapshot.daily_frame)
        feature_columns = MARKET_FEATURE_COLUMNS
    else:
        dataset = build_combined_market_dataset(snapshot.index_frame, snapshot.daily_frame)
        feature_columns = COMBINED_MARKET_FEATURE_COLUMNS
    result = evaluate_nested_class_weights(dataset)

    rows = []
    for model_name, predictions in result.oos_predictions.groupby("model", sort=False):
        actual = predictions["actual"].to_numpy(dtype=int)
        predicted = predictions["predicted"].to_numpy(dtype=int)
        metrics = classification_metrics(actual, predicted)
        predicted_counts = {
            str(label): int(np.sum(predicted == label)) for label in (-1, 0, 1)
        }
        model_folds = result.outer_results.loc[result.outer_results["model"] == model_name]
        rows.append(
            {
                "model": model_name,
                **metrics,
                "accuracy_fold_std": float(model_folds["accuracy"].std(ddof=1)),
                "core_harmonic_fold_std": float(
                    model_folds["core_harmonic_mean"].std(ddof=1)
                ),
                "delta_sharpe_net_median": float(model_folds["delta_sharpe_net"].median()),
                "predicted_counts": predicted_counts,
            }
        )

    first_model = result.oos_predictions["model"].iloc[0]
    baseline_actual = result.oos_predictions.loc[
        result.oos_predictions["model"] == first_model, "actual"
    ].to_numpy(dtype=int)
    labels, counts = np.unique(baseline_actual, return_counts=True)
    majority_class = int(labels[np.argmax(counts)])
    majority_predicted = np.full_like(baseline_actual, majority_class)
    baseline = {
        "class": majority_class,
        "count": int(counts.max()),
        "samples": int(len(baseline_actual)),
        **classification_metrics(baseline_actual, majority_predicted),
    }

    selected_counts = (
        result.outer_results.assign(
            selected_class_weight=result.outer_results["selected_class_weight"].fillna("None")
        )
        .groupby(["model", "selected_class_weight"])
        .size()
        .rename("folds")
        .reset_index()
    )
    summary = sorted(rows, key=lambda row: row["core_harmonic_mean"], reverse=True)
    print(f"HF 커밋: {snapshot.repo_sha}")
    print(f"학습 가능 기간: {dataset.frame['bas_dd'].min()}~{dataset.frame['bas_dd'].max()}")
    print(f"학습 가능 행: {len(dataset.frame):,}")
    print(f"다수 클래스 기준선: {majority_class} / Accuracy {baseline['accuracy']:.6f}")
    print("\n네 모델 OOS 통합 결과")
    for row in summary:
        print(
            f"{row['model']}: Accuracy={row['accuracy']:.6f}, "
            f"MacroF1={row['macro_f1']:.6f}, DownRecall={row['down_recall']:.6f}, "
            f"조화평균={row['core_harmonic_mean']:.6f}, "
            f"예측수={row['predicted_counts']}"
        )

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "repo": REPO_ID,
            "repo_sha": snapshot.repo_sha,
            "index_sha256": snapshot.index_sha256,
            "daily_sha256": snapshot.daily_sha256,
            "dev_end": snapshot.dev_end,
        },
        "dataset": {
            "combination": dataset.combination,
            "feature_columns": list(feature_columns),
            "first_date": dataset.frame["bas_dd"].min(),
            "last_date": dataset.frame["bas_dd"].max(),
            "rows": len(dataset.frame),
        },
        "rules": {
            "outer_window": "expanding",
            "outer_folds": N_FOLDS,
            "outer_minimum_train": MIN_TRAIN_SIZE,
            "outer_valid_size": VALID_SIZE,
            "outer_gap": LABEL_HORIZON,
            "inner_valid_size": VALID_SIZE,
            "inner_gap": LABEL_HORIZON,
            "class_weight_candidates": [None, "balanced"],
            "class_weight_selection": "Accuracy·Macro F1·하락 Recall 조화평균 최대",
            "oos_feature_selection": "없음: OOS 확인 전 다섯 피처와 룩백을 고정",
        },
        "majority_baseline": baseline,
        "summary": summary,
        "selected_class_weight_counts": selected_counts.to_dict(orient="records"),
        "outer_results": result.outer_results.to_dict(orient="records"),
    }
    output = OUTPUTS[args.mode]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과 저장: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
