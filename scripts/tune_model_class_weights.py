"""대표 조합에서 중첩 시계열 검증으로 모델별 클래스 가중치를 다시 고른다."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from features.model_dataset import build_model_dataset  # noqa: E402
from models.experiment import (  # noqa: E402
    LABEL_HORIZON,
    MIN_TRAIN_SIZE,
    N_FOLDS,
    VALID_SIZE,
    evaluate_nested_class_weights,
)
from supply.hf_model_data import REPO_ID, load_hf_index_prices  # noqa: E402

OUTPUT = ROOT / "reports" / "class_weight_tuning.json"


def main() -> int:
    snapshot = load_hf_index_prices()
    dataset = build_model_dataset(
        snapshot.frame,
        "F",
        return_features=("daily_return", "five_day_return"),
    )
    result = evaluate_nested_class_weights(dataset)
    summary = (
        result.outer_results.groupby("model", as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            macro_f1=("macro_f1", "mean"),
            down_recall=("down_recall", "mean"),
            core_harmonic_mean=("core_harmonic_mean", "mean"),
            delta_sharpe_net=("delta_sharpe_net", "median"),
            all_cash_folds=("all_cash", "sum"),
        )
    )
    counts = (
        result.outer_results.assign(
            selected_class_weight=result.outer_results["selected_class_weight"].fillna("None")
        )
        .groupby(["model", "selected_class_weight"])
        .size()
        .rename("folds")
        .reset_index()
    )
    print("대표 조합 F + Daily_Return + 5Day_Return")
    print("\n외부 OOS 평균")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\n폴드별 선택 횟수")
    print(counts.to_string(index=False))

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "repo": REPO_ID,
            "repo_sha": snapshot.repo_sha,
            "index_sha256": snapshot.file_sha256,
            "dev_end": snapshot.dev_end,
        },
        "representative": {
            "combination": "F",
            "return_features": ["daily_return", "five_day_return"],
        },
        "rules": {
            "outer_window": "expanding",
            "outer_folds": N_FOLDS,
            "outer_minimum_train": MIN_TRAIN_SIZE,
            "outer_valid_size": VALID_SIZE,
            "inner_valid_size": VALID_SIZE,
            "inner_gap": LABEL_HORIZON,
            "candidates": [None, "balanced"],
            "selection": "Accuracy·Macro F1·하락 Recall 조화평균 최대, 동률이면 None",
            "all_cash_sharpe": 0.0,
        },
        "summary": summary.to_dict(orient="records"),
        "selected_counts": counts.to_dict(orient="records"),
        "inner_results": result.inner_results.to_dict(orient="records"),
        "outer_results": result.outer_results.to_dict(orient="records"),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과 저장: {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
