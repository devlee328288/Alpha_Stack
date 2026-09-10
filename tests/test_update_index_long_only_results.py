from pathlib import Path

from scripts.build_model_notebooks import COMBINATION_DIRS, MODELS
from scripts.update_index_long_only_results import (
    RANK_PREFIXES,
    select_best_results,
    sync_long_only_best_results,
)


def _candidate(rank: int, combination: str, model: str) -> dict[str, object]:
    return {
        "rank": rank,
        "experiment": {
            "combination": combination,
            "model": model,
            "return_features": [],
        },
        "feature_columns": ["rsi_14"],
        "summary": {
            "passes_operational_gate": True,
            "accuracy": 0.45,
            "training_majority_baseline_accuracy": 0.40,
            "accuracy_minus_training_majority_baseline": 0.05,
            "baseline_win_folds": 7,
            "folds": 12,
            "macro_f1": 0.40,
            "pr_auc_up": 1.0 / rank,
            "up_precision": 0.42,
            "up_recall": 0.50,
            "buy_signals": 300,
            "delta_sharpe_net_median": 0.20,
            "selected_up_threshold_median": 0.35,
            "selected_up_threshold_min": 0.20,
            "selected_up_threshold_max": 0.50,
        },
    }


def _report() -> dict[str, object]:
    candidates = []
    rank = 1
    for combination in ("C", "B", "E", "A", "D", "F", "G"):
        for model in MODELS:
            candidates.append(_candidate(rank, combination, model))
            rank += 1
    return {
        "status": "complete",
        "holdout_used": False,
        "expected_candidate_count": len(candidates),
        "source": {"generated_at": "now", "index_sha256": "sha"},
        "candidates": candidates,
    }


def test_모델별_long_only_보고서_순위가_가장_높은_변형을_고른다():
    candidates = [
        _candidate(2, "C", "LogisticRegression"),
        {
            **_candidate(1, "C", "LogisticRegression"),
            "experiment": {
                "combination": "C",
                "model": "LogisticRegression",
                "return_features": ["daily_return"],
            },
        },
        _candidate(3, "C", "RandomForest"),
        _candidate(4, "C", "XGBoost"),
        _candidate(5, "C", "LightGBM"),
    ]

    selected = select_best_results(candidates, "C")

    assert selected["LogisticRegression"]["rank"] == 1


def test_조합별_best_result_폴더와_모델표시를_새_순위로_동기화한다(tmp_path: Path):
    for combination in COMBINATION_DIRS:
        directory = tmp_path / f"조합{combination}"
        directory.mkdir()
        (directory / "보존.txt").write_text("보존", encoding="utf-8")

    written = sync_long_only_best_results(_report(), best_root=tmp_path)

    assert written["C"].name == f"{RANK_PREFIXES[0]}조합C"
    assert written["B"].name == f"{RANK_PREFIXES[1]}조합B"
    assert written["E"].name == f"{RANK_PREFIXES[2]}조합E"
    assert (written["C"] / "⭐01.LogisticRegression.ipynb").is_file()
    assert (written["C"] / "보존.txt").read_text(encoding="utf-8") == "보존"
