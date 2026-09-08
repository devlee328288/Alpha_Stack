import numpy as np
import pandas as pd
import pytest

from evaluation.multiple_comparison import compare_accuracy_to_baseline, holm_adjust


def test_holm_보정은_원래_순서와_단조성을_보존한다():
    adjusted = holm_adjust([0.03, 0.001, 0.02])

    assert np.allclose(adjusted, [0.04, 0.003, 0.04])


def test_후보전체를_한_검정군으로_묶어_보정한다():
    rows = []
    for combination, deltas in {"A": [0.1] * 12, "B": [-0.1] * 12}.items():
        for fold, delta in enumerate(deltas, start=1):
            rows.append(
                {
                    "combination": combination,
                    "model": "모델",
                    "fold": fold,
                    "accuracy": 0.4 + delta,
                    "training_majority_baseline_accuracy": 0.4,
                }
            )

    report = compare_accuracy_to_baseline(pd.DataFrame(rows))
    by_combination = {row["combination"]: row for row in report["results"]}

    assert report["comparisons"] == 2
    assert by_combination["A"]["significant_after_holm"]
    assert not by_combination["B"]["significant_after_holm"]
    assert by_combination["A"]["positive_folds"] == 12


def test_조합모델폴드가_중복되면_조용히_검정하지_않는다():
    frame = pd.DataFrame(
        {
            "combination": ["A", "A"],
            "model": ["모델", "모델"],
            "fold": [1, 1],
            "accuracy": [0.5, 0.5],
            "training_majority_baseline_accuracy": [0.4, 0.4],
        }
    )

    with pytest.raises(ValueError, match="중복"):
        compare_accuracy_to_baseline(frame)
