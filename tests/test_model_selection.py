import pytest

from models.selection import rank_final_model_candidates


def test_최종모델은_기준선대비정확도를_가장먼저본다():
    candidates = [
        {
            "combination": "F",
            "model": "LightGBM",
            "accuracy_minus_training_majority_baseline": -0.01,
            "macro_f1": 0.40,
            "baseline_win_folds": 8,
        },
        {
            "combination": "A",
            "model": "LightGBM",
            "accuracy_minus_training_majority_baseline": 0.01,
            "macro_f1": 0.37,
            "baseline_win_folds": 6,
        },
    ]

    ranked = rank_final_model_candidates(candidates)

    assert ranked[0]["combination"] == "A"
    assert [row["final_selection_rank"] for row in ranked] == [1, 2]


def test_최종모델선정지표가빠지면실패한다():
    with pytest.raises(ValueError, match="선정 지표"):
        rank_final_model_candidates([{"combination": "A", "model": "모델"}])
