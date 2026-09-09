import json

import pytest

from models.final_models import (
    INDEX_POLICY_ACCURACY_MACRO_HARMONIC,
    INDEX_POLICY_ACCURACY_THEN_MACRO,
    INDEX_POLICY_DELTA_REPORTED_MAJORITY,
    compare_index_model_selection_policies,
    load_winning_models,
    select_best_index_model,
    select_best_stock_model,
)


def test_kospi200_보고서에서_조화평균_1위를_가져온다():
    report = {
        "experiments": [
            {
                "experiment": {
                    "combination": "A",
                    "model": "LightGBM",
                    "return_features": [],
                },
                "summary": {
                    "core_harmonic_mean": 0.35,
                    "accuracy": 0.42,
                    "macro_f1": 0.36,
                    "feature_columns": ["rsi_14"],
                },
            },
            {
                "experiment": {
                    "combination": "E",
                    "model": "RandomForest",
                    "return_features": ["five_day_return"],
                },
                "summary": {
                    "core_harmonic_mean": 0.39,
                    "accuracy": 0.38,
                    "macro_f1": 0.37,
                    "feature_columns": ["atr_ratio", "five_day_return"],
                },
            },
        ]
    }

    winner = select_best_index_model(report)

    assert winner.combination == "E"
    assert winner.model == "RandomForest"
    assert winner.return_features == ("five_day_return",)


def test_개별종목_보고서에서_adr_선정_1위를_가져온다():
    report = {
        "final_selection": {
            "primary": "accuracy_minus_training_majority_baseline",
            "selected": {
                "combination": "K",
                "model": "LogisticRegression",
                "feature_columns": ["atr_ratio", "hv_regime"],
                "accuracy_minus_training_majority_baseline": 0.024,
            },
        }
    }

    winner = select_best_stock_model(report)

    assert winner.combination == "K"
    assert winner.model == "LogisticRegression"
    assert winner.feature_columns == ("atr_ratio", "hv_regime")
    assert winner.selection_value == 0.024


def test_두_트랙_1위는_보고서_파일에서_함께_읽는다(tmp_path):
    index_path = tmp_path / "index.json"
    stock_path = tmp_path / "stock.json"
    index_path.write_text(
        json.dumps(
            {
                "experiments": [
                    {
                        "experiment": {"combination": "E", "model": "RandomForest"},
                        "summary": {
                            "core_harmonic_mean": 0.39,
                            "accuracy": 0.38,
                            "macro_f1": 0.37,
                            "feature_columns": ["hv_regime"],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    stock_path.write_text(
        json.dumps(
            {
                "final_selection": {
                    "selected": {
                        "combination": "K",
                        "model": "LogisticRegression",
                        "feature_columns": ["hv_regime"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    index, stock = load_winning_models(index_path, stock_path)

    assert (index.combination, stock.combination) == ("E", "K")


def test_개별종목_최종선정이_없으면_임의로_다시_고르지_않는다():
    with pytest.raises(ValueError, match="final_selection"):
        select_best_stock_model({"combination_winners": []})


def test_두_평가_보고서의_지수_데이터가_다르면_최종실행을_막는다(tmp_path):
    index_path = tmp_path / "index.json"
    stock_path = tmp_path / "stock.json"
    index_path.write_text(
        json.dumps(
            {
                "source": {"index_sha256": "index-a"},
                "experiments": [
                    {
                        "experiment": {"combination": "E", "model": "RandomForest"},
                        "summary": {
                            "core_harmonic_mean": 0.39,
                            "accuracy": 0.38,
                            "macro_f1": 0.37,
                            "feature_columns": ["hv_regime"],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    stock_path.write_text(
        json.dumps(
            {
                "source": {"index_sha256": "index-b"},
                "final_selection": {
                    "selected": {
                        "combination": "K",
                        "model": "LogisticRegression",
                        "feature_columns": ["hv_regime"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="SHA-256"):
        load_winning_models(index_path, stock_path)


def _정책비교용_후보(
    combination,
    accuracy,
    macro_f1,
    down_recall,
    majority_accuracy,
):
    return {
        "experiment": {
            "combination": combination,
            "model": "LogisticRegression",
            "return_features": [],
        },
        "summary": {
            "feature_columns": ["feature"],
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "down_recall": down_recall,
            "core_harmonic_mean": 3.0
            / (1.0 / accuracy + 1.0 / macro_f1 + 1.0 / down_recall),
            "majority_accuracy": majority_accuracy,
            "balanced_accuracy": macro_f1,
            "mcc": 0.1,
            "up_recall": 0.4,
            "pr_auc_up": 0.4,
            "predicted_down": 10,
            "predicted_neutral": 20,
            "predicted_up": 30,
        },
    }


def test_kospi200_선정정책별_1위를_비교해도_운영기본값은_바뀌지_않는다():
    report = {
        "experiments": [
            _정책비교용_후보("A", 0.50, 0.30, 0.05, 0.45),
            _정책비교용_후보("B", 0.43, 0.42, 0.35, 0.42),
        ]
    }

    comparison = compare_index_model_selection_policies(report)
    winners = {
        row["policy"]: row["winner"]["combination"]
        for row in comparison["policies"]
    }

    assert select_best_index_model(report).combination == "B"
    assert winners[INDEX_POLICY_ACCURACY_THEN_MACRO] == "A"
    assert winners[INDEX_POLICY_ACCURACY_MACRO_HARMONIC] == "B"
    assert winners[INDEX_POLICY_DELTA_REPORTED_MAJORITY] == "A"
    assert comparison["holdout_used"] is False
