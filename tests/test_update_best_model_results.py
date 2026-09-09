from scripts.update_best_model_results import select_best_results


def test_모델별_네_실험_중_조화평균이_가장_높은_결과를_고른다():
    records = []
    variants = [(), ("daily_return",), ("five_day_return",), ("daily_return", "five_day_return")]
    for model in ("LogisticRegression", "RandomForest", "XGBoost", "LightGBM"):
        for index, returns in enumerate(variants):
            records.append(
                {
                    "experiment": {
                        "combination": "A",
                        "model": model,
                        "return_features": list(returns),
                    },
                    "summary": {"core_harmonic_mean": float(index)},
                }
            )

    selected = select_best_results(records, "A")

    assert set(selected) == {
        "LogisticRegression",
        "RandomForest",
        "XGBoost",
        "LightGBM",
    }
    assert all(
        item["experiment"]["return_features"] == ["daily_return", "five_day_return"]
        for item in selected.values()
    )
