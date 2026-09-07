import numpy as np
import pandas as pd

from features.stock_model_dataset import STOCK_FEATURE_COLUMNS, StockModelDataset
from models.stock_experiment import (
    evaluate_stock_models,
    inner_group_class_weight_split,
)


class _ProbabilityModel:
    def __init__(self, *, class_weight=None):
        self.class_weight = class_weight

    def fit(self, x, y):
        self.classes_ = np.array([-1, 0, 1])
        return self

    def predict(self, x):
        values = x.iloc[:, 0].to_numpy()
        return np.where(values < -0.2, -1, np.where(values > 0.2, 1, 0))

    def predict_proba(self, x):
        predicted = self.predict(x)
        probabilities = np.full((len(x), 3), 0.1)
        probabilities[np.arange(len(x)), predicted + 1] = 0.8
        return probabilities


def _dataset(date_count: int = 25) -> StockModelDataset:
    rows = []
    for date_index in range(date_count):
        date = f"202301{date_index + 1:02d}"
        for stock_index, value in enumerate((-1.0, 0.0, 1.0)):
            row = {
                "bas_dd": date,
                "code": f"{stock_index + 1:06d}",
                "label_numeric": int(value),
            }
            row.update({feature: value for feature in STOCK_FEATURE_COLUMNS})
            rows.append(row)
    return StockModelDataset(pd.DataFrame(rows))


def test_내부가중치검증도_같은날짜종목을쪼개지않는다():
    dataset = _dataset()
    outer_train = np.arange(18 * 3)

    inner_train, inner_valid = inner_group_class_weight_split(
        dataset.groups,
        outer_train,
        valid_dates=3,
        gap_dates=5,
    )

    assert len(set(dataset.groups[inner_train]) & set(dataset.groups[inner_valid])) == 0
    assert len(set(dataset.groups[inner_valid])) == 3
    assert int(inner_valid[0]) - int(inner_train[-1]) == 5 * 3 + 1


def test_종목모델평가는_날짜그룹폴드와확률세칸을남긴다():
    dataset = _dataset()

    result = evaluate_stock_models(
        dataset,
        model_builders={"시험모델": _ProbabilityModel},
        n_folds=2,
        min_train_dates=12,
        valid_dates=3,
        gap_dates=5,
    )

    assert len(result.inner_results) == 4
    assert len(result.outer_results) == 2
    assert result.outer_results["valid_dates"].tolist() == [3, 3]
    assert len(result.oos_predictions) == 2 * 3 * 3
    probability_sum = result.oos_predictions[["p_down", "p_neutral", "p_up"]].sum(axis=1)
    assert np.allclose(probability_sum, 1.0)
    assert set(result.oos_predictions["predicted"]) == {-1, 0, 1}
