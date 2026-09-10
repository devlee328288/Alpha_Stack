import numpy as np
import pandas as pd
import pytest

from features.model_dataset import ModelDataset
from models.index_long_only import (
    evaluate_long_only_thresholds,
    predict_with_up_threshold,
    restrict_dataset_to_dates,
    select_inner_threshold,
)


def _dataset(rows: int = 20) -> ModelDataset:
    frame = pd.DataFrame(
        {
            "bas_dd": [f"202401{day:02d}" for day in range(1, rows + 1)],
            "raw_position": np.arange(rows),
            "label_numeric": np.resize(np.array([-1, 0, 1]), rows),
            "feature": np.arange(rows, dtype=float),
        }
    )
    raw = pd.DataFrame({"open": np.linspace(100.0, 125.0, rows + 10)})
    return ModelDataset(
        frame=frame,
        raw_prices=raw,
        feature_columns=("feature",),
        combination="A",
    )


def test_상승_임계값을_넘지_못하면_하락과_보합_확률만_비교한다():
    probabilities = np.array(
        [
            [0.20, 0.30, 0.50],
            [0.45, 0.35, 0.20],
            [0.30, 0.40, 0.30],
        ]
    )

    predicted = predict_with_up_threshold(probabilities, threshold=0.40)

    np.testing.assert_array_equal(predicted, np.array([1, -1, 0]))


def test_상승_임계값은_확률_배열_검증을_건너뛰지_않는다():
    with pytest.raises(ValueError, match="합이 1"):
        predict_with_up_threshold(np.array([[0.2, 0.2, 0.2]]), threshold=0.4)


def test_공통_거래일로_줄여도_원시가격_위치는_그대로_보존한다():
    dataset = _dataset(10)
    dates = ("20240103", "20240104", "20240105")

    restricted = restrict_dataset_to_dates(dataset, dates)

    assert restricted.frame["bas_dd"].tolist() == list(dates)
    assert restricted.signal_positions.tolist() == [2, 3, 4]
    assert restricted.raw_prices is dataset.raw_prices


def test_내부_임계값은_최소_매수기회가_없는_후보를_선택하지_않는다():
    dataset = _dataset(20)
    indices = np.arange(5, 15)
    probabilities = np.tile(np.array([0.20, 0.45, 0.35]), (len(indices), 1))

    selected, candidates = select_inner_threshold(
        dataset,
        indices,
        probabilities,
        thresholds=(0.30, 0.40),
        min_buy_signals=2,
    )

    assert len(candidates) == 2
    assert selected["threshold"] == 0.30
    assert selected["buy_signals"] == len(indices)


def test_모든_내부_임계값이_매수기회를_잃으면_중단한다():
    dataset = _dataset(20)
    indices = np.arange(5, 15)
    probabilities = np.tile(np.array([0.25, 0.65, 0.10]), (len(indices), 1))

    with pytest.raises(RuntimeError, match="최소 내부 매수 신호"):
        select_inner_threshold(
            dataset,
            indices,
            probabilities,
            thresholds=(0.50, 0.60),
            min_buy_signals=2,
        )


def test_외부_검증은_학습구간에서_고른_임계값과_운영_기준선을_기록한다():
    rows = 830
    labels = np.resize(np.array([-1, 0, 1]), rows)
    frame = pd.DataFrame(
        {
            "bas_dd": [f"{day:08d}" for day in range(1, rows + 1)],
            "raw_position": np.arange(rows),
            "label_numeric": labels,
            "feature": labels + np.sin(np.arange(rows)) * 0.1,
        }
    )
    dataset = ModelDataset(
        frame=frame,
        raw_prices=pd.DataFrame(
            {"open": 100.0 + np.arange(rows + 10) * 0.1 + np.sin(np.arange(rows + 10))}
        ),
        feature_columns=("feature",),
        combination="A",
    )
    splits = [(np.arange(750), np.arange(755, 815))]

    result = evaluate_long_only_thresholds(
        dataset,
        "LogisticRegression",
        splits,
        thresholds=(0.20, 0.80),
    )

    assert len(result.inner_results) == 2
    assert len(result.outer_results) == 1
    assert len(result.oos_predictions) == 60
    assert result.outer_results.loc[0, "selected_up_threshold"] == 0.20
    assert "training_majority_baseline_accuracy" in result.outer_results
