import numpy as np
import pandas as pd

from features.model_dataset import ModelDataset
from models.experiment import NestedWeightResult
from models.notebook_experiment import summarize_notebook_experiment


def test_노트북_요약은_전체_oos의_하락_recall과_조화평균을_계산한다():
    dataset = ModelDataset(
        frame=pd.DataFrame({"bas_dd": ["20240101", "20240102"], "x": [1.0, 2.0]}),
        raw_prices=pd.DataFrame(),
        feature_columns=("x",),
        combination="A",
    )
    outer = pd.DataFrame(
        {
            "model": ["모델"],
            "selected_class_weight": [None],
            "delta_sharpe_net": [0.25],
            "all_cash": [False],
        }
    )
    predictions = pd.DataFrame(
        {
            "model": ["모델"] * 6,
            "fold": [1] * 6,
            "bas_dd": [str(index) for index in range(6)],
            "actual": np.array([-1, -1, 0, 0, 1, 1]),
            "predicted": np.array([-1, 0, 0, 0, 1, 0]),
        }
    )
    nested = NestedWeightResult(
        inner_results=pd.DataFrame(),
        outer_results=outer,
        oos_predictions=predictions,
    )

    result = summarize_notebook_experiment(dataset, nested, "모델")

    assert result.summary["down_recall"] == 0.5
    assert result.summary["core_harmonic_mean"] > 0.0
    assert result.summary["delta_sharpe_net_median"] == 0.25
    assert result.weight_counts.loc[0, "클래스 가중치"] == "None"
