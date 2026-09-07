import numpy as np
import pandas as pd

from models.stock_ranking import add_probability_ranks, select_for_index_direction


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": ["모델"] * 3,
            "bas_dd": ["20240102"] * 3,
            "code": ["000001", "000002", "000003"],
            "industry_index_name": ["건설", "건설", "금속"],
            "p_down": [0.1, 0.7, 0.2],
            "p_neutral": [0.2, 0.2, 0.6],
            "p_up": [0.7, 0.1, 0.2],
        }
    )


def test_하락중립상승확률순위를_각각남긴다():
    result = add_probability_ranks(_predictions())

    assert result.set_index("code").loc["000002", "down_rank"] == 1
    assert result.set_index("code").loc["000003", "neutral_rank"] == 1
    assert result.set_index("code").loc["000001", "up_rank"] == 1


def test_지수예측클래스확률로_상위종목을고른다():
    index_predictions = pd.DataFrame({"bas_dd": ["20240102"], "predicted": [-1]})

    result = select_for_index_direction(_predictions(), index_predictions, top_n=2)

    assert result["code"].tolist() == ["000002", "000003"]
    assert result["index_predicted"].tolist() == [-1, -1]
    assert np.allclose(result["selected_probability"], [0.7, 0.2])
