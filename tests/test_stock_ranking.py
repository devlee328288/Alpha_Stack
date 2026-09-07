import numpy as np
import pandas as pd

from models.stock_ranking import (
    add_probability_ranks,
    aligned_index_splits,
    aligned_panel_splits,
    build_common_validation_schedule,
    select_for_index_direction,
    summarize_direction_ranking,
)


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


def test_지수방향과_실제종목라벨의_top_n_적중률을계산한다():
    stocks = _predictions().assign(label_numeric=[1, 0, 1])
    index_predictions = pd.DataFrame({"bas_dd": ["20240102"], "predicted": [1]})

    summary = summarize_direction_ranking(stocks, index_predictions, cutoffs=(1, 2))

    assert summary["top_n"].tolist() == [1, 2]
    assert summary["direction_hit_rate"].tolist() == [1.0, 1.0]
    assert np.allclose(summary["mean_selected_probability"], [0.7, 0.45])


def test_종목검증일정과_같은날짜로_지수분할을만든다():
    dates = pd.date_range("2020-01-01", periods=20, freq="D").strftime("%Y%m%d")
    schedule = pd.DataFrame(
        {
            "fold": [1, 1, 2, 2],
            "bas_dd": [dates[10], dates[11], dates[16], dates[17]],
        }
    )

    splits = aligned_index_splits(
        dates,
        schedule,
        gap_dates=2,
        minimum_train_dates=5,
    )

    assert splits[0][0].tolist() == list(range(8))
    assert splits[0][1].tolist() == [10, 11]
    assert splits[1][0].tolist() == list(range(14))
    assert splits[1][1].tolist() == [16, 17]


def test_두모델의_공통거래일로_동일한_검증일정을만든다():
    first = pd.date_range("2020-01-01", periods=20, freq="D").strftime("%Y%m%d")
    second = first[2:]

    schedule = build_common_validation_schedule(
        first,
        second,
        n_folds=2,
        minimum_train_dates=5,
        valid_dates=2,
        gap_dates=1,
    )

    assert schedule.groupby("fold")["bas_dd"].count().tolist() == [2, 2]
    assert set(schedule["bas_dd"]).issubset(set(first) & set(second))


def test_패널의_같은날짜종목을_한꺼번에분할한다():
    dates = pd.date_range("2020-01-01", periods=10, freq="D").strftime("%Y%m%d")
    panel_dates = np.repeat(dates, 2)
    schedule = pd.DataFrame({"fold": [1, 1], "bas_dd": [dates[7], dates[8]]})

    splits = aligned_panel_splits(
        panel_dates,
        schedule,
        gap_dates=1,
        minimum_train_dates=3,
    )

    train, valid = splits[0]
    assert train.tolist() == list(range(12))
    assert valid.tolist() == [14, 15, 16, 17]
