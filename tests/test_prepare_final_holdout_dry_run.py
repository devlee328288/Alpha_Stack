import pandas as pd

from scripts.prepare_final_holdout_dry_run import split_dry_run_frames


def test_마지막60일을평가로떼고_직전5일은학습에서버린다():
    dates = [f"2023{month:02d}{day:02d}" for month in range(1, 5) for day in range(1, 21)]
    index = pd.DataFrame({"bas_dd": dates})
    stock = pd.DataFrame(
        [{"bas_dd": date, "code": code} for date in dates for code in ("000001", "000002")]
    )

    result = split_dry_run_frames(index, stock, evaluation_dates=60, gap_dates=5)

    assert result["index_holdout"]["bas_dd"].nunique() == 60
    assert result["stock_holdout"]["bas_dd"].nunique() == 60
    assert result["index_dev"]["bas_dd"].max() == dates[14]
    assert result["stock_dev"]["bas_dd"].max() == dates[14]
    assert result["index_holdout"]["bas_dd"].min() == dates[20]
