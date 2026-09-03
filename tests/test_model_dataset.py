import numpy as np
import pandas as pd

from features.model_dataset import build_model_dataset


def _index_prices(rows: int = 400) -> pd.DataFrame:
    dates = pd.bdate_range("2022-01-03", periods=rows).strftime("%Y%m%d")
    step = np.arange(rows, dtype=float)
    close = 100.0 + step * 0.03 + 5.0 * np.sin(step / 5.0)
    return pd.DataFrame(
        {
            "bas_dd": dates,
            "index_name": "코스피 200",
            "index_class": "KOSPI",
            "open": close * (1.0 + 0.001 * np.cos(step / 3.0)),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000_000.0 + 10_000.0 * np.sin(step / 7.0),
        }
    )


def test_F조합은_파생피처_결측을_먼저_제거한_학습표를_만든다():
    dataset = build_model_dataset(
        _index_prices(),
        "F",
        return_features=("daily_return", "five_day_return"),
    )

    assert dataset.feature_columns[-2:] == ("daily_return", "five_day_return")
    assert np.isfinite(dataset.x.to_numpy()).all()
    assert len(dataset.frame) > 0
    assert set(dataset.y) == {-1, 0, 1}


def test_마지막_신호의_청산_시가는_원시_개발구간_안에_남는다():
    dataset = build_model_dataset(_index_prices(), "A")

    last_signal = int(dataset.signal_positions[-1])
    assert last_signal + 6 < len(dataset.raw_prices)
    assert dataset.frame["bas_dd"].max() < "20240901"
