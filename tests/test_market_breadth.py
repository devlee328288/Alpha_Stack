import numpy as np
import pandas as pd

from features.market_breadth import (
    COMBINED_MARKET_FEATURE_COLUMNS,
    MARKET_FEATURE_COLUMNS,
    build_combined_market_dataset,
    build_kospi_market_features,
    build_market_feature_dataset,
)
from features.model_dataset import build_model_dataset


def _prices(rows: int = 280) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2022-01-03", periods=rows).strftime("%Y%m%d")
    step = np.arange(rows, dtype=float)
    index_close = 300.0 + 0.03 * step + 8.0 * np.sin(step / 8.0)
    index_prices = pd.DataFrame(
        {
            "bas_dd": dates,
            "index_name": "코스피 200",
            "index_class": "KOSPI",
            "open": index_close * (1.0 + 0.002 * np.cos(step / 4.0)),
            "high": index_close * 1.01,
            "low": index_close * 0.99,
            "close": index_close,
            "volume": 1_000_000.0,
        }
    )

    daily_parts = []
    price_paths = (
        100.0 + 0.10 * step,
        200.0 + 0.20 * step,
        200.0 - 0.10 * step,
        300.0 - 0.05 * step,
    )
    for number, prices in enumerate(price_paths):
        daily_parts.append(
            pd.DataFrame(
                {
                    "bas_dd": dates,
                    "code": f"{number + 1:06d}",
                    "market": "KOSPI",
                    "volume": 1_000.0 + number * 100.0,
                    "adj_close": prices,
                }
            )
        )
    return index_prices, pd.concat(daily_parts, ignore_index=True)


def test_시장피처는_수정종가와_당일까지의_값만_사용한다():
    _, daily = _prices()
    original = build_kospi_market_features(daily)
    changed = daily.copy()
    last_date = changed["bas_dd"].max()
    changed.loc[changed["bas_dd"] == last_date, "adj_close"] *= 10.0
    recalculated = build_kospi_market_features(changed)

    cutoff = original["bas_dd"].iloc[-2]
    expected = original.loc[original["bas_dd"] <= cutoff].reset_index(drop=True)
    actual = recalculated.loc[recalculated["bas_dd"] <= cutoff].reset_index(drop=True)
    pd.testing.assert_frame_equal(expected, actual)


def test_시장피처_데이터셋은_MA200_워밍업과_미래라벨을_제거한다():
    index_prices, daily = _prices()

    dataset = build_market_feature_dataset(index_prices, daily)

    assert dataset.feature_columns == MARKET_FEATURE_COLUMNS
    assert dataset.combination == "G_MARKET_INTERNALS"
    assert np.isfinite(dataset.x.to_numpy()).all()
    assert dataset.frame["bas_dd"].min() >= daily["bas_dd"].unique()[199]
    assert dataset.signal_positions[-1] + 6 < len(dataset.raw_prices)


def test_F와_시장피처_결합은_기존_F와_같은_OOS후보날짜를_보존한다():
    index_prices, daily = _prices(rows=500)

    base = build_model_dataset(index_prices, "F", return_features=("five_day_return",))
    combined = build_combined_market_dataset(index_prices, daily)

    assert combined.feature_columns == COMBINED_MARKET_FEATURE_COLUMNS
    assert combined.frame["bas_dd"].tolist() == base.frame["bas_dd"].tolist()
