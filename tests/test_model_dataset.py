import numpy as np
import pandas as pd

from features.indicators import macd_hist_atr
from features.model_dataset import COMBINATION_FEATURES, build_model_dataset
from features.volatility import atr, atr_ratio, hv_regime
from features.volume import obv_slope_20


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


def test_지수파생피처는_인라인이아니라_원자함수와같은값을낸다():
    """PR #147 로 원자 함수가 생긴 넷을 이 파일이 다시 손으로 짜지 않는지 본다(#155).

    같은 공식이 두 곳에 있으면 원자 함수만 고쳐졌을 때 값이 조용히 갈린다.
    """
    prices = _index_prices()
    dataset = build_model_dataset(
        prices, "F", return_features=("daily_return", "five_day_return")
    )
    frame = dataset.frame.set_index("bas_dd")

    high = prices["high"].to_numpy(dtype=float)
    low = prices["low"].to_numpy(dtype=float)
    close = prices["close"].to_numpy(dtype=float)
    volume = prices["volume"].to_numpy(dtype=float)
    expected = pd.DataFrame(
        {
            "bas_dd": prices["bas_dd"],
            "atr_ratio": atr_ratio(high, low, close, 14),
            "hv_regime": hv_regime(close, 20, 250),
            "obv_slope_20": obv_slope_20(close, volume, 20),
            "macd_hist_atr": macd_hist_atr(close, atr(high, low, close, 14)),
        }
    ).set_index("bas_dd")

    shared = frame.index.intersection(expected.index)
    assert len(shared) > 0
    for column in ("atr_ratio", "hv_regime", "obv_slope_20", "macd_hist_atr"):
        # 칸이 사라졌으면 조용히 건너뛰지 않고 실패한다 — 그래야 시험이 실제로 잰다.
        assert column in frame.columns
        assert np.allclose(
            frame.loc[shared, column].to_numpy(dtype=float),
            expected.loc[shared, column].to_numpy(dtype=float),
            equal_nan=True,
        )


def test_G조합은_E와_B의_선정피처_합집합이다():
    assert COMBINATION_FEATURES["G"] == (
        "atr_ratio",
        "bb_bandwidth",
        "hv_regime",
        "five_day_return",
        "sma_gap_5_20",
        "macd_hist_ratio",
        "rsi_14",
        "hv_20",
    )
    dataset = build_model_dataset(_index_prices(), "G")
    assert dataset.feature_columns == COMBINATION_FEATURES["G"]
