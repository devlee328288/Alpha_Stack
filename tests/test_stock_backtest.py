import numpy as np
import pandas as pd
import pytest

from evaluation.stock_backtest import run_overlapping_stock_backtest


def _ranking(index_predicted: int, stock_predicted: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold": [1],
            "model": ["모델"],
            "bas_dd": ["20240101"],
            "code": ["000001"],
            "industry_index_name": ["건설"],
            "index_predicted": [index_predicted],
            "predicted": [stock_predicted],
            "selected_probability": [0.7],
            "index_direction_rank": [1],
            "entry_bas_dd": ["20240102"],
            "exit_bas_dd": ["20240107"],
            "entry_adj_open": [100.0],
            "exit_adj_open": [100.0],
            "label_numeric": [0],
            "p_down": [0.1],
            "p_neutral": [0.2],
            "p_up": [0.7],
        }
    )


def _prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bas_dd": [f"2024010{day}" for day in range(2, 8)],
            "code": ["000001"] * 6,
            "adj_open": [100.0] * 6,
        }
    )


def test_상승예측은_t1진입_t6청산비용을모두낸다():
    result = run_overlapping_stock_backtest(
        _ranking(1),
        _prices(),
        top_n=1,
        round_trip_cost=0.01,
    )

    assert result.summary["trade_rows"] == 1
    assert result.trade_log.loc[0, "entry_bas_dd"] == "20240102"
    assert result.trade_log.loc[0, "exit_bas_dd"] == "20240107"
    assert result.trade_log.loc[0, "index_predicted"] == 1
    assert result.trade_log.loc[0, "predicted"] == 1
    expected = 0.2 * (np.prod([0.995, 0.995]) - 1.0)
    assert np.isclose(result.summary["total_return"], expected)


def test_중립하락예측일은종목순위를남겨도매수하지않는다():
    result = run_overlapping_stock_backtest(
        _ranking(0),
        _prices(),
        top_n=1,
        round_trip_cost=0.01,
    )

    assert result.summary["trade_rows"] == 0
    assert result.summary["total_return"] == 0.0
    assert result.summary["turnover"] == 0.0


def test_지수가_상승이어도_종목이_상승_예측이_아니면_매수하지_않는다():
    result = run_overlapping_stock_backtest(
        _ranking(1, stock_predicted=0),
        _prices(),
        top_n=1,
        round_trip_cost=0.01,
    )

    assert result.summary["trade_rows"] == 0
    assert result.summary["total_return"] == 0.0
    assert result.summary["turnover"] == 0.0


def test_실제보유일의_0원시가는_전방채움하지않고오류로막는다():
    prices = _prices()
    prices.loc[prices["bas_dd"].eq("20240104"), "adj_open"] = 0.0

    with pytest.raises(ValueError, match="20240104 000001"):
        run_overlapping_stock_backtest(
            _ranking(1),
            prices,
            top_n=1,
            round_trip_cost=0.0,
        )


def test_거래하지않는종목의_결측시가는_전체백테스트를막지않는다():
    prices = pd.concat(
        [
            _prices(),
            pd.DataFrame(
                {
                    "bas_dd": ["20240104"],
                    "code": ["999999"],
                    "adj_open": [np.nan],
                }
            ),
        ],
        ignore_index=True,
    )

    result = run_overlapping_stock_backtest(
        _ranking(1),
        prices,
        top_n=1,
        round_trip_cost=0.0,
    )

    assert result.summary["total_return"] == 0.0
