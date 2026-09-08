import numpy as np
import pandas as pd

from evaluation.stock_ledger import build_stock_ledgers, verify_stock_ledgers
from supply.backtest_ledger import SIGNAL_LOG_COLUMNS, TRADE_LOG_COLUMNS


def _ranking(direction: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold": [1, 1],
            "bas_dd": ["20240102", "20240102"],
            "code": ["000001", "000002"],
            "industry_index_name": ["건설", "금속"],
            "index_predicted": [direction, direction],
            "index_direction_rank": [1, 2],
            "selected_probability": [0.7, 0.6],
            "entry_bas_dd": ["20240103", "20240103"],
            "exit_bas_dd": ["20240110", "20240110"],
            "entry_adj_open": [100.0, 200.0],
            "exit_adj_open": [110.0, 180.0],
            "p_down": [0.1, 0.2],
            "p_neutral": [0.2, 0.2],
            "p_up": [0.7, 0.6],
        }
    )


def test_개별종목출력이_공통17칸22칸원장계약을통과한다():
    result = build_stock_ledgers(
        _ranking(),
        top_n=2,
        cost_rate=0.0028,
        model_id="A-LogisticRegression+E-RandomForest",
        model_rev="test-rev",
        run_id="test-run",
    )

    assert list(result.signal_log.columns[:17]) == list(SIGNAL_LOG_COLUMNS)
    assert list(result.trade_log.columns[:22]) == list(TRADE_LOG_COLUMNS)
    assert result.verification["signal"]["problems"] == []
    assert result.verification["trade"]["problems"] == []
    assert set(result.trade_log["action"]) == {"buy", "sell"}
    assert len(result.trade_log) == 4
    assert np.allclose(result.signal_log["realized_return_5d"], [0.1, -0.1])


def test_중립예측은신호원장에남고거래원장은빈다():
    result = build_stock_ledgers(
        _ranking(0),
        top_n=1,
        cost_rate=0.0043,
        model_id="model",
        model_rev="rev",
        run_id="run",
    )

    assert len(result.signal_log) == 1
    assert result.signal_log.loc[0, "signal"] == "중립"
    assert result.trade_log.empty


def test_청산일이홀드아웃이면별도실행검사가막는다():
    signal = build_stock_ledgers(
        _ranking(),
        top_n=1,
        cost_rate=0.0,
        model_id="model",
        model_rev="rev",
        run_id="run",
    ).signal_log
    signal["exit_execution_date"] = "20240902"

    verification = verify_stock_ledgers(
        signal,
        pd.DataFrame(columns=TRADE_LOG_COLUMNS),
    )

    assert any("홀드아웃" in problem for problem in verification["stock_execution"]["problems"])


def test_진입청산가격수익률이틀리면검산에서잡는다():
    result = build_stock_ledgers(
        _ranking(),
        top_n=1,
        cost_rate=0.0,
        model_id="model",
        model_rev="rev",
        run_id="run",
    )
    signal = result.signal_log.copy()
    signal.loc[0, "realized_return_5d"] = 0.5

    verification = verify_stock_ledgers(signal, result.trade_log)

    assert any("다릅니다" in problem for problem in verification["stock_execution"]["problems"])
