# dashboard/services/backtest_service.py
"""
백테스트 서비스 — backtest/backtest_strategies 엔진 호출.
Streamlit caching 안 씀 (stdout ASCII 캡처 문제 회피).
"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_BT_DIR = _REPO_ROOT / "backtest"
if str(_BT_DIR) not in sys.path:
    sys.path.insert(0, str(_BT_DIR))

import pandas as pd

_IMPORT_ERR: str | None = None
_HAS_BT = False
_HAS_COST = False

try:
    from backtest_strategies import (     # type: ignore
        load_data, predict_5d_after, run_backtest,
    )
    _HAS_BT = True
except Exception as e:
    _IMPORT_ERR = f"backtest_strategies: {type(e).__name__}: {e}"

try:
    from run_cost_sensitivity import (    # type: ignore
        find_breakeven_cost, COST_PRESETS, COST_LABELS, STRATEGIES,
    )
    _HAS_COST = True
except Exception as e:
    if _IMPORT_ERR is None:
        _IMPORT_ERR = f"run_cost_sensitivity: {type(e).__name__}: {e}"


def engine_status() -> str | None:
    return _IMPORT_ERR


def cost_available() -> bool:
    return _HAS_COST


STRATEGIES = ("A", "B", "C")


def _silence():
    return (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    )


# ═══════════════════════════════════════════════════════════
# 수동 캐시
# ═══════════════════════════════════════════════════════════
_DF_CACHE: dict = {}
_BT_CACHE: dict = {}
_COST_GRID_CACHE: dict = {}
_BE_CACHE: dict = {}


def get_market_data() -> pd.DataFrame:
    if "df" not in _DF_CACHE:
        _out, _err = _silence()
        with _out, _err:
            _DF_CACHE["df"] = load_data()
    return _DF_CACHE["df"]


def run_single(
    strategy: str,
    start: str,
    end: str,
    initial_cash: float = 100.0,
    trade_cost: float = 0.001,
    predictor: str = "random",
) -> dict:
    """단일 전략 실행. `strategy` in A/B/C."""
    if not _HAS_BT:
        raise RuntimeError(f"backtest import 실패: {_IMPORT_ERR}")

    key = (strategy, start, end, float(initial_cash), float(trade_cost), predictor)
    if key in _BT_CACHE:
        return _BT_CACHE[key]

    df = get_market_data()

    _out, _err = _silence()
    with _out, _err:
        result = run_backtest(
            market_data=df,
            start_date=pd.Timestamp(start),
            end_date=pd.Timestamp(end),
            predict_func=predict_5d_after,
            strategy=strategy,
            initial_cash=float(initial_cash),
            trade_cost=float(trade_cost),
            model_id=f"{predictor}-v0",
        )

    # DataFrame을 dict로 변환, 시계열은 별도 저장
    out = {
        "strategy": strategy,
        "start": start,
        "end": end,
        "initial_cash": float(initial_cash),
        "trade_cost": float(trade_cost),
        "metrics": {
            "total_return": float(result["total_return"]),
            "annual_return": float(result["annual_return"]),
            "volatility": float(result["volatility"]),
            "sharpe_ratio": float(result["sharpe_ratio"]),
            "max_drawdown": float(result["max_drawdown"]),
            "win_rate_daily": float(result["win_rate_daily"]),
            "avg_win": float(result["avg_win"]),
            "avg_loss": float(result["avg_loss"]),
            "profit_factor": float(result["profit_factor"]),
            "max_win_streak": int(result["max_win_streak"]),
            "max_loss_streak": int(result["max_loss_streak"]),
            "num_trades": int(result["num_trades"]),
            "buy_trades": int(result["buy_trades"]),
            "sell_trades": int(result["sell_trades"]),
            "total_transaction_cost": float(result["total_transaction_cost"]),
            "final_portfolio_value": float(result["final_portfolio_value"]),
            "final_position_ratio": float(result["final_position_ratio"]),
        },
        "equity": result["portfolio_series"],
        "daily_returns": result["daily_returns"],
        "signal_log": result["signal_log"].tail(50).to_dict("records"),
    }
    _BT_CACHE[key] = out
    return out


def run_all_strategies(
    start: str,
    end: str,
    initial_cash: float = 100.0,
    trade_cost: float = 0.001,
    predictor: str = "random",
) -> dict[str, dict]:
    out = {}
    for s in STRATEGIES:
        out[s] = run_single(s, start, end, initial_cash, trade_cost, predictor)
    return out


# ═══════════════════════════════════════════════════════════
# Cost Sensitivity
# ═══════════════════════════════════════════════════════════
def run_cost_grid(
    start: str,
    end: str,
    initial_cash: float = 100.0,
    predictor: str = "random",
) -> pd.DataFrame:
    """COST_PRESETS × STRATEGIES 그리드."""
    if not _HAS_COST:
        raise RuntimeError(f"cost import 실패: {_IMPORT_ERR}")

    key = (start, end, float(initial_cash), predictor)
    if key in _COST_GRID_CACHE:
        return _COST_GRID_CACHE[key]

    rows = []
    for cost_key, cost_rate in COST_PRESETS.items():
        for s in STRATEGIES:
            r = run_single(s, start, end, initial_cash, cost_rate, predictor)
            m = r["metrics"]
            rows.append({
                "strategy": s,
                "cost_key": cost_key,
                "cost_label": COST_LABELS[cost_key],
                "cost_rate": cost_rate,
                "sharpe": m["sharpe_ratio"],
                "cagr": m["annual_return"],
                "mdd": m["max_drawdown"],
                "total_return": m["total_return"],
                "num_trades": m["num_trades"],
            })
    df = pd.DataFrame(rows)
    _COST_GRID_CACHE[key] = df
    return df


def run_breakeven(
    start: str,
    end: str,
    initial_cash: float = 100.0,
) -> pd.DataFrame:
    if not _HAS_COST:
        raise RuntimeError(f"cost import 실패: {_IMPORT_ERR}")

    key = (start, end, float(initial_cash))
    if key in _BE_CACHE:
        return _BE_CACHE[key]

    df = get_market_data()
    rows = []
    for s in STRATEGIES:
        _out, _err = _silence()
        with _out, _err:
            be = find_breakeven_cost(
                strategy=s,
                market_data=df,
                predict_func=predict_5d_after,
                start_date=pd.Timestamp(start),
                end_date=pd.Timestamp(end),
            )
        rows.append({"strategy": s, "breakeven_cost": float(be)})
    out = pd.DataFrame(rows)
    _BE_CACHE[key] = out
    return out


def clear_cache() -> None:
    _DF_CACHE.clear()
    _BT_CACHE.clear()
    _COST_GRID_CACHE.clear()
    _BE_CACHE.clear()