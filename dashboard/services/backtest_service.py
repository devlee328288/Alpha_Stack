# dashboard/services/backtest_service.py
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

from services import evaluation_service

# ── 실 엔진 ──────────────────────────────────────────────
_ERRORS: list[str] = []
try:
    from backtest.backtest_strategies import (
        get_trade_ratio as _real_get_trade_ratio,
    )
    from backtest.backtest_strategies import (
        update_signal_streak as _real_update_streak,
    )
    _HAS_BT = True
except Exception as _e:
    _HAS_BT = False
    _ERRORS.append(f"backtest_strategies: {_e}")


def engine_status() -> list[str]:
    return list(_ERRORS)


# ── 신호 → 한국어 라벨 ────────────────────────────────────
_SIG_LABEL = {-1: "하락", 0: "보합", 1: "상승"}


def _to_kr(v) -> str:
    try:
        return _SIG_LABEL.get(int(v), "보합")
    except Exception:
        return "보합"


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    drawdown: pd.Series
    positions: pd.Series
    trades: int
    metrics: dict
    strategy: str
    engine: str          # "real" | "mock"


def _to_series(x) -> pd.Series:
    if isinstance(x, pd.DataFrame):
        for col in ("close", "Close", "adj_close", "Adj Close"):
            if col in x.columns:
                return x[col].astype(float)
        return x.select_dtypes("number").iloc[:, 0].astype(float)
    return pd.Series(x).astype(float)


def _positions_from_strategy(signals: pd.Series, strategy: str
                              ) -> tuple[pd.Series, str]:
    """실 엔진의 get_trade_ratio + update_signal_streak 로 포지션 시퀀스 생성."""
    if not _HAS_BT:
        return signals.astype(float).clip(-1, 1), "mock"

    try:
        ratios = np.zeros(len(signals), dtype=float)
        up_s = down_s = 0
        for i, v in enumerate(signals.values):
            sig = _to_kr(v)
            try:
                ratios[i] = float(_real_get_trade_ratio(strategy, sig, up_s, down_s))
            except Exception:
                ratios[i] = 0.0
            try:
                up_s, down_s = _real_update_streak(sig, up_s, down_s)
            except Exception:
                up_s = up_s + 1 if sig == "상승" else 0
                down_s = down_s + 1 if sig == "하락" else 0
        return pd.Series(ratios, index=signals.index), "real"
    except Exception as e:
        _ERRORS.append(f"strategy loop 실패: {e}")
        return signals.astype(float).clip(-1, 1), "mock"


# ── 백테스트 ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_backtest(
    prices,
    signals,
    cost: float,
    strategy: str = "A",
    position_size: float = 1.0,
) -> BacktestResult:
    prices = _to_series(prices)
    signals = _to_series(signals)

    # 1) 실 전략 로직으로 포지션 비율
    ratio, engine = _positions_from_strategy(signals, strategy)
    ratio = ratio * position_size

    # 2) 다음날 포지션 적용 (look-ahead 방지)
    pos = ratio.shift(1).fillna(0.0)

    # 3) 수익률
    ret = prices.pct_change().fillna(0.0)
    turnover = pos.diff().abs().fillna(0.0)
    strat = pos * ret - turnover * cost

    equity = (1 + strat).cumprod()
    dd = equity / equity.cummax() - 1
    trades = int((turnover > 1e-9).sum())

    metrics = evaluation_service.risk_metrics(strat)

    return BacktestResult(
        equity=equity,
        returns=strat,
        drawdown=dd,
        positions=pos,
        trades=trades,
        metrics=metrics,
        strategy=strategy,
        engine=engine,
    )


@st.cache_data(show_spinner=False)
def cost_sensitivity(
    prices,
    signals,
    cost_grid: tuple[float, ...],
    strategy: str = "A",
    position_size: float = 1.0,
) -> pd.DataFrame:
    rows = []
    for c in cost_grid:
        r = run_backtest(prices, signals, c, strategy, position_size)
        row = {"cost": c, "trades": r.trades}
        for k, v in r.metrics.items():
            try:
                row[k] = float(v)
            except Exception:
                row[k] = v
        rows.append(row)
    return pd.DataFrame(rows).set_index("cost")
