# dashboard/services/backtest_5day_service.py
"""
5-Day Horizon Backtest.

모델 라벨이 T+1 open → T+6 open (5거래일) 이므로 매매도 5일 홀드로 통일.

- A5: 신호 1개 → 5일 홀드 1포지션 → 다음 신호까지 대기 (5일 텀)
- D5: 매일 신호 → 5일 홀드 슬리브 추가 (최대 N개 겹침)
"""

from __future__ import annotations
import contextlib, io, sys
from collections import deque
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
from services import data_loader, backtest_service


def _silence():
    return (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    )


def _sleeve_size(sig: str, up: int, dn: int) -> float:
    if sig == "상승":
        return 0.20 if up <= 1 else (0.30 if up == 2 else 0.50)
    if sig == "하락":
        return -0.20 if dn <= 1 else (-0.30 if dn == 2 else -0.50)
    return 0.0


def _streak(sig: str, up: int, dn: int):
    if sig == "상승":
        return up + 1, 0
    if sig == "하락":
        return 0, dn + 1
    return 0, 0


def _signals(prices, predict_func, trading_days):
    out = {}
    for d in trading_days:
        try:
            r = predict_func(d, prices)
            out[d] = r[0] if isinstance(r, tuple) else r
        except Exception:
            out[d] = "중립"
    return pd.Series(out)


def _metrics(equity, rets, initial_cash):
    total_ret = equity.iloc[-1] / initial_cash - 1
    years = len(equity) / 252
    ann = (equity.iloc[-1] / initial_cash) ** (1 / years) - 1 if years > 0 else np.nan
    vol = rets.std(ddof=1) * np.sqrt(252) if len(rets) > 1 else np.nan
    shrp = (
        rets.mean() / rets.std(ddof=1) * np.sqrt(252)
        if len(rets) > 1 and rets.std(ddof=1) > 0
        else np.nan
    )
    mdd = (equity / equity.cummax() - 1).min()
    wr = (rets > 0).mean() if len(rets) else 0.0
    gain = rets[rets > 0].sum()
    loss = abs(rets[rets < 0].sum())
    pf = float(gain / loss) if loss > 0 else float("inf")
    return {
        "total_return": float(total_ret),
        "annual_return": float(ann),
        "volatility": float(vol),
        "sharpe_ratio": float(shrp) if np.isfinite(shrp) else None,
        "max_drawdown": float(mdd),
        "win_rate_daily": float(wr),
        "profit_factor": pf,
        "final_portfolio_value": float(equity.iloc[-1]),
    }


# ═══════════════════════════════════════════════════════════
# A5: 5일 홀드, 1포지션, 5일 텀
# ═══════════════════════════════════════════════════════════
def _run_a5(
    prices, signals, trading_days, hold_days=5, initial_cash=100.0, trade_cost=0.001
):
    cash = float(initial_cash)
    shares = 0.0
    entry_idx = None
    next_free_idx = 0
    up = dn = 0

    eq_list, ret_list, pos_list = [], [], []
    trade_count = 0
    total_cost = 0.0
    log = []

    for i, date in enumerate(trading_days):
        # 청산 체크
        if entry_idx is not None and i >= entry_idx + hold_days:
            # 오늘 시가에 청산
            open_today = float(prices.loc[date, "open"])
            proceeds = shares * open_today
            cost = abs(proceeds) * trade_cost
            cash += proceeds - cost
            shares = 0.0
            total_cost += cost
            trade_count += 1
            log.append({"date": date, "action": "exit", "reason": f"hold {hold_days}d"})
            entry_idx = None
            next_free_idx = i

        # 진입 체크 (포지션 없고, 진입 가능 시점)
        if entry_idx is None and i == next_free_idx and i < len(trading_days) - 1:
            sig = signals.loc[date]
            up, dn = _streak(sig, up, dn)
            size = _sleeve_size(sig, up, dn)
            if abs(size) > 1e-8:
                next_open = float(prices.loc[trading_days[i + 1], "open"])
                # 오늘 종가 기준 PV로 사이징
                close_today = float(prices.loc[date, "close"])
                pv = cash + shares * close_today
                target_val = pv * size
                qty = target_val / next_open
                cost = abs(target_val) * trade_cost
                cash -= target_val + cost
                shares += qty
                total_cost += cost
                trade_count += 1
                entry_idx = i + 1
                log.append(
                    {
                        "date": date,
                        "action": "enter",
                        "signal": sig,
                        "size": size,
                        "exec": trading_days[i + 1],
                    }
                )

        # 매일 valuation
        close_today = float(prices.loc[date, "close"])
        pv = cash + shares * close_today
        eq_list.append(pv)
        pos_list.append((shares * close_today / pv) if pv > 0 else 0.0)
        if i > 0:
            prev = eq_list[-2]
            ret_list.append(pv / prev - 1 if prev > 0 else 0.0)

    eq = pd.Series(eq_list, index=trading_days)
    rets = pd.Series(ret_list, index=trading_days[1:])
    pos = pd.Series(pos_list, index=trading_days)
    m = _metrics(eq, rets, initial_cash)
    m["num_trades"] = trade_count
    m["total_transaction_cost"] = total_cost
    return {
        "equity": eq,
        "daily_returns": rets,
        "position_ratios": pos,
        "trade_log": pd.DataFrame(log),
        "metrics": m,
    }


# ═══════════════════════════════════════════════════════════
# D5: 매일 슬리브 진입, 5일 유지, 겹침
# ═══════════════════════════════════════════════════════════
def _run_d5(
    prices, signals, trading_days, sleeve_days=5, initial_cash=100.0, trade_cost=0.001
):
    cash = float(initial_cash)
    sleeves = deque()  # (entry_idx, shares)
    up = dn = 0

    eq_list, ret_list, pos_list, cnt_list = [], [], [], []
    trade_count = 0
    total_cost = 0.0
    log = []

    for i, date in enumerate(trading_days):
        # 5일 지난 슬리브 청산
        while sleeves and i >= sleeves[0][0] + sleeve_days:
            _, sh = sleeves.popleft()
            open_today = float(prices.loc[date, "open"])
            proceeds = sh * open_today
            cost = abs(proceeds) * trade_cost
            cash += proceeds - cost
            total_cost += cost
            trade_count += 1
            log.append({"date": date, "action": "exit", "shares": sh})

        # 매일 새 슬리브 진입 (다음날 시가)
        if i < len(trading_days) - 1:
            sig = signals.loc[date]
            up, dn = _streak(sig, up, dn)
            size = _sleeve_size(sig, up, dn)
            if abs(size) > 1e-8:
                close_today = float(prices.loc[date, "close"])
                stock_val = sum(sh * close_today for _, sh in sleeves)
                pv = cash + stock_val
                next_open = float(prices.loc[trading_days[i + 1], "open"])
                target_val = pv * size
                qty = target_val / next_open
                cost = abs(target_val) * trade_cost
                cash -= target_val + cost
                sleeves.append((i + 1, qty))
                total_cost += cost
                trade_count += 1
                log.append(
                    {
                        "date": date,
                        "action": "enter",
                        "signal": sig,
                        "size": size,
                        "exec": trading_days[i + 1],
                    }
                )

        close_today = float(prices.loc[date, "close"])
        stock_val = sum(sh * close_today for _, sh in sleeves)
        pv = cash + stock_val
        eq_list.append(pv)
        pos_list.append((stock_val / pv) if pv > 0 else 0.0)
        cnt_list.append(len(sleeves))
        if i > 0:
            prev = eq_list[-2]
            ret_list.append(pv / prev - 1 if prev > 0 else 0.0)

    eq = pd.Series(eq_list, index=trading_days)
    rets = pd.Series(ret_list, index=trading_days[1:])
    pos = pd.Series(pos_list, index=trading_days)
    cnt = pd.Series(cnt_list, index=trading_days)
    m = _metrics(eq, rets, initial_cash)
    m["num_trades"] = trade_count
    m["total_transaction_cost"] = total_cost
    return {
        "equity": eq,
        "daily_returns": rets,
        "position_ratios": pos,
        "sleeve_counts": cnt,
        "trade_log": pd.DataFrame(log),
        "metrics": m,
    }


# ═══════════════════════════════════════════════════════════
# Wrapper
# ═══════════════════════════════════════════════════════════
def run_5day_comparison(
    scope: str,
    ticker: str,
    predictor: str = "RandomForest",
    baseline_kind: str = "fwd_return",
    start: str = "2023-01-01",
    end: str = "2024-08-22",
    initial_cash: float = 100.0,
    trade_cost: float = 0.001,
    hold_days: int = 5,
    source: str = "stocks30",
) -> dict:
    if source == "full" and scope == "STOCK":
        prices = data_loader.load_market_data_full(ticker)
    else:
        prices = data_loader.load_market_data(scope, ticker)

    if predictor == "Random":
        predict_func = backtest_service._make_random_predictor()
    else:
        predict_func, _ = backtest_service._make_model_predictor(
            scope=scope,
            ticker=ticker,
            model_name=predictor,
            baseline_kind=baseline_kind,  # ← 추가
            source=source,
        )

    s, e = pd.Timestamp(start), pd.Timestamp(end)
    days = prices.index[(prices.index >= s) & (prices.index <= e)]
    if len(days) < hold_days + 20:
        raise ValueError(f"기간 부족: {len(days)}일")

    _o, _e = _silence()
    with _o, _e:
        sigs = _signals(prices, predict_func, days)

    a5 = _run_a5(prices, sigs, days, hold_days, initial_cash, trade_cost)
    d5 = _run_d5(prices, sigs, days, hold_days, initial_cash, trade_cost)

    return {
        "A5": a5,
        "D5": d5,
        "signals": sigs,
        "hold_days": int(hold_days),
        "baseline_kind": baseline_kind,
        "predictor": predictor,
        "start": start,
        "end": end,
        "initial_cash": float(initial_cash),
        "trade_cost": float(trade_cost),
        "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def clear_cache() -> None:
    pass
