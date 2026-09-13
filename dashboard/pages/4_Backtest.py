import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Backtest · AlphaStack", layout="wide")

import theme
from theme import (
    metric_row, section_header, top_strip, panel,
    show_table, plotly_chart,
)
theme.inject()

from components import sidebar_controls
from services import backtest_service

sidebar_controls()

st.markdown('<div class="as-title">Backtest</div>', unsafe_allow_html=True)

_err = backtest_service.engine_status()
if _err:
    top_strip(["BACKTEST"], status_text="IMPORT FAIL", status_tone="warn")
    st.error(_err)
    st.stop()

top_strip(
    ["KOSPI200", "STRATEGY A/B/C", "SIGNAL · 5D HORIZON", "RANDOM PREDICTOR"],
    status_text="READY", status_tone="neutral",
)

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════
section_header("CONFIG")
with panel():
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1], gap="small")
    with c1:
        start = st.date_input("Start", value=pd.Timestamp("2023-01-01")).strftime("%Y-%m-%d")
    with c2:
        end = st.date_input("End", value=pd.Timestamp("2024-08-22")).strftime("%Y-%m-%d")
    with c3:
        initial_cash = st.number_input("Initial Cash", 100.0, 1e7, 100.0, 100.0)
        trade_cost = st.number_input(
            "Trade Cost", 0.0, 0.01, 0.001, 0.0005, format="%.4f",
        )
    with c4:
        st.write("")
        run = st.button("▶  RUN ALL (A/B/C)", type="primary",
                        use_container_width=True)

# ═══════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════
if run:
    import traceback
    with st.spinner("Running A/B/C backtest…"):
        try:
            results = backtest_service.run_all_strategies(
                start, end, initial_cash, trade_cost,
            )
            st.session_state["_bt_results"] = results
        except Exception as e:
            st.error(f"실행 실패: {type(e).__name__}: {e}")
            with st.expander("Traceback", expanded=True):
                st.code(traceback.format_exc())

results = st.session_state.get("_bt_results")
if not results:
    st.caption("Start/End를 확인하고 RUN ALL을 누르세요.")
    st.stop()

# ═══════════════════════════════════════════════════════════
# COMPARISON METRIC
# ═══════════════════════════════════════════════════════════
section_header("STRATEGY COMPARISON")

rows = []
for s, r in results.items():
    m = r["metrics"]
    rows.append({
        "STRATEGY": s,
        "TOTAL RETURN": m["total_return"],
        "ANNUAL RETURN": m["annual_return"],
        "VOLATILITY": m["volatility"],
        "SHARPE": m["sharpe_ratio"],
        "MDD": -abs(m["max_drawdown"]),
        "WIN RATE": m["win_rate_daily"],
        "PROFIT FACTOR": m["profit_factor"],
        "TRADES": m["num_trades"],
        "FINAL VALUE": m["final_portfolio_value"],
    })
cmp_df = pd.DataFrame(rows)

with panel(f"{len(results)} STRATEGIES"):
    show_table(
        cmp_df,
        num_cols=["TOTAL RETURN", "ANNUAL RETURN", "VOLATILITY", "SHARPE",
                  "MDD", "WIN RATE", "PROFIT FACTOR", "FINAL VALUE"],
        precision=4,
    )

# ═══════════════════════════════════════════════════════════
# EQUITY CURVE
# ═══════════════════════════════════════════════════════════
section_header("EQUITY CURVE")

with panel("A / B / C · PORTFOLIO VALUE"):
    fig = go.Figure()
    colors = {"A": "#7fd1ff", "B": "#4ade80", "C": "#fbbf24"}
    for s, r in results.items():
        eq = r["equity"]
        fig.add_trace(go.Scatter(
            x=eq.index, y=eq.values, name=f"Strategy {s}",
            line=dict(color=colors.get(s, "#9aa0a6"), width=1.6),
        ))
    fig.update_layout(
        height=380,
        xaxis=dict(title=""),
        yaxis=dict(title="Portfolio Value"),
    )
    plotly_chart(fig)

# ═══════════════════════════════════════════════════════════
# DRAWDOWN
# ═══════════════════════════════════════════════════════════
section_header("DRAWDOWN")

with panel("A / B / C"):
    fig = go.Figure()
    for s, r in results.items():
        eq = r["equity"]
        dd = eq / eq.cummax() - 1
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values, name=f"Strategy {s}",
            line=dict(color=colors.get(s, "#9aa0a6"), width=1.2),
            fill="tozeroy",
            fillcolor="rgba(0,0,0,0)",
        ))
    fig.update_layout(height=300, yaxis=dict(tickformat=".1%", title=""))
    plotly_chart(fig)

# ═══════════════════════════════════════════════════════════
# DETAIL
# ═══════════════════════════════════════════════════════════
section_header("STRATEGY DETAIL")
pick = st.selectbox("Inspect", list(results.keys()), index=0)
m = results[pick]["metrics"]

metric_row([
    dict(label="TOTAL RETURN", value=f"{m['total_return']*100:.2f}%",
         tone="up" if m["total_return"] > 0 else "down"),
    dict(label="ANNUAL RETURN", value=f"{m['annual_return']*100:.2f}%",
         tone="up" if m["annual_return"] > 0 else "down"),
    dict(label="SHARPE", value=f"{m['sharpe_ratio']:.4f}",
         tone="up" if m["sharpe_ratio"] > 0 else "down"),
    dict(label="MDD", value=f"{-abs(m['max_drawdown'])*100:.2f}%", tone="down"),
    dict(label="WIN RATE", value=f"{m['win_rate_daily']*100:.2f}%"),
    dict(label="PROFIT FACTOR", value=f"{m['profit_factor']:.2f}"),
], cols=6)

metric_row([
    dict(label="AVG WIN", value=f"{m['avg_win']*100:.4f}%", tone="up"),
    dict(label="AVG LOSS", value=f"{m['avg_loss']*100:.4f}%", tone="down"),
    dict(label="MAX WIN STREAK", value=f"{m['max_win_streak']}"),
    dict(label="MAX LOSS STREAK", value=f"{m['max_loss_streak']}"),
    dict(label="TOTAL TRADES", value=f"{m['num_trades']}"),
    dict(label="TRANSACTION COST", value=f"{m['total_transaction_cost']:.4f}"),
], cols=6)