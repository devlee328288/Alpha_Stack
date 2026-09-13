import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Cost Sensitivity · AlphaStack", layout="wide")

import theme
from theme import (
    metric_row, section_header, top_strip, panel,
    show_table, plotly_chart,
)
theme.inject()

from components import sidebar_controls
from services import backtest_service

sidebar_controls()

st.markdown('<div class="as-title">Cost Sensitivity</div>', unsafe_allow_html=True)

if not backtest_service.cost_available():
    top_strip(["COST SENS"], status_text="IMPORT FAIL", status_tone="warn")
    st.error(backtest_service.engine_status())
    st.stop()

top_strip(
    ["KOSPI200", "4 COST PRESETS × 3 STRATEGIES", "BREAKEVEN (Sharpe=0)"],
    status_text="READY", status_tone="neutral",
)

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════
section_header("CONFIG")
with panel():
    c1, c2, c3 = st.columns([1.2, 1.2, 1], gap="small")
    with c1:
        start = st.date_input("Start", value=pd.Timestamp("2023-01-01")).strftime("%Y-%m-%d")
    with c2:
        end = st.date_input("End", value=pd.Timestamp("2024-08-22")).strftime("%Y-%m-%d")
    with c3:
        st.write("")
        run = st.button("▶  RUN", type="primary", use_container_width=True)

# ═══════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════
if run:
    import traceback
    with st.spinner("Running cost grid (4 × 3 = 12 backtests)…"):
        try:
            grid = backtest_service.run_cost_grid(start, end)
            be = backtest_service.run_breakeven(start, end)
            st.session_state["_cost_grid"] = grid
            st.session_state["_cost_be"] = be
        except Exception as e:
            st.error(f"실행 실패: {type(e).__name__}: {e}")
            with st.expander("Traceback", expanded=True):
                st.code(traceback.format_exc())

grid = st.session_state.get("_cost_grid")
be = st.session_state.get("_cost_be")
if grid is None or be is None:
    st.caption("Start/End 확인 후 RUN을 누르세요.")
    st.stop()

# ═══════════════════════════════════════════════════════════
# BREAKEVEN
# ═══════════════════════════════════════════════════════════
section_header("BREAKEVEN COST · SHARPE = 0")
be_rows = be.to_dict("records")
metric_row([
    dict(label=f"STRATEGY {r['strategy']}",
         value=f"{r['breakeven_cost']*100:.3f}%",
         tone="up" if r["breakeven_cost"] > 0.002 else "warn",
         accent=True)
    for r in be_rows
], cols=len(be_rows))

# ═══════════════════════════════════════════════════════════
# GRID TABLE
# ═══════════════════════════════════════════════════════════
section_header("COST GRID · 4 PRESETS × 3 STRATEGIES")
with panel("SHARPE · CAGR · MDD"):
    disp = grid.copy()
    disp["cost_pct"] = (disp["cost_rate"] * 100).round(3).astype(str) + "%"
    disp = disp[["strategy", "cost_label", "cost_pct",
                 "sharpe", "cagr", "mdd", "total_return", "num_trades"]]
    show_table(
        disp,
        num_cols=["sharpe", "cagr", "mdd", "total_return"],
        precision=4,
    )

# ═══════════════════════════════════════════════════════════
# SHARPE vs COST
# ═══════════════════════════════════════════════════════════
section_header("SHARPE vs COST")

with panel("STRATEGY A / B / C"):
    fig = go.Figure()
    colors = {"A": "#7fd1ff", "B": "#4ade80", "C": "#fbbf24"}
    for s in sorted(grid["strategy"].unique()):
        sub = grid[grid["strategy"] == s].sort_values("cost_rate")
        fig.add_trace(go.Scatter(
            x=sub["cost_rate"] * 100, y=sub["sharpe"],
            name=f"Strategy {s}",
            mode="lines+markers",
            line=dict(color=colors.get(s, "#9aa0a6"), width=2),
            marker=dict(size=8),
        ))
    # breakeven 표시
    for r in be_rows:
        fig.add_vline(
            x=r["breakeven_cost"] * 100,
            line_dash="dash",
            line_color=colors.get(r["strategy"], "#9aa0a6"),
            opacity=0.5,
        )
    fig.update_layout(
        height=380,
        xaxis=dict(title="Cost (%)"),
        yaxis=dict(title="Sharpe"),
        hovermode="x unified",
    )
    plotly_chart(fig)

# ═══════════════════════════════════════════════════════════
# DETAIL BY STRATEGY
# ═══════════════════════════════════════════════════════════
section_header("DETAIL BY STRATEGY")
pick = st.selectbox("Strategy", sorted(grid["strategy"].unique()), index=0)
sub = grid[grid["strategy"] == pick].sort_values("cost_rate").copy()
sub["cost_pct"] = (sub["cost_rate"] * 100).round(3).astype(str) + "%"
with panel(f"STRATEGY {pick}"):
    show_table(
        sub[["cost_pct", "cost_label", "sharpe", "cagr", "mdd",
             "total_return", "num_trades"]],
        num_cols=["sharpe", "cagr", "mdd", "total_return"],
        precision=4,
    )