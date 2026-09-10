import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Cost Sensitivity · AlphaStack", layout="wide")

import theme
from theme import (
    metric_row,
    panel,
    section_header,
    styled_df,
    top_strip,
)

theme.inject()

from components import sidebar_controls
from services import backtest_service, data_service
from state import ctx

sidebar_controls()
c = ctx()

# ═══════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════
st.markdown('<div class="as-title">Cost Sensitivity</div>', unsafe_allow_html=True)

res = st.session_state.get("latest_model_result")
if res is None:
    top_strip([c["dataset"].upper(), "NO MODEL"],
              status_text="IDLE", status_tone="neutral")
    st.info("Overview에서 먼저 Run Analysis를 실행하세요.")
    st.stop()

prices = data_service.load_market_data(c["dataset"], c["start"], c["end"])
prices_test = prices.loc[res.test_index, "close"]
sig = pd.Series(res.y_pred, index=res.test_index)

top_strip(
    [c["dataset"].upper(),
     f"{c['start']} – {c['end']}",
     f"TEST {len(res.test_index)}d"],
    status_text="READY", status_tone="up",
)

# ═══════════════════════════════════════════════════════════
# ① CONTROLS
# ═══════════════════════════════════════════════════════════
section_header("CONTROLS")
with panel():
    c1, c2 = st.columns(2, gap="small")
    with c1:
        strategy = st.selectbox("Strategy", ["A", "B", "C"], index=0)
    with c2:
        position_size = st.slider(
            "Position Size (%)", 10, 100, 100, 10) / 100.0

    grid_text = st.text_input(
        "Costs (%, comma-separated)",
        value="0.00, 0.05, 0.10, 0.20, 0.30, 0.50")

try:
    grid = tuple(float(x.strip()) / 100.0
                 for x in grid_text.split(",") if x.strip())
except Exception:
    st.error("숫자를 콤마로 구분해서 입력하세요.")
    st.stop()
if not grid:
    st.stop()

# ═══════════════════════════════════════════════════════════
# ② RUN
# ═══════════════════════════════════════════════════════════
sens = backtest_service.cost_sensitivity(
    prices_test, sig, grid, strategy, position_size)


def _breakeven(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns or len(df) < 2:
        return None
    xs, ys = df.index.values, df[col].values
    for i in range(len(ys) - 1):
        if np.isnan(ys[i]) or np.isnan(ys[i + 1]):
            continue
        if (ys[i] >= 0) != (ys[i + 1] >= 0):
            x0, x1, y0, y1 = xs[i], xs[i + 1], ys[i], ys[i + 1]
            if y1 == y0:
                return float(x0)
            return float(x0 - y0 * (x1 - x0) / (y1 - y0))
    return None


be_sharpe = _breakeven(sens, "Sharpe")
be_cagr = _breakeven(sens, "CAGR")

# ═══════════════════════════════════════════════════════════
# ③ KPI
# ═══════════════════════════════════════════════════════════
base_trades = sens["trades"].iloc[0] if "trades" in sens else "-"

metric_row([
    dict(label="BREAK-EVEN · SHARPE=0",
         value=f"{be_sharpe*100:.3f}%" if be_sharpe is not None else "N/A",
         tone="warn" if be_sharpe is not None else "neutral",
         accent=True),
    dict(label="BREAK-EVEN · CAGR=0",
         value=f"{be_cagr*100:.3f}%" if be_cagr is not None else "N/A",
         tone="warn" if be_cagr is not None else "neutral"),
    dict(label="TRADES @ BASE COST",
         value=f"{base_trades}"),
], cols=3)

# ═══════════════════════════════════════════════════════════
# ④ METRICS vs COST (chart)
# ═══════════════════════════════════════════════════════════
section_header("METRICS vs COST")
with panel("SHARPE / CAGR / MDD", "GRID", "accent"):
    cost_pct = [x * 100 for x in sens.index.tolist()]
    fig = go.Figure()
    for col, color in [("Sharpe", "#7fd1ff"),
                       ("CAGR",   "#4ade80"),
                       ("MDD",    "#ff5c5c")]:
        if col in sens.columns:
            fig.add_trace(go.Scatter(
                x=cost_pct, y=sens[col], name=col,
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=6),
            ))
    fig.update_layout(
        height=380,
        showlegend=True,
        xaxis=dict(title="Cost (%)"),
        yaxis=dict(title=""),
    )
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# ⑤ GRID TABLE
# ═══════════════════════════════════════════════════════════
section_header("GRID TABLE")
show = sens.reset_index().copy()
if "cost" in show.columns:
    show["cost"] = (show["cost"] * 100).round(3).astype(str) + "%"

num_cols = [col for col in
            ["CAGR", "Vol", "Sharpe", "Sortino", "MDD",
             "Calmar", "Sterling"]
            if col in show.columns]

with panel(f"{len(show)} COST POINTS"):
    st.dataframe(
        styled_df(show, num_cols=num_cols, precision=3),
        use_container_width=True,
        hide_index=True,
    )

# ═══════════════════════════════════════════════════════════
# ⑥ REAL ENGINE
# ═══════════════════════════════════════════════════════════
section_header("REAL ENGINE")
try:
    from backtest.run_cost_sensitivity import find_breakeven_cost as _real_be

    with panel("backtest.run_cost_sensitivity.find_breakeven_cost",
               "IMPORT OK", "up"):
        try:
            real_be = _real_be(prices_test.values, sig.values)
            metric_row([
                dict(label="REAL ENGINE BREAK-EVEN",
                     value=(f"{float(real_be)*100:.3f}%"
                            if real_be is not None else "N/A"),
                     tone="warn" if real_be is not None else "neutral",
                     accent=True),
            ], cols=1)
        except Exception as e:
            st.caption(f"호출 시그니처 확인 필요: {e}")
except Exception as e:
    with panel("backtest.run_cost_sensitivity.find_breakeven_cost",
               "IMPORT FAIL", "warn"):
        st.caption(f"실 엔진 import 실패: {e}")
