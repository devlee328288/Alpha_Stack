import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Backtest · AlphaStack", layout="wide")

import theme
from theme import metric_row, panel, section_header, styled_df, top_strip

theme.inject()

import charts
from components import sidebar_controls
from services import backtest_service, data_service
from state import ctx

sidebar_controls()
c = ctx()

# ═══════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════
st.markdown('<div class="as-title">Backtest</div>', unsafe_allow_html=True)

res = st.session_state.get("latest_model_result")
if res is None:
    top_strip([c["dataset"].upper(), "NO MODEL"], status_text="IDLE", status_tone="neutral")
    st.info("Overview에서 먼저 Run Analysis를 실행하세요.")
    st.stop()

for e in backtest_service.engine_status():
    st.warning(f"⚠ backtest engine: {e}")

prices = data_service.load_market_data(c["dataset"], c["start"], c["end"])
prices_test = prices.loc[res.test_index, "close"]
sig = pd.Series(res.y_pred, index=res.test_index)

# ═══════════════════════════════════════════════════════════
# ① CONTROLS
# ═══════════════════════════════════════════════════════════
section_header("CONTROLS")
with panel():
    c1, c2, c3 = st.columns(3, gap="small")
    with c1:
        strategy = st.selectbox(
            "Strategy", ["A", "B", "C"], index=0,
            help="A: 단계별 분할매매 · B: 고정 25% · C: 올인/아웃",
        )
    with c2:
        position_size = st.slider(
            "Position Size (%)", 10, 100, 100, 10) / 100.0
    with c3:
        cost_pct = st.slider(
            "Transaction Cost (%)", 0.0, 0.5, c["cost"] * 100, 0.05)

# ═══════════════════════════════════════════════════════════
# ② RUN
# ═══════════════════════════════════════════════════════════
bt = backtest_service.run_backtest(
    prices_test, sig, cost_pct / 100.0, strategy, position_size)
m = bt.metrics
bh = (1 + prices_test.pct_change().fillna(0)).cumprod()

top_strip(
    [c["dataset"].upper(), f"STRATEGY {strategy}",
     f"COST {cost_pct/100:.2%}", f"POSITION {position_size:.0%}",
     f"TRADES {bt.trades}"],
    status_text=f"ENGINE · {bt.engine}", status_tone="up",
)

# ═══════════════════════════════════════════════════════════
# ③ KPI
# ═══════════════════════════════════════════════════════════
metric_row([
    dict(label="CAGR",   value=f"{m.get('CAGR', 0.0) * 100:.1f}%"),
    dict(label="SHARPE", value=f"{m.get('Sharpe', 0.0):.2f}"),
    dict(label="MDD",    value=f"{-abs(m.get('MDD', 0.0)) * 100:.1f}%",
         tone="down"),
    dict(label="CALMAR", value=f"{m.get('Calmar', 0.0):.2f}"),
    dict(label="TRADES", value=f"{bt.trades}"),
], cols=5)

# ═══════════════════════════════════════════════════════════
# ④ EQUITY (2fr) : POSITION (1fr)
# ═══════════════════════════════════════════════════════════
col_eq, col_pos = st.columns([2, 1], gap="small")

with col_eq:
    with panel(f"EQUITY — STRATEGY {strategy}", "LIVE", "up"):
        st.plotly_chart(
            charts.equity_curve(
                {"Strategy": bt.equity, "Buy & Hold": bh},
                ""),
            use_container_width=True,
        )

with col_pos:
    with panel("POSITION"):
        st.line_chart(bt.positions.rename("position"), height=300)

# ═══════════════════════════════════════════════════════════
# ⑤ DRAWDOWN
# ═══════════════════════════════════════════════════════════
section_header("DRAWDOWN")
with panel():
    st.plotly_chart(
        charts.drawdown_area(bt.drawdown),
        use_container_width=True,
    )

# ═══════════════════════════════════════════════════════════
# ⑥ COST SENSITIVITY
# ═══════════════════════════════════════════════════════════
section_header("COST SENSITIVITY")
grid = (0.0, 0.0005, 0.0010, 0.0020, 0.0030, 0.0050)
sens = backtest_service.cost_sensitivity(
    prices_test, sig, grid, strategy, position_size)

# break-even 추정 (실패해도 조용히 무시)
be_text = None
try:
    sh = sens["Sharpe"].astype(float)
    costs = sens.index.astype(float) if not pd.api.types.is_numeric_dtype(sens.index) else pd.Series(sens.index, index=sens.index)
    pos_mask = sh > 0
    if pos_mask.all():
        be_text = f"BREAK-EVEN > {max(costs)*100:.2f}%"
    elif not pos_mask.any():
        be_text = "BREAK-EVEN < 0.00%"
    else:
        first_neg_idx = int(np.argmax(~pos_mask.values))
        be_cost = float(costs.iloc[first_neg_idx]) if hasattr(costs, "iloc") else float(costs[first_neg_idx])
        be_text = f"BREAK-EVEN ≈ {be_cost*100:.2f}%"
except Exception:
    pass

with panel(
    "GRID 0.00% – 0.50%",
    status_text=be_text or "",
    status_tone="warn" if be_text else "neutral",
):
    sens_num = [col for col in
                ["CAGR", "Vol", "Sharpe", "Sortino", "MDD",
                 "Calmar", "Sterling"]
                if col in sens.columns]
    st.dataframe(
        styled_df(sens, num_cols=sens_num, precision=3),
        use_container_width=True,
    )
