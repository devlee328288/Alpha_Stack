import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Backtest · AlphaStack", layout="wide")

import charts
from components import kpi_row, sidebar_controls
from services import backtest_service, data_service
from state import ctx

sidebar_controls()
c = ctx()
st.title("Backtest")

res = st.session_state.get("latest_model_result")
if res is None:
    st.info("Overview에서 먼저 Run Analysis를 실행하세요.")
    st.stop()

for e in backtest_service.engine_status():
    st.warning(f"⚠ backtest engine: {e}")

prices = data_service.load_market_data(c["dataset"], c["start"], c["end"])
prices_test = prices.loc[res.test_index, "close"]
sig = pd.Series(res.y_pred, index=res.test_index)


def _fmt(v, nd=3):
    if v is None:
        return "-"
    try:
        arr = np.asarray(v)
        if arr.size == 0:
            return "-"
        if arr.size == 1:
            return f"{float(arr.reshape(-1)[0]):.{nd}f}"
        return f"{float(np.nanmean(arr.astype(float))):.{nd}f}"
    except Exception:
        return str(v)


# 컨트롤
c1, c2, c3 = st.columns(3)
strategy = c1.selectbox("Strategy", ["A", "B", "C"], index=0,
                        help="A: 단계별 분할매매 · B: 고정 25% · C: 올인/아웃")
position_size = c2.slider("Position Size (%)", 10, 100, 100, 10) / 100.0
cost_pct = c3.slider("Transaction Cost (%)", 0.0, 0.5, c["cost"] * 100, 0.05)

# 실행
bt = backtest_service.run_backtest(
    prices_test, sig, cost_pct / 100.0, strategy, position_size)
m = bt.metrics
bh = (1 + prices_test.pct_change().fillna(0)).cumprod()

st.caption(f"engine: **{bt.engine}** · strategy {bt.strategy} · "
           f"trades {bt.trades} · position {position_size:.0%}")

kpi_row([
    ("CAGR",   f"{m.get('CAGR', 0.0) * 100:.1f}%"),
    ("Sharpe", f"{m.get('Sharpe', 0.0):.2f}"),
    ("MDD",    f"{-abs(m.get('MDD', 0.0)) * 100:.1f}%"),
    ("Calmar", f"{m.get('Calmar', 0.0):.2f}"),
    ("Trades", f"{bt.trades}"),
])

st.plotly_chart(
    charts.equity_curve({"Strategy": bt.equity, "Buy & Hold": bh},
                        f"Equity — Strategy {strategy}"),
    use_container_width=True)

col_a, col_b = st.columns(2)
with col_a:
    st.plotly_chart(charts.drawdown_area(bt.drawdown), use_container_width=True)
with col_b:
    st.markdown("#### Positions")
    st.line_chart(bt.positions.rename("position"), height=320)

st.subheader("Cost sensitivity")
grid = (0.0, 0.0005, 0.0010, 0.0020, 0.0030, 0.0050)
sens = backtest_service.cost_sensitivity(
    prices_test, sig, grid, strategy, position_size)

st.dataframe(
    sens.style.format({
        "CAGR": "{:.3f}", "Vol": "{:.3f}", "Sharpe": "{:.2f}",
        "Sortino": "{:.2f}", "MDD": "{:.3f}", "Calmar": "{:.2f}",
        "Sterling": "{:.2f}", "trades": "{:d}",
    }, na_rep="-"),
    use_container_width=True)
