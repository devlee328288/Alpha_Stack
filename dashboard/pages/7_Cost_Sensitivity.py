import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Cost Sensitivity · AlphaStack", layout="wide")

from components import kpi_row, sidebar_controls
from services import backtest_service, data_service
from state import ctx

sidebar_controls()
c = ctx()
st.title("Cost Sensitivity")

res = st.session_state.get("latest_model_result")
if res is None:
    st.info("Overview에서 먼저 Run Analysis를 실행하세요.")
    st.stop()

prices = data_service.load_market_data(c["dataset"], c["start"], c["end"])
prices_test = prices.loc[res.test_index, "close"]
sig = pd.Series(res.y_pred, index=res.test_index)

# ── 컨트롤 ────────────────────────────────────────────────
c1, c2 = st.columns(2)
strategy = c1.selectbox("Strategy", ["A", "B", "C"], index=0)
position_size = c2.slider("Position Size (%)", 10, 100, 100, 10) / 100.0

grid_text = st.text_input(
    "Costs (%, comma-separated)",
    value="0.00, 0.05, 0.10, 0.20, 0.30, 0.50")

try:
    grid = tuple(float(x.strip()) / 100.0 for x in grid_text.split(",") if x.strip())
except Exception:
    st.error("숫자를 콤마로 구분해서 입력하세요.")
    st.stop()
if not grid:
    st.stop()

# ── 실행 ──────────────────────────────────────────────────
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

kpi_row([
    ("Break-even (Sharpe=0)",
     f"{be_sharpe*100:.3f}%" if be_sharpe is not None else "N/A"),
    ("Break-even (CAGR=0)",
     f"{be_cagr*100:.3f}%" if be_cagr is not None else "N/A"),
    ("Trades @ base cost", f"{sens['trades'].iloc[0] if 'trades' in sens else '-'}"),
])

# ── 테이블 ────────────────────────────────────────────────
show = sens.reset_index().copy()
show["cost"] = (show["cost"] * 100).round(3).astype(str) + "%"
st.dataframe(show, use_container_width=True, hide_index=True)

# ── 차트 ──────────────────────────────────────────────────
cost_pct = [x * 100 for x in sens.index.tolist()]
fig = go.Figure()
for col, color in [("Sharpe", "#7fd1ff"), ("CAGR", "#4ade80"), ("MDD", "#ff6b6b")]:
    if col in sens.columns:
        fig.add_trace(go.Scatter(
            x=cost_pct, y=sens[col], name=col, mode="lines+markers",
            line=dict(color=color, width=2)))
fig.update_layout(
    title="Metrics vs Cost",
    template="plotly_dark", height=380,
    margin=dict(l=10, r=10, t=40, b=10),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(title="Cost (%)", gridcolor="rgba(255,255,255,0.06)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
)
st.plotly_chart(fig, use_container_width=True)

# ── 실 엔진 시도 ──────────────────────────────────────────
st.divider()
try:
    from backtest.run_cost_sensitivity import find_breakeven_cost as _real_be
    st.success("실 엔진 `backtest.run_cost_sensitivity.find_breakeven_cost` import 성공")
    try:
        real_be = _real_be(prices_test.values, sig.values)
        st.metric("Real engine break-even cost",
                  f"{float(real_be)*100:.3f}%" if real_be is not None else "N/A")
    except Exception as e:
        st.caption(f"호출 시그니처 확인 필요: {e}")
except Exception as e:
    st.caption(f"실 엔진 import 실패: {e}")
