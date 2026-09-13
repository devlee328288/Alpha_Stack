import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Risk · AlphaStack", layout="wide")

import theme
from theme import metric_row, section_header, top_strip, panel, show_table
theme.inject()

from components import sidebar_controls
from services import backtest_service, risk_service

sidebar_controls()

st.markdown('<div class="as-title">Risk</div>', unsafe_allow_html=True)

_err = risk_service.engine_status()
if _err and not risk_service.risk_available():
    top_strip(["RISK"], status_text="IMPORT FAIL", status_tone="warn")
    st.error(_err)
    st.stop()

top_strip(
    ["KOSPI200", "evaluation_risk · evaluation_backtest", "DSR · STERLING"],
    status_text="READY", status_tone="neutral",
)

# ═══════════════════════════════════════════════════════════
# SOURCE SELECT
# ═══════════════════════════════════════════════════════════
section_header("SOURCE")

bt = st.session_state.get("_bt_results")

source = st.radio(
    "Returns source",
    ["Backtest (A/B/C)", "Model Lab best model"],
    horizontal=True,
)

returns = None
source_label = ""

if source == "Backtest (A/B/C)":
    if not bt:
        st.warning("Backtest 페이지에서 먼저 RUN ALL을 실행하세요.")
        st.stop()
    pick = st.selectbox("Strategy", list(bt.keys()), index=0)
    returns = bt[pick]["daily_returns"].to_numpy()
    source_label = f"Backtest · Strategy {pick}"
else:
    from services import model_service
    models = model_service.load_results()
    if not models:
        st.warning("Model Lab 페이지에서 먼저 RUN을 실행하세요.")
        st.stop()
    pick = st.selectbox("Model", list(models.keys()), index=0)
    fold = pd.DataFrame(models[pick]["fold_results"])
    # fold별 sharpe를 만들 수 없으니 daily returns 대용으로 fold 지표를 사용할 수 없음
    # → backtest 엔진을 각 모델의 OOS 구간에 대해 돌리는 게 정석이지만 지금은 fold sharpe로 대체
    st.info("⚠️ 모델별 daily returns를 저장하지 않아 Backtest A/B/C로만 리스크 계산 가능합니다.")
    st.stop()

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════
section_header("CONFIG")
with panel():
    c1, c2, c3 = st.columns(3, gap="small")
    with c1:
        rf = st.number_input("Risk-free rate (annual)", 0.0, 0.1, 0.02, 0.005, format="%.4f")
    with c2:
        n_trials = st.number_input("N trials (DSR)", 1, 1000, 50, 1)
    with c3:
        top_k = st.number_input("Sterling top-K MDD", 1, 10, 3, 1)

# ═══════════════════════════════════════════════════════════
# CALC
# ═══════════════════════════════════════════════════════════
section_header(f"RISK METRICS · {source_label}")

try:
    m = risk_service.calculate_risk(
        returns,
        risk_free_rate=float(rf),
        n_trials=int(n_trials),
        sterling_top_k=int(top_k),
    )
except Exception as e:
    st.error(f"계산 실패: {type(e).__name__}: {e}")
    st.stop()

metric_row([
    dict(label="MDD",
         value=f"{-abs(m.get('MDD') or 0)*100:.2f}%", tone="down"),
    dict(label="SHARPE",
         value=f"{m.get('Sharpe Ratio') or 0:.4f}",
         tone="up" if (m.get("Sharpe Ratio") or 0) > 0 else "down"),
    dict(label="SORTINO",
         value=f"{m.get('Sortino Ratio') or 0:.4f}",
         tone="up" if (m.get("Sortino Ratio") or 0) > 0 else "down"),
    dict(label="CALMAR",
         value=f"{m.get('Calmar Ratio') or 0:.4f}"),
    dict(label="STERLING",
         value=f"{m.get('Sterling Ratio') or 0:.4f}"),
    dict(label="DSR",
         value=f"{(m.get('Deflated Sharpe Ratio') or 0)*100:.2f}%"),
], cols=6)

# ═══════════════════════════════════════════════════════════
# DETAIL TABLE
# ═══════════════════════════════════════════════════════════
section_header("DETAIL")
with panel():
    det = pd.DataFrame([{
        "METRIC": k,
        "VALUE": v,
    } for k, v in m.items()])
    show_table(det, num_cols=["VALUE"], precision=4)

# ═══════════════════════════════════════════════════════════
# DISTRIBUTION
# ═══════════════════════════════════════════════════════════
section_header("RETURN DISTRIBUTION")
with panel("DAILY RETURNS"):
    import plotly.graph_objects as go
    fig = go.Figure(go.Histogram(
        x=returns, nbinsx=50,
        marker_color="#7fd1ff",
        opacity=0.75,
    ))
    fig.update_layout(height=320, xaxis=dict(title="Daily Return"),
                      yaxis=dict(title="Count"), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# FOLD STATS (Backtest A/B/C만)
# ═══════════════════════════════════════════════════════════
if source == "Backtest (A/B/C)":
    section_header("ROLLING 60D")
    roll = pd.Series(returns).rolling(60).mean() / pd.Series(returns).rolling(60).std() * np.sqrt(252)
    with panel("ROLLING SHARPE · 60D"):
        import plotly.graph_objects as go
        fig = go.Figure(go.Scatter(
            y=roll.values, mode="lines",
            line=dict(color="#4ade80", width=1.4),
        ))
        fig.update_layout(height=280, yaxis=dict(title="Sharpe (60D)"))
        st.plotly_chart(fig, use_container_width=True)