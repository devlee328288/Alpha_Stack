import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Risk & Evaluation · AlphaStack", layout="wide")

import charts
from components import kpi_row, sidebar_controls
from services import backtest_service, data_service, evaluation_service
from state import ctx

sidebar_controls()
c = ctx()
st.title("Risk & Evaluation")

res = st.session_state.get("latest_model_result")
if res is None:
    st.info("Overview에서 먼저 Run Analysis를 실행하세요.")
    st.stop()

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


tab1, tab2, tab3 = st.tabs(["Classification", "Risk", "Walk Forward"])

# ── Classification ────────────────────────────────────────
with tab1:
    m = evaluation_service.classification_metrics(res.y_true, res.y_pred)
    kpi_row([(k, _fmt(v)) for k, v in list(m.items())[:4]])

    st.markdown("#### Class Distribution")
    cd = evaluation_service.class_distribution(res.y_true, res.y_pred)

    # 테이블
    show = cd.copy()
    show["actual_share"] = show["actual_share"].apply(lambda x: f"{x*100:.1f}%")
    if "pred_share" in show.columns:
        show["pred_share"] = show["pred_share"].apply(lambda x: f"{x*100:.1f}%")
    st.dataframe(show, use_container_width=True)

    # 차트
    st.plotly_chart(
        evaluation_service.class_distribution_chart(cd),
        use_container_width=True,
    )

    st.markdown("#### Confusion Matrix")
    cm = evaluation_service.confusion(res.y_true, res.y_pred)
    st.plotly_chart(charts.confusion_heatmap(cm), use_container_width=True)
    st.dataframe(cm, use_container_width=True)

# ── Risk ──────────────────────────────────────────────────
with tab2:
    bt = backtest_service.run_backtest(prices_test, sig, c["cost"])
    m = bt.metrics

    items = [
        ("CAGR",      f"{m.get('CAGR', 0.0) * 100:.2f}%"),
        ("Vol",       f"{m.get('Vol', 0.0) * 100:.2f}%"),
        ("Sharpe",    f"{m.get('Sharpe', 0.0):.3f}"),
        ("Sortino",   f"{m.get('Sortino', 0.0):.3f}"),
        ("MDD",       f"{-abs(m.get('MDD', 0.0)) * 100:.2f}%"),
        ("Calmar",    f"{m.get('Calmar', 0.0):.3f}"),
        ("Sterling",  f"{m.get('Sterling', 0.0):.3f}"),
        ("Trades",    f"{bt.trades}"),
    ]
    kpi_row(items)
    st.plotly_chart(charts.drawdown_area(bt.drawdown), use_container_width=True)

    st.markdown("**All metrics (raw)**")
    st.dataframe(
        pd.DataFrame([{k: _fmt(v) for k, v in m.items()}]),
        use_container_width=True)

# ── Walk Forward ──────────────────────────────────────────
with tab3:
    wf = evaluation_service.walk_forward(prices_test, sig, c["cost"], n_folds=6)
    k1, k2, k3 = st.columns(3)
    k1.metric("Mean Sharpe", f"{wf.mean_sharpe:.2f}")
    k2.metric("Std Sharpe", f"{wf.std_sharpe:.2f}")
    k3.metric("Positive Folds", f"{wf.positive_folds}/{wf.total_folds}")
    st.dataframe(wf.folds, use_container_width=True)
