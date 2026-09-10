import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Risk & Evaluation · AlphaStack", layout="wide")

import theme
from theme import (
    metric_grid,
    metric_row,
    panel,
    section_header,
    top_strip,
)

theme.inject()

import charts
from components import sidebar_controls
from services import backtest_service, data_service, evaluation_service
from state import ctx

sidebar_controls()
c = ctx()

# ═══════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════
st.markdown('<div class="as-title">Risk & Evaluation</div>', unsafe_allow_html=True)

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
     f"COST {c['cost']:.2%}",
     f"TEST {len(res.test_index)}d"],
    status_text="LIVE", status_tone="up",
)


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


# ═══════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["CLASSIFICATION", "RISK", "WALK FORWARD"])

# ── Classification ────────────────────────────────────────
with tab1:
    m = evaluation_service.classification_metrics(res.y_true, res.y_pred)
    pairs = list(m.items())[:4]

    metric_row([
        dict(label=k.upper().replace("_", " "),
             value=_fmt(v, 3),
             tone="neutral")
        for k, v in pairs
    ], cols=4)

    # ── Class Distribution ────────────────────────────────
    section_header("CLASS DISTRIBUTION")
    cd = evaluation_service.class_distribution(res.y_true, res.y_pred)

    show = cd.copy()
    show["actual_share"] = show["actual_share"].apply(lambda x: f"{x*100:.1f}%")
    if "pred_share" in show.columns:
        show["pred_share"] = show["pred_share"].apply(lambda x: f"{x*100:.1f}%")

    col_t, col_c = st.columns([1, 2], gap="small")
    with col_t:
        with panel("SHARES"):
            st.dataframe(show, use_container_width=True)
    with col_c:
        with panel("ACTUAL vs PREDICTED"):
            st.plotly_chart(
                evaluation_service.class_distribution_chart(cd),
                use_container_width=True,
            )

    # ── Confusion Matrix ──────────────────────────────────
    section_header("CONFUSION MATRIX")
    cm = evaluation_service.confusion(res.y_true, res.y_pred)

    col_h, col_t = st.columns([2, 1], gap="small")
    with col_h:
        with panel("HEATMAP"):
            st.plotly_chart(
                charts.confusion_heatmap(cm),
                use_container_width=True,
            )
    with col_t:
        with panel("COUNTS"):
            st.dataframe(cm, use_container_width=True)

# ── Risk ──────────────────────────────────────────────────
with tab2:
    bt = backtest_service.run_backtest(prices_test, sig, c["cost"])
    m = bt.metrics

    items = [
        dict(label="CAGR",     value=f"{m.get('CAGR', 0.0) * 100:.2f}%",
             tone="neutral"),
        dict(label="VOL",      value=f"{m.get('Vol', 0.0) * 100:.2f}%",
             tone="neutral"),
        dict(label="SHARPE",   value=f"{m.get('Sharpe', 0.0):.3f}",
             tone="up" if m.get("Sharpe", 0) > 0 else "down"),
        dict(label="SORTINO",  value=f"{m.get('Sortino', 0.0):.3f}",
             tone="up" if m.get("Sortino", 0) > 0 else "down"),
        dict(label="MDD",      value=f"{-abs(m.get('MDD', 0.0)) * 100:.2f}%",
             tone="down"),
        dict(label="CALMAR",   value=f"{m.get('Calmar', 0.0):.3f}",
             tone="up" if m.get("Calmar", 0) > 0 else "down"),
        dict(label="STERLING", value=f"{m.get('Sterling', 0.0):.3f}",
             tone="neutral"),
        dict(label="TRADES",   value=f"{bt.trades}",
             tone="neutral"),
    ]
    metric_grid(items, cols=4)

    section_header("DRAWDOWN")
    with panel("UNDERWATER"):
        st.plotly_chart(
            charts.drawdown_area(bt.drawdown),
            use_container_width=True,
        )

    section_header("ALL METRICS · RAW")
    with panel():
        st.dataframe(
            pd.DataFrame([{k: _fmt(v) for k, v in m.items()}]),
            use_container_width=True,
        )

# ── Walk Forward ──────────────────────────────────────────
with tab3:
    wf = evaluation_service.walk_forward(
        prices_test, sig, c["cost"], n_folds=6)

    pos_ratio = wf.positive_folds / max(wf.total_folds, 1)

    metric_row([
        dict(label="MEAN SHARPE",  value=f"{wf.mean_sharpe:.2f}",
             tone="up" if wf.mean_sharpe > 0 else "down"),
        dict(label="STD SHARPE",   value=f"{wf.std_sharpe:.2f}",
             tone="neutral"),
        dict(label="POSITIVE FOLDS",
             value=f"{wf.positive_folds}/{wf.total_folds}",
             delta=f"{pos_ratio * 100:.0f}%",
             tone="up" if pos_ratio >= 0.5 else "down"),
    ], cols=3)

    section_header("FOLD TABLE")
    with panel(f"{wf.total_folds} FOLDS"):
        st.dataframe(wf.folds, use_container_width=True)
