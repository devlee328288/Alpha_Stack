import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Prediction · AlphaStack", layout="wide")

import theme
from theme import (
    metric_row,
    panel,
    section_header,
    signal_chip,
    top_strip,
)

theme.inject()

import charts
from components import sidebar_controls

sidebar_controls()

# ═══════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════
st.markdown('<div class="as-title">Prediction</div>', unsafe_allow_html=True)

res = st.session_state.get("latest_model_result")
if res is None:
    top_strip(["NO MODEL"], status_text="IDLE", status_tone="neutral")
    st.info("Overview에서 먼저 Run Analysis를 실행하세요.")
    st.stop()

top_strip(
    [f"TEST {len(res.test_index)}d",
     f"LAST {pd.Timestamp(res.test_index[-1]).strftime('%Y-%m-%d')}",
     "PROBA MODEL"],
    status_text="LIVE", status_tone="up",
)

# ── 라벨 순서: -1 / 0 / 1 ─────────────────────────────────
proba = pd.DataFrame(
    res.y_proba,
    index=res.test_index,
    columns=["P(Down)", "P(Neutral)", "P(Up)"],
)
label_map = {-1: "DOWN", 0: "NEUTRAL", 1: "UP"}

# ═══════════════════════════════════════════════════════════
# ① LATEST — hero KPI
# ═══════════════════════════════════════════════════════════
last_proba = proba.iloc[-1]
signal = label_map[int(res.y_pred[-1])]
p_up = float(last_proba["P(Up)"])
confidence = float(last_proba.max())

_tone = {"UP": "up", "DOWN": "down", "NEUTRAL": "neutral"}.get(signal, "neutral")
_conf_label = "HIGH" if confidence >= 0.6 else ("MED" if confidence >= 0.45 else "LOW")

metric_row([
    dict(label="SIGNAL",     value=signal,
         delta=signal_chip(signal, p_up if signal == "UP" else None),
         tone=_tone, variant="hero"),
    dict(label="P(UP)",      value=f"{p_up * 100:.1f}%",
         tone="up", variant="hero"),
    dict(label="CONFIDENCE", value=f"{confidence * 100:.1f}%",
         delta=_conf_label, tone="accent", variant="hero"),
], cols=3)

# ═══════════════════════════════════════════════════════════
# ② PROBABILITY STACK (최근 120일)
# ═══════════════════════════════════════════════════════════
section_header("PROBABILITY STACK · LAST 120D")
with panel("P(DOWN) / P(NEUTRAL) / P(UP)", "STREAM", "accent"):
    st.plotly_chart(
        charts.proba_stack(proba.tail(120)),
        use_container_width=True,
    )

# ═══════════════════════════════════════════════════════════
# ③ FEATURE CONTRIBUTION (마지막 예측)
# ═══════════════════════════════════════════════════════════
fi = res.feature_importance.head(15)
if isinstance(fi, pd.Series):
    fi_df = fi.rename("importance").to_frame().reset_index()
    fi_df.columns = ["feature", "importance"]
else:
    fi_df = fi.copy()
    fi_df.columns = ["feature", "importance"]
fi_df = fi_df.sort_values("importance")

section_header("FEATURE CONTRIBUTION · LATEST")
with panel("TOP 15"):
    colors = [
        "#4ade80" if v > 0 else ("#ff5c5c" if v < 0 else "#9aa0a6")
        for v in fi_df["importance"]
    ]
    fig = go.Figure(go.Bar(
        x=fi_df["importance"],
        y=fi_df["feature"],
        orientation="h",
        marker=dict(color=colors),
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        height=max(260, 20 * len(fi_df)),
        showlegend=False,
        xaxis=dict(title="", zeroline=True, zerolinecolor="rgba(255,255,255,0.09)"),
        yaxis=dict(title=""),
    )
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# ④ RECENT PREDICTIONS (최근 20일)
# ═══════════════════════════════════════════════════════════
section_header("RECENT PREDICTIONS · LAST 20D")
latest = proba.tail(20).copy()
latest["Signal"] = [label_map[int(i)] for i in res.y_pred[-20:]]

with panel():
    st.dataframe(
        theme.styled_df(
            latest.iloc[::-1],
            num_cols=["P(Down)", "P(Neutral)", "P(Up)"],
            precision=3,
        ),
        use_container_width=True,
    )
