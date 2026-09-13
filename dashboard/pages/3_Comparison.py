import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Comparison · AlphaStack", layout="wide")

import theme
from theme import (
    metric_row, section_header, top_strip, panel, show_table, plotly_chart,
)
theme.inject()

from components import sidebar_controls
from services import baseline_service, model_service, comparison_service

sidebar_controls()

# ═══════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════
st.markdown('<div class="as-title">Comparison</div>', unsafe_allow_html=True)

baseline = baseline_service.load_results()
models = model_service.load_results()

n_models = len(models) if models else 0
parts = ["KOSPI200"]
if baseline:
    parts.append(f"BASELINE {baseline['threshold']*100:.0f}%")
if models:
    parts.append(f"{n_models} MODELS")

_ready = bool(baseline and models)
top_strip(parts, status_text="READY" if _ready else "INCOMPLETE",
          status_tone="up" if _ready else "warn")

# ═══════════════════════════════════════════════════════════
# 사전 조건 체크
# ═══════════════════════════════════════════════════════════
if not baseline and not models:
    st.info("**Baseline** 과 **Model Lab** 페이지에서 각각 RUN을 먼저 실행하세요.")
    st.stop()

c1, c2 = st.columns(2, gap="small")
with c1:
    if baseline:
        st.success(f"✅ Baseline (threshold={baseline['threshold']*100:.0f}%, "
                   f"folds={baseline['total_folds']})")
    else:
        st.warning("⚠️ Baseline 미실행 → **Baseline** 페이지에서 RUN")
with c2:
    if models:
        st.success(f"✅ Model Lab ({len(models)} models)")
    else:
        st.warning("⚠️ Model Lab 미실행 → **Model Lab** 페이지에서 RUN")

if not _ready:
    st.stop()

# ═══════════════════════════════════════════════════════════
# TOP LINE · BASELINE vs BEST MODEL
# ═══════════════════════════════════════════════════════════
best_name = comparison_service.best_model(models)
best_summary = models[best_name]["summary"]

b_perf = baseline.get("perf_metrics", {})
b_cls = baseline.get("cls_metrics", {})

section_header("TOP LINE · BASELINE vs BEST MODEL")

col_b, col_m = st.columns(2, gap="small")

with col_b:
    with panel(
        f"BASELINE · 6-PARAM · {baseline['threshold']*100:.0f}%",
        status_text=f"{baseline['total_folds']} FOLDS",
        status_tone="accent",
    ):
        metric_row([
            dict(label="SHARPE", value=f"{b_perf.get('sharpe', 0):.4f}",
                 tone="up" if b_perf.get("sharpe", 0) > 0 else "down"),
            dict(label="CAGR", value=f"{b_perf.get('cagr', 0)*100:.2f}%",
                 tone="up" if b_perf.get("cagr", 0) > 0 else "down"),
            dict(label="MDD", value=f"{-abs(b_perf.get('mdd', 0))*100:.2f}%",
                 tone="down"),
            dict(label="WIN RATE", value=f"{b_perf.get('win_rate', 0)*100:.2f}%"),
        ], cols=4)

with col_m:
    with panel(
        f"BEST MODEL · {best_name}",
        status_text="BEST", status_tone="up",
    ):
        _ds = best_summary.get("delta_sharpe_net_median", 0)
        metric_row([
            dict(label="ACC", value=f"{best_summary['accuracy']:.4f}"),
            dict(label="MACRO F1", value=f"{best_summary['macro_f1']:.4f}"),
            dict(label="DOWN REC", value=f"{best_summary['down_recall']:.4f}"),
            dict(label="ΔSHARPE", value=f"{_ds:+.4f}",
                 tone="up" if _ds > 0 else "down"),
        ], cols=4)

# ═══════════════════════════════════════════════════════════
# STRATEGY TABLE
# ═══════════════════════════════════════════════════════════
cross = comparison_service.build_cross_table(baseline, models)

section_header("STRATEGY · OOS PERFORMANCE")
with panel("SHARPE · CAGR · MDD · WIN RATE · ΔSHARPE"):
    show_table(
        cross["strategy"],
        num_cols=["SHARPE", "CAGR", "MDD", "WIN RATE", "ΔSHARPE"],
        precision=4,
    )

st.caption(
    "⚠️ BASELINE: 전체 OOS pooled Sharpe ·  "
    "MODEL: fold별 ΔSharpe 중앙값 (전략 Sharpe − Buy&Hold Sharpe). 정의가 다름."
)

# ═══════════════════════════════════════════════════════════
# CLASSIFICATION TABLE
# ═══════════════════════════════════════════════════════════
section_header("CLASSIFICATION · OOS")
with panel("ACCURACY · MACRO F1 · BALANCED ACC · DOWN RECALL"):
    show_table(
        cross["classification"],
        num_cols=["ACC", "MACRO F1", "BAL ACC", "DOWN REC", "MAJORITY"],
        precision=4,
    )

st.caption("⚠️ BASELINE: step5가 반환하는 f1_macro · balanced_acc만 존재.")

# ═══════════════════════════════════════════════════════════
# MODEL RANKING · HARMONIC
# ═══════════════════════════════════════════════════════════
section_header("MODEL RANKING · HARMONIC MEAN")
with panel("ACC · MACRO F1 · DOWN RECALL 조화평균"):
    cmp_df = comparison_service.build_comparison_table(models)
    colors = ["#4ade80" if i == 0 else "#7fd1ff" for i in range(len(cmp_df))]

    fig = go.Figure(go.Bar(
        x=cmp_df["core_harmonic_mean"],
        y=cmp_df["model"],
        orientation="h",
        marker=dict(color=colors),
        text=[f"{v:.4f}" for v in cmp_df["core_harmonic_mean"]],
        textposition="outside",
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        height=max(200, 55 * len(cmp_df)),
        showlegend=False,
        xaxis=dict(title="",
                   range=[0, float(cmp_df["core_harmonic_mean"].max()) * 1.18]),
        yaxis=dict(title="", autorange="reversed"),
    )
    plotly_chart(fig)

# ═══════════════════════════════════════════════════════════
# FULL DETAIL TABLE
# ═══════════════════════════════════════════════════════════
section_header("FULL DETAIL · 4 MODELS")
with panel(f"{len(models)} MODELS · SORT BY HARMONIC",
           status_text=f"BEST · {best_name}", status_tone="up"):
    display = cmp_df.rename(columns=comparison_service.COMPARISON_COLUMNS)
    show_table(
        display,
        num_cols=["ACC", "MACRO F1", "DOWN RECALL", "HARMONIC",
                  "BAL ACC", "MAJORITY", "ΔSHARPE"],
        precision=4,
        highlight_row=0,
    )