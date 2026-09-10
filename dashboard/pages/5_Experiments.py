import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Experiments · AlphaStack", layout="wide")

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
from state import ctx

sidebar_controls()
c = ctx()

# ═══════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════
st.markdown('<div class="as-title">Experiments</div>', unsafe_allow_html=True)

exp_list = st.session_state.get("experiments", [])
top_strip(
    [c["dataset"].upper(),
     f"MODEL {c['model'].upper()}",
     f"FEATURE {c['feature_set'].upper()}"],
    status_text=f"{len(exp_list)} RUN{'S' if len(exp_list) != 1 else ''}",
    status_tone=("up" if exp_list else "neutral"),
)

# ═══════════════════════════════════════════════════════════
# ① ACTIONS
# ═══════════════════════════════════════════════════════════
section_header("ACTIONS")
with panel():
    col_a, col_b = st.columns([1, 4], gap="small")
    with col_a:
        save_clicked = st.button(
            "▶  SAVE CURRENT RUN",
            type="primary",
            use_container_width=True,
        )
    with col_b:
        st.caption(
            "최신 Run Analysis 결과를 실험 로그에 저장합니다. "
            "실행 이력은 세션에 유지됩니다."
        )

if save_clicked:
    res = st.session_state.get("latest_model_result")
    bt = st.session_state.get("latest_backtest")
    if res is None or bt is None:
        st.warning("Overview에서 먼저 Run Analysis를 실행하세요.")
    else:
        row = {
            "id": len(st.session_state.experiments) + 1,
            **ctx(),
            "accuracy": round(res.accuracy, 4),
            "mcc": round(res.mcc, 4),
            "sharpe": round(bt.metrics["Sharpe"], 4),
            "mdd": round(bt.metrics["MDD"], 4),
        }
        st.session_state.experiments.append(row)
        st.success(f"Saved run #{row['id']}")

# ═══════════════════════════════════════════════════════════
# ② EXPERIMENT LOG
# ═══════════════════════════════════════════════════════════
section_header("EXPERIMENT LOG")
if st.session_state.experiments:
    df = pd.DataFrame(st.session_state.experiments)

    # KPI — 가장 최근 실험 요약
    latest_row = df.iloc[-1]
    metric_row([
        dict(label="TOTAL RUNS", value=str(len(df))),
        dict(label="LATEST ACC", value=f"{latest_row['accuracy']:.4f}",
             tone="up" if latest_row["accuracy"] > 0.5 else "down"),
        dict(label="LATEST MCC", value=f"{latest_row['mcc']:.4f}",
             tone="up" if latest_row["mcc"] > 0 else "down"),
        dict(label="LATEST SHARPE", value=f"{latest_row['sharpe']:.4f}",
             tone="up" if latest_row["sharpe"] > 0 else "down"),
    ], cols=4)

    # 표
    with panel(f"{len(df)} RUN{'S' if len(df) != 1 else ''}"):
        # 숫자 컬럼만 골라내기
        num_cols = [col for col in df.columns
                    if col in ("accuracy", "mcc", "sharpe", "mdd")]
        st.dataframe(
            styled_df(df, num_cols=num_cols, precision=4),
            use_container_width=True,
            hide_index=True,
        )
else:
    with panel("EMPTY"):
        st.markdown(
            '<div style="padding:16px 4px;color:var(--text-secondary);'
            'font-size:12px;text-align:center;">'
            '저장된 실험이 아직 없습니다. 위 버튼으로 첫 실행을 기록하세요.'
            '</div>',
            unsafe_allow_html=True,
        )
