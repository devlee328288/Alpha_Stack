import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Model Lab · AlphaStack", layout="wide")

import theme
from theme import (
    metric_row, section_header, top_strip, panel, show_table,
)
theme.inject()

from components import sidebar_controls
from services import model_service, comparison_service
from state import ctx, get_result, set_result

sidebar_controls()
c = ctx()
scope = c["scope"]
ticker = c["ticker"]

# ── UNIVERSE scope 는 전용 페이지에서 ──
if scope == "UNIVERSE":
    st.markdown('<div class="as-title">전용 페이지로 이동</div>', unsafe_allow_html=True)
    st.info(
        "**UNIVERSE scope 는 Universe 페이지에서만 사용합니다.**  \n"
        "사이드바에서 **UNIVERSE** 버튼을 다시 누르거나, "
        "페이지 목록의 **Universe** 를 여세요."
    )
    if st.button("▶  Universe 페이지로", type="primary"):
        # 사이드바 페이지 이름이 파일명에 따라 다를 수 있음 — 안내만
        st.switch_page("pages/8_Universe.py")
    st.stop()

# ═══════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════
st.markdown('<div class="as-title">Model Lab</div>', unsafe_allow_html=True)

_err = model_service.engine_status()
if _err:
    top_strip([scope, ticker], status_text="IMPORT FAIL", status_tone="warn")
    st.error(f"repo engine import 실패: {_err}")
    st.stop()

top_strip(
    [scope, ticker, "COMBINATION E", "RETURN · 5DAY", "12-FOLD EXPANDING"],
    status_text="READY", status_tone="neutral",
)

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════
section_header("CONFIG")
with panel():
    c1, c2 = st.columns([3, 1], gap="small")
    with c1:
        selected = st.multiselect(
            "Models", list(model_service.MODELS),
            default=list(model_service.MODELS),
        )
    with c2:
        st.write("")
        run = st.button("▶  RUN", type="primary", use_container_width=True)

    st.caption(
        "⚠️ 모델 1개당 1~10분 (종목/데이터 크기에 따라), "
        "4모델 전체는 오래 걸립니다. 빠른 테스트는 RandomForest 하나만."
    )

# ═══════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════
if run and selected:
    import traceback
    progress = st.progress(0.0)
    status = st.empty()

    def _cb(i, total, name):
        if total > 0:
            progress.progress(min(i / total, 1.0))
        status.caption(f"[{i}/{total}] {name}")

    try:
        results = model_service.run_all_models(
            scope=scope,
            ticker=ticker,
            models=tuple(selected),
            progress_cb=_cb,
        )
        set_result("model_lab", scope, ticker, results)
        progress.progress(1.0)
        status.caption(f"완료 · {len(results)} 모델")
    except Exception as e:
        st.error(f"실행 실패: {type(e).__name__}: {e}")
        with st.expander("Traceback", expanded=True):
            st.code(traceback.format_exc())

# ═══════════════════════════════════════════════════════════
# RESULT
# ═══════════════════════════════════════════════════════════
results = get_result("model_lab", scope, ticker)
if not results:
    st.caption(f"{scope}:{ticker} 에 대한 결과가 아직 없습니다.")
    st.stop()

kpi = comparison_service.build_summary_kpis(results)
cmp_df = comparison_service.build_comparison_table(results)

section_header(f"SUMMARY · {scope}:{ticker}")
metric_row([
    dict(label="BEST MODEL",  value=kpi["best_model"], tone="up", accent=True),
    dict(label="HARMONIC",    value=f"{kpi['best_harmonic']:.4f}", tone="up"),
    dict(label="ACCURACY",    value=f"{kpi['best_accuracy']:.4f}"),
    dict(label="MACRO F1",    value=f"{kpi['best_macro_f1']:.4f}"),
    dict(label="DOWN RECALL", value=f"{kpi['best_down_recall']:.4f}"),
    dict(label="ΔSHARPE",     value=f"{kpi['best_delta_sharpe']:+.4f}",
         tone="up" if kpi['best_delta_sharpe'] > 0 else "down"),
], cols=6)

section_header(f"MODEL COMPARISON · {len(cmp_df)} MODELS")
with panel(
    "12-FOLD OOS · SORT BY HARMONIC",
    status_text=f"BEST · {kpi['best_model']}",
    status_tone="up",
):
    display = cmp_df.rename(columns=comparison_service.COMPARISON_COLUMNS)
    show_table(
        display,
        num_cols=["ACC", "MACRO F1", "DOWN RECALL", "HARMONIC",
                  "BAL ACC", "MAJORITY", "ΔSHARPE"],
        precision=4,
        highlight_row=0,
    )

# MODEL DETAIL
section_header("MODEL DETAIL")
pick = st.selectbox("Inspect model", list(results.keys()), index=0)
r = results[pick]
s = r["summary"]

metric_row([
    dict(label="OOS ROWS",   value=f"{s['oos_rows']}"),
    dict(label="ACC",        value=f"{s['accuracy']:.4f}"),
    dict(label="MACRO F1",   value=f"{s['macro_f1']:.4f}"),
    dict(label="DOWN REC",   value=f"{s['down_recall']:.4f}"),
    dict(label="NEUTRAL REC", value=f"{s['neutral_recall']:.4f}"),
    dict(label="UP REC",     value=f"{s['up_recall']:.4f}"),
], cols=6)

fold_df = pd.DataFrame(r["fold_results"])
if not fold_df.empty:
    section_header("FOLD RESULTS")
    with panel(f"{len(fold_df)} FOLDS"):
        show_table(
            fold_df,
            num_cols=["accuracy", "macro_f1", "down_recall",
                      "core_harmonic_mean", "delta_sharpe_net",
                      "strategy_sharpe_net", "buyhold_sharpe_net"],
            precision=4,
        )

wc = pd.DataFrame(r["weight_counts"])
if not wc.empty:
    section_header("CLASS WEIGHT SELECTION")
    with panel():
        show_table(wc, num_cols=["선택 폴드 수"], precision=0)