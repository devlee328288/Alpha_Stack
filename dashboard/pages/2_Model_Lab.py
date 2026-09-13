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

import traceback

from components import sidebar_controls
from services import model_service, comparison_service

sidebar_controls()

# ═══════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════
st.markdown('<div class="as-title">Model Lab</div>', unsafe_allow_html=True)

_err = model_service.engine_status()
if _err:
    top_strip(["REPO ENGINE"], status_text="IMPORT FAIL", status_tone="warn")
    st.error(f"repo engine import 실패: {_err}")
    st.stop()

top_strip(
    ["KOSPI200", "COMBINATION E", "RETURN · 5DAY", "12-FOLD EXPANDING"],
    status_text="READY", status_tone="neutral",
)

# ═══════════════════════════════════════════════════════════
# ① CONFIG
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
        "⚠️ 모델 1개당 5~10분, 4모델 전체 20~40분 소요. "
        "빠른 테스트는 RandomForest 하나만 선택하세요."
    )

# ═══════════════════════════════════════════════════════════
# ② RUN
# ═══════════════════════════════════════════════════════════
if run and selected:
    import contextlib
    import io

    progress = st.progress(0.0)
    status = st.empty()

    def _cb(i, total, name):
        if total > 0:
            progress.progress(min(i / total, 1.0))
        status.caption(f"[{i}/{total}] {name}")

    # ★ repo 코드의 모든 print 를 StringIO 로 흡수 (Streamlit stream 우회)
    _sink = io.StringIO()
    try:
        with contextlib.redirect_stdout(_sink), contextlib.redirect_stderr(_sink):
            results = model_service.run_all_models(
                models=tuple(selected),
                progress_cb=_cb,
            )
        model_service.save_results(results)
        progress.progress(1.0)
        status.caption(f"완료 · {len(results)} 모델")
    except Exception as e:
        st.error(f"실행 실패: {type(e).__name__}: {e}")
        with st.expander("🔍 Traceback"):
            st.code(traceback.format_exc(), language="text")
        with st.expander("🔍 Captured stdout/stderr"):
            st.text(_sink.getvalue()[-3000:] or "(empty)")

# ═══════════════════════════════════════════════════════════
# ③ RESULT
# ═══════════════════════════════════════════════════════════
results = model_service.load_results()
if not results:
    st.caption("모델을 선택하고 RUN을 누르세요. (12폴드 walk-forward, 수 분 소요)")
    st.stop()

kpi = comparison_service.build_summary_kpis(results)
cmp_df = comparison_service.build_comparison_table(results)

# ── SUMMARY ────────────────────────────────────────────────
section_header("SUMMARY")
metric_row([
    dict(label="BEST MODEL",  value=kpi["best_model"], tone="up", accent=True),
    dict(label="HARMONIC",    value=f"{kpi['best_harmonic']:.4f}", tone="up"),
    dict(label="ACCURACY",    value=f"{kpi['best_accuracy']:.4f}"),
    dict(label="MACRO F1",    value=f"{kpi['best_macro_f1']:.4f}"),
    dict(label="DOWN RECALL", value=f"{kpi['best_down_recall']:.4f}"),
    dict(label="ΔSHARPE",     value=f"{kpi['best_delta_sharpe']:+.4f}",
         tone="up" if kpi['best_delta_sharpe'] > 0 else "down"),
], cols=6)

# ── MODEL COMPARISON ───────────────────────────────────────
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

# ── MODEL DETAIL ───────────────────────────────────────────
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

# ── FOLD RESULTS ───────────────────────────────────────────
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

# ── CLASS WEIGHT SELECTION ─────────────────────────────────
wc = pd.DataFrame(r["weight_counts"])
if not wc.empty:
    section_header("CLASS WEIGHT SELECTION")
    with panel():
        show_table(wc, num_cols=["선택 폴드 수"], precision=0)