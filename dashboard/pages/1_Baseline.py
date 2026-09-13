import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
st.set_page_config(page_title="Baseline · AlphaStack", layout="wide")

# ═══════════════════════════════════════════════════════════
# 진단용 import (실패 시 어디서 죽는지 표시)
# ═══════════════════════════════════════════════════════════
try:
    import theme
    from theme import (
        metric_row, section_header, top_strip, panel, show_table,
    )
except Exception as e:
    import traceback
    st.error(f"❌ theme import 실패: {type(e).__name__}: {e}")
    st.code(traceback.format_exc())
    st.stop()

theme.inject()

try:
    from components import sidebar_controls
    from services import baseline_service
except Exception as e:
    import traceback
    st.error(f"❌ components/services import 실패: {type(e).__name__}: {e}")
    st.code(traceback.format_exc())
    st.stop()

try:
    sidebar_controls()
except Exception as e:
    import traceback
    st.error(f"❌ sidebar_controls 실패: {type(e).__name__}: {e}")
    st.code(traceback.format_exc())
    st.stop()

# ═══════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════
st.markdown('<div class="as-title">Baseline Classifier</div>', unsafe_allow_html=True)

_err = baseline_service.engine_status()
if _err:
    top_strip(["6-PARAM THRESHOLD"], status_text="IMPORT FAIL", status_tone="warn")
    st.error(f"repo engine import 실패: {_err}")
    st.stop()

top_strip(
    ["KOSPI200", "6-PARAM THRESHOLD", "12-FOLD EXPANDING", "ADR-AS-0002"],
    status_text="READY", status_tone="neutral",
)

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════
section_header("CONFIG")
with panel():
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1, 1.4], gap="small")
    with c1:
        threshold = st.selectbox(
            "Threshold", [0.01, 0.02],
            format_func=lambda v: f"{v*100:.0f}%",
            index=0,
        )
    with c2:
        mode = st.selectbox(
            "CMA-ES mode", ["QUICK (30)", "FULL (300)"], index=0,
        )
        max_evals = 30 if mode.startswith("QUICK") else 300
    with c3:
        st.write("")
        include_focal = st.checkbox(
            "Focal MLP", value=False,
            disabled=not baseline_service.focal_available(),
            help="step7 Focal walk-forward (torch, 수 분 소요)",
        )
    with c4:
        st.write("")
        run = st.button("▶  RUN BASELINE", type="primary",
                        use_container_width=True)

# ═══════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════
if run:
    import traceback
    with st.spinner("12-fold expanding walk-forward (CMA-ES)…"):
        try:
            results = baseline_service.run_baseline(
                threshold=threshold,
                max_evals=max_evals,
                include_focal=include_focal,
            )
            baseline_service.save_results(results)
            st.success("✅ 실행 완료")
        except Exception as e:
            st.error(f"실행 실패: {type(e).__name__}: {e}")
            with st.expander("🔍 Traceback", expanded=True):
                st.code(traceback.format_exc(), language="text")

# ═══════════════════════════════════════════════════════════
# RESULT
# ═══════════════════════════════════════════════════════════
results = baseline_service.load_results()
if not results:
    st.caption("Threshold와 mode를 선택하고 RUN BASELINE을 누르세요.")
    st.stop()

s = results
perf = s["perf_metrics"]
cls = s["cls_metrics"]

# ── SUMMARY ────────────────────────────────────────────────
section_header(f"SUMMARY · THRESHOLD {s['threshold']*100:.0f}%")
metric_row([
    dict(label="FOLDS",       value=f"{s['total_folds']}"),
    dict(label="SHARPE",      value=f"{perf.get('sharpe', 0):.4f}",
         tone="up" if perf.get("sharpe", 0) > 0 else "down"),
    dict(label="CAGR",        value=f"{perf.get('cagr', 0)*100:.2f}%",
         tone="up" if perf.get("cagr", 0) > 0 else "down"),
    dict(label="MDD",         value=f"{-abs(perf.get('mdd', 0))*100:.2f}%",
         tone="down"),
    dict(label="WIN RATE",    value=f"{perf.get('win_rate', 0)*100:.2f}%"),
    dict(label="MACRO F1",    value=f"{cls.get('f1_macro', 0):.4f}"),
], cols=6)

metric_row([
    dict(label="CALMAR",       value=f"{perf.get('calmar', 0):.4f}"),
    dict(label="PROFIT FCT",   value=f"{perf.get('profit_factor', 0):.4f}"),
    dict(label="BALANCED ACC", value=f"{cls.get('balanced_acc', 0):.4f}"),
    dict(label="RATIO UP",     value=f"{cls.get('ratio_up', 0)*100:.2f}%"),
    dict(label="RATIO NEUT",   value=f"{cls.get('ratio_neutral', 0)*100:.2f}%"),
    dict(label="RATIO DOWN",   value=f"{cls.get('ratio_down', 0)*100:.2f}%"),
], cols=6)

# ── MEDIAN 6-PARAMS ────────────────────────────────────────
section_header("MEDIAN 6-PARAMETERS")
pm = s["params_median"]
metric_row([
    dict(label="α UP",          value=f"{pm.get('alpha_up', 0):.4f}"),
    dict(label="α DOWN",        value=f"{pm.get('alpha_down', 0):.4f}"),
    dict(label="β UP",          value=f"{pm.get('beta_up', 0):.4f}"),
    dict(label="β DOWN",        value=f"{pm.get('beta_down', 0):.4f}"),
    dict(label="VOL PERIOD",    value=f"{pm.get('vol_period', 0):.0f}"),
    dict(label="VOLUME PERIOD", value=f"{pm.get('volume_period', 0):.0f}"),
], cols=6)

# ── FOLD DETAILS ───────────────────────────────────────────
import pandas as pd
fold_df = pd.DataFrame(s["fold_details"])
if not fold_df.empty:
    section_header(f"FOLD DETAILS · {len(fold_df)} FOLDS")
    with panel():
        drop_cols = [c for c in ["train_start", "train_end"]
                     if c in fold_df.columns]
        view = fold_df.drop(columns=drop_cols)

        num_cols = [
            "alpha_up", "alpha_down", "beta_up", "beta_down",
            "vol_period", "volume_period", "is_fitness", "oos_ret_mean",
        ]
        num_cols = [c for c in num_cols if c in view.columns]
        show_table(view, num_cols=num_cols, precision=4)

# ── FOCAL (옵션) ───────────────────────────────────────────
if "focal" in s:
    focal = s["focal"]
    section_header("FOCAL MLP · STEP 7")
    metric_row([
        dict(label="SHARPE",    value=f"{focal.get('sharpe', 0):.4f}",
             tone="up" if focal.get("sharpe", 0) > 0 else "down"),
        dict(label="CAGR",      value=f"{focal.get('cagr', 0)*100:.2f}%",
             tone="up" if focal.get("cagr", 0) > 0 else "down"),
        dict(label="MDD",       value=f"{-abs(focal.get('mdd', 0))*100:.2f}%",
             tone="down"),
        dict(label="WIN RATE",  value=f"{focal.get('win_rate', 0)*100:.2f}%"),
        dict(label="ACC",       value=f"{focal.get('accuracy', 0):.4f}"),
        dict(label="MACRO F1",  value=f"{focal.get('f1_macro', 0):.4f}"),
    ], cols=6)

elif "focal_error" in s:
    st.warning(f"Focal MLP 실행 실패: {s['focal_error']}")