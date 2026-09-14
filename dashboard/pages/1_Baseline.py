import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Baseline · AlphaStack", layout="wide")

import theme
from theme import (
    metric_row,
    section_header,
    top_strip,
    panel,
    show_table,
)

theme.inject()

from components import sidebar_controls
from services import baseline_service, data_loader
from state import ctx, get_result, set_result

sidebar_controls()
c = ctx()
scope = c["scope"]
ticker = c["ticker"]

# ── UNIVERSE scope 는 전용 페이지에서 ──
if scope == "UNIVERSE":
    st.markdown(
        '<div class="as-title">전용 페이지로 이동</div>', unsafe_allow_html=True
    )
    st.info(
        "**UNIVERSE scope 는 Universe 페이지에서만 사용합니다.**  \n"
        "사이드바에서 **UNIVERSE** 버튼을 다시 누르거나, "
        "페이지 목록의 **Universe** 를 여세요."
    )
    if st.button("▶  Universe 페이지로", type="primary"):
        st.switch_page("pages/8_Universe.py")
    st.stop()

# ═══════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════
st.markdown('<div class="as-title">Baseline Classifier</div>', unsafe_allow_html=True)

_err = baseline_service.engine_status()
if _err:
    top_strip([scope, ticker], status_text="IMPORT FAIL", status_tone="warn")
    st.error(f"repo engine import 실패: {_err}")
    st.stop()

top_strip(
    [scope, ticker, "6-PARAM THRESHOLD", "12-FOLD EXPANDING", "ADR-AS-0002"],
    status_text="READY",
    status_tone="neutral",
)

# ═══════════════════════════════════════════════════════════
# CONFIG · ADAPTIVE (per-fold 최적화)
# ═══════════════════════════════════════════════════════════
section_header("ADAPTIVE CONFIG · 12-FOLD WALK-FORWARD")
with panel():
    c1, c2, c3 = st.columns([1.2, 1.2, 1.6], gap="small")
    with c1:
        threshold = st.selectbox(
            "Threshold",
            [0.01, 0.02],
            format_func=lambda v: f"{v*100:.0f}%",
            index=0,
            key="_ad_threshold",
        )
    with c2:
        mode = st.selectbox(
            "CMA-ES mode",
            ["QUICK (30)", "FULL (300)"],
            index=0,
            key="_ad_mode",
        )
        max_evals = 30 if mode.startswith("QUICK") else 300
    with c3:
        st.write("")
        run = st.button(
            "▶  RUN BASELINE",
            type="primary",
            use_container_width=True,
            key="_ad_run",
        )

# ═══════════════════════════════════════════════════════════
# RUN — ADAPTIVE
# ═══════════════════════════════════════════════════════════
if run:
    import traceback

    with st.spinner(f"12-fold walk-forward · {scope}:{ticker}…"):
        try:
            results = baseline_service.run_baseline(
                scope=scope,
                ticker=ticker,
                threshold=threshold,
                max_evals=max_evals,
            )
            set_result("baseline", scope, ticker, results)
            st.success(f"✅ 완료 · {scope}:{ticker}")
        except Exception as e:
            st.error(f"실행 실패: {type(e).__name__}: {e}")
            with st.expander("Traceback", expanded=True):
                st.code(traceback.format_exc())

# ═══════════════════════════════════════════════════════════
# RESULT — ADAPTIVE
# ═══════════════════════════════════════════════════════════
results = get_result("baseline", scope, ticker)
if results:
    s = results
    perf = s["perf_metrics"]
    cls = s["cls_metrics"]

    # ── SUMMARY ────────────────────────────────────────────
    section_header(
        f"ADAPTIVE SUMMARY · {scope}:{ticker} · THRESHOLD {s['threshold']*100:.0f}%"
    )
    metric_row(
        [
            dict(label="FOLDS", value=f"{s['total_folds']}"),
            dict(
                label="SHARPE",
                value=f"{perf.get('sharpe', 0):.4f}",
                tone="up" if perf.get("sharpe", 0) > 0 else "down",
            ),
            dict(
                label="CAGR",
                value=f"{perf.get('cagr', 0)*100:.2f}%",
                tone="up" if perf.get("cagr", 0) > 0 else "down",
            ),
            dict(
                label="MDD", value=f"{-abs(perf.get('mdd', 0))*100:.2f}%", tone="down"
            ),
            dict(label="WIN RATE", value=f"{perf.get('win_rate', 0)*100:.2f}%"),
            dict(label="MACRO F1", value=f"{cls.get('f1_macro', 0):.4f}"),
        ],
        cols=6,
    )

    metric_row(
        [
            dict(label="CALMAR", value=f"{perf.get('calmar', 0):.4f}"),
            dict(label="PROFIT FCT", value=f"{perf.get('profit_factor', 0):.4f}"),
            dict(label="BALANCED ACC", value=f"{cls.get('balanced_acc', 0):.4f}"),
            dict(label="RATIO UP", value=f"{cls.get('ratio_up', 0)*100:.2f}%"),
            dict(label="RATIO NEUT", value=f"{cls.get('ratio_neutral', 0)*100:.2f}%"),
            dict(label="RATIO DOWN", value=f"{cls.get('ratio_down', 0)*100:.2f}%"),
        ],
        cols=6,
    )

    # ── MEDIAN 6-PARAMS ────────────────────────────────────
    section_header("MEDIAN 6-PARAMETERS (참고용 · 실제 신호엔 per-fold 값)")
    pm = s["params_median"]
    metric_row(
        [
            dict(label="α UP", value=f"{pm.get('alpha_up', 0):.4f}"),
            dict(label="α DOWN", value=f"{pm.get('alpha_down', 0):.4f}"),
            dict(label="β UP", value=f"{pm.get('beta_up', 0):.4f}"),
            dict(label="β DOWN", value=f"{pm.get('beta_down', 0):.4f}"),
            dict(label="VOL PERIOD", value=f"{pm.get('vol_period', 0):.0f}"),
            dict(label="VOLUME PERIOD", value=f"{pm.get('volume_period', 0):.0f}"),
        ],
        cols=6,
    )

    # ── FOLD DETAILS ───────────────────────────────────────
    fold_df = pd.DataFrame(s["fold_details"])
    if not fold_df.empty:
        section_header(f"FOLD DETAILS · {len(fold_df)} FOLDS")
        with panel():
            drop_cols = [
                c for c in ["train_start", "train_end"] if c in fold_df.columns
            ]
            view = fold_df.drop(columns=drop_cols)
            num_cols = [
                "alpha_up",
                "alpha_down",
                "beta_up",
                "beta_down",
                "vol_period",
                "volume_period",
                "is_fitness",
                "oos_ret_mean",
            ]
            num_cols = [c for c in num_cols if c in view.columns]
            show_table(view, num_cols=num_cols, precision=4)
else:
    st.caption(
        f"{scope}:{ticker} · Adaptive 결과가 아직 없습니다. RUN BASELINE 을 누르세요."
    )
