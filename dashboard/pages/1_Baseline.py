import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Baseline · AlphaStack", layout="wide")

import theme
from theme import (
    metric_row,
    section_header,
    top_strip,
    panel,
    show_table,
    plotly_chart,
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


# ═══════════════════════════════════════════════════════════
# FROZEN CLASSIFIER (문제 1)
# ═══════════════════════════════════════════════════════════
section_header("FROZEN CLASSIFIER · 고정 α/β 세트")

st.caption(
    "ℹ️ **전체 구간 1회 최적화**: Baseline은 **판단기**(지금이 상승/중립/하락?)입니다. "
    "미래 예측이 아니라 과거 판정이므로 train/test split 없이 "
    "**역사 전체를 가장 잘 설명하는 기준**을 찾습니다."
)

with panel():
    c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1.4], gap="small")
    with c1:
        _fz_objective = st.selectbox(
            "Objective",
            ["harmonic", "f1"],
            index=0,
            format_func=lambda x: {
                "harmonic": "조화평균 (ACC·F1·DownRec)",
                "f1": "Macro F1",
            }[x],
            key="_fz_objective",
        )
    with c2:
        _fz_max_evals = st.number_input(
            "Max Evals",
            10,
            500,
            100,
            10,
            key="_fz_max_evals",
        )
    with c3:
        _fz_threshold = st.selectbox(
            "Threshold",
            [0.01, 0.02],
            format_func=lambda v: f"{v*100:.0f}%",
            index=0,
            key="_fz_threshold",
        )
    with c4:
        st.write("")
        _fz_run = st.button(
            "▶  RUN FROZEN",
            type="primary",
            use_container_width=True,
            key="_fz_run",
        )

if _fz_run:
    import traceback

    with st.spinner(f"Frozen classifier · {scope}:{ticker} · CMA-ES…"):
        try:
            _fz_result = baseline_service.run_frozen_classifier(
                scope=scope,
                ticker=ticker,
                threshold=float(_fz_threshold),
                max_evals=int(_fz_max_evals),
                objective=_fz_objective,
            )
            st.session_state[f"_frozen_{scope}:{ticker}"] = _fz_result
            st.success(f"✅ 완료 · {scope}:{ticker}")
            st.rerun()
        except Exception as e:
            st.error(f"실행 실패: {type(e).__name__}: {e}")
            with st.expander("Traceback", expanded=True):
                st.code(traceback.format_exc())

# ── 결과 표시 ───────────────────────────────────────
_fz_result = st.session_state.get(f"_frozen_{scope}:{ticker}")
if _fz_result:
    fp = _fz_result["frozen_params"]
    fm = _fz_result["metrics"]
    rng = _fz_result["range"]

    # ① Frozen 파라미터
    section_header("FROZEN PARAMETERS")
    metric_row(
        [
            dict(label="α UP", value=f"{fp['alpha_up']:.4f}"),
            dict(label="α DOWN", value=f"{fp['alpha_down']:.4f}"),
            dict(label="β UP", value=f"{fp['beta_up']:.4f}"),
            dict(label="β DOWN", value=f"{fp['beta_down']:.4f}"),
            dict(label="VOL PERIOD", value=f"{fp['vol_period']}"),
            dict(label="VOLUME PERIOD", value=f"{fp['volume_period']}"),
        ],
        cols=6,
    )

    # ② 전체 구간 성능
    section_header(f"WHOLE-PERIOD METRICS · {rng[0][:4]}–{rng[1][:4]}")
    _n = fm.get("n_valid", 0)

    def _fz_fmt(v, d=4):
        if v is None or (isinstance(v, float) and v != v):
            return "—"
        return f"{v:.{d}f}"

    metric_row(
        [
            dict(label="SAMPLES", value=f"{_n}"),
            dict(label="ACCURACY", value=_fz_fmt(fm.get("accuracy"))),
            dict(label="MACRO F1", value=_fz_fmt(fm.get("macro_f1"))),
            dict(label="BALANCED ACC", value=_fz_fmt(fm.get("balanced_acc"))),
            dict(
                label="DOWN RECALL",
                value=_fz_fmt(fm.get("down_recall")),
                tone="up" if (fm.get("down_recall") or 0) > 0.3 else "warn",
            ),
            dict(
                label="HARMONIC",
                value=_fz_fmt(fm.get("harmonic")),
                tone="up",
                accent=True,
            ),
        ],
        cols=6,
    )

    metric_row(
        [
            dict(label="UP RECALL", value=_fz_fmt(fm.get("up_recall"))),
            dict(label="NEUTRAL RECALL", value=_fz_fmt(fm.get("neutral_recall"))),
            dict(label="OBJECTIVE", value=_fz_result["objective"]),
            dict(label="THRESHOLD", value=f"{_fz_result['threshold']*100:.0f}%"),
        ],
        cols=4,
    )

    # ③ Per-year 안정성
    section_header("PER-YEAR STABILITY")
    _py = pd.DataFrame(_fz_result.get("per_year", []))
    if not _py.empty:
        with panel("연도별 판정 성능"):
            show_table(
                _py,
                num_cols=["accuracy", "macro_f1", "down_recall", "balanced_acc"],
                precision=4,
            )
        with panel("MACRO F1 / DOWN RECALL / BALANCED ACC 안정성"):
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=_py["year"],
                    y=_py["macro_f1"],
                    name="Macro F1",
                    mode="lines+markers",
                    line=dict(color="#7fd1ff", width=2),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=_py["year"],
                    y=_py["down_recall"],
                    name="Down Recall",
                    mode="lines+markers",
                    line=dict(color="#ff5c5c", width=2),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=_py["year"],
                    y=_py["balanced_acc"],
                    name="Balanced Acc",
                    mode="lines+markers",
                    line=dict(color="#4ade80", width=2),
                )
            )
            fig.update_layout(
                height=280,
                xaxis=dict(title="Year"),
                yaxis=dict(title=""),
                hovermode="x unified",
            )
            plotly_chart(fig)
    else:
        st.caption("Per-year 데이터 없음")

    st.caption(
        f"ℹ️ 전체 구간 1회 최적화 · {_fz_result['n_days']}일 · "
        f"{_fz_result['max_evals']} evals · objective={_fz_result['objective']}"
    )
else:
    st.caption("FROZEN classifier를 실행하면 결과가 표시됩니다.")
