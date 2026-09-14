import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Comparison · AlphaStack", layout="wide")

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
from services import comparison_service
from state import ctx, get_result

sidebar_controls()
c = ctx()
scope = c["scope"]
ticker = c["ticker"]

# ── UNIVERSE scope 가드 ──
if scope == "UNIVERSE":
    st.markdown(
        '<div class="as-title">전용 페이지로 이동</div>', unsafe_allow_html=True
    )
    st.info("**UNIVERSE scope 는 Universe 페이지에서만 사용합니다.**")
    if st.button("▶  Universe 페이지로", type="primary"):
        st.switch_page("pages/8_Universe.py")
    st.stop()

# ═══════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════
st.markdown('<div class="as-title">Comparison</div>', unsafe_allow_html=True)

baseline = get_result("baseline", scope, ticker)
models_fwd = get_result("model_lab", scope, ticker)
models_ad = get_result("model_lab_adaptive", scope, ticker)

top_strip(
    [
        scope,
        ticker,
        f"BASELINE {'OK' if baseline else '—'}",
        f"MODELS {sum(1 for x in [models_fwd, models_ad] if x)}/2",
    ],
    status_text="READY",
    status_tone="up",
)

if not baseline and not models_fwd and not models_ad:
    st.info(
        "**Baseline 페이지**에서 RUN BASELINE, "
        "**Model Lab**에서 각 라벨 소스로 RUN 하세요."
    )
    st.stop()

# ═══════════════════════════════════════════════════════════
# ① BASELINE COMPARISON · 라벨 분포
# ═══════════════════════════════════════════════════════════
section_header("BASELINE COMPARISON · 기준선 라벨 분포")
st.caption(
    "ℹ️ **기준선 = 라벨을 정의하는 규칙.**  \n"
    "- fwd_return ±1%: 미래 5일 수익률 (예측 라벨)  \n"
    "- Adaptive 6-param: 폴드별 최적 band (시기별 기준 변화)"
)

# fwd_return 라벨 생성 (표시용)
_fwd_labels = None
if baseline is not None:
    try:
        from services import data_loader
        from step5_optimize_6params import make_labels as _mk_fwd_labels

        if scope == "full" and isinstance(ticker, str) and len(ticker) == 6:
            df_ = data_loader.load_market_data_full(ticker, price_col="adj_close")
        else:
            df_ = data_loader.load_market_data(scope, ticker)
        _fwd_labels = _mk_fwd_labels(df_, threshold=0.01).tolist()
    except Exception:
        _fwd_labels = None

_base_cmp = comparison_service.build_baseline_comparison(
    baseline_adaptive=baseline,
    fwd_labels=_fwd_labels,
)

if not _base_cmp.empty:
    with panel(f"{len(_base_cmp)} 기준선"):
        show_table(
            _base_cmp,
            num_cols=["UP %", "NEUTRAL %", "DOWN %"],
            precision=4,
        )

    with panel("라벨 분포 비교"):
        fig = go.Figure()
        classes = ["UP %", "NEUTRAL %", "DOWN %"]
        colors = {"UP %": "#4ade80", "NEUTRAL %": "#9aa0a6", "DOWN %": "#ff5c5c"}
        for cl in classes:
            fig.add_trace(
                go.Bar(
                    x=_base_cmp["SOURCE"],
                    y=_base_cmp[cl],
                    name=cl,
                    marker_color=colors[cl],
                    text=[f"{v*100:.1f}%" for v in _base_cmp[cl]],
                    textposition="outside",
                )
            )
        fig.update_layout(
            barmode="group",
            height=320,
            yaxis=dict(title="비율", tickformat=".0%"),
            xaxis=dict(title=""),
            legend=dict(orientation="h", y=1.1, x=0),
        )
        plotly_chart(fig)

# ═══════════════════════════════════════════════════════════
# ② MODEL PER BASELINE
# ═══════════════════════════════════════════════════════════
_all_slots = {
    "fwd_return": models_fwd,
    "adaptive": models_ad,
}
_available = [k for k, v in _all_slots.items() if v]

if _available:
    section_header("MODEL PER BASELINE · 라벨 소스별 ML 성능")
    st.caption("ℹ️ 각 기준선이 정의한 라벨을 ML이 얼마나 잘 학습하는가.")

    _pivot = comparison_service.build_metric_pivot(_all_slots)
    if not _pivot.empty:
        with panel(f"{len(_pivot)} 라벨 소스"):
            show_table(
                _pivot,
                num_cols=["ACC", "MACRO F1", "DOWN REC", "HARMONIC", "ΔSHARPE"],
                precision=4,
            )

        with panel("HARMONIC 비교"):
            fig = go.Figure(
                go.Bar(
                    x=_pivot["LABEL SOURCE"],
                    y=_pivot["HARMONIC"],
                    marker_color=[
                        "#4ade80" if i == _pivot["HARMONIC"].idxmax() else "#7fd1ff"
                        for i in range(len(_pivot))
                    ],
                    text=[f"{v:.4f}" for v in _pivot["HARMONIC"]],
                    textposition="outside",
                )
            )
            fig.update_layout(
                height=300,
                yaxis=dict(
                    title="HARMONIC",
                    range=[0, float(_pivot["HARMONIC"].max()) * 1.2],
                ),
                xaxis=dict(title=""),
                showlegend=False,
            )
            plotly_chart(fig)

# ═══════════════════════════════════════════════════════════
# ③ MODEL DETAIL + REPRODUCTION
# ═══════════════════════════════════════════════════════════
if _available:
    section_header("MODEL DETAIL · 라벨 소스별 랭킹")

    _view = st.radio(
        "VIEW",
        _available,
        format_func=lambda x: {
            "fwd_return": "fwd_return 결과",
            "adaptive": "Adaptive 결과",
        }[x],
        horizontal=True,
        key="_cmp_view_src",
    )

    _results = _all_slots[_view]
    _kpi = comparison_service.build_summary_kpis(_results)
    _cmp_df = comparison_service.build_comparison_table(_results)

    with panel(
        f"{_view} · {len(_cmp_df)} MODELS",
        status_text=f"BEST · {_kpi['best_model']}",
        status_tone="up",
    ):
        display = _cmp_df.rename(columns=comparison_service.COMPARISON_COLUMNS)
        show_table(
            display,
            num_cols=[
                "ACC",
                "MACRO F1",
                "DOWN RECALL",
                "HARMONIC",
                "BAL ACC",
                "MAJORITY",
                "ΔSHARPE",
            ],
            precision=4,
            highlight_row=0,
        )

    with panel(f"BEST · {_kpi['best_model']}"):
        metric_row(
            [
                dict(label="ACC", value=f"{_kpi['best_accuracy']:.4f}"),
                dict(label="MACRO F1", value=f"{_kpi['best_macro_f1']:.4f}"),
                dict(label="DOWN REC", value=f"{_kpi['best_down_recall']:.4f}"),
                dict(
                    label="HARMONIC",
                    value=f"{_kpi['best_harmonic']:.4f}",
                    tone="up",
                    accent=True,
                ),
                dict(
                    label="ΔSHARPE",
                    value=f"{_kpi['best_delta_sharpe']:+.4f}",
                    tone="up" if _kpi["best_delta_sharpe"] > 0 else "down",
                ),
            ],
            cols=5,
        )

    # ── REPRODUCTION ──────────────────────────────
    section_header("REPRODUCTION · ML이 Baseline 판정을 얼마나 재현?")
    st.caption(
        "ℹ️ **Baseline 라벨과 ML 예측을 bas_dd 로 매칭**해 일치도 측정.  \n"
        "Agreement = 같은 시점에서 ML 예측 == Baseline 판정.  \n"
        "Kappa = 우연 일치를 보정한 일치도 (0.6 양호 / 0.8 우수)."
    )

    # baseline 기준 후보: adaptive (baseline 결과 있으면)
    _base_choices = []
    if baseline is not None:
        _base_choices.append("adaptive")
    if models_fwd:
        _base_choices.append("fwd_return")

    if not _base_choices:
        st.info("Baseline 페이지에서 RUN BASELINE 필요.")
    else:
        c1, c2 = st.columns([2, 3], gap="small")
        with c1:
            _base_pick = st.radio(
                "BASELINE 기준",
                _base_choices,
                format_func=lambda x: {
                    "adaptive": "Adaptive 6-param",
                    "fwd_return": "fwd_return ±1%",
                }[x],
                key="_rep_base_pick",
                horizontal=False,
            )

        # baseline 기준에 맞는 ML 슬롯 자동 선택
        _slot_for_base = {
            "adaptive": "adaptive",
            "fwd_return": "fwd_return",
        }
        _target_slot = _slot_for_base.get(_base_pick, "fwd_return")

        if not _all_slots.get(_target_slot):
            st.warning(
                f"⚠️ **{_base_pick}** 라벨로 학습한 ML이 없습니다. "
                f"Model Lab 에서 Label Source = `{_base_pick}` 로 RUN 하세요."
            )
            _ml_pick = None
        else:
            with c2:
                _ml_pick = st.radio(
                    "ML 모델",
                    list(_all_slots[_target_slot].keys()),
                    key="_rep_ml_pick",
                    horizontal=True,
                )
            st.caption(
                f"ℹ️ baseline=`{_base_pick}` ↔ ML=`{_target_slot}` 슬롯 자동 매칭"
            )

            # baseline 라벨 + dates 준비
            _bl_labels = None
            _bl_dates = None

            if _base_pick == "adaptive" and baseline is not None:
                try:
                    from services import baseline_service as _bs

                    _ad = _bs.make_adaptive_labels_from_baseline(
                        scope,
                        ticker,
                        baseline.get("fold_details", []),
                    )
                    _bl_labels = _ad.get("labels")
                    _bl_dates = _ad.get("dates")
                except Exception as e:
                    st.warning(f"Adaptive 라벨 재생성 실패: {e}")

            elif _base_pick == "fwd_return":
                try:
                    from services import data_loader
                    from step5_optimize_6params import make_labels as _mk_fwd_labels

                    if scope == "full" and isinstance(ticker, str) and len(ticker) == 6:
                        df_ = data_loader.load_market_data_full(
                            ticker, price_col="adj_close"
                        )
                    else:
                        df_ = data_loader.load_market_data(scope, ticker)
                    fwd_lab = _mk_fwd_labels(df_, threshold=0.01)
                    _bl_labels = fwd_lab.tolist()
                    _bl_dates = [str(d) for d in df_.index]
                except Exception as e:
                    st.warning(f"fwd_return 라벨 생성 실패: {e}")

            if not _bl_labels or not _bl_dates:
                st.info(
                    "Baseline 라벨 또는 dates 없음.  \n"
                    "**Baseline 페이지에서 RUN BASELINE 을 다시 실행**하세요."
                )
            elif _ml_pick:
                _ml_res = _all_slots[_target_slot].get(_ml_pick)
                if _ml_res is None:
                    st.info("선택한 ML 결과 없음.")
                else:
                    _rep = comparison_service.compute_reproduction(
                        _ml_res,
                        _bl_labels,
                        _bl_dates,
                    )

                    if "error" in _rep:
                        st.warning(f"재현도 계산 불가: {_rep['error']}")
                    else:
                        metric_row(
                            [
                                dict(
                                    label="AGREEMENT",
                                    value=f"{_rep['agreement']*100:.2f}%",
                                    tone="up" if _rep["agreement"] > 0.6 else "warn",
                                    accent=True,
                                ),
                                dict(
                                    label="COHEN'S KAPPA",
                                    value=f"{_rep['kappa']:.4f}",
                                    tone="up" if _rep["kappa"] > 0.6 else "warn",
                                ),
                                dict(label="N MATCHED", value=f"{_rep['n_matched']}"),
                                dict(label="BASELINE", value=_base_pick),
                                dict(label="ML MODEL", value=_ml_pick),
                            ],
                            cols=5,
                        )

                        # Per-class
                        _pc_rows = []
                        for cls_name, v in _rep["per_class"].items():
                            _pc_rows.append(
                                {
                                    "CLASS": cls_name,
                                    "N": v["n"],
                                    "AGREEMENT": v["agreement"],
                                }
                            )
                        section_header("PER-CLASS AGREEMENT")
                        with panel(f"{_base_pick} 판정을 ML이 얼마나 따라갔나"):
                            show_table(
                                pd.DataFrame(_pc_rows),
                                num_cols=["AGREEMENT"],
                                precision=4,
                            )

                        # Confusion heatmap
                        section_header("CONFUSION · rows=Baseline, cols=ML")
                        with panel("3 × 3"):
                            cls_labels_x = list(_rep["classes"])
                            cls_labels_y = list(reversed(_rep["classes"]))
                            cm = np.array(_rep["confusion"])[::-1, :]
                            fig = go.Figure(
                                go.Heatmap(
                                    z=cm,
                                    x=[f"ML·{c}" for c in cls_labels_x],
                                    y=[f"BASE·{c}" for c in cls_labels_y],
                                    colorscale=[
                                        [0.0, "#0e1420"],
                                        [0.4, "#1e4a6e"],
                                        [1.0, "#7fd1ff"],
                                    ],
                                    showscale=False,
                                    text=cm,
                                    texttemplate="%{text}",
                                    textfont=dict(
                                        size=14,
                                        color="#e6e8ec",
                                        family="JetBrains Mono, monospace",
                                    ),
                                )
                            )
                            fig.update_layout(
                                height=320,
                                xaxis=dict(title="ML Prediction", side="top"),
                                yaxis=dict(title="Baseline Label"),
                            )
                            plotly_chart(fig)

                            cm_df = pd.DataFrame(
                                cm,
                                index=[f"base_{c}" for c in cls_labels_y],
                                columns=[f"ml_{c}" for c in cls_labels_x],
                            )
                            show_table(cm_df, num_cols=list(cm_df.columns), precision=0)
