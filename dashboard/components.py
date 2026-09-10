import streamlit as st
from dash_config import DATASETS, FEATURE_SET_LABELS, FEATURE_SETS, MODELS
from state import fingerprint, init_state
from theme import (
    metric_row as _metric_row,
)
from theme import (
    sidebar_header,
    sidebar_meta,
    sidebar_section,
)


def sidebar_controls() -> bool:
    """
    A안 사이드바. 위젯 키/시그니처는 이전과 100% 동일.
    반환: Run Analysis 버튼 클릭 여부(bool)
    """
    init_state()

    with st.sidebar:
        # ── 브랜딩 ────────────────────────────
        sidebar_header("QUANT RESEARCH TERMINAL")

        # ── 컨텍스트 위젯 ─────────────────────
        sidebar_section("CONTEXT")
        st.selectbox("Dataset", DATASETS, key="dataset")
        st.selectbox("Model", MODELS, key="model")
        st.selectbox(
            "Feature Set",
            FEATURE_SETS,
            key="feature_set",
            format_func=lambda k: FEATURE_SET_LABELS.get(k, k),
        )

        c1, c2 = st.columns(2)
        c1.date_input("Start", key="start")
        c2.date_input("End", key="end")

        st.number_input(
            "Cost (per turnover)",
            min_value=0.0, max_value=0.01,
            step=0.0005, format="%.4f", key="cost",
            help="0.0010 = 0.10%",
        )
        st.number_input(
            "Seed",
            min_value=0, max_value=10000, step=1, key="seed",
        )

        # ── 실행 ──────────────────────────────
        sidebar_section("RUN")
        run = st.button(
            "▶  RUN ANALYSIS",
            type="primary",
            use_container_width=True,
        )

        # ── 세션 메타 (하단) ──────────────────
        sidebar_meta([
            ("Fingerprint", fingerprint()[:20]),
        ])

        return run


def kpi_row(items):
    """
    하위 호환용. 내부적으로 A안 metric_row 사용.

    items: list of (label, value) 또는 (label, value, delta) 또는
                     (label, value, delta, tone)
    """
    metrics = []
    for item in items:
        if len(item) >= 3:
            label, value, delta = item[0], item[1], item[2]
            tone = item[3] if len(item) >= 4 else "neutral"
        else:
            label, value = item[0], item[1]
            delta = None
            tone = "neutral"

        metrics.append(dict(
            label=str(label).upper(),
            value=str(value),
            delta=delta,
            tone=tone,
        ))
    if metrics:
        _metric_row(metrics, cols=len(metrics))
