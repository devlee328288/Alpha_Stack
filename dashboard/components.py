import streamlit as st
from dash_config import DATASETS, FEATURE_SET_LABELS, FEATURE_SETS, MODELS
from state import fingerprint, init_state


def sidebar_controls() -> bool:
    init_state()
    with st.sidebar:
        st.markdown("### ALPHASTACK")
        st.caption("Quant Research Terminal")
        st.divider()

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

        st.number_input("Cost (per turnover)", min_value=0.0, max_value=0.01,
                        step=0.0005, format="%.4f", key="cost",
                        help="0.0010 = 0.10%")
        st.number_input("Seed", min_value=0, max_value=10000, step=1, key="seed")

        st.divider()
        run = st.button("Run Analysis", type="primary", use_container_width=True)
        st.caption(f"fingerprint · {fingerprint()}")
        return run


def kpi_row(items):
    """
    items: list of (label, value) 또는 (label, value, delta)
    """
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        if len(item) == 3:
            label, value, delta = item
        else:
            label, value = item
            delta = None
        col.metric(label, value, delta)
