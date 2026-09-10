import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Model Lab · AlphaStack", layout="wide")

from components import sidebar_controls
from services import data_service, evaluation_service, feature_service, model_service
from state import ctx

sidebar_controls()
c = ctx()
st.title("Model Lab")

prices = data_service.load_market_data(c["dataset"], c["start"], c["end"])
X, y = feature_service.build_dataset(prices, c["feature_set"])

ALL_MODELS = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"]
selected = st.multiselect("Models", ALL_MODELS, default=ALL_MODELS)

if st.button("Train & Compare", type="primary") and selected:
    rows = []
    bar = st.progress(0.0)
    for i, name in enumerate(selected):
        with st.spinner(f"Training {name}..."):
            r = model_service.train(name, X, y, seed=c["seed"])
        rows.append({
            "Model": name,
            "Engine": r.engine,
            "Accuracy": r.accuracy * 100,
            "Balanced Acc": r.balanced_accuracy * 100,
            "MCC": r.mcc,
            "macro F1": r.macro_f1,
            "Down Recall": r.recalls["down"] * 100,
            "Neutral Recall": r.recalls["neutral"] * 100,
            "Up Recall": r.recalls["up"] * 100,
            "Runtime (s)": r.runtime_sec,
        })
        bar.progress((i + 1) / len(selected))
    df = pd.DataFrame(rows).set_index("Model")
    st.session_state["_model_compare"] = df

df = st.session_state.get("_model_compare")
if df is not None:
    st.dataframe(
        df.style.format({
            "Accuracy": "{:.1f}", "Balanced Acc": "{:.1f}", "MCC": "{:.2f}",
            "macro F1": "{:.2f}",
            "Down Recall": "{:.1f}", "Neutral Recall": "{:.1f}",
            "Up Recall": "{:.1f}", "Runtime (s)": "{:.2f}",
        }),
        use_container_width=True)

    st.subheader("Baseline 대비")
    _ytr = y.loc[:X.index[int(len(X) * 0.8) - 1]]
    _yvl = y.loc[X.index[int(len(X) * 0.8):]]
    bl = evaluation_service.baseline_comparison(
        _ytr.values, _yvl.values, _yvl.values)   # y_pred 자리에 y_valid (baseline 전용)
    bc1, bc2, bc3, bc4 = st.columns(4)
    bc1.metric("Majority", f"{bl['majority']*100:.1f}%")
    bc2.metric("Always Up", f"{bl['always_up']*100:.1f}%")
    bc3.metric("Prev Direction", f"{bl['previous_direction']*100:.1f}%")
    bc4.metric("Best Baseline", f"{bl['best_baseline']*100:.1f}%")
else:
    st.caption("모델을 선택하고 Train & Compare를 누르세요.")

# ── 엔진 상태 ─────────────────────────────────────────────
_errs = evaluation_service.engine_status()
if _errs:
    with st.expander(f"⚠ engine fallback {len(_errs)}건", expanded=False):
        for e in _errs:
            st.write(f"- {e}")
