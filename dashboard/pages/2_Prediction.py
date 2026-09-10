# dashboard/pages/2_Prediction.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Prediction · AlphaStack", layout="wide")

import charts
from components import sidebar_controls

sidebar_controls()
st.title("Prediction")

res = st.session_state.get("latest_model_result")
if res is None:
    st.info("Overview에서 먼저 Run Analysis를 실행하세요.")
    st.stop()

# ── 라벨 순서: -1 / 0 / 1 ─────────────────────────────────
proba = pd.DataFrame(res.y_proba, index=res.test_index,
                     columns=["P(Down)", "P(Neutral)", "P(Up)"])
label_map = {-1: "DOWN", 0: "NEUTRAL", 1: "UP"}

latest = proba.tail(20).copy()
latest["Signal"] = [label_map[int(i)] for i in res.y_pred[-20:]]
st.subheader("Recent predictions")
st.dataframe(latest.iloc[::-1], use_container_width=True)

st.plotly_chart(charts.proba_stack(proba.tail(120)), use_container_width=True)

st.subheader("Top features")
st.dataframe(res.feature_importance.head(15).rename("importance").to_frame(),
             use_container_width=True)
