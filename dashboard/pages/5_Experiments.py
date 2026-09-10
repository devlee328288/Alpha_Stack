import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Experiments · AlphaStack", layout="wide")

from components import sidebar_controls
from state import ctx

sidebar_controls()
st.title("Experiments")

if st.button("Save current run"):
    res = st.session_state.get("latest_model_result")
    bt = st.session_state.get("latest_backtest")
    if res is None or bt is None:
        st.warning("Overview에서 먼저 Run Analysis를 실행하세요.")
    else:
        row = {
            "id": len(st.session_state.experiments) + 1,
            **ctx(),
            "accuracy": round(res.accuracy, 4),
            "mcc": round(res.mcc, 4),
            "sharpe": round(bt.metrics["Sharpe"], 4),
            "mdd": round(bt.metrics["MDD"], 4),
        }
        st.session_state.experiments.append(row)

if st.session_state.experiments:
    st.dataframe(pd.DataFrame(st.session_state.experiments), use_container_width=True)
else:
    st.caption("저장된 실험이 아직 없습니다.")
