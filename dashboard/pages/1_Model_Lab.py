import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Model Lab · AlphaStack", layout="wide")

import theme
from theme import metric_row, panel, section_header, styled_df, top_strip

theme.inject()

from components import sidebar_controls
from services import data_service, evaluation_service, feature_service, model_service
from state import ctx

sidebar_controls()
c = ctx()

# ═══════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════
st.markdown('<div class="as-title">Model Lab</div>', unsafe_allow_html=True)
top_strip(
    [c["dataset"].upper(), c["feature_set"].upper(),
     f"{c['start']} – {c['end']}", "5-FOLD CV"],
    status_text="READY", status_tone="neutral",
)

prices = data_service.load_market_data(c["dataset"], c["start"], c["end"])
X, y = feature_service.build_dataset(prices, c["feature_set"])

ALL_MODELS = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"]

# ═══════════════════════════════════════════════════════════
# ① CONTROLS
# ═══════════════════════════════════════════════════════════
section_header("CONTROLS")
with panel():
    c1, c2 = st.columns([3, 1], gap="small")
    with c1:
        selected = st.multiselect("Models", ALL_MODELS, default=ALL_MODELS)
    with c2:
        st.write("")  # vertical align
        run = st.button("▶  TRAIN", type="primary", use_container_width=True)

# ═══════════════════════════════════════════════════════════
# ② TRAIN
# ═══════════════════════════════════════════════════════════
if run and selected:
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

# ═══════════════════════════════════════════════════════════
# ③ MODEL COMPARISON (최고 성능 행 highlight)
# ═══════════════════════════════════════════════════════════
df = st.session_state.get("_model_compare")
if df is not None:
    best_row = df["Accuracy"].idxmax()          # 인덱스가 모델명 → 문자열 비교 OK

    section_header("MODEL COMPARISON")
    with panel(
        f"{len(df)} MODELS · ENGINE REAL",
        status_text=f"BEST · {best_row}",
        status_tone="up",
    ):
        st.dataframe(
            styled_df(
                df,
                num_cols=["Accuracy", "Balanced Acc", "MCC", "macro F1",
                          "Down Recall", "Neutral Recall", "Up Recall",
                          "Runtime (s)"],
                precision=2,
                highlight_row=best_row,
            ),
            use_container_width=True,
        )

    # ── BASELINE ──────────────────────────────────────────
    section_header("BASELINE COMPARISON")
    _ytr = y.loc[:X.index[int(len(X) * 0.8) - 1]]
    _yvl = y.loc[X.index[int(len(X) * 0.8):]]
    bl = evaluation_service.baseline_comparison(
        _ytr.values, _yvl.values, _yvl.values)

    metric_row([
        dict(label="MAJORITY",         value=f"{bl['majority']*100:.1f}"),
        dict(label="ALWAYS UP",        value=f"{bl['always_up']*100:.1f}"),
        dict(label="PREV DIRECTION",   value=f"{bl['previous_direction']*100:.1f}"),
        dict(label="BEST BASELINE",    value=f"{bl['best_baseline']*100:.1f}",
             accent=True, tone="warn"),
    ], cols=4)
else:
    st.caption("모델을 선택하고 TRAIN을 누르세요.")

# ═══════════════════════════════════════════════════════════
# ④ ENGINE STATUS
# ═══════════════════════════════════════════════════════════
_errs = evaluation_service.engine_status()
if _errs:
    section_header("ENGINE STATUS")
    with panel(f"FALLBACK {len(_errs)}건", status_text="WARN", status_tone="warn"):
        for e in _errs:
            st.markdown(f"- {e}")
