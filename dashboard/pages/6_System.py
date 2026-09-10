import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

st.set_page_config(page_title="System · AlphaStack", layout="wide")

from components import sidebar_controls
from services import backtest_service, data_service, evaluation_service, feature_service
from state import ctx, fingerprint

sidebar_controls()
c = ctx()
st.title("System / Pipeline")

# ── 파이프라인 상태 ────────────────────────────────────────
st.subheader("Pipeline Status")

_has_model = st.session_state.get("latest_model_result") is not None
_has_bt = st.session_state.get("latest_backtest") is not None

stages = [
    ("Data Ingestion",     "● OK",       f"{c['dataset']} · HF"),
    ("Feature Generation", "● OK",       f"set {c['feature_set']}"),
    ("Model Training",     "● OK" if _has_model else "○ PENDING", c["model"]),
    ("Prediction",         "● OK" if _has_model else "○ PENDING", "-"),
    ("Backtest",           "● OK" if _has_bt else "○ PENDING", f"cost {c['cost']*100:.2f}%"),
    ("Evaluation",         "● OK" if _has_model else "○ PENDING", "-"),
]
st.dataframe(
    pd.DataFrame(stages, columns=["Stage", "Status", "Detail"]),
    use_container_width=True, hide_index=True)

# 파이프라인 실 엔진 이력 시도
try:
    from pipelines import refresh as _refresh
    running = None
    if hasattr(_refresh, "진행중인_실행"):
        running = _refresh.진행중인_실행()
    if running:
        st.info(f"파이프라인 실행 중: {running}")
    else:
        st.caption("파이프라인 진행 중인 실행 없음 (`pipelines.refresh.진행중인_실행`)")
except Exception as e:
    st.caption(f"파이프라인 이력 조회 실패: {e}")

# ── 데이터 요약 ────────────────────────────────────────────
st.subheader("Data")
try:
    prices = data_service.load_market_data(c["dataset"], c["start"], c["end"])
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Rows", f"{len(prices):,}")
    d2.metric("Columns", f"{len(prices.columns)}")
    d3.metric("Missing", f"{int(prices.isna().sum().sum()):,}")
    d4.metric("Range", f"{prices.index.min().date()} ~ {prices.index.max().date()}")
except Exception as e:
    st.warning(f"데이터 로드 실패: {e}")
    prices = None

# ── 피처 요약 ─────────────────────────────────────────────
if prices is not None:
    st.subheader("Feature / Model")
    try:
        X, y = feature_service.build_dataset(prices, c["feature_set"])
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("X shape", f"{X.shape}")
        f2.metric("y classes", f"{sorted(pd.unique(y).tolist())}")
        f3.metric("Feature Set", c["feature_set"])
        f4.metric("Model", c["model"])
    except Exception as e:
        st.warning(f"피처 로드 실패: {e}")

# ── 엔진 상태 ─────────────────────────────────────────────
st.subheader("Engine Status")
rows = []
for name, svc in [("feature", feature_service),
                  ("evaluation", evaluation_service),
                  ("backtest", backtest_service)]:
    errs = svc.engine_status()
    if not errs:
        rows.append({"Service": name, "Status": "● real", "Message": ""})
    else:
        for e in errs:
            rows.append({"Service": name, "Status": "○ fallback", "Message": e})
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── 세션 ──────────────────────────────────────────────────
st.subheader("Session")
s1, s2, s3 = st.columns(3)
s1.metric("Fingerprint", fingerprint()[:26] + "…")
s2.metric("Last Run", st.session_state.get("latest_run_at", "-"))
s3.metric("Experiments", len(st.session_state.get("experiments", [])))
