import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

st.set_page_config(page_title="System · AlphaStack", layout="wide")

import theme
from theme import (
    metric_row,
    panel,
    section_header,
    status_dot,
    top_strip,
)

theme.inject()

from components import sidebar_controls
from services import backtest_service, data_service, evaluation_service, feature_service
from state import ctx, fingerprint

sidebar_controls()
c = ctx()

# ═══════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════
st.markdown('<div class="as-title">System / Pipeline</div>', unsafe_allow_html=True)

_has_model = st.session_state.get("latest_model_result") is not None
_has_bt = st.session_state.get("latest_backtest") is not None

_overall_tone = "up" if (_has_model and _has_bt) else ("warn" if _has_model else "neutral")
_overall_text = ("LIVE" if (_has_model and _has_bt)
                 else ("MODEL READY" if _has_model else "IDLE"))

top_strip(
    [c["dataset"].upper(),
     f"MODEL {c['model'].upper()}",
     f"FEATURE {c['feature_set'].upper()}",
     f"{c['start']} – {c['end']}"],
    status_text=_overall_text, status_tone=_overall_tone,
)

# ═══════════════════════════════════════════════════════════
# ① PIPELINE STATUS
# ═══════════════════════════════════════════════════════════
section_header("PIPELINE STATUS")

stages = [
    ("Data Ingestion",     "up",   f"{c['dataset']} · HF"),
    ("Feature Generation", "up",   f"set {c['feature_set']}"),
    ("Model Training",     "up" if _has_model else "neutral",
                           c["model"] if _has_model else "PENDING"),
    ("Prediction",         "up" if _has_model else "neutral",
                           "-" if not _has_model else "OK"),
    ("Backtest",           "up" if _has_bt else "neutral",
                           f"cost {c['cost']*100:.2f}%" if _has_bt else "PENDING"),
    ("Evaluation",         "up" if _has_model else "neutral",
                           "-" if not _has_model else "OK"),
]

with panel("6 STAGES", status_text=_overall_text, status_tone=_overall_tone):
    for name, tone, detail in stages:
        st.markdown(
            f'<div class="as-status-line" style="justify-content:space-between;'
            f'padding:4px 2px;border-bottom:1px solid var(--border-subtle);">'
            f'  <span style="color:var(--text-primary);">{name}</span>'
            f'  <span>{status_dot(tone)} '
            f'<span style="color:var(--text-secondary);">{detail}</span></span>'
            f'</div>',
            unsafe_allow_html=True,
        )

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

# ═══════════════════════════════════════════════════════════
# ② DATA SUMMARY
# ═══════════════════════════════════════════════════════════
section_header("DATA")

prices = None
try:
    prices = data_service.load_market_data(c["dataset"], c["start"], c["end"])
    metric_row([
        dict(label="ROWS",     value=f"{len(prices):,}"),
        dict(label="COLUMNS",  value=f"{len(prices.columns)}"),
        dict(label="MISSING",  value=f"{int(prices.isna().sum().sum()):,}",
             tone="warn" if prices.isna().sum().sum() > 0 else "neutral"),
        dict(label="RANGE",
             value=f"{prices.index.min().date()} → {prices.index.max().date()}",
             variant="compact"),
    ], cols=4)
except Exception as e:
    st.warning(f"데이터 로드 실패: {e}")

# ═══════════════════════════════════════════════════════════
# ③ FEATURE / MODEL
# ═══════════════════════════════════════════════════════════
if prices is not None:
    section_header("FEATURE / MODEL")
    try:
        X, y = feature_service.build_dataset(prices, c["feature_set"])
        metric_row([
            dict(label="X SHAPE",     value=str(X.shape)),
            dict(label="Y CLASSES",   value=str(sorted(pd.unique(y).tolist()))),
            dict(label="FEATURE SET", value=c["feature_set"]),
            dict(label="MODEL",       value=c["model"]),
        ], cols=4)
    except Exception as e:
        st.warning(f"피처 로드 실패: {e}")

# ═══════════════════════════════════════════════════════════
# ④ ENGINE STATUS
# ═══════════════════════════════════════════════════════════
section_header("ENGINE STATUS")

rows = []
for name, svc in [("feature", feature_service),
                  ("evaluation", evaluation_service),
                  ("backtest", backtest_service)]:
    errs = svc.engine_status()
    if not errs:
        rows.append({"Service": name, "Status": "real", "Message": ""})
    else:
        for e in errs:
            rows.append({"Service": name, "Status": "fallback", "Message": e})

_all_real = all(r["Status"] == "real" for r in rows) if rows else True
with panel(
    f"{len(rows)} SERVICE{'S' if len(rows) != 1 else ''}",
    status_text=("ALL REAL" if _all_real else "FALLBACK"),
    status_tone=("up" if _all_real else "warn"),
):
    for r in rows:
        tone = "up" if r["Status"] == "real" else "warn"
        msg = r["Message"] or ("engine ready" if tone == "up" else "")
        st.markdown(
            f'<div class="as-status-line" style="justify-content:space-between;'
            f'padding:4px 2px;border-bottom:1px solid var(--border-subtle);">'
            f'  <span style="color:var(--text-primary);">'
            f'    {status_dot(tone)} {r["Service"]}</span>'
            f'  <span style="color:var(--text-secondary);">{msg}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ═══════════════════════════════════════════════════════════
# ⑤ SESSION
# ═══════════════════════════════════════════════════════════
section_header("SESSION")
fp = fingerprint()
metric_row([
    dict(label="FINGERPRINT", value=f"{fp[:26]}…", variant="compact"),
    dict(label="LAST RUN",    value=str(st.session_state.get("latest_run_at", "-"))),
    dict(label="EXPERIMENTS", value=str(len(st.session_state.get("experiments", [])))),
], cols=3)
