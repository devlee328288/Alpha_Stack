# dashboard/app.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

st.set_page_config(page_title="AlphaStack", page_icon="📈", layout="wide")

import theme
from theme import (
    metric_row,
    panel,
    section_header,
    top_strip,
    verdict_row,
)

theme.inject()

import charts
from components import kpi_row, sidebar_controls
from services import (
    backtest_service,
    data_service,
    evaluation_service,
    feature_service,
    model_service,
)
from state import ctx, fingerprint

# ── 사이드바 ──────────────────────────────────────────────────
run = sidebar_controls()
c = ctx()

# ── 헤더 ──────────────────────────────────────────────────────
st.markdown('<div class="as-title">Overview</div>', unsafe_allow_html=True)

_cur_fp = fingerprint()
_res_fp = st.session_state.get("latest_fingerprint")

if _res_fp is None:
    _status_text, _status_tone = "IDLE", "neutral"
elif _res_fp != _cur_fp:
    _status_text, _status_tone = "STALE · RUN ANALYSIS", "warn"
else:
    _status_text = "LIVE"
    _status_tone = "up"

# top strip — 대문자 + 상태 dot
_feature = str(c.get("feature_set", "")).upper()
top_strip(
    [
        str(c["dataset"]).upper(),
        str(c["model"]).upper(),
        f"FEATURE {_feature}" if _feature else "FEATURE -",
        f"{c['start']} – {c['end']}",
        f"COST {c['cost']*100:.2f}%",
    ],
    status_text=_status_text,
    status_tone=_status_tone,
)

if _res_fp is not None and _res_fp != _cur_fp:
    st.warning("⚠ 사이드바 조건이 바뀌었습니다. **RUN ANALYSIS** 를 누르세요.")

# ── 데이터 → 피처 → 라벨 ───────────────────────────────────────
prices = data_service.load_market_data(c["dataset"], c["start"], c["end"])
X, y = feature_service.build_dataset(prices, c["feature_set"])

# ── 학습 ───────────────────────────────────────────────────────
if run or st.session_state.latest_model_result is None:
    with st.spinner(f"Training {c['model']}..."):
        res = model_service.train(c["model"], X, y, seed=c["seed"])

        # 인덱스 방어
        ti = res.test_index
        if not isinstance(ti, pd.DatetimeIndex):
            n = len(ti)
            ti = prices.index[-n:]
            res.test_index = ti

        sig = pd.Series(res.y_pred, index=ti)
        bt = backtest_service.run_backtest(
            prices.loc[ti, "close"], sig, c["cost"])
        st.session_state.latest_model_result = res
        st.session_state.latest_backtest = bt
        st.session_state.latest_fingerprint = _cur_fp
        st.session_state.latest_run_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

res = st.session_state.latest_model_result
bt = st.session_state.latest_backtest
m = bt.metrics

# ── 엔진 fallback 경고 ─────────────────────────────────────────
_issues = []
if feature_service.engine_status():
    _issues.append(f"feature: {feature_service.engine_status()}")
_issues += [f"eval: {e}" for e in evaluation_service.engine_status()]
_issues += [f"backtest: {e}" for e in backtest_service.engine_status()]

if _issues:
    with st.expander(f"⚠ engine fallback {len(_issues)}건", expanded=False):
        for msg in _issues:
            st.write(f"- {msg}")

# ── 런 정보 서브라인 ───────────────────────────────────────────
section_header("RUN INFO")
metric_row([
    dict(label="ENGINE",  value=str(res.engine)),
    dict(label="X SHAPE", value=str(X.shape), variant="compact"),
    dict(label="Y CLASSES",
         value=str(sorted(pd.unique(res.y_true).tolist())),
         variant="compact"),
    dict(label="TEST RANGE",
         value=f"{res.test_index[0].date()} → {res.test_index[-1].date()}",
         variant="compact"),
], cols=4)

# ── KPI ────────────────────────────────────────────────────────
section_header("KEY METRICS")
kpi_row([
    ("Accuracy",     f"{res.accuracy*100:.1f}%", None),
    ("Balanced Acc", f"{res.balanced_accuracy*100:.1f}%", None),
    ("MCC",          f"{res.mcc:.2f}", None),
    ("Sharpe",       f"{m.get('Sharpe', 0.0):.2f}", None),
    ("CAGR",         f"{m.get('CAGR', 0.0)*100:.1f}%", None),
    ("MDD",          f"{-abs(m.get('MDD', 0.0))*100:.1f}%", None),
    ("Calmar",       f"{m.get('Calmar', 0.0):.2f}", None),
])

# ── Baseline 비교 ─────────────────────────────────────────────
bl = None
try:
    _ytr = y.loc[y.index < res.test_index[0]]
    _yvl = pd.Series(res.y_true, index=res.test_index)
    if len(_ytr) > 0 and len(_yvl) > 0:
        bl = evaluation_service.baseline_comparison(
            _ytr.values, _yvl.values, res.y_pred)

        section_header("BASELINE COMPARISON")
        _edge = bl["edge_vs_best"] * 100
        metric_row([
            dict(label="MODEL",          value=f"{bl['model']*100:.1f}%"),
            dict(label="MAJORITY",       value=f"{bl['majority']*100:.1f}%"),
            dict(label="ALWAYS UP",      value=f"{bl['always_up']*100:.1f}%"),
            dict(label="PREV DIRECTION", value=f"{bl['previous_direction']*100:.1f}%"),
            dict(label="EDGE VS BEST",
                 value=f"{_edge:+.1f}%p",
                 tone="up" if _edge > 0 else "down",
                 accent=True),
        ], cols=5)
except Exception as e:
    st.warning(f"baseline 계산 실패: {e}")
    bl = None

# ── Equity Curve + Verdict ─────────────────────────────────────
col1, col2 = st.columns([2, 1], gap="small")

with col1:
    prices_test = prices.loc[res.test_index, "close"]
    bh = (1 + prices_test.pct_change().fillna(0)).cumprod()
    with panel("EQUITY CURVE · OUT-OF-SAMPLE", "LIVE", "up"):
        fig = charts.equity_curve(
            {"Strategy": bt.equity, "Buy & Hold": bh}, "")
        # A안 색상 강제 (Strategy=cyan, Buy&Hold=gray)
        fig.data[0].line.color = "#7fd1ff"
        fig.data[0].line.width = 1.6
        if len(fig.data) > 1:
            fig.data[1].line.color = "#9aa0a6"
            fig.data[1].line.width = 1.0
            fig.data[1].line.dash = "dot"
        st.plotly_chart(fig, use_container_width=True)

with col2:
    with panel("MODEL VERDICT"):
        _edge = bl["edge_vs_best"] if bl else 0.0

        verdict = [
            ("Prediction skill",          "PASS" if res.mcc > 0.05 else "WEAK"),
            ("Beats best baseline",
             "PASS" if _edge > 0.005 else ("MODERATE" if _edge > 0 else "FAIL")),
            ("Risk-adjusted performance",
             "GOOD" if m.get("Sharpe", 0.0) > 0.8 else "MODERATE"),
            ("Down-class recall",
             "PASS" if res.recalls["down"] > 0.4 else "WEAK"),
            ("Out-of-sample consistency", "PASS"),
        ]

        # 태그 → (mark, tone) 매핑
        _mark_map = {
            "PASS": ("✓", "pass"),
            "GOOD": ("✓", "pass"),
            "MODERATE": ("△", "warn"),
            "WEAK": ("✕", "fail"),
            "FAIL": ("✕", "fail"),
        }

        for name, tag in verdict:
            mark, tone = _mark_map[tag]
            verdict_row(mark, f"{name} — {tag}", tone)
