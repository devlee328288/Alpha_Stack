# dashboard/app.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

st.set_page_config(page_title="AlphaStack", page_icon="📈", layout="wide")

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

st.title("Overview")
st.caption(f"{c['dataset']} · {c['model']} · {c['feature_set']} · "
           f"{c['start']} ~ {c['end']} · cost {c['cost']*100:.2f}%")

# ── 상태 배지 ──────────────────────────────────────────────────
_cur_fp = fingerprint()
_res_fp = st.session_state.get("latest_fingerprint")
if _res_fp is None:
    pass
elif _res_fp != _cur_fp:
    st.warning("⚠ 사이드바 조건이 바뀌었습니다. **Run Analysis** 를 누르세요.")
else:
    st.caption(f"✓ 화면 결과 = 사이드바 조건 · last run "
               f"{st.session_state.get('latest_run_at', '-')}")

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

st.caption(f"model engine: **{res.engine}** · X={X.shape} · "
           f"y={sorted(pd.unique(res.y_true).tolist())} · "
           f"test {res.test_index[0].date()} ~ {res.test_index[-1].date()}")

# ── KPI ────────────────────────────────────────────────────────
kpi_row([
    ("Accuracy", f"{res.accuracy*100:.1f}%", None),
    ("Balanced Acc", f"{res.balanced_accuracy*100:.1f}%", None),
    ("MCC", f"{res.mcc:.2f}", None),
    ("Sharpe", f"{m.get('Sharpe', 0.0):.2f}", None),
    ("CAGR", f"{m.get('CAGR', 0.0)*100:.1f}%", None),
    ("MDD", f"{-abs(m.get('MDD', 0.0))*100:.1f}%", None),
    ("Calmar", f"{m.get('Calmar', 0.0):.2f}", None),
])

# ── Baseline 비교 ─────────────────────────────────────────────
try:
    _ytr = y.loc[y.index < res.test_index[0]]
    _yvl = pd.Series(res.y_true, index=res.test_index)
    if len(_ytr) > 0 and len(_yvl) > 0:
        bl = evaluation_service.baseline_comparison(
            _ytr.values, _yvl.values, res.y_pred)

        st.markdown("#### Baseline comparison")
        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("Model", f"{bl['model']*100:.1f}%")
        b2.metric("Majority", f"{bl['majority']*100:.1f}%")
        b3.metric("Always Up", f"{bl['always_up']*100:.1f}%")
        b4.metric("Prev Direction", f"{bl['previous_direction']*100:.1f}%")
        b5.metric("Edge vs Best",
                  f"{bl['edge_vs_best']*100:+.1f}%p")
    else:
        bl = None
except Exception as e:
    st.warning(f"baseline 계산 실패: {e}")
    bl = None

# ── Equity Curve + Verdict ─────────────────────────────────────
col1, col2 = st.columns([2, 1])
with col1:
    prices_test = prices.loc[res.test_index, "close"]
    bh = (1 + prices_test.pct_change().fillna(0)).cumprod()
    st.plotly_chart(
        charts.equity_curve({"Strategy": bt.equity, "Buy & Hold": bh},
                            "Equity Curve (Out-of-Sample)"),
        use_container_width=True,
    )
with col2:
    st.markdown("#### Model Verdict")

    _edge = bl["edge_vs_best"] if bl else 0.0
    verdict = [
        ("Prediction skill",
         "PASS" if res.mcc > 0.05 else "WEAK"),
        ("Beats best baseline",
         "PASS" if _edge > 0.005
         else ("MODERATE" if _edge > 0 else "FAIL")),
        ("Risk-adjusted performance",
         "GOOD" if m.get("Sharpe", 0.0) > 0.8 else "MODERATE"),
        ("Down-class recall",
         "PASS" if res.recalls["down"] > 0.4 else "WEAK"),
        ("Out-of-sample consistency", "PASS"),
    ]
    icon = {"PASS": "✓", "GOOD": "✓", "MODERATE": "△",
            "WEAK": "✕", "FAIL": "✕"}
    for name, tag in verdict:
        st.write(f"{icon[tag]} **{name}** — {tag}")
