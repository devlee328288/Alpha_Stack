import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Backtest · AlphaStack", layout="wide")

import theme
from theme import (
    metric_row, section_header, top_strip, panel, show_table, plotly_chart,
)
theme.inject()

from components import sidebar_controls
from services import backtest_service
from state import ctx, get_result, set_result

sidebar_controls()
c = ctx()
scope = c["scope"]
ticker = c["ticker"]

# ── UNIVERSE scope 는 전용 페이지에서 ──
if scope == "UNIVERSE":
    st.markdown('<div class="as-title">전용 페이지로 이동</div>', unsafe_allow_html=True)
    st.info(
        "**UNIVERSE scope 는 Universe 페이지에서만 사용합니다.**  \n"
        "사이드바에서 **UNIVERSE** 버튼을 다시 누르거나, "
        "페이지 목록의 **Universe** 를 여세요."
    )
    if st.button("▶  Universe 페이지로", type="primary"):
        # 사이드바 페이지 이름이 파일명에 따라 다를 수 있음 — 안내만
        st.switch_page("pages/8_Universe.py")
    st.stop()

st.markdown('<div class="as-title">Backtest</div>', unsafe_allow_html=True)

_err = backtest_service.engine_status()
if _err:
    top_strip([scope, ticker], status_text="IMPORT FAIL", status_tone="warn")
    st.error(_err)
    st.stop()

_bt_prev = get_result("bt", scope, ticker)
_bt_pred_disp = "—"
if _bt_prev:
    _bt_pred_disp = _bt_prev.get("A", {}).get("predictor", "Random")

top_strip(
    [scope, ticker, "A/B/C", "5D", f"PREDICTOR {_bt_pred_disp}"],
    status_text="READY", status_tone="neutral",
)

section_header("CONFIG")
with panel():
    # ── Row 1: Predictor ─────────────────────────
    c1, c2 = st.columns([3, 1], gap="small")
    with c1:
        predictor = st.selectbox(
            "Predictor",
            list(backtest_service.PREDICTOR_OPTIONS),
            index=1,   # 0=Random, 1=RandomForest
            help=(
                "Random = 구조 검증용 랜덤  \n"
                "모델명 = Model Lab 의 12-fold OOS 예측을 백테스트 신호로 사용 "
                "(첫 실행 시 학습 5~10분)"
            ),
        )
    with c2:
        st.write("")
        run = st.button("▶  RUN ALL (A/B/C)", type="primary",
                        use_container_width=True)

    if predictor != "Random":
        st.warning(
            f"⚠️ **{predictor}** predictor 사용: 12-fold walk-forward OOS 예측을 "
            f"백테스트 신호로 씁니다. 첫 실행은 학습에 5~10분 걸립니다."
        )

    # ── Row 2: Period / Cost ─────────────────────
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.2], gap="small")
    with c1:
        start = st.date_input(
            "Start", value=pd.Timestamp("2023-01-01"),
        ).strftime("%Y-%m-%d")
    with c2:
        end = st.date_input(
            "End", value=pd.Timestamp("2024-08-22"),
        ).strftime("%Y-%m-%d")
    with c3:
        initial_cash = st.number_input("Initial Cash", 100.0, 1e7, 100.0, 100.0)
    with c4:
        trade_cost = st.number_input(
            "Trade Cost", 0.0, 0.01, 0.001, 0.0005, format="%.4f",
        )

if run:
    import traceback

    msg = f"A/B/C · {scope}:{ticker} · {predictor}"
    if predictor != "Random":
        msg += " · 학습 포함 (5~10분 예상)"

    with st.spinner(msg):
        try:
            results = backtest_service.run_all_strategies(
                scope=scope, ticker=ticker,
                start=start, end=end,
                initial_cash=initial_cash, trade_cost=trade_cost,
                predictor=predictor,
            )
            set_result("bt", scope, ticker, results)
            st.success(f"✅ 완료 · {scope}:{ticker} · predictor={predictor}")
            st.rerun()
        except Exception as e:
            st.error(f"실행 실패: {type(e).__name__}: {e}")
            with st.expander("Traceback", expanded=True):
                st.code(traceback.format_exc())

results = get_result("bt", scope, ticker)
if not results:
    st.caption(f"{scope}:{ticker} 결과가 아직 없습니다.")
    st.stop()

section_header(f"STRATEGY COMPARISON · {scope}:{ticker}")
rows = []
for s, r in results.items():
    m = r["metrics"]
    rows.append({
        "STRATEGY": s,
        "TOTAL RETURN": m["total_return"],
        "ANNUAL RETURN": m["annual_return"],
        "VOLATILITY": m["volatility"],
        "SHARPE": m["sharpe_ratio"],
        "MDD": -abs(m["max_drawdown"]),
        "WIN RATE": m["win_rate_daily"],
        "PROFIT FACTOR": m["profit_factor"],
        "TRADES": m["num_trades"],
        "FINAL VALUE": m["final_portfolio_value"],
    })
cmp_df = pd.DataFrame(rows)

with panel(f"{len(results)} STRATEGIES"):
    show_table(
        cmp_df,
        num_cols=["TOTAL RETURN", "ANNUAL RETURN", "VOLATILITY", "SHARPE",
                  "MDD", "WIN RATE", "PROFIT FACTOR", "FINAL VALUE"],
        precision=4,
    )

section_header("EQUITY CURVE")
with panel("A / B / C · PORTFOLIO VALUE"):
    fig = go.Figure()
    colors = {"A": "#7fd1ff", "B": "#4ade80", "C": "#fbbf24"}
    for s, r in results.items():
        eq = r["equity"]
        fig.add_trace(go.Scatter(
            x=eq.index, y=eq.values, name=f"Strategy {s}",
            line=dict(color=colors.get(s, "#9aa0a6"), width=1.6),
        ))
    fig.update_layout(height=380, xaxis=dict(title=""),
                      yaxis=dict(title="Portfolio Value"))
    plotly_chart(fig)

section_header("DRAWDOWN")
with panel("A / B / C"):
    fig = go.Figure()
    for s, r in results.items():
        eq = r["equity"]
        dd = eq / eq.cummax() - 1
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values, name=f"Strategy {s}",
            line=dict(color=colors.get(s, "#9aa0a6"), width=1.2),
            fill="tozeroy", fillcolor="rgba(0,0,0,0)",
        ))
    fig.update_layout(height=300, yaxis=dict(tickformat=".1%", title=""))
    plotly_chart(fig)

section_header("STRATEGY DETAIL")
pick = st.selectbox("Inspect", list(results.keys()), index=0)
m = results[pick]["metrics"]

metric_row([
    dict(label="TOTAL RETURN", value=f"{m['total_return']*100:.2f}%",
         tone="up" if m["total_return"] > 0 else "down"),
    dict(label="ANNUAL RETURN", value=f"{m['annual_return']*100:.2f}%",
         tone="up" if m["annual_return"] > 0 else "down"),
    dict(label="SHARPE", value=f"{m['sharpe_ratio']:.4f}",
         tone="up" if m["sharpe_ratio"] > 0 else "down"),
    dict(label="MDD", value=f"{-abs(m['max_drawdown'])*100:.2f}%", tone="down"),
    dict(label="WIN RATE", value=f"{m['win_rate_daily']*100:.2f}%"),
    dict(label="PROFIT FACTOR", value=f"{m['profit_factor']:.2f}"),
], cols=6)

metric_row([
    dict(label="AVG WIN", value=f"{m['avg_win']*100:.4f}%", tone="up"),
    dict(label="AVG LOSS", value=f"{m['avg_loss']*100:.4f}%", tone="down"),
    dict(label="MAX WIN STREAK", value=f"{m['max_win_streak']}"),
    dict(label="MAX LOSS STREAK", value=f"{m['max_loss_streak']}"),
    dict(label="TOTAL TRADES", value=f"{m['num_trades']}"),
    dict(label="TRANSACTION COST", value=f"{m['total_transaction_cost']:.4f}"),
], cols=6)