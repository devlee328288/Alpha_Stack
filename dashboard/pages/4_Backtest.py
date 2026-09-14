import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Backtest · AlphaStack", layout="wide")

import theme
from theme import (
    metric_row,
    section_header,
    top_strip,
    panel,
    show_table,
    plotly_chart,
)

theme.inject()

from components import sidebar_controls
from services import backtest_service
from services.experiment_config import ExperimentConfig
from state import ctx, get_result, set_result

sidebar_controls()
c = ctx()
scope = c["scope"]
ticker = c["ticker"]

# ── UNIVERSE scope 는 전용 페이지에서 ──
if scope == "UNIVERSE":
    st.markdown(
        '<div class="as-title">전용 페이지로 이동</div>', unsafe_allow_html=True
    )
    st.info(
        "**UNIVERSE scope 는 Universe 페이지에서만 사용합니다.**  \n"
        "사이드바에서 **UNIVERSE** 버튼을 다시 누르거나, "
        "페이지 목록의 **Universe** 를 여세요."
    )
    if st.button("▶  Universe 페이지로", type="primary"):
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
_bt_base_disp = "—"
if _bt_prev:
    _bt_pred_disp = _bt_prev.get("A", {}).get("predictor", "Random")
    _bt_base_disp = _bt_prev.get("A", {}).get("baseline_kind", "fwd_return")

top_strip(
    [
        scope,
        ticker,
        "A/B/C · D/E/F(5D LOCK)",
        "5D",
        f"MODEL {_bt_pred_disp}",
        f"BASE {_bt_base_disp}",
    ],
    status_text="READY",
    status_tone="neutral",
)

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════
section_header("CONFIG")
with panel():
    # ── Row 1: 모델(4) × Baseline(2) + RUN ─────────
    c1, c2, c3 = st.columns([2, 2, 1], gap="small")

    with c1:
        model_name = st.selectbox(
            "① Model  ·  AI 모델 선택 (4종)",
            list(backtest_service.MODEL_OPTIONS),
            index=0,
            help=(
                "Model Lab 12-fold walk-forward OOS 예측을 백테스트 신호로 사용.  \n"
                "첫 실행 시 학습에 5~10분 소요."
            ),
        )

    with c2:
        baseline_kind = st.selectbox(
            "② Baseline  ·  판단 기준 (2종)",
            list(backtest_service.BASELINE_OPTIONS),
            index=0,
            format_func=lambda x: backtest_service.BASELINE_LABELS[x],
            help=(
                "**fwd_return** = 미래 5일 수익률 ±1% 상수 밴드  \n"
                "**adaptive**  = Baseline per-fold 최적 6-param 밴드"
            ),
        )

    with c3:
        st.write("")
        run = st.button(
            "▶  RUN ALL (A~F)",
            type="primary",
            use_container_width=True,
        )

    st.caption(
        f"ℹ️ 선택: **{model_name}** × "
        f"**{backtest_service.BASELINE_LABELS[baseline_kind]}** "
        f"→ 이 조합으로 모델 학습 후 OOS 예측을 매매 신호로 사용합니다."
    )
    st.caption(
        "🔒 **A / B / C** = 매일 매매  ·  "
        "**D / E / F** = A/B/C + **5거래일 락** (그 사이 신규·청산 모두 금지)"
    )

    if baseline_kind == "adaptive":
        _bl = get_result("baseline", scope, ticker)
        if _bl is None:
            st.warning(
                "⚠️ **Adaptive** 는 Baseline 결과가 필요합니다. "
                "**Baseline 페이지에서 RUN BASELINE** 을 먼저 실행하세요."
            )
        else:
            st.info(
                f"**Adaptive 6-param** · Baseline {_bl['total_folds']}개 fold의 "
                f"per-fold 최적 밴드로 라벨을 생성합니다."
            )
    else:
        st.info(
            "**fwd_return ±1%** · 미래 5일 수익률 상수 밴드로 라벨을 생성합니다. "
            "별도 사전 실행 없이 즉시 사용 가능."
        )

    # ── Row 2: Period / Cash / Cost ─────────────────
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.2], gap="small")
    with c1:
        start = st.date_input(
            "Start",
            value=pd.Timestamp("2010-01-01"),
        ).strftime("%Y-%m-%d")
    with c2:
        end = st.date_input(
            "End",
            value=pd.Timestamp("2025-01-01"),
        ).strftime("%Y-%m-%d")
    with c3:
        initial_cash = st.number_input("Initial Cash", 100.0, 1e7, 100.0, 100.0)
    with c4:
        trade_cost = st.number_input(
            "Trade Cost",
            0.0,
            0.01,
            0.001,
            0.0005,
            format="%.4f",
        )

    # ── 공용 실험 설정 (SSOT) ─────────────────────
    exp_config = ExperimentConfig(
        scope=scope,
        ticker=ticker,
        start=start,
        end=end,
        source="stocks30",
        predictor=model_name,
        model_revision="v1",
        baseline_kind=baseline_kind,
        initial_cash=float(initial_cash),
        trade_cost=float(trade_cost),
        seed=None,
    )

    st.divider()
    section_header("EXPERIMENT CONFIG · SSOT")
    show_table(pd.DataFrame(exp_config.summary_rows()), num_cols=[], precision=4)
    st.caption(
        "ℹ️ **Cost Sensitivity 페이지가 이 설정을 그대로 사용합니다.** "
        "Cost Sens 는 이 설정에서 `trade_cost` 만 바꿔 실행합니다."
    )

# ═══════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════
if run:
    import traceback

    msg = (
        f"A~F · {scope}:{ticker} · "
        f"{model_name} × {backtest_service.BASELINE_LABELS[baseline_kind]}"
    )
    msg += " · 학습 포함 (5~10분 예상)"

    with st.spinner(msg):
        try:
            results = backtest_service.run_all_strategies(config=exp_config)
            set_result("bt", scope, ticker, results)
            # ★ Backtest 설정 저장 (Cost Sens 가 읽음)
            st.session_state[f"_bt_cfg_{scope}:{ticker}"] = exp_config.to_dict()
            st.success(
                f"✅ 완료 · {scope}:{ticker} · " f"{model_name} × {baseline_kind}"
            )
            st.rerun()
        except Exception as e:
            st.error(f"실행 실패: {type(e).__name__}: {e}")
            with st.expander("Traceback", expanded=True):
                st.code(traceback.format_exc())

# ═══════════════════════════════════════════════════════════
# RESULT
# ═══════════════════════════════════════════════════════════
results = get_result("bt", scope, ticker)
if not results:
    st.caption(f"{scope}:{ticker} 결과가 아직 없습니다. RUN 을 누르세요.")
    st.stop()

# ── 저장된 설정 표시 ─────────────────────────────────
_saved_cfg = st.session_state.get(f"_bt_cfg_{scope}:{ticker}")
if _saved_cfg:
    with st.expander("📌 LAST EXECUTED CONFIG (Cost Sens 기준)", expanded=False):
        show_table(
            pd.DataFrame([{"KEY": k, "VALUE": v} for k, v in _saved_cfg.items()]),
            num_cols=[],
            precision=4,
        )

# ── 요약 배너 ────────────────────────────────────────
_first = next(iter(results.values()))
section_header(
    f"STRATEGY COMPARISON · {scope}:{ticker} · "
    f"{_first.get('predictor', '—')} × {_first.get('baseline_kind', '—')}"
)

rows = []
for s, r in results.items():
    m = r["metrics"]
    rows.append(
        {
            "STRATEGY": s,
            "BASE": r.get("base_strategy", s),
            "LOCK": r.get("cooldown_days", 0),
            "TOTAL RETURN": m["total_return"],
            "ANNUAL RETURN": m["annual_return"],
            "VOLATILITY": m["volatility"],
            "SHARPE": m["sharpe_ratio"],
            "MDD": -abs(m["max_drawdown"]),
            "WIN RATE": m["win_rate_daily"],
            "PROFIT FACTOR": m["profit_factor"],
            "TRADES": m["num_trades"],
            "FINAL VALUE": m["final_portfolio_value"],
        }
    )
cmp_df = pd.DataFrame(rows)

with panel(f"{len(results)} STRATEGIES"):
    show_table(
        cmp_df,
        num_cols=[
            "LOCK",
            "TOTAL RETURN",
            "ANNUAL RETURN",
            "VOLATILITY",
            "SHARPE",
            "MDD",
            "WIN RATE",
            "PROFIT FACTOR",
            "FINAL VALUE",
        ],
        precision=4,
    )

# ── 색상 / 선 스타일 ────────────────────────────────
_colors = {
    "A": "#7fd1ff",
    "B": "#4ade80",
    "C": "#fbbf24",
    "D": "#7fd1ff",
    "E": "#4ade80",
    "F": "#fbbf24",
}
_dash = {
    "A": "solid",
    "B": "solid",
    "C": "solid",
    "D": "dash",
    "E": "dash",
    "F": "dash",
}

section_header("EQUITY CURVE")
with panel("A/B/C (실선)  ·  D/E/F (5거래일 락, 점선)"):
    fig = go.Figure()
    for s, r in results.items():
        eq = r["equity"]
        fig.add_trace(
            go.Scatter(
                x=eq.index,
                y=eq.values,
                name=f"Strategy {s}",
                line=dict(
                    color=_colors.get(s, "#9aa0a6"),
                    width=1.6,
                    dash=_dash.get(s, "solid"),
                ),
            )
        )
    fig.update_layout(
        height=420,
        xaxis=dict(title=""),
        yaxis=dict(title="Portfolio Value"),
    )
    plotly_chart(fig)

section_header("DRAWDOWN")
with panel("A/B/C (실선)  ·  D/E/F (점선)"):
    fig = go.Figure()
    for s, r in results.items():
        eq = r["equity"]
        dd = eq / eq.cummax() - 1
        fig.add_trace(
            go.Scatter(
                x=dd.index,
                y=dd.values,
                name=f"Strategy {s}",
                line=dict(
                    color=_colors.get(s, "#9aa0a6"),
                    width=1.2,
                    dash=_dash.get(s, "solid"),
                ),
                fill="tozeroy",
                fillcolor="rgba(0,0,0,0)",
            )
        )
    fig.update_layout(height=340, yaxis=dict(tickformat=".1%", title=""))
    plotly_chart(fig)

section_header("STRATEGY DETAIL")
pick = st.selectbox("Inspect", list(results.keys()), index=0)
_r = results[pick]
m = _r["metrics"]

# ── 락 정보 배너 ────────────────────────────────
_lock = _r.get("cooldown_days", 0)
_base = _r.get("base_strategy", pick)
if _lock > 0:
    st.info(
        f"🔒 **Strategy {pick}** = Strategy {_base} + **{_lock}거래일 락** "
        f"(실제 거래 발생 시점부터 5거래일간 신규·청산 모두 금지)"
    )
else:
    st.caption(f"Strategy {pick} · 매일 매매 (락 없음)")

metric_row(
    [
        dict(
            label="TOTAL RETURN",
            value=f"{m['total_return']*100:.2f}%",
            tone="up" if m["total_return"] > 0 else "down",
        ),
        dict(
            label="ANNUAL RETURN",
            value=f"{m['annual_return']*100:.2f}%",
            tone="up" if m["annual_return"] > 0 else "down",
        ),
        dict(
            label="SHARPE",
            value=f"{m['sharpe_ratio']:.4f}",
            tone="up" if m["sharpe_ratio"] > 0 else "down",
        ),
        dict(label="MDD", value=f"{-abs(m['max_drawdown'])*100:.2f}%", tone="down"),
        dict(label="WIN RATE", value=f"{m['win_rate_daily']*100:.2f}%"),
        dict(label="PROFIT FACTOR", value=f"{m['profit_factor']:.2f}"),
    ],
    cols=6,
)

metric_row(
    [
        dict(label="AVG WIN", value=f"{m['avg_win']*100:.4f}%", tone="up"),
        dict(label="AVG LOSS", value=f"{m['avg_loss']*100:.4f}%", tone="down"),
        dict(label="MAX WIN STREAK", value=f"{m['max_win_streak']}"),
        dict(label="MAX LOSS STREAK", value=f"{m['max_loss_streak']}"),
        dict(label="TOTAL TRADES", value=f"{m['num_trades']}"),
        dict(
            label="TRANSACTION COST",
            value=f"{m['total_transaction_cost']:.4f}",
        ),
    ],
    cols=6,
)
