import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Cost Sensitivity · AlphaStack", layout="wide")

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

st.markdown('<div class="as-title">Cost Sensitivity</div>', unsafe_allow_html=True)

if not backtest_service.cost_available():
    top_strip([scope, ticker], status_text="IMPORT FAIL", status_tone="warn")
    st.error(backtest_service.engine_status())
    st.stop()

# ═══════════════════════════════════════════════════════════
# CONFIG · SSOT (Backtest 에서 넘어옴)
# ═══════════════════════════════════════════════════════════
section_header("CONFIG · SSOT (from Backtest)")

_cfg_key = f"_bt_cfg_{scope}:{ticker}"
_cfg_dict = st.session_state.get(_cfg_key)

if _cfg_dict is None:
    top_strip([scope, ticker], status_text="NO CONFIG", status_tone="warn")
    st.warning(
        f"⚠️ **{scope}:{ticker}** 에 대한 Backtest 설정이 없습니다.\n\n"
        "**Backtest 페이지에서 RUN ALL 을 먼저 실행하세요.**  \n"
        "Cost Sens 는 Backtest 와 **동일 조건**에서 비용만 바꿔 실행하는 페이지입니다."
    )
    if st.button("▶  Backtest 페이지로", type="primary"):
        st.switch_page("pages/4_Backtest.py")
    st.stop()

try:
    exp_config = ExperimentConfig(**_cfg_dict)
except Exception as e:
    st.error(f"Config 복원 실패: {type(e).__name__}: {e}")
    st.stop()

with panel("EXPERIMENT CONFIG (Backtest 와 동일)"):
    show_table(
        pd.DataFrame(exp_config.summary_rows()),
        num_cols=[],
        precision=4,
    )
    st.caption(
        "ℹ️ Backtest 와 **동일한 예측·신호**를 사용합니다. "
        "비용(`trade_cost`) 프리셋만 여러 값으로 바꿔 체결·회계를 재계산합니다."
    )

# ── RUN ──────────────────────────────────────────────
c1, c2 = st.columns([1, 3], gap="small")
with c1:
    run = st.button("▶  RUN Cost Grid", type="primary", use_container_width=True)
with c2:
    st.caption(
        f"대상: **4 COST PRESETS × {len(backtest_service.STRATEGIES)} STRATEGIES** "
        f"· predictor **{exp_config.predictor}** "
        f"· baseline **{exp_config.baseline_kind}**"
    )

# ── HEADER STRIP ─────────────────────────────────────
top_strip(
    [
        scope,
        ticker,
        f"4 PRESETS × {len(backtest_service.STRATEGIES)} STRATEGIES",
        f"PRED {exp_config.predictor}",
        f"BASE {exp_config.baseline_kind}",
    ],
    status_text="READY",
    status_tone="neutral",
)

# ═══════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════
if run:
    import traceback

    msg = f"cost grid · {scope}:{ticker} · {exp_config.predictor}"
    if exp_config.predictor != "Random":
        msg += " · 학습 캐시 사용"

    with st.spinner(msg):
        try:
            grid = backtest_service.run_cost_grid(exp_config)
            be = backtest_service.run_breakeven(exp_config)

            set_result("cost_grid", scope, ticker, grid)
            set_result("cost_be", scope, ticker, be)
            # 결과가 어떤 config 로 만들어졌는지 기록
            st.session_state[f"_cost_cfg_{scope}:{ticker}"] = exp_config.to_dict()
            st.success(f"✅ 완료 · {scope}:{ticker} · {exp_config.predictor}")
            st.rerun()
        except Exception as e:
            st.error(f"실행 실패: {type(e).__name__}: {e}")
            with st.expander("Traceback", expanded=True):
                st.code(traceback.format_exc())

# ═══════════════════════════════════════════════════════════
# RESULT
# ═══════════════════════════════════════════════════════════
grid = get_result("cost_grid", scope, ticker)
be = get_result("cost_be", scope, ticker)

if grid is None or be is None:
    st.caption(f"{scope}:{ticker} Cost Sens 결과가 아직 없습니다. RUN 을 누르세요.")
    st.stop()

# ── 결과가 오래된 config 로 만들어졌는지 체크 ────────
_cost_cfg = st.session_state.get(f"_cost_cfg_{scope}:{ticker}")
if _cost_cfg is not None and _cost_cfg != exp_config.to_dict():
    st.warning(
        "⚠️ Backtest 설정이 변경되었습니다. " "**RUN Cost Grid 를 다시 실행**해 주세요."
    )

# ═══════════════════════════════════════════════════════════
# BREAKEVEN
# ═══════════════════════════════════════════════════════════
section_header("BREAKEVEN COST · SHARPE = 0")
be_rows = be.to_dict("records") if hasattr(be, "to_dict") else be

metric_row(
    [
        dict(
            label=f"STRATEGY {r['strategy']}"
            + (f" ({r['base_strategy']}+🔒)" if r.get("cooldown_days", 0) > 0 else ""),
            value=(
                f"{r['breakeven_cost']*100:.3f}%"
                if pd.notna(r.get("breakeven_cost"))
                else "—"
            ),
            tone="up" if (r.get("breakeven_cost") or 0) > 0.002 else "warn",
            accent=True,
        )
        for r in be_rows
    ],
    cols=len(be_rows),
)

# ═══════════════════════════════════════════════════════════
# GRID TABLE
# ═══════════════════════════════════════════════════════════
section_header(f"COST GRID · 4 PRESETS × {len(backtest_service.STRATEGIES)} STRATEGIES")
with panel("SHARPE · CAGR · MDD"):
    disp = grid.copy()
    disp["cost_pct"] = (disp["cost_rate"] * 100).round(3).astype(str) + "%"
    disp = disp[
        [
            "strategy",
            "base_strategy",
            "cooldown_days",
            "cost_label",
            "cost_pct",
            "sharpe",
            "cagr",
            "mdd",
            "total_return",
            "num_trades",
        ]
    ]
    show_table(
        disp,
        num_cols=["cooldown_days", "sharpe", "cagr", "mdd", "total_return"],
        precision=4,
    )

# ═══════════════════════════════════════════════════════════
# SHARPE vs COST
# ═══════════════════════════════════════════════════════════
section_header("SHARPE vs COST")

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

with panel("STRATEGY A~F  ·  A/B/C 실선 · D/E/F 점선 (5일 락)"):
    fig = go.Figure()
    for s in sorted(grid["strategy"].unique()):
        sub = grid[grid["strategy"] == s].sort_values("cost_rate")
        fig.add_trace(
            go.Scatter(
                x=sub["cost_rate"] * 100,
                y=sub["sharpe"],
                name=f"Strategy {s}",
                mode="lines+markers",
                line=dict(
                    color=_colors.get(s, "#9aa0a6"),
                    width=2,
                    dash=_dash.get(s, "solid"),
                ),
                marker=dict(size=8),
            )
        )
    # breakeven 수직선 (점선, 전략별 색)
    for r in be_rows:
        bc = r.get("breakeven_cost")
        if bc is None or pd.isna(bc):
            continue
        fig.add_vline(
            x=bc * 100,
            line_dash="dot",
            line_color=_colors.get(r["strategy"], "#9aa0a6"),
            opacity=0.4,
        )
    fig.update_layout(
        height=420,
        xaxis=dict(title="Cost (%)"),
        yaxis=dict(title="Sharpe"),
        hovermode="x unified",
    )
    plotly_chart(fig)

# ═══════════════════════════════════════════════════════════
# DETAIL BY STRATEGY
# ═══════════════════════════════════════════════════════════
section_header("DETAIL BY STRATEGY")
pick = st.selectbox("Strategy", sorted(grid["strategy"].unique()), index=0)
sub = grid[grid["strategy"] == pick].sort_values("cost_rate").copy()
sub["cost_pct"] = (sub["cost_rate"] * 100).round(3).astype(str) + "%"

_pick_row = next((r for r in be_rows if r["strategy"] == pick), None)
if _pick_row and _pick_row.get("cooldown_days", 0) > 0:
    st.info(
        f"🔒 Strategy {pick} = Strategy {_pick_row['base_strategy']} + "
        f"**{_pick_row['cooldown_days']}거래일 락**"
    )

with panel(f"STRATEGY {pick}"):
    show_table(
        sub[
            [
                "cost_pct",
                "cost_label",
                "sharpe",
                "cagr",
                "mdd",
                "total_return",
                "num_trades",
            ]
        ],
        num_cols=["sharpe", "cagr", "mdd", "total_return"],
        precision=4,
    )
