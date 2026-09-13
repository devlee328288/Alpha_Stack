# dashboard/app.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="AlphaStack", page_icon="📈", layout="wide")

import theme
from theme import (
    metric_row, section_header, top_strip, panel,
    show_table, plotly_chart, status_dot,
)
theme.inject()

from components import sidebar_controls
from services import baseline_service, model_service, comparison_service

sidebar_controls()

# ═══════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════
st.markdown('<div class="as-title">Overview</div>', unsafe_allow_html=True)

baseline = baseline_service.load_results()
models = model_service.load_results()
bt = st.session_state.get("_bt_results")
cost_grid = st.session_state.get("_cost_grid")

_ready = bool(baseline and models)
top_strip(
    ["ALPHASTACK", "KOSPI200", "COMBINATION E"],
    status_text="READY" if _ready else "INCOMPLETE",
    status_tone="up" if _ready else "warn",
)

# ═══════════════════════════════════════════════════════════
# ① STATUS ROW
# ═══════════════════════════════════════════════════════════
section_header("PIPELINE")

def _status_dot_line(label: str, ready: bool, detail: str = "") -> str:
    tone = "up" if ready else "neutral"
    status = "ready" if ready else "empty"
    return (
        f'<div style="display:flex;justify-content:space-between;'
        f'align-items:baseline;padding:4px 0;">'
        f'<span style="font-family:var(--font-ui);font-size:11px;'
        f'color:var(--text-secondary);">{label}</span>'
        f'<span style="font-family:var(--font-num);font-size:11px;'
        f'color:var(--text-primary);">{status_dot(tone)} {status}'
        f'  <span style="color:var(--text-muted);margin-left:6px;">{detail}</span>'
        f'</span>'
        f'</div>'
    )

with panel("4 STAGES", status_text="LIVE" if _ready else "PARTIAL",
           status_tone="up" if _ready else "warn"):
    def _ready(v) -> bool:
        if v is None:
            return False
        if hasattr(v, "empty") and hasattr(v, "shape"):
            try:
                return not bool(v.empty)
            except Exception:
                return False
        if hasattr(v, "__len__"):
            try:
                return len(v) > 0
            except Exception:
                return False
        return bool(v)

    _cost_n = len(cost_grid) if cost_grid is not None and hasattr(cost_grid, "__len__") else 0

    st.markdown(
        _status_dot_line("Baseline", _ready(baseline),
                         f"{baseline['total_folds']} folds" if baseline else "") +
        _status_dot_line("Model Lab", _ready(models),
                         f"{len(models)} models" if models else "") +
        _status_dot_line("Backtest", _ready(bt),
                         "A/B/C" if _ready(bt) else "") +
        _status_dot_line("Cost Sens", _ready(cost_grid),
                         f"{_cost_n} grid" if _ready(cost_grid) else ""),
        unsafe_allow_html=True,
    )

if not _ready:
    st.info(
        "**Baseline 과 Model Lab 페이지에서 각각 RUN 을 실행하세요.**  \n"
        "두 결과가 준비되면 이 Overview 가 자동으로 채워집니다."
    )
    st.stop()

# ═══════════════════════════════════════════════════════════
# ② TOP LINE
# ═══════════════════════════════════════════════════════════
best_name = comparison_service.best_model(models)
best_summary = models[best_name]["summary"]
b_perf = baseline.get("perf_metrics", {})
b_cls = baseline.get("cls_metrics", {})

section_header("TOP LINE · BEST MODEL vs BASELINE")

col_b, col_m = st.columns(2, gap="small")

with col_b:
    _age = baseline.get("run_at", "—")
    with panel(
        f"BASELINE · 6-PARAM · {baseline['threshold']*100:.0f}%",
        status_text=f"{baseline['total_folds']} FOLDS · {_age}",
        status_tone="accent",
    ):
        metric_row([
            dict(label="SHARPE", value=f"{b_perf.get('sharpe', 0):.4f}",
                 tone="up" if b_perf.get("sharpe", 0) > 0 else "down"),
            dict(label="CAGR", value=f"{b_perf.get('cagr', 0)*100:.2f}%",
                 tone="up" if b_perf.get("cagr", 0) > 0 else "down"),
            dict(label="MDD", value=f"{-abs(b_perf.get('mdd', 0))*100:.2f}%",
                 tone="down"),
            dict(label="MACRO F1", value=f"{b_cls.get('f1_macro', 0):.4f}"),
        ], cols=4)

with col_m:
    _age = models[best_name].get("run_at", "—")
    with panel(
        f"BEST MODEL · {best_name}",
        status_text=f"BEST · {_age}", status_tone="up",
    ):
        _ds = best_summary.get("delta_sharpe_net_median", 0)
        metric_row([
            dict(label="ACC", value=f"{best_summary['accuracy']:.4f}"),
            dict(label="MACRO F1", value=f"{best_summary['macro_f1']:.4f}"),
            dict(label="DOWN REC", value=f"{best_summary['down_recall']:.4f}"),
            dict(label="ΔSHARPE", value=f"{_ds:+.4f}",
                 tone="up" if _ds > 0 else "down"),
        ], cols=4)

# ═══════════════════════════════════════════════════════════
# ③ FOLD TIMELINE · ΔSHARPE per fold
# ═══════════════════════════════════════════════════════════
section_header(f"FOLD TIMELINE · {best_name}")

fold_df = pd.DataFrame(models[best_name]["fold_results"])
if not fold_df.empty and "delta_sharpe_net" in fold_df.columns:
    with panel(f"{len(fold_df)} FOLDS · ΔSHARPE (STRATEGY − BUY&HOLD)"):
        colors = [
            "#4ade80" if v > 0 else "#ff5c5c"
            for v in fold_df["delta_sharpe_net"]
        ]
        hover_text = [
            f"{row['valid_start']} ~ {row['valid_end']}"
            if "valid_start" in fold_df.columns else f"fold {int(row['fold'])}"
            for _, row in fold_df.iterrows()
        ]
        fig = go.Figure(go.Bar(
            x=[f"F{int(i)}" for i in fold_df["fold"]],
            y=fold_df["delta_sharpe_net"],
            marker=dict(color=colors),
            text=[f"{v:+.2f}" for v in fold_df["delta_sharpe_net"]],
            textposition="outside",
            hovertext=hover_text,
            hovertemplate="%{x}<br>%{hovertext}<br>ΔSharpe=%{y:.3f}<extra></extra>",
        ))
        fig.update_layout(
            height=280,
            showlegend=False,
            xaxis=dict(title=""),
            yaxis=dict(title="ΔSharpe", zeroline=True,
                       zerolinecolor="rgba(255,255,255,0.15)"),
            margin=dict(l=8, r=8, t=28, b=8),
        )
        plotly_chart(fig)

        # fold 요약 통계
        _pos = int((fold_df["delta_sharpe_net"] > 0).sum())
        _tot = len(fold_df)
        _med = float(fold_df["delta_sharpe_net"].median())
        _max = float(fold_df["delta_sharpe_net"].max())
        _min = float(fold_df["delta_sharpe_net"].min())

        metric_row([
            dict(label="POSITIVE FOLDS", value=f"{_pos}/{_tot}",
                 tone="up" if _pos > _tot / 2 else "warn"),
            dict(label="MEDIAN ΔSHP", value=f"{_med:+.4f}",
                 tone="up" if _med > 0 else "down"),
            dict(label="MAX", value=f"{_max:+.4f}", tone="up"),
            dict(label="MIN", value=f"{_min:+.4f}", tone="down"),
        ], cols=4)

# ═══════════════════════════════════════════════════════════
# ④ MODEL COMPARISON
# ═══════════════════════════════════════════════════════════
section_header(f"MODEL COMPARISON · {len(models)} MODELS")
cmp_df = comparison_service.build_comparison_table(models)

with panel(
    "12-FOLD OOS · SORT BY HARMONIC",
    status_text=f"BEST · {best_name}", status_tone="up",
):
    display = cmp_df.rename(columns=comparison_service.COMPARISON_COLUMNS)
    show_table(
        display,
        num_cols=["ACC", "MACRO F1", "DOWN RECALL", "HARMONIC",
                  "BAL ACC", "MAJORITY", "ΔSHARPE"],
        precision=4,
        highlight_row=0,
    )

# ═══════════════════════════════════════════════════════════
# ⑤ BACKTEST + COST (있을 때만)
# ═══════════════════════════════════════════════════════════
if bt:
    section_header("BACKTEST · A/B/C")
    with panel("3 STRATEGIES"):
        bt_rows = []
        for s, r in bt.items():
            m = r["metrics"]
            bt_rows.append({
                "STRATEGY": s,
                "TOTAL RETURN": m["total_return"],
                "ANNUAL RETURN": m["annual_return"],
                "SHARPE": m["sharpe_ratio"],
                "MDD": -abs(m["max_drawdown"]),
                "TRADES": m["num_trades"],
                "FINAL VALUE": m["final_portfolio_value"],
            })
        show_table(
            pd.DataFrame(bt_rows),
            num_cols=["TOTAL RETURN", "ANNUAL RETURN", "SHARPE",
                      "MDD", "FINAL VALUE"],
            precision=4,
        )

if cost_grid is not None:
    section_header("COST · BREAKEVEN")
    be = st.session_state.get("_cost_be")
    if be is not None and not be.empty:
        be_rows = be.to_dict("records")
        metric_row([
            dict(label=f"STRATEGY {r['strategy']}",
                 value=f"{r['breakeven_cost']*100:.3f}%",
                 tone="up" if r["breakeven_cost"] > 0.002 else "warn",
                 accent=True)
            for r in be_rows
        ], cols=len(be_rows))
    else:
        st.caption("Cost Sensitivity 페이지에서 RUN 한 후 표시됩니다.")

# ═══════════════════════════════════════════════════════════
# ⑥ RECENT OOS PREDICTIONS · BEST MODEL
# ═══════════════════════════════════════════════════════════
_oos = models[best_name].get("oos_predictions")
if _oos:
    oos_df = pd.DataFrame(_oos).tail(20).copy()
    # 라벨 변환
    label_map = {-1: "DOWN", 0: "NEUTRAL", 1: "UP"}
    oos_df["actual_label"] = oos_df["actual"].map(label_map)
    oos_df["pred_label"] = oos_df["predicted"].map(label_map)
    oos_df["correct"] = oos_df["actual"] == oos_df["predicted"]

    section_header(f"RECENT OOS · {best_name} · LAST 20")

    with panel("LAST 20 DAYS"):
        view = oos_df[[
            "bas_dd", "actual_label", "pred_label",
            "p_down", "p_neutral", "p_up", "correct",
        ]].rename(columns={
            "bas_dd": "DATE",
            "actual_label": "ACTUAL",
            "pred_label": "PRED",
            "p_down": "P(DN)",
            "p_neutral": "P(NT)",
            "p_up": "P(UP)",
            "correct": "✓",
        })
        show_table(
            view.iloc[::-1].reset_index(drop=True),
            num_cols=["P(DN)", "P(NT)", "P(UP)"],
            precision=3,
        )

        _acc20 = float(oos_df["correct"].mean())
        metric_row([
            dict(label="ACCURACY (LAST 20)",
                 value=f"{_acc20*100:.1f}%",
                 tone="up" if _acc20 > 0.4 else "down"),
            dict(label="UP PRED", value=f"{int((oos_df['predicted'] == 1).sum())}"),
            dict(label="NEUTRAL PRED", value=f"{int((oos_df['predicted'] == 0).sum())}"),
            dict(label="DOWN PRED", value=f"{int((oos_df['predicted'] == -1).sum())}"),
        ], cols=4)

# ═══════════════════════════════════════════════════════════
# ⑦ NEXT STEPS (결과 없을 때만)
# ═══════════════════════════════════════════════════════════
if not bt:
    section_header("NEXT")
    with panel("RECOMMENDED"):
        st.markdown("""
- **Backtest** — A/B/C 전략 백테스트 (실행 후 이 Overview 에 요약 표시)
- **Cost Sensitivity** — 4가지 비용 프리셋 × 3 전략 (breakeven 자동 계산)
""")