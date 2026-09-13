import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="QFRS · AlphaStack", layout="wide")

import theme
from theme import (
    metric_row, section_header, top_strip, panel,
    show_table, plotly_chart,
)
theme.inject()

from components import sidebar_controls
from services import risk_service
from state import ctx, get_result

sidebar_controls()
c = ctx()
scope = c["scope"]
ticker = c["ticker"]

# ═══════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════
st.markdown('<div class="as-title">QFRS · Quant Research Summary</div>',
            unsafe_allow_html=True)

bt = get_result("bt", scope, ticker)
models = get_result("model_lab", scope, ticker)

if not bt and not models:
    top_strip([scope, ticker], status_text="NO DATA", status_tone="warn")
    st.warning(
        f"**Backtest** 또는 **Model Lab** 페이지에서 {scope}:{ticker} 결과를 먼저 만들어주세요.\n\n"
        "- RISK 탭: Backtest 필요\n"
        "- BACKTEST 탭: Backtest signal_log 또는 Model Lab oos_predictions 필요"
    )
    st.stop()

top_strip(
    [scope, ticker, "QFRS", "RISK + PERFORMANCE"],
    status_text="LIVE", status_tone="up",
)

# ═══════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════
tab_risk, tab_bt = st.tabs(["RISK", "BACKTEST"])

# ═══════════════════════════════════════════════════════════
# ═══ TAB 1 · RISK (QFRS Score) ═══════════════════════════════
# ═══════════════════════════════════════════════════════════
with tab_risk:
    if not bt:
        st.info("Backtest 페이지에서 RUN ALL 을 먼저 실행하세요.")
        st.stop()

    # ── Source & Params ────────────────────────────────
    section_header("SOURCE & PARAMS")

    with panel():
        c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1], gap="small")
        with c1:
            pick = st.selectbox(
                "Backtest Strategy", list(bt.keys()),
                index=0, key="_qfrs_pick",
            )
        with c2:
            rf = st.number_input(
                "Rf", 0.0, 0.10, 0.02, 0.005,
                format="%.4f", key="_qfrs_rf",
            )
        with c3:
            periods = st.number_input(
                "Periods/yr", 52, 252, 252, 1, key="_qfrs_periods",
            )
        with c4:
            n_trials = st.number_input(
                "n_trials", 1, 1000, 50, 1, key="_qfrs_trials",
            )
        with c5:
            top_k = st.number_input(
                "top_k MDD", 1, 10, 3, 1, key="_qfrs_topk",
            )

    returns = bt[pick]["daily_returns"].to_numpy()
    source_label = f"Strategy {pick} · {scope}:{ticker}"

    # ── Calculate QFRS ─────────────────────────────────
    try:
        m = risk_service.calculate_risk(
            returns,
            risk_free_rate=float(rf),
            periods_per_year=int(periods),
            n_trials=int(n_trials),
            sterling_top_k=int(top_k),
        )
    except Exception as e:
        st.error(f"Risk 계산 실패: {type(e).__name__}: {e}")
        st.stop()

    from scipy.stats import skew, kurtosis

    def _s(v, default=np.nan):
        try:
            fv = float(v)
            return fv if np.isfinite(fv) else default
        except Exception:
            return default

    _cagr = _s(np.prod(1 + returns) ** (252 / max(len(returns), 1)) - 1)
    _vol = _s(np.std(returns, ddof=1) * np.sqrt(252))
    _skw = _s(skew(returns))
    _krt = _s(kurtosis(returns, fisher=False))
    _v95 = _s(np.percentile(returns, 5))
    _cv95 = _s(returns[returns <= _v95].mean()) if _v95 == _v95 else np.nan
    _win = _s((returns > 0).mean())
    _gain = returns[returns > 0].sum()
    _loss = abs(returns[returns < 0].sum())
    _pf = _s(_gain / _loss) if _loss > 0 else np.inf

    _mws = _mls = _cw = _cl = 0
    for r in returns:
        if r > 0:
            _cw += 1; _cl = 0; _mws = max(_mws, _cw)
        elif r < 0:
            _cl += 1; _cw = 0; _mls = max(_mls, _cl)

    _mdd = m.get("MDD")
    _shp = m.get("Sharpe Ratio")
    _srt = m.get("Sortino Ratio")
    _clm = m.get("Calmar Ratio")
    _stl = m.get("Sterling Ratio")
    _dsr = m.get("Deflated Sharpe Ratio")

    # ── Verdict 함수들 ─────────────────────────────────
    def _v_sharpe(v):
        if v is None or v != v: return ("—", "neutral", 0)
        if v > 1.5: return ("STRONG", "up", 100)
        if v > 1.0: return ("GOOD", "up", 80)
        if v > 0.5: return ("MODERATE", "neutral", 60)
        if v > 0: return ("WEAK", "warn", 40)
        return ("FAIL", "down", 10)

    def _v_sortino(v):
        if v is None or v != v: return ("—", "neutral", 0)
        if v > 2.0: return ("STRONG", "up", 100)
        if v > 1.5: return ("GOOD", "up", 80)
        if v > 0.7: return ("MODERATE", "neutral", 60)
        if v > 0: return ("WEAK", "warn", 40)
        return ("FAIL", "down", 10)

    def _v_mdd(v):
        if v is None or v != v: return ("—", "neutral", 0)
        a = abs(v)
        if a < 0.05: return ("EXCELLENT", "up", 100)
        if a < 0.10: return ("LOW", "up", 80)
        if a < 0.20: return ("MODERATE", "neutral", 60)
        if a < 0.35: return ("HIGH", "warn", 40)
        return ("SEVERE", "down", 10)

    def _v_calmar(v):
        if v is None or v != v: return ("—", "neutral", 0)
        if v > 1.0: return ("STRONG", "up", 100)
        if v > 0.5: return ("GOOD", "up", 80)
        if v > 0.2: return ("MODERATE", "neutral", 60)
        if v > 0: return ("WEAK", "warn", 40)
        return ("FAIL", "down", 10)

    def _v_sterling(v):
        if v is None or v != v: return ("—", "neutral", 0)
        if v > 0.8: return ("STRONG", "up", 100)
        if v > 0.4: return ("GOOD", "up", 80)
        if v > 0.2: return ("MODERATE", "neutral", 60)
        if v > 0: return ("WEAK", "warn", 40)
        return ("FAIL", "down", 10)

    def _v_dsr(v):
        if v is None or v != v: return ("—", "neutral", 0)
        if v > 0.95: return ("PASS", "up", 100)
        if v > 0.80: return ("LIKELY", "up", 75)
        if v > 0.50: return ("PARTIAL", "warn", 50)
        return ("FAIL", "down", 20)

    _verdicts = {
        "Sharpe":   _v_sharpe(_shp),
        "Sortino":  _v_sortino(_srt),
        "MDD":      _v_mdd(_mdd),
        "Calmar":   _v_calmar(_clm),
        "Sterling": _v_sterling(_stl),
        "DSR":      _v_dsr(_dsr),
    }

    _qfrs_score = int(np.mean([v[2] for v in _verdicts.values()]))

    if _qfrs_score >= 80: _qt = "up"; _ql = "EXCELLENT"
    elif _qfrs_score >= 60: _qt = "up"; _ql = "GOOD"
    elif _qfrs_score >= 40: _qt = "warn"; _ql = "MODERATE"
    else: _qt = "down"; _ql = "POOR"

    # ═══════════════════════════════════════════════════
    # SCORE CARD
    # ═══════════════════════════════════════════════════
    section_header("QFRS SCORE CARD")

    with panel(source_label,
               status_text=f"{_ql} · {_qfrs_score}/100",
               status_tone=_qt):
        c1, c2 = st.columns([1, 2], gap="small")

        with c1:
            _color = {"up": "#4ade80", "warn": "#fbbf24",
                      "down": "#ff5c5c", "neutral": "#9aa0a6"}[_qt]
            st.markdown(
                f'<div style="text-align:center;padding:20px 0;">'
                f'  <div style="font-family:var(--font-num);font-size:64px;'
                f'              font-weight:600;color:{_color};line-height:1;">'
                f'    {_qfrs_score}'
                f'  </div>'
                f'  <div style="font-family:var(--font-num);font-size:14px;'
                f'              color:var(--text-muted);margin-top:4px;">/ 100</div>'
                f'  <div style="font-family:var(--font-ui);font-size:11px;'
                f'              font-weight:500;letter-spacing:0.14em;'
                f'              color:{_color};margin-top:12px;">{_ql}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with c2:
            for name, (label, tone, score) in _verdicts.items():
                _tc = {"up": "#4ade80", "warn": "#fbbf24",
                       "down": "#ff5c5c", "neutral": "#9aa0a6"}[tone]
                st.markdown(
                    f'<div style="padding:4px 0;">'
                    f'  <div style="display:flex;justify-content:space-between;'
                    f'              align-items:baseline;margin-bottom:3px;">'
                    f'    <span style="font-family:var(--font-ui);font-size:11px;'
                    f'                 color:var(--text-secondary);">{name}</span>'
                    f'    <span style="font-family:var(--font-num);font-size:11px;'
                    f'                 color:{_tc};">{label}</span>'
                    f'  </div>'
                    f'  <div style="height:4px;background:var(--bg-elevated);'
                    f'              border-radius:2px;overflow:hidden;">'
                    f'    <div style="height:100%;width:{score}%;'
                    f'                background:{_tc};"></div>'
                    f'  </div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ═══════════════════════════════════════════════════
    # CORE 6 METRICS
    # ═══════════════════════════════════════════════════
    section_header("CORE 6 METRICS")

    metric_row([
        dict(label="MDD",
             value=f"{-abs(_mdd or 0)*100:.2f}%",
             tone=_verdicts["MDD"][1]),
        dict(label="SHARPE",
             value=f"{_shp or 0:.4f}",
             tone=_verdicts["Sharpe"][1]),
        dict(label="SORTINO",
             value=f"{_srt or 0:.4f}",
             tone=_verdicts["Sortino"][1]),
        dict(label="CALMAR",
             value=f"{_clm or 0:.4f}",
             tone=_verdicts["Calmar"][1]),
        dict(label="STERLING",
             value=f"{_stl or 0:.4f}",
             tone=_verdicts["Sterling"][1]),
        dict(label="DSR",
             value=f"{(_dsr or 0)*100:.2f}%",
             tone=_verdicts["DSR"][1]),
    ], cols=6)

    # ═══════════════════════════════════════════════════
    # RADAR
    # ═══════════════════════════════════════════════════
    section_header("RADAR · NORMALIZED 0-100")

    cats = list(_verdicts.keys())
    vals = [_verdicts[k][2] for k in cats]
    _cats = cats + [cats[0]]
    _vals = vals + [vals[0]]

    with panel("6 METRICS · VERDICT SCORE"):
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=_vals, theta=_cats,
            fill="toself",
            line=dict(color="#7fd1ff", width=2),
            fillcolor="rgba(127,209,255,0.20)",
        ))
        fig.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(
                    visible=True, range=[0, 100],
                    gridcolor="rgba(255,255,255,0.06)",
                    tickfont=dict(size=9, color="#9aa0a6"),
                ),
                angularaxis=dict(
                    gridcolor="rgba(255,255,255,0.06)",
                    tickfont=dict(size=10, color="#9aa0a6"),
                ),
            ),
            height=420,
            showlegend=False,
            margin=dict(l=60, r=60, t=40, b=40),
        )
        plotly_chart(fig)

    # ═══════════════════════════════════════════════════
    # VERDICT TABLE
    # ═══════════════════════════════════════════════════
    section_header("VERDICT TABLE")

    _vr = []
    for name, (label, tone, score) in _verdicts.items():
        if name == "MDD":
            vd = f"{-abs(_mdd or 0)*100:.4f}%"
        elif name == "DSR":
            vd = f"{(_dsr or 0)*100:.4f}%"
        elif name == "Sharpe":
            vd = f"{_shp or 0:.4f}"
        elif name == "Sortino":
            vd = f"{_srt or 0:.4f}"
        elif name == "Calmar":
            vd = f"{_clm or 0:.4f}"
        elif name == "Sterling":
            vd = f"{_stl or 0:.4f}"
        else:
            vd = "—"
        _vr.append({"METRIC": name, "VALUE": vd,
                    "VERDICT": label, "SCORE": score})

    with panel(f"{len(_vr)} VERDICTS"):
        show_table(pd.DataFrame(_vr), num_cols=["SCORE"], precision=0)

    # ═══════════════════════════════════════════════════
    # SUPPORTING STATISTICS
    # ═══════════════════════════════════════════════════
    section_header("SUPPORTING STATISTICS")

    metric_row([
        dict(label="CAGR", value=f"{_cagr*100:.2f}%",
             tone="up" if (_cagr or 0) > 0 else "down"),
        dict(label="VOL (ANN)", value=f"{_vol*100:.2f}%"),
        dict(label="MEAN DAILY", value=f"{_s(returns.mean())*100:.4f}%"),
        dict(label="WIN RATE", value=f"{_win*100:.2f}%"),
        dict(label="PROFIT FACTOR",
             value=f"{_pf:.2f}" if _pf == _pf else "—"),
        dict(label="OBS", value=f"{len(returns)}"),
    ], cols=6)

    metric_row([
        dict(label="SKEWNESS", value=f"{_skw:+.3f}",
             tone="up" if (_skw or 0) > 0 else "down"),
        dict(label="KURTOSIS", value=f"{_krt:.3f}"),
        dict(label="VaR 95%", value=f"{_v95*100:.3f}%", tone="down"),
        dict(label="CVaR 95%",
             value=f"{_cv95*100:.3f}%" if _cv95 == _cv95 else "—",
             tone="down"),
        dict(label="MAX WIN STREAK", value=f"{_mws}"),
        dict(label="MAX LOSS STREAK", value=f"{_mls}"),
    ], cols=6)

    # ═══════════════════════════════════════════════════
    # CHARTS
    # ═══════════════════════════════════════════════════
    section_header("DISTRIBUTION")

    with panel("DAILY RETURNS HISTOGRAM"):
        fig = go.Figure(go.Histogram(
            x=returns, nbinsx=60,
            marker_color="#7fd1ff", opacity=0.75,
        ))
        fig.add_vline(x=float(_v95), line_dash="dash", line_color="#ff5c5c",
                      annotation_text=f"VaR95 {_v95*100:.2f}%")
        fig.add_vline(x=float(_cv95), line_dash="dot", line_color="#fbbf24",
                      annotation_text=f"CVaR95 {_cv95*100:.2f}%")
        fig.update_layout(
            height=340, showlegend=False,
            xaxis=dict(title="Daily Return"),
            yaxis=dict(title="Count"),
        )
        plotly_chart(fig)

    section_header("ROLLING 60D SHARPE")
    roll = (
        pd.Series(returns).rolling(60).mean()
        / pd.Series(returns).rolling(60).std()
        * np.sqrt(252)
    )
    with panel("60-DAY WINDOW"):
        fig = go.Figure(go.Scatter(
            y=roll.values, mode="lines",
            line=dict(color="#4ade80", width=1.4),
        ))
        fig.add_hline(y=0, line_dash="dot",
                      line_color="rgba(255,255,255,0.2)")
        fig.update_layout(height=280, yaxis=dict(title="Sharpe (60D)"))
        plotly_chart(fig)

    section_header("CUMULATIVE RETURN")
    with panel("EQUITY (BACKTEST)"):
        eq = (1 + pd.Series(returns)).cumprod()
        fig = go.Figure(go.Scatter(
            y=eq.values, mode="lines",
            line=dict(color="#7fd1ff", width=1.6),
            fill="tozeroy", fillcolor="rgba(127,209,255,0.08)",
        ))
        fig.update_layout(height=280, yaxis=dict(title="Cumulative"))
        plotly_chart(fig)

    # ═══════════════════════════════════════════════════
    # RAW
    # ═══════════════════════════════════════════════════
    section_header("ALL METRICS · RAW")

    with panel("evaluation_risk.calculate_all_metrics() RETURN"):
        show_table(
            pd.DataFrame([
                {"METRIC": k, "VALUE": v if v is not None else "—"}
                for k, v in m.items()
            ]),
            num_cols=["VALUE"],
            precision=6,
        )

    st.caption(
        f"QFRS Score = 6 verdict scores 평균 · "
        f"Rf={rf*100:.2f}% · periods={int(periods)} · "
        f"n_trials={int(n_trials)} · sterling_top_k={int(top_k)}"
    )


# ═══════════════════════════════════════════════════════════
# ═══ TAB 2 · BACKTEST (Performance) ══════════════════════════
# ═══════════════════════════════════════════════════════════
with tab_bt:
    section_header("PERFORMANCE · evaluation_backtest.py")

    # ── Source 후보 ────────────────────────────────────
    sources = []
    if models:
        sources.append("Model Lab · best")
    if bt:
        for s in bt.keys():
            sources.append(f"Backtest · Strategy {s}")

    if not sources:
        st.info("Model Lab 또는 Backtest 결과가 필요합니다.")
        st.stop()

    csel = st.radio(
        "Source", sources,
        horizontal=True, index=0, key="_bt_src",
    )

    y_true_arr = None
    y_pred_arr = None
    y_pred_proba_arr = None
    labels_list = None
    regression_pred = None
    regression_ret = None

    # ── Model Lab best ────────────────────────────────
    if csel == "Model Lab · best" and models:
        try:
            from services import comparison_service
            best_name = comparison_service.best_model(models)
        except Exception:
            best_name = list(models.keys())[0]

        oos = pd.DataFrame(models[best_name].get("oos_predictions", []))
        if not oos.empty:
            y_true_arr = oos["actual"].to_numpy(dtype=int)
            y_pred_arr = oos["predicted"].to_numpy(dtype=int)
            y_pred_proba_arr = oos[[
                "p_down", "p_neutral", "p_up",
            ]].to_numpy(dtype=float)
            labels_list = [-1, 0, 1]
            st.caption(f"Source: Model Lab · {best_name} · OOS {len(oos)} rows")
        else:
            st.warning("oos_predictions 없음. Model Lab RUN 다시 필요.")

    # ── Backtest Strategy ─────────────────────────────
    elif csel.startswith("Backtest · Strategy"):
        strat = csel.replace("Backtest · Strategy ", "")
        if strat in bt:
            sl = pd.DataFrame(bt[strat].get("signal_log", []))
            st.caption(f"Source: Backtest · Strategy {strat} · signal_log {len(sl)} rows")

            if not sl.empty:
                label_map = {"상승": 1, "중립": 0, "하락": -1}
                if "signal" in sl.columns:
                    sl["_y_pred"] = sl["signal"].map(label_map)
                if "realized_return_5d" in sl.columns:
                    sl["_y_true"] = np.where(
                        sl["realized_return_5d"] > 0.01, 1,
                        np.where(sl["realized_return_5d"] < -0.01, -1, 0),
                    )

                if "_y_true" in sl.columns and "_y_pred" in sl.columns:
                    mask = sl["_y_true"].notna() & sl["_y_pred"].notna()
                    sub = sl[mask]
                    if not sub.empty:
                        y_true_arr = sub["_y_true"].to_numpy(dtype=int)
                        y_pred_arr = sub["_y_pred"].to_numpy(dtype=int)
                        labels_list = [-1, 0, 1]
                        if all(c in sub.columns for c in ("p_down", "p_flat", "p_up")):
                            y_pred_proba_arr = sub[[
                                "p_down", "p_flat", "p_up",
                            ]].to_numpy(dtype=float)

                if "p_up" in sl.columns and "realized_return_5d" in sl.columns:
                    reg_mask = sl["p_up"].notna() & sl["realized_return_5d"].notna()
                    regression_pred = sl.loc[reg_mask, "p_up"].to_numpy(dtype=float)
                    regression_ret = sl.loc[reg_mask, "realized_return_5d"].to_numpy(dtype=float)

    # ═══════════════════════════════════════════════════
    # CLASSIFICATION METRICS
    # ═══════════════════════════════════════════════════
    section_header("CLASSIFICATION METRICS")

    if y_true_arr is None or y_pred_arr is None:
        st.caption("분류 지표에 필요한 y_true / y_pred 가 없습니다.")
    else:
        try:
            cls_res = risk_service.calculate_classification(
                y_true=y_true_arr,
                y_pred=y_pred_arr,
                y_pred_proba=y_pred_proba_arr,
                labels=labels_list,
            )
        except Exception as e:
            st.error(f"분류 지표 계산 실패: {type(e).__name__}: {e}")
            cls_res = None

        if cls_res:
            _ba = cls_res.get("Balanced Accuracy")
            _mcc = cls_res.get("Multiclass MCC")
            _ap = cls_res.get("Macro Average Precision")

            metric_row([
                dict(label="BALANCED ACC",
                     value=f"{_ba:.4f}" if _ba is not None else "—",
                     tone="up" if (_ba or 0) > 0.4 else "neutral"),
                dict(label="MULTICLASS MCC",
                     value=f"{_mcc:.4f}" if _mcc is not None else "—",
                     tone="up" if (_mcc or 0) > 0 else "down"),
                dict(label="MACRO AVG PRECISION",
                     value=f"{_ap:.4f}" if _ap is not None else "—"),
                dict(label="SAMPLES", value=f"{len(y_true_arr)}"),
            ], cols=4)

            # Confusion Matrix
            cm = cls_res.get("Confusion Matrix")
            if cm is not None:
                cm_arr = np.asarray(cm)
                labels_disp = ["DOWN", "NEUTRAL", "UP"]
                section_header("CONFUSION MATRIX")
                with panel("ACTUAL × PREDICTED"):
                    fig = go.Figure(go.Heatmap(
                        z=cm_arr,
                        x=[f"P·{l}" for l in labels_disp],
                        y=[f"A·{l}" for l in labels_disp],
                        colorscale=[
                            [0.0, "#0e1420"],
                            [0.4, "#1e4a6e"],
                            [1.0, "#7fd1ff"],
                        ],
                        showscale=False,
                        text=cm_arr,
                        texttemplate="%{text}",
                        textfont=dict(size=14, color="#e6e8ec",
                                      family="JetBrains Mono, monospace"),
                    ))
                    fig.update_layout(
                        height=320,
                        xaxis=dict(title="Predicted", side="top"),
                        yaxis=dict(title="Actual", autorange="reversed"),
                    )
                    plotly_chart(fig)

                    cm_df = pd.DataFrame(
                        cm_arr,
                        index=[f"actual_{l}" for l in labels_disp],
                        columns=[f"pred_{l}" for l in labels_disp],
                    )
                    show_table(cm_df, num_cols=list(cm_df.columns), precision=0)

            # Class Distribution
            cd = cls_res.get("Class Distribution")
            if cd:
                section_header("CLASS DISTRIBUTION")
                with panel():
                    show_table(
                        pd.DataFrame([
                            {"CLASS": str(k), "SHARE": v} for k, v in cd.items()
                        ]),
                        num_cols=["SHARE"], precision=4,
                    )

            # All Classification
            section_header("ALL CLASSIFICATION METRICS")
            cls_rows = [
                {"METRIC": k, "VALUE": float(v)}
                for k, v in cls_res.items()
                if k not in ("Confusion Matrix", "Class Distribution")
                and isinstance(v, (int, float, np.floating, np.integer))
            ]
            if cls_rows:
                with panel():
                    show_table(pd.DataFrame(cls_rows),
                               num_cols=["VALUE"], precision=6)

    # ═══════════════════════════════════════════════════
    # REGRESSION METRICS
    # ═══════════════════════════════════════════════════
    section_header("REGRESSION METRICS")

    if regression_pred is None or regression_ret is None:
        st.caption(
            "회귀 지표(IC)에 필요한 p_up · realized_return_5d 가 없습니다. "
            "(Backtest Strategy 소스 선택 시 제공)"
        )
    elif not risk_service.regression_available():
        st.caption("evaluation_backtest 회귀 지표 import 실패.")
    else:
        try:
            reg_res = risk_service.calculate_regression(
                predictions=regression_pred,
                returns=regression_ret,
            )
        except Exception as e:
            st.error(f"회귀 지표 계산 실패: {type(e).__name__}: {e}")
            reg_res = None

        if reg_res:
            _mae = reg_res.get("MAE")
            _rmse = reg_res.get("RMSE")
            _pic = reg_res.get("Pearson IC")
            _ric = reg_res.get("Rank IC")

            metric_row([
                dict(label="MAE",
                     value=f"{_mae:.6f}" if _mae is not None else "—"),
                dict(label="RMSE",
                     value=f"{_rmse:.6f}" if _rmse is not None else "—"),
                dict(label="PEARSON IC",
                     value=f"{_pic:+.4f}" if _pic is not None else "—",
                     tone="up" if (_pic or 0) > 0 else "down"),
                dict(label="RANK IC",
                     value=f"{_ric:+.4f}" if _ric is not None else "—",
                     tone="up" if (_ric or 0) > 0 else "down"),
            ], cols=4)

            with panel("ALL REGRESSION METRICS"):
                show_table(
                    pd.DataFrame([
                        {"METRIC": k, "VALUE": v if v is not None else "—"}
                        for k, v in reg_res.items()
                    ]),
                    num_cols=["VALUE"],
                    precision=6,
                )

            st.caption(
                "IC: 예측 p_up 과 realized_return_5d 의 상관. "
                "cross-sectional IC 는 종목별로 별도 계산 필요."
            )