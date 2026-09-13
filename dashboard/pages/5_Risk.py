import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Risk · AlphaStack", layout="wide")

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
st.markdown('<div class="as-title">Risk</div>', unsafe_allow_html=True)

_err = risk_service.engine_status()
if _err and not risk_service.risk_available():
    top_strip([scope, ticker], status_text="IMPORT FAIL", status_tone="warn")
    st.error(_err)
    st.stop()

top_strip(
    [scope, ticker, "evaluation_risk · evaluation_backtest",
     "QFRS · CLASSIFICATION · REGRESSION"],
    status_text="READY", status_tone="neutral",
)

# ═══════════════════════════════════════════════════════════
# SOURCE
# ═══════════════════════════════════════════════════════════
section_header("SOURCE")

bt = get_result("bt", scope, ticker)
models = get_result("model_lab", scope, ticker)

if not bt and not models:
    st.warning(
        f"**Backtest** 또는 **Model Lab** 페이지에서 {scope}:{ticker} 결과를 먼저 만들어주세요."
    )
    st.stop()

# ── Backtest 선택 ─────────────────────────────────────
bt_returns = None
bt_signal_log = None
if bt:
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1], gap="small")
    with c1:
        pick = st.selectbox("Backtest Strategy", list(bt.keys()), index=0)
    with c2:
        rf = st.number_input("Risk-free (annual)", 0.0, 0.10, 0.02, 0.005, format="%.4f")
    with c3:
        n_trials = st.number_input("N trials (DSR)", 1, 1000, 50, 1)
    with c4:
        top_k = st.number_input("Sterling top-K", 1, 10, 3, 1)

    bt_returns = bt[pick]["daily_returns"].to_numpy()
    bt_signal_log = pd.DataFrame(bt[pick].get("signal_log", []))
    source_label = f"Backtest · Strategy {pick} · {scope}:{ticker}"
else:
    rf, n_trials, top_k = 0.02, 50, 3
    bt_returns = None
    source_label = f"Model Lab · {scope}:{ticker}"

# ═══════════════════════════════════════════════════════════
# QFRS 사전 계산 (backtest 있을 때만)
# ═══════════════════════════════════════════════════════════
m = {}
if bt_returns is not None:
    try:
        m = risk_service.calculate_risk(
            bt_returns,
            risk_free_rate=float(rf),
            periods_per_year=252,
            n_trials=int(n_trials),
            sterling_top_k=int(top_k),
        )
    except Exception as e:
        st.error(f"Risk 계산 실패: {type(e).__name__}: {e}")

# ═══════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════
tab_ov, tab_qfrs, tab_perf, tab_dist, tab_raw = st.tabs(
    ["OVERVIEW", "QFRS", "PERFORMANCE", "DISTRIBUTION", "RAW"]
)

# ─────────────────────────────────────────────────────────
# TAB 1 · OVERVIEW
# ─────────────────────────────────────────────────────────
with tab_ov:
    if not m:
        st.info("Backtest 결과가 있어야 리스크 지표를 계산합니다.")
    else:
        section_header(f"RISK METRICS · {source_label}")

        metric_row([
            dict(label="MDD",
                 value=f"{-abs(m.get('MDD') or 0)*100:.2f}%",
                 tone="down"),
            dict(label="SHARPE",
                 value=f"{m.get('Sharpe Ratio') or 0:.4f}",
                 tone="up" if (m.get("Sharpe Ratio") or 0) > 0 else "down"),
            dict(label="SORTINO",
                 value=f"{m.get('Sortino Ratio') or 0:.4f}",
                 tone="up" if (m.get("Sortino Ratio") or 0) > 0 else "down"),
            dict(label="CALMAR",
                 value=f"{m.get('Calmar Ratio') or 0:.4f}"),
            dict(label="STERLING",
                 value=f"{m.get('Sterling Ratio') or 0:.4f}"),
            dict(label="DSR",
                 value=f"{(m.get('Deflated Sharpe Ratio') or 0)*100:.2f}%"),
        ], cols=6)

# ─────────────────────────────────────────────────────────
# TAB 2 · QFRS
# ─────────────────────────────────────────────────────────
with tab_qfrs:
    if not m:
        st.info("Backtest 결과가 있어야 QFRS 지표를 계산합니다.")
    else:
        section_header("QFRS · QUANTITATIVE RISK SCORE")
        st.caption(
            f"Source: **{source_label}** · Rf = {rf*100:.2f}% · "
            f"n_trials = {n_trials} · sterling_top_k = {top_k}"
        )

        _mdd = m.get("MDD"); _shp = m.get("Sharpe Ratio")
        _srt = m.get("Sortino Ratio"); _clm = m.get("Calmar Ratio")
        _stl = m.get("Sterling Ratio"); _dsr = m.get("Deflated Sharpe Ratio")

        core_rows = [
            {"METRIC": "MDD", "FULL NAME": "Maximum Drawdown",
             "VALUE": f"{-abs(_mdd or 0)*100:.4f}%",
             "DESCRIPTION": "최대 낙폭 (고점 대비 저점)"},
            {"METRIC": "SHARPE", "FULL NAME": "Sharpe Ratio (ann.)",
             "VALUE": f"{_shp or 0:.4f}",
             "DESCRIPTION": "위험 1단위당 초과수익 (연율)"},
            {"METRIC": "SORTINO", "FULL NAME": "Sortino Ratio (ann.)",
             "VALUE": f"{_srt or 0:.4f}",
             "DESCRIPTION": "하방위험 1단위당 초과수익 (연율)"},
            {"METRIC": "CALMAR", "FULL NAME": "Calmar Ratio",
             "VALUE": f"{_clm or 0:.4f}",
             "DESCRIPTION": "CAGR − Rf / MDD"},
            {"METRIC": "STERLING", "FULL NAME": "Sterling Ratio",
             "VALUE": f"{_stl or 0:.4f}",
             "DESCRIPTION": f"CAGR − Rf / 상위 {int(top_k)} MDD 평균"},
            {"METRIC": "DSR", "FULL NAME": "Deflated Sharpe Ratio",
             "VALUE": f"{(_dsr or 0)*100:.4f}%",
             "DESCRIPTION": f"우연이 아닐 확률 (n_trials={n_trials})"},
        ]
        with panel(f"{len(core_rows)} METRICS · CORE"):
            show_table(pd.DataFrame(core_rows))

        # Supporting
        ret = bt_returns
        from scipy.stats import skew, kurtosis
        _cagr = float(np.prod(1 + ret) ** (252 / max(len(ret), 1)) - 1)
        _vol = float(np.std(ret, ddof=1) * np.sqrt(252))
        _skw = float(skew(ret)); _krt = float(kurtosis(ret, fisher=False))
        _v95 = float(np.percentile(ret, 5))
        _cv95 = float(ret[ret <= _v95].mean()) if len(ret[ret <= _v95]) else np.nan
        _win = float((ret > 0).mean())
        _gain = ret[ret > 0].sum(); _loss = abs(ret[ret < 0].sum())
        _pf = float(_gain / _loss) if _loss > 0 else float("inf")

        sup_rows = [
            {"METRIC": "OBSERVATIONS", "VALUE": f"{len(ret)}",
             "DESCRIPTION": "일별 수익률 개수"},
            {"METRIC": "CAGR", "VALUE": f"{_cagr*100:.4f}%",
             "DESCRIPTION": "연평균 복리 수익률"},
            {"METRIC": "VOL (ANNUAL)", "VALUE": f"{_vol*100:.4f}%",
             "DESCRIPTION": "연환산 변동성"},
            {"METRIC": "MEAN DAILY", "VALUE": f"{float(ret.mean())*100:.4f}%",
             "DESCRIPTION": "일평균 수익률"},
            {"METRIC": "WIN RATE", "VALUE": f"{_win*100:.4f}%",
             "DESCRIPTION": "양(+) 수익률 비율"},
            {"METRIC": "PROFIT FACTOR", "VALUE": f"{_pf:.4f}",
             "DESCRIPTION": "총이익 / 총손실"},
            {"METRIC": "SKEWNESS", "VALUE": f"{_skw:+.4f}",
             "DESCRIPTION": "왜도 (정규=0)"},
            {"METRIC": "KURTOSIS", "VALUE": f"{_krt:.4f}",
             "DESCRIPTION": "첨도 (정규=3)"},
            {"METRIC": "VaR (95%)", "VALUE": f"{_v95*100:.4f}%",
             "DESCRIPTION": "5% 분위 일별 수익률"},
            {"METRIC": "CVaR (95%)", "VALUE": f"{_cv95*100:.4f}%",
             "DESCRIPTION": "VaR 이하 평균 (ES)"},
        ]
        with panel(f"{len(sup_rows)} STATISTICS"):
            show_table(pd.DataFrame(sup_rows))

        # Verdict
        verdict_rows = [
            {"METRIC": "Sharpe Ratio", "VALUE": f"{_shp or 0:.4f}",
             "VERDICT": ("STRONG" if (_shp or 0) > 1.0
                         else "GOOD" if (_shp or 0) > 0.5
                         else "WEAK" if (_shp or 0) > 0 else "FAIL")},
            {"METRIC": "Sortino Ratio", "VALUE": f"{_srt or 0:.4f}",
             "VERDICT": ("STRONG" if (_srt or 0) > 1.5
                         else "GOOD" if (_srt or 0) > 0.7
                         else "WEAK" if (_srt or 0) > 0 else "FAIL")},
            {"METRIC": "MDD", "VALUE": f"{-abs(_mdd or 0)*100:.2f}%",
             "VERDICT": ("LOW" if abs(_mdd or 0) < 0.10
                         else "MODERATE" if abs(_mdd or 0) < 0.20
                         else "HIGH" if abs(_mdd or 0) < 0.35 else "SEVERE")},
            {"METRIC": "Calmar Ratio", "VALUE": f"{_clm or 0:.4f}",
             "VERDICT": ("STRONG" if (_clm or 0) > 0.5
                         else "GOOD" if (_clm or 0) > 0.2
                         else "WEAK" if (_clm or 0) > 0 else "FAIL")},
            {"METRIC": "Deflated Sharpe", "VALUE": f"{(_dsr or 0)*100:.2f}%",
             "VERDICT": ("PASS" if (_dsr or 0) > 0.95
                         else "PARTIAL" if (_dsr or 0) > 0.5 else "FAIL")},
        ]
        with panel(f"{len(verdict_rows)} VERDICTS"):
            show_table(pd.DataFrame(verdict_rows))

# ─────────────────────────────────────────────────────────
# TAB 3 · PERFORMANCE (evaluation_backtest)
# ─────────────────────────────────────────────────────────
with tab_perf:
    section_header("PERFORMANCE · evaluation_backtest.py")

    # ── Source selector ────────────────────────────────
    perf_sources = []
    if models:
        perf_sources.append("Model Lab · best")
    if bt_signal_log is not None and not bt_signal_log.empty:
        perf_sources.append("Backtest · signal_log")

    if not perf_sources:
        st.info("Model Lab 또는 Backtest signal_log 결과가 있어야 계산 가능합니다.")
    else:
        csel = st.radio(
            "Source", perf_sources,
            horizontal=True, index=0, key="_perf_src",
        )

        y_true_arr = None
        y_pred_arr = None
        y_pred_proba_arr = None
        labels_list = None
        regression_pred = None
        regression_ret = None

        # ── 준비: Model Lab best ────────────────────────
        if csel == "Model Lab · best" and models:
            best = models[list(models.keys())[0]]  # 또는 comparison_service.best_model
            # best model 선정
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

        # ── 준비: Backtest signal_log ────────────────────
        elif csel == "Backtest · signal_log":
            sl = bt_signal_log.copy()
            # y_true / y_pred (signal → 숫자)
            label_map = {"상승": 1, "중립": 0, "하락": -1}
            if "signal" in sl.columns:
                sl["_y_pred"] = sl["signal"].map(label_map)
            if "realized_return_5d" in sl.columns:
                # realized_return_5d 를 y_true 로 쓸 수 있지만 이건 회귀지 분류 아님
                # 분류: realized_return 부호로 매핑
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

            # Regression: p_up (continuous) vs realized_return_5d
            if "p_up" in sl.columns and "realized_return_5d" in sl.columns:
                reg_mask = sl["p_up"].notna() & sl["realized_return_5d"].notna()
                regression_pred = sl.loc[reg_mask, "p_up"].to_numpy(dtype=float)
                regression_ret = sl.loc[reg_mask, "realized_return_5d"].to_numpy(dtype=float)

            st.caption(f"Source: Backtest signal_log · {len(sl)} rows")

        # ═══════════════════════════════════════════════
        # CLASSIFICATION METRICS
        # ═══════════════════════════════════════════════
        section_header("CLASSIFICATION METRICS")

        if y_true_arr is None or y_pred_arr is None:
            st.caption("분류 지표 계산에 필요한 y_true / y_pred 가 없습니다.")
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
                # 상단 KPI
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
                    cd_rows = []
                    for k, v in cd.items():
                        cd_rows.append({
                            "CLASS": str(k),
                            "SHARE": v,
                        })
                    with panel():
                        show_table(
                            pd.DataFrame(cd_rows),
                            num_cols=["SHARE"],
                            precision=4,
                        )

                # All classification metrics
                section_header("ALL CLASSIFICATION METRICS")
                cls_rows = []
                for k, v in cls_res.items():
                    if k in ("Confusion Matrix", "Class Distribution"):
                        continue
                    if isinstance(v, (int, float, np.floating, np.integer)):
                        cls_rows.append({"METRIC": k, "VALUE": float(v)})
                if cls_rows:
                    with panel():
                        show_table(
                            pd.DataFrame(cls_rows),
                            num_cols=["VALUE"],
                            precision=6,
                        )

        # ═══════════════════════════════════════════════
        # REGRESSION METRICS
        # ═══════════════════════════════════════════════
        section_header("REGRESSION METRICS")

        if regression_pred is None or regression_ret is None:
            st.caption(
                "회귀 지표(IC) 계산에 필요한 예측값(p_up)과 "
                "실현 수익률(realized_return_5d)이 없습니다."
            )
        elif not risk_service.regression_available():
            st.caption("evaluation_backtest 의 회귀 지표 import 실패.")
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
                    reg_rows = [
                        {"METRIC": k, "VALUE": v if v is not None else "—"}
                        for k, v in reg_res.items()
                    ]
                    show_table(
                        pd.DataFrame(reg_rows),
                        num_cols=["VALUE"],
                        precision=6,
                    )

                st.caption(
                    "IC: 예측 p_up 과 realized_return_5d 의 상관. "
                    "cross-sectional IC 는 종목별로 별도 계산 필요."
                )

# ─────────────────────────────────────────────────────────
# TAB 4 · DISTRIBUTION
# ─────────────────────────────────────────────────────────
with tab_dist:
    if bt_returns is None:
        st.info("Backtest 결과가 있어야 분포를 표시합니다.")
    else:
        section_header("RETURN DISTRIBUTION")
        with panel("DAILY RETURNS HISTOGRAM"):
            _v95 = float(np.percentile(bt_returns, 5))
            _cv95 = float(bt_returns[bt_returns <= _v95].mean())
            fig = go.Figure(go.Histogram(
                x=bt_returns, nbinsx=60,
                marker_color="#7fd1ff", opacity=0.75,
            ))
            fig.add_vline(x=_v95, line_dash="dash", line_color="#ff5c5c",
                          annotation_text=f"VaR95 {_v95*100:.2f}%")
            fig.add_vline(x=_cv95, line_dash="dot", line_color="#fbbf24",
                          annotation_text=f"CVaR95 {_cv95*100:.2f}%")
            fig.update_layout(height=360, showlegend=False,
                              xaxis=dict(title="Daily Return"),
                              yaxis=dict(title="Count"))
            plotly_chart(fig)

        section_header("ROLLING 60D SHARPE")
        roll = (
            pd.Series(bt_returns).rolling(60).mean()
            / pd.Series(bt_returns).rolling(60).std()
            * np.sqrt(252)
        )
        with panel("60-DAY WINDOW"):
            fig = go.Figure(go.Scatter(
                y=roll.values, mode="lines",
                line=dict(color="#4ade80", width=1.4),
            ))
            fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)")
            fig.update_layout(height=280, yaxis=dict(title="Sharpe (60D)"))
            plotly_chart(fig)

# ─────────────────────────────────────────────────────────
# TAB 5 · RAW
# ─────────────────────────────────────────────────────────
with tab_raw:
    if m:
        section_header("RAW · evaluation_risk")
        with panel("calculate_all_metrics() RETURN"):
            show_table(
                pd.DataFrame([
                    {"KEY": k, "VALUE": v if v is not None else "—"}
                    for k, v in m.items()
                ]),
                num_cols=["VALUE"],
                precision=6,
            )
    else:
        st.info("Backtest 결과 없음.")