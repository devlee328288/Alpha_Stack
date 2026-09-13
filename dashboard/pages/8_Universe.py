import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Universe · AlphaStack", layout="wide")

import theme
from theme import (
    metric_row, section_header, top_strip, panel, show_table, plotly_chart,
)
theme.inject()

from components import sidebar_controls
from services import universe_service
from state import (
    ctx,
    universe_get, universe_set, universe_clear,
)

sidebar_controls()
c = ctx()

# ═══════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════
st.markdown('<div class="as-title">Universe Ranking</div>', unsafe_allow_html=True)

baseline_u = universe_get("baseline")
model_u = universe_get("model")

# ═══════════════════════════════════════════════════════════
# SOURCE SELECTOR
# ═══════════════════════════════════════════════════════════
section_header("SOURCE")

with panel():
    c1, c2 = st.columns([1, 3], gap="small")
    with c1:
        source = st.radio(
            "Universe",
            ["stocks30", "full"],
            index=0,
            horizontal=False,
            help=(
                "stocks30 = 30종목 개발용 샘플 · 빠름  \n"
                "full = 전종목 parquet (3462종목, 444MB) · 필터 필요"
            ),
        )
    with c2:
        if source == "full":
            st.warning(
                "⚠️ **전종목 모드**: 444MB parquet 로드 (첫 실행 시 30초~1분)  \n"
                "필터를 좁히지 않으면 실행 시간이 매우 오래 걸립니다."
            )
        else:
            st.info("개발용 30종목 샘플 · 로드 빠름 (2초)")

# ═══════════════════════════════════════════════════════════
# FILTERS (full 전용)
# ═══════════════════════════════════════════════════════════
filter_kwargs: dict = {}

if source == "full":
    section_header("FILTERS · FULL UNIVERSE")

    with panel():
        c1, c2, c3 = st.columns(3, gap="small")
        with c1:
            market = st.selectbox(
                "Market", ["KOSPI", "KOSPI+KOSDAQ", "ALL"], index=0,
            )
            filter_kwargs["market"] = market

        with c2:
            top_n_options = {
                "Top 50": 50, "Top 100": 100, "Top 200": 200,
                "Top 500": 500, "전체 (None)": None,
            }
            top_n_sel = st.selectbox(
                "Market Cap Top-N", list(top_n_options), index=2,
            )
            filter_kwargs["top_n"] = top_n_options[top_n_sel]

        with c3:
            price_col = st.selectbox(
                "Price column",
                ["adj_close", "adj_close_tr", "close"],
                index=0,
                help="adj_close = 배당·분할 조정 (권장)",
            )
            # price_col은 로더에서 사용

        c1, c2, c3 = st.columns(3, gap="small")
        with c1:
            ex_halt = st.checkbox("거래정지 제외", value=True)
            filter_kwargs["exclude_halted"] = ex_halt
        with c2:
            ex_liq = st.checkbox("상장폐지 제외", value=True)
            filter_kwargs["exclude_liquidation"] = ex_liq
        with c3:
            ex_first = st.checkbox("신규상장 제외", value=True)
            filter_kwargs["exclude_first_listing"] = ex_first

        c1, c2 = st.columns(2, gap="small")
        with c1:
            min_cap_eok = st.number_input(
                "최소 시가총액 (억원)", 0, 100000, 0, 100,
                help="0 이면 필터 없음",
            )
            filter_kwargs["min_market_cap"] = float(min_cap_eok) * 1e8
        with c2:
            st.caption("")  # spacing

# ═══════════════════════════════════════════════════════════
# UNIVERSE LIST
# ═══════════════════════════════════════════════════════════
try:
    all_tickers = universe_service.list_universe(source=source, **filter_kwargs)
    codes_all = [t["code"] for t in all_tickers]
except Exception as e:
    st.error(f"종목 리스트 로드 실패: {type(e).__name__}: {e}")
    st.stop()

top_strip(
    [source.upper(), f"{len(codes_all)} TICKERS",
     f"BASELINE {len(baseline_u)}", f"MODEL {len(model_u)}"],
    status_text="READY" if baseline_u else "IDLE",
    status_tone="up" if baseline_u else "neutral",
)

# ═══════════════════════════════════════════════════════════
# CONFIG — RUN
# ═══════════════════════════════════════════════════════════
section_header("RUN CONFIG")

with panel():
    c1, c2, c3, c4 = st.columns([1.4, 1.4, 1.2, 1.2], gap="small")

    with c1:
        subset_options = ["전체", "앞 5", "앞 10", "앞 20", "앞 50", "앞 100"]
        subset_mode = st.selectbox("Subset", subset_options, index=0)
        n_map = {
            "전체": len(codes_all), "앞 5": 5, "앞 10": 10,
            "앞 20": 20, "앞 50": 50, "앞 100": 100,
        }
        n_take = min(n_map[subset_mode], len(codes_all))
        tickers_sel = codes_all[:n_take]

    with c2:
        threshold = st.selectbox(
            "Threshold", [0.01, 0.02],
            format_func=lambda v: f"{v*100:.0f}%", index=0,
        )
        max_evals = st.selectbox(
            "CMA-ES",
            ["QUICK (10)", "MED (30)", "FULL (100)", "MAX (300)"],
            index=0,
        )
        max_evals_n = {
            "QUICK (10)": 10, "MED (30)": 30,
            "FULL (100)": 100, "MAX (300)": 300,
        }[max_evals]

    with c3:
        n_jobs = st.selectbox(
            "Parallel jobs", [1, 2, 4, 6, 8], index=2,
            help="1 = 순차 (느림, 안전) · 8 = 전체 코어 (메모리 주의)",
        )

    with c4:
        skip_existing = st.checkbox(
            "Skip done", value=True,
            help="이미 실행된 종목/모델은 건너뜀 (중단 후 재개)",
        )

    # 예상 시간 계산
    _baseline_per_item = 3 if max_evals_n >= 100 else 0.5
    _model_per_item = 5
    _n_models_est = 1
    _baseline_min = int(n_take * _baseline_per_item / max(n_jobs, 1))
    _model_min = int(n_take * _model_per_item / max(n_jobs, 1))

    st.caption(
        f"선택: **{n_take}종목** · baseline 예상 **{_baseline_min}분** "
        f"(n_jobs={n_jobs}) · model 예상 **{_model_min}분/모델**"
    )

    # ── RUN BASELINE ────────────────────────────────────
    cA, cB = st.columns([1, 3], gap="small")
    with cA:
        run_baseline_btn = st.button(
            "▶  RUN BASELINE", type="primary", use_container_width=True,
        )
    with cB:
        if n_take > 50 or n_jobs == 1 and n_take > 20:
            st.warning(f"⚠️ {n_take}종목 · 시간 오래 걸림. 백그라운드로 두세요.")
        else:
            st.caption("")

    # ── RUN MODEL (multi) ──────────────────────────────
    cC, cD = st.columns([2, 2], gap="small")
    with cC:
        model_names_sel = st.multiselect(
            "Models (multi-select)",
            ["RandomForest", "LightGBM", "XGBoost", "LogisticRegression"],
            default=["RandomForest"],
        )
    with cD:
        run_model_btn = st.button(
            "▶  RUN MODEL(S)", type="secondary", use_container_width=True,
        )

    total_runs = len(model_names_sel) * n_take
    if total_runs > 100:
        st.error(
            f"🚨 **{total_runs}회 실행** · 예상 **{int(total_runs * _model_per_item / max(n_jobs,1))}분**. "
            f"Subset을 줄이거나 n_jobs를 늘리세요."
        )
    else:
        st.caption(f"Model: {len(model_names_sel)}개 × {n_take}종목 = {total_runs}회")

    # ── Clear ───────────────────────────────────────────
    c1, c2 = st.columns([1, 1], gap="small")
    with c1:
        if st.button("Clear baseline cache", use_container_width=True):
            universe_clear("baseline")
            st.rerun()
    with c2:
        if st.button("Clear model cache", use_container_width=True):
            universe_clear("model")
            st.rerun()

# ═══════════════════════════════════════════════════════════
# RUN — BASELINE (parallel)
# ═══════════════════════════════════════════════════════════
if run_baseline_btn:
    progress_bar = st.progress(0.0)
    status_box = st.empty()
    log_box = st.empty()
    logs = []

    def _cb(i, tot, ticker, phase):
        progress_bar.progress(min(i / tot, 1.0))
        if phase in ("done", "error"):
            logs.append(f"[{i}/{tot}] {ticker} · {phase}")
            status_box.caption(f"[{i}/{tot}] {ticker} · {phase}")
            log_box.code("\n".join(logs[-10:]))

    skip = baseline_u if skip_existing else None

    with st.spinner(f"Baseline · {n_take}종목 · n_jobs={n_jobs}…"):
        results = universe_service.run_universe_baseline_parallel(
            tickers=tickers_sel,
            threshold=float(threshold),
            max_evals=int(max_evals_n),
            source=source,
            n_jobs=int(n_jobs),
            skip_existing=skip,
            progress_cb=_cb,
        )
    for tk, r in results.items():
        universe_set("baseline", tk, r)
    st.success(f"✅ Baseline 완료 · {len(results)} 종목")
    st.rerun()

# ═══════════════════════════════════════════════════════════
# RUN — MODEL (parallel)
# ═══════════════════════════════════════════════════════════
if run_model_btn and model_names_sel:
    progress_bar = st.progress(0.0)
    status_box = st.empty()
    log_box = st.empty()
    logs = []

    def _cb2(done, total_steps, ticker, m_name, phase):
        progress_bar.progress(min(done / total_steps, 1.0))
        if phase in ("done", "error"):
            logs.append(f"[{done}/{total_steps}] {ticker} · {m_name} · {phase}")
            status_box.caption(f"[{done}/{total_steps}] {ticker} · {m_name} · {phase}")
            log_box.code("\n".join(logs[-12:]))

    skip = model_u if skip_existing else None

    with st.spinner(f"Model · {len(model_names_sel)}개 × {n_take}종목 · n_jobs={n_jobs}…"):
        results = universe_service.run_universe_models_multi_parallel(
            tickers=tickers_sel,
            model_names=model_names_sel,
            source=source,
            n_jobs=int(n_jobs),
            skip_existing=skip,
            progress_cb=_cb2,
        )
    for ck, r in results.items():
        universe_set("model", ck, r)
    st.success(f"✅ Model 완료 · {len(results)} entries")
    st.rerun()

# ═══════════════════════════════════════════════════════════
# BASELINE RANKING
# ═══════════════════════════════════════════════════════════
baseline_u = universe_get("baseline")
if baseline_u:
    section_header(f"BASELINE RANKING · {len(baseline_u)} STOCKS")
    rank_df = universe_service.build_universe_ranking(baseline_u)

    valid = rank_df[rank_df["ERROR"] == ""]
    if not valid.empty:
        _top = valid.iloc[0]

        def _fmt(v, digits=4, signed=False):
            if v is None or pd.isna(v):
                return "—"
            return f"{v:+.{digits}f}" if signed else f"{v:.{digits}f}"

        _mean_mf1 = valid["MACRO F1"].mean() if "MACRO F1" in valid.columns else None
        _mean_shp = valid["SHARPE"].mean() if "SHARPE" in valid.columns else None

        metric_row([
            dict(label="TOP STOCK",
                 value=f"{_top['CODE']} · {_top['NAME']}",
                 tone="up", accent=True),
            dict(label="TOP MACRO F1", value=_fmt(_top.get("MACRO F1")),
                 tone="up"),
            dict(label="TOP BAL ACC", value=_fmt(_top.get("BAL ACC"))),
            dict(label="MEAN MACRO F1", value=_fmt(_mean_mf1)),
            dict(label="MEAN SHARPE", value=_fmt(_mean_shp),
                 tone="up" if (_mean_shp or 0) > 0 else "down"),
        ], cols=5)

    with panel(f"{len(rank_df)} STOCKS · SORT BY MACRO F1",
               status_text=f"BEST · {valid.iloc[0]['CODE']}" if not valid.empty else "",
               status_tone="up"):
        show_table(
            rank_df,
            num_cols=["MACRO F1", "BAL ACC",
                      "SHARPE", "CAGR", "MDD", "WIN RATE"],
            precision=4,
            highlight_row=0,
        )

    section_header("TOP 15 · MACRO F1")
    top15 = rank_df[rank_df["ERROR"] == ""].head(15)
    if not top15.empty:
        with panel("MACRO F1 (SORTED DESC)"):
            fig = go.Figure(go.Bar(
                x=top15["MACRO F1"],
                y=[f"{r['CODE']} · {r['NAME']}" for _, r in top15.iterrows()],
                orientation="h",
                marker=dict(color=["#4ade80" if i == 0 else "#7fd1ff"
                                    for i in range(len(top15))]),
                text=[f"{v:.4f}" for v in top15["MACRO F1"]],
                textposition="outside",
            ))
            fig.update_layout(
                height=max(300, 32 * len(top15)),
                showlegend=False,
                xaxis=dict(title="", range=[0, float(top15["MACRO F1"].max()) * 1.15]),
                yaxis=dict(title="", autorange="reversed"),
            )
            plotly_chart(fig)

# ═══════════════════════════════════════════════════════════
# MODEL RANKING · MATRIX
# ═══════════════════════════════════════════════════════════
model_u = universe_get("model")
if model_u:
    section_header(f"MODEL RANKING · {len(model_u)} ENTRIES")

    matrix = universe_service.build_universe_models_matrix(model_u)
    if not matrix.empty:
        model_cols = [c for c in matrix.columns
                      if c not in ("RANK", "CODE", "NAME")]

        section_header("MATRIX · HARMONIC (TICKER × MODEL)")
        with panel(f"{len(matrix)} STOCKS × {len(model_cols)} MODELS"):
            show_table(
                matrix,
                num_cols=model_cols,
                precision=4,
                highlight_row=0,
            )

    section_header("FULL DETAIL · LONG")
    long_df = universe_service.build_models_by_ticker_table(model_u)

    with panel(f"{len(long_df)} ROWS · SORT BY HARMONIC"):
        show_table(
            long_df.head(100),
            num_cols=["ACC", "MACRO F1", "DOWN RECALL", "HARMONIC", "ΔSHARPE"],
            precision=4,
            highlight_row=0,
        )

# ═══════════════════════════════════════════════════════════
# DRILL-DOWN
# ═══════════════════════════════════════════════════════════
if baseline_u:
    section_header("DRILL-DOWN · 종목 → STOCK 모드")

    with panel("종목 선택 → 해당 종목만 STOCK scope 에서 재실행"):
        rank_for_drill = universe_service.build_universe_ranking(baseline_u)
        top10 = rank_for_drill[rank_for_drill["ERROR"] == ""].head(10)

        c1, c2 = st.columns([3, 1], gap="small")
        with c1:
            options = [
                f"{r['CODE']} · {r['NAME']} · MACRO F1 {r['MACRO F1']:.4f}"
                for _, r in top10.iterrows()
            ]
            picked = st.selectbox("종목 선택 (baseline TOP 10)",
                                  options, index=0, key="_drill_pick")
        with c2:
            st.write("")
            if st.button("▶  STOCK 모드", type="primary",
                         use_container_width=True, key="_drill_btn"):
                code = picked.split(" · ")[0]
                st.session_state["scope"] = "STOCK"
                st.session_state["ticker"] = code
                st.rerun()

# ═══════════════════════════════════════════════════════════
# EMPTY STATE
# ═══════════════════════════════════════════════════════════
if not baseline_u and not model_u:
    section_header("START")
    with panel("HOW TO RUN"):
        st.markdown("""
**Source 선택**
- `stocks30` — 30종목 개발용 (빠름)
- `full` — 전종목 parquet (3462종목, 필터 권장)

**필터 (full 전용)**
- Market: KOSPI / KOSPI+KOSDAQ / ALL
- Top-N: 시가총액 상위 (권장 200)
- 잡음 제외: 거래정지·상장폐지·신규상장

**실행**
1. Subset, Threshold, CMA-ES, n_jobs 설정
2. **▶ RUN BASELINE** (종목별 6-param walk-forward)
3. **Models (multi)** 선택 → **▶ RUN MODEL(S)**
4. **Skip done** 체크 → 중단 후 재개

**주의**
- n_jobs 늘리면 빠르지만 메모리·CPU 사용 ↑
- 200종목 × QUICK × n_jobs=4 = 약 25분
- Subset 줄여서 먼저 테스트 권장
""")