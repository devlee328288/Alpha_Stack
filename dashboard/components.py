# dashboard/components.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from state import fingerprint, init_state, has_result
from theme import (
    metric_row as _metric_row,
    sidebar_header,
    sidebar_meta,
    sidebar_section,
)
from services import hf_service, data_loader


_RESULTS_MAP = [
    ("Baseline",  "baseline"),
    ("Model Lab", "model_lab"),
    ("Backtest",  "bt"),
    ("Cost Sens", "cost_grid"),
]


# ═══════════════════════════════════════════════════════════
# SCOPE
# ═══════════════════════════════════════════════════════════
def _scope_block() -> None:
    sidebar_section("SCOPE")
    scope = st.session_state.get("scope", "MARKET")

    # 2열 + 1열 (사이드바 폭 좁아서 wrap 방지)
    c1, c2 = st.columns(2)
    if c1.button(
        "MARKET",
        key="_scope_market",
        type="primary" if scope == "MARKET" else "secondary",
        use_container_width=True,
    ):
        if scope != "MARKET":
            st.session_state["scope"] = "MARKET"
            st.session_state["ticker"] = "KOSPI200"
            st.rerun()

    if c2.button(
        "STOCK",
        key="_scope_stock",
        type="primary" if scope == "STOCK" else "secondary",
        use_container_width=True,
    ):
        if scope != "STOCK":
            try:
                tickers = data_loader.list_universe_tickers()
            except Exception:
                tickers = []
            if tickers:
                st.session_state["scope"] = "STOCK"
                st.session_state["ticker"] = tickers[0]["code"]
                st.rerun()

    if st.button(
        "UNIVERSE  (30종목 랭킹)",
        key="_scope_univ",
        type="primary" if scope == "UNIVERSE" else "secondary",
        use_container_width=True,
        help="30종목 전체 랭킹 (Universe 페이지로)",
    ):
        st.session_state["scope"] = "UNIVERSE"
        st.session_state["ticker"] = "ALL"
        try:
            st.switch_page("pages/8_Universe.py")
        except Exception:
            st.rerun()

    # ── STOCK 선택 시 종목 selectbox ────────────────────
    if scope == "STOCK":
        try:
            tickers = data_loader.list_universe_tickers()
        except Exception:
            tickers = []

        if not tickers:
            st.warning("stocks30 데이터를 불러오지 못했습니다.")
            return

        options = [t["code"] for t in tickers]
        label_map = {
            t["code"]: f"{t['code']} · {t.get('name') or '?'}"
            for t in tickers
        }
        current = st.session_state.get("ticker", options[0])
        if current not in options:
            current = options[0]

        picked = st.selectbox(
            "Ticker",
            options,
            index=options.index(current),
            format_func=lambda c: label_map.get(c, c),
            key="_ticker_select",
        )
        st.session_state["ticker"] = picked

    # ── 현재 TICKER 표시 ────────────────────────────────
    ticker_now = st.session_state.get("ticker", "KOSPI200")
    if scope == "UNIVERSE":
        disp = "STOCKS30 · ALL"
    else:
        disp = ticker_now

    st.markdown(
        f'<div style="padding:6px 14px 2px 14px;font-family:var(--font-num);'
        f'font-size:11px;color:var(--text-secondary);">'
        f'<span style="color:var(--text-muted);">TICKER</span> &nbsp; {disp}'
        f"</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════
# HF SNAPSHOT
# ═══════════════════════════════════════════════════════════
def _hf_block() -> None:
    sidebar_section("HF SNAPSHOT")
    meta = hf_service.get_hf_meta()
    if meta.get("ok"):
        sidebar_meta([
            ("repo_sha", meta.get("repo_sha", "—")),
            ("dev_end",  meta.get("dev_end", "—")),
            ("file_sha", meta.get("file_sha", "—")),
        ])
    else:
        st.markdown(
            f'<div style="padding:6px 14px 2px 14px;font-size:10px;'
            f'color:var(--color-warn);">'
            f'HF 메타 조회 실패: {meta.get("err", "")[:60]}'
            f"</div>",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════
def _results_block() -> None:
    sidebar_section("RESULTS")

    scope = st.session_state.get("scope", "MARKET")
    ticker = st.session_state.get("ticker", "KOSPI200")

    rows_html = []
    for label, kind in _RESULTS_MAP:
        ready = has_result(kind, scope, ticker)
        dot_color = "var(--color-up)" if ready else "var(--text-muted)"
        status = "ready" if ready else "empty"
        rows_html.append(
            f'<div style="display:flex;justify-content:space-between;'
            f'align-items:baseline;padding:3px 0;">'
            f'<span style="font-family:var(--font-ui);font-size:11px;'
            f'color:var(--text-secondary);">{label}</span>'
            f'<span style="font-family:var(--font-num);font-size:10px;'
            f'color:{dot_color};">● {status}</span>'
            f"</div>"
        )
    st.markdown(
        '<div style="padding:6px 14px 4px 14px;">'
        + "".join(rows_html)
        + "</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def sidebar_controls() -> bool:
    init_state()

    with st.sidebar:
        sidebar_header("QUANT RESEARCH TERMINAL")
        _scope_block()
        _hf_block()
        _results_block()

        # FINGERPRINT
        st.markdown(
            f'<div style="padding:10px 14px 14px 14px;font-family:var(--font-num);'
            f'font-size:10px;color:var(--text-muted);'
            f'border-top:1px solid var(--border-subtle);margin-top:8px;">'
            f"FINGERPRINT &nbsp; {fingerprint()[:28]}"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── ANALYTICS 바로가기 ──────────────────────
        sidebar_section("ANALYTICS")
        if st.button(
            "QFRS  ·  Risk + Performance",
            key="_qfr_link",
            use_container_width=True,
            help="RISK · BACKTEST 지표 통합 페이지",
        ):
            try:
                st.switch_page("pages/9_QFRS.py")
            except Exception:
                st.rerun()

        return False


# ═══════════════════════════════════════════════════════════
# kpi_row (하위 호환)
# ═══════════════════════════════════════════════════════════
def kpi_row(items):
    metrics = []
    for item in items:
        if len(item) >= 3:
            label, value, delta = item[0], item[1], item[2]
            tone = item[3] if len(item) >= 4 else "neutral"
        else:
            label, value = item[0], item[1]
            delta = None
            tone = "neutral"

        metrics.append(dict(
            label=str(label).upper(),
            value=str(value),
            delta=delta,
            tone=tone,
        ))
    if metrics:
        _metric_row(metrics, cols=len(metrics))