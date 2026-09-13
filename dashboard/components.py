# dashboard/components.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from dash_config import tickers_for
from state import fingerprint, init_state
from theme import (
    metric_row as _metric_row,
    sidebar_header,
    sidebar_meta,
    sidebar_section,
)
from services import hf_service


_RESULTS_MAP = [
    ("Baseline",   "_baseline_results"),
    ("Model Lab",  "_model_lab_results"),
    ("Backtest",   "_bt_results"),
    ("Cost Sens",  "_cost_grid"),
]


def _scope_block() -> None:
    sidebar_section("SCOPE")
    scope = st.session_state.get("scope", "MARKET")
    ticker = st.session_state.get("ticker", "KOSPI200")

    c1, c2 = st.columns(2)
    if c1.button(
        "MARKET",
        key="_scope_market",
        type="primary" if scope == "MARKET" else "secondary",
        use_container_width=True,
    ):
        st.session_state["scope"] = "MARKET"
        st.session_state["ticker"] = "KOSPI200"
        st.rerun()

    if c2.button(
        "STOCK",
        key="_scope_stock",
        type="primary" if scope == "STOCK" else "secondary",
        use_container_width=True,
        disabled=True,
        help="V2 개별종목 확장에서 활성화됩니다.",
    ):
        pass

    # 현재 ticker 표시
    st.markdown(
        f'<div style="padding:6px 14px 2px 14px;font-family:var(--font-num);'
        f'font-size:11px;color:var(--text-secondary);">'
        f'<span style="color:var(--text-muted);">TICKER</span> &nbsp; {ticker}'
        f"</div>",
        unsafe_allow_html=True,
    )


def _hf_block() -> None:
    sidebar_section("HF SNAPSHOT")
    meta = hf_service.get_hf_meta()
    if meta.get("ok"):
        sidebar_meta([
            ("repo_sha", meta.get("repo_sha", "—")),
            ("dev_end", meta.get("dev_end", "—")),
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


def _is_ready(v) -> bool:
    """세션 값이 '결과 있음' 상태인지 안전하게 판정.
    DataFrame / Series / dict / list / None 다 커버."""
    if v is None:
        return False
    # DataFrame / Series
    if hasattr(v, "empty") and hasattr(v, "shape"):
        try:
            return not bool(v.empty)
        except Exception:
            return False
    # dict / list / set
    if hasattr(v, "__len__"):
        try:
            return len(v) > 0
        except Exception:
            return False
    return bool(v)


def _results_block() -> None:
    sidebar_section("RESULTS")
    rows_html = []
    for label, key in _RESULTS_MAP:
        has = _is_ready(st.session_state.get(key))
        dot_color = "var(--color-up)" if has else "var(--text-muted)"
        status = "ready" if has else "empty"
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


def sidebar_controls() -> bool:
    """
    A안 사이드바.
    이제 각 페이지가 자체 RUN 버튼을 가지므로 반환값은 사용되지 않음.
    """
    init_state()

    with st.sidebar:
        sidebar_header("QUANT RESEARCH TERMINAL")

        # SCOPE (V2 대비)
        _scope_block()

        # HF SNAPSHOT
        _hf_block()

        # RESULTS 상태
        _results_block()

        # THEME

        # FINGERPRINT
        st.markdown(
            f'<div style="padding:10px 14px 14px 14px;font-family:var(--font-num);'
            f'font-size:10px;color:var(--text-muted);'
            f'border-top:1px solid var(--border-subtle);margin-top:8px;">'
            f"FINGERPRINT &nbsp; {fingerprint()[:28]}"
            f"</div>",
            unsafe_allow_html=True,
        )

        return False


def kpi_row(items):
    """하위 호환용. 내부적으로 A안 metric_row 사용."""
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