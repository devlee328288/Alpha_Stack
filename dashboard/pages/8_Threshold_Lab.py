import sys
from pathlib import Path

# dashboard/ 루트
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 레포 루트
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# threshold_tuning 폴더 자체를 path에 (step1~7 상호 import)
_TH_DIR = _REPO_ROOT / "evaluation" / "threshold_tuning"
if str(_TH_DIR) not in sys.path:
    sys.path.insert(0, str(_TH_DIR))

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Threshold Lab · AlphaStack", layout="wide")

import theme
from theme import (
    metric_row,
    panel,
    section_header,
    top_strip,
)

theme.inject()

import charts
from components import sidebar_controls
from services import data_service
from state import ctx

sidebar_controls()
c = ctx()

# ═══════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════
st.markdown('<div class="as-title">Threshold Lab</div>', unsafe_allow_html=True)

prices = data_service.load_market_data(c["dataset"], c["start"], c["end"])

# ── 엔진 import ───────────────────────────────────────────
_HAS_TH = False
_ERR_TH = None
try:
    from step5_optimize_6params import (  # type: ignore
        calculate_metrics as _real_metrics,
    )
    from step5_optimize_6params import (  # type: ignore
        compute_bands_flexible as _real_bands,
    )
    from step5_optimize_6params import (  # type: ignore
        get_positions_6params as _real_positions,
    )
    _HAS_TH = True
except Exception as e:
    _ERR_TH = str(e)

if not _HAS_TH:
    top_strip([c["dataset"].upper(), "THRESHOLD ENGINE"],
              status_text="IMPORT FAIL", status_tone="warn")
    st.warning(f"⚠ threshold_tuning import 실패: {_ERR_TH}")
    st.stop()

top_strip(
    [c["dataset"].upper(),
     f"{c['start']} – {c['end']}",
     "6-PARAM THRESHOLD"],
    status_text="READY", status_tone="up",
)

# ═══════════════════════════════════════════════════════════
# ① PARAMETERS
# ═══════════════════════════════════════════════════════════
section_header("6-PARAMETER THRESHOLD")
with panel("PARAMETERS"):
    p1, p2, p3 = st.columns(3, gap="small")
    p4, p5, p6 = st.columns(3, gap="small")

    with p1:
        alpha_up = st.slider("alpha_up", 0.0, 2.0, 0.5, 0.05)
    with p2:
        alpha_down = st.slider("alpha_down", 0.0, 2.0, 0.5, 0.05)
    with p3:
        beta_up = st.slider("beta_up", 0.0, 2.0, 0.5, 0.05)
    with p4:
        beta_down = st.slider("beta_down", 0.0, 2.0, 0.5, 0.05)
    with p5:
        vol_period = st.slider("vol_period", 5, 60, 20, 1)
    with p6:
        volume_period = st.slider("volume_period", 5, 60, 20, 1)

    st.write("")
    run = st.button("▶  RUN THRESHOLD ANALYSIS",
                    type="primary", use_container_width=False)

# ═══════════════════════════════════════════════════════════
# ② RUN
# ═══════════════════════════════════════════════════════════
if run:
    with st.spinner("Running 6-param threshold..."):
        try:
            pos_result = _real_positions(
                prices,
                alpha_up, alpha_down, beta_up, beta_down,
                vol_period, volume_period,
            )

            # 반환: tuple 이면 첫 원소가 포지션
            if isinstance(pos_result, tuple):
                pos = pos_result[0]
                extras = pos_result[1:]
            else:
                pos = pos_result
                extras = ()

            pos_series = (
                pos if isinstance(pos, pd.Series)
                else pd.Series(np.asarray(pos), index=prices.index)
            )

            ret = prices["close"].pct_change().fillna(0)
            strat = pos_series.shift(1).fillna(0) * ret
            eq = (1 + strat).cumprod()

            # ── KPI (metrics 계산 가능하면) ────────────────
            try:
                m = _real_metrics(strat.values)
                if isinstance(m, dict):
                    items = []
                    for k, v in list(m.items())[:6]:
                        try:
                            fv = float(v)
                            tone = ("up" if fv > 0 else
                                    ("down" if fv < 0 else "neutral"))
                            items.append(dict(
                                label=str(k).upper().replace("_", " "),
                                value=f"{fv:.3f}",
                                tone=tone,
                            ))
                        except Exception:
                            items.append(dict(
                                label=str(k).upper().replace("_", " "),
                                value=str(v),
                            ))
                    if items:
                        section_header("PERFORMANCE METRICS")
                        metric_row(items, cols=min(len(items), 6))
            except Exception as e:
                st.caption(f"metrics skip: {e}")

            # ── Position ────────────────────────────────────
            section_header("POSITION")
            with panel("SIGNAL POSITION"):
                st.line_chart(pos_series.rename("position"), height=300)

            # ── Equity ──────────────────────────────────────
            section_header("EQUITY")
            with panel("THRESHOLD STRATEGY", "LIVE", "up"):
                st.plotly_chart(
                    charts.equity_curve({"Threshold": eq}, ""),
                    use_container_width=True,
                )

            # ── Extra outputs (bands 등) ────────────────────
            for i, ex in enumerate(extras):
                if isinstance(ex, pd.DataFrame):
                    section_header(f"EXTRA OUTPUT #{i+1}")
                    with panel():
                        st.dataframe(ex.tail(30), use_container_width=True)
                elif isinstance(ex, pd.Series):
                    section_header(f"EXTRA OUTPUT #{i+1}")
                    with panel():
                        st.line_chart(ex.rename(f"extra_{i+1}"), height=240)

        except Exception as e:
            st.error(f"포지션 계산 실패: {e}")
            st.caption(
                "시그니처: get_positions_6params("
                "df, alpha_up, alpha_down, beta_up, beta_down, "
                "vol_period, volume_period) -> tuple")

# ═══════════════════════════════════════════════════════════
# ③ DEV / REFERENCE (접힌 상태)
# ═══════════════════════════════════════════════════════════
section_header("ENGINE REFERENCE")
col_a, col_b = st.columns(2, gap="small")

with col_a:
    with panel("compute_bands_flexible SIGNATURE"):
        try:
            import inspect
            sig = inspect.signature(_real_bands)
            st.code(f"compute_bands_flexible{sig}", language="python")
        except Exception as e:
            st.caption(f"inspect 실패: {e}")

with col_b:
    with panel("FUNCTIONS"):
        st.markdown(
            "**get_positions_6params** — "
            "`(df, alpha_up, alpha_down, beta_up, beta_down, "
            "vol_period, volume_period) -> tuple`\n\n"
            "**calculate_metrics** — returns → 지표 dict\n\n"
            "**compute_bands_flexible** — 동적 임계값 밴드"
        )
        st.caption("원본: `evaluation/threshold_tuning/step5_optimize_6params.py`")
