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

import charts
from components import kpi_row, sidebar_controls
from services import data_service
from state import ctx

sidebar_controls()
c = ctx()
st.title("Threshold Lab")

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
    st.warning(f"⚠ threshold_tuning import 실패: {_ERR_TH}")
    st.stop()

# ── 파라미터 (실 엔진 시그니처 순서) ──────────────────────
st.markdown("#### 6-Parameter Threshold")
p1, p2, p3 = st.columns(3)
p4, p5, p6 = st.columns(3)

alpha_up      = p1.slider("alpha_up",       0.0, 2.0, 0.5, 0.05)
alpha_down    = p2.slider("alpha_down",     0.0, 2.0, 0.5, 0.05)
beta_up       = p3.slider("beta_up",        0.0, 2.0, 0.5, 0.05)
beta_down     = p4.slider("beta_down",      0.0, 2.0, 0.5, 0.05)
vol_period    = p5.slider("vol_period",     5, 60, 20, 1)
volume_period = p6.slider("volume_period",  5, 60, 20, 1)

run = st.button("Run Threshold Analysis", type="primary")

if run:
    with st.spinner("Running 6-param threshold..."):
        # ── positions ─────────────────────────────────────
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

            pos_series = (pos if isinstance(pos, pd.Series)
                          else pd.Series(np.asarray(pos), index=prices.index))

            st.markdown("**Position**")
            st.line_chart(pos_series.rename("position"), height=320)

            ret = prices["close"].pct_change().fillna(0)
            strat = pos_series.shift(1).fillna(0) * ret
            eq = (1 + strat).cumprod()

            st.markdown("**Equity**")
            st.plotly_chart(
                charts.equity_curve({"Threshold": eq}, "Equity"),
                use_container_width=True)

            # ── metrics ───────────────────────────────────
            try:
                m = _real_metrics(strat.values)
                if isinstance(m, dict):
                    items = []
                    for k, v in list(m.items())[:6]:
                        try:
                            items.append((k, f"{float(v):.3f}"))
                        except Exception:
                            items.append((k, str(v)))
                    kpi_row(items)
            except Exception as e:
                st.caption(f"metrics skip: {e}")

            # tuple 추가 반환값 (bands 등)
            for i, ex in enumerate(extras):
                if isinstance(ex, pd.DataFrame):
                    st.markdown(f"**Extra output #{i+1}**")
                    st.dataframe(ex.tail(30), use_container_width=True)
                elif isinstance(ex, pd.Series):
                    st.markdown(f"**Extra output #{i+1}**")
                    st.line_chart(ex.rename(f"extra_{i+1}"), height=240)

        except Exception as e:
            st.error(f"포지션 계산 실패: {e}")
            st.caption(
                "시그니처: get_positions_6params("
                "df, alpha_up, alpha_down, beta_up, beta_down, "
                "vol_period, volume_period) -> tuple")

# ── bands (별도 시도, 시그니처 다르면 스킵) ───────────────
with st.expander("compute_bands_flexible 시그니처 확인"):
    try:
        import inspect
        sig = inspect.signature(_real_bands)
        st.code(f"compute_bands_flexible{sig}")
    except Exception as e:
        st.caption(f"inspect 실패: {e}")

with st.expander("실 엔진 함수 정보"):
    st.write("**get_positions_6params** — "
             "`(df, alpha_up, alpha_down, beta_up, beta_down, "
             "vol_period, volume_period) -> tuple`")
    st.write("**calculate_metrics** — returns → 지표 dict")
    st.write("**compute_bands_flexible** — 동적 임계값 밴드")
    st.caption("원본: `evaluation/threshold_tuning/step5_optimize_6params.py`")
