# dashboard/services/data_service.py
from __future__ import annotations

import sys
from pathlib import Path

_DASH_ROOT = Path(__file__).resolve().parents[1]
if str(_DASH_ROOT) not in sys.path:
    sys.path.insert(0, str(_DASH_ROOT))

from datetime import date

import app_streamlit_data as hf_data
import pandas as pd
import streamlit as st

# ── 데이터셋 alias: 지금 HF 저장소엔 KOSPI200 지수 하나뿐 ──────
_DATASET_ALIAS = {
    "KOSPI200": "KOSPI200",
    "KOSPI":    "KOSPI200",   # 같은 파일로 매핑 (추후 실제 KOSPI 파일 생기면 교체)
    "KOSPI200_DEV": "KOSPI200",
}


@st.cache_data(show_spinner=False)
def load_market_data(dataset: str, start: date, end: date) -> pd.DataFrame:
    key = _DATASET_ALIAS.get(dataset)
    if key is None:
        raise ValueError(
            f"지원하지 않는 dataset: {dataset}\n"
            f"가능: {list(_DATASET_ALIAS)}"
        )

    df = hf_data.load_kospi200()
    df = df.loc[str(start):str(end)]
    if df.empty:
        raise ValueError(f"{dataset}에 {start}~{end} 구간 데이터가 없습니다.")
    return df


@st.cache_data(show_spinner=False)
def list_stocks() -> list[str]:
    return hf_data.list_stocks()


@st.cache_data(show_spinner=False)
def load_stock(code: str, start: date | None = None, end: date | None = None) -> pd.DataFrame:
    df = hf_data.load_stock(code)
    if start is not None and end is not None:
        df = df.loc[str(start):str(end)]
    return df
