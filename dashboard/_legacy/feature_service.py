# dashboard/services/feature_service.py
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]   # services → dashboard → Alpha_Stack
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import streamlit as st
from dash_config import HORIZON

# ── 실제 엔진 시도 ──────────────────────────────────────────
_ENGINE_ERROR: str | None = None
try:
    from features.model_dataset import (
        build_kospi200_feature_frame,
    )
    from features.model_dataset import (
        build_model_dataset as _real_build_model_dataset,
    )
    _HAS_ENGINE = True
except Exception as _e:
    _HAS_ENGINE = False
    _ENGINE_ERROR = f"features.model_dataset import 실패: {_e}"


# ── fallback: CSV의 한글 라벨 직접 매핑 ─────────────────────
_LABEL_MAP = {
    "상승": 1, "up": 1, "bull": 1, "buy": 1,
    "보합": 0, "중립": 0, "neutral": 0, "hold": 0,
    "하락": -1, "down": -1, "bear": -1, "sell": -1,
}
_LEAKAGE = {"label", "fwd_return_5d", "bas_dd", "label_numeric", "raw_position"}
_META = {"index_name", "index_class"}


def _align_xy(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    X = X.replace([np.inf, -np.inf], np.nan)
    data = X.join(y.rename("__y__")).dropna()
    return data[X.columns].astype("float64"), data["__y__"].astype(int)


def _build_fallback(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if "label" not in df.columns:
        raise ValueError(f"'label' 컬럼 없음. 컬럼: {list(df.columns)[:30]}")
    raw = df["label"].astype(str).str.strip()
    y = raw.map(_LABEL_MAP)
    if y.isna().any():
        unknowns = sorted(raw[y.isna()].dropna().unique().tolist())
        raise ValueError(f"매핑되지 않은 라벨: {unknowns}")
    exclude = _LEAKAGE | _META
    feat_cols = [c for c in df.columns
                 if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    if not feat_cols:
        raise ValueError("숫자형 피처가 없습니다.")
    return _align_xy(df[feat_cols], y.astype(int))


# ── 최종 진입점 ──────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def build_dataset(df: pd.DataFrame, feature_set: str, horizon: int = HORIZON):
    """OHLCV(+피처/라벨) → (X, y)."""
    if _HAS_ENGINE:
        key = feature_set if feature_set in list("ABCDEF") else "F"
        try:
            return _build_real(df, key)
        except Exception as e:
            global _ENGINE_ERROR
            _ENGINE_ERROR = f"build_model_dataset({key}) 실패 → fallback: {e}"
    return _build_fallback(df)


# ── 인덱스를 날짜로 강제하는 헬퍼 ──────────────────────────
def _ensure_date_index(X: pd.DataFrame, y: pd.Series,
                       src_df: pd.DataFrame,
                       frame: pd.DataFrame | None = None):
    """X, y 의 인덱스를 DatetimeIndex 로 만든다.
    우선순위: 이미 날짜 → frame['bas_dd'] → src_df.index(위치 기반)"""
    if isinstance(X.index, pd.DatetimeIndex):
        return X, y

    # 1) 엔진 frame 에 bas_dd 가 있으면 사용
    if frame is not None and "bas_dd" in frame.columns and len(frame) == len(X):
        try:
            dates = pd.to_datetime(frame["bas_dd"].astype(str).values)
            X = X.copy(); y = y.copy()
            X.index = dates
            y.index = dates
            return X, y
        except Exception:
            pass

    # 2) 원본 df 와 길이가 같으면 위치 기반 매핑
    if len(src_df) == len(X) and isinstance(src_df.index, pd.DatetimeIndex):
        X = X.copy(); y = y.copy()
        X.index = src_df.index
        y.index = src_df.index
        return X, y

    # 3) 매핑 불가 → 명시적 에러
    raise ValueError(
        f"X({len(X)}) 의 인덱스를 날짜로 복원할 수 없습니다. "
        f"src_df({len(src_df)}) · frame({len(frame) if frame is not None else 'None'})"
    )


def _build_real(df: pd.DataFrame, combination: str) -> tuple[pd.DataFrame, pd.Series]:
    ff = build_kospi200_feature_frame(df)
    ds = _real_build_model_dataset(ff, combination)

    # ── 조합 필터 무시하고 frame 전체에서 피처 추출 ────────
    frame = ds.frame
    if "label_numeric" not in frame.columns:
        # label_numeric 없으면 ds.x/ds.y 로 폴백
        X = ds.x.copy()
        y = pd.Series(np.asarray(ds.y, dtype=int), index=X.index, name="y")
        X, y = _ensure_date_index(X, y, df, frame=frame)
        return _align_xy(X, y)

    # 라벨 컬럼
    y = frame["label_numeric"].astype(int).copy()

    # 제외: 라벨·미래정보·식별자
    _EXCLUDE = {
        "label_numeric", "label", "raw_position",
        "fwd_return_5d", "fwd_return_1d", "bas_dd",
        "index_name", "index_class",
    }
    feat_cols = [
        c for c in frame.columns
        if c not in _EXCLUDE and pd.api.types.is_numeric_dtype(frame[c])
    ]
    X = frame[feat_cols].copy()

    X, y = _ensure_date_index(X, y, df, frame=frame)
    return _align_xy(X, y)


def engine_status() -> str | None:
    """디버깅용: 실 엔진 사용 여부와 에러 메시지."""
    if _HAS_ENGINE and _ENGINE_ERROR is None:
        return None
    return _ENGINE_ERROR or "features.model_dataset 를 불러오지 못했습니다."
