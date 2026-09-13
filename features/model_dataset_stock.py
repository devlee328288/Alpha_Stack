# features/model_dataset_stock.py
"""
features/model_dataset.py 를 수정하지 않고 개별종목/유니버스를 지원하는 wrapper.

개별종목 DataFrame 을 KOSPI200 형식으로 위장해서 원본 함수에 전달.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from features.model_dataset import build_model_dataset as _build_model_dataset
from features.model_dataset import build_kospi200_feature_frame as _build_frame


def mimic_kospi200(df: pd.DataFrame) -> pd.DataFrame:
    """
    개별종목 DataFrame 을 원본 build_kospi200_feature_frame 이 통과시키도록
    KOSPI200 위장 컬럼 추가.
    """
    out = df.copy()

    # bas_dd (YYYYMMDD string)
    if "bas_dd" not in out.columns:
        if "date" in out.columns:
            out["bas_dd"] = pd.to_datetime(out["date"]).dt.strftime("%Y%m%d")
        elif isinstance(out.index, pd.DatetimeIndex):
            out["bas_dd"] = out.index.strftime("%Y%m%d")
        else:
            raise ValueError("bas_dd 또는 date 컬럼이 필요합니다.")

    # index_name / index_class 위장
    out["index_name"] = "코스피 200"
    out["index_class"] = "지수"

    # raw_position (0..N-1)
    if "raw_position" not in out.columns:
        out["raw_position"] = np.arange(len(out))

    # date 컬럼 제거 (원본이 bas_dd 로 재생성)
    if "date" in out.columns:
        out = out.drop(columns=["date"])

    return out.reset_index(drop=True)


def build_model_dataset_any(
    df: pd.DataFrame,
    combination: str,
    *,
    return_features: tuple = (),
    holdout_start: str | None = None,
):
    """
    지수/개별종목 무관 wrapper.

    Parameters
    ----------
    df : DataFrame
        지수면 그대로, 개별종목이면 stocks30 에서 필터한 것.
        (DatetimeIndex 또는 date 컬럼 무관)
    combination : "A"~"F"
    return_features : ("five_day_return",) 등
    holdout_start : None 이면 원본 디폴트
    """
    df_mimic = mimic_kospi200(df)

    kwargs = {"return_features": return_features}
    if holdout_start is not None:
        kwargs["holdout_start"] = holdout_start

    return _build_model_dataset(df_mimic, combination, **kwargs)


def build_feature_frame_any(
    df: pd.DataFrame,
    *,
    holdout_start: str | None = None,
):
    """build_kospi200_feature_frame 의 wrapper."""
    df_mimic = mimic_kospi200(df)
    kwargs = {}
    if holdout_start is not None:
        kwargs["holdout_start"] = holdout_start
    return _build_frame(df_mimic, **kwargs)