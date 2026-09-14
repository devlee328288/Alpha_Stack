# features/model_dataset_baseline_labeled.py
from __future__ import annotations
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
from features.model_dataset_stock import build_model_dataset_any as _build_any


def build_model_dataset_with_baseline_labels(
    df,
    combination,
    *,
    return_features=(),
    holdout_start=None,
    baseline_labels=None,
):
    """baseline_labels=None 이면 원본 라벨 유지. 값이 있으면 오버라이드."""
    ds = _build_any(
        df,
        combination,
        return_features=return_features,
        holdout_start=holdout_start,
    )
    if baseline_labels is None:
        return ds

    if isinstance(baseline_labels, pd.Series):
        bl_ser = baseline_labels.copy()
        if not isinstance(bl_ser.index, pd.DatetimeIndex):
            try:
                bl_ser.index = pd.to_datetime(bl_ser.index)
            except Exception:
                return ds
    else:
        bl_ser = pd.Series(np.asarray(baseline_labels), index=pd.to_datetime(df.index))

    try:
        frame_dates = pd.to_datetime(ds.frame["bas_dd"].astype(str), format="%Y%m%d")
    except Exception:
        return ds

    aligned = bl_ser.reindex(frame_dates.values)
    new_labels = np.asarray(aligned.values, dtype="float64")
    mask_arr = np.isfinite(new_labels)
    if not mask_arr.all():
        drop_idx = ds.frame.index[~mask_arr]
        ds.frame.drop(index=drop_idx, inplace=True)
        new_labels = new_labels[mask_arr]
        ds.frame.reset_index(drop=True, inplace=True)

    labels_ml = new_labels.astype(np.int64) - 1
    ds.frame["label_numeric"] = pd.array(labels_ml, dtype="Int64")
    return ds
