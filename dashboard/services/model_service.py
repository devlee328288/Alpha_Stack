# dashboard/services/model_service.py
"""
4모델 nested walk-forward 실행 서비스.

- combination: scope 별 자동 선택 (combo_config)
  · MARKET   → C
  · STOCK    → K
  · UNIVERSE → K
- label_source: "fwd_return" (기본) | "adaptive" (per-fold baseline 판정)
"""

from __future__ import annotations
import contextlib
import io
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from typing import Callable
import numpy as np
import pandas as pd

MODELS = ("LogisticRegression", "RandomForest", "XGBoost", "LightGBM")

_IMPORT_ERR: str | None = None
try:
    from models.experiment import evaluate_nested_class_weights
    from models.notebook_experiment import summarize_notebook_experiment
    from features.model_dataset_baseline_labeled import (
        build_model_dataset_with_baseline_labels as _build_dataset_with_labels,
    )
except Exception as e:
    _IMPORT_ERR = f"{type(e).__name__}: {e}"


def engine_status() -> str | None:
    return _IMPORT_ERR


_MODEL_CACHE: dict = {}


def _silence():
    return (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    )


def _apply_label_override(dataset, df, label_array):
    """dataset.frame['label_numeric'] 을 label_array 로 오버라이드."""
    if label_array is None:
        return dataset

    arr = np.asarray(label_array, dtype="float64")
    if len(arr) != len(df):
        raise ValueError(f"label_array 길이({len(arr)}) != df({len(df)})")

    try:
        df_dates = pd.to_datetime(df.index).strftime("%Y%m%d").values
        frame_dates = dataset.frame["bas_dd"].astype(str).values
    except Exception as e:
        raise ValueError(f"bas_dd 매핑 실패: {e}")

    label_map = dict(zip(df_dates, arr))
    new_labels = np.array(
        [label_map.get(bd, np.nan) for bd in frame_dates],
        dtype="float64",
    )

    mask = np.isfinite(new_labels)
    if int(mask.sum()) == 0:
        raise ValueError(
            f"오버라이드 후 유효 라벨 0개. "
            f"(df rows={len(df)}, frame rows={len(frame_dates)}, "
            f"arr non-nan={int(np.isfinite(arr).sum())})"
        )

    if not mask.all():
        drop_idx = dataset.frame.index[~mask]
        dataset.frame.drop(index=drop_idx, inplace=True)
        new_labels = new_labels[mask]
        dataset.frame.reset_index(drop=True, inplace=True)

    # {0,1,2} → {-1,0,1}
    labels_ml = new_labels.astype(np.int64) - 1
    dataset.frame["label_numeric"] = pd.array(labels_ml, dtype="Int64")
    return dataset


def _load_dataset(
    scope,
    ticker,
    combination,
    return_features,
    source="stocks30",
    label_override=None,
):
    from services import data_loader

    _out, _err = _silence()
    with _out, _err:
        if source == "full":
            df = data_loader.load_market_data_full(ticker, price_col="adj_close")
        else:
            df = data_loader.load_market_data(scope, ticker)

        dataset = _build_dataset_with_labels(
            df,
            combination,
            return_features=return_features,
            baseline_labels=None,
        )

    if label_override is not None:
        dataset = _apply_label_override(dataset, df, label_override)

    return df, dataset


def run_single_model(
    scope: str,
    ticker: str,
    model_name: str,
    combination: str | None = None,  # None → combo_config 자동
    return_features: tuple | None = None,  # None → combo_config 자동
    source: str = "stocks30",
    label_source: str = "fwd_return",
    adaptive_labels=None,
) -> dict:
    if _IMPORT_ERR:
        raise RuntimeError(f"repo engine import 실패: {_IMPORT_ERR}")

    from services.combo_config import combination_of, return_features_of

    if combination is None:
        combination = combination_of(scope)
    if return_features is None:
        return_features = return_features_of(scope)

    cache_key = (
        scope,
        ticker,
        model_name,
        combination,
        tuple(return_features),
        source,
        label_source,
    )
    if label_source == "adaptive" and adaptive_labels is not None:
        arr = np.asarray(adaptive_labels, dtype="float64")
        fingerprint = (
            int(np.isfinite(arr).sum()),
            tuple(np.nan_to_num(arr[:10], nan=-999).round(4).tolist()),
        )
        cache_key = cache_key + (fingerprint,)

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    override = adaptive_labels if label_source == "adaptive" else None

    df, dataset = _load_dataset(
        scope,
        ticker,
        combination,
        return_features,
        source,
        label_override=override,
    )

    _out, _err = _silence()
    with _out, _err:
        nested = evaluate_nested_class_weights(dataset, model_names=(model_name,))
        summary_obj = summarize_notebook_experiment(dataset, nested, model_name)

    result = {
        "model_name": model_name,
        "label_source": label_source,
        "combination": combination,
        "summary": dict(summary_obj.summary),
        "fold_results": summary_obj.fold_results.to_dict("records"),
        "inner_results": summary_obj.inner_results.to_dict("records"),
        "weight_counts": summary_obj.weight_counts.to_dict("records"),
        "confusion": summary_obj.confusion.to_dict(),
        "class_report": summary_obj.class_report.to_dict(),
        "oos_predictions": nested.oos_predictions.to_dict("records"),
        "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": {
            "scope": scope,
            "ticker": ticker,
            "dataset_rows": len(dataset.frame),
            "n_features": len(dataset.feature_columns),
        },
    }
    _MODEL_CACHE[cache_key] = result
    return result


def run_all_models(
    scope: str,
    ticker: str,
    models: tuple = MODELS,
    combination: str | None = None,
    return_features: tuple | None = None,
    source: str = "stocks30",
    label_source: str = "fwd_return",
    adaptive_labels=None,
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> dict:
    out: dict = {}
    total = len(models)
    for i, name in enumerate(models):
        if progress_cb:
            progress_cb(i, total, name)
        out[name] = run_single_model(
            scope,
            ticker,
            name,
            combination=combination,
            return_features=return_features,
            source=source,
            label_source=label_source,
            adaptive_labels=adaptive_labels,
        )
    if progress_cb:
        progress_cb(total, total, "done")
    return out


def clear_cache() -> None:
    _MODEL_CACHE.clear()
