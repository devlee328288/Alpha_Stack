# dashboard/services/model_service.py
"""
4모델 nested walk-forward 실행 서비스.
레포의 models.experiment 엔진을 호출.
scope/ticker 지원 (MARKET | STOCK).
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

import streamlit as st

MODELS = ("LogisticRegression", "RandomForest", "XGBoost", "LightGBM")
DEFAULT_COMBINATION = "E"
DEFAULT_RETURN_FEATURES = ("five_day_return",)

_IMPORT_ERR: str | None = None
try:
    from models.experiment import evaluate_nested_class_weights
    from models.notebook_experiment import summarize_notebook_experiment
    from features.model_dataset_stock import build_model_dataset_any
except Exception as e:
    _IMPORT_ERR = f"{type(e).__name__}: {e}"


def engine_status() -> str | None:
    return _IMPORT_ERR


# ═══════════════════════════════════════════════════════════
# 수동 캐시
# ═══════════════════════════════════════════════════════════
_MODEL_CACHE: dict = {}


def _silence():
    return (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    )


def _load_dataset(
    scope: str,
    ticker: str,
    combination: str,
    return_features: tuple,
    source: str = "stocks30",
):
    from services import data_loader

    _out, _err = _silence()
    with _out, _err:
        if source == "full":
            df = data_loader.load_market_data_full(ticker, price_col="adj_close")
        else:
            df = data_loader.load_market_data(scope, ticker)
        dataset = build_model_dataset_any(
            df, combination, return_features=return_features,
        )
    return df, dataset


def run_single_model(
    scope: str,
    ticker: str,
    model_name: str,
    combination: str = DEFAULT_COMBINATION,
    return_features: tuple = DEFAULT_RETURN_FEATURES,
    source: str = "stocks30",
) -> dict:
    if _IMPORT_ERR:
        raise RuntimeError(f"repo engine import 실패: {_IMPORT_ERR}")

    cache_key = (scope, ticker, model_name, combination,
                 tuple(return_features), source)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    df, dataset = _load_dataset(scope, ticker, combination, return_features, source)

    _out, _err = _silence()
    with _out, _err:
        nested = evaluate_nested_class_weights(
            dataset, model_names=(model_name,),
        )
        summary_obj = summarize_notebook_experiment(
            dataset, nested, model_name,
        )

    result = {
        "scope": scope,
        "ticker": ticker,
        "model_name": model_name,
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
        },
    }
    _MODEL_CACHE[cache_key] = result
    return result


def run_all_models(
    scope: str,
    ticker: str,
    models: tuple = MODELS,
    combination: str = DEFAULT_COMBINATION,
    return_features: tuple = DEFAULT_RETURN_FEATURES,
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> dict:
    out: dict = {}
    total = len(models)
    for i, name in enumerate(models):
        if progress_cb:
            progress_cb(i, total, name)
        out[name] = run_single_model(
            scope, ticker, name, combination, return_features,
        )
    if progress_cb:
        progress_cb(total, total, "done")
    return out


def clear_cache() -> None:
    _MODEL_CACHE.clear()