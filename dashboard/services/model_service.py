# dashboard/services/model_service.py
"""
4모델 nested walk-forward 실행 서비스.
레포의 models.experiment 엔진을 호출. UI 로직 없음.

주의: Streamlit @st.cache_data / @st.cache_resource 를 쓰지 않는다.
      Streamlit의 캐시 래퍼가 stdout 을 ASCII 로 잡아서 repo 코드의
      emoji/한글 print("① ...") 가 UnicodeEncodeError 로 터지는 문제가
      있었음. 대신 모듈 레벨 dict 로 수동 캐시.
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
    from supply.hf_model_data import load_hf_index_prices
    from features.model_dataset import build_model_dataset
    from models.experiment import evaluate_nested_class_weights
    from models.notebook_experiment import summarize_notebook_experiment
except Exception as e:
    _IMPORT_ERR = f"{type(e).__name__}: {e}"


def engine_status() -> str | None:
    return _IMPORT_ERR


# ═══════════════════════════════════════════════════════════
# 수동 캐시 (Streamlit caching 우회)
# ═══════════════════════════════════════════════════════════
_DS_CACHE: dict = {}
_MODEL_CACHE: dict = {}


def _silence_stdout():
    """repo 코드의 print 를 StringIO 로 흡수하는 컨텍스트 매니저."""
    return contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO())


def _load_dataset(combination: str, return_features: tuple[str, ...]):
    key = (combination, tuple(return_features))
    if key in _DS_CACHE:
        return _DS_CACHE[key]

    _out, _err = _silence_stdout()
    with _out, _err:
        snapshot = load_hf_index_prices()
        dataset = build_model_dataset(
            snapshot.frame,
            combination,
            return_features=return_features,
        )

    _DS_CACHE[key] = (snapshot, dataset)
    return _DS_CACHE[key]


def run_single_model(
    model_name: str,
    combination: str = DEFAULT_COMBINATION,
    return_features: tuple[str, ...] = DEFAULT_RETURN_FEATURES,
) -> dict:
    if _IMPORT_ERR:
        raise RuntimeError(f"repo engine import 실패: {_IMPORT_ERR}")

    cache_key = (model_name, combination, tuple(return_features))
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    snapshot, dataset = _load_dataset(combination, return_features)

    _out, _err = _silence_stdout()
    with _out, _err:
        nested = evaluate_nested_class_weights(
            dataset, model_names=(model_name,),
        )
        summary_obj = summarize_notebook_experiment(
            dataset, nested, model_name,
        )

    result = {
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
            "repo_sha": snapshot.repo_sha,
            "index_sha256": snapshot.file_sha256,
            "dev_end": snapshot.dev_end,
        },
    }
    _MODEL_CACHE[cache_key] = result
    return result


def run_all_models(
    models: tuple[str, ...] = MODELS,
    combination: str = DEFAULT_COMBINATION,
    return_features: tuple[str, ...] = DEFAULT_RETURN_FEATURES,
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    total = len(models)
    for i, name in enumerate(models):
        if progress_cb:
            progress_cb(i, total, name)
        out[name] = run_single_model(name, combination, return_features)
    if progress_cb:
        progress_cb(total, total, "done")
    return out


def clear_cache() -> None:
    """디버깅용: 메모리 캐시 초기화."""
    _DS_CACHE.clear()
    _MODEL_CACHE.clear()


# ═══════════════════════════════════════════════════════════
# Session state 저장/로드
# ═══════════════════════════════════════════════════════════
SESSION_KEY = "_model_lab_results"


def save_results(results: dict[str, dict]) -> None:
    st.session_state[SESSION_KEY] = results


def load_results() -> dict[str, dict] | None:
    return st.session_state.get(SESSION_KEY)


def clear_results() -> None:
    st.session_state.pop(SESSION_KEY, None)