# dashboard/services/baseline_service.py
"""
Baseline 서비스 — 6-param threshold walk-forward (+옵션 Focal).
레포의 evaluation/threshold_tuning 엔진을 호출.

주의: Streamlit @st.cache_data 를 쓰지 않는다 (stdout ASCII 캡처 문제 회피).
"""
from __future__ import annotations

import contextlib
import io
import sys
from datetime import datetime
from pathlib import Path

# stdout/stderr reconfigure (Streamlit 이전에 미리)
try:
    for _s in (sys.stdout, sys.stderr):
        if hasattr(_s, "reconfigure"):
            _s.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_TH_DIR = _REPO_ROOT / "evaluation" / "threshold_tuning"
if str(_TH_DIR) not in sys.path:
    sys.path.insert(0, str(_TH_DIR))

import numpy as np
import pandas as pd
import streamlit as st

_IMPORT_ERR: str | None = None
_HAS_FOCAL = False

try:
    from step1_core_features import load_data                    # type: ignore
    from step5_optimize_6params import run_walkforward_6params   # type: ignore
    from focal_classifier import FocalConfig, make_labels        # type: ignore
    try:
        from step7_focal_walkforward import (                     # type: ignore
            evaluate_signals, generate_signals_rolling,
        )
        _HAS_FOCAL = True
    except Exception:
        pass
except Exception as e:
    _IMPORT_ERR = f"{type(e).__name__}: {e}"


def engine_status() -> str | None:
    return _IMPORT_ERR


def focal_available() -> bool:
    return _HAS_FOCAL


# ═══════════════════════════════════════════════════════════
# 수동 캐시 (Streamlit caching 우회)
# ═══════════════════════════════════════════════════════════
_DF_CACHE: dict = {}
_BASELINE_CACHE: dict = {}


def _silence():
    return (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    )


def _load_df() -> pd.DataFrame:
    if "df" not in _DF_CACHE:
        _out, _err = _silence()
        with _out, _err:
            _DF_CACHE["df"] = load_data()
    return _DF_CACHE["df"]


def _sanitize_metrics(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, (int, float, np.floating, np.integer)):
            out[k] = float(v)
        elif isinstance(v, (str, bool, type(None))):
            out[k] = v
    return out


def run_baseline(
    threshold: float = 0.01,
    max_evals: int = 30,
    include_focal: bool = False,
) -> dict:
    if _IMPORT_ERR:
        raise RuntimeError(f"repo engine import 실패: {_IMPORT_ERR}")

    cache_key = (float(threshold), int(max_evals), bool(include_focal))
    if cache_key in _BASELINE_CACHE:
        return _BASELINE_CACHE[cache_key]

    df = _load_df()

    _out, _err = _silence()
    with _out, _err:
        step5 = run_walkforward_6params(
            df,
            train_years=2,
            val_months=3,
            step_months=1,
            max_evals=max_evals,
            threshold=threshold,
        )

    out: dict = {
        "threshold": float(threshold),
        "max_evals": int(max_evals),
        "include_focal": bool(include_focal),
        "total_folds": int(step5["total_folds"]),
        "perf_metrics": _sanitize_metrics(step5["perf_metrics"]),
        "cls_metrics": _sanitize_metrics(step5["cls_metrics"]),
        "params_median": _sanitize_metrics(step5["params_median"]),
        "fold_details": step5["fold_details"].to_dict("records"),
        "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    if include_focal and _HAS_FOCAL:
        try:
            _out2, _err2 = _silence()
            with _out2, _err2:
                cfg = FocalConfig(threshold=float(threshold))
                signals = generate_signals_rolling(
                    df, step5["fold_details"], focal_config=cfg,
                )
                y_true = make_labels(df, threshold=float(threshold)).to_numpy()
                metrics = evaluate_signals(signals, y_true)

            out["focal"] = _sanitize_metrics(metrics)
            out["focal_signals_tail"] = (
                signals.tail(30)
                .reset_index()
                .rename(columns={"index": "date"})
                .to_dict("records")
            )
        except Exception as e:
            out["focal_error"] = f"{type(e).__name__}: {e}"

    _BASELINE_CACHE[cache_key] = out
    return out


def clear_cache() -> None:
    _DF_CACHE.clear()
    _BASELINE_CACHE.clear()


SESSION_KEY = "_baseline_results"


def save_results(results: dict) -> None:
    st.session_state[SESSION_KEY] = results


def load_results() -> dict | None:
    return st.session_state.get(SESSION_KEY)


def clear_results() -> None:
    st.session_state.pop(SESSION_KEY, None)