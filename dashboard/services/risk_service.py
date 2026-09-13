# dashboard/services/risk_service.py
"""
리스크 서비스 — evaluation/evaluation_risk.py, evaluation_backtest.py 호출.
이 두 모듈은 import 시점에 HF 다운로드가 돌기 때문에 stdout 리다이렉트 필수.
"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_EVAL_DIR = _REPO_ROOT / "evaluation"
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

import numpy as np

_IMPORT_ERR: str | None = None
_HAS_RISK = False
_HAS_CLS = False
_HAS_REG = False

try:
    _out, _err = contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO())
    with _out, _err:
        from evaluation_risk import calculate_all_metrics as _risk_all  # type: ignore
    _HAS_RISK = True
except Exception as e:
    _IMPORT_ERR = f"evaluation_risk: {type(e).__name__}: {e}"

try:
    _out2, _err2 = contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO())
    with _out2, _err2:
        from evaluation_backtest import (  # type: ignore
            calculate_all_classification_metrics as _cls_all,
        )
        from evaluation_backtest import (  # type: ignore
            calculate_all_regression_metrics as _reg_all,
        )
    _HAS_CLS = True
    _HAS_REG = True
except Exception as e:
    _HAS_CLS = False
    _HAS_REG = False
    if _IMPORT_ERR is None:
        _IMPORT_ERR = f"evaluation_backtest: {type(e).__name__}: {e}"


def engine_status() -> str | None:
    return _IMPORT_ERR


def risk_available() -> bool:
    return _HAS_RISK


def classification_available() -> bool:
    return _HAS_CLS


def _sanitize(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, (int, float, np.floating, np.integer)):
            fv = float(v)
            out[k] = fv if np.isfinite(fv) else None
        elif isinstance(v, (str, bool, type(None))):
            out[k] = v
    return out


def calculate_risk(
    returns,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
    n_trials: int = 50,
    sterling_top_k: int = 3,
) -> dict:
    if not _HAS_RISK:
        raise RuntimeError(f"risk import 실패: {_IMPORT_ERR}")

    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        raise ValueError("유효한 수익률이 없습니다.")

    _out, _err = contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO())
    with _out, _err:
        result = _risk_all(
            returns=arr,
            risk_free_rate=risk_free_rate,
            target_return=0.0,
            periods_per_year=periods_per_year,
            n_trials=n_trials,
            sterling_top_k=sterling_top_k,
        )
    return _sanitize(dict(result))


def calculate_classification(
    y_true, y_pred, y_pred_proba=None, labels=None,
) -> dict:
    if not _HAS_CLS:
        raise RuntimeError(f"classification import 실패: {_IMPORT_ERR}")

    _out, _err = contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO())
    with _out, _err:
        result = _cls_all(
            y_true=np.asarray(y_true),
            y_pred=np.asarray(y_pred),
            y_pred_proba=None if y_pred_proba is None else np.asarray(y_pred_proba),
            labels=list(labels) if labels else None,
        )

    out = {}
    for k, v in result.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, dict):
            out[k] = {str(kk): float(vv) if isinstance(vv, (int, float, np.floating)) else vv
                      for kk, vv in v.items()}
        elif isinstance(v, (int, float, np.floating, np.integer)):
            fv = float(v)
            out[k] = fv if np.isfinite(fv) else None
        else:
            out[k] = v
    return out


def regression_available() -> bool:
    return _HAS_REG


def calculate_regression(
    predictions,
    returns,
    ic_series=None,
) -> dict:
    """evaluation_backtest.calculate_all_regression_metrics 래퍼."""
    if not _HAS_REG:
        raise RuntimeError(f"regression import 실패: {_IMPORT_ERR}")

    _out, _err = contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO())
    with _out, _err:
        result = _reg_all(
            predictions=np.asarray(predictions, dtype=float),
            returns=np.asarray(returns, dtype=float),
            ic_series=None if ic_series is None else np.asarray(ic_series, dtype=float),
        )

    out = {}
    for k, v in result.items():
        if isinstance(v, (int, float, np.floating, np.integer)):
            fv = float(v)
            out[k] = fv if np.isfinite(fv) else None
        else:
            out[k] = v
    return out