# dashboard/services/system_service.py
"""
System 상태 수집 — HF 스냅샷, 엔진 상태, 패키지 버전, 세션 정보.
"""
from __future__ import annotations

import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st


REPO_ID = "qurious-quant/alphastack-krx-dev"
CACHE_DIR = _REPO_ROOT / "data" / "raw" / "hf_cache"

# 표시할 패키지 (없으면 skip)
_PACKAGES = [
    "streamlit",
    "pandas",
    "numpy",
    "plotly",
    "scikit-learn",
    "torch",
    "cma",
    "lightgbm",
    "xgboost",
    "huggingface-hub",
    "datasets",
    "exchange-calendars",
    "scipy",
    "tqdm",
]


def hf_snapshot() -> dict:
    """HF 스냅샷 상세 (session_state 우선)."""
    from services import hf_service

    meta = hf_service.get_hf_meta()
    return {
        "repo_id": REPO_ID,
        "repo_sha": meta.get("repo_sha", "—"),
        "dev_end": meta.get("dev_end", "—"),
        "index_sha256": meta.get("file_sha", "—"),
        "source": meta.get("source", "—"),
        "ok": meta.get("ok", False),
        "err": meta.get("err"),
        "cache_dir": str(CACHE_DIR),
        "cache_exists": CACHE_DIR.exists(),
    }


def _import_status() -> dict:
    """각 서비스 import 성공/실패."""
    out = {}

    try:
        from services import baseline_service
        err = baseline_service.engine_status()
        out["baseline_service"] = {
            "ok": err is None,
            "err": err,
            "focal_available": baseline_service.focal_available(),
        }
    except Exception as e:
        out["baseline_service"] = {"ok": False, "err": f"{type(e).__name__}: {e}"}

    try:
        from services import model_service
        err = model_service.engine_status()
        out["model_service"] = {"ok": err is None, "err": err}
    except Exception as e:
        out["model_service"] = {"ok": False, "err": f"{type(e).__name__}: {e}"}

    try:
        from services import backtest_service
        err = backtest_service.engine_status()
        out["backtest_service"] = {
            "ok": err is None,
            "err": err,
            "cost_available": backtest_service.cost_available(),
        }
    except Exception as e:
        out["backtest_service"] = {"ok": False, "err": f"{type(e).__name__}: {e}"}

    try:
        from services import risk_service
        err = risk_service.engine_status()
        out["risk_service"] = {
            "ok": risk_service.risk_available(),
            "err": err,
            "classification_available": risk_service.classification_available(),
        }
    except Exception as e:
        out["risk_service"] = {"ok": False, "err": f"{type(e).__name__}: {e}"}

    return out


def packages() -> list[dict]:
    rows = []
    for name in _PACKAGES:
        try:
            rows.append({"package": name, "version": version(name)})
        except PackageNotFoundError:
            rows.append({"package": name, "version": "(not installed)"})
        except Exception as e:
            rows.append({"package": name, "version": f"error: {e}"})
    return rows


def session_info() -> dict:
    keys = sorted(k for k in st.session_state.keys() if k.startswith("_"))
    # 결과 저장된 키만 (실제 dict 인 것)
    result_keys = []
    for k in keys:
        v = st.session_state.get(k)
        if isinstance(v, dict) and v:
            result_keys.append((k, len(v) if isinstance(v, dict) else 0))

    return {
        "scope": st.session_state.get("scope", "—"),
        "ticker": st.session_state.get("ticker", "—"),
        "all_underscore_keys": keys,
        "result_keys": result_keys,
        "n_session_keys": len(st.session_state.keys()),
    }


def collect_all() -> dict:
    return {
        "hf": hf_snapshot(),
        "engines": _import_status(),
        "packages": packages(),
        "session": session_info(),
        "collected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }