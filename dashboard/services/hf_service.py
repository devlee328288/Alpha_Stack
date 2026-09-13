# dashboard/services/hf_service.py
"""
HF 스냅샷 메타데이터 (사이드바 / System 페이지 표시용).
"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

_REPO_ID = "qurious-quant/alphastack-krx-dev"
_CACHE: dict = {}


def _silence():
    return (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    )


def get_hf_meta() -> dict:
    """
    우선순위:
      1) session_state에 저장된 결과의 source 정보
      2) HfApi.repo_info 로 lazy fetch (모듈 캐시)
    """
    # 1) session_state 에서
    for key in ("_baseline_results", "_model_lab_results"):
        r = st.session_state.get(key)
        if isinstance(r, dict):
            src = r.get("source")
            if isinstance(src, dict) and src.get("repo_sha"):
                return {
                    "ok": True,
                    "repo_sha": str(src["repo_sha"])[:12],
                    "dev_end": str(src.get("dev_end") or "—"),
                    "file_sha": str(src.get("index_sha256") or "")[:8] or "—",
                    "source": "session",
                }

    # 2) lazy fetch
    if "meta" in _CACHE:
        return _CACHE["meta"]

    try:
        from huggingface_hub import HfApi
        from common.secrets import load_key

        _out, _err = _silence()
        with _out, _err:
            token, _ = load_key(
                ("HUGGINGFACE_ACCESS_TOKEN", "HF_TOKEN", "HUGGINGFACE_TOKEN")
            )
            info = HfApi(token=token).repo_info(
                repo_id=_REPO_ID, repo_type="dataset"
            )

        meta = {
            "ok": True,
            "repo_sha": str(info.sha)[:12],
            "dev_end": "—",
            "file_sha": "—",
            "source": "hf_api",
        }
    except Exception as e:
        meta = {"ok": False, "err": f"{type(e).__name__}: {e}"}

    _CACHE["meta"] = meta
    return meta


def clear_cache() -> None:
    _CACHE.clear()