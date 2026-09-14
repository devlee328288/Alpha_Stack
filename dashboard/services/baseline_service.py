# dashboard/services/baseline_service.py
"""
Baseline 서비스 — 6-param threshold walk-forward.
scope/ticker 지원 (MARKET | STOCK).
Streamlit @st.cache_data 안 씀 (stdout ASCII 캡처 문제 회피).
"""

from __future__ import annotations

import contextlib
import io
import sys
from datetime import datetime
from pathlib import Path

# stdout/stderr reconfigure
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

_IMPORT_ERR: str | None = None
_HAS_FOCAL = False
try:
    from step1_core_features import load_data  # type: ignore
    from step5_optimize_6params import run_walkforward_6params  # type: ignore
    from focal_classifier import FocalConfig, make_labels  # type: ignore

    try:
        from step7_focal_walkforward import (  # type: ignore
            evaluate_signals,
            generate_signals_rolling,
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
# 수동 캐시
# ═══════════════════════════════════════════════════════════
_BASELINE_CACHE: dict = {}


def _silence():
    return (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    )


def _sanitize_metrics(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, (int, float, np.floating, np.integer)):
            out[k] = float(v)
        elif isinstance(v, (str, bool, type(None))):
            out[k] = v
    return out


def run_baseline(
    scope: str = "MARKET",
    ticker: str = "KOSPI200",
    threshold: float | str = 0.01,
    max_evals: int = 30,
    include_focal: bool = False,
    source: str = "stocks30",
) -> dict:
    if _IMPORT_ERR:
        raise RuntimeError(f"repo engine import 실패: {_IMPORT_ERR}")

    # ── threshold 정규화 ("adaptive" 지원) ────────────
    if isinstance(threshold, str) and threshold == "adaptive":
        threshold_key = "adaptive"
        threshold_float = 0.01  # focal/metrics 용 (adaptive에선 미사용)
    else:
        threshold_key = float(threshold)
        threshold_float = float(threshold)

    cache_key = (
        scope,
        ticker,
        threshold_key,
        int(max_evals),
        bool(include_focal),
        source,
    )
    if cache_key in _BASELINE_CACHE:
        return _BASELINE_CACHE[cache_key]

    # 데이터 로드 (scope/ticker/source 별)
    from services import data_loader

    if source == "full":
        df = data_loader.load_market_data_full(ticker, price_col="adj_close")
    else:
        df = data_loader.load_market_data(scope, ticker)

    _out, _err = _silence()
    with _out, _err:
        step5 = run_walkforward_6params(
            df,
            train_years=2,
            val_months=3,
            step_months=1,
            max_evals=max_evals,
            threshold=threshold,  # ★ "adaptive" or float 그대로 전달
        )

    out: dict = {
        "scope": scope,
        "ticker": ticker,
        "threshold": threshold_key,  # ★ "adaptive" or float 저장
        "max_evals": int(max_evals),
        "include_focal": bool(include_focal),
        "total_folds": int(step5["total_folds"]),
        "perf_metrics": _sanitize_metrics(step5["perf_metrics"]),
        "cls_metrics": _sanitize_metrics(step5["cls_metrics"]),
        "params_median": _sanitize_metrics(step5["params_median"]),
        "fold_details": step5["fold_details"].to_dict("records"),
        "label_kind": step5.get("label_kind", "fwd_return"),
        "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    if include_focal and _HAS_FOCAL:
        try:
            _out2, _err2 = _silence()
            with _out2, _err2:
                cfg = FocalConfig(threshold=float(threshold_float))
                signals = generate_signals_rolling(
                    df,
                    step5["fold_details"],
                    focal_config=cfg,
                )
                y_true = make_labels(df, threshold=float(threshold_float)).to_numpy()
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
    _BASELINE_CACHE.clear()


def make_adaptive_labels_from_baseline(
    scope: str | None = None,
    ticker: str = "KOSPI200",
    fold_details=None,
    source: str | None = None,
    **kwargs,
):
    """
    Baseline 의 fold_details (per-fold 파라미터) 로 전체 df 라벨 생성.

    시그니처 유연화:
      - scope: "MARKET" | "STOCK"  (권장)
      - source: "MARKET"/"STOCK" 이면 scope 로 해석
                "stocks30"/"full"   이면 session_state['scope'] 폴백

    Returns
    -------
    dict : {"labels": list[float], "dates": list[str], "n_valid": int, "n_total": int}
    """
    import numpy as np
    import pandas as pd
    from services import data_loader
    from step5_optimize_6params import make_band_labels

    if _IMPORT_ERR:
        raise RuntimeError(f"repo engine import 실패: {_IMPORT_ERR}")

    # ── scope 정규화 ─────────────────────────────
    # 1) scope 우선
    _resolved_scope = scope
    # 2) source 가 MARKET/STOCK 이면 그걸 scope 로
    if _resolved_scope is None and source in ("MARKET", "STOCK"):
        _resolved_scope = source
    # 3) 그 외엔 session_state 폴백
    if _resolved_scope is None:
        try:
            import streamlit as st

            _resolved_scope = st.session_state.get("scope", "MARKET")
        except Exception:
            _resolved_scope = "MARKET"
    scope = _resolved_scope

    df = data_loader.load_market_data(scope, ticker)
    n = len(df)
    dates_list = [str(d) for d in df.index]

    fold_df = pd.DataFrame(fold_details)
    if fold_df.empty:
        raise ValueError("fold_details 비어있음")
    if "train_end" not in fold_df.columns:
        raise ValueError("train_end 컬럼 없음")

    # ── train_end Timestamp → index 위치 매핑 ────
    date_to_idx = {d: i for i, d in enumerate(df.index)}

    fold_positions = []
    for _, row in fold_df.iterrows():
        try:
            te = pd.Timestamp(row["train_end"])
        except Exception:
            continue
        idx = date_to_idx.get(te)
        if idx is None:
            try:
                pos = df.index.get_indexer([te], method="nearest")
                if len(pos) > 0 and pos[0] >= 0:
                    idx = int(pos[0])
            except Exception:
                pass
        if idx is None:
            continue
        fold_positions.append((idx, row))

    if not fold_positions:
        raise ValueError(
            f"fold_details 의 train_end 가 df.index 와 매칭 안 됨 "
            f"(n_fold={len(fold_df)})"
        )

    fold_positions.sort(key=lambda x: x[0])
    train_ends = [p[0] for p in fold_positions]

    fold_assign = np.full(n, -1, dtype=int)
    for i, te in enumerate(train_ends):
        next_te = train_ends[i + 1] if i + 1 < len(train_ends) else n
        fold_assign[te:next_te] = i

    labels = np.full(n, np.nan, dtype=float)
    for i, (te, row) in enumerate(fold_positions):
        mask = fold_assign == i
        if not mask.any():
            continue
        try:
            fold_labels = make_band_labels(
                df,
                float(row["alpha_up"]),
                float(row["alpha_down"]),
                float(row["beta_up"]),
                float(row["beta_down"]),
                int(round(float(row["vol_period"]))),
                int(round(float(row["volume_period"]))),
            )
            labels[mask] = fold_labels[mask]
        except Exception:
            continue

    valid_mask = np.isfinite(labels)
    return {
        "labels": labels.tolist(),
        "dates": dates_list,
        "n_valid": int(valid_mask.sum()),
        "n_total": n,
    }
