# dashboard/services/universe_service.py
"""
UNIVERSE 랭킹 서비스 — stocks30 / full 유니버스 × baseline / models.

Progressive: 종목별 완료 즉시 callback → session_state 저장.
Skip: 이미 결과가 있으면 건너뜀.
Parallel: joblib loky backend (Windows 안전).
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

import pandas as pd

from services import data_loader, baseline_service, model_service


# ═══════════════════════════════════════════════════════════
# 유니버스 목록
# ═══════════════════════════════════════════════════════════
def list_universe(source: str = "stocks30", **filter_kwargs) -> list[dict]:
    """
    source : "stocks30" | "full"
    filter_kwargs (full 일 때): market, top_n, exclude_halted, ...
    """
    if source == "full":
        return data_loader.list_full_universe(**filter_kwargs)
    return data_loader.list_universe_tickers()


def _name_map(source: str = "stocks30", **filter_kwargs) -> dict:
    """{code: name} 매핑."""
    try:
        return {m["code"]: m.get("name", "") for m in list_universe(source, **filter_kwargs)}
    except Exception:
        return {}


# ═══════════════════════════════════════════════════════════
# 순차 실행 (baseline)
# ═══════════════════════════════════════════════════════════
def run_universe_baseline(
    tickers: list[str],
    threshold: float = 0.01,
    max_evals: int = 10,
    source: str = "stocks30",
    skip_existing: dict | None = None,
    progress_cb: Callable[[int, int, str, str], None] | None = None,
) -> dict:
    """
    종목 리스트 순차 실행.

    progress_cb(i, total, ticker, phase)
      phase ∈ {"skip", "running", "done", "error"}
    """
    results = dict(skip_existing or {})
    total = len(tickers)
    name_map = _name_map(source)

    for i, ticker in enumerate(tickers):
        if ticker in results and not results[ticker].get("error"):
            if progress_cb:
                progress_cb(i, total, ticker, "skip")
            continue

        if progress_cb:
            progress_cb(i, total, ticker, "running")

        try:
            r = baseline_service.run_baseline(
                scope="STOCK",
                ticker=ticker,
                threshold=threshold,
                max_evals=max_evals,
                include_focal=False,
                source=source,
            )
            r["name"] = name_map.get(ticker, "")
            results[ticker] = r
            if progress_cb:
                progress_cb(i + 1, total, ticker, "done")
        except Exception as e:
            results[ticker] = {
                "error": f"{type(e).__name__}: {e}",
                "ticker": ticker,
                "name": name_map.get(ticker, ""),
            }
            if progress_cb:
                progress_cb(i + 1, total, ticker, "error")

    return results


# ═══════════════════════════════════════════════════════════
# 순차 실행 (단일 모델)
# ═══════════════════════════════════════════════════════════
def run_universe_model(
    tickers: list[str],
    model_name: str = "RandomForest",
    combination: str = "E",
    return_features: tuple = ("five_day_return",),
    source: str = "stocks30",
    skip_existing: dict | None = None,
    progress_cb: Callable[[int, int, str, str], None] | None = None,
) -> dict:
    """종목 × 단일 모델. 결과 {ticker: result}."""
    results = dict(skip_existing or {})
    total = len(tickers)
    name_map = _name_map(source)

    for i, ticker in enumerate(tickers):
        if ticker in results and not results[ticker].get("error"):
            if progress_cb:
                progress_cb(i, total, ticker, "skip")
            continue

        if progress_cb:
            progress_cb(i, total, ticker, "running")

        try:
            r = model_service.run_single_model(
                scope="STOCK",
                ticker=ticker,
                model_name=model_name,
                combination=combination,
                return_features=return_features,
                source=source,
            )
            r["name"] = name_map.get(ticker, "")
            results[ticker] = r
            if progress_cb:
                progress_cb(i + 1, total, ticker, "done")
        except Exception as e:
            results[ticker] = {
                "error": f"{type(e).__name__}: {e}",
                "ticker": ticker,
                "name": name_map.get(ticker, ""),
            }
            if progress_cb:
                progress_cb(i + 1, total, ticker, "error")

    return results


# ═══════════════════════════════════════════════════════════
# 순차 실행 (multi-model)
# ═══════════════════════════════════════════════════════════
def run_universe_models_multi(
    tickers: list[str],
    model_names: list[str],
    combination: str = "E",
    return_features: tuple = ("five_day_return",),
    source: str = "stocks30",
    skip_existing: dict | None = None,
    progress_cb: Callable[[int, int, str, str, str], None] | None = None,
) -> dict:
    """
    종목 × N개 모델. 결과 키 = f"{ticker}::{model_name}".
    progress_cb(done, total_steps, ticker, model_name, phase)
    """
    results = dict(skip_existing or {})
    name_map = _name_map(source)
    total_steps = len(tickers) * len(model_names)
    done = 0

    for m_name in model_names:
        for ticker in tickers:
            composite_key = f"{ticker}::{m_name}"

            if composite_key in results and not results[composite_key].get("error"):
                done += 1
                if progress_cb:
                    progress_cb(done, total_steps, ticker, m_name, "skip")
                continue

            if progress_cb:
                progress_cb(done, total_steps, ticker, m_name, "running")

            try:
                r = model_service.run_single_model(
                    scope="STOCK",
                    ticker=ticker,
                    model_name=m_name,
                    combination=combination,
                    return_features=return_features,
                    source=source,
                )
                r["name"] = name_map.get(ticker, "")
                r["_model_name"] = m_name
                r["_ticker"] = ticker
                results[composite_key] = r
                done += 1
                if progress_cb:
                    progress_cb(done, total_steps, ticker, m_name, "done")
            except Exception as e:
                results[composite_key] = {
                    "error": f"{type(e).__name__}: {e}",
                    "ticker": ticker,
                    "name": name_map.get(ticker, ""),
                    "_model_name": m_name,
                }
                done += 1
                if progress_cb:
                    progress_cb(done, total_steps, ticker, m_name, "error")

    return results


# ═══════════════════════════════════════════════════════════
# 랭킹 DataFrame 빌더
# ═══════════════════════════════════════════════════════════
def build_universe_ranking(baseline_results: dict) -> pd.DataFrame:
    """{ticker: baseline result} → 랭킹 DataFrame."""
    rows = []
    for ticker, r in baseline_results.items():
        _name = r.get("name", "")

        if r.get("error"):
            rows.append({
                "RANK": None, "CODE": ticker, "NAME": _name,
                "MACRO F1": None, "BAL ACC": None,
                "SHARPE": None, "CAGR": None, "MDD": None, "WIN RATE": None,
                "FOLDS": None,
                "ERROR": r["error"][:40],
            })
            continue

        perf = r.get("perf_metrics", {})
        cls = r.get("cls_metrics", {})
        rows.append({
            "RANK": None,
            "CODE": ticker,
            "NAME": _name,
            "MACRO F1": cls.get("f1_macro"),
            "BAL ACC": cls.get("balanced_acc"),
            "SHARPE": perf.get("sharpe"),
            "CAGR": perf.get("cagr"),
            "MDD": -abs(perf.get("mdd", 0)) if perf.get("mdd") is not None else None,
            "WIN RATE": perf.get("win_rate"),
            "FOLDS": r.get("total_folds"),
            "ERROR": "",
        })

    df = pd.DataFrame(rows)
    if "MACRO F1" in df.columns:
        df = df.sort_values(
            by=["MACRO F1", "SHARPE"],
            ascending=[False, False],
            na_position="last",
            kind="stable",
        ).reset_index(drop=True)
        df["RANK"] = range(1, len(df) + 1)

    cols = ["RANK", "CODE", "NAME", "MACRO F1", "BAL ACC",
            "SHARPE", "CAGR", "MDD", "WIN RATE", "FOLDS", "ERROR"]
    cols = [c for c in cols if c in df.columns]
    return df[cols]


def build_universe_model_ranking(model_results: dict) -> pd.DataFrame:
    """{ticker: run_single_model dict} → 랭킹."""
    rows = []
    for ticker, r in model_results.items():
        _name = r.get("name", "")

        if r.get("error"):
            rows.append({
                "RANK": None, "CODE": ticker, "NAME": _name,
                "ACC": None, "MACRO F1": None, "DOWN RECALL": None,
                "HARMONIC": None, "BAL ACC": None,
                "ΔSHARPE": None, "CASH FOLDS": None,
                "ERROR": r["error"][:40],
            })
            continue

        s = r.get("summary", {})
        rows.append({
            "RANK": None,
            "CODE": ticker,
            "NAME": _name,
            "ACC": s.get("accuracy"),
            "MACRO F1": s.get("macro_f1"),
            "DOWN RECALL": s.get("down_recall"),
            "HARMONIC": s.get("core_harmonic_mean"),
            "BAL ACC": s.get("balanced_accuracy"),
            "ΔSHARPE": s.get("delta_sharpe_net_median"),
            "CASH FOLDS": s.get("all_cash_folds"),
            "ERROR": "",
        })

    df = pd.DataFrame(rows)
    if "HARMONIC" in df.columns:
        df = df.sort_values(
            by=["HARMONIC"],
            ascending=False,
            na_position="last",
            kind="stable",
        ).reset_index(drop=True)
        df["RANK"] = range(1, len(df) + 1)

    cols = ["RANK", "CODE", "NAME", "ACC", "MACRO F1", "DOWN RECALL",
            "HARMONIC", "BAL ACC", "ΔSHARPE", "CASH FOLDS", "ERROR"]
    cols = [c for c in cols if c in df.columns]
    return df[cols]


# ═══════════════════════════════════════════════════════════
# Multi-model Matrix / Long
# ═══════════════════════════════════════════════════════════
def build_universe_models_matrix(
    model_results: dict,
    source: str = "stocks30",
) -> pd.DataFrame:
    """
    키가 "ticker::model" 인 dict → wide matrix (ticker × model harmonic).
    """
    fallback_name_map = _name_map(source)

    rows = {}
    for composite_key, r in model_results.items():
        if "::" not in composite_key:
            continue
        ticker, m_name = composite_key.split("::", 1)

        if not m_name or m_name == "?":
            continue

        if ticker not in rows:
            rows[ticker] = {
                "CODE": ticker,
                "NAME": r.get("name") or fallback_name_map.get(ticker, ""),
            }

        if r.get("error"):
            rows[ticker][m_name] = None
        else:
            s = r.get("summary", {})
            rows[ticker][m_name] = s.get("core_harmonic_mean")

    df = pd.DataFrame(list(rows.values()))
    if df.empty:
        return df

    # 유효 모델 컬럼 (빈 이름/`?` 제거)
    model_cols = [
        c for c in df.columns
        if c not in ("CODE", "NAME") and c and c != "?"
    ]

    if model_cols:
        df["_MEAN"] = df[model_cols].mean(axis=1)
        df = df.sort_values("_MEAN", ascending=False, kind="stable").reset_index(drop=True)
        df["RANK"] = range(1, len(df) + 1)
        df = df.drop(columns=["_MEAN"])

    cols = ["RANK", "CODE", "NAME"] + model_cols
    return df[[c for c in cols if c in df.columns]]


def build_models_by_ticker_table(
    model_results: dict,
    source: str = "stocks30",
) -> pd.DataFrame:
    """키가 "ticker::model" 인 dict → long format."""
    fallback_name_map = _name_map(source)

    rows = []
    for composite_key, r in model_results.items():
        if "::" not in composite_key:
            continue
        ticker, m_name = composite_key.split("::", 1)

        if not m_name or m_name == "?":
            continue

        _name = r.get("name") or fallback_name_map.get(ticker, "")

        if r.get("error"):
            rows.append({
                "TICKER": ticker, "NAME": _name, "MODEL": m_name,
                "ACC": None, "MACRO F1": None, "HARMONIC": None,
                "DOWN RECALL": None, "ΔSHARPE": None,
                "ERROR": r["error"][:40],
            })
            continue

        s = r.get("summary", {})
        rows.append({
            "TICKER": ticker,
            "NAME": _name,
            "MODEL": m_name,
            "ACC": s.get("accuracy"),
            "MACRO F1": s.get("macro_f1"),
            "HARMONIC": s.get("core_harmonic_mean"),
            "DOWN RECALL": s.get("down_recall"),
            "ΔSHARPE": s.get("delta_sharpe_net_median"),
            "ERROR": "",
        })

    df = pd.DataFrame(rows)
    if "HARMONIC" in df.columns and not df.empty:
        df = df.sort_values("HARMONIC", ascending=False,
                            na_position="last", kind="stable").reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════
# Worker (joblib)
# ═══════════════════════════════════════════════════════════
def _worker_baseline(ticker, threshold, max_evals, source, name):
    try:
        r = baseline_service.run_baseline(
            scope="STOCK",
            ticker=ticker,
            threshold=threshold,
            max_evals=max_evals,
            include_focal=False,
            source=source,
        )
        r["name"] = name
        return ticker, r
    except Exception as e:
        return ticker, {
            "error": f"{type(e).__name__}: {e}",
            "ticker": ticker,
            "name": name,
        }


def _worker_model(ticker, model_name, source, name):
    try:
        r = model_service.run_single_model(
            scope="STOCK",
            ticker=ticker,
            model_name=model_name,
            source=source,
        )
        r["name"] = name
        r["_model_name"] = model_name
        r["_ticker"] = ticker
        return ticker, model_name, r
    except Exception as e:
        return ticker, model_name, {
            "error": f"{type(e).__name__}: {e}",
            "ticker": ticker,
            "name": name,
            "_model_name": model_name,
        }


# ═══════════════════════════════════════════════════════════
# 병렬 실행 (baseline)
# ═══════════════════════════════════════════════════════════
def run_universe_baseline_parallel(
    tickers: list[str],
    threshold: float = 0.01,
    max_evals: int = 30,
    source: str = "stocks30",
    n_jobs: int = 4,
    skip_existing: dict | None = None,
    progress_cb=None,
) -> dict:
    """병렬. n_jobs=1 이면 순차."""
    from joblib import Parallel, delayed

    results = dict(skip_existing or {})
    todo = [t for t in tickers if t not in results or results[t].get("error")]
    total = len(todo)
    if total == 0:
        return results

    name_map = _name_map(source)

    # 순차
    if n_jobs <= 1:
        for i, tk in enumerate(todo):
            _, r = _worker_baseline(
                tk, threshold, max_evals, source, name_map.get(tk, ""),
            )
            results[tk] = r
            if progress_cb:
                progress_cb(i + 1, total, tk,
                            "done" if "error" not in r else "error")
        return results

    # 병렬
    try:
        gen = Parallel(n_jobs=n_jobs, backend="loky",
                       return_as="generator_unordered")(
            delayed(_worker_baseline)(
                tk, threshold, max_evals, source, name_map.get(tk, ""),
            )
            for tk in todo
        )
        for i, (tk, r) in enumerate(gen):
            results[tk] = r
            if progress_cb:
                progress_cb(i + 1, total, tk,
                            "done" if "error" not in r else "error")
    except Exception:
        # 폴백: 순차
        for i, tk in enumerate(todo):
            if tk in results and not results[tk].get("error"):
                continue
            _, r = _worker_baseline(
                tk, threshold, max_evals, source, name_map.get(tk, ""),
            )
            results[tk] = r
            if progress_cb:
                progress_cb(i + 1, total, tk,
                            "done" if "error" not in r else "error")

    return results


# ═══════════════════════════════════════════════════════════
# 병렬 실행 (multi-model)
# ═══════════════════════════════════════════════════════════
def run_universe_models_multi_parallel(
    tickers: list[str],
    model_names: list[str],
    source: str = "stocks30",
    n_jobs: int = 4,
    skip_existing: dict | None = None,
    progress_cb=None,
) -> dict:
    """병렬. 결과 키 = f"{ticker}::{model_name}"."""
    from joblib import Parallel, delayed

    results = dict(skip_existing or {})
    name_map = _name_map(source)

    tasks = []
    for m_name in model_names:
        for tk in tickers:
            ck = f"{tk}::{m_name}"
            if ck in results and not results[ck].get("error"):
                continue
            tasks.append((tk, m_name))

    total = len(tasks)
    if total == 0:
        return results

    # 순차
    if n_jobs <= 1:
        for i, (tk, m_name) in enumerate(tasks):
            _, mname, r = _worker_model(
                tk, m_name, source, name_map.get(tk, ""),
            )
            results[f"{tk}::{mname}"] = r
            if progress_cb:
                progress_cb(i + 1, total, tk, m_name,
                            "done" if "error" not in r else "error")
        return results

    # 병렬
    try:
        gen = Parallel(n_jobs=n_jobs, backend="loky",
                       return_as="generator_unordered")(
            delayed(_worker_model)(
                tk, m_name, source, name_map.get(tk, ""),
            )
            for tk, m_name in tasks
        )
        for i, (tk, m_name, r) in enumerate(gen):
            results[f"{tk}::{m_name}"] = r
            if progress_cb:
                progress_cb(i + 1, total, tk, m_name,
                            "done" if "error" not in r else "error")
    except Exception:
        # 폴백: 순차
        for i, (tk, m_name) in enumerate(tasks):
            ck = f"{tk}::{m_name}"
            if ck in results and not results[ck].get("error"):
                continue
            _, mname, r = _worker_model(
                tk, m_name, source, name_map.get(tk, ""),
            )
            results[f"{tk}::{mname}"] = r
            if progress_cb:
                progress_cb(i + 1, total, tk, m_name,
                            "done" if "error" not in r else "error")

    return results


# ═══════════════════════════════════════════════════════════
# 캐시
# ═══════════════════════════════════════════════════════════
def clear_cache() -> None:
    pass  # session_state 로 관리