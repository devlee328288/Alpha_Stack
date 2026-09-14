# dashboard/services/backtest_service.py
"""
백테스트 서비스 — backtest/backtest_strategies 엔진 호출.

공용 설정(ExperimentConfig) 을 인자로 받아 실행.
predictor 기본값 통일, 캐시 키 = 설정 해시.
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

_BT_DIR = _REPO_ROOT / "backtest"
if str(_BT_DIR) not in sys.path:
    sys.path.insert(0, str(_BT_DIR))

import numpy as np
import pandas as pd

from services.experiment_config import ExperimentConfig

_IMPORT_ERR: str | None = None
_HAS_BT = False
_HAS_COST = False

try:
    from backtest_strategies import (  # type: ignore
        predict_5d_after,
        run_backtest,
    )

    _HAS_BT = True
except Exception as e:
    _IMPORT_ERR = f"backtest_strategies: {type(e).__name__}: {e}"

try:
    from run_cost_sensitivity import (  # type: ignore
        find_breakeven_cost,
        COST_PRESETS,
        COST_LABELS,
    )

    _HAS_COST = True
except Exception as e:
    if _IMPORT_ERR is None:
        _IMPORT_ERR = f"run_cost_sensitivity: {type(e).__name__}: {e}"


def engine_status() -> str | None:
    return _IMPORT_ERR


def cost_available() -> bool:
    return _HAS_COST


# ── 전략 ────────────────────────────────────────────────
STRATEGIES = ("A", "B", "C", "D", "E", "F")
_COOLDOWN_OF = {"D": 5, "E": 5, "F": 5}
_BASE_OF = {"D": "A", "E": "B", "F": "C"}

# ── UI 옵션 ────────────────────────────────────────────
MODEL_OPTIONS = (
    "RandomForest",
    "LightGBM",
    "XGBoost",
    "LogisticRegression",
)
BASELINE_OPTIONS = ("fwd_return", "adaptive")
BASELINE_LABELS = {
    "fwd_return": "fwd_return · ±1% 상수",
    "adaptive": "Adaptive 6-param · per-fold",
}
# Cost Sens 등 호환용 (Backtest 페이지는 MODEL_OPTIONS 사용)
PREDICTOR_OPTIONS = ("Random",) + MODEL_OPTIONS


# ═══════════════════════════════════════════════════════════
# 유틸
# ═══════════════════════════════════════════════════════════
def _silence():
    return (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    )


def _asset_code(scope: str, ticker: str) -> str:
    return "KOSPI200" if scope == "MARKET" else ticker


_LABEL_MAP = {-1: "하락", 0: "중립", 1: "상승"}
_FALLBACK_PROBS = {"p_up": 0.33, "p_flat": 0.34, "p_down": 0.33}


def _make_random_predictor():
    return predict_5d_after


def _prepare_baseline_payload(scope, ticker, baseline_kind, source="stocks30"):
    adaptive_labels = None
    if baseline_kind == "adaptive":
        from services import baseline_service
        from state import get_result

        _bl = get_result("baseline", scope, ticker)
        if _bl is None:
            raise ValueError(
                "Adaptive baseline 결과가 없습니다. "
                "**Baseline 페이지에서 RUN BASELINE** 을 먼저 실행하세요."
            )
        fd = _bl.get("fold_details", [])
        if not fd:
            raise ValueError("Baseline fold_details 가 비어 있습니다.")
        _ad = baseline_service.make_adaptive_labels_from_baseline(
            scope,
            ticker,
            fd,
            source=source,
        )
        adaptive_labels = _ad["labels"]
    elif baseline_kind == "fwd_return":
        pass
    else:
        raise ValueError(f"알 수 없는 baseline_kind: {baseline_kind}")
    return adaptive_labels


def _make_model_predictor(
    scope,
    ticker,
    model_name,
    baseline_kind="fwd_return",
    source="stocks30",
    model_revision="v1",
):
    """반환: (predict_func, oos_rows_count, model_id)."""
    from services import model_service

    adaptive_labels = _prepare_baseline_payload(
        scope,
        ticker,
        baseline_kind,
        source,
    )

    _out, _err = _silence()
    with _out, _err:
        r_model = model_service.run_single_model(
            scope=scope,
            ticker=ticker,
            model_name=model_name,
            source=source,
            label_source=baseline_kind,
            adaptive_labels=adaptive_labels,
        )

    oos = pd.DataFrame(r_model.get("oos_predictions", []))
    if oos.empty:
        raise ValueError(f"{model_name} · {baseline_kind}: oos_predictions 없음")
    if "bas_dd" not in oos.columns:
        raise ValueError(f"{model_name}: bas_dd 컬럼 없음")

    oos["bas_dd"] = pd.to_datetime(oos["bas_dd"], errors="coerce")
    oos = oos.dropna(subset=["bas_dd"]).sort_values("bas_dd")
    if oos["bas_dd"].duplicated().any():
        oos = oos.drop_duplicates(subset=["bas_dd"], keep="last")
    oos = oos.set_index("bas_dd").sort_index()
    if oos.empty:
        raise ValueError(f"{model_name}: 유효 bas_dd 없음")
    if not oos.index.is_unique:
        oos = oos[~oos.index.duplicated(keep="last")].sort_index()

    def _predict_from_model(base_date, market_data):
        ts = pd.Timestamp(base_date)
        if ts in oos.index:
            row = oos.loc[ts]
            sig = _LABEL_MAP[int(row["predicted"])]
            probs = {
                "p_up": float(row.get("p_up", 0.33)),
                "p_flat": float(row.get("p_neutral", 0.34)),
                "p_down": float(row.get("p_down", 0.33)),
            }
            return sig, probs
        return "중립", dict(_FALLBACK_PROBS)

    model_id = f"{model_name}-{baseline_kind}-{model_revision}"
    return _predict_from_model, len(oos), model_id


# ═══════════════════════════════════════════════════════════
# 캐시 (설정 해시 기준)
# ═══════════════════════════════════════════════════════════
_BT_CACHE: dict = {}
_COST_GRID_CACHE: dict = {}
_BE_CACHE: dict = {}
_PREDICT_FUNC_CACHE: dict = {}


def _get_predictor(config: ExperimentConfig):
    """config 기준 predictor 반환 (캐시). 반환: (predict_func, oos_rows, model_id)."""
    key = config.signal_key()
    if key in _PREDICT_FUNC_CACHE:
        return _PREDICT_FUNC_CACHE[key]

    if config.predictor == "Random":
        predict_func = _make_random_predictor()
        model_id = "random-v0"
        oos_rows = None
    else:
        predict_func, oos_rows, model_id = _make_model_predictor(
            config.scope,
            config.ticker,
            config.predictor,
            config.baseline_kind,
            config.source,
            config.model_revision,
        )

    _PREDICT_FUNC_CACHE[key] = (predict_func, oos_rows, model_id)
    return predict_func, oos_rows, model_id


# ═══════════════════════════════════════════════════════════
# 단일 백테스트
# ═══════════════════════════════════════════════════════════
def run_single(strategy: str, config: ExperimentConfig) -> dict:
    if not _HAS_BT:
        raise RuntimeError(f"backtest import 실패: {_IMPORT_ERR}")

    key = (config.cache_key(), strategy)
    if key in _BT_CACHE:
        return _BT_CACHE[key]

    from services import data_loader

    df = data_loader.load_market_data(config.scope, config.ticker)
    asset_code = _asset_code(config.scope, config.ticker)

    predict_func, oos_rows, model_id = _get_predictor(config)
    cooldown_days = _COOLDOWN_OF.get(strategy, 0)

    _out, _err = _silence()
    with _out, _err:
        result = run_backtest(
            market_data=df,
            start_date=pd.Timestamp(config.start),
            end_date=pd.Timestamp(config.end),
            predict_func=predict_func,
            strategy=strategy,
            initial_cash=float(config.initial_cash),
            trade_cost=float(config.trade_cost),
            model_id=model_id,
            asset_code=asset_code,
            cooldown_days=cooldown_days,
        )

    out = {
        "scope": config.scope,
        "ticker": config.ticker,
        "asset_code": asset_code,
        "strategy": strategy,
        "base_strategy": result.get("base_strategy", strategy),
        "cooldown_days": result.get("cooldown_days", 0),
        "predictor": config.predictor,
        "model_id": model_id,
        "baseline_kind": config.baseline_kind,
        "source": config.source,
        "oos_rows": int(oos_rows) if oos_rows is not None else None,
        "start": config.start,
        "end": config.end,
        "initial_cash": float(config.initial_cash),
        "trade_cost": float(config.trade_cost),
        "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": config.to_dict(),
        "metrics": {
            "total_return": float(result["total_return"]),
            "annual_return": float(result["annual_return"]),
            "volatility": float(result["volatility"]),
            "sharpe_ratio": float(result["sharpe_ratio"]),
            "max_drawdown": float(result["max_drawdown"]),
            "win_rate_daily": float(result["win_rate_daily"]),
            "avg_win": float(result["avg_win"]),
            "avg_loss": float(result["avg_loss"]),
            "profit_factor": float(result["profit_factor"]),
            "max_win_streak": int(result["max_win_streak"]),
            "max_loss_streak": int(result["max_loss_streak"]),
            "num_trades": int(result["num_trades"]),
            "buy_trades": int(result["buy_trades"]),
            "sell_trades": int(result["sell_trades"]),
            "total_transaction_cost": float(result["total_transaction_cost"]),
            "final_portfolio_value": float(result["final_portfolio_value"]),
            "final_position_ratio": float(result["final_position_ratio"]),
        },
        "equity": result["portfolio_series"],
        "daily_returns": result["daily_returns"],
        "signal_log": result["signal_log"].to_dict("records"),
    }
    _BT_CACHE[key] = out
    return out


def run_all_strategies(config: ExperimentConfig) -> dict:
    return {s: run_single(s, config) for s in STRATEGIES}


# ═══════════════════════════════════════════════════════════
# Cost Sensitivity
# ═══════════════════════════════════════════════════════════
def run_cost_grid(config: ExperimentConfig) -> pd.DataFrame:
    """config 의 trade_cost 를 무시하고, COST_PRESETS 각 값으로 실행."""
    if not _HAS_COST:
        raise RuntimeError(f"cost import 실패: {_IMPORT_ERR}")

    key = ("grid", *config.signal_key(), float(config.initial_cash))
    if key in _COST_GRID_CACHE:
        return _COST_GRID_CACHE[key]

    rows = []
    for cost_key, cost_rate in COST_PRESETS.items():
        cfg_cost = config.with_cost(cost_rate)
        for s in STRATEGIES:
            r = run_single(s, cfg_cost)
            m = r["metrics"]
            rows.append(
                {
                    "strategy": s,
                    "base_strategy": r.get("base_strategy", s),
                    "cooldown_days": r.get("cooldown_days", 0),
                    "cost_key": cost_key,
                    "cost_label": COST_LABELS[cost_key],
                    "cost_rate": cost_rate,
                    "sharpe": m["sharpe_ratio"],
                    "cagr": m["annual_return"],
                    "mdd": m["max_drawdown"],
                    "total_return": m["total_return"],
                    "num_trades": m["num_trades"],
                }
            )
    df = pd.DataFrame(rows)
    df.attrs["config"] = config.to_dict()
    _COST_GRID_CACHE[key] = df
    return df


def run_breakeven(config: ExperimentConfig) -> pd.DataFrame:
    """config 기준 전략별 손익분기 비용."""
    if not _HAS_COST:
        raise RuntimeError(f"cost import 실패: {_IMPORT_ERR}")

    key = ("be", *config.signal_key())
    if key in _BE_CACHE:
        return _BE_CACHE[key]

    from services import data_loader

    df = data_loader.load_market_data(config.scope, config.ticker)
    predict_func, _oos, _model_id = _get_predictor(config)

    rows = []
    for s in STRATEGIES:
        _out, _err = _silence()
        with _out, _err:
            try:
                be = find_breakeven_cost(
                    strategy=s,
                    market_data=df,
                    predict_func=predict_func,
                    start_date=pd.Timestamp(config.start),
                    end_date=pd.Timestamp(config.end),
                )
                rows.append(
                    {
                        "strategy": s,
                        "base_strategy": _BASE_OF.get(s, s),
                        "cooldown_days": _COOLDOWN_OF.get(s, 0),
                        "breakeven_cost": float(be),
                    }
                )
            except Exception as e:
                rows.append(
                    {
                        "strategy": s,
                        "base_strategy": _BASE_OF.get(s, s),
                        "cooldown_days": _COOLDOWN_OF.get(s, 0),
                        "breakeven_cost": float("nan"),
                        "error": f"{type(e).__name__}: {e}",
                    }
                )
    out = pd.DataFrame(rows)
    out.attrs["config"] = config.to_dict()
    _BE_CACHE[key] = out
    return out


def clear_cache() -> None:
    _BT_CACHE.clear()
    _COST_GRID_CACHE.clear()
    _BE_CACHE.clear()
    _PREDICT_FUNC_CACHE.clear()
