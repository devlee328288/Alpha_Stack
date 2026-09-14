# dashboard/services/backtest_service.py
"""
백테스트 서비스 — backtest/backtest_strategies 엔진 호출.
scope/ticker/predictor/baseline_kind 지원.

predictor:
  "Random"       → 기존 랜덤 (구조 검증용)
  "<모델명>"      → Model Lab 의 12-fold OOS 예측을 백테스트 신호로 사용

baseline_kind (모델 학습 라벨 = 매매 판단 기준):
  "fwd_return"   → 미래 5일 수익률 ±1% 상수 밴드
  "adaptive"     → Baseline per-fold 최적 밴드
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


STRATEGIES = ("A", "B", "C")

# ── UI 옵션 ────────────────────────────────────────────────
# AI 모델 4종 (label_source 와 무관)
MODEL_OPTIONS = (
    "RandomForest",
    "LightGBM",
    "XGBoost",
    "LogisticRegression",
)

# Baseline(라벨 규칙) 2종
BASELINE_OPTIONS = ("fwd_return", "adaptive")

BASELINE_LABELS = {
    "fwd_return": "fwd_return · ±1% 상수",
    "adaptive": "Adaptive 6-param · per-fold",
}

# 기존 호환 (Cost Sensitivity 페이지 등이 참조)
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


# ═══════════════════════════════════════════════════════════
# predict_func 빌더
# ═══════════════════════════════════════════════════════════
def _make_random_predictor():
    """기존 backtest_strategies.predict_5d_after (랜덤)."""
    return predict_5d_after


def _prepare_baseline_payload(
    scope: str,
    ticker: str,
    baseline_kind: str,
    source: str = "stocks30",
):
    """
    baseline_kind 에 따라 adaptive_labels 를 준비.
    Model Lab 과 동일한 라벨 소스 계약을 사용.
    """
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
    scope: str,
    ticker: str,
    model_name: str,
    baseline_kind: str = "fwd_return",
    source: str = "stocks30",
):
    """
    (모델 × baseline_kind) 로 학습한 Model Lab OOS 예측을
    백테스트 신호 predict_func 로 변환.
    반환: (predict_func, oos_rows_count)
    """
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

    # ── (신규) 중복 날짜 정리 ─────────────────────────
    # 같은 bas_dd 가 여러 행이면 신호가 하루에 여러 개가 되므로
    # 가장 마지막 OOS 예측 하나만 남긴다. (walk-forward 라면 최신 fold 결과)
    _dup = int(oos["bas_dd"].duplicated().sum())
    if _dup:
        # 진단용 (원하면 제거)
        # print(f"[{model_name}/{baseline_kind}] duplicate bas_dd rows: {_dup}")
        oos = oos.drop_duplicates(subset=["bas_dd"], keep="last")

    oos = oos.set_index("bas_dd").sort_index()

    if oos.empty:
        raise ValueError(f"{model_name}: 유효 bas_dd 없음")
    if not oos.index.is_unique:
        # 방어적: 그래도 남아있으면 강제 dedupe
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

    return _predict_from_model, len(oos)


# ═══════════════════════════════════════════════════════════
# 수동 캐시
# ═══════════════════════════════════════════════════════════
_BT_CACHE: dict = {}
_COST_GRID_CACHE: dict = {}
_BE_CACHE: dict = {}


# ═══════════════════════════════════════════════════════════
# 단일 백테스트
# ═══════════════════════════════════════════════════════════
def run_single(
    scope: str,
    ticker: str,
    strategy: str,
    start: str,
    end: str,
    initial_cash: float = 100.0,
    trade_cost: float = 0.001,
    predictor: str = "Random",
    source: str = "stocks30",
    baseline_kind: str = "fwd_return",
) -> dict:
    if not _HAS_BT:
        raise RuntimeError(f"backtest import 실패: {_IMPORT_ERR}")

    key = (
        scope,
        ticker,
        strategy,
        start,
        end,
        float(initial_cash),
        float(trade_cost),
        predictor,
        source,
        baseline_kind,
    )
    if key in _BT_CACHE:
        return _BT_CACHE[key]

    from services import data_loader

    df = data_loader.load_market_data(scope, ticker)
    asset_code = _asset_code(scope, ticker)

    # ── predict_func 결정 ─────────────────────────────
    oos_rows = None
    if predictor == "Random":
        predict_func = _make_random_predictor()
        model_id = "random-v0"
    else:
        predict_func, oos_rows = _make_model_predictor(
            scope,
            ticker,
            predictor,
            baseline_kind,
            source,
        )
        model_id = f"{predictor}-{baseline_kind}-v1"

    _out, _err = _silence()
    with _out, _err:
        result = run_backtest(
            market_data=df,
            start_date=pd.Timestamp(start),
            end_date=pd.Timestamp(end),
            predict_func=predict_func,
            strategy=strategy,
            initial_cash=float(initial_cash),
            trade_cost=float(trade_cost),
            model_id=model_id,
            asset_code=asset_code,
        )

    out = {
        "scope": scope,
        "ticker": ticker,
        "asset_code": asset_code,
        "strategy": strategy,
        "predictor": predictor,
        "baseline_kind": baseline_kind,
        "source": source,
        "oos_rows": int(oos_rows) if oos_rows is not None else None,
        "start": start,
        "end": end,
        "initial_cash": float(initial_cash),
        "trade_cost": float(trade_cost),
        "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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


def run_all_strategies(
    scope: str,
    ticker: str,
    start: str,
    end: str,
    initial_cash: float = 100.0,
    trade_cost: float = 0.001,
    predictor: str = "Random",
    source: str = "stocks30",
    baseline_kind: str = "fwd_return",
) -> dict:
    out = {}
    for s in STRATEGIES:
        out[s] = run_single(
            scope,
            ticker,
            s,
            start,
            end,
            initial_cash,
            trade_cost,
            predictor,
            source,
            baseline_kind,
        )
    return out


# ═══════════════════════════════════════════════════════════
# Cost Sensitivity
# ═══════════════════════════════════════════════════════════
def run_cost_grid(
    scope: str,
    ticker: str,
    start: str,
    end: str,
    initial_cash: float = 100.0,
    predictor: str = "Random",
    source: str = "stocks30",
    baseline_kind: str = "fwd_return",
) -> pd.DataFrame:
    if not _HAS_COST:
        raise RuntimeError(f"cost import 실패: {_IMPORT_ERR}")

    key = (
        scope,
        ticker,
        start,
        end,
        float(initial_cash),
        predictor,
        source,
        baseline_kind,
    )
    if key in _COST_GRID_CACHE:
        return _COST_GRID_CACHE[key]

    rows = []
    for cost_key, cost_rate in COST_PRESETS.items():
        for s in STRATEGIES:
            r = run_single(
                scope,
                ticker,
                s,
                start,
                end,
                initial_cash,
                cost_rate,
                predictor,
                source,
                baseline_kind,
            )
            m = r["metrics"]
            rows.append(
                {
                    "strategy": s,
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
    _COST_GRID_CACHE[key] = df
    return df


def run_breakeven(
    scope: str,
    ticker: str,
    start: str,
    end: str,
    initial_cash: float = 100.0,
    predictor: str = "RandomForest",
    source: str = "stocks30",
    baseline_kind: str = "fwd_return",
) -> pd.DataFrame:
    if not _HAS_COST:
        raise RuntimeError(f"cost import 실패: {_IMPORT_ERR}")

    key = (
        scope,
        ticker,
        start,
        end,
        float(initial_cash),
        predictor,
        source,
        baseline_kind,
    )
    if key in _BE_CACHE:
        return _BE_CACHE[key]

    from services import data_loader

    df = data_loader.load_market_data(scope, ticker)

    # ── predict_func 결정 (run_single 과 동일) ─────────
    if predictor == "Random":
        predict_func = _make_random_predictor()
    else:
        predict_func, _ = _make_model_predictor(
            scope,
            ticker,
            predictor,
            baseline_kind,
            source,
        )

    rows = []
    for s in STRATEGIES:
        _out, _err = _silence()
        with _out, _err:
            be = find_breakeven_cost(
                strategy=s,
                market_data=df,
                predict_func=predict_func,
                start_date=pd.Timestamp(start),
                end_date=pd.Timestamp(end),
            )
        rows.append({"strategy": s, "breakeven_cost": float(be)})
    out = pd.DataFrame(rows)
    _BE_CACHE[key] = out
    return out


def clear_cache() -> None:
    _BT_CACHE.clear()
    _COST_GRID_CACHE.clear()
    _BE_CACHE.clear()
