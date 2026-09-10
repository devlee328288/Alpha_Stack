# dashboard/services/evaluation_service.py
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# 실 엔진 import
# ============================================================
_ERRORS: list[str] = []

# ── risk ───────────────────────────────────────────────────
try:
    from evaluation.evaluation_risk import (
        calculate_all_metrics as _risk_all_metrics,
    )
    _HAS_RISK = True
except Exception as _e:
    _HAS_RISK = False
    _ERRORS.append(f"evaluation_risk: {_e}")

# ── classification ─────────────────────────────────────────
try:
    from evaluation.evaluation_backtest import (
        calculate_all_classification_metrics as _real_cls_metrics,
    )
    from evaluation.evaluation_backtest import (
        confusion_matrix_multiclass as _real_cm,
    )
    _HAS_CLS = True
except Exception as _e:
    _HAS_CLS = False
    _ERRORS.append(f"evaluation_backtest: {_e}")

# ── walk forward ───────────────────────────────────────────
try:
    from evaluation.walk_forward import (
        expanding_splits as _real_expanding_splits,
    )
    from evaluation.walk_forward import (
        iter_splits as _real_iter_splits,
    )
    _HAS_WF = True
except Exception as _e:
    _HAS_WF = False
    _ERRORS.append(f"walk_forward: {_e}")


def engine_status() -> list[str]:
    return list(_ERRORS)


# ============================================================
# 공통 유틸
# ============================================================
_LABELS = [-1, 0, 1]


def _scalar(v):
    """numpy array / list / 0-d array / None → float 스칼라. dict/object 는 nan."""
    if v is None:
        return float("nan")
    if isinstance(v, dict):
        return float("nan")
    try:
        arr = np.asarray(v)
        if arr.dtype == object:
            return float("nan")
        if arr.size == 0:
            return float("nan")
        if arr.size == 1:
            return float(arr.reshape(-1)[0])
        return float(np.nanmean(arr.astype(float)))
    except Exception:
        return float("nan")


_RISK_ALIAS = {
    "MDD": "MDD",
    "Maximum Drawdown": "MDD",
    "Sharpe Ratio": "Sharpe",
    "Sharpe": "Sharpe",
    "Sortino Ratio": "Sortino",
    "Sortino": "Sortino",
    "Sterling Ratio": "Sterling",
    "Sterling": "Sterling",
    "Calmar Ratio": "Calmar",
    "Calmar": "Calmar",
    "CAGR": "CAGR",
    "Annualized Return": "CAGR",
    "Volatility": "Vol",
    "Annualized Volatility": "Vol",
}


def _normalize_risk_keys(d: dict) -> dict:
    return {_RISK_ALIAS.get(k, k): v for k, v in d.items()}


# ============================================================
# Risk 지표
# ============================================================
def _mock_risk_metrics(returns: pd.Series) -> dict:
    returns = pd.Series(returns).dropna()
    if len(returns) == 0:
        return {"CAGR": 0.0, "Vol": 0.0, "Sharpe": 0.0,
                "Sortino": 0.0, "MDD": 0.0, "Calmar": 0.0, "Sterling": 0.0}
    ann = 252
    mu = float(returns.mean() * ann)
    vol = float(returns.std() * np.sqrt(ann))
    sharpe = mu / vol if vol > 0 else 0.0
    eq = (1 + returns).cumprod()
    dd = eq / eq.cummax() - 1
    mdd = float(dd.min())
    cagr = float(eq.iloc[-1] ** (ann / max(len(returns), 1)) - 1)
    calmar = cagr / abs(mdd) if mdd < 0 else 0.0
    downside = float(returns[returns < 0].std() * np.sqrt(ann))
    sortino = mu / downside if downside > 0 else 0.0
    return {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "Sortino": sortino,
            "MDD": mdd, "Calmar": calmar, "Sterling": 0.0}


def risk_metrics(returns) -> dict:
    """실 엔진 우선. 실패 시 mock."""
    s = pd.Series(returns).dropna()
    arr = s.to_numpy(dtype=float)
    if _HAS_RISK and len(arr) > 0:
        try:
            raw = _risk_all_metrics(arr)
            out = _normalize_risk_keys(dict(raw))
            for k, v in _mock_risk_metrics(s).items():
                out.setdefault(k, v)
            # 스칼라 보장
            return {k: _scalar(v) for k, v in out.items()}
        except Exception as e:
            _ERRORS.append(f"calculate_all_metrics 실패: {e}")
    return _mock_risk_metrics(s)


# ============================================================
# Classification 지표
# ============================================================
# 분류 지표: 스칼라가 아닌 항목은 제외
_SKIP_CLS_KEYS = {"class_distribution", "confusion_matrix", "labels"}


def classification_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    raw: dict = {}
    if _HAS_CLS:
        try:
            raw = dict(_real_cls_metrics(y_true, y_pred))
        except Exception as e:
            _ERRORS.append(f"calculate_all_classification_metrics 실패: {e}")

    if not raw:
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            f1_score,
            matthews_corrcoef,
            precision_score,
            recall_score,
        )
        raw = {
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "mcc": matthews_corrcoef(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        }

    out: dict = {}
    for k, v in raw.items():
        # dict/object 스킵
        if k.lower() in _SKIP_CLS_KEYS or isinstance(v, dict):
            continue
        lk = k.lower()
        if "balanced" in lk and "acc" in lk:
            key = "balanced_accuracy"
        elif "accuracy" in lk or lk == "acc":
            key = "accuracy"
        elif lk == "mcc":
            key = "mcc"
        elif "macro" in lk and "f1" in lk:
            key = "macro_f1"
        elif lk in ("f1", "f1_score"):
            key = "macro_f1"
        elif "macro" in lk and "precision" in lk:
            key = "macro_precision"
        elif "macro" in lk and "recall" in lk:
            key = "macro_recall"
        else:
            key = k
        val = _scalar(v)
        if not np.isfinite(val):
            continue   # nan 은 아예 빼버림
        out[key] = val
    return out


def confusion(y_true, y_pred) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if _HAS_CLS:
        try:
            cm = _real_cm(y_true, y_pred, labels=_LABELS)
            if isinstance(cm, pd.DataFrame):
                return cm
            return pd.DataFrame(
                np.asarray(cm),
                index=["true_down", "true_neutral", "true_up"],
                columns=["pred_down", "pred_neutral", "pred_up"])
        except Exception as e:
            _ERRORS.append(f"confusion_matrix_multiclass 실패: {e}")

    from sklearn.metrics import confusion_matrix as _cm
    return pd.DataFrame(
        _cm(y_true, y_pred, labels=_LABELS),
        index=["true_down", "true_neutral", "true_up"],
        columns=["pred_down", "pred_neutral", "pred_up"])


# ============================================================
# Walk Forward
# ============================================================
@dataclass
class WalkForwardResult:
    folds: pd.DataFrame
    mean_sharpe: float
    std_sharpe: float
    positive_folds: int
    total_folds: int


def _fold_row(i, p, sig, cost):
    ret = p.pct_change().fillna(0)
    pos = sig.shift(1).fillna(0)
    turn = pos.diff().abs().fillna(0)
    strat = pos * ret - turn * cost
    m = risk_metrics(strat)
    eq = (1 + strat).cumprod()
    return {
        "fold": i + 1,
        "start": p.index[0].date(),
        "end": p.index[-1].date(),
        "sharpe": _scalar(m.get("Sharpe", 0.0)),
        "return": float(eq.iloc[-1] - 1) if len(eq) else 0.0,
        "mdd": _scalar(m.get("MDD", 0.0)),
    }


def _wrap_wf(df: pd.DataFrame) -> WalkForwardResult:
    return WalkForwardResult(
        folds=df,
        mean_sharpe=float(df["sharpe"].mean()) if len(df) else 0.0,
        std_sharpe=float(df["sharpe"].std(ddof=0)) if len(df) else 0.0,
        positive_folds=int((df["sharpe"] > 0).sum()) if len(df) else 0,
        total_folds=len(df),
    )


def _mock_walk_forward(prices, signals, cost, n_folds) -> WalkForwardResult:
    edges = np.linspace(0, len(prices), n_folds + 1).astype(int)
    rows = []
    for i in range(n_folds):
        s, e = edges[i], edges[i + 1]
        if e - s < 30:
            continue
        rows.append(_fold_row(i, prices.iloc[s:e], signals.iloc[s:e], cost))
    return _wrap_wf(pd.DataFrame(rows))


@st.cache_data(show_spinner=False)
def walk_forward(prices: pd.Series, signals: pd.Series,
                 cost: float, n_folds: int = 6) -> WalkForwardResult:
    """실 엔진(expanding_splits) 우선. 실패 시 mock."""
    if _HAS_WF:
        try:
            n = len(prices)
            splits = _real_expanding_splits(
                n, n_folds=n_folds, min_train=100, horizon=5)
            rows = []
            for i, tr, va in _real_iter_splits(splits):
                p = prices.iloc[va]
                sig = signals.iloc[va]
                if len(p) < 10:
                    continue
                rows.append(_fold_row(i, p, sig, cost))
            if rows:
                return _wrap_wf(pd.DataFrame(rows))
        except Exception as e:
            _ERRORS.append(f"walk_forward 실패: {e}")
    return _mock_walk_forward(prices, signals, cost, n_folds)


# ============================================================
# Baseline 비교
# ============================================================
try:
    from evaluation.baseline import (
        always_up as _real_always_up,
    )
    from evaluation.baseline import (
        multiclass_majority_class as _real_multi_majority,
    )
    from evaluation.baseline import (
        previous_direction as _real_prev_dir,
    )
    _HAS_BASELINE = True
except Exception as _e:
    _HAS_BASELINE = False
    _ERRORS.append(f"baseline: {_e}")


def baseline_comparison(y_train, y_valid, y_pred_model) -> dict:
    """모델 + baseline 3종 정확도. 실 엔진 우선, fallback sklearn."""
    from collections import Counter

    from sklearn.metrics import accuracy_score

    y_train = np.asarray(y_train, dtype=int)
    y_valid = np.asarray(y_valid, dtype=int)
    y_pred = np.asarray(y_pred_model, dtype=int)

    out = {"model": float(accuracy_score(y_valid, y_pred))}

    # majority class (fallback 공통)
    mc = Counter(y_train).most_common(1)[0][0]
    out["majority"] = float((y_valid == mc).mean())

    # always_up (fallback 공통)
    out["always_up"] = float((y_valid == 1).mean())

    # previous_direction (persistence)
    y_prev = np.concatenate([[y_train[-1]], y_valid[:-1]])
    out["previous_direction"] = float((y_valid == y_prev).mean())

    # 실 엔진 값으로 덮어쓰기 (성공한 것만)
    if _HAS_BASELINE:
        n = len(y_valid)
        try:
            out["always_up"] = float(accuracy_score(y_valid, _real_always_up(n)))
        except Exception as e:
            _ERRORS.append(f"always_up: {e}")
        try:
            pred = _real_multi_majority(y_train, n)
            out["majority"] = float(accuracy_score(y_valid, pred))
        except Exception:
            try:
                pred = _real_multi_majority(y_train, n_valid=n)
                out["majority"] = float(accuracy_score(y_valid, pred))
            except Exception as e:
                _ERRORS.append(f"majority: {e}")
        try:
            pred = _real_prev_dir(y_prev)
            out["previous_direction"] = float(accuracy_score(y_valid, pred))
        except Exception as e:
            _ERRORS.append(f"previous_direction: {e}")

    out["best_baseline"] = max(
        out["majority"], out["always_up"], out["previous_direction"])
    out["edge_vs_best"] = out["model"] - out["best_baseline"]
    return out


# ============================================================
# Class Distribution
# ============================================================
_CLASS_NAMES = {-1: "down", 0: "neutral", 1: "up"}


def class_distribution(y_true, y_pred=None) -> pd.DataFrame:
    """실제 라벨과 (옵션) 예측 라벨의 클래스 분포.

    Returns
    -------
    DataFrame with columns: class, actual_count, actual_share,
                            pred_count, pred_share
    """
    y_true = np.asarray(y_true, dtype=int)
    n = max(len(y_true), 1)
    labels = [-1, 0, 1]

    rows = []
    for l in labels:
        row = {
            "class": _CLASS_NAMES[l],
            "actual_count": int((y_true == l).sum()),
            "actual_share": float((y_true == l).sum() / n),
        }
        if y_pred is not None:
            yp = np.asarray(y_pred, dtype=int)
            row["pred_count"] = int((yp == l).sum())
            row["pred_share"] = float((yp == l).sum() / max(len(yp), 1))
        rows.append(row)

    df = pd.DataFrame(rows).set_index("class")

    # 실 엔진 함수가 있으면 actual_share 값을 그걸로 재확인 (선택)
    try:
        from evaluation.evaluation_backtest import class_distribution as _real_cd
        real = _real_cd(y_true, labels=labels)
        # real: {label: share} 형태로 반환된다고 가정
        if isinstance(real, dict):
            for l, v in real.items():
                name = _CLASS_NAMES.get(int(l))
                if name in df.index:
                    df.loc[name, "actual_share"] = float(v)
    except Exception:
        pass

    return df


def class_distribution_chart(df: pd.DataFrame):
    """Class Distribution 시각화용 plotly Figure."""
    import plotly.graph_objects as go

    classes = df.index.tolist()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Actual",
        x=classes, y=df["actual_share"].values,
        marker_color="#7fd1ff",
        text=[f"{v*100:.1f}%" for v in df["actual_share"].values],
        textposition="outside",
    ))
    if "pred_share" in df.columns:
        fig.add_trace(go.Bar(
            name="Predicted",
            x=classes, y=df["pred_share"].values,
            marker_color="#ffb347",
            text=[f"{v*100:.1f}%" for v in df["pred_share"].values],
            textposition="outside",
        ))
    fig.update_layout(
        barmode="group",
        template="plotly_dark",
        height=320,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(tickformat=".0%", gridcolor="rgba(255,255,255,0.06)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        legend=dict(orientation="h", y=1.12, x=0),
    )
    return fig
