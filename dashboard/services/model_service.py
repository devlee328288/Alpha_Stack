# dashboard/services/model_service.py
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# ── 실제 모델 빌더 ──────────────────────────────────────────
_HAS_REAL_BUILDERS = False
_BUILDERS: dict = {}
try:
    from models.lightgbm import build_lightgbm_baseline
    from models.logistic import build_logistic_baseline
    from models.random_forest import build_random_forest_baseline
    from models.xgboost import build_xgboost_baseline
    _BUILDERS = {
        "Logistic Regression": build_logistic_baseline,
        "Random Forest":       build_random_forest_baseline,
        "LightGBM":            build_lightgbm_baseline,
        "XGBoost":             build_xgboost_baseline,
    }
    _HAS_REAL_BUILDERS = True
except Exception:
    pass

# ── 실제 분류 지표 ──────────────────────────────────────────
try:
    from models.experiment import (
        classification_metrics as _real_cls_metrics,
    )
    _HAS_REAL_METRICS = True
except Exception:
    _HAS_REAL_METRICS = False


@dataclass
class ModelResult:
    name: str
    accuracy: float
    balanced_accuracy: float
    mcc: float
    macro_f1: float
    recalls: dict          # {"down": ..., "neutral": ..., "up": ...}
    y_true: np.ndarray     # -1/0/1
    y_pred: np.ndarray     # -1/0/1
    y_proba: np.ndarray    # columns: [down, neutral, up]
    test_index: pd.Index
    feature_importance: pd.Series
    runtime_sec: float
    engine: str            # "real" | "sklearn"


# ── 모델 빌드 ─────────────────────────────────────────────
def _build(name: str, seed: int):
    if _HAS_REAL_BUILDERS and name in _BUILDERS:
        try:
            return _BUILDERS[name](random_state=seed)
        except TypeError:
            # 일부 빌더가 random_state 를 안 받을 수도
            return _BUILDERS[name]()
    # fallback (sklearn)
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, random_state=seed)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed)
    if name == "XGBoost":
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                                 eval_metric="mlogloss", random_state=seed, n_jobs=-1)
        except Exception:
            return GradientBoostingClassifier(random_state=seed)
    if name == "LightGBM":
        try:
            from lightgbm import LGBMClassifier
            return LGBMClassifier(n_estimators=500, num_leaves=31, learning_rate=0.05,
                                  random_state=seed, n_jobs=-1)
        except Exception:
            return GradientBoostingClassifier(random_state=seed)
    raise ValueError(f"알 수 없는 모델: {name}")


@st.cache_resource(show_spinner=False)
def _fit_cached(name: str, key: str, seed: int, X: np.ndarray, y: np.ndarray):
    clf = _build(name, seed)
    clf.fit(X, y)
    return clf


def _safe_recalls(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import recall_score
    out = {}
    for lab, key in [(-1, "down"), (0, "neutral"), (1, "up")]:
        r = recall_score(y_true, y_pred, labels=[lab], average=None, zero_division=0)
        out[key] = float(r[0]) if len(r) else 0.0
    return out


def train(name: str, X: pd.DataFrame, y: pd.Series,
          seed: int = 42, test_ratio: float = 0.2) -> ModelResult:
    # ── 방어: NaN/inf 제거, 라벨은 -1/0/1 만 ──────────────
    df = X.join(y.rename("__y__"), how="inner")
    df = df.replace([np.inf, -np.inf], pd.NA).dropna()
    df = df[df["__y__"].isin([-1, 0, 1])]
    X = df.drop(columns="__y__").astype("float64")
    y = df["__y__"].astype(int)

    n = len(X)
    if n < 100:
        raise ValueError(f"학습 표본 부족: n={n}")

    split = int(n * (1 - test_ratio))
    Xtr, Xte = X.iloc[:split], X.iloc[split:]
    ytr, yte = y.iloc[:split], y.iloc[split:]

    t0 = time.time()
    key = f"{X.shape}|{Xtr.index[0]}|{Xtr.index[-1]}|{seed}"
    clf = _fit_cached(name, key, seed, Xtr.values, ytr.values)
    runtime = time.time() - t0

    proba_raw = clf.predict_proba(Xte.values)
    classes = list(getattr(clf, "classes_", [-1, 0, 1]))
    # classes_ 순서를 [-1, 0, 1] 로 강제 정렬
    order = [classes.index(c) for c in (-1, 0, 1) if c in classes]
    proba = proba_raw[:, order] if len(order) == proba_raw.shape[1] else proba_raw
    pred = np.array([-1, 0, 1])[proba.argmax(axis=1)]

    # ── 실제 지표 시도 ────────────────────────────────────
    engine = "sklearn"
    if _HAS_REAL_METRICS:
        try:
            m = _real_cls_metrics(yte.values, pred)
            accuracy = m.get("accuracy", float((pred == yte.values).mean()))
            macro_f1 = m.get("macro_f1", 0.0)
            down_recall = m.get("down_recall", 0.0)
            # 나머지 지표는 sklearn 으로
            from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
            bal = float(balanced_accuracy_score(yte.values, pred))
            mcc = float(matthews_corrcoef(yte.values, pred))
            recalls = _safe_recalls(yte.values, pred)
            recalls["down"] = float(down_recall)  # real 값으로 덮어씀
            engine = "real"
        except Exception:
            m = None
    if engine != "real":
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            f1_score,
            matthews_corrcoef,
        )
        accuracy = float(accuracy_score(yte.values, pred))
        bal = float(balanced_accuracy_score(yte.values, pred))
        mcc = float(matthews_corrcoef(yte.values, pred))
        macro_f1 = float(f1_score(yte.values, pred, average="macro", zero_division=0))
        recalls = _safe_recalls(yte.values, pred)

    if hasattr(clf, "feature_importances_"):
        imp = pd.Series(clf.feature_importances_, index=X.columns)
    elif hasattr(clf, "coef_"):
        imp = pd.Series(np.abs(clf.coef_).mean(axis=0), index=X.columns)
    else:
        imp = pd.Series(0.0, index=X.columns)
    imp = imp.sort_values(ascending=False)

    return ModelResult(
        name=name,
        accuracy=float(accuracy),
        balanced_accuracy=bal,
        mcc=mcc,
        macro_f1=macro_f1,
        recalls=recalls,
        y_true=yte.values.astype(int),
        y_pred=pred.astype(int),
        y_proba=proba,
        test_index=Xte.index,
        feature_importance=imp,
        runtime_sec=runtime,
        engine=engine,
    )
