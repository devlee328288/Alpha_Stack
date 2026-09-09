import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)
from tqdm import tqdm

from step1_core_features import compute_atr, compute_base, compute_log_rv
from focal_classifier import (
    FocalConfig,
    build_features,
    make_labels,
    predict_focal,
    train_focal_model,
)

warnings.filterwarnings("ignore")

LABELS = [0, 1, 2]
GAP_DAYS = 5


def compute_bands_flexible(
    df: pd.DataFrame,
    vol_period: int = 14,
    volume_period: int = 20,
    base_type: str = "SMA20",
    alpha_up: float = 1.0,
    alpha_down: float = 1.0,
    beta_up: float = 0.0,
    beta_down: float = 0.0,
) -> pd.DataFrame:
    base = compute_base(df, base_type)
    atr = compute_atr(df, period=vol_period)
    log_rv = compute_log_rv(df, period=volume_period)
    raw_up = alpha_up + beta_up * log_rv
    raw_down = alpha_down + beta_down * log_rv
    upper_width = atr * np.maximum(0.05, raw_up)
    lower_width = atr * np.maximum(0.05, raw_down)
    result = df.copy()
    result["base"] = base
    result["upper"] = base + upper_width
    result["lower"] = base - lower_width
    result["alpha_up"] = alpha_up
    result["alpha_down"] = alpha_down
    result["beta_up"] = beta_up
    result["beta_down"] = beta_down
    result["vol_period"] = vol_period
    result["volume_period"] = volume_period
    return result


def _params(row: pd.Series) -> dict:
    return {
        "alpha_up": float(row["alpha_up"]),
        "alpha_down": float(row["alpha_down"]),
        "beta_up": float(row["beta_up"]),
        "beta_down": float(row["beta_down"]),
        "vol_period": int(round(row["vol_period"])),
        "volume_period": int(round(row["volume_period"])),
    }


def _attach_bands_features(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    bands = compute_bands_flexible(df, **params)
    features = build_features(df, bands)
    return features, bands


def _signal_from_probability(proba: np.ndarray) -> np.ndarray:
    pred = np.full(len(proba), np.nan)
    valid = np.isfinite(proba).all(axis=1)
    if valid.any():
        pred[valid] = np.argmax(proba[valid], axis=1)
    return pred


def _strategy_returns(df: pd.DataFrame, signal: np.ndarray) -> np.ndarray:
    position = np.where(signal == 2, 1.0, np.where(signal == 0, -1.0, 0.0))
    market_ret = df["close"].pct_change().to_numpy(dtype=float)
    pos_shifted = np.concatenate([[0.0], position[:-1]])
    return pos_shifted * market_ret, position


def generate_signals_rolling(
    df: pd.DataFrame,
    fold_details: pd.DataFrame,
    focal_config: Optional[FocalConfig] = None,
) -> pd.DataFrame:
    """Train a fresh Focal classifier inside each expanding fold and predict only OOS.

    The inner validation split is chronological and occurs entirely before OOS.
    This replaces the previous hard rule/proxy-probability classifier.
    """
    cfg = focal_config or FocalConfig()
    y = make_labels(df)
    result = pd.DataFrame(index=df.index)
    for c in [
        "close",
        "signal",
        "position",
        "strategy_return",
        "p_down",
        "p_neutral",
        "p_up",
        "base",
        "upper",
        "lower",
    ]:
        result[c] = np.nan
    for c in [
        "alpha_up",
        "alpha_down",
        "beta_up",
        "beta_down",
        "vol_period",
        "volume_period",
    ]:
        result[c] = np.nan

    date_to_idx = {d: i for i, d in enumerate(df.index)}
    prev_position = 0.0
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    print(
        f"🧠 Focal classifier: γ={cfg.gamma:.2f}, α=[{cfg.alpha_down:.2f}, {cfg.alpha_neutral:.2f}, {cfg.alpha_up:.2f}]"
    )
    print(f"🧠 Device: {device}")

    for fold_no, row in tqdm(
        fold_details.iterrows(), total=len(fold_details), desc="Focal OOS 폴드"
    ):
        val_start = pd.Timestamp(row["val_start"])
        val_end = pd.Timestamp(row["val_end"])
        start_idx = date_to_idx.get(val_start)
        end_idx = date_to_idx.get(val_end)
        if start_idx is None or end_idx is None:
            continue

        # Outer training boundary is GAP days before OOS.
        train_end = start_idx - GAP_DAYS
        if train_end <= 150:
            continue

        params = _params(row)
        # IMPORTANT: the classifier needs the entire historical training window.
        # Using only a 35-day lookback here made train_local_end ~= 35, so the
        # training-data guard rejected every fold. Bands/features are causal,
        # therefore calculating them on all data up to OOS end is safe.
        calc_start = 0
        calc_end = end_idx + 1
        df_calc = df.iloc[calc_start:calc_end].copy()
        bands_calc = compute_bands_flexible(df_calc, **params)
        features_calc = build_features(df_calc, bands_calc)

        # Train only on rows before the gap. Labels near train_end need future data,
        # which is available in the historical dataset and does not touch OOS features.
        train_local_end = train_end
        try:
            model, scaler, feature_cols, stats = train_focal_model(
                features_calc,
                make_labels(df_calc),
                train_end=train_local_end,
                config=cfg,
                device=device,
            )
        except Exception as exc:
            print(f"⚠️ Focal fold {fold_no + 1} 실패: {exc}")
            continue

        oos_features = features_calc.iloc[
            start_idx - calc_start : end_idx - calc_start + 1
        ]
        proba = predict_focal(model, scaler, feature_cols, oos_features, device=device)
        signal = _signal_from_probability(proba)

        df_oos = df.iloc[start_idx : end_idx + 1]
        strategy_ret, position = _strategy_returns(df_oos, signal)
        pos_shifted = np.concatenate([[prev_position], position[:-1]])
        market_ret = df_oos["close"].pct_change().to_numpy(dtype=float)
        strategy_ret = pos_shifted * market_ret
        if len(position):
            prev_position = position[-1]

        mask = (df.index >= val_start) & (df.index <= val_end)
        result.loc[mask, "close"] = df.loc[mask, "close"]
        result.loc[mask, "signal"] = signal
        result.loc[mask, "position"] = position
        result.loc[mask, "strategy_return"] = strategy_ret
        result.loc[mask, "p_down"] = proba[:, 0]
        result.loc[mask, "p_neutral"] = proba[:, 1]
        result.loc[mask, "p_up"] = proba[:, 2]
        result.loc[mask, "base"] = bands_calc.iloc[
            start_idx - calc_start : end_idx - calc_start + 1
        ]["base"].to_numpy()
        result.loc[mask, "upper"] = bands_calc.iloc[
            start_idx - calc_start : end_idx - calc_start + 1
        ]["upper"].to_numpy()
        result.loc[mask, "lower"] = bands_calc.iloc[
            start_idx - calc_start : end_idx - calc_start + 1
        ]["lower"].to_numpy()
        for c in [
            "alpha_up",
            "alpha_down",
            "beta_up",
            "beta_down",
            "vol_period",
            "volume_period",
        ]:
            result.loc[mask, c] = params[c]

        print(
            f"   Fold {fold_no + 1}: inner Macro-F1={stats['inner_macro_f1']:.4f}, "
            f"epoch={int(stats['best_epoch'])}, features={int(stats['n_features'])}"
        )

    return result


def generate_signals_single(
    df: pd.DataFrame, params: dict, focal_config: Optional[FocalConfig] = None
) -> pd.DataFrame:
    """Median-parameter benchmark using a chronological train/holdout split."""
    cfg = focal_config or FocalConfig()
    bands = compute_bands_flexible(df, **params)
    features = build_features(df, bands)
    y = make_labels(df)
    train_end = max(150, len(df) - 63)
    model, scaler, cols, _ = train_focal_model(
        features, y, train_end=train_end, config=cfg
    )
    proba = predict_focal(
        model,
        scaler,
        cols,
        features,
        device="cuda" if __import__("torch").cuda.is_available() else "cpu",
    )
    signal = _signal_from_probability(proba)
    strategy_ret, position = _strategy_returns(df, signal)
    out = bands.copy()
    out["signal"] = signal
    out["position"] = position
    out["strategy_return"] = strategy_ret
    out["p_down"] = proba[:, 0]
    out["p_neutral"] = proba[:, 1]
    out["p_up"] = proba[:, 2]
    return out


def calculate_metrics(returns: np.ndarray) -> dict:
    clean = returns[np.isfinite(returns)]
    if len(clean) == 0:
        return {
            k: np.nan
            for k in ["sharpe", "cagr", "mdd", "calmar", "win_rate", "profit_factor"]
        }
    std = np.std(clean)
    sharpe = np.mean(clean) / std * np.sqrt(252) if std > 0 else 0.0
    wealth = np.cumprod(1 + clean)
    cagr = wealth[-1] ** (252 / len(clean)) - 1 if wealth[-1] > 0 else -1.0
    peak = np.maximum.accumulate(wealth)
    mdd = np.max((peak - wealth) / peak) if len(wealth) else 0.0
    gains = clean[clean > 0].sum()
    losses = abs(clean[clean < 0].sum())
    return {
        "sharpe": sharpe,
        "cagr": cagr,
        "mdd": mdd,
        "calmar": cagr / mdd if mdd > 0 else 0.0,
        "win_rate": np.mean(clean > 0),
        "profit_factor": gains / losses if losses > 0 else np.inf,
    }


def evaluate_signals(df_signals: pd.DataFrame, y_true: np.ndarray) -> dict:
    mask = np.isfinite(df_signals["signal"].to_numpy()) & np.isfinite(y_true)
    if mask.sum() == 0:
        return {
            **calculate_metrics(np.array([])),
            "accuracy": np.nan,
            "f1_macro": np.nan,
            "balanced_acc": np.nan,
            "mcc": np.nan,
            "ratio_down": np.nan,
            "ratio_neutral": np.nan,
            "ratio_up": np.nan,
            "confusion_matrix": None,
            "report": "",
            "macro_pr_auc": np.nan,
            "pr_auc_down": np.nan,
            "pr_auc_neutral": np.nan,
            "pr_auc_up": np.nan,
        }
    yt = y_true[mask].astype(int)
    yp = df_signals.loc[mask, "signal"].to_numpy().astype(int)
    result = calculate_metrics(
        df_signals.loc[mask, "strategy_return"].to_numpy(dtype=float)
    )
    result.update(
        {
            "accuracy": accuracy_score(yt, yp),
            "f1_macro": f1_score(yt, yp, average="macro", zero_division=0),
            "balanced_acc": balanced_accuracy_score(yt, yp),
            "mcc": matthews_corrcoef(yt, yp),
            "ratio_down": np.mean(yp == 0),
            "ratio_neutral": np.mean(yp == 1),
            "ratio_up": np.mean(yp == 2),
            "confusion_matrix": confusion_matrix(yt, yp, labels=LABELS),
            "report": classification_report(
                yt,
                yp,
                labels=LABELS,
                target_names=["하락", "중립", "상승"],
                digits=4,
                zero_division=0,
            ),
        }
    )
    if all(c in df_signals.columns for c in ["p_down", "p_neutral", "p_up"]):
        proba = df_signals.loc[mask, ["p_down", "p_neutral", "p_up"]].to_numpy(
            dtype=float
        )
        if np.isfinite(proba).all():
            y_onehot = np.eye(3)[yt]
            result["macro_pr_auc"] = average_precision_score(
                y_onehot, proba, average="macro"
            )
            result["pr_auc_down"] = average_precision_score(y_onehot[:, 0], proba[:, 0])
            result["pr_auc_neutral"] = average_precision_score(
                y_onehot[:, 1], proba[:, 1]
            )
            result["pr_auc_up"] = average_precision_score(y_onehot[:, 2], proba[:, 2])
    return result
