"""임시 mock. 실제 레포 함수가 준비되면 services/*에서 이 모듈 호출만 지우면 된다."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


def ohlcv(dataset: str, start: date, end: date, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end, name="date")
    n = len(idx)
    rets = rng.normal(0.0003, 0.011, n)
    close = 100 * np.exp(np.cumsum(rets))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
    volume = rng.integers(1_000_000, 20_000_000, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def features(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["ret_1"] = df["close"].pct_change()
    out["ret_5"] = df["close"].pct_change(5)
    out["ret_20"] = df["close"].pct_change(20)
    out["vol_20"] = out["ret_1"].rolling(20).std()
    out["ma_gap"] = df["close"].rolling(5).mean() / df["close"].rolling(20).mean() - 1
    out["rsi_14"] = _rsi(df["close"], 14)
    out["macd"] = _macd(df["close"])
    out["volume_z"] = ((df["volume"] - df["volume"].rolling(20).mean())
                       / df["volume"].rolling(20).std())
    if feature_set in ("Technical + Market", "Full"):
        out["market_ret_1"] = out["ret_1"] * 0.9 + 0.0001
    if feature_set in ("Technical + Macro", "Full"):
        out["macro_proxy"] = np.sin(np.arange(len(df)) / 60.0)
    return out


def labels(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    fwd = df["close"].shift(-horizon) / df["close"] - 1
    q1, q2 = fwd.quantile([1 / 3, 2 / 3])
    lab = pd.Series(1, index=df.index, dtype=int)
    lab[fwd <= q1] = 0
    lab[fwd >= q2] = 2
    lab[fwd.isna()] = -1
    return lab


def _rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _macd(close: pd.Series) -> pd.Series:
    return (close.ewm(span=12, adjust=False).mean()
            - close.ewm(span=26, adjust=False).mean())
