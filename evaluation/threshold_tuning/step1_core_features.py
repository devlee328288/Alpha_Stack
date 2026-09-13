import functools

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

_REPO_ID = "qurious-quant/alphastack-krx-dev"
_KOSPI200_CSV = "small/features_labels_kospi200_dev.csv"
_STOCKS30_CSV = "small/features_labels_stocks30_dev.csv"

ALL_TICKERS = "__ALL__"
MARKET_TICKER = "KOSPI200"


# ============================================================
# 0. 데이터 로드 — 지수 / 개별종목 / 유니버스
# ============================================================
def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if "code" in df.columns and df["code"].nunique() > 1:
        df = df.sort_values(["code", "date"])
    else:
        df = df.sort_values("date")
    df = df.set_index("date")
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"데이터에 필요한 컬럼이 없습니다: {missing}")
    return df


@functools.lru_cache(maxsize=2)
def _load_kospi200_raw() -> pd.DataFrame:
    path = hf_hub_download(
        repo_id=_REPO_ID, filename=_KOSPI200_CSV, repo_type="dataset",
    )
    return pd.read_csv(path)


@functools.lru_cache(maxsize=2)
def _load_stocks30_raw() -> pd.DataFrame:
    path = hf_hub_download(
        repo_id=_REPO_ID, filename=_STOCKS30_CSV, repo_type="dataset",
    )
    return pd.read_csv(
        path, dtype={"code": str}, low_memory=False,
    )


def load_data(ticker=None) -> pd.DataFrame:
    """
    Parameters
    ----------
    ticker : None | "KOSPI200" | "<code>" | "__ALL__"
        None / "KOSPI200" : 지수 (기존 동작)
        "<code>"          : stocks30 개별종목 (예 "005930")
        "__ALL__"         : stocks30 전체 (UNIVERSE 용, code 컬럼 유지)
    """
    if ticker is None or ticker == MARKET_TICKER:
        return _finalize(_load_kospi200_raw())

    if ticker == ALL_TICKERS:
        return _finalize(_load_stocks30_raw())

    df_all = _load_stocks30_raw()

    # 종목코드 정규화: 6자리 zero-padding
    code_str = str(ticker).strip()
    candidates = []
    # 1) 원본
    candidates.append(code_str)
    # 2) 6자리 zero-pad
    if code_str.isdigit():
        candidates.append(code_str.zfill(6))

    sub = None
    for cand in candidates:
        sub = df_all[df_all["code"] == cand]
        if not sub.empty:
            break

    if sub is None or sub.empty:
        avail = sorted(df_all["code"].unique().tolist())
        raise ValueError(
            f"종목코드 {ticker!r} 를 stocks30 에서 찾지 못했습니다. "
            f"앞 10개: {avail[:10]}"
        )
    return _finalize(sub)


def list_universe_tickers() -> list:
    """[{code, name, market, sector}, ...] — UNIVERSE 선택 UI 용."""
    df = _load_stocks30_raw()
    meta = (
        df[["code", "name", "market", "sector"]]
        .drop_duplicates(subset=["code"])
        .sort_values("code")
        .reset_index(drop=True)
    )
    return meta.to_dict("records")


# ============================================================
# 1. Base (중심선) 계산
# ============================================================
def compute_base(df: pd.DataFrame, base_type: str = "SMA20") -> pd.Series:
    close = df["close"]
    if base_type == "SMA20":
        return close.rolling(20).mean()
    elif base_type == "EMA20":
        return close.ewm(span=20, adjust=False).mean()
    elif base_type == "EMA30":
        return close.ewm(span=30, adjust=False).mean()
    elif base_type == "SMA60":
        return close.rolling(60).mean()
    else:
        raise ValueError(f"지원하지 않는 Base 타입: {base_type}")


# ============================================================
# 2. Volatility (변동성) 계산
# ============================================================
def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"].shift(1)
    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_natr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    atr = compute_atr(df, period)
    return atr / df["close"]


def compute_std_ret(df: pd.DataFrame, period: int = 20) -> pd.Series:
    if 'code' in df.columns:
        ret = df.groupby('code')['close'].pct_change()
    else:
        ret = df['close'].pct_change()
    return ret.rolling(period).std()


# ============================================================
# 3. Volume (거래량) 계산
# ============================================================
def compute_log_rv(df: pd.DataFrame, period: int = 20) -> pd.Series:
    volume = df["volume"]
    sma_vol = volume.rolling(period).mean()
    return np.log(volume / sma_vol)


def compute_volume_zscore(df: pd.DataFrame, period: int = 20) -> pd.Series:
    volume = df["volume"]
    sma_vol = volume.rolling(period).mean()
    std_vol = volume.rolling(period).std()
    return (volume - sma_vol) / std_vol


def compute_volume_shock(df: pd.DataFrame, period: int = 20) -> pd.Series:
    volume = df["volume"]
    sma_vol = volume.rolling(period).mean()
    return volume / sma_vol - 1


# ============================================================
# 4. 기준선 (Upper / Lower)
# ============================================================
def compute_bands(
    df: pd.DataFrame,
    base_type: str = "SMA20",
    vol_type: str = "ATR14",
    volume_type: str = "LogRV20",
    alpha: float = 1.0,
    beta: float = 0.0,
    asym: bool = False,
    alpha_up: float = None,
    alpha_down: float = None,
    beta_up: float = None,
    beta_down: float = None,
) -> pd.DataFrame:
    base = compute_base(df, base_type)

    if vol_type == "ATR14":
        vol = compute_atr(df, 14)
    elif vol_type == "NATR14":
        vol = compute_natr(df, 14)
    elif vol_type == "STD20":
        vol = compute_std_ret(df, 20)
    else:
        raise ValueError(f"지원하지 않는 Volatility 타입: {vol_type}")

    if volume_type is None:
        vol_effect = 1.0
    elif volume_type == "LogRV20":
        vol_effect = compute_log_rv(df, 20)
    elif volume_type == "Zscore20":
        vol_effect = compute_volume_zscore(df, 20)
    elif volume_type == "Shock20":
        vol_effect = compute_volume_shock(df, 20)
    else:
        raise ValueError(f"지원하지 않는 Volume 타입: {volume_type}")

    if asym:
        if alpha_up is None or alpha_down is None:
            raise ValueError("비대칭 모드에서는 alpha_up, alpha_down 필수")
        _beta_up = beta_up if beta_up is not None else beta
        _beta_down = beta_down if beta_down is not None else beta
        upper_width = vol * np.exp(alpha_up + _beta_up * vol_effect)
        lower_width = vol * np.exp(alpha_down + _beta_down * vol_effect)
    else:
        width = vol * np.exp(alpha + beta * vol_effect)
        upper_width = width
        lower_width = width

    result = df.copy()
    result["base"] = base
    result["upper"] = base + upper_width
    result["lower"] = base - lower_width
    return result


# ============================================================
# 5. 모든 피처 한 번에
# ============================================================
def prepare_all_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["base_SMA20"] = compute_base(df, "SMA20")
    df["base_EMA20"] = compute_base(df, "EMA20")
    df["base_EMA30"] = compute_base(df, "EMA30")
    df["base_SMA60"] = compute_base(df, "SMA60")
    df["ATR14"] = compute_atr(df, 14)
    df["NATR14"] = compute_natr(df, 14)
    df["STD20"] = compute_std_ret(df, 20)
    df["LogRV20"] = compute_log_rv(df, 20)
    df["VolZ20"] = compute_volume_zscore(df, 20)
    df["VolShock20"] = compute_volume_shock(df, 20)
    return df


if __name__ == "__main__":
    pass