# dashboard/services/data_loader.py
"""
지수 / 개별종목 / 유니버스 데이터 로더.
step1_core_features 와 동일 로직을 dashboard 에서 재사용.
"""
from __future__ import annotations

import functools
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

_REPO_ID = "qurious-quant/alphastack-krx-dev"
_KOSPI200_CSV = "small/features_labels_kospi200_dev.csv"
_STOCKS30_CSV = "small/features_labels_stocks30_dev.csv"

MARKET_TICKER = "KOSPI200"
ALL_TICKERS = "__ALL__"


@functools.lru_cache(maxsize=2)
def _load_kospi200_raw() -> pd.DataFrame:
    path = hf_hub_download(
        repo_id=_REPO_ID, filename=_KOSPI200_CSV, repo_type="dataset",
    )
    return pd.read_csv(path, low_memory=False)


@functools.lru_cache(maxsize=2)
def _load_stocks30_raw() -> pd.DataFrame:
    path = hf_hub_download(
        repo_id=_REPO_ID, filename=_STOCKS30_CSV, repo_type="dataset",
    )
    return pd.read_csv(path, dtype={"code": str}, low_memory=False)


def _finalize(df: pd.DataFrame, multi_code: bool = False) -> pd.DataFrame:
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    if multi_code and "code" in df.columns:
        df = df.sort_values(["code", "date"]).set_index("date")
    else:
        df = df.sort_values("date").set_index("date")
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 없음: {missing}")
    return df


def _normalize_code(ticker) -> list[str]:
    """짧은 코드도 6자리 zero-pad 로 시도."""
    s = str(ticker).strip()
    candidates = [s]
    if s.isdigit():
        z = s.zfill(6)
        if z not in candidates:
            candidates.append(z)
    return candidates


def load_market_data(scope: str, ticker: str) -> pd.DataFrame:
    """
    scope : "MARKET" | "STOCK"
    ticker : "KOSPI200" (MARKET) | "<code>" (STOCK)
    """
    if scope == "MARKET":
        return _finalize(_load_kospi200_raw())

    df_all = _load_stocks30_raw()
    for cand in _normalize_code(ticker):
        sub = df_all[df_all["code"] == cand]
        if not sub.empty:
            return _finalize(sub, multi_code=False)

    avail = sorted(df_all["code"].unique().tolist())
    raise ValueError(
        f"종목코드 {ticker!r} 를 stocks30 에서 찾지 못했습니다. "
        f"앞 10개: {avail[:10]}"
    )


def list_universe_tickers() -> list[dict]:
    df = _load_stocks30_raw()
    meta = (
        df[["code", "name", "market", "sector"]]
        .drop_duplicates(subset=["code"])
        .sort_values("code")
        .reset_index(drop=True)
    )
    return meta.to_dict("records")


def ticker_label(ticker: str) -> str:
    """005930 → '005930 · 삼성전자' 형태."""
    if ticker == MARKET_TICKER:
        return "KOSPI200"
    for m in list_universe_tickers():
        if m["code"] == ticker:
            name = m.get("name") or ""
            return f"{ticker} · {name}" if name else ticker
    return ticker


def clear_cache() -> None:
    _load_kospi200_raw.cache_clear()
    _load_stocks30_raw.cache_clear()


# ═══════════════════════════════════════════════════════════
# Full Universe (전종목)
# ═══════════════════════════════════════════════════════════
_FULL_CSV = "full/daily_price_dev.parquet"
_MIN_COLS = [
    "bas_dd", "code", "name", "market", "sector",
    "open", "high", "low", "close", "volume",
    "adj_open", "adj_high", "adj_low", "adj_close", "adj_close_tr",
    "market_cap", "value",
    "is_halted", "is_liquidation", "is_first_listing",
]


@functools.lru_cache(maxsize=1)
def _load_full_daily_raw() -> pd.DataFrame:
    """full/daily_price_dev.parquet — 444MB, 첫 로드 후 캐시."""
    path = hf_hub_download(
        repo_id=_REPO_ID, filename=_FULL_CSV, repo_type="dataset",
    )
    df = pd.read_parquet(path)
    keep = [c for c in _MIN_COLS if c in df.columns]
    return df[keep].copy()


def list_full_universe(
    market: str = "KOSPI",
    top_n: int | None = 200,
    exclude_halted: bool = True,
    exclude_liquidation: bool = True,
    exclude_first_listing: bool = True,
    min_market_cap: float = 0.0,
    price_col: str = "adj_close",
) -> list[dict]:
    """
    필터 적용된 종목 리스트.

    Parameters
    ----------
    market : "KOSPI" | "KOSPI+KOSDAQ" | "ALL"
    top_n : 시가총액 상위 N. None 이면 전체
    price_col : "adj_close" | "close" | "adj_close_tr"
    """
    df = _load_full_daily_raw()

    # 1) 최신 거래일만 (시가총액 정렬 기준)
    latest_dd = df["bas_dd"].max()
    latest = df[df["bas_dd"] == latest_dd].copy()

    # 2) 시장 필터
    if market == "KOSPI":
        latest = latest[latest["market"] == "KOSPI"]
    elif market == "KOSPI+KOSDAQ":
        latest = latest[latest["market"].isin(["KOSPI", "KOSDAQ"])]

    # 3) 잡음 제외
    for flag, do_exclude in [
        ("is_halted", exclude_halted),
        ("is_liquidation", exclude_liquidation),
        ("is_first_listing", exclude_first_listing),
    ]:
        if do_exclude and flag in latest.columns:
            latest = latest[~latest[flag].fillna(False).astype(bool)]

    # 4) 시가총액 필터
    if min_market_cap > 0 and "market_cap" in latest.columns:
        latest = latest[latest["market_cap"] >= min_market_cap]

    # 5) 시총 정렬 + 상위 N
    if "market_cap" in latest.columns:
        latest = latest.sort_values("market_cap", ascending=False, kind="stable")
    if top_n is not None:
        latest = latest.head(top_n)

    # 6) 반환
    out = latest[["code", "name", "market", "sector"]].drop_duplicates("code")
    return out.to_dict("records")


def load_market_data_full(
    ticker: str,
    price_col: str = "adj_close",
) -> pd.DataFrame:
    """full parquet 에서 종목 하나 추출. price_col 을 'close' 로 통일."""
    df_all = _load_full_daily_raw()
    code_str = str(ticker).strip()
    if code_str.isdigit():
        code_str = code_str.zfill(6)

    sub = df_all[df_all["code"] == code_str]
    if sub.empty:
        raise ValueError(f"종목코드 {ticker!r} 를 full universe 에서 찾지 못했습니다.")

    sub = sub.copy()
    # 가격 컬럼 통일 (기존 stocks30 은 'close' 가 이미 adj)
    if price_col != "close" and price_col in sub.columns:
        sub["close"] = sub[price_col]
        if price_col in ("adj_close", "adj_close_tr"):
            # high/low/open 도 조정가로
            if "adj_open" in sub.columns:
                sub["open"] = sub["adj_open"]
            if "adj_high" in sub.columns:
                sub["high"] = sub["adj_high"]
            if "adj_low" in sub.columns:
                sub["low"] = sub["adj_low"]

    # bas_dd → date
    sub["date"] = pd.to_datetime(sub["bas_dd"], format="%Y%m%d", errors="coerce")
    sub = sub.dropna(subset=["date"]).sort_values("date").set_index("date")

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in sub.columns]
    if missing:
        raise ValueError(f"필수 컬럼 없음: {missing}")
    return sub


def clear_full_cache() -> None:
    _load_full_daily_raw.cache_clear()