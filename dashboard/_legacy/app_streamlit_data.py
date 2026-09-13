# app_streamlit_data.py
"""
Streamlit 앱에서 HuggingFace 데이터셋을 직접 로드하는 모듈.

사용:
    import streamlit as st
    from app_streamlit_data import load_kospi200, load_stock, list_stocks

    df_index = load_kospi200()
    df_stock = load_stock("005930")          # 삼성전자
    codes    = list_stocks()                  # 전체 종목코드 목록
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

# ============================================================
# 설정
# ============================================================
REPO_ID = "qurious-quant/alphastack-krx-dev"
REPO_TYPE = "dataset"

KOSPI200_PATH = "small/features_labels_kospi200_dev.csv"

STOCK_FEATURES_PATH = "small/features_labels_stocks30_dev.csv"
STOCK_CODES_PATH    = "small/sample_codes.json"

_TICKER_CANDIDATES = ("ticker", "code", "stock_code", "symbol",
                      "종목코드", "단축코드", "isu_cd", "isu_cd_short")

OHLCV = ["open", "high", "low", "close", "volume"]


# ============================================================
# 내부 유틸
# ============================================================
def _download(path: str) -> str:
    return hf_hub_download(repo_id=REPO_ID, filename=path, repo_type=REPO_TYPE)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
    return df


def _require_ohlcv(df: pd.DataFrame, source: str) -> None:
    missing = [c for c in OHLCV if c not in df.columns]
    if missing:
        raise ValueError(f"{source} 에 필요한 컬럼이 없습니다: {missing}")


# ============================================================
# ① KOSPI200 지수
# ============================================================
@st.cache_data(show_spinner="KOSPI200 데이터 불러오는 중…")
def load_kospi200() -> pd.DataFrame:
    path = _download(KOSPI200_PATH)
    df = pd.read_csv(path)
    df = _normalize(df)
    _require_ohlcv(df, "KOSPI200")
    return df


# ============================================================
# ② 개별종목 (30종목 통합 CSV)
# ============================================================
@st.cache_data(show_spinner=False)
def _stock_raw() -> pd.DataFrame:
    path = _download(STOCK_FEATURES_PATH)
    df = pd.read_csv(path, dtype=str)
    for c in df.columns:
        if c in _TICKER_CANDIDATES:
            continue
        try:
            df[c] = pd.to_numeric(df[c])
        except (ValueError, TypeError):
            pass
    df = _normalize(df)
    return df


@st.cache_data(show_spinner=False)
def _ticker_col() -> str:
    cols = _stock_raw().reset_index().columns
    for c in _TICKER_CANDIDATES:
        if c in cols:
            return c
    raise ValueError(
        f"종목 식별 컬럼을 찾을 수 없습니다.\n"
        f"실제 컬럼: {list(cols)}\n"
        f"_TICKER_CANDIDATES 에 추가하세요."
    )


@st.cache_data(show_spinner=False)
def list_stocks() -> list[str]:
    df = _stock_raw().reset_index()
    col = _ticker_col()
    codes = df[col].dropna().astype(str).str.strip()
    return sorted(codes.unique().tolist())


@st.cache_data(show_spinner="종목 데이터 불러오는 중…")
def load_stock(code: str) -> pd.DataFrame:
    df = _stock_raw().reset_index()
    col = _ticker_col()
    sub = df[df[col].astype(str).str.strip() == str(code).strip()]

    if sub.empty:
        avail = list_stocks()
        raise KeyError(
            f"종목코드 {code} 없음. "
            f"사용 가능: {avail[:5]} … (총 {len(avail)}개)"
        )

    sub = sub.drop(columns=[col]).set_index("date").sort_index()
    _require_ohlcv(sub, f"종목 {code}")
    return sub


@st.cache_data(show_spinner=False)
def load_stocks(codes: Iterable[str]) -> dict[str, pd.DataFrame]:
    return {c: load_stock(c) for c in codes}


@st.cache_data(show_spinner=False)
def stock_metadata() -> pd.DataFrame:
    import json
    try:
        path = _download(STOCK_CODES_PATH)
    except Exception:
        return pd.DataFrame(columns=["code", "name"])

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        rows = [{"code": str(k), **(v if isinstance(v, dict) else {"name": v})}
                for k, v in raw.items()]
    elif isinstance(raw, list):
        rows = []
        for item in raw:
            if isinstance(item, dict):
                rows.append({k: str(v) for k, v in item.items()})
            else:
                rows.append({"code": str(item)})
    else:
        return pd.DataFrame(columns=["code", "name"])

    out = pd.DataFrame(rows)
    for c in ("code", "ticker", "stock_code", "symbol", "종목코드"):
        if c in out.columns:
            out = out.rename(columns={c: "code"})
            break
    return out


# ============================================================
# ③ 사용 예 — Streamlit 페이지
# ============================================================
def render_page() -> None:
    st.set_page_config(page_title="AlphaStack 데이터 뷰어", layout="wide")
    st.title("AlphaStack 데이터 뷰어")

    tab_index, tab_stock = st.tabs(["KOSPI200", "개별종목"])

    with tab_index:
        df = load_kospi200()
        st.caption(f"{len(df):,}행 · {df.index.min().date()} ~ {df.index.max().date()}")
        start, end = st.date_input(
            "기간",
            value=(df.index.min().date(), df.index.max().date()),
            key="kospi_range",
        )
        view = df.loc[str(start):str(end)]
        st.line_chart(view["close"], height=320)
        st.dataframe(view.tail(200), use_container_width=True)

    with tab_stock:
        codes = list_stocks()
        if not codes:
            st.warning("종목코드를 찾지 못했습니다.")
            return

        meta = stock_metadata()
        label_map = {}
        if not meta.empty and "name" in meta.columns:
            label_map = dict(zip(meta["code"], meta["name"]))

        def _fmt(c):
            n = label_map.get(str(c))
            return f"{c} · {n}" if n else str(c)

        code = st.selectbox("종목", codes, format_func=_fmt, key="stock_code")
        sdf = load_stock(code)
        st.caption(
            f"{_fmt(code)} · {len(sdf):,}행 · "
            f"{sdf.index.min().date()} ~ {sdf.index.max().date()}"
        )
        st.line_chart(sdf["close"], height=320)
        st.dataframe(sdf.tail(200), use_container_width=True)


if __name__ == "__main__":
    render_page()
