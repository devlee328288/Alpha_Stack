"""개별 종목 랭킹 모델에 넣을 시점 정합 학습 표를 만든다.

업종 매핑이 준비되기 전 MVP는 매 거래일 KOSPI 보통주 중 시가총액 상위 50개를
후보로 삼는다. 이 모듈은 종목 피처를 만들기 전 단계인 후보 선정과 5거래일 라벨만
담당한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype

from evaluation.horizon import HOLDOUT_START
from features.indicators import bollinger_bands, macd_hist_ratio, percent_b, rsi, sma_gap
from features.returns import n_day_return
from features.volatility import atr, historical_volatility
from features.volume import obv, volume_ratio, volume_sma

DEFAULT_TOP_N = 50
STOCK_LABEL_HORIZON = 5
STOCK_NEUTRAL_BAND = 0.02
LABEL_TO_NUMBER = {"하락": -1, "중립": 0, "상승": 1}

STOCK_FEATURE_COLUMNS = (
    "sma_gap_5_20",
    "sma_gap_20_60",
    "rsi_14",
    "macd_hist_ratio",
    "bb_bandwidth",
    "bb_position",
    "atr_ratio",
    "hv_20",
    "vol_ratio_20",
    "obv_slope_20",
    "daily_return",
    "five_day_return",
)

REQUIRED_COLUMNS = {
    "bas_dd",
    "code",
    "market",
    "market_cap",
    "adj_open",
    "is_common_stock",
}

PANEL_PRICE_COLUMNS = {
    "bas_dd",
    "code",
    "market",
    "adj_open",
    "adj_high",
    "adj_low",
    "adj_close",
    "volume",
}


@dataclass(frozen=True)
class StockModelDataset:
    """날짜 그룹 워크포워드에 바로 넣을 개별종목 패널."""

    frame: pd.DataFrame
    feature_columns: tuple[str, ...] = STOCK_FEATURE_COLUMNS

    @property
    def x(self) -> pd.DataFrame:
        return self.frame.loc[:, self.feature_columns]

    @property
    def y(self) -> np.ndarray:
        return self.frame["label_numeric"].to_numpy(dtype=int)

    @property
    def groups(self) -> np.ndarray:
        return self.frame["bas_dd"].to_numpy(dtype=str)


def _normalize_dates(frame: pd.DataFrame) -> pd.Series:
    dates = (
        frame["bas_dd"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(8)
    )
    invalid = dates.isna() | ~dates.str.fullmatch(r"\d{8}")
    if invalid.any():
        examples = frame.loc[invalid, "bas_dd"].head(3).tolist()
        raise ValueError(f"YYYYMMDD로 해석할 수 없는 기준일이 있습니다: {examples}")
    return dates


def _validate_arguments(
    *,
    holdout_start: str,
    top_n: int,
    horizon: int,
    neutral_band: float,
) -> None:
    if not isinstance(holdout_start, str) or not holdout_start.isdigit():
        raise ValueError(f"holdout_start는 YYYYMMDD 문자열이어야 합니다: {holdout_start!r}")
    if len(holdout_start) != 8:
        raise ValueError(f"holdout_start는 YYYYMMDD 문자열이어야 합니다: {holdout_start!r}")
    if top_n <= 0:
        raise ValueError(f"top_n은 1 이상이어야 합니다: {top_n}")
    if horizon <= 0:
        raise ValueError(f"horizon은 1 이상이어야 합니다: {horizon}")
    if neutral_band < 0.0:
        raise ValueError(f"neutral_band는 0 이상이어야 합니다: {neutral_band}")


def build_stock_training_frame(
    daily_prices: pd.DataFrame,
    *,
    holdout_start: str = HOLDOUT_START,
    top_n: int = DEFAULT_TOP_N,
    horizon: int = STOCK_LABEL_HORIZON,
    neutral_band: float = STOCK_NEUTRAL_BAND,
) -> pd.DataFrame:
    """날짜별 KOSPI 보통주 시총 상위 N개의 5거래일 라벨을 만든다.

    ``is_common_stock``은 종목명이나 코드 모양으로 추측하지 않고 데이터 계층에서
    판정해 넘겨야 한다. 이 열이 없으면 조용히 전체 종목을 쓰지 않고 즉시 중단한다.

    계획서의 체결 규칙에 맞춰 T일 정보로 판단하고 T+1 수정시가에 진입한 뒤 T+6
    수정시가에 평가한다. 날짜는 종목별 행 번호가 아니라 KOSPI 전체 거래일 달력으로
    센다. 거래정지 등으로 정확한 진입·평가일 가격이 없으면 그 후보를 버리되, 가격이
    있는 시총 차순위 종목으로 채우지 않는다. 그렇게 채우면 미래의 거래 가능 여부로
    오늘의 후보를 바꾸는 누수가 생긴다.
    """
    _validate_arguments(
        holdout_start=holdout_start,
        top_n=top_n,
        horizon=horizon,
        neutral_band=neutral_band,
    )
    missing = REQUIRED_COLUMNS - set(daily_prices.columns)
    if missing:
        raise ValueError(f"개별 종목 학습표 필수 열이 없습니다: {sorted(missing)}")
    if not is_bool_dtype(daily_prices["is_common_stock"].dtype):
        raise TypeError("is_common_stock은 추측 문자열이 아니라 bool 열이어야 합니다.")
    if daily_prices["is_common_stock"].isna().any():
        raise ValueError("is_common_stock에 판정되지 않은 행이 있습니다.")

    source = daily_prices.copy()
    source["bas_dd"] = _normalize_dates(source)
    if (source["bas_dd"] >= holdout_start).any():
        first = str(source.loc[source["bas_dd"] >= holdout_start, "bas_dd"].min())
        raise RuntimeError(f"개별 종목 원천에 홀드아웃 행이 들어 있습니다: {first}")

    kospi = source.loc[source["market"].eq("KOSPI")].copy()
    if kospi.empty:
        raise ValueError("KOSPI 개별 종목 행이 없습니다.")
    if kospi.duplicated(["bas_dd", "code"]).any():
        duplicate = kospi.loc[
            kospi.duplicated(["bas_dd", "code"], keep=False), ["bas_dd", "code"]
        ].iloc[0]
        raise ValueError(
            "KOSPI에 같은 날짜·종목코드가 두 번 이상 있습니다: "
            f"{duplicate['bas_dd']} {duplicate['code']}"
        )

    calendar = sorted(kospi["bas_dd"].unique().tolist())
    if len(calendar) <= horizon + 1:
        raise ValueError(
            "T+1 진입·T+6 평가 라벨을 만들 거래일이 부족합니다: "
            f"거래일 {len(calendar)}개, 최소 {horizon + 2}개"
        )

    kospi["market_cap"] = pd.to_numeric(kospi["market_cap"], errors="coerce")
    kospi["adj_open"] = pd.to_numeric(kospi["adj_open"], errors="coerce")
    valid_market_cap = kospi["market_cap"].notna() & (kospi["market_cap"] > 0.0)
    ranked_source = kospi.loc[valid_market_cap].copy()
    common = ranked_source.loc[ranked_source["is_common_stock"]].copy()
    if common.empty:
        raise ValueError("KOSPI 보통주로 판정된 유효 행이 없습니다.")

    # 먼저 오늘 알 수 있는 시가총액으로 후보를 확정한다. 같은 시총이면 코드순으로
    # 고정해 실행할 때마다 동일한 50종목이 나오게 한다.
    ranked = common.sort_values(
        ["bas_dd", "market_cap", "code"],
        ascending=[True, False, True],
        kind="stable",
    )
    ranked["market_cap_rank"] = ranked.groupby("bas_dd", sort=False).cumcount() + 1
    candidates = ranked.loc[ranked["market_cap_rank"] <= top_n].copy()

    calendar_frame = pd.DataFrame({"bas_dd": calendar})
    calendar_frame["entry_bas_dd"] = calendar_frame["bas_dd"].shift(-1)
    calendar_frame["exit_bas_dd"] = calendar_frame["bas_dd"].shift(-(horizon + 1))
    candidates = candidates.merge(
        calendar_frame,
        on="bas_dd",
        how="left",
        validate="many_to_one",
    )
    selected_rows = len(candidates)
    # 마지막 horizon+1 거래일은 개발구간 안에 청산일이 없다. 빈 날짜끼리 조인하면
    # pandas가 여러 NaN을 같은 키로 세므로, 가격표를 붙이기 전에 명시적으로 비운다.
    candidates = candidates.loc[candidates["exit_bas_dd"].notna()].copy()

    valid_open = kospi["adj_open"].notna() & (kospi["adj_open"] > 0.0)
    open_prices = kospi.loc[valid_open, ["bas_dd", "code", "adj_open"]]
    entry_prices = open_prices.rename(
        columns={"bas_dd": "entry_bas_dd", "adj_open": "entry_adj_open"}
    )
    candidates = candidates.merge(
        entry_prices,
        on=["entry_bas_dd", "code"],
        how="left",
        validate="one_to_one",
    )
    exit_prices = open_prices.rename(
        columns={"bas_dd": "exit_bas_dd", "adj_open": "exit_adj_open"}
    )
    candidates = candidates.merge(
        exit_prices,
        on=["exit_bas_dd", "code"],
        how="left",
        validate="one_to_one",
    )
    candidates = candidates.loc[
        candidates["entry_adj_open"].notna() & candidates["exit_adj_open"].notna()
    ].copy()
    if candidates.empty:
        raise ValueError("정확한 T+1·T+6 수정시가가 있는 학습 후보가 없습니다.")

    candidates["fwd_return_5d"] = (
        candidates["exit_adj_open"] / candidates["entry_adj_open"] - 1.0
    )
    # 수익률을 먼저 계산해 ±2%와 비교하면 102/100-1이 0.020000...으로 표현되어
    # 정확히 경계인 값이 상승으로 넘어갈 수 있다. 계획서의 초과·미만 규칙을 가격
    # 경계로 직접 비교해 +2%와 -2%는 중립에 남긴다.
    up = candidates["exit_adj_open"] > candidates["entry_adj_open"] * (
        1.0 + neutral_band
    )
    down = candidates["exit_adj_open"] < candidates["entry_adj_open"] * (
        1.0 - neutral_band
    )
    candidates["label"] = np.select([up, down], ["상승", "하락"], default="중립")
    candidates["label_numeric"] = candidates["label"].map(LABEL_TO_NUMBER).astype("int8")
    candidates["market_cap_rank"] = candidates["market_cap_rank"].astype("int16")
    candidates = candidates.sort_values(
        ["bas_dd", "market_cap_rank", "code"], kind="stable"
    ).reset_index(drop=True)

    if candidates["bas_dd"].max() >= holdout_start:
        raise RuntimeError("개별 종목 학습표에 홀드아웃 행이 들어왔습니다.")
    candidates.attrs["stock_training_filter"] = {
        "holdout_start": holdout_start,
        "top_n": top_n,
        "horizon": horizon,
        "neutral_band": neutral_band,
        "source_rows": int(len(daily_prices)),
        "kospi_rows": int(len(kospi)),
        "invalid_market_cap_rows": int((~valid_market_cap).sum()),
        "common_stock_rows": int(len(common)),
        "selected_rows": int(selected_rows),
        "missing_entry_or_exit_rows": int(selected_rows - len(candidates)),
        "training_rows": int(len(candidates)),
        "first_date": str(candidates["bas_dd"].min()),
        "last_date": str(candidates["bas_dd"].max()),
    }
    return candidates


def _build_one_stock_features(group: pd.DataFrame) -> pd.DataFrame:
    """한 종목의 전체 시계열에서 가격 수준에 무관한 피처를 계산한다."""

    ordered = group.sort_values("bas_dd", kind="stable").copy()
    close = ordered["adj_close"].to_numpy(dtype=float)
    high = ordered["adj_high"].to_numpy(dtype=float)
    low = ordered["adj_low"].to_numpy(dtype=float)
    volume = ordered["volume"].to_numpy(dtype=float)

    atr_14 = atr(high, low, close, 14)
    bands = bollinger_bands(close, 20)
    hv_20 = historical_volatility(close, 20)
    obv_values = obv(close, volume)
    volume_average = volume_sma(volume, 20)

    with np.errstate(divide="ignore", invalid="ignore"):
        ordered["sma_gap_5_20"] = sma_gap(close, 5, 20)
        ordered["sma_gap_20_60"] = sma_gap(close, 20, 60)
        ordered["rsi_14"] = rsi(close, 14)
        ordered["macd_hist_ratio"] = macd_hist_ratio(close)
        ordered["bb_bandwidth"] = bands["bandwidth"]
        ordered["bb_position"] = percent_b(close, 20)
        ordered["atr_ratio"] = atr_14 / close
        ordered["hv_20"] = hv_20
        ordered["vol_ratio_20"] = volume_ratio(volume, 20)
        ordered["obv_slope_20"] = (
            obv_values - pd.Series(obv_values).shift(20).to_numpy()
        ) / (volume_average * 20.0)
        ordered["daily_return"] = n_day_return(close, 1)
        ordered["five_day_return"] = n_day_return(close, 5)
    return ordered.loc[:, ["bas_dd", "code", *STOCK_FEATURE_COLUMNS]].replace(
        [np.inf, -np.inf], np.nan
    )


def build_sector_stock_model_dataset(
    daily_prices: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    holdout_start: str = HOLDOUT_START,
    horizon: int = STOCK_LABEL_HORIZON,
    neutral_band: float = STOCK_NEUTRAL_BAND,
) -> StockModelDataset:
    """업종 후보에 종목별 기술적 피처와 T+1→T+6 라벨을 붙인다.

    피처는 후보 행만 이어 붙여 계산하지 않는다. 한 번이라도 후보가 된 종목의 전체
    개발 시계열에서 먼저 계산한 뒤 ``(bas_dd, code)``로 후보와 조인한다. 따라서
    후보에서 빠졌다가 다시 들어온 기간도 20·60일 창의 실제 과거로 남는다.
    """

    _validate_arguments(
        holdout_start=holdout_start,
        top_n=1,
        horizon=horizon,
        neutral_band=neutral_band,
    )
    missing_prices = PANEL_PRICE_COLUMNS - set(daily_prices.columns)
    missing_candidates = {"bas_dd", "code"} - set(candidates.columns)
    if missing_prices:
        raise ValueError(f"종목 패널 가격 열이 없습니다: {sorted(missing_prices)}")
    if missing_candidates:
        raise ValueError(f"종목 후보 키가 없습니다: {sorted(missing_candidates)}")
    if candidates.empty:
        raise ValueError("종목 후보가 비어 있습니다.")

    source = daily_prices.loc[:, sorted(PANEL_PRICE_COLUMNS)].copy()
    source["bas_dd"] = _normalize_dates(source)
    source["code"] = source["code"].astype("string").str.strip().str.zfill(6)
    if (source["bas_dd"] >= holdout_start).any():
        first = str(source.loc[source["bas_dd"] >= holdout_start, "bas_dd"].min())
        raise RuntimeError(f"개별 종목 원천에 홀드아웃 행이 들어 있습니다: {first}")
    source = source.loc[source["market"].eq("KOSPI")].copy()
    if source.duplicated(["bas_dd", "code"]).any():
        raise ValueError("KOSPI에 같은 날짜·종목코드가 두 번 이상 있습니다.")
    for column in ("adj_open", "adj_high", "adj_low", "adj_close", "volume"):
        source[column] = pd.to_numeric(source[column], errors="coerce")

    selected = candidates.copy()
    selected["bas_dd"] = _normalize_dates(selected)
    selected["code"] = selected["code"].astype("string").str.strip().str.zfill(6)
    if (selected["bas_dd"] >= holdout_start).any():
        raise RuntimeError("종목 후보에 홀드아웃 행이 들어 있습니다.")
    if selected.duplicated(["bas_dd", "code"]).any():
        raise ValueError("종목 후보에 같은 날짜·코드가 두 번 이상 있습니다.")

    # 후보 종목의 모든 과거 행만 남긴다. 후보 날짜만 남기면 이동창이 편입 때마다
    # 끊기지만, 종목 단위로 줄이는 것은 해당 종목의 과거를 훼손하지 않는다.
    selected_codes = selected["code"].unique()
    feature_source = source.loc[source["code"].isin(selected_codes)].copy()
    feature_source = feature_source.sort_values(["code", "bas_dd"], kind="stable")
    feature_parts = [
        _build_one_stock_features(group)
        for _, group in feature_source.groupby("code", sort=False, observed=True)
    ]
    features = pd.concat(feature_parts, ignore_index=True)
    selected = selected.merge(
        features,
        on=["bas_dd", "code"],
        how="left",
        validate="one_to_one",
    )

    calendar = sorted(source["bas_dd"].unique().tolist())
    calendar_frame = pd.DataFrame({"bas_dd": calendar})
    calendar_frame["entry_bas_dd"] = calendar_frame["bas_dd"].shift(-1)
    calendar_frame["exit_bas_dd"] = calendar_frame["bas_dd"].shift(-(horizon + 1))
    selected = selected.merge(
        calendar_frame,
        on="bas_dd",
        how="left",
        validate="many_to_one",
    )
    selected_before_prices = len(selected)
    selected = selected.loc[selected["exit_bas_dd"].notna()].copy()

    valid_open = source["adj_open"].notna() & source["adj_open"].gt(0.0)
    open_prices = source.loc[valid_open, ["bas_dd", "code", "adj_open"]]
    selected = selected.merge(
        open_prices.rename(
            columns={"bas_dd": "entry_bas_dd", "adj_open": "entry_adj_open"}
        ),
        on=["entry_bas_dd", "code"],
        how="left",
        validate="one_to_one",
    )
    selected = selected.merge(
        open_prices.rename(
            columns={"bas_dd": "exit_bas_dd", "adj_open": "exit_adj_open"}
        ),
        on=["exit_bas_dd", "code"],
        how="left",
        validate="one_to_one",
    )
    valid_prices = selected["entry_adj_open"].notna() & selected["exit_adj_open"].notna()
    finite_features = np.isfinite(
        selected.loc[:, STOCK_FEATURE_COLUMNS].to_numpy(dtype=float)
    ).all(axis=1)
    selected = selected.loc[valid_prices & finite_features].copy()
    if selected.empty:
        raise ValueError("피처와 정확한 T+1·T+6 수정시가가 있는 후보가 없습니다.")

    selected["fwd_return_5d"] = (
        selected["exit_adj_open"] / selected["entry_adj_open"] - 1.0
    )
    up = selected["exit_adj_open"] > selected["entry_adj_open"] * (1.0 + neutral_band)
    down = selected["exit_adj_open"] < selected["entry_adj_open"] * (1.0 - neutral_band)
    selected["label"] = np.select([up, down], ["상승", "하락"], default="중립")
    selected["label_numeric"] = selected["label"].map(LABEL_TO_NUMBER).astype("int8")

    sort_columns = ["bas_dd"]
    if "candidate_rank" in selected.columns:
        sort_columns.append("candidate_rank")
    sort_columns.append("code")
    selected = selected.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    if selected["bas_dd"].max() >= holdout_start:
        raise RuntimeError("개별 종목 모델 입력에 홀드아웃 행이 들어왔습니다.")
    selected.attrs["stock_panel"] = {
        "holdout_start": holdout_start,
        "horizon": horizon,
        "neutral_band": neutral_band,
        "candidate_rows": int(selected_before_prices),
        "model_rows": int(len(selected)),
        "dates": int(selected["bas_dd"].nunique()),
        "stocks": int(selected["code"].nunique()),
        "first_date": str(selected["bas_dd"].min()),
        "last_date": str(selected["bas_dd"].max()),
    }
    return StockModelDataset(frame=selected)


__all__ = [
    "STOCK_FEATURE_COLUMNS",
    "StockModelDataset",
    "build_sector_stock_model_dataset",
    "build_stock_training_frame",
]
