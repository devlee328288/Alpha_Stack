"""개별 종목 랭킹 모델에 넣을 시점 정합 학습 표를 만든다.

업종 매핑이 준비되기 전 MVP는 매 거래일 KOSPI 보통주 중 시가총액 상위 50개를
후보로 삼는다. 이 모듈은 종목 피처를 만들기 전 단계인 후보 선정과 5거래일 라벨만
담당한다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype

from evaluation.horizon import HOLDOUT_START
from features.indicators import bollinger_bands, macd_hist_ratio, percent_b, rsi, sma_gap
from features.model_dataset import KOSPI200_NAME
from features.returns import n_day_return
from features.volatility import atr_ratio, historical_volatility
from features.volume import obv_slope_20, volume_ratio
from supply.sector import index_name_for

DEFAULT_TOP_N = 50
STOCK_LABEL_HORIZON = 5
STOCK_NEUTRAL_BAND = 0.02
LABEL_TO_NUMBER = {"하락": -1, "중립": 0, "상승": 1}

STOCK_COMBINATION_FEATURES = {
    "A": (
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
    ),
    "B": (
        "ret_5",
        "ret_20",
        "sma_gap_5_20",
        "sma_gap_20_60",
        "rsi_14",
        "dist_high_20",
        "dist_high_60",
    ),
    "C": (
        "ret_5",
        "overnight_gap",
        "intraday_return",
        "close_location",
        "bb_position",
        "rsi_14",
        "volume_z_20",
    ),
    "D": (
        "atr_ratio",
        "hv_20",
        "range_1",
        "range_20",
        "bb_bandwidth",
        "volume_z_20",
        "turnover_20",
        "log_amihud_20",
    ),
    "E": (
        "sector_ret_5",
        "sector_ret_20",
        "relative_ret_5_sector",
        "relative_ret_20_sector",
        "relative_ret_5_market",
        "sector_hv_20",
        "sector_beta_60",
    ),
    "F": (
        "ret_5_rank",
        "sector_relative_rank",
        "turnover_rank",
        "hv_20_rank",
        "market_cap_percentile",
        "sector_market_cap_rank",
        "industry_stock_rank",
        "volume_z_20",
        "bb_position",
    ),
    "G": (
        "dist_high_60",
        "sma_gap_20_60",
        "relative_ret_5_market",
        "rsi_14",
        "hv_20",
        "turnover_20",
    ),
}

# 기존 호출은 조합 A를 뜻한다. 전체 조합의 합집합은 공통 패널 캐시 검증에 사용한다.
STOCK_FEATURE_COLUMNS = STOCK_COMBINATION_FEATURES["A"]
ALL_STOCK_FEATURE_COLUMNS = tuple(
    dict.fromkeys(
        feature
        for combination in STOCK_COMBINATION_FEATURES.values()
        for feature in combination
    )
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

OPTIONAL_PANEL_PRICE_COLUMNS = {"value", "market_cap", "industry"}


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
    """한 종목의 전체 시계열에서 가격·거래량·유동성 피처를 계산한다."""

    ordered = group.sort_values("bas_dd", kind="stable").reset_index(drop=True).copy()
    close = ordered["adj_close"].to_numpy(dtype=float)
    open_ = ordered["adj_open"].to_numpy(dtype=float)
    high = ordered["adj_high"].to_numpy(dtype=float)
    low = ordered["adj_low"].to_numpy(dtype=float)
    volume = ordered["volume"].to_numpy(dtype=float)
    value = pd.to_numeric(
        ordered.get("value", pd.Series(np.nan, index=ordered.index)), errors="coerce"
    ).reset_index(drop=True)
    market_cap = pd.to_numeric(
        ordered.get("market_cap", pd.Series(np.nan, index=ordered.index)), errors="coerce"
    ).reset_index(drop=True)

    bands = bollinger_bands(close, 20)

    # 파생 지표는 여기서 손으로 다시 짜지 않고 원자 함수를 그대로 부른다. 같은 공식을
    # 두 곳에 두면 한쪽만 고쳐졌을 때 값이 조용히 갈린다(신장환 님 지적, #155).
    with np.errstate(divide="ignore", invalid="ignore"):
        ordered["sma_gap_5_20"] = sma_gap(close, 5, 20)
        ordered["sma_gap_20_60"] = sma_gap(close, 20, 60)
        ordered["rsi_14"] = rsi(close, 14)
        ordered["macd_hist_ratio"] = macd_hist_ratio(close)
        ordered["bb_bandwidth"] = bands["bandwidth"]
        ordered["bb_position"] = percent_b(close, 20)
        ordered["atr_ratio"] = atr_ratio(high, low, close, 14)
        ordered["hv_20"] = historical_volatility(close, 20)
        ordered["vol_ratio_20"] = volume_ratio(volume, 20)
        ordered["obv_slope_20"] = obv_slope_20(close, volume, 20)
        ordered["daily_return"] = n_day_return(close, 1)
        ordered["five_day_return"] = n_day_return(close, 5)
        ordered["ret_1"] = ordered["daily_return"]
        ordered["ret_5"] = ordered["five_day_return"]
        ordered["ret_20"] = n_day_return(close, 20)
        ordered["dist_high_20"] = close / pd.Series(close).rolling(20).max() - 1.0
        ordered["dist_high_60"] = close / pd.Series(close).rolling(60).max() - 1.0
        previous_close = pd.Series(close).shift(1).to_numpy()
        ordered["overnight_gap"] = open_ / previous_close - 1.0
        ordered["intraday_return"] = close / open_ - 1.0
        price_range = high - low
        ordered["close_location"] = np.where(
            price_range > 0.0,
            (close - low) / price_range,
            np.nan,
        )
        ordered["range_1"] = price_range / close
        ordered["range_20"] = pd.Series(ordered["range_1"]).rolling(20).mean()

        log_volume = np.log1p(np.where(volume >= 0.0, volume, np.nan))
        log_volume_series = pd.Series(log_volume)
        volume_mean = log_volume_series.rolling(20).mean()
        volume_std = log_volume_series.rolling(20).std(ddof=0)
        ordered["volume_z_20"] = (log_volume_series - volume_mean) / volume_std

        turnover = value / market_cap
        turnover = turnover.where((value > 0.0) & (market_cap > 0.0))
        ordered["turnover_20"] = turnover.rolling(20).mean()
        amihud = pd.Series(np.abs(ordered["ret_1"])) / value
        amihud = amihud.where(value > 0.0)
        ordered["log_amihud_20"] = np.log(amihud.rolling(20).mean() + 1e-18)

    # ret_1은 조합 C의 입력에서는 제외하지만 업종 베타와 Amihud 계산에는 필요하다.
    # 모델 피처 목록과 파생 계산용 내부 열을 같은 목록으로 취급하면, 조합을 바꿀 때 후속
    # 파생피처가 조용히 깨질 수 있으므로 내부 열로 따로 유지한다.
    output_columns = list(
        dict.fromkeys(
            [
                "bas_dd",
                "code",
                "ret_1",
                *[
                    feature
                    for feature in ALL_STOCK_FEATURE_COLUMNS
                    if feature in ordered.columns
                ],
            ]
        )
    )
    return ordered.loc[:, output_columns].replace(
        [np.inf, -np.inf], np.nan
    )


def _attach_index_relative_features(
    features: pd.DataFrame,
    source: pd.DataFrame,
    index_prices: pd.DataFrame,
) -> pd.DataFrame:
    """종목의 그날 업종과 KOSPI200 과거 수익률만 이용해 상대 피처를 붙인다."""

    required = {"bas_dd", "index_name", "index_class", "close"}
    missing = required - set(index_prices.columns)
    if missing:
        raise ValueError(f"업종 상대강도 지수 열이 없습니다: {sorted(missing)}")
    if "industry" not in source.columns:
        raise ValueError("업종 상대강도 계산에 종목별 industry 열이 필요합니다.")

    indices = index_prices.loc[:, sorted(required)].copy()
    indices["bas_dd"] = _normalize_dates(indices)
    if (indices["bas_dd"] >= HOLDOUT_START).any():
        raise RuntimeError("업종 상대강도 원천에 홀드아웃 행이 들어 있습니다.")
    indices = indices.loc[indices["index_class"].eq("KOSPI")].copy()
    indices["close"] = pd.to_numeric(indices["close"], errors="coerce")
    if indices.duplicated(["bas_dd", "index_name"]).any():
        raise ValueError("업종 상대강도 원천에 같은 날짜·지수가 두 번 이상 있습니다.")
    indices = indices.sort_values(["index_name", "bas_dd"], kind="stable")
    grouped_close = indices.groupby("index_name", sort=False)["close"]
    indices["sector_ret_1"] = grouped_close.pct_change(fill_method=None)
    indices["sector_ret_5"] = grouped_close.pct_change(5, fill_method=None)
    indices["sector_ret_20"] = grouped_close.pct_change(20, fill_method=None)
    log_return = grouped_close.transform(lambda values: np.log(values).diff())
    indices["sector_hv_20"] = log_return.groupby(
        indices["index_name"], sort=False
    ).transform(lambda values: values.rolling(20).std(ddof=1) * np.sqrt(252.0))

    identity = source.loc[:, ["bas_dd", "code", "industry"]].copy()
    identity["industry_index_name"] = identity["industry"].astype("string").map(
        index_name_for
    )
    out = features.merge(identity, on=["bas_dd", "code"], how="left", validate="one_to_one")
    sector_columns = [
        "bas_dd",
        "index_name",
        "sector_ret_1",
        "sector_ret_5",
        "sector_ret_20",
        "sector_hv_20",
    ]
    out = out.merge(
        indices.loc[:, sector_columns].rename(columns={"index_name": "industry_index_name"}),
        on=["bas_dd", "industry_index_name"],
        how="left",
        validate="many_to_one",
    )
    market = indices.loc[
        indices["index_name"].eq(KOSPI200_NAME),
        ["bas_dd", "sector_ret_5"],
    ].rename(columns={"sector_ret_5": "market_ret_5"})
    if market.empty:
        raise ValueError(f"업종 상대강도 원천에 {KOSPI200_NAME!r} 지수가 없습니다.")
    out = out.merge(market, on="bas_dd", how="left", validate="many_to_one")
    out["relative_ret_5_sector"] = out["ret_5"] - out["sector_ret_5"]
    out["relative_ret_20_sector"] = out["ret_20"] - out["sector_ret_20"]
    out["relative_ret_5_market"] = out["ret_5"] - out["market_ret_5"]

    def _rolling_sector_beta(group: pd.DataFrame) -> pd.Series:
        stock_return = group["ret_1"]
        sector_return = group["sector_ret_1"]
        covariance = stock_return.rolling(60).cov(sector_return)
        variance = sector_return.rolling(60).var()
        return covariance / variance

    out = out.sort_values(["code", "bas_dd"], kind="stable")
    out["sector_beta_60"] = (
        out.groupby("code", sort=False, group_keys=False)
        .apply(_rolling_sector_beta, include_groups=False)
        .reset_index(level=0, drop=True)
        .reindex(out.index)
    )
    return out.drop(columns=["industry", "industry_index_name", "market_ret_5", "sector_ret_1"])


def build_sector_stock_model_dataset(
    daily_prices: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    index_prices: pd.DataFrame | None = None,
    feature_columns: tuple[str, ...] = STOCK_FEATURE_COLUMNS,
    drop_incomplete_features: bool = True,
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
    unknown_features = set(feature_columns) - set(ALL_STOCK_FEATURE_COLUMNS)
    if unknown_features:
        raise ValueError(f"아직 계산하지 않는 개별종목 피처입니다: {sorted(unknown_features)}")
    if not feature_columns or len(set(feature_columns)) != len(feature_columns):
        raise ValueError("개별종목 피처 조합은 비어 있거나 중복될 수 없습니다.")
    needs_index = bool(set(feature_columns) & set(STOCK_COMBINATION_FEATURES["E"])) or bool(
        set(feature_columns) & {"sector_relative_rank"}
    )
    if needs_index and index_prices is None:
        raise ValueError("조합 E·F의 업종 상대강도 피처에는 index_prices가 필요합니다.")

    missing_prices = PANEL_PRICE_COLUMNS - set(daily_prices.columns)
    missing_candidates = {"bas_dd", "code"} - set(candidates.columns)
    if missing_prices:
        raise ValueError(f"종목 패널 가격 열이 없습니다: {sorted(missing_prices)}")
    if missing_candidates:
        raise ValueError(f"종목 후보 키가 없습니다: {sorted(missing_candidates)}")
    if candidates.empty:
        raise ValueError("종목 후보가 비어 있습니다.")

    source_columns = PANEL_PRICE_COLUMNS | (
        OPTIONAL_PANEL_PRICE_COLUMNS & set(daily_prices.columns)
    )
    source = daily_prices.loc[:, sorted(source_columns)].copy()
    source["bas_dd"] = _normalize_dates(source)
    source["code"] = source["code"].astype("string").str.strip().str.zfill(6)
    if (source["bas_dd"] >= holdout_start).any():
        first = str(source.loc[source["bas_dd"] >= holdout_start, "bas_dd"].min())
        raise RuntimeError(f"개별 종목 원천에 홀드아웃 행이 들어 있습니다: {first}")
    source = source.loc[source["market"].eq("KOSPI")].copy()
    if source.duplicated(["bas_dd", "code"]).any():
        raise ValueError("KOSPI에 같은 날짜·종목코드가 두 번 이상 있습니다.")
    for column in (
        "adj_open",
        "adj_high",
        "adj_low",
        "adj_close",
        "volume",
        "value",
        "market_cap",
    ):
        if column not in source.columns:
            continue
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
    if index_prices is not None:
        features = _attach_index_relative_features(features, feature_source, index_prices)
    selected = selected.merge(
        features,
        on=["bas_dd", "code"],
        how="left",
        validate="one_to_one",
    )

    # 횡단면 순위는 그날 확정된 최대 50개 후보 안에서만 계산한다. 미래 날짜나
    # 후보 밖 종목을 사용하지 않으며, 값이 클수록 1에 가까운 백분위로 통일한다.
    if "ret_5" in selected.columns:
        selected["ret_5_rank"] = selected.groupby("bas_dd", sort=False)["ret_5"].rank(
            pct=True, method="average"
        )
    if "relative_ret_5_sector" in selected.columns:
        selected["sector_relative_rank"] = selected.groupby("bas_dd", sort=False)[
            "relative_ret_5_sector"
        ].rank(pct=True, method="average")
    if "turnover_20" in selected.columns:
        selected["turnover_rank"] = selected.groupby("bas_dd", sort=False)[
            "turnover_20"
        ].rank(pct=True, method="average")
    if "hv_20" in selected.columns:
        selected["hv_20_rank"] = selected.groupby("bas_dd", sort=False)["hv_20"].rank(
            pct=True, method="average"
        )
    if "market_cap" in selected.columns:
        market_cap = pd.to_numeric(selected["market_cap"], errors="coerce")
        selected["market_cap_percentile"] = market_cap.groupby(selected["bas_dd"]).rank(
            pct=True, method="average"
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
    missing_calculated = set(feature_columns) - set(selected.columns)
    if missing_calculated:
        raise ValueError(f"선택한 피처를 계산하지 못했습니다: {sorted(missing_calculated)}")
    finite_features = np.isfinite(
        selected.loc[:, feature_columns].to_numpy(dtype=float)
    ).all(axis=1)
    keep = valid_prices & finite_features if drop_incomplete_features else valid_prices
    selected = selected.loc[keep].copy()
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
        "feature_columns": list(feature_columns),
    }
    return StockModelDataset(frame=selected, feature_columns=tuple(feature_columns))


def select_stock_feature_dataset(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> StockModelDataset:
    """공통 후보 패널에서 한 조합의 유효 피처 행만 선택한다."""

    unknown = set(feature_columns) - set(ALL_STOCK_FEATURE_COLUMNS)
    missing = set(feature_columns) - set(frame.columns)
    if unknown:
        raise ValueError(f"아직 계산하지 않는 개별종목 피처입니다: {sorted(unknown)}")
    if missing:
        raise ValueError(f"공통 종목 패널에 선택 피처가 없습니다: {sorted(missing)}")
    if not feature_columns or len(set(feature_columns)) != len(feature_columns):
        raise ValueError("개별종목 피처 조합은 비어 있거나 중복될 수 없습니다.")
    finite = np.isfinite(frame.loc[:, feature_columns].to_numpy(dtype=float)).all(axis=1)
    selected = frame.loc[finite].copy().reset_index(drop=True)
    if selected.empty:
        raise ValueError("선택한 피처 조합에 유효한 종목 행이 없습니다.")
    selected.attrs.update(frame.attrs)
    panel_rule = dict(selected.attrs.get("stock_panel", {}))
    panel_rule.update(
        {
            "model_rows": int(len(selected)),
            "dates": int(selected["bas_dd"].nunique()),
            "stocks": int(selected["code"].nunique()),
            "first_date": str(selected["bas_dd"].min()),
            "last_date": str(selected["bas_dd"].max()),
            "feature_columns": list(feature_columns),
        }
    )
    selected.attrs["stock_panel"] = panel_rule
    return StockModelDataset(frame=selected, feature_columns=tuple(feature_columns))


def align_stock_feature_datasets(
    datasets: Mapping[str, StockModelDataset],
) -> dict[str, StockModelDataset]:
    """조합별 패널을 공통 ``(bas_dd, code)`` 행으로 맞춘다.

    공통 날짜만 남기면 같은 날짜 안에서도 조합별 워밍업 결측 때문에 종목 구성이 달라질 수
    있다. 날짜와 종목코드가 모두 같은 행만 남기고 모든 조합을 같은 순서로 정렬해, 표본
    차이가 피처 조합의 성능 차이로 섞이지 않게 한다.
    """

    if not datasets:
        raise ValueError("정렬할 개별종목 피처 조합이 없습니다.")

    normalized_frames: dict[str, pd.DataFrame] = {}
    common_keys: pd.MultiIndex | None = None
    for name, dataset in datasets.items():
        missing = {"bas_dd", "code", "label_numeric"} - set(dataset.frame.columns)
        if missing:
            raise ValueError(f"조합 {name}에 공통 행 정렬 열이 없습니다: {sorted(missing)}")

        frame = dataset.frame.copy()
        frame["bas_dd"] = _normalize_dates(frame)
        frame["code"] = frame["code"].astype("string").str.strip().str.zfill(6).astype(str)
        if frame.duplicated(["bas_dd", "code"]).any():
            raise ValueError(f"조합 {name}에 같은 날짜·종목 행이 두 번 이상 있습니다.")

        keys = pd.MultiIndex.from_frame(frame.loc[:, ["bas_dd", "code"]])
        common_keys = keys if common_keys is None else common_keys.intersection(keys, sort=False)
        normalized_frames[name] = frame

    if common_keys is None or common_keys.empty:
        raise ValueError("피처 조합 사이에 공통 날짜·종목 행이 없습니다.")

    canonical = common_keys.to_frame(index=False).sort_values(
        ["bas_dd", "code"], kind="stable"
    )
    canonical = canonical.reset_index(drop=True)
    aligned: dict[str, StockModelDataset] = {}
    reference_labels: np.ndarray | None = None
    for name, dataset in datasets.items():
        frame = canonical.merge(
            normalized_frames[name],
            on=["bas_dd", "code"],
            how="left",
            validate="one_to_one",
            sort=False,
        )
        labels = frame["label_numeric"].to_numpy(dtype=int)
        if reference_labels is None:
            reference_labels = labels
        elif not np.array_equal(labels, reference_labels):
            raise ValueError(f"조합 {name}의 공통 날짜·종목 라벨이 다른 조합과 다릅니다.")

        frame.attrs.update(dataset.frame.attrs)
        panel_rule = dict(frame.attrs.get("stock_panel", {}))
        panel_rule.update(
            {
                "model_rows": int(len(frame)),
                "dates": int(frame["bas_dd"].nunique()),
                "stocks": int(frame["code"].nunique()),
                "first_date": str(frame["bas_dd"].min()),
                "last_date": str(frame["bas_dd"].max()),
                "feature_columns": list(dataset.feature_columns),
                "alignment_keys": ["bas_dd", "code"],
            }
        )
        frame.attrs["stock_panel"] = panel_rule
        aligned[name] = StockModelDataset(
            frame=frame,
            feature_columns=dataset.feature_columns,
        )

    return aligned


__all__ = [
    "ALL_STOCK_FEATURE_COLUMNS",
    "STOCK_COMBINATION_FEATURES",
    "STOCK_FEATURE_COLUMNS",
    "StockModelDataset",
    "align_stock_feature_datasets",
    "build_sector_stock_model_dataset",
    "build_stock_training_frame",
    "select_stock_feature_dataset",
]
