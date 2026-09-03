"""HF 전 종목 수정주가를 KOSPI 시장 내부 상태 피처로 집계한다."""

from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.horizon import HOLDOUT_START
from features.model_dataset import (
    COMBINATION_FEATURES,
    LABEL_HORIZON,
    ModelDataset,
    build_kospi200_feature_frame,
    build_model_dataset,
)

MARKET_FEATURE_COLUMNS = (
    "ad_percent",
    "above_ma200_ratio",
    "log_trin",
    "return_dispersion",
    "unchanged_ratio",
)

COMBINED_MARKET_FEATURE_COLUMNS = (
    *COMBINATION_FEATURES["F"],
    "five_day_return",
    *MARKET_FEATURE_COLUMNS,
)

DAILY_REQUIRED_COLUMNS = {
    "bas_dd",
    "code",
    "market",
    "volume",
    "adj_close",
}


def _normalize_dates(frame: pd.DataFrame) -> pd.Series:
    dates = (
        frame["bas_dd"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(8)
    )
    if dates.isna().any() or (~dates.str.fullmatch(r"\d{8}")).any():
        raise ValueError("전 종목 bas_dd에 YYYYMMDD가 아닌 값이 있습니다.")
    return dates


def build_kospi_market_features(
    daily_prices: pd.DataFrame,
    *,
    holdout_start: str = HOLDOUT_START,
) -> pd.DataFrame:
    """각 거래일 KOSPI 전 종목을 다섯 개 시장 피처로 집계한다.

    종목별 수익률과 이동평균은 액면분할을 반영한 ``adj_close``로만 계산한다.
    그날 실제로 존재한 KOSPI 종목 전체를 집계하므로 미래 시점의 구성 종목을
    과거에 소급하지 않는다.
    """

    missing = DAILY_REQUIRED_COLUMNS - set(daily_prices.columns)
    if missing:
        raise ValueError(f"전 종목 Parquet 필수 열이 없습니다: {sorted(missing)}")

    source = daily_prices.loc[:, sorted(DAILY_REQUIRED_COLUMNS)].copy()
    source["bas_dd"] = _normalize_dates(source)
    source = source.loc[
        (source["bas_dd"] < holdout_start) & (source["market"] == "KOSPI")
    ].copy()
    if source.empty:
        raise ValueError(f"{holdout_start} 이전 KOSPI 종목 데이터가 없습니다.")

    source["code"] = source["code"].astype("string").str.zfill(6)
    for column in ("volume", "adj_close"):
        source[column] = pd.to_numeric(source[column], errors="raise").astype(float)
    if (source["adj_close"] <= 0.0).any():
        raise ValueError("KOSPI 종목 adj_close에 0 이하 값이 있습니다.")
    if source.duplicated(["code", "bas_dd"]).any():
        raise ValueError("같은 KOSPI 종목·거래일 행이 두 번 이상 들어 있습니다.")

    source = source.sort_values(["code", "bas_dd"], kind="stable").reset_index(drop=True)
    grouped_close = source.groupby("code", sort=False)["adj_close"]
    previous_close = grouped_close.shift(1)
    source["return_1d"] = source["adj_close"] / previous_close - 1.0
    source["ma200"] = grouped_close.transform(
        lambda values: values.rolling(200, min_periods=200).mean()
    )

    valid_return = source["return_1d"].notna()
    source["advance"] = valid_return & (source["return_1d"] > 0.0)
    source["decline"] = valid_return & (source["return_1d"] < 0.0)
    source["unchanged"] = valid_return & (source["return_1d"] == 0.0)
    source["advance_volume"] = source["volume"].where(source["advance"], 0.0)
    source["decline_volume"] = source["volume"].where(source["decline"], 0.0)
    source["above_ma200"] = (source["adj_close"] > source["ma200"]).where(
        source["ma200"].notna()
    )

    daily = source.groupby("bas_dd", sort=True).agg(
        advance_count=("advance", "sum"),
        decline_count=("decline", "sum"),
        unchanged_count=("unchanged", "sum"),
        return_count=("return_1d", "count"),
        advance_volume=("advance_volume", "sum"),
        decline_volume=("decline_volume", "sum"),
        above_ma200_count=("above_ma200", "sum"),
        ma200_count=("above_ma200", "count"),
        return_dispersion=("return_1d", "std"),
    )
    # nullable bool 합계는 pandas에서 object가 될 수 있다. 0으로 나누기를
    # NaN으로 남기려면 벡터 연산 전에 명시적으로 실수형으로 맞춘다.
    daily = daily.astype(float)

    directional_count = daily["advance_count"] + daily["decline_count"]
    with np.errstate(divide="ignore", invalid="ignore"):
        daily["ad_percent"] = (
            (daily["advance_count"] - daily["decline_count"]) / directional_count
        )
        daily["above_ma200_ratio"] = daily["above_ma200_count"] / daily["ma200_count"]
        daily["unchanged_ratio"] = daily["unchanged_count"] / daily["return_count"]
        daily["log_trin"] = np.log(
            (daily["advance_count"] / daily["decline_count"])
            / (daily["advance_volume"] / daily["decline_volume"])
        )

    result = daily.loc[:, MARKET_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    return result.reset_index()


def build_market_feature_dataset(
    index_prices: pd.DataFrame,
    daily_prices: pd.DataFrame,
    *,
    holdout_start: str = HOLDOUT_START,
) -> ModelDataset:
    """KOSPI200 라벨에 같은 날까지 알려진 시장 내부 피처를 결합한다."""

    raw = build_kospi200_feature_frame(index_prices, holdout_start=holdout_start)
    market = build_kospi_market_features(daily_prices, holdout_start=holdout_start)
    merged = raw.merge(market, on="bas_dd", how="left", validate="one_to_one")
    finite = np.isfinite(merged.loc[:, MARKET_FEATURE_COLUMNS].to_numpy(dtype=float)).all(
        axis=1
    )
    usable = finite & merged["label_numeric"].notna().to_numpy()
    model_frame = merged.loc[usable].copy().reset_index(drop=True)
    if model_frame.empty:
        raise ValueError("시장 내부 피처로 학습 가능한 행이 없습니다.")
    if model_frame["bas_dd"].max() >= holdout_start:
        raise RuntimeError("시장 내부 모델 입력에 홀드아웃 행이 들어왔습니다.")
    if model_frame["raw_position"].iloc[-1] + LABEL_HORIZON + 1 >= len(raw):
        raise RuntimeError("마지막 모델 행의 5거래일 청산 시가가 개발구간 밖에 있습니다.")

    return ModelDataset(
        frame=model_frame,
        raw_prices=raw,
        feature_columns=MARKET_FEATURE_COLUMNS,
        combination="G_MARKET_INTERNALS",
    )


def build_combined_market_dataset(
    index_prices: pd.DataFrame,
    daily_prices: pd.DataFrame,
    *,
    holdout_start: str = HOLDOUT_START,
) -> ModelDataset:
    """기존 F+5일 수익률에 시장 내부 피처를 더하되 OOS 날짜는 보존한다."""

    base = build_model_dataset(
        index_prices,
        "F",
        return_features=("five_day_return",),
        holdout_start=holdout_start,
    )
    market = build_kospi_market_features(daily_prices, holdout_start=holdout_start)
    merged = base.frame.merge(market, on="bas_dd", how="left", validate="one_to_one")
    finite = np.isfinite(
        merged.loc[:, COMBINED_MARKET_FEATURE_COLUMNS].to_numpy(dtype=float)
    ).all(axis=1)
    if not finite.all():
        failed_dates = merged.loc[~finite, "bas_dd"].head().tolist()
        raise ValueError(f"기존 F+5Day 날짜에 시장 내부 피처가 없습니다: {failed_dates}")

    return ModelDataset(
        frame=merged,
        raw_prices=base.raw_prices,
        feature_columns=COMBINED_MARKET_FEATURE_COLUMNS,
        combination="G_FIVE_DAY_PLUS_MARKET_INTERNALS",
    )
