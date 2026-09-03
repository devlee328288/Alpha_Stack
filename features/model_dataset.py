"""HF 지수 원시 Parquet에서 A~F 모델 입력을 재현한다."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from evaluation.horizon import HOLDOUT_START, NEUTRAL_BAND
from features.indicators import bollinger_bands, ema, macd, rsi, sma
from features.returns import n_day_return
from features.volatility import atr, historical_volatility, parkinson_volatility, true_range
from features.volume import obv, volume_ratio, volume_roc, volume_sma, vwap

KOSPI200_NAME = "코스피 200"
LABEL_HORIZON = 5
LABEL_TO_NUMBER = {"하락": -1, "중립": 0, "상승": 1}

RAW_COLUMNS = {
    "bas_dd",
    "index_name",
    "index_class",
    "open",
    "high",
    "low",
    "close",
    "volume",
}

BASE_FEATURE_COLUMNS = (
    "sma_5",
    "sma_20",
    "sma_60",
    "ema_12",
    "ema_26",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_mid",
    "bb_upper",
    "bb_lower",
    "bb_bandwidth",
    "true_range",
    "atr_14",
    "hv_20",
    "parkinson_20",
    "vol_sma_20",
    "vol_ratio_20",
    "obv",
    "vwap_20",
    "vol_roc_5",
)

COMBINATION_FEATURES = {
    "A": ("rsi_14", "bb_bandwidth", "hv_20", "vol_ratio_20"),
    "B": ("sma_gap_5_20", "macd_hist_ratio", "rsi_14", "hv_20"),
    "C": (
        "sma_gap_5_20",
        "macd_hist_ratio",
        "rsi_14",
        "bb_position",
        "hv_20",
        "vol_ratio_20",
    ),
    "D": (
        "sma_gap_20_60",
        "rsi_14",
        "atr_ratio",
        "bb_bandwidth",
        "hv_regime",
        "obv_slope_20",
    ),
    "E": ("atr_ratio", "bb_bandwidth", "hv_regime"),
    "F": (
        "sma_gap_20_60",
        "sma_gap_5_20",
        "rsi_14",
        "macd_hist_atr",
        "atr_ratio",
        "bb_bandwidth",
        "hv_regime",
        "obv_slope_20",
    ),
}

RETURN_FEATURES = {"daily_return", "five_day_return"}


@dataclass(frozen=True)
class ModelDataset:
    """한 피처 조합의 학습 표와 백테스트용 원시 시가."""

    frame: pd.DataFrame
    raw_prices: pd.DataFrame
    feature_columns: tuple[str, ...]
    combination: str

    @property
    def x(self) -> pd.DataFrame:
        return self.frame.loc[:, self.feature_columns]

    @property
    def y(self) -> np.ndarray:
        return self.frame["label_numeric"].to_numpy(dtype=int)

    @property
    def signal_positions(self) -> np.ndarray:
        return self.frame["raw_position"].to_numpy(dtype=int)

    @property
    def opens(self) -> np.ndarray:
        return self.raw_prices["open"].to_numpy(dtype=float)


def _normalize_dates(frame: pd.DataFrame) -> pd.Series:
    dates = (
        frame["bas_dd"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(8)
    )
    if dates.isna().any() or (~dates.str.fullmatch(r"\d{8}")).any():
        raise ValueError("bas_dd에 YYYYMMDD로 해석할 수 없는 값이 있습니다.")
    return dates


def build_kospi200_feature_frame(
    index_prices: pd.DataFrame,
    *,
    holdout_start: str = HOLDOUT_START,
    horizon: int = LABEL_HORIZON,
    neutral_band: float = NEUTRAL_BAND,
) -> pd.DataFrame:
    """전체 지수 원시표에서 KOSPI200 피처와 5거래일 라벨을 계산한다."""

    missing = RAW_COLUMNS - set(index_prices.columns)
    if missing:
        raise ValueError(f"지수 Parquet 필수 열이 없습니다: {sorted(missing)}")

    source = index_prices.copy()
    source["bas_dd"] = _normalize_dates(source)
    source = source.loc[source["bas_dd"] < holdout_start].copy()
    if source.empty:
        raise ValueError(f"{holdout_start} 이전 지수 데이터가 없습니다.")

    calendar = sorted(source["bas_dd"].unique().tolist())
    target = source.loc[source["index_name"] == KOSPI200_NAME].copy()
    target = target.sort_values("bas_dd", kind="stable").reset_index(drop=True)
    if target.empty:
        raise ValueError(f"정확한 지수명 {KOSPI200_NAME!r}을 찾지 못했습니다.")
    if target["bas_dd"].duplicated().any():
        raise ValueError("KOSPI200에 같은 거래일이 두 번 이상 들어 있습니다.")
    if target["bas_dd"].tolist() != calendar:
        missing_dates = sorted(set(calendar) - set(target["bas_dd"]))
        raise ValueError(f"KOSPI200 거래일에 구멍이 있습니다: {missing_dates[:5]}")

    for column in ("open", "high", "low", "close", "volume"):
        target[column] = pd.to_numeric(target[column], errors="raise").astype(float)
    if (target[["open", "high", "low", "close"]] <= 0.0).any().any():
        raise ValueError("KOSPI200 시고저종가에 0 이하 값이 있습니다.")

    close = target["close"].to_numpy()
    high = target["high"].to_numpy()
    low = target["low"].to_numpy()
    volume = target["volume"].to_numpy()

    target["sma_5"] = sma(close, 5)
    target["sma_20"] = sma(close, 20)
    target["sma_60"] = sma(close, 60)
    target["ema_12"] = ema(close, 12)
    target["ema_26"] = ema(close, 26)
    target["rsi_14"] = rsi(close, 14)
    macd_values = macd(close)
    target["macd"] = macd_values["macd"]
    target["macd_signal"] = macd_values["signal"]
    target["macd_hist"] = macd_values["hist"]
    bands = bollinger_bands(close, 20)
    target["bb_mid"] = bands["mid"]
    target["bb_upper"] = bands["upper"]
    target["bb_lower"] = bands["lower"]
    target["bb_bandwidth"] = bands["bandwidth"]
    target["true_range"] = true_range(high, low, close)
    target["atr_14"] = atr(high, low, close, 14)
    target["hv_20"] = historical_volatility(close, 20)
    target["parkinson_20"] = parkinson_volatility(high, low, 20)
    target["vol_sma_20"] = volume_sma(volume, 20)
    target["vol_ratio_20"] = volume_ratio(volume, 20)
    target["obv"] = obv(close, volume)
    target["vwap_20"] = vwap(close, volume, 20)
    target["vol_roc_5"] = volume_roc(volume, 5)

    entry_open = target["open"].shift(-1)
    exit_open = target["open"].shift(-(horizon + 1))
    target["fwd_return_5d"] = exit_open / entry_open - 1.0
    target["label"] = np.select(
        [target["fwd_return_5d"] > neutral_band, target["fwd_return_5d"] < -neutral_band],
        ["상승", "하락"],
        default="중립",
    )
    target.loc[target["fwd_return_5d"].isna(), "label"] = pd.NA
    target["label_numeric"] = target["label"].map(LABEL_TO_NUMBER).astype("Int64")
    target["raw_position"] = np.arange(len(target), dtype=int)
    return target


def _add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    close = out["close"]
    bb_range = out["bb_upper"] - out["bb_lower"]

    with np.errstate(divide="ignore", invalid="ignore"):
        out["sma_gap_5_20"] = out["sma_5"] / out["sma_20"] - 1.0
        out["sma_gap_20_60"] = out["sma_20"] / out["sma_60"] - 1.0
        out["macd_hist_ratio"] = out["macd_hist"] / close
        out["macd_hist_atr"] = out["macd_hist"] / out["atr_14"]
        out["bb_position"] = (close - out["bb_lower"]) / bb_range
        out["atr_ratio"] = out["atr_14"] / close
        out["hv_regime"] = out["hv_20"] / out["hv_20"].rolling(
            250, min_periods=250
        ).mean()
        out["obv_slope_20"] = (
            (out["obv"] - out["obv"].shift(20)) / (out["vol_sma_20"] * 20.0)
        )
    out["daily_return"] = n_day_return(close, 1)
    out["five_day_return"] = n_day_return(close, 5)
    return out.replace([np.inf, -np.inf], np.nan)


def build_model_dataset(
    index_prices: pd.DataFrame,
    combination: str,
    *,
    return_features: Sequence[str] = (),
    holdout_start: str = HOLDOUT_START,
) -> ModelDataset:
    """A~F와 선택한 수익률 피처를 만들고, 쓸 수 있는 행만 남긴다."""

    key = combination.upper()
    if key not in COMBINATION_FEATURES:
        raise ValueError(f"알 수 없는 피처 조합입니다: {combination}")
    returns = tuple(return_features)
    unknown_returns = set(returns) - RETURN_FEATURES
    if unknown_returns:
        raise ValueError(f"알 수 없는 수익률 피처입니다: {sorted(unknown_returns)}")
    if len(set(returns)) != len(returns):
        raise ValueError("수익률 피처가 중복되었습니다.")

    raw = build_kospi200_feature_frame(index_prices, holdout_start=holdout_start)
    derived = _add_derived_features(raw)
    feature_columns = (*COMBINATION_FEATURES[key], *returns)
    finite = np.isfinite(derived.loc[:, feature_columns].to_numpy(dtype=float)).all(axis=1)
    usable = finite & derived["label_numeric"].notna().to_numpy()
    model_frame = derived.loc[usable].copy().reset_index(drop=True)
    if model_frame.empty:
        raise ValueError(f"조합 {key}에 학습 가능한 행이 없습니다.")
    if model_frame["bas_dd"].max() >= holdout_start:
        raise RuntimeError("모델 입력에 홀드아웃 행이 들어왔습니다.")
    if model_frame["raw_position"].iloc[-1] + LABEL_HORIZON + 1 >= len(raw):
        raise RuntimeError("마지막 모델 행의 5거래일 청산 시가가 개발구간 밖에 있습니다.")

    return ModelDataset(
        frame=model_frame,
        raw_prices=raw,
        feature_columns=feature_columns,
        combination=key,
    )
