"""확정된 KOSPI200 C·개별종목 K 모델을 한 번의 평가구간에 적용한다.

이 모듈은 후보를 비교하거나 다시 고르지 않는다. 개발구간 보고서에서 이미 확정한
LogisticRegression 두 개와 사전 고정한 설정만 받아 전체 개발구간에 fit하고, 별도로
전달된 평가구간에는 predict만 수행한다.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from features.model_dataset import COMBINATION_FEATURES
from features.stock_model_dataset import STOCK_COMBINATION_FEATURES
from models.experiment import classification_probability_metrics, ordered_class_probabilities
from models.index_long_only import predict_with_up_threshold
from models.logistic import build_logistic_baseline

INDEX_COMBINATION = "C"
STOCK_COMBINATION = "K"
MODEL_NAME = "LogisticRegression"
INDEX_FEATURES = COMBINATION_FEATURES[INDEX_COMBINATION]
STOCK_FEATURES = STOCK_COMBINATION_FEATURES[STOCK_COMBINATION]
LABEL_COLUMN = "label_numeric"


@dataclass(frozen=True)
class FinalModelConfig:
    """홀드아웃을 보기 전에 파일로 봉인할 두 모델의 단일 설정."""

    schema_version: int
    index_combination: str
    index_model: str
    index_features: tuple[str, ...]
    index_class_weight: str | None
    index_up_threshold: float
    stock_combination: str
    stock_model: str
    stock_features: tuple[str, ...]
    stock_class_weight: str | None

    def validate(self) -> None:
        """현재 ADR 0007 선정 결과와 다른 설정이면 실행 전에 중단한다."""

        expected = {
            "schema_version": 1,
            "index_combination": INDEX_COMBINATION,
            "index_model": MODEL_NAME,
            "index_features": INDEX_FEATURES,
            "stock_combination": STOCK_COMBINATION,
            "stock_model": MODEL_NAME,
            "stock_features": STOCK_FEATURES,
        }
        actual = asdict(self)
        for key, value in expected.items():
            if actual[key] != value:
                raise ValueError(f"최종 모델 설정 {key}가 확정값과 다릅니다: {actual[key]!r}")
        if self.index_class_weight not in (None, "balanced"):
            raise ValueError("KOSPI200 class_weight는 None 또는 balanced여야 합니다.")
        if self.stock_class_weight not in (None, "balanced"):
            raise ValueError("개별종목 class_weight는 None 또는 balanced여야 합니다.")
        if not 0.0 <= self.index_up_threshold <= 1.0:
            raise ValueError("KOSPI200 상승 임계값은 0~1이어야 합니다.")

    def canonical_json(self) -> str:
        """설정 지문이 운영체제나 들여쓰기에 따라 달라지지 않는 JSON을 만든다."""

        self.validate()
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FinalHoldoutResult:
    """두 트랙 예측과 결합 매수 후보 및 평가 요약."""

    index_predictions: pd.DataFrame
    stock_predictions: pd.DataFrame
    combined_predictions: pd.DataFrame
    metrics: dict[str, dict[str, object]]
    config_sha256: str


def _normalize_frame(
    frame: pd.DataFrame,
    *,
    features: tuple[str, ...],
    identity: tuple[str, ...],
    context: str,
) -> pd.DataFrame:
    required = {"bas_dd", LABEL_COLUMN, *features, *identity}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{context} 입력 열이 없습니다: {sorted(missing)}")
    out = frame.loc[:, [*identity, "bas_dd", *features, LABEL_COLUMN]].copy()
    out["bas_dd"] = (
        out["bas_dd"].astype("string").str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(8)
    )
    if out["bas_dd"].isna().any() or (~out["bas_dd"].str.fullmatch(r"\d{8}")).any():
        raise ValueError(f"{context} bas_dd에 YYYYMMDD가 아닌 값이 있습니다.")
    values = out.loc[:, features].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{context} 피처에 결측 또는 무한대가 있습니다.")
    labels = pd.to_numeric(out[LABEL_COLUMN], errors="raise").astype(int)
    unknown = set(labels.tolist()) - {-1, 0, 1}
    if unknown:
        raise ValueError(f"{context} 라벨에 -1·0·1 이외 값이 있습니다: {sorted(unknown)}")
    out[LABEL_COLUMN] = labels
    return out


def _validate_periods(dev: pd.DataFrame, holdout: pd.DataFrame, *, context: str) -> None:
    if dev.empty or holdout.empty:
        raise ValueError(f"{context} 개발구간과 평가구간은 비어 있을 수 없습니다.")
    if str(dev["bas_dd"].max()) >= str(holdout["bas_dd"].min()):
        raise ValueError(f"{context} 개발구간과 평가구간 날짜가 겹치거나 역전됐습니다.")


def _metric_summary(
    frame: pd.DataFrame,
    actual_column: str,
    predicted_column: str,
) -> dict[str, object]:
    return classification_probability_metrics(
        frame[actual_column].to_numpy(dtype=int),
        frame[predicted_column].to_numpy(dtype=int),
        frame[["p_down", "p_neutral", "p_up"]].to_numpy(dtype=float),
    )


def run_final_holdout(
    index_dev: pd.DataFrame,
    index_holdout: pd.DataFrame,
    stock_dev: pd.DataFrame,
    stock_holdout: pd.DataFrame,
    config: FinalModelConfig,
) -> FinalHoldoutResult:
    """확정 설정으로 전체 개발구간을 학습하고 평가구간을 정확히 한 번 추론한다."""

    config.validate()
    index_train = _normalize_frame(
        index_dev,
        features=config.index_features,
        identity=(),
        context="KOSPI200 개발구간",
    ).sort_values("bas_dd", kind="stable")
    index_test = _normalize_frame(
        index_holdout,
        features=config.index_features,
        identity=(),
        context="KOSPI200 평가구간",
    ).sort_values("bas_dd", kind="stable")
    stock_train = _normalize_frame(
        stock_dev,
        features=config.stock_features,
        identity=("code",),
        context="개별종목 개발구간",
    ).sort_values(["bas_dd", "code"], kind="stable")
    stock_test = _normalize_frame(
        stock_holdout,
        features=config.stock_features,
        identity=("code",),
        context="개별종목 평가구간",
    ).sort_values(["bas_dd", "code"], kind="stable")
    _validate_periods(index_train, index_test, context="KOSPI200")
    _validate_periods(stock_train, stock_test, context="개별종목")
    if index_train["bas_dd"].duplicated().any() or index_test["bas_dd"].duplicated().any():
        raise ValueError("KOSPI200에는 거래일마다 한 행만 있어야 합니다.")
    if stock_train.duplicated(["bas_dd", "code"]).any() or stock_test.duplicated(
        ["bas_dd", "code"]
    ).any():
        raise ValueError("개별종목에는 같은 날짜·종목코드가 중복될 수 없습니다.")

    index_model = build_logistic_baseline(class_weight=config.index_class_weight)
    index_model.fit(index_train.loc[:, config.index_features], index_train[LABEL_COLUMN])
    index_probability = ordered_class_probabilities(
        index_model,
        index_test.loc[:, config.index_features],
    )
    index_predicted = predict_with_up_threshold(
        index_probability,
        config.index_up_threshold,
    )
    index_output = index_test.loc[:, ["bas_dd", LABEL_COLUMN]].rename(
        columns={LABEL_COLUMN: "actual"}
    )
    index_output["predicted"] = index_predicted
    index_output[["p_down", "p_neutral", "p_up"]] = index_probability
    index_output["selected_up_threshold"] = config.index_up_threshold

    stock_model = build_logistic_baseline(class_weight=config.stock_class_weight)
    stock_model.fit(stock_train.loc[:, config.stock_features], stock_train[LABEL_COLUMN])
    stock_probability = ordered_class_probabilities(
        stock_model,
        stock_test.loc[:, config.stock_features],
    )
    stock_predicted = np.asarray(
        stock_model.predict(stock_test.loc[:, config.stock_features]),
        dtype=int,
    )
    stock_output = stock_test.loc[:, ["bas_dd", "code", LABEL_COLUMN]].rename(
        columns={LABEL_COLUMN: "actual"}
    )
    stock_output["predicted"] = stock_predicted
    stock_output[["p_down", "p_neutral", "p_up"]] = stock_probability

    combined = stock_output.merge(
        index_output.loc[:, ["bas_dd", "predicted", "p_up"]].rename(
            columns={"predicted": "index_predicted", "p_up": "index_p_up"}
        ),
        on="bas_dd",
        how="inner",
        validate="many_to_one",
    )
    if len(combined) != len(stock_output):
        missing_dates = sorted(set(stock_output["bas_dd"]) - set(index_output["bas_dd"]))
        raise ValueError(f"개별종목 평가일에 KOSPI200 예측이 없습니다: {missing_dates[:5]}")
    combined = combined.rename(
        columns={"predicted": "stock_predicted", "actual": "stock_actual"}
    )
    combined["buy_candidate"] = combined["index_predicted"].eq(1) & combined[
        "stock_predicted"
    ].eq(1)

    return FinalHoldoutResult(
        index_predictions=index_output.reset_index(drop=True),
        stock_predictions=stock_output.reset_index(drop=True),
        combined_predictions=combined.reset_index(drop=True),
        metrics={
            "index": _metric_summary(index_output, "actual", "predicted"),
            "stock": _metric_summary(stock_output, "actual", "predicted"),
        },
        config_sha256=config.sha256,
    )


def config_from_dict(value: dict[str, Any]) -> FinalModelConfig:
    """JSON 설정을 읽되 피처 배열은 비교 가능한 튜플로 정규화한다."""

    normalized = dict(value)
    for key in ("index_features", "stock_features"):
        if key in normalized:
            normalized[key] = tuple(normalized[key])
    config = FinalModelConfig(**normalized)
    config.validate()
    return config


__all__ = [
    "FinalHoldoutResult",
    "FinalModelConfig",
    "INDEX_FEATURES",
    "STOCK_FEATURES",
    "config_from_dict",
    "run_final_holdout",
]
