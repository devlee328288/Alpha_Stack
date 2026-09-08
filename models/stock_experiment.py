"""개별종목 패널의 날짜 단위 워크포워드와 확률 예측."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from evaluation.walk_forward import expanding_group_splits
from features.stock_model_dataset import StockModelDataset
from models.experiment import (
    CLASS_WEIGHT_CANDIDATES,
    MODEL_BUILDERS,
    classification_metrics,
    classification_probability_metrics,
    ordered_class_probabilities,
)

N_FOLDS = 12
MIN_TRAIN_DATES = 750
VALID_DATES = 60
LABEL_HORIZON = 5

ModelBuilder = Callable[..., object]

# 세 클래스의 학습 빈도가 정확히 같을 때는 포지션을 만들지 않는 중립을 우선한다.
# 검증 정답을 보고 동률을 푸는 순간 기준선에 미래정보가 들어가므로 순서는 고정한다.
MAJORITY_TIE_BREAK = (0, -1, 1)


@dataclass(frozen=True)
class StockExperimentResult:
    """내부 가중치 비교, 외부 OOS 점수와 종목별 확률."""

    inner_results: pd.DataFrame
    outer_results: pd.DataFrame
    oos_predictions: pd.DataFrame


def fold_classification_baselines(
    y_train: object,
    y_valid: object,
) -> dict[str, int | float]:
    """학습 최빈 기준선과 검증 분포의 사후 참고값을 서로 분리해 계산한다."""

    train = np.asarray(y_train, dtype=int)
    valid = np.asarray(y_valid, dtype=int)
    if train.ndim != 1 or valid.ndim != 1 or train.size == 0 or valid.size == 0:
        raise ValueError("학습·검증 라벨은 비어 있지 않은 1차원 배열이어야 합니다.")
    unknown = (set(train.tolist()) | set(valid.tolist())) - {-1, 0, 1}
    if unknown:
        raise ValueError(f"3분류 라벨이 아닌 값이 있습니다: {sorted(unknown)}")

    train_counts = {label: int(np.sum(train == label)) for label in MAJORITY_TIE_BREAK}
    valid_counts = {label: int(np.sum(valid == label)) for label in MAJORITY_TIE_BREAK}
    train_majority = max(MAJORITY_TIE_BREAK, key=train_counts.__getitem__)
    valid_majority = max(MAJORITY_TIE_BREAK, key=valid_counts.__getitem__)
    return {
        "training_majority_class": train_majority,
        "training_majority_baseline_accuracy": float(np.mean(valid == train_majority)),
        # 이 값은 검증 정답 분포를 본 사후 통계다. 모델 비교 기준선으로 사용하지 않는다.
        "validation_majority_class": valid_majority,
        "validation_majority_oracle_accuracy": float(np.mean(valid == valid_majority)),
        "validation_down_rate": valid_counts[-1] / int(valid.size),
        "validation_neutral_rate": valid_counts[0] / int(valid.size),
        "validation_up_rate": valid_counts[1] / int(valid.size),
    }


def inner_group_class_weight_split(
    groups: object,
    outer_train_index: np.ndarray,
    *,
    valid_dates: int = VALID_DATES,
    gap_dates: int = LABEL_HORIZON,
) -> tuple[np.ndarray, np.ndarray]:
    """외부 학습의 마지막 60거래일을 내부 검증, 직전 5거래일을 갭으로 둔다."""

    all_groups = np.asarray(groups)
    indices = np.asarray(outer_train_index, dtype=int)
    if all_groups.ndim != 1 or indices.ndim != 1 or indices.size == 0:
        raise ValueError("groups와 외부 학습 인덱스는 비어 있지 않은 1차원이어야 합니다.")
    selected_groups = all_groups[indices]
    codes, unique_groups = pd.factorize(selected_groups, sort=False)
    if len(unique_groups) < valid_dates + gap_dates + 1:
        raise ValueError("내부 학습·5일 갭·60일 검증을 만들 거래일이 부족합니다.")
    if not pd.Index(unique_groups).is_monotonic_increasing:
        raise ValueError("외부 학습 그룹은 과거부터 미래 순서여야 합니다.")
    starts = np.concatenate(([0], np.flatnonzero(codes[1:] != codes[:-1]) + 1))
    if len(starts) != len(unique_groups):
        raise ValueError("같은 거래일의 종목 행은 한곳에 연속해서 모여 있어야 합니다.")

    valid_group_start = len(unique_groups) - valid_dates
    train_group_end = valid_group_start - gap_dates
    inner_train = indices[codes < train_group_end]
    inner_valid = indices[codes >= valid_group_start]
    return inner_train, inner_valid


def evaluate_stock_models(
    dataset: StockModelDataset,
    *,
    model_builders: Mapping[str, ModelBuilder] = MODEL_BUILDERS,
    n_folds: int = N_FOLDS,
    min_train_dates: int = MIN_TRAIN_DATES,
    valid_dates: int = VALID_DATES,
    gap_dates: int = LABEL_HORIZON,
    outer_splits: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
) -> StockExperimentResult:
    """네 모델을 날짜 그룹 12폴드로 평가하고 폴드 안에서 가중치를 다시 고른다."""

    if not model_builders:
        raise ValueError("평가할 모델이 없습니다.")
    splits = list(outer_splits) if outer_splits is not None else expanding_group_splits(
        dataset.groups,
        n_folds=n_folds,
        min_train=min_train_dates,
        horizon=valid_dates,
        gap=gap_dates,
        label_horizon=LABEL_HORIZON,
    )
    if not splits:
        raise ValueError("외부 OOS 분할이 없습니다.")
    inner_rows: list[dict[str, object]] = []
    outer_rows: list[dict[str, object]] = []
    prediction_parts: list[pd.DataFrame] = []

    for model_name, builder in model_builders.items():
        for fold, (outer_train, outer_valid) in enumerate(splits, start=1):
            inner_train, inner_valid = inner_group_class_weight_split(
                dataset.groups,
                outer_train,
                valid_dates=valid_dates,
                gap_dates=gap_dates,
            )
            candidates: list[dict[str, object]] = []
            for class_weight in CLASS_WEIGHT_CANDIDATES:
                model = builder(class_weight=class_weight)
                model.fit(dataset.x.iloc[inner_train], dataset.y[inner_train])
                predicted = np.asarray(model.predict(dataset.x.iloc[inner_valid]), dtype=int)
                row = {
                    "model": model_name,
                    "fold": fold,
                    "class_weight": class_weight,
                    "inner_train_dates": int(
                        pd.Series(dataset.groups[inner_train]).nunique()
                    ),
                    "inner_valid_dates": int(
                        pd.Series(dataset.groups[inner_valid]).nunique()
                    ),
                    "inner_train_rows": int(len(inner_train)),
                    "inner_valid_rows": int(len(inner_valid)),
                    **classification_metrics(dataset.y[inner_valid], predicted),
                }
                candidates.append(row)
                inner_rows.append(row)

            # 같은 점수면 추가 가정을 하지 않는 기본 가중치를 선택한다.
            best = max(
                candidates,
                key=lambda row: (
                    float(row["core_harmonic_mean"]),
                    row["class_weight"] is None,
                ),
            )
            selected_weight = best["class_weight"]
            final_model = builder(class_weight=selected_weight)
            final_model.fit(dataset.x.iloc[outer_train], dataset.y[outer_train])
            valid_x = dataset.x.iloc[outer_valid]
            predicted = np.asarray(final_model.predict(valid_x), dtype=int)
            probabilities = ordered_class_probabilities(final_model, valid_x)
            metrics = classification_probability_metrics(
                dataset.y[outer_valid],
                predicted,
                probabilities,
            )
            metrics.pop("confusion_matrix")
            baselines = fold_classification_baselines(
                dataset.y[outer_train],
                dataset.y[outer_valid],
            )
            outer_rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "selected_class_weight": selected_weight,
                    "train_dates": int(pd.Series(dataset.groups[outer_train]).nunique()),
                    "valid_dates": int(pd.Series(dataset.groups[outer_valid]).nunique()),
                    "train_rows": int(len(outer_train)),
                    "valid_rows": int(len(outer_valid)),
                    "train_end": str(dataset.groups[outer_train][-1]),
                    "valid_start": str(dataset.groups[outer_valid][0]),
                    "valid_end": str(dataset.groups[outer_valid][-1]),
                    **metrics,
                    **baselines,
                    "accuracy_minus_training_majority_baseline": (
                        float(metrics["accuracy"])
                        - float(baselines["training_majority_baseline_accuracy"])
                    ),
                }
            )

            identity_columns = [
                column
                for column in (
                    "bas_dd",
                    "code",
                    "name",
                    "industry",
                    "industry_index_name",
                    "sector_market_cap_rank",
                    "industry_stock_rank",
                    "candidate_rank",
                    "entry_bas_dd",
                    "exit_bas_dd",
                    "entry_adj_open",
                    "exit_adj_open",
                    "fwd_return_5d",
                    "label_numeric",
                )
                if column in dataset.frame.columns
            ]
            fold_predictions = dataset.frame.iloc[outer_valid][identity_columns].copy()
            fold_predictions.insert(0, "fold", fold)
            fold_predictions.insert(0, "model", model_name)
            fold_predictions["predicted"] = predicted
            fold_predictions["p_down"] = probabilities[:, 0]
            fold_predictions["p_neutral"] = probabilities[:, 1]
            fold_predictions["p_up"] = probabilities[:, 2]
            prediction_parts.append(fold_predictions)

    predictions = pd.concat(prediction_parts, ignore_index=True)
    return StockExperimentResult(
        inner_results=pd.DataFrame(inner_rows),
        outer_results=pd.DataFrame(outer_rows),
        oos_predictions=predictions,
    )


__all__ = [
    "StockExperimentResult",
    "evaluate_stock_models",
    "fold_classification_baselines",
    "inner_group_class_weight_split",
]
