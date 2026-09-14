"""KOSPI200 long-only 목적에 맞춰 상승 확률 임계값을 내부 검증에서 고른다.

최종 매매는 상승 예측만 매수하고 보합·하락은 모두 현금이다. 따라서 이 모듈은
하락 Recall을 선택축으로 사용하지 않는다. 외부 검증구간의 확률이나 정답으로 임계값을
고르면 OOS가 아니므로, 각 외부 폴드의 학습구간 끝에 별도 내부 검증구간을 둔다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score

from evaluation.baseline import multiclass_majority_class
from evaluation.overlapping import overlapping_long_only_returns
from features.model_dataset import ModelDataset
from models.experiment import (
    CLASS_WEIGHT_CANDIDATES,
    LABEL_HORIZON,
    MODEL_BUILDERS,
    ROUND_TRIP_COST,
    classification_probability_metrics,
    inner_class_weight_split,
    ordered_class_probabilities,
    window_selection_sharpe,
)

DEFAULT_UP_THRESHOLDS = tuple(np.round(np.arange(0.20, 0.801, 0.025), 3))
MIN_INNER_BUY_SIGNALS = 3


@dataclass(frozen=True)
class LongOnlyThresholdResult:
    """한 후보의 내부 선택과 외부 OOS 평가 결과."""

    inner_results: pd.DataFrame
    outer_results: pd.DataFrame
    oos_predictions: pd.DataFrame


def predict_with_up_threshold(probabilities: object, threshold: float) -> np.ndarray:
    """상승 확률이 임계값 이상이면 상승, 아니면 하락·보합 중 큰 확률을 고른다."""

    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("확률은 하락·보합·상승 3열의 2차원 배열이어야 합니다.")
    if not np.isfinite(values).all() or (values < 0.0).any() or (values > 1.0).any():
        raise ValueError("확률 배열에 결측·무한대 또는 0~1 밖의 값이 있습니다.")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("하락·보합·상승 확률의 합이 1이 아닙니다.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"상승 임계값은 0~1이어야 합니다: {threshold}")

    # 하락과 보합은 매매상 모두 현금이지만 3분류 보고를 유지한다. 동률이면 보합을 택한다.
    non_up = np.where(values[:, 0] > values[:, 1], -1, 0)
    return np.where(values[:, 2] >= threshold, 1, non_up).astype(int)


def _up_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    buy_mask = predicted == 1
    buy_signals = int(buy_mask.sum())
    opportunity_rate = float(buy_mask.mean())
    up_precision = float(
        precision_score(actual == 1, buy_mask, zero_division=0)
    )
    up_recall = float(recall_score(actual == 1, buy_mask, zero_division=0))
    # precision × opportunity는 전체 관측치 중 실제 상승을 맞혀 매수한 비율과 같다.
    correct_up_opportunity = float(np.mean(buy_mask & (actual == 1)))
    return {
        "up_precision": up_precision,
        "up_recall": up_recall,
        "buy_signals": buy_signals,
        "opportunity_rate": opportunity_rate,
        "correct_up_opportunity": correct_up_opportunity,
    }


def _threshold_evaluation(
    dataset: ModelDataset,
    indices: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    actual = dataset.y[indices]
    predicted = predict_with_up_threshold(probabilities, threshold)
    probability_metrics = classification_probability_metrics(
        actual,
        predicted,
        probabilities,
    )
    probability_metrics.pop("confusion_matrix")
    portfolio = overlapping_long_only_returns(
        dataset.opens,
        dataset.signal_positions[indices],
        predicted,
        horizon=LABEL_HORIZON,
        round_trip_cost=ROUND_TRIP_COST,
    )
    delta, strategy_sharpe, buyhold_sharpe, all_cash = window_selection_sharpe(portfolio)
    return {
        "threshold": float(threshold),
        **probability_metrics,
        **_up_metrics(actual, predicted),
        "strategy_sharpe_net": strategy_sharpe,
        "buyhold_sharpe_net": buyhold_sharpe,
        "delta_sharpe_net": delta,
        "all_cash": all_cash,
    }


def select_inner_threshold(
    dataset: ModelDataset,
    indices: Sequence[int],
    probabilities: object,
    *,
    thresholds: Sequence[float] = DEFAULT_UP_THRESHOLDS,
    min_buy_signals: int = MIN_INNER_BUY_SIGNALS,
) -> tuple[dict[str, object], pd.DataFrame]:
    """내부 검증 결과만으로 long-only 상승 임계값 하나를 고른다.

    비용 차감 ΔSharpe를 우선하되 최소 매수 기회가 없는 임계값은 제외한다. 동률이면
    ``상승 Precision × 기회비율``인 정답 매수 비율, 상승 Precision, 기회비율 순으로
    고른다. 이는 서로 단위가 다른 값을 임의 가중합으로 합치지 않기 위한 사전 규칙이다.
    """

    valid_indices = np.asarray(indices, dtype=int)
    probability_values = np.asarray(probabilities, dtype=float)
    if valid_indices.ndim != 1 or valid_indices.size == 0:
        raise ValueError("내부 검증 인덱스는 비어 있지 않은 1차원이어야 합니다.")
    if probability_values.shape != (len(valid_indices), 3):
        raise ValueError("내부 검증 확률의 행 수가 검증 인덱스와 다릅니다.")
    configured = tuple(float(value) for value in thresholds)
    if not configured or len(configured) != len(set(configured)):
        raise ValueError("상승 임계값은 중복 없는 비어 있지 않은 목록이어야 합니다.")
    if min_buy_signals < 1:
        raise ValueError("최소 내부 매수 신호 수는 1 이상이어야 합니다.")

    rows = [
        _threshold_evaluation(dataset, valid_indices, probability_values, threshold)
        for threshold in configured
    ]
    eligible = [row for row in rows if int(row["buy_signals"]) >= min_buy_signals]
    if not eligible:
        raise RuntimeError(
            "설정한 임계값에서 최소 내부 매수 신호를 확보하지 못했습니다: "
            f"최대 {max(int(row['buy_signals']) for row in rows)}개"
        )
    selected = max(
        eligible,
        key=lambda row: (
            float(row["delta_sharpe_net"]),
            float(row["correct_up_opportunity"]),
            float(row["up_precision"]),
            float(row["opportunity_rate"]),
            -float(row["threshold"]),
        ),
    )
    return selected, pd.DataFrame(rows)


def restrict_dataset_to_dates(
    dataset: ModelDataset,
    dates: Sequence[str],
) -> ModelDataset:
    """후보마다 같은 거래일과 같은 외부 폴드를 쓰도록 데이터셋을 교집합으로 줄인다."""

    expected = tuple(str(value) for value in dates)
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("공통 거래일은 중복 없는 비어 있지 않은 목록이어야 합니다.")
    selected = dataset.frame.loc[dataset.frame["bas_dd"].isin(expected)].copy()
    selected = selected.sort_values("bas_dd", kind="stable").reset_index(drop=True)
    actual = tuple(selected["bas_dd"].astype(str))
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        raise ValueError(f"후보 데이터셋에 공통 거래일이 없습니다: {missing[:5]}")
    return ModelDataset(
        frame=selected,
        raw_prices=dataset.raw_prices,
        feature_columns=dataset.feature_columns,
        combination=dataset.combination,
    )


def evaluate_long_only_thresholds(
    dataset: ModelDataset,
    model_name: str,
    outer_splits: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    thresholds: Sequence[float] = DEFAULT_UP_THRESHOLDS,
) -> LongOnlyThresholdResult:
    """한 KOSPI200 후보를 누수 없는 내부 임계값 선택 후 외부 폴드에서 평가한다."""

    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"알 수 없는 모델입니다: {model_name}")
    splits = list(outer_splits)
    if not splits:
        raise ValueError("외부 OOS 분할이 없습니다.")

    builder = MODEL_BUILDERS[model_name]
    inner_rows: list[dict[str, object]] = []
    outer_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    for fold, (outer_train, outer_valid) in enumerate(splits, start=1):
        train = np.asarray(outer_train, dtype=int)
        valid = np.asarray(outer_valid, dtype=int)
        inner_train, inner_valid = inner_class_weight_split(train)
        weight_candidates: list[dict[str, object]] = []

        for class_weight in CLASS_WEIGHT_CANDIDATES:
            model = builder(class_weight=class_weight)
            model.fit(dataset.x.iloc[inner_train], dataset.y[inner_train])
            inner_probabilities = ordered_class_probabilities(
                model,
                dataset.x.iloc[inner_valid],
            )
            selected_threshold, threshold_rows = select_inner_threshold(
                dataset,
                inner_valid,
                inner_probabilities,
                thresholds=thresholds,
            )
            inner_pr_auc_up = float(
                average_precision_score(
                    dataset.y[inner_valid] == 1,
                    inner_probabilities[:, 2],
                )
            )
            candidate = {
                "fold": fold,
                "class_weight": class_weight,
                "inner_train_size": len(inner_train),
                "inner_valid_size": len(inner_valid),
                "inner_train_end": dataset.frame.iloc[inner_train[-1]]["bas_dd"],
                "inner_valid_start": dataset.frame.iloc[inner_valid[0]]["bas_dd"],
                "inner_valid_end": dataset.frame.iloc[inner_valid[-1]]["bas_dd"],
                "inner_pr_auc_up": inner_pr_auc_up,
                **selected_threshold,
            }
            weight_candidates.append(candidate)
            inner_rows.append(candidate)
            # 전체 임계값 표는 결과 JSON을 과도하게 키우지 않고, 선택 재현에 필요한
            # 후보 수와 범위만 선택 행에 남긴다.
            candidate["threshold_candidate_count"] = len(threshold_rows)

        selected_weight = max(
            weight_candidates,
            key=lambda row: (
                float(row["inner_pr_auc_up"]),
                float(row["delta_sharpe_net"]),
                float(row["correct_up_opportunity"]),
                float(row["up_precision"]),
                row["class_weight"] is None,
            ),
        )
        class_weight = selected_weight["class_weight"]
        threshold = float(selected_weight["threshold"])
        final_model = builder(class_weight=class_weight)
        final_model.fit(dataset.x.iloc[train], dataset.y[train])
        probabilities = ordered_class_probabilities(final_model, dataset.x.iloc[valid])
        evaluated = _threshold_evaluation(dataset, valid, probabilities, threshold)
        predicted = predict_with_up_threshold(probabilities, threshold)
        argmax_predicted = np.asarray((-1, 0, 1))[np.argmax(probabilities, axis=1)]
        majority_predicted = multiclass_majority_class(dataset.y[train], len(valid))
        baseline_accuracy = float(np.mean(majority_predicted == dataset.y[valid]))
        accuracy = float(evaluated["accuracy"])

        outer_rows.append(
            {
                "model": model_name,
                "fold": fold,
                "selected_class_weight": class_weight,
                "selected_up_threshold": threshold,
                "train_size": len(train),
                "valid_size": len(valid),
                "train_end": dataset.frame.iloc[train[-1]]["bas_dd"],
                "valid_start": dataset.frame.iloc[valid[0]]["bas_dd"],
                "valid_end": dataset.frame.iloc[valid[-1]]["bas_dd"],
                **evaluated,
                "training_majority_baseline_accuracy": baseline_accuracy,
                "accuracy_minus_training_majority_baseline": accuracy - baseline_accuracy,
                "beats_training_majority_baseline": accuracy > baseline_accuracy,
            }
        )
        prediction_rows.extend(
            {
                "model": model_name,
                "fold": fold,
                "bas_dd": dataset.frame.iloc[index]["bas_dd"],
                "actual": int(actual),
                "predicted": int(selected),
                "argmax_predicted": int(argmax_value),
                "training_majority_predicted": int(baseline),
                "selected_up_threshold": threshold,
                "p_down": float(probability[0]),
                "p_neutral": float(probability[1]),
                "p_up": float(probability[2]),
            }
            for index, actual, selected, argmax_value, baseline, probability in zip(
                valid,
                dataset.y[valid],
                predicted,
                argmax_predicted,
                majority_predicted,
                probabilities,
                strict=True,
            )
        )

    return LongOnlyThresholdResult(
        inner_results=pd.DataFrame(inner_rows),
        outer_results=pd.DataFrame(outer_rows),
        oos_predictions=pd.DataFrame(prediction_rows),
    )


def summarize_long_only_result(result: LongOnlyThresholdResult) -> dict[str, object]:
    """한 후보의 12폴드 OOS 확률·분류·경제성 결과를 한 행으로 요약한다."""

    predictions = result.oos_predictions
    outer = result.outer_results
    if predictions.empty or outer.empty:
        raise ValueError("요약할 long-only OOS 결과가 없습니다.")
    actual = predictions["actual"].to_numpy(dtype=int)
    predicted = predictions["predicted"].to_numpy(dtype=int)
    probabilities = predictions[["p_down", "p_neutral", "p_up"]].to_numpy(dtype=float)
    metrics = classification_probability_metrics(actual, predicted, probabilities)
    metrics.pop("confusion_matrix")
    up = _up_metrics(actual, predicted)
    baseline = predictions["training_majority_predicted"].to_numpy(dtype=int)
    counts = pd.Series(actual).value_counts()
    in_sample_majority_accuracy = float(counts.max() / counts.sum())
    operational_baseline_accuracy = float(np.mean(actual == baseline))
    return {
        **metrics,
        **up,
        "in_sample_majority_accuracy": in_sample_majority_accuracy,
        "training_majority_baseline_accuracy": operational_baseline_accuracy,
        "accuracy_minus_training_majority_baseline": (
            float(metrics["accuracy"]) - operational_baseline_accuracy
        ),
        "baseline_win_folds": int(outer["beats_training_majority_baseline"].sum()),
        "folds": int(len(outer)),
        "delta_sharpe_net_median": float(outer["delta_sharpe_net"].median()),
        "delta_sharpe_net_mean": float(outer["delta_sharpe_net"].mean()),
        "selected_up_threshold_median": float(outer["selected_up_threshold"].median()),
        "selected_up_threshold_min": float(outer["selected_up_threshold"].min()),
        "selected_up_threshold_max": float(outer["selected_up_threshold"].max()),
        "all_cash_folds": int(outer["all_cash"].sum()),
    }


__all__ = [
    "DEFAULT_UP_THRESHOLDS",
    "LongOnlyThresholdResult",
    "evaluate_long_only_thresholds",
    "predict_with_up_threshold",
    "restrict_dataset_to_dates",
    "select_inner_threshold",
    "summarize_long_only_result",
]
