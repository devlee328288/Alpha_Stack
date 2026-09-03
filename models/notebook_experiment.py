"""모델 실험 노트북이 공통으로 사용하는 결과 집계 함수."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix

from features.model_dataset import ModelDataset
from models.experiment import NestedWeightResult, classification_metrics

CLASS_LABELS = (-1, 0, 1)
CLASS_NAMES = ("하락", "중립", "상승")


@dataclass(frozen=True)
class NotebookExperimentSummary:
    """노트북 표와 비교 파일을 만드는 데 필요한 한 모델의 OOS 결과."""

    summary: dict[str, object]
    fold_results: pd.DataFrame
    inner_results: pd.DataFrame
    weight_counts: pd.DataFrame
    confusion: pd.DataFrame
    class_report: pd.DataFrame


def summarize_notebook_experiment(
    dataset: ModelDataset,
    nested: NestedWeightResult,
    model_name: str,
) -> NotebookExperimentSummary:
    """12개 외부 OOS 예측을 합쳐 기존 노트북 형식의 지표를 만든다."""

    predictions = nested.oos_predictions
    if predictions.empty or set(predictions["model"]) != {model_name}:
        raise ValueError(f"{model_name}의 OOS 예측만 정확히 들어 있어야 합니다.")
    actual = predictions["actual"].to_numpy(dtype=int)
    predicted = predictions["predicted"].to_numpy(dtype=int)
    metrics = classification_metrics(actual, predicted)
    matrix = confusion_matrix(actual, predicted, labels=CLASS_LABELS)
    report = pd.DataFrame(
        classification_report(
            actual,
            predicted,
            labels=CLASS_LABELS,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    counts = pd.Series(actual).value_counts().reindex(CLASS_LABELS, fill_value=0)
    majority_accuracy = float(counts.max() / counts.sum())
    fold_results = nested.outer_results.copy()
    inner_results = nested.inner_results.copy()
    weight_counts = (
        fold_results.assign(
            selected_class_weight=fold_results["selected_class_weight"].fillna("None")
        )
        .groupby("selected_class_weight")
        .size()
        .rename("선택 폴드 수")
        .reset_index()
        .rename(columns={"selected_class_weight": "클래스 가중치"})
    )
    summary: dict[str, object] = {
        "model": model_name,
        "combination": dataset.combination,
        "feature_columns": list(dataset.feature_columns),
        "feature_count": len(dataset.feature_columns),
        "dataset_rows": len(dataset.frame),
        "dataset_start": str(dataset.frame["bas_dd"].min()),
        "dataset_end": str(dataset.frame["bas_dd"].max()),
        "oos_rows": len(actual),
        **metrics,
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "majority_accuracy": majority_accuracy,
        "down_recall": float(report.loc["하락", "recall"]),
        "neutral_recall": float(report.loc["중립", "recall"]),
        "up_recall": float(report.loc["상승", "recall"]),
        "predicted_down": int(np.sum(predicted == -1)),
        "predicted_neutral": int(np.sum(predicted == 0)),
        "predicted_up": int(np.sum(predicted == 1)),
        "delta_sharpe_net_median": float(fold_results["delta_sharpe_net"].median()),
        "delta_sharpe_net_mean": float(fold_results["delta_sharpe_net"].mean()),
        "all_cash_folds": int(fold_results["all_cash"].sum()),
    }
    return NotebookExperimentSummary(
        summary=summary,
        fold_results=fold_results,
        inner_results=inner_results,
        weight_counts=weight_counts,
        confusion=pd.DataFrame(
            matrix,
            index=[f"실제 {name}" for name in CLASS_NAMES],
            columns=[f"예측 {name}" for name in CLASS_NAMES],
        ),
        class_report=report,
    )
