"""개발구간 평가 보고서에서 트랙별 1위 모델을 읽어온다.

최종 산출 코드가 특정 조합 문자를 하드코딩하면 재평가 뒤 1위가 바뀌어도 예전 모델을
계속 사용할 수 있다. 이 모듈은 각 트랙이 이미 확정해 보고서에 기록한 선정 기준만 읽으며,
홀드아웃 결과로 모델을 다시 고르지 않는다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

INDEX_POLICY_CORE_HARMONIC = "core_harmonic_mean"
INDEX_POLICY_ACCURACY_THEN_MACRO = "accuracy_then_macro_f1"
INDEX_POLICY_ACCURACY_MACRO_HARMONIC = "accuracy_macro_f1_harmonic"
INDEX_POLICY_DELTA_REPORTED_MAJORITY = "accuracy_minus_reported_majority_then_macro_f1"
INDEX_POLICY_LONG_ONLY = "operational_gate_then_pr_auc_up"
INDEX_SELECTION_POLICIES = (
    INDEX_POLICY_CORE_HARMONIC,
    INDEX_POLICY_ACCURACY_THEN_MACRO,
    INDEX_POLICY_ACCURACY_MACRO_HARMONIC,
    INDEX_POLICY_DELTA_REPORTED_MAJORITY,
)


@dataclass(frozen=True)
class WinningModel:
    """최종 실행에 필요한 1위 모델 정보."""

    track: str
    combination: str
    model: str
    feature_columns: tuple[str, ...]
    return_features: tuple[str, ...] = ()
    selection_metric: str = ""
    selection_value: float | None = None


def _read_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"모델 평가 보고서가 없습니다: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError(f"모델 평가 보고서의 최상위 값은 객체여야 합니다: {path}")
    return report


def _feature_tuple(value: Any, context: str) -> tuple[str, ...]:
    """비어 있거나 문자열이 아닌 피처 목록이 최종 실행으로 넘어가지 않게 막는다."""

    if not isinstance(value, list) or not value:
        raise ValueError(f"{context} 피처 목록이 비어 있거나 배열이 아닙니다.")
    features = tuple(str(item).strip() for item in value)
    if any(not item for item in features):
        raise ValueError(f"{context} 피처 목록에 빈 이름이 있습니다.")
    if len(features) != len(set(features)):
        raise ValueError(f"{context} 피처 목록에 중복이 있습니다.")
    return features


def _two_metric_harmonic_mean(first: float, second: float) -> float:
    """두 양수 지표의 동일 가중 조화평균을 계산한다."""

    denominator = first + second
    return 0.0 if denominator == 0.0 else 2.0 * first * second / denominator


def _index_selection_key(
    record: dict[str, Any],
    policy: str,
) -> tuple[Any, ...]:
    """KOSPI200 후보를 정책별로 비교하되 항상 같은 동률 순서를 만든다."""

    experiment = record.get("experiment", {})
    summary = record.get("summary", {})
    required = {"core_harmonic_mean", "accuracy", "macro_f1", "feature_columns"}
    if policy == INDEX_POLICY_DELTA_REPORTED_MAJORITY:
        required.add("majority_accuracy")
    missing = required - set(summary)
    if missing:
        raise ValueError(f"KOSPI200 평가 지표가 없습니다: {sorted(missing)}")

    accuracy = float(summary["accuracy"])
    macro_f1 = float(summary["macro_f1"])
    core_harmonic = float(summary["core_harmonic_mean"])
    if policy == INDEX_POLICY_CORE_HARMONIC:
        metrics = (core_harmonic, accuracy, macro_f1)
    elif policy == INDEX_POLICY_ACCURACY_THEN_MACRO:
        metrics = (accuracy, macro_f1, core_harmonic)
    elif policy == INDEX_POLICY_ACCURACY_MACRO_HARMONIC:
        metrics = (_two_metric_harmonic_mean(accuracy, macro_f1), accuracy, macro_f1)
    elif policy == INDEX_POLICY_DELTA_REPORTED_MAJORITY:
        metrics = (
            accuracy - float(summary["majority_accuracy"]),
            macro_f1,
            accuracy,
        )
    else:
        raise ValueError(f"지원하지 않는 KOSPI200 선정 정책입니다: {policy}")

    # 수치가 모두 같은 후보도 입력 순서에 따라 결과가 바뀌지 않게 식별자를 붙인다.
    return (
        *metrics,
        str(experiment.get("combination", "")),
        str(experiment.get("model", "")),
        json.dumps(experiment.get("return_features", []), ensure_ascii=False),
    )


def _index_selection_value(record: dict[str, Any], policy: str) -> tuple[str, float]:
    """정책 이름과 보고서에 남길 대표 선택값을 반환한다."""

    summary = record["summary"]
    accuracy = float(summary["accuracy"])
    macro_f1 = float(summary["macro_f1"])
    if policy == INDEX_POLICY_CORE_HARMONIC:
        return "core_harmonic_mean", float(summary["core_harmonic_mean"])
    if policy == INDEX_POLICY_ACCURACY_THEN_MACRO:
        return "accuracy_then_macro_f1", accuracy
    if policy == INDEX_POLICY_ACCURACY_MACRO_HARMONIC:
        return "accuracy_macro_f1_harmonic", _two_metric_harmonic_mean(
            accuracy,
            macro_f1,
        )
    if policy == INDEX_POLICY_DELTA_REPORTED_MAJORITY:
        return (
            "accuracy_minus_reported_majority_accuracy",
            accuracy - float(summary["majority_accuracy"]),
        )
    raise ValueError(f"지원하지 않는 KOSPI200 선정 정책입니다: {policy}")


def select_best_index_model(
    report: dict[str, Any],
    *,
    policy: str = INDEX_POLICY_CORE_HARMONIC,
) -> WinningModel:
    """KOSPI200 실험 1위를 반환한다. 기본값은 현재 운영 중인 조화평균이다."""

    experiments = report.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("KOSPI200 평가 보고서에 experiments가 없습니다.")

    winner = max(experiments, key=lambda record: _index_selection_key(record, policy))
    experiment = winner["experiment"]
    summary = winner["summary"]
    combination = str(experiment.get("combination", ""))
    model = str(experiment.get("model", ""))
    if not combination or not model:
        raise ValueError("KOSPI200 1위 모델의 조합 또는 모델명이 비어 있습니다.")
    selection_metric, selection_value = _index_selection_value(winner, policy)
    return WinningModel(
        track="KOSPI200",
        combination=combination,
        model=model,
        feature_columns=_feature_tuple(summary["feature_columns"], "KOSPI200 1위 모델"),
        return_features=tuple(
            str(value) for value in experiment.get("return_features", [])
        ),
        selection_metric=selection_metric,
        selection_value=selection_value,
    )


def select_best_long_only_index_model(report: dict[str, Any]) -> WinningModel:
    """완료된 long-only 100후보 보고서의 잠정 1위를 반환한다.

    순위는 실행기가 기록한 운영 기준선 게이트와 상승 PR-AUC 순서를 그대로 따른다.
    보고서가 부분 실행 상태이거나 순위가 중복되면 중간 결과를 최종 모델처럼 쓰지 않는다.
    """

    if report.get("status") != "complete":
        raise ValueError("KOSPI200 long-only 평가 보고서가 완료 상태가 아닙니다.")
    if report.get("holdout_used") is not False:
        raise ValueError("KOSPI200 long-only 모델 선택에 홀드아웃이 사용됐습니다.")
    candidates = report.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("KOSPI200 long-only 평가 보고서에 candidates가 없습니다.")
    expected = int(report.get("expected_candidate_count", len(candidates)))
    if len(candidates) != expected:
        raise ValueError(
            f"KOSPI200 long-only 후보 수가 완료 조건과 다릅니다: {len(candidates)}/{expected}"
        )
    ranks = [int(record.get("rank", 0)) for record in candidates]
    if sorted(ranks) != list(range(1, len(candidates) + 1)):
        raise ValueError("KOSPI200 long-only 후보 순위가 중복되거나 빠졌습니다.")

    winner = min(candidates, key=lambda record: int(record["rank"]))
    experiment = winner.get("experiment", {})
    summary = winner.get("summary", {})
    combination = str(experiment.get("combination", "")).strip()
    model = str(experiment.get("model", "")).strip()
    if not combination or not model:
        raise ValueError("KOSPI200 long-only 1위 모델의 조합 또는 모델명이 비어 있습니다.")
    required = {"pr_auc_up", "passes_operational_gate"}
    missing = required - set(summary)
    if missing:
        raise ValueError(f"KOSPI200 long-only 1위 지표가 없습니다: {sorted(missing)}")
    if summary["passes_operational_gate"] is not True:
        raise ValueError("KOSPI200 long-only 1위가 운영 기준선 게이트를 통과하지 못했습니다.")
    return WinningModel(
        track="KOSPI200",
        combination=combination,
        model=model,
        feature_columns=_feature_tuple(winner.get("feature_columns"), "KOSPI200 long-only 1위"),
        return_features=tuple(str(value) for value in experiment.get("return_features", [])),
        selection_metric=INDEX_POLICY_LONG_ONLY,
        selection_value=float(summary["pr_auc_up"]),
    )


def compare_index_model_selection_policies(report: dict[str, Any]) -> dict[str, Any]:
    """홀드아웃을 보지 않고 개발구간 후보의 선정 정책별 1위를 비교한다."""

    experiments = report.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("KOSPI200 평가 보고서에 experiments가 없습니다.")

    descriptions = {
        INDEX_POLICY_CORE_HARMONIC: "Accuracy·Macro F1·하락 Recall 조화평균",
        INDEX_POLICY_ACCURACY_THEN_MACRO: "Accuracy 우선, Macro F1 차순",
        INDEX_POLICY_ACCURACY_MACRO_HARMONIC: "Accuracy·Macro F1 동일 가중 조화평균",
        INDEX_POLICY_DELTA_REPORTED_MAJORITY: (
            "Accuracy-보고서 majority_accuracy 우선, Macro F1 차순"
        ),
    }
    comparisons: list[dict[str, Any]] = []
    for policy in INDEX_SELECTION_POLICIES:
        winner_record = max(
            experiments,
            key=lambda record, selected_policy=policy: _index_selection_key(
                record,
                selected_policy,
            ),
        )
        winner = select_best_index_model(report, policy=policy)
        summary = winner_record["summary"]
        accuracy = float(summary["accuracy"])
        macro_f1 = float(summary["macro_f1"])
        majority_accuracy = summary.get("majority_accuracy")
        comparisons.append(
            {
                "policy": policy,
                "description": descriptions[policy],
                "production_default": policy == INDEX_POLICY_CORE_HARMONIC,
                "winner": {
                    "combination": winner.combination,
                    "return_features": list(winner.return_features),
                    "model": winner.model,
                    "feature_columns": list(winner.feature_columns),
                    "selection_metric": winner.selection_metric,
                    "selection_value": winner.selection_value,
                    "accuracy": accuracy,
                    "macro_f1": macro_f1,
                    "accuracy_macro_f1_harmonic": _two_metric_harmonic_mean(
                        accuracy,
                        macro_f1,
                    ),
                    "down_recall": float(summary["down_recall"]),
                    "core_harmonic_mean": float(summary["core_harmonic_mean"]),
                    "balanced_accuracy": float(summary["balanced_accuracy"]),
                    "mcc": float(summary["mcc"]),
                    "up_recall": float(summary["up_recall"]),
                    "pr_auc_up": float(summary["pr_auc_up"]),
                    "predicted_down": int(summary["predicted_down"]),
                    "predicted_neutral": int(summary["predicted_neutral"]),
                    "predicted_up": int(summary["predicted_up"]),
                    "majority_accuracy": (
                        float(majority_accuracy) if majority_accuracy is not None else None
                    ),
                    "accuracy_minus_reported_majority": (
                        accuracy - float(majority_accuracy)
                        if majority_accuracy is not None
                        else None
                    ),
                },
            }
        )
    return {
        "status": "comparison_only_pending_issue_203",
        "holdout_used": False,
        "candidate_count": len(experiments),
        "production_policy": INDEX_POLICY_CORE_HARMONIC,
        "policies": comparisons,
    }


def select_best_stock_model(report: dict[str, Any]) -> WinningModel:
    """개별종목 보고서에 ADR 0007 기준으로 기록된 최종 1위를 반환한다."""

    final_selection = report.get("final_selection")
    if not isinstance(final_selection, dict):
        raise ValueError("개별종목 평가 보고서에 final_selection이 없습니다.")
    selected = final_selection.get("selected")
    if not isinstance(selected, dict):
        raise ValueError("개별종목 평가 보고서에 final_selection.selected가 없습니다.")
    required = {"combination", "model", "feature_columns"}
    missing = required - set(selected)
    if missing:
        raise ValueError(f"개별종목 1위 모델 정보가 없습니다: {sorted(missing)}")
    combination = str(selected["combination"]).strip()
    model = str(selected["model"]).strip()
    if not combination or not model:
        raise ValueError("개별종목 1위 모델의 조합 또는 모델명이 비어 있습니다.")
    primary = str(final_selection.get("primary", "accuracy_minus_training_majority_baseline"))
    value = selected.get(primary)
    return WinningModel(
        track="개별종목",
        combination=combination,
        model=model,
        feature_columns=_feature_tuple(selected["feature_columns"], "개별종목 1위 모델"),
        selection_metric=primary,
        selection_value=float(value) if value is not None else None,
    )


def load_winning_models(
    index_report_path: Path,
    stock_report_path: Path,
) -> tuple[WinningModel, WinningModel]:
    """KOSPI200과 개별종목의 현재 1위 모델을 평가 보고서에서 함께 읽는다."""

    index_report = _read_report(index_report_path)
    stock_report = _read_report(stock_report_path)
    index_sha = index_report.get("source", {}).get("index_sha256")
    stock_index_sha = stock_report.get("source", {}).get("index_sha256")
    if index_sha and stock_index_sha and index_sha != stock_index_sha:
        raise ValueError("KOSPI200과 개별종목 평가 보고서의 지수 데이터 SHA-256이 다릅니다.")
    index_winner = (
        select_best_long_only_index_model(index_report)
        if "candidates" in index_report
        else select_best_index_model(index_report)
    )
    return (
        index_winner,
        select_best_stock_model(stock_report),
    )


__all__ = [
    "INDEX_POLICY_ACCURACY_MACRO_HARMONIC",
    "INDEX_POLICY_ACCURACY_THEN_MACRO",
    "INDEX_POLICY_CORE_HARMONIC",
    "INDEX_POLICY_DELTA_REPORTED_MAJORITY",
    "INDEX_POLICY_LONG_ONLY",
    "INDEX_SELECTION_POLICIES",
    "WinningModel",
    "compare_index_model_selection_policies",
    "load_winning_models",
    "select_best_index_model",
    "select_best_long_only_index_model",
    "select_best_stock_model",
]
