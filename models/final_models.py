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


def select_best_index_model(report: dict[str, Any]) -> WinningModel:
    """KOSPI200 실험 중 기존 평가 기준인 조화평균 1위를 반환한다."""

    experiments = report.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("KOSPI200 평가 보고서에 experiments가 없습니다.")

    def selection_key(record: dict[str, Any]) -> tuple[float, float, float, str, str]:
        experiment = record.get("experiment", {})
        summary = record.get("summary", {})
        required = {"core_harmonic_mean", "accuracy", "macro_f1", "feature_columns"}
        missing = required - set(summary)
        if missing:
            raise ValueError(f"KOSPI200 평가 지표가 없습니다: {sorted(missing)}")
        return (
            float(summary["core_harmonic_mean"]),
            float(summary["accuracy"]),
            float(summary["macro_f1"]),
            str(experiment.get("combination", "")),
            str(experiment.get("model", "")),
        )

    winner = max(experiments, key=selection_key)
    experiment = winner["experiment"]
    summary = winner["summary"]
    combination = str(experiment.get("combination", ""))
    model = str(experiment.get("model", ""))
    if not combination or not model:
        raise ValueError("KOSPI200 1위 모델의 조합 또는 모델명이 비어 있습니다.")
    return WinningModel(
        track="KOSPI200",
        combination=combination,
        model=model,
        feature_columns=tuple(str(value) for value in summary["feature_columns"]),
        return_features=tuple(
            str(value) for value in experiment.get("return_features", [])
        ),
        selection_metric="core_harmonic_mean",
        selection_value=float(summary["core_harmonic_mean"]),
    )


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
    primary = str(final_selection.get("primary", "accuracy_minus_training_majority_baseline"))
    value = selected.get(primary)
    return WinningModel(
        track="개별종목",
        combination=str(selected["combination"]),
        model=str(selected["model"]),
        feature_columns=tuple(str(item) for item in selected["feature_columns"]),
        selection_metric=primary,
        selection_value=float(value) if value is not None else None,
    )


def load_winning_models(
    index_report_path: Path,
    stock_report_path: Path,
) -> tuple[WinningModel, WinningModel]:
    """KOSPI200과 개별종목의 현재 1위 모델을 평가 보고서에서 함께 읽는다."""

    return (
        select_best_index_model(_read_report(index_report_path)),
        select_best_stock_model(_read_report(stock_report_path)),
    )


__all__ = [
    "WinningModel",
    "load_winning_models",
    "select_best_index_model",
    "select_best_stock_model",
]
