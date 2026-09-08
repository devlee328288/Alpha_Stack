"""같은 외부 폴드에서 여러 모델을 기준선과 비교하는 다중검정 도구."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """가족오류율을 통제하는 Holm 보정 p-value를 원래 순서로 돌려준다."""

    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Holm 보정에는 하나 이상의 1차원 p-value가 필요합니다.")
    if not np.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("p-value는 0 이상 1 이하의 유한한 값이어야 합니다.")

    order = np.argsort(values, kind="stable")
    ranked = values[order]
    factors = np.arange(values.size, 0, -1, dtype=float)
    adjusted_ranked = np.minimum(1.0, np.maximum.accumulate(ranked * factors))
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted


def compare_accuracy_to_baseline(
    fold_results: pd.DataFrame,
    *,
    alpha: float = 0.05,
) -> dict[str, object]:
    """각 조합·모델의 폴드 Accuracy가 학습 최빈 기준선보다 큰지 검정한다.

    각 후보의 12개 폴드 차이에 단측 Wilcoxon signed-rank 검정을 적용하고, 후보 전체를
    하나의 검정군으로 묶어 Holm 보정한다. expanding 학습창은 서로 겹치므로 p-value를
    완전히 독립된 반복실험처럼 해석할 수 없다는 한계를 보고서에 명시한다.
    """

    required = {
        "combination",
        "model",
        "fold",
        "accuracy",
        "training_majority_baseline_accuracy",
    }
    missing = required - set(fold_results.columns)
    if missing:
        raise ValueError(f"다중비교 입력 열이 없습니다: {sorted(missing)}")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha는 0과 1 사이여야 합니다.")
    if fold_results.duplicated(["combination", "model", "fold"]).any():
        raise ValueError("조합·모델·폴드 결과가 중복되었습니다.")

    rows: list[dict[str, object]] = []
    for (combination, model), group in fold_results.groupby(
        ["combination", "model"], sort=True
    ):
        ordered = group.sort_values("fold")
        delta = (
            ordered["accuracy"].to_numpy(dtype=float)
            - ordered["training_majority_baseline_accuracy"].to_numpy(dtype=float)
        )
        if len(delta) < 2 or not np.isfinite(delta).all():
            raise ValueError(f"조합{combination} {model}의 폴드 차이를 검정할 수 없습니다.")

        nonzero = int(np.count_nonzero(delta))
        if nonzero == 0:
            statistic = 0.0
            p_value = 1.0
        else:
            result = wilcoxon(
                delta,
                alternative="greater",
                zero_method="pratt",
                method="auto",
            )
            statistic = float(result.statistic)
            p_value = float(result.pvalue)
        rows.append(
            {
                "combination": str(combination),
                "model": str(model),
                "folds": int(len(delta)),
                "positive_folds": int((delta > 0.0).sum()),
                "negative_folds": int((delta < 0.0).sum()),
                "tie_folds": int((delta == 0.0).sum()),
                "mean_accuracy_delta": float(delta.mean()),
                "wilcoxon_statistic": statistic,
                "raw_p_value": p_value,
            }
        )

    adjusted = holm_adjust([float(row["raw_p_value"]) for row in rows])
    for row, p_value in zip(rows, adjusted, strict=True):
        row["holm_adjusted_p_value"] = float(p_value)
        row["significant_after_holm"] = bool(p_value < alpha)
    rows.sort(
        key=lambda row: (
            float(row["holm_adjusted_p_value"]),
            -float(row["mean_accuracy_delta"]),
            str(row["combination"]),
            str(row["model"]),
        )
    )

    significant = sum(bool(row["significant_after_holm"]) for row in rows)
    return {
        "family": "A~H 8개 조합 × 4개 모델의 학습 최빈 기준선 대비 Accuracy",
        "hypothesis": "각 후보의 폴드 Accuracy 차이 중앙값이 0보다 크다",
        "test": "one-sided Wilcoxon signed-rank",
        "correction": "Holm family-wise error rate",
        "alpha": alpha,
        "comparisons": len(rows),
        "significant_after_correction": int(significant),
        "results": rows,
        "limitations": [
            "표본이 후보당 12폴드뿐이어서 검정력이 제한된다.",
            "expanding 학습창이 서로 겹치므로 폴드 차이가 완전히 독립이라는 보장은 없다.",
            "통계적 유의성은 수익성과 같지 않으며 최종 선정 규칙을 대체하지 않는다.",
        ],
    }


__all__ = ["compare_accuracy_to_baseline", "holm_adjust"]
