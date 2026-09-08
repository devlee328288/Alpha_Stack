"""개발구간에서 최종 분류 모델 하나를 고르는 사전등록 규칙."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

PRIMARY_METRIC = "accuracy_minus_training_majority_baseline"
SECONDARY_METRIC = "macro_f1"
TERTIARY_METRIC = "baseline_win_folds"


def final_model_selection_key(candidate: Mapping[str, object]) -> tuple[float, float, int]:
    """후보를 비교할 세 지표를 우선순위 순으로 돌려준다.

    현재 포지션 정책은 상승만 매수하고 중립·하락은 현금이므로 하락 Recall을 최종
    선택 축에 넣지 않는다. 이름으로 동률을 푸는 일은 정렬의 재현성만 위한 것이며,
    성능 우위로 해석하지 않는다.
    """

    missing = {
        PRIMARY_METRIC,
        SECONDARY_METRIC,
        TERTIARY_METRIC,
    } - set(candidate)
    if missing:
        raise ValueError(f"최종 모델 선정 지표가 없습니다: {sorted(missing)}")
    return (
        float(candidate[PRIMARY_METRIC]),
        float(candidate[SECONDARY_METRIC]),
        int(candidate[TERTIARY_METRIC]),
    )


def rank_final_model_candidates(
    candidates: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    """사전등록한 지표로 후보를 정렬하고 1부터 순위를 붙인다."""

    rows = [dict(candidate) for candidate in candidates]
    if not rows:
        raise ValueError("최종 모델 후보가 비어 있습니다.")
    for row in rows:
        final_model_selection_key(row)
    rows.sort(
        key=lambda row: (
            -final_model_selection_key(row)[0],
            -final_model_selection_key(row)[1],
            -final_model_selection_key(row)[2],
            str(row.get("combination", "")),
            str(row.get("model", "")),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["final_selection_rank"] = rank
    return rows


__all__ = [
    "PRIMARY_METRIC",
    "SECONDARY_METRIC",
    "TERTIARY_METRIC",
    "final_model_selection_key",
    "rank_final_model_candidates",
]
