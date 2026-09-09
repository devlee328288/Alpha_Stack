"""개별종목 3분류 확률을 날짜별 랭킹으로 바꾼다."""

from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.walk_forward import expanding_splits

PROBABILITY_COLUMNS = {-1: "p_down", 0: "p_neutral", 1: "p_up"}


def build_common_validation_schedule(
    first_dates: object,
    second_dates: object,
    *,
    n_folds: int = 12,
    minimum_train_dates: int = 750,
    valid_dates: int = 60,
    gap_dates: int = 5,
) -> pd.DataFrame:
    """두 모델이 모두 가진 거래일에서 동일한 expanding OOS 일정을 만든다."""

    left = pd.Index(pd.Series(first_dates, dtype="string").drop_duplicates())
    right = pd.Index(pd.Series(second_dates, dtype="string").drop_duplicates())
    if not left.is_monotonic_increasing or not right.is_monotonic_increasing:
        raise ValueError("두 모델 날짜는 과거부터 미래 순서여야 합니다.")
    common = left.intersection(right, sort=False)
    if not common.is_monotonic_increasing:
        common = common.sort_values()
    splits = expanding_splits(
        len(common),
        n_folds=n_folds,
        min_train=minimum_train_dates,
        horizon=valid_dates,
        gap=gap_dates,
        label_horizon=gap_dates,
    )
    rows = [
        {"fold": fold, "bas_dd": str(common[position])}
        for fold, (_train, valid) in enumerate(splits, start=1)
        for position in valid
    ]
    return pd.DataFrame(rows)


def aligned_index_splits(
    index_dates: object,
    stock_validation_schedule: pd.DataFrame,
    *,
    gap_dates: int = 5,
    minimum_train_dates: int = 750,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """개별종목 OOS 날짜와 정확히 같은 날짜로 지수 expanding 분할을 만든다."""

    if gap_dates < 0 or minimum_train_dates <= 0:
        raise ValueError("갭은 0 이상이고 최소 학습일은 양수여야 합니다.")
    dates = pd.Index(pd.Series(index_dates, dtype="string"))
    if dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("KOSPI200 날짜는 결측·중복 없는 오름차순이어야 합니다.")
    missing = {"fold", "bas_dd"} - set(stock_validation_schedule.columns)
    if missing:
        raise ValueError(f"개별종목 검증 일정 열이 없습니다: {sorted(missing)}")

    schedule = stock_validation_schedule.loc[:, ["fold", "bas_dd"]].copy()
    schedule["bas_dd"] = schedule["bas_dd"].astype("string")
    schedule = schedule.drop_duplicates()
    if schedule["bas_dd"].duplicated().any():
        raise ValueError("같은 검증 날짜가 둘 이상의 폴드에 들어 있습니다.")
    if schedule.empty:
        raise ValueError("개별종목 검증 일정이 비어 있습니다.")

    positions = pd.Series(np.arange(len(dates), dtype=int), index=dates)
    unknown_dates = sorted(set(schedule["bas_dd"]) - set(dates))
    if unknown_dates:
        raise ValueError(f"KOSPI200에 없는 검증 날짜가 있습니다: {unknown_dates[:5]}")

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for _fold, group in schedule.groupby("fold", sort=True):
        valid_dates = group["bas_dd"].sort_values(kind="stable")
        valid = positions.loc[valid_dates].to_numpy(dtype=int)
        if len(valid) == 0 or (len(valid) > 1 and not np.all(np.diff(valid) == 1)):
            raise ValueError("한 폴드의 검증 날짜가 KOSPI200 거래일에서 연속하지 않습니다.")
        train_end = int(valid[0]) - gap_dates
        if train_end < minimum_train_dates:
            raise ValueError("날짜 정렬 분할의 최초 학습 거래일이 부족합니다.")
        splits.append((np.arange(train_end, dtype=int), valid))
    return splits


def aligned_panel_splits(
    panel_dates: object,
    validation_schedule: pd.DataFrame,
    *,
    gap_dates: int = 5,
    minimum_train_dates: int = 750,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """반복된 날짜의 모든 종목 행을 같은 학습·검증 구간으로 묶는다."""

    groups = pd.Series(panel_dates, dtype="string")
    if groups.isna().any() or not groups.is_monotonic_increasing:
        raise ValueError("개별종목 패널 날짜는 결측 없이 오름차순이어야 합니다.")
    codes, unique_dates = pd.factorize(groups, sort=False)
    starts = np.concatenate(([0], np.flatnonzero(codes[1:] != codes[:-1]) + 1))
    if len(starts) != len(unique_dates):
        raise ValueError("같은 개별종목 패널 날짜는 연속해서 모여 있어야 합니다.")
    ends = np.concatenate((starts[1:], [len(groups)]))
    group_splits = aligned_index_splits(
        unique_dates,
        validation_schedule,
        gap_dates=gap_dates,
        minimum_train_dates=minimum_train_dates,
    )
    return [
        (
            np.arange(starts[train[0]], ends[train[-1]], dtype=int),
            np.arange(starts[valid[0]], ends[valid[-1]], dtype=int),
        )
        for train, valid in group_splits
    ]


def add_probability_ranks(predictions: pd.DataFrame) -> pd.DataFrame:
    """모델·날짜마다 하락·중립·상승 확률 순위를 모두 남긴다."""

    required = {"model", "bas_dd", "code", *PROBABILITY_COLUMNS.values()}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"종목 확률 랭킹 열이 없습니다: {sorted(missing)}")
    if predictions.duplicated(["model", "bas_dd", "code"]).any():
        raise ValueError("같은 모델·날짜·종목의 OOS 확률이 두 번 이상 있습니다.")

    out = predictions.copy()
    probabilities = out.loc[:, list(PROBABILITY_COLUMNS.values())].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all():
        raise ValueError("종목 확률에 결측 또는 무한대가 있습니다.")
    if (probabilities < 0.0).any() or (probabilities > 1.0).any():
        raise ValueError("종목 확률이 0~1 범위를 벗어났습니다.")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("종목 확률 세 칸의 합이 1이 아닙니다.")

    groups = ["model", "bas_dd"]
    for label, probability_column in PROBABILITY_COLUMNS.items():
        suffix = {-1: "down", 0: "neutral", 1: "up"}[label]
        out[f"{suffix}_rank"] = (
            out.groupby(groups, sort=False)[probability_column]
            .rank(method="first", ascending=False)
            .astype("int16")
        )
        if "industry_index_name" in out.columns:
            out[f"{suffix}_sector_rank"] = (
                out.groupby([*groups, "industry_index_name"], sort=False)[
                    probability_column
                ]
                .rank(method="first", ascending=False)
                .astype("int16")
            )
    return out


def select_for_index_direction(
    stock_predictions: pd.DataFrame,
    index_predictions: pd.DataFrame,
    *,
    top_n: int = 5,
) -> pd.DataFrame:
    """KOSPI200 예측 클래스와 같은 종목 확률을 골라 상위 N개를 남긴다."""

    if top_n <= 0:
        raise ValueError("top_n은 1 이상이어야 합니다.")
    missing_index = {"bas_dd", "predicted"} - set(index_predictions.columns)
    if missing_index:
        raise ValueError(f"KOSPI200 예측 열이 없습니다: {sorted(missing_index)}")
    if index_predictions["bas_dd"].duplicated().any():
        raise ValueError("같은 날짜의 KOSPI200 예측이 두 번 이상 있습니다.")
    unknown = set(index_predictions["predicted"].astype(int)) - set(PROBABILITY_COLUMNS)
    if unknown:
        raise ValueError(f"KOSPI200 예측에 -1·0·1 외 값이 있습니다: {sorted(unknown)}")

    ranked = add_probability_ranks(stock_predictions)
    index_frame = index_predictions.loc[:, ["bas_dd", "predicted"]].rename(
        columns={"predicted": "index_predicted"}
    )
    combined = ranked.merge(index_frame, on="bas_dd", how="inner", validate="many_to_one")
    if combined.empty:
        raise ValueError("KOSPI200과 개별종목 OOS 예측 날짜가 겹치지 않습니다.")

    combined["selected_probability"] = np.select(
        [
            combined["index_predicted"].eq(-1),
            combined["index_predicted"].eq(0),
            combined["index_predicted"].eq(1),
        ],
        [combined["p_down"], combined["p_neutral"], combined["p_up"]],
        default=np.nan,
    )
    combined = combined.sort_values(
        ["model", "bas_dd", "selected_probability", "code"],
        ascending=[True, True, False, True],
        kind="stable",
    )
    combined["index_direction_rank"] = (
        combined.groupby(["model", "bas_dd"], sort=False).cumcount() + 1
    ).astype("int16")
    return combined.loc[combined["index_direction_rank"] <= top_n].reset_index(drop=True)


def summarize_direction_ranking(
    stock_predictions: pd.DataFrame,
    index_predictions: pd.DataFrame,
    *,
    cutoffs: tuple[int, ...] = (1, 3, 5),
) -> pd.DataFrame:
    """KOSPI200 예측 방향과 실제 종목 라벨이 맞는 비율을 순위 구간별로 계산한다."""

    if not cutoffs or any(cutoff <= 0 for cutoff in cutoffs):
        raise ValueError("랭킹 평가 구간은 하나 이상의 양수여야 합니다.")
    if len(set(cutoffs)) != len(cutoffs):
        raise ValueError("랭킹 평가 구간이 중복되었습니다.")
    missing = {"label_numeric"} - set(stock_predictions.columns)
    if missing:
        raise ValueError(f"종목 랭킹 평가 열이 없습니다: {sorted(missing)}")

    ranked = select_for_index_direction(
        stock_predictions,
        index_predictions,
        top_n=max(cutoffs),
    )
    rows: list[dict[str, object]] = []
    for cutoff in sorted(cutoffs):
        selected = ranked.loc[ranked["index_direction_rank"] <= cutoff].copy()
        selected["direction_hit"] = selected["label_numeric"].eq(
            selected["index_predicted"]
        )
        rows.append(
            {
                "top_n": cutoff,
                "rows": int(len(selected)),
                "dates": int(selected["bas_dd"].nunique()),
                "direction_hit_rate": float(selected["direction_hit"].mean()),
                "mean_selected_probability": float(
                    selected["selected_probability"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_random_ranking_baseline(
    stock_predictions: pd.DataFrame,
    index_predictions: pd.DataFrame,
    *,
    cutoffs: tuple[int, ...] = (1, 3, 5),
) -> pd.DataFrame:
    """공통 OOS의 날짜·지수 예측 방향 안에서 무작위 Top-K의 기대 적중률을 계산한다."""

    if not cutoffs or any(cutoff <= 0 for cutoff in cutoffs):
        raise ValueError("무작위 기준선의 Top-K는 하나 이상의 양수여야 합니다.")
    if len(set(cutoffs)) != len(cutoffs):
        raise ValueError("무작위 기준선의 Top-K가 중복되었습니다.")
    required = {"model", "bas_dd", "code", "label_numeric"}
    missing = required - set(stock_predictions.columns)
    if missing:
        raise ValueError(f"무작위 랭킹 기준선 입력 열이 없습니다: {sorted(missing)}")
    if stock_predictions.duplicated(["model", "bas_dd", "code"]).any():
        raise ValueError("같은 모델·날짜·종목의 OOS 예측이 중복되었습니다.")
    if index_predictions["bas_dd"].duplicated().any():
        raise ValueError("같은 날짜의 KOSPI200 예측이 중복되었습니다.")

    index_frame = index_predictions.loc[:, ["bas_dd", "predicted"]].rename(
        columns={"predicted": "index_predicted"}
    )
    merged = stock_predictions.loc[:, list(required)].merge(
        index_frame,
        on="bas_dd",
        how="inner",
        validate="many_to_one",
    )
    if merged.empty:
        raise ValueError("KOSPI200과 개별종목의 공통 OOS 날짜가 없습니다.")
    merged["direction_hit"] = merged["label_numeric"].eq(merged["index_predicted"])
    dates = (
        merged.groupby(["model", "bas_dd", "index_predicted"], sort=False)
        .agg(candidate_rows=("code", "size"), matching_rows=("direction_hit", "sum"))
        .reset_index()
    )

    rows: list[dict[str, object]] = []
    for model, model_dates in dates.groupby("model", sort=False):
        scopes: list[tuple[int | None, pd.DataFrame]] = [(None, model_dates)]
        scopes.extend(
            (int(direction), group)
            for direction, group in model_dates.groupby("index_predicted", sort=True)
        )
        for direction, scope in scopes:
            for cutoff in sorted(cutoffs):
                selected_rows = np.minimum(scope["candidate_rows"].to_numpy(), cutoff)
                expected_hits = selected_rows * (
                    scope["matching_rows"].to_numpy()
                    / scope["candidate_rows"].to_numpy()
                )
                rows.append(
                    {
                        "model": model,
                        "top_n": cutoff,
                        "index_predicted": direction,
                        "dates": int(len(scope)),
                        "expected_selected_rows": int(selected_rows.sum()),
                        "random_direction_hit_rate": float(
                            expected_hits.sum() / selected_rows.sum()
                        ),
                    }
                )
    return pd.DataFrame(rows)


__all__ = [
    "add_probability_ranks",
    "aligned_index_splits",
    "aligned_panel_splits",
    "build_common_validation_schedule",
    "select_for_index_direction",
    "summarize_direction_ranking",
    "summarize_random_ranking_baseline",
]
