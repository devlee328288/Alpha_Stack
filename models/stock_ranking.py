"""개별종목 3분류 확률을 날짜별 랭킹으로 바꾼다."""

from __future__ import annotations

import numpy as np
import pandas as pd

PROBABILITY_COLUMNS = {-1: "p_down", 0: "p_neutral", 1: "p_up"}


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


__all__ = ["add_probability_ranks", "select_for_index_direction"]
