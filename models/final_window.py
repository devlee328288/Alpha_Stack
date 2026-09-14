"""개봉 자료의 마지막 T+1→T+6 한 구간을 누수 없이 분리한다."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from features.model_dataset import LABEL_HORIZON


@dataclass(frozen=True)
class FinalWindow:
    """원본의 마지막 거래일을 청산일로 삼은 최종 평가 구간."""

    decision_date: str
    entry_date: str
    exit_date: str


def derive_final_window(common_dates: list[str], *, horizon: int = LABEL_HORIZON) -> FinalWindow:
    """마지막 공통 거래일에서 거래일 기준 6칸 전을 판단일로 정한다."""

    dates = sorted(set(str(value) for value in common_dates))
    future_steps = horizon + 1
    if len(dates) <= future_steps:
        raise ValueError(f"최종 T+1→T+6 구간에는 거래일이 최소 {future_steps + 1}개 필요합니다.")
    return FinalWindow(
        decision_date=dates[-(future_steps + 1)],
        entry_date=dates[-future_steps],
        exit_date=dates[-1],
    )


def split_index_window(
    frame: pd.DataFrame,
    window: FinalWindow,
    *,
    horizon: int = LABEL_HORIZON,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """판단일 이후 가격을 쓰는 라벨을 학습에서 빼고 판단일 한 행을 남긴다."""

    required = {"bas_dd", "raw_position", "label_numeric"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"KOSPI200 최종 구간 분리에 필요한 열이 없습니다: {sorted(missing)}")
    normalized = frame.copy()
    normalized["bas_dd"] = normalized["bas_dd"].astype(str)
    test = normalized.loc[normalized["bas_dd"].eq(window.decision_date)].copy()
    if len(test) != 1:
        raise ValueError(f"KOSPI200 판단일 행은 하나여야 합니다: {window.decision_date}")
    decision_position = int(test["raw_position"].iloc[0])
    # 한 학습 행의 라벨은 raw_position + 6의 시가를 사용한다. 그 청산 시점이
    # 판단일을 넘는 행은 최종 정답 구간을 미리 본 것이므로 fit에서 제외한다.
    train = normalized.loc[
        normalized["raw_position"].astype(int) + horizon + 1 <= decision_position
    ].copy()
    if train.empty:
        raise ValueError("KOSPI200 최종 학습구간이 비었습니다.")
    return train, test


def split_stock_window(
    frame: pd.DataFrame,
    window: FinalWindow,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """청산일이 판단일 이하인 종목 행만 학습하고 판단일 후보만 평가한다."""

    required = {"bas_dd", "code", "exit_bas_dd", "label_numeric"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"개별종목 최종 구간 분리에 필요한 열이 없습니다: {sorted(missing)}")
    normalized = frame.copy()
    normalized["bas_dd"] = normalized["bas_dd"].astype(str)
    normalized["exit_bas_dd"] = normalized["exit_bas_dd"].astype(str)
    train = normalized.loc[normalized["exit_bas_dd"].le(window.decision_date)].copy()
    test = normalized.loc[normalized["bas_dd"].eq(window.decision_date)].copy()
    if train.empty:
        raise ValueError("개별종목 최종 학습구간이 비었습니다.")
    if test.empty:
        raise ValueError(f"개별종목 판단일 후보가 없습니다: {window.decision_date}")
    return train, test


__all__ = [
    "FinalWindow",
    "derive_final_window",
    "split_index_window",
    "split_stock_window",
]
