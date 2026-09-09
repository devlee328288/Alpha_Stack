"""개별 종목 패널의 날짜 그룹 walk-forward 분할을 검증한다."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.walk_forward import expanding_group_splits


def _panel_dates(n_dates: int, rows_per_date: int) -> np.ndarray:
    dates = pd.bdate_range("2020-01-02", periods=n_dates)
    return np.repeat(dates.to_numpy(), rows_per_date)


def test_12폴드마다_검증60거래일과_gap5거래일을_지킨다():
    groups = _panel_dates(n_dates=1600, rows_per_date=50)

    splits = expanding_group_splits(
        groups,
        n_folds=12,
        min_train=750,
        horizon=60,
        label_horizon=5,
    )

    assert len(splits) == 12
    all_dates = pd.Index(groups).unique()
    for train_idx, valid_idx in splits:
        train_dates = pd.Index(groups[train_idx]).unique()
        valid_dates = pd.Index(groups[valid_idx]).unique()

        assert len(train_dates) >= 750
        assert len(valid_dates) == 60
        assert train_dates.intersection(valid_dates).empty

        train_end = all_dates.get_loc(train_dates[-1])
        valid_start = all_dates.get_loc(valid_dates[0])
        assert valid_start - train_end - 1 == 5


def test_같은_거래일의_모든_종목은_같은_구간에_남는다():
    groups = _panel_dates(n_dates=40, rows_per_date=7)
    splits = expanding_group_splits(
        groups,
        n_folds=3,
        min_train=15,
        horizon=5,
        label_horizon=2,
    )

    for train_idx, valid_idx in splits:
        train_dates = set(groups[train_idx])
        valid_dates = set(groups[valid_idx])
        assert train_dates.isdisjoint(valid_dates)
        assert len(train_idx) % 7 == 0
        assert len(valid_idx) == 5 * 7


def test_날짜별_종목수가_달라도_전체_날짜행을_보존한다():
    dates = pd.bdate_range("2024-01-02", periods=18).to_numpy()
    counts = np.arange(1, 19)
    groups = np.repeat(dates, counts)

    splits = expanding_group_splits(
        groups,
        n_folds=2,
        min_train=8,
        horizon=3,
        label_horizon=2,
    )

    for _, valid_idx in splits:
        for date in np.unique(groups[valid_idx]):
            assert np.count_nonzero(groups[valid_idx] == date) == np.count_nonzero(groups == date)


@pytest.mark.parametrize(
    "groups, message",
    [
        (["2024-01-02", "2024-01-03", "2024-01-02"], "연속"),
        (["2024-01-03", "2024-01-02", "2024-01-01"], "정렬"),
        (["2024-01-02", None, "2024-01-03"], "결측"),
    ],
)
def test_잘못된_그룹배열은_조용히_분할하지_않는다(groups, message):
    with pytest.raises(ValueError, match=message):
        expanding_group_splits(
            groups,
            n_folds=1,
            min_train=1,
            horizon=1,
            label_horizon=0,
        )


def test_2차원_그룹배열을_거부한다():
    with pytest.raises(ValueError, match="1차원"):
        expanding_group_splits(np.array([[1, 1], [2, 2]]))
