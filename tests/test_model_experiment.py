import numpy as np
import pandas as pd
import pytest

from evaluation.overlapping import OverlappingResult
from models.experiment import (
    MODEL_BUILDERS,
    N_FOLDS,
    WINDOW_NAMES,
    classification_metrics,
    common_window_splits,
    evaluate_nested_class_weights,
    inner_class_weight_split,
    summarize_window_results,
    window_selection_sharpe,
)


def test_전기간_현금이면_윈도평가에서_전략_sharpe를_0으로_둔다():
    result = OverlappingResult(
        strategy_net=np.zeros(10),
        buyhold_net=np.array([0.01, -0.005] * 5),
        strategy_gross=np.zeros(10),
        buyhold_gross=np.array([0.01, -0.005] * 5),
        first_entry_position=1,
        last_exit_position=11,
    )

    delta, strategy_sharpe, buyhold_sharpe, all_cash = window_selection_sharpe(result)

    assert strategy_sharpe == 0.0
    assert delta == -buyhold_sharpe
    assert all_cash is True


def test_내부_가중치_검증은_마지막_60일과_직전_5일을_분리한다():
    inner_train, inner_valid = inner_class_weight_split(np.arange(750))

    np.testing.assert_array_equal(inner_train, np.arange(685))
    np.testing.assert_array_equal(inner_valid, np.arange(690, 750))
    assert inner_valid[0] - inner_train[-1] - 1 == 5


def test_핵심지표_중_하나가_0이면_조화평균도_0이다():
    actual = np.array([-1, 0, 1, -1, 0, 1])
    predicted = np.array([0, 0, 1, 0, 0, 1])

    metrics = classification_metrics(actual, predicted)

    assert metrics["down_recall"] == 0.0
    assert metrics["core_harmonic_mean"] == 0.0


def test_알_수_없는_모델만_요청하면_학습_전에_막는다():
    with pytest.raises(ValueError, match="알 수 없는 모델"):
        # 모델 이름 검사는 데이터 접근보다 먼저 일어나야 하므로 최소 모형만 전달합니다.
        from features.model_dataset import ModelDataset

        empty = ModelDataset(
            frame=pd.DataFrame(),
            raw_prices=pd.DataFrame(),
            feature_columns=(),
            combination="A",
        )
        evaluate_nested_class_weights(empty, model_names=("없는모델",))


def test_학습창_후보는_모두_같은_검증_인덱스를_쓴다():
    splits = common_window_splits(3400)
    reference = splits["expanding"]

    assert all(len(candidate) == N_FOLDS for candidate in splits.values())
    for candidate in splits.values():
        for (_, expected_valid), (_, actual_valid) in zip(reference, candidate, strict=True):
            np.testing.assert_array_equal(actual_valid, expected_valid)
    assert {len(train) for train, _ in splits["rolling_750"]} == {750}
    assert {len(train) for train, _ in splits["rolling_1250"]} == {1250}
    assert {len(train) for train, _ in splits["rolling_2000"]} == {2000}


def test_학습창_중앙값이_동률이면_expanding을_고른다():
    rows = []
    for window in WINDOW_NAMES:
        for model in MODEL_BUILDERS:
            for fold in range(1, N_FOLDS + 1):
                rows.append(
                    {
                        "window": window,
                        "model": model,
                        "fold": fold,
                        "delta_sharpe_net": 1.0,
                        "accuracy": 0.4,
                        "macro_f1": 0.3,
                        "down_recall": 0.2,
                    }
                )

    _, selected = summarize_window_results(pd.DataFrame(rows))

    assert selected == "expanding"
