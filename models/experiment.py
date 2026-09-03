"""A~F 모델 실험에서 공유하는 분할·학습·평가 함수."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score

from evaluation.metrics import sharpe_ratio
from evaluation.overlapping import OverlappingResult, overlapping_long_only_returns
from evaluation.walk_forward import expanding_splits
from features.model_dataset import ModelDataset
from models.lightgbm import build_lightgbm_baseline
from models.logistic import build_logistic_baseline
from models.random_forest import build_random_forest_baseline
from models.xgboost import build_xgboost_baseline

N_FOLDS = 12
MIN_TRAIN_SIZE = 750
VALID_SIZE = 60
LABEL_HORIZON = 5
ROUND_TRIP_COST = 0.0005
ROLLING_WINDOWS = (750, 1250, 2000)
WINDOW_NAMES = ("expanding", "rolling_750", "rolling_1250", "rolling_2000")
CLASS_WEIGHT_CANDIDATES = (None, "balanced")

ModelBuilder = Callable[..., object]
MODEL_BUILDERS: dict[str, ModelBuilder] = {
    "LogisticRegression": build_logistic_baseline,
    "RandomForest": build_random_forest_baseline,
    "XGBoost": build_xgboost_baseline,
    "LightGBM": build_lightgbm_baseline,
}


@dataclass(frozen=True)
class NestedWeightResult:
    """내부 가중치 비교와 선택한 가중치의 외부 OOS 결과."""

    inner_results: pd.DataFrame
    outer_results: pd.DataFrame
    oos_predictions: pd.DataFrame


def classification_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """세 핵심 분류 지표와 그 조화평균을 같은 규칙으로 계산한다."""

    accuracy = float(accuracy_score(actual, predicted))
    macro_f1 = float(
        f1_score(actual, predicted, labels=[-1, 0, 1], average="macro", zero_division=0)
    )
    down_recall = float(
        recall_score(actual, predicted, labels=[-1], average="macro", zero_division=0)
    )
    values = np.asarray([accuracy, macro_f1, down_recall], dtype=float)
    # 어느 한 지표라도 0이면 조화평균도 0이다. 작은 수를 더해 0점을 살려내지 않는다.
    harmonic = 0.0 if np.any(values == 0.0) else float(len(values) / np.sum(1.0 / values))
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "down_recall": down_recall,
        "core_harmonic_mean": harmonic,
    }


def inner_class_weight_split(
    outer_train_index: np.ndarray,
    *,
    valid_size: int = VALID_SIZE,
    gap: int = LABEL_HORIZON,
) -> tuple[np.ndarray, np.ndarray]:
    """외부 학습 끝 60일을 내부 검증으로 떼고 그 직전 5일을 버린다."""

    indices = np.asarray(outer_train_index, dtype=int)
    if indices.ndim != 1 or indices.size < valid_size + gap + 1:
        raise ValueError("내부 학습·5일 갭·60일 검증을 만들 표본이 부족합니다.")
    if np.any(np.diff(indices) <= 0):
        raise ValueError("외부 학습 인덱스는 중복 없는 오름차순이어야 합니다.")
    inner_train = indices[: -(valid_size + gap)]
    inner_valid = indices[-valid_size:]
    actual_gap = int(inner_valid[0]) - int(inner_train[-1]) - 1
    if actual_gap != gap:
        raise ValueError(f"내부 검증 갭이 {gap}거래일이 아닙니다: {actual_gap}")
    return inner_train, inner_valid


def evaluate_nested_class_weights(
    dataset: ModelDataset,
    *,
    model_names: tuple[str, ...] | None = None,
) -> NestedWeightResult:
    """각 외부 폴드 안에서 기본·balanced를 고른 뒤 OOS를 평가한다."""

    selected_builders = MODEL_BUILDERS
    if model_names is not None:
        unknown = set(model_names) - set(MODEL_BUILDERS)
        if unknown:
            raise ValueError(f"알 수 없는 모델입니다: {sorted(unknown)}")
        selected_builders = {name: MODEL_BUILDERS[name] for name in model_names}
    outer_splits = expanding_splits(
        n_samples=len(dataset.frame),
        n_folds=N_FOLDS,
        min_train=MIN_TRAIN_SIZE,
        horizon=VALID_SIZE,
        gap=LABEL_HORIZON,
        label_horizon=LABEL_HORIZON,
    )
    inner_rows: list[dict[str, object]] = []
    outer_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for model_name, builder in selected_builders.items():
        for fold, (outer_train, outer_valid) in enumerate(outer_splits, start=1):
            inner_train, inner_valid = inner_class_weight_split(outer_train)
            candidate_rows = []
            for class_weight in CLASS_WEIGHT_CANDIDATES:
                model = builder(class_weight=class_weight)
                model.fit(dataset.x.iloc[inner_train], dataset.y[inner_train])
                predicted = np.asarray(model.predict(dataset.x.iloc[inner_valid]), dtype=int)
                metrics = classification_metrics(dataset.y[inner_valid], predicted)
                row = {
                    "model": model_name,
                    "fold": fold,
                    "class_weight": class_weight,
                    "inner_train_size": len(inner_train),
                    "inner_valid_size": len(inner_valid),
                    "inner_train_end": dataset.frame.iloc[inner_train[-1]]["bas_dd"],
                    "inner_valid_start": dataset.frame.iloc[inner_valid[0]]["bas_dd"],
                    "inner_valid_end": dataset.frame.iloc[inner_valid[-1]]["bas_dd"],
                    **metrics,
                }
                candidate_rows.append(row)
                inner_rows.append(row)

            # 점수가 같으면 추가 가정을 하지 않는 기본값(None)을 고른다.
            best = max(
                candidate_rows,
                key=lambda row: (
                    float(row["core_harmonic_mean"]),
                    row["class_weight"] is None,
                ),
            )
            selected_weight = best["class_weight"]
            final_model = builder(class_weight=selected_weight)
            final_model.fit(dataset.x.iloc[outer_train], dataset.y[outer_train])
            outer_predicted = np.asarray(
                final_model.predict(dataset.x.iloc[outer_valid]), dtype=int
            )
            outer_metrics = classification_metrics(dataset.y[outer_valid], outer_predicted)
            portfolio = overlapping_long_only_returns(
                dataset.opens,
                dataset.signal_positions[outer_valid],
                outer_predicted,
                horizon=LABEL_HORIZON,
                round_trip_cost=ROUND_TRIP_COST,
            )
            delta, strategy_sharpe, buyhold_sharpe, all_cash = window_selection_sharpe(
                portfolio
            )
            outer_rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "selected_class_weight": selected_weight,
                    "train_size": len(outer_train),
                    "valid_size": len(outer_valid),
                    "train_end": dataset.frame.iloc[outer_train[-1]]["bas_dd"],
                    "valid_start": dataset.frame.iloc[outer_valid[0]]["bas_dd"],
                    "valid_end": dataset.frame.iloc[outer_valid[-1]]["bas_dd"],
                    **outer_metrics,
                    "strategy_sharpe_net": strategy_sharpe,
                    "buyhold_sharpe_net": buyhold_sharpe,
                    "delta_sharpe_net": delta,
                    "all_cash": all_cash,
                }
            )
            prediction_rows.extend(
                {
                    "model": model_name,
                    "fold": fold,
                    "bas_dd": dataset.frame.iloc[index]["bas_dd"],
                    "actual": int(actual),
                    "predicted": int(predicted),
                }
                for index, actual, predicted in zip(
                    outer_valid,
                    dataset.y[outer_valid],
                    outer_predicted,
                    strict=True,
                )
            )
    return NestedWeightResult(
        inner_results=pd.DataFrame(inner_rows),
        outer_results=pd.DataFrame(outer_rows),
        oos_predictions=pd.DataFrame(prediction_rows),
    )


def window_selection_sharpe(
    result: OverlappingResult,
) -> tuple[float, float, float, bool]:
    """윈도 선택용 전략·벤치마크 Sharpe와 차이를 계산한다.

    상승 예측이 한 번도 없으면 전략은 전 기간 현금이고 일별 수익률도 모두 0이다.
    일반 성과 함수에서는 이 경우 Sharpe가 정의되지 않지만, 후보 윈도 평가에서는
    사용자가 정한 규칙에 따라 전략 Sharpe를 0으로 둔다. 상수 수익률처럼 현금 보유가
    아닌 다른 무변동 사례까지 0으로 바꾸면 오류를 숨길 수 있으므로 그대로 실패시킨다.
    """

    strategy_sharpe = sharpe_ratio(result.strategy_net)
    buyhold_sharpe = sharpe_ratio(result.buyhold_net)
    all_cash = not np.any(result.strategy_net)

    if strategy_sharpe is None and all_cash:
        strategy_sharpe = 0.0
    if strategy_sharpe is None or buyhold_sharpe is None:
        raise RuntimeError("현금 보유가 아닌 무변동 수익률의 Sharpe를 계산할 수 없습니다.")
    return (
        strategy_sharpe - buyhold_sharpe,
        strategy_sharpe,
        buyhold_sharpe,
        all_cash,
    )


def common_window_splits(
    n_samples: int,
    *,
    n_folds: int = N_FOLDS,
    valid_size: int = VALID_SIZE,
    gap: int = LABEL_HORIZON,
    rolling_windows: tuple[int, ...] = ROLLING_WINDOWS,
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """모든 창 후보가 같은 검증 인덱스를 사용하도록 학습 인덱스만 바꾼다."""

    largest_window = max(rolling_windows)
    anchors = expanding_splits(
        n_samples=n_samples,
        n_folds=n_folds,
        min_train=largest_window,
        horizon=valid_size,
        gap=gap,
        label_horizon=gap,
    )
    result = {"expanding": anchors}
    for window in rolling_windows:
        candidate = []
        for expanding_train, valid in anchors:
            train_end = int(expanding_train[-1]) + 1
            candidate.append((np.arange(train_end - window, train_end), valid.copy()))
        result[f"rolling_{window}"] = candidate
    return result


def evaluate_window_candidates(
    dataset: ModelDataset,
    *,
    class_weight: str | None = None,
) -> pd.DataFrame:
    """대표 데이터에서 네 모델과 네 창 후보의 폴드별 성과를 계산한다."""

    splits_by_window = common_window_splits(len(dataset.frame))
    rows = []
    for window_name in WINDOW_NAMES:
        splits = splits_by_window[window_name]
        for model_name, builder in MODEL_BUILDERS.items():
            for fold, (train_index, valid_index) in enumerate(splits, start=1):
                model = builder(class_weight=class_weight)
                model.fit(dataset.x.iloc[train_index], dataset.y[train_index])
                predicted = np.asarray(model.predict(dataset.x.iloc[valid_index]), dtype=int)
                actual = dataset.y[valid_index]
                portfolio = overlapping_long_only_returns(
                    dataset.opens,
                    dataset.signal_positions[valid_index],
                    predicted,
                    horizon=LABEL_HORIZON,
                    round_trip_cost=ROUND_TRIP_COST,
                )
                try:
                    delta, strategy_sharpe, buyhold_sharpe, all_cash = (
                        window_selection_sharpe(portfolio)
                    )
                except RuntimeError as error:
                    raise RuntimeError(
                        f"{window_name} {model_name} {fold}폴드의 Sharpe를 계산할 수 없습니다."
                    ) from error
                rows.append(
                    {
                        "window": window_name,
                        "model": model_name,
                        "fold": fold,
                        "train_size": len(train_index),
                        "valid_size": len(valid_index),
                        "train_end": dataset.frame.iloc[train_index[-1]]["bas_dd"],
                        "valid_start": dataset.frame.iloc[valid_index[0]]["bas_dd"],
                        "valid_end": dataset.frame.iloc[valid_index[-1]]["bas_dd"],
                        **classification_metrics(actual, predicted),
                        "strategy_sharpe_net": strategy_sharpe,
                        "buyhold_sharpe_net": buyhold_sharpe,
                        "delta_sharpe_net": delta,
                        "all_cash": all_cash,
                    }
                )
    return pd.DataFrame(rows)


def summarize_window_results(results: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """48개 폴드·모델 결과의 창별 중앙값과 선택된 전역 창을 돌려준다."""

    required = {"window", "model", "fold", "delta_sharpe_net"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"학습창 결과 열이 없습니다: {sorted(missing)}")
    counts = results.groupby("window").size()
    expected = len(MODEL_BUILDERS) * N_FOLDS
    if set(counts.index) != set(WINDOW_NAMES) or (counts != expected).any():
        raise ValueError(f"각 학습창에는 {expected}개 결과가 있어야 합니다: {counts.to_dict()}")

    summary = (
        results.groupby("window", as_index=False)
        .agg(
            delta_sharpe_median=("delta_sharpe_net", "median"),
            delta_sharpe_mean=("delta_sharpe_net", "mean"),
            accuracy_median=("accuracy", "median"),
            macro_f1_median=("macro_f1", "median"),
            down_recall_median=("down_recall", "median"),
        )
        .sort_values("delta_sharpe_median", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    best_value = float(summary.loc[0, "delta_sharpe_median"])
    tied = summary.loc[
        np.isclose(summary["delta_sharpe_median"], best_value, rtol=0.0, atol=1e-12),
        "window",
    ].tolist()
    selected = "expanding" if "expanding" in tied else tied[0]
    return summary, selected
