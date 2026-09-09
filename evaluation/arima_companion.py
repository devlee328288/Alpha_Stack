"""지도학습 모델과 같은 외부 폴드에서 재는 ARIMA 동반 기준선."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from timeseries.forecast import point
from timeseries.models import fit_best

FitFunction = Callable[[Sequence[float]], dict[str, object]]
ForecastFunction = Callable[[dict[str, object], int], np.ndarray]


@dataclass(frozen=True)
class ArimaCompanionResult:
    """폴드별 차수와 날짜별 3분류 예측."""

    fold_results: pd.DataFrame
    predictions: pd.DataFrame
    trial_results: pd.DataFrame


def evaluate_arima_companion(
    opens: Sequence[float],
    signal_positions: Sequence[int],
    labels: Sequence[int],
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    neutral_band: float = 0.01,
    horizon: int = 5,
    fit_function: FitFunction = fit_best,
    forecast_function: ForecastFunction = point,
) -> ArimaCompanionResult:
    """동일 학습 종료점에서 검증 60일 전체를 한 번에 예측한다.

    검증기간 실제 시가로 모형을 갱신하지 않는다. 따라서 실시간 rolling ARIMA보다
    보수적이지만, 지도학습 모델과 같은 학습 정보만 본다는 비교 조건은 지킨다.
    """

    open_values = np.asarray(opens, dtype=float)
    positions = np.asarray(signal_positions, dtype=int)
    actual = np.asarray(labels, dtype=int)
    if positions.shape != actual.shape or positions.ndim != 1:
        raise ValueError("신호 위치와 라벨은 길이가 같은 1차원이어야 합니다.")
    if not 0.0 < neutral_band < 1.0 or horizon <= 0:
        raise ValueError("중립대는 0~1 사이이고 예측기간은 양수여야 합니다.")
    if not np.isfinite(open_values).all() or (open_values <= 0.0).any():
        raise ValueError("ARIMA 입력 시가는 유한한 양수여야 합니다.")

    fold_rows = []
    prediction_rows = []
    trial_rows = []
    for fold, (train, valid) in enumerate(splits, start=1):
        train = np.asarray(train, dtype=int)
        valid = np.asarray(valid, dtype=int)
        if train.size == 0 or valid.size == 0:
            raise ValueError("ARIMA 학습·검증 인덱스가 비어 있습니다.")
        train_end = int(positions[train[-1]])
        valid_positions = positions[valid]
        entry_steps = valid_positions + 1 - train_end
        exit_steps = valid_positions + 1 + horizon - train_end
        if entry_steps.min() <= 0 or exit_steps.max() > len(open_values) - train_end - 1:
            raise ValueError("ARIMA 예측에 필요한 미래 시가 위치가 원시표 밖입니다.")

        fitted = fit_function(open_values[: train_end + 1])
        model = fitted.get("model")
        if model is None:
            raise RuntimeError(f"ARIMA {fold}폴드 적합에 실패했습니다.")
        forecast = np.asarray(forecast_function(model, int(exit_steps.max())), dtype=float)
        if len(forecast) < int(exit_steps.max()) or not np.isfinite(forecast).all():
            raise RuntimeError(f"ARIMA {fold}폴드 예측 길이가 부족하거나 유한하지 않습니다.")
        entry = forecast[entry_steps - 1]
        exit_values = forecast[exit_steps - 1]
        returns = exit_values / entry - 1.0
        predicted = np.where(
            returns > neutral_band,
            1,
            np.where(returns < -neutral_band, -1, 0),
        )
        order = fitted.get("order", {})
        seen_orders: set[tuple[int, int]] = set()
        candidates = list(order.get("candidates", [])) + list(
            order.get("rejected_unstable", [])
        )
        for candidate in candidates:
            candidate_key = (int(candidate["p"]), int(candidate["q"]))
            if candidate_key in seen_orders:
                continue
            seen_orders.add(candidate_key)
            trial_rows.append(
                {
                    "fold": fold,
                    "phase": "order_search",
                    "p": candidate_key[0],
                    "d": int(order.get("d", 0)),
                    "q": candidate_key[1],
                    "train_rows": int(len(train)),
                    "valid_rows": int(len(valid)),
                    "aic": candidate.get("aic"),
                    "bic": candidate.get("bic"),
                    "stable": candidate.get("stable"),
                    "white_noise": candidate.get("white_noise"),
                    "selected": candidate_key
                    == (int(order.get("p", 0)), int(order.get("q", 0))),
                }
            )
        trial_rows.append(
            {
                "fold": fold,
                "phase": "outer_evaluation",
                "p": int(order.get("p", 0)),
                "d": int(order.get("d", 0)),
                "q": int(order.get("q", 0)),
                "train_rows": int(len(train)),
                "valid_rows": int(len(valid)),
                "accuracy": float(np.mean(predicted == actual[valid])),
                "selected": True,
            }
        )
        fold_rows.append(
            {
                "fold": fold,
                "train_rows": int(len(train)),
                "valid_rows": int(len(valid)),
                "train_end_raw_position": train_end,
                "p": order.get("p"),
                "d": order.get("d"),
                "q": order.get("q"),
            }
        )
        prediction_rows.extend(
            {
                "fold": fold,
                "row_index": int(row_index),
                "actual": int(label),
                "predicted": int(prediction),
                "forecast_return_5d": float(forecast_return),
            }
            for row_index, label, prediction, forecast_return in zip(
                valid,
                actual[valid],
                predicted,
                returns,
                strict=True,
            )
        )
    return ArimaCompanionResult(
        fold_results=pd.DataFrame(fold_rows),
        predictions=pd.DataFrame(prediction_rows),
        trial_results=pd.DataFrame(trial_rows),
    )


__all__ = ["ArimaCompanionResult", "evaluate_arima_companion"]
