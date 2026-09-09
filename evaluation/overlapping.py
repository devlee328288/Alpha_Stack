"""5거래일 예측을 매일 실행하는 중첩 포지션의 일별 성과 계산."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from evaluation.metrics import sharpe_ratio


@dataclass(frozen=True)
class OverlappingResult:
    """5개 슬리브 전략과 같은 기간 Buy&Hold의 일별 순수익률."""

    strategy_net: np.ndarray
    buyhold_net: np.ndarray
    strategy_gross: np.ndarray
    buyhold_gross: np.ndarray
    first_entry_position: int
    last_exit_position: int


def _validate_inputs(
    opens: np.ndarray,
    signal_positions: np.ndarray,
    predictions: np.ndarray,
    horizon: int,
    round_trip_cost: float,
) -> None:
    if horizon < 1:
        raise ValueError(f"horizon은 1 이상이어야 합니다: {horizon}")
    if not 0.0 <= round_trip_cost < 1.0:
        raise ValueError(f"왕복비용은 0 이상 1 미만이어야 합니다: {round_trip_cost}")
    if signal_positions.size != predictions.size:
        raise ValueError("신호 위치와 예측값의 길이가 다릅니다.")
    if signal_positions.size == 0:
        raise ValueError("평가할 신호가 없습니다.")
    if np.any(np.diff(signal_positions) <= 0):
        raise ValueError("신호 위치는 중복 없이 오름차순이어야 합니다.")
    if signal_positions[0] < 0 or signal_positions[-1] + horizon + 1 >= opens.size:
        raise ValueError("5거래일 보유를 계산할 미래 시가가 부족합니다.")
    if not np.all(np.isin(predictions, [-1, 0, 1])):
        raise ValueError("예측값은 하락=-1, 중립=0, 상승=1만 허용합니다.")
    if not np.all(np.isfinite(opens)) or np.any(opens <= 0.0):
        raise ValueError("시가는 모두 유한한 양수여야 합니다.")


def _equity_returns(curve: list[float]) -> np.ndarray:
    values = np.asarray(curve, dtype=float)
    return values[1:] / values[:-1] - 1.0


def _simulate_sleeves(
    opens: np.ndarray,
    signal_positions: np.ndarray,
    predictions: np.ndarray,
    horizon: int,
    round_trip_cost: float,
) -> np.ndarray:
    """자본을 ``horizon``개 슬리브로 나눠 매일 한 슬리브만 교체한다."""

    first_entry = int(signal_positions[0]) + 1
    last_exit = int(signal_positions[-1]) + 1 + horizon
    entries = {
        int(position) + 1: int(prediction)
        for position, prediction in zip(signal_positions, predictions, strict=True)
    }

    sleeve_values = np.full(horizon, 1.0 / horizon, dtype=float)
    invested = np.zeros(horizon, dtype=bool)
    exits = np.full(horizon, -1, dtype=int)
    half_cost = round_trip_cost / 2.0
    equity = [1.0]

    for interval_start in range(first_entry, last_exit):
        slot = interval_start % horizon
        prediction = entries.get(interval_start)
        if prediction is not None:
            if invested[slot]:
                raise RuntimeError("5거래일이 지나기 전에 같은 슬리브를 다시 사용했습니다.")
            if prediction == 1:
                # 진입 비용은 해당 슬리브에만 적용한다. 나머지 네 슬리브에는 영향이 없다.
                sleeve_values[slot] *= 1.0 - half_cost
                invested[slot] = True
                exits[slot] = interval_start + horizon

        daily_return = opens[interval_start + 1] / opens[interval_start] - 1.0
        sleeve_values[invested] *= 1.0 + daily_return

        exit_slots = np.flatnonzero(invested & (exits == interval_start + 1))
        for exit_slot in exit_slots:
            sleeve_values[exit_slot] *= 1.0 - half_cost
            invested[exit_slot] = False
            exits[exit_slot] = -1

        equity.append(float(sleeve_values.sum()))

    if np.any(invested):
        raise RuntimeError("평가 종료 뒤에도 청산되지 않은 슬리브가 있습니다.")
    return _equity_returns(equity)


def _simulate_buyhold(
    opens: np.ndarray,
    first_entry: int,
    last_exit: int,
    round_trip_cost: float,
) -> np.ndarray:
    half_cost = round_trip_cost / 2.0
    net_equity = [1.0]
    value = 1.0 * (1.0 - half_cost)

    for interval_start in range(first_entry, last_exit):
        value *= opens[interval_start + 1] / opens[interval_start]
        if interval_start + 1 == last_exit:
            value *= 1.0 - half_cost
        net_equity.append(float(value))
    return _equity_returns(net_equity)


def overlapping_long_only_returns(
    opens: Sequence[float],
    signal_positions: Sequence[int],
    predictions: Sequence[int],
    *,
    horizon: int = 5,
    round_trip_cost: float = 0.0005,
) -> OverlappingResult:
    """매일의 5일 예측을 겹쳐 실행한 전략과 Buy&Hold 수익률을 만든다.

    상승 예측만 다음 거래일 시가에 진입하고 중립·하락은 현금으로 둔다. 자본을 다섯
    슬리브로 나눠 각 슬리브가 정확히 5거래일 뒤 청산되므로, 매일 신호를 실행하면서도
    한 신호의 보유기간을 임의로 단축하지 않는다. 각 폴드는 빈 포지션에서 시작하고
    마지막 검증 신호의 포지션까지 모두 청산한 뒤 끝낸다.
    """

    open_values = np.asarray(opens, dtype=float)
    positions = np.asarray(signal_positions, dtype=int)
    predicted = np.asarray(predictions, dtype=int)
    _validate_inputs(open_values, positions, predicted, horizon, round_trip_cost)

    first_entry = int(positions[0]) + 1
    last_exit = int(positions[-1]) + 1 + horizon
    strategy_net = _simulate_sleeves(
        open_values,
        positions,
        predicted,
        horizon,
        round_trip_cost,
    )
    strategy_gross = _simulate_sleeves(open_values, positions, predicted, horizon, 0.0)
    buyhold_net = _simulate_buyhold(open_values, first_entry, last_exit, round_trip_cost)
    buyhold_gross = _simulate_buyhold(open_values, first_entry, last_exit, 0.0)
    return OverlappingResult(
        strategy_net=strategy_net,
        buyhold_net=buyhold_net,
        strategy_gross=strategy_gross,
        buyhold_gross=buyhold_gross,
        first_entry_position=first_entry,
        last_exit_position=last_exit,
    )


def delta_sharpe_net(result: OverlappingResult) -> float | None:
    """전략 순수익률 Sharpe에서 같은 기간 Buy&Hold 순수익률 Sharpe를 뺀다."""

    strategy_sharpe = sharpe_ratio(result.strategy_net)
    buyhold_sharpe = sharpe_ratio(result.buyhold_net)
    if strategy_sharpe is None or buyhold_sharpe is None:
        return None
    return strategy_sharpe - buyhold_sharpe
