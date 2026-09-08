"""개별종목 Top-K를 T+1~T+6 시가로 운용하는 5슬리브 OOS 백테스트."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from evaluation.metrics import equity_curve, max_drawdown, sharpe_ratio

HORIZON = 5
PROBABILITY_COLUMNS = ("p_down", "p_neutral", "p_up")


@dataclass(frozen=True)
class StockBacktestResult:
    """전체 5슬리브 성과와 재현 가능한 로그."""

    summary: dict[str, object]
    daily_returns: pd.DataFrame
    trade_log: pd.DataFrame
    track_summary: pd.DataFrame


def validate_stock_ranking_contract(ranking: pd.DataFrame) -> None:
    """모델 출력과 백테스트 사이의 최소 열·유일성·확률 계약을 검사한다."""

    required = {
        "fold",
        "model",
        "bas_dd",
        "code",
        "industry_index_name",
        "index_predicted",
        "selected_probability",
        "index_direction_rank",
        "entry_bas_dd",
        "exit_bas_dd",
        "entry_adj_open",
        "exit_adj_open",
        "label_numeric",
        *PROBABILITY_COLUMNS,
    }
    missing = required - set(ranking.columns)
    if missing:
        raise ValueError(f"종목 랭킹-백테스트 계약 열이 없습니다: {sorted(missing)}")
    if ranking.empty:
        raise ValueError("종목 랭킹이 비어 있습니다.")
    if ranking.duplicated(["fold", "bas_dd", "code"]).any():
        raise ValueError("같은 폴드·날짜·종목 랭킹이 중복되었습니다.")
    if (ranking["index_direction_rank"].astype(int) <= 0).any():
        raise ValueError("종목 순위는 1 이상의 정수여야 합니다.")
    probabilities = ranking.loc[:, PROBABILITY_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or not np.allclose(
        probabilities.sum(axis=1),
        1.0,
        atol=1e-6,
    ):
        raise ValueError("하락·중립·상승 확률이 유한하지 않거나 합이 1이 아닙니다.")
    if not set(ranking["index_predicted"].astype(int)).issubset({-1, 0, 1}):
        raise ValueError("KOSPI200 예측 클래스는 -1·0·1만 허용합니다.")


def _signal_table(ranking: pd.DataFrame, top_n: int) -> pd.DataFrame:
    selected = ranking.loc[ranking["index_direction_rank"].astype(int) <= top_n].copy()
    if selected.empty:
        raise ValueError(f"Top {top_n}에 해당하는 종목이 없습니다.")
    keys = ["fold", "bas_dd"]
    consistency_columns = ["index_predicted", "entry_bas_dd", "exit_bas_dd"]
    for column in consistency_columns:
        if selected.groupby(keys, sort=False)[column].nunique().gt(1).any():
            raise ValueError(f"같은 신호 날짜의 {column} 값이 서로 다릅니다.")
    dates = selected[keys + consistency_columns].drop_duplicates().sort_values(keys)
    dates["track"] = dates.groupby("fold", sort=False).cumcount() % HORIZON
    return selected.merge(dates, on=keys + consistency_columns, validate="many_to_one")


def _price_lookup(daily_prices: pd.DataFrame) -> pd.Series:
    required = {"bas_dd", "code", "adj_open"}
    missing = required - set(daily_prices.columns)
    if missing:
        raise ValueError(f"수정 시가 열이 없습니다: {sorted(missing)}")
    prices = daily_prices.loc[:, ["bas_dd", "code", "adj_open"]].copy()
    prices["bas_dd"] = prices["bas_dd"].astype("string")
    prices["code"] = prices["code"].astype("string").str.zfill(6)
    if prices.duplicated(["bas_dd", "code"]).any():
        raise ValueError("같은 날짜·종목의 수정 시가가 중복되었습니다.")
    # 가격 파일 전체에 거래정지 결측이 있는 것은 정상이다. 여기서 값을 채우거나 전체
    # 실행을 막지 않고, 실제 보유 종목의 평가일에 조회될 때만 `_get_price()`가 검사한다.
    # 그래야 백테스트와 무관한 종목의 거래정지가 실행을 중단시키지 않는다.
    values = prices["adj_open"].to_numpy(dtype=float)
    prices["adj_open"] = np.where(np.isfinite(values) & (values > 0.0), values, np.nan)
    return prices.set_index(["bas_dd", "code"])["adj_open"]


def _get_price(price_lookup: pd.Series, date: str, code: str) -> float:
    """실제 거래·평가에 필요한 수정 시가가 없으면 조용히 통과시키지 않는다."""

    try:
        price = float(price_lookup.loc[(date, code)])
    except KeyError as error:
        raise ValueError(f"수정 시가가 없습니다: {date} {code}") from error
    if not np.isfinite(price) or price <= 0.0:
        raise ValueError(f"사용 가능한 과거 수정 시가가 없습니다: {date} {code}")
    return price


def _simulate_fold(
    fold_signals: pd.DataFrame,
    price_lookup: pd.Series,
    *,
    round_trip_cost: float,
    active_tracks: tuple[int, ...],
) -> tuple[pd.DataFrame, float]:
    """한 외부 폴드에서 지정한 슬리브를 독립적으로 시뮬레이션한다."""

    signals = fold_signals.loc[fold_signals["track"].isin(active_tracks)].copy()
    if signals.empty:
        return pd.DataFrame(columns=["bas_dd", "net_return"]), 0.0
    remap = {track: index for index, track in enumerate(active_tracks)}
    signals["slot"] = signals["track"].map(remap)
    signal_dates = signals[
        ["bas_dd", "entry_bas_dd", "exit_bas_dd", "slot", "index_predicted"]
    ].drop_duplicates()
    entry_groups = {
        str(date): group
        for date, group in signals.groupby("entry_bas_dd", sort=False)
    }
    exit_events: dict[str, list[int]] = {}
    for row in signal_dates.itertuples(index=False):
        if int(row.index_predicted) == 1:
            exit_events.setdefault(str(row.exit_bas_dd), []).append(int(row.slot))
    first_entry = str(signal_dates["entry_bas_dd"].min())
    last_exit = str(signal_dates["exit_bas_dd"].max())
    calendar = sorted(
        {
            str(date)
            for date in price_lookup.index.get_level_values("bas_dd")
            if first_entry <= str(date) <= last_exit
        }
    )
    if not calendar or calendar[0] != first_entry or calendar[-1] != last_exit:
        raise ValueError("신호의 진입·청산 날짜를 수정 시가 달력에서 찾지 못했습니다.")

    slot_count = len(active_tracks)
    cash = np.full(slot_count, 1.0 / slot_count, dtype=float)
    holdings: dict[int, dict[str, float]] = {}
    half_cost = round_trip_cost / 2.0
    previous_equity = 1.0
    turnover = 0.0
    rows: list[dict[str, object]] = []

    for date in calendar:
        slot_values = cash.copy()
        for slot, shares in holdings.items():
            slot_values[slot] = sum(
                quantity * _get_price(price_lookup, date, code)
                for code, quantity in shares.items()
            )

        for slot in sorted(exit_events.get(date, [])):
            if slot not in holdings:
                raise RuntimeError("활성 포지션이 없는 슬리브를 청산하려 했습니다.")
            equity_before = float(slot_values.sum())
            turnover += slot_values[slot] / equity_before
            slot_values[slot] *= 1.0 - half_cost
            cash[slot] = slot_values[slot]
            del holdings[slot]

        entry = entry_groups.get(date)
        if entry is not None:
            for slot, slot_rows in entry.groupby("slot", sort=True):
                slot = int(slot)
                if slot in holdings:
                    raise RuntimeError("5거래일 전에 같은 슬리브에 다시 진입했습니다.")
                if int(slot_rows["index_predicted"].iloc[0]) != 1:
                    continue
                equity_before = float(slot_values.sum())
                turnover += slot_values[slot] / equity_before
                slot_values[slot] *= 1.0 - half_cost
                codes = slot_rows["code"].astype("string").str.zfill(6).tolist()
                allocation = slot_values[slot] / len(codes)
                holdings[slot] = {
                    code: allocation / _get_price(price_lookup, date, code)
                    for code in codes
                }
                cash[slot] = 0.0

        equity = float(slot_values.sum())
        rows.append({"bas_dd": date, "net_return": equity / previous_equity - 1.0})
        previous_equity = equity

    if holdings:
        raise RuntimeError("폴드 마지막 날짜 뒤에도 청산되지 않은 슬리브가 있습니다.")
    return pd.DataFrame(rows), float(turnover)


def _summary(returns: np.ndarray, turnover: float) -> dict[str, object]:
    curve = equity_curve(returns)
    return {
        "observations": int(len(returns)),
        "total_return": float(np.prod(1.0 + returns) - 1.0),
        "sharpe": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(curve),
        "turnover": turnover,
    }


def run_overlapping_stock_backtest(
    ranking: pd.DataFrame,
    daily_prices: pd.DataFrame,
    *,
    top_n: int,
    round_trip_cost: float,
) -> StockBacktestResult:
    """상승 지수 예측일만 Top-K를 매수하고 다섯 슬리브로 겹쳐 운용한다."""

    if top_n <= 0 or not 0.0 <= round_trip_cost < 1.0:
        raise ValueError("Top-K는 양수이고 왕복비용은 0 이상 1 미만이어야 합니다.")
    validate_stock_ranking_contract(ranking)
    signals = _signal_table(ranking, top_n)
    lookup = _price_lookup(daily_prices)

    daily_parts = []
    turnover = 0.0
    for fold, fold_signals in signals.groupby("fold", sort=True):
        daily, fold_turnover = _simulate_fold(
            fold_signals,
            lookup,
            round_trip_cost=round_trip_cost,
            active_tracks=tuple(range(HORIZON)),
        )
        daily.insert(0, "fold", int(fold))
        daily_parts.append(daily)
        turnover += fold_turnover
    daily_returns = pd.concat(daily_parts, ignore_index=True)

    track_rows = []
    for track in range(HORIZON):
        parts = []
        track_turnover = 0.0
        for _fold, fold_signals in signals.groupby("fold", sort=True):
            daily, fold_turnover = _simulate_fold(
                fold_signals,
                lookup,
                round_trip_cost=round_trip_cost,
                active_tracks=(track,),
            )
            parts.append(daily)
            track_turnover += fold_turnover
        track_returns = pd.concat(parts, ignore_index=True)["net_return"].to_numpy(float)
        track_rows.append({"track": track + 1, **_summary(track_returns, track_turnover)})

    positioned = signals.loc[signals["index_predicted"].eq(1)].copy()
    positioned["gross_return"] = (
        positioned["exit_adj_open"] / positioned["entry_adj_open"] - 1.0
    )
    positioned["net_return"] = (
        (1.0 + positioned["gross_return"]) * (1.0 - round_trip_cost / 2.0) ** 2 - 1.0
    )
    trade_columns = [
        "fold",
        "track",
        "bas_dd",
        "entry_bas_dd",
        "exit_bas_dd",
        "code",
        "industry_index_name",
        "index_direction_rank",
        "selected_probability",
        "entry_adj_open",
        "exit_adj_open",
        "gross_return",
        "net_return",
    ]
    returns = daily_returns["net_return"].to_numpy(dtype=float)
    summary = {
        **_summary(returns, turnover),
        "top_n": top_n,
        "round_trip_cost": round_trip_cost,
        "position_policy": "KOSPI200 상승 예측일만 매수, 중립·하락은 현금",
        "entry": "T+1 adj_open",
        "exit": "T+6 adj_open",
        "tracks": HORIZON,
        "trade_rows": int(len(positioned)),
        "signal_dates": int(signals["bas_dd"].nunique()),
    }
    return StockBacktestResult(
        summary=summary,
        daily_returns=daily_returns,
        trade_log=positioned.loc[:, trade_columns].reset_index(drop=True),
        track_summary=pd.DataFrame(track_rows),
    )


__all__ = [
    "StockBacktestResult",
    "run_overlapping_stock_backtest",
    "validate_stock_ranking_contract",
]
