"""개별종목 확률 랭킹을 공통 백테스트 원장 계약으로 변환한다."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from evaluation.horizon import HOLDOUT_START
from supply.backtest_ledger import (
    SIGNAL_LOG_COLUMNS,
    TRADE_LOG_COLUMNS,
    verify_execution_log,
    verify_trade_log,
)

HORIZON = 5
LABEL_NAMES = {-1: "하락", 0: "중립", 1: "상승"}
EXTRA_COLUMNS = (
    "fold",
    "top_n",
    "ranking_method",
    "industry_index_name",
    "index_direction_rank",
    "selected_probability",
    "exit_execution_date",
    "entry_adj_open",
    "exit_adj_open",
)


@dataclass(frozen=True)
class StockLedgerResult:
    """공통 계약 열과 개별종목 실행 보조 열을 함께 가진 두 원장."""

    signal_log: pd.DataFrame
    trade_log: pd.DataFrame
    verification: dict[str, object]


def _signal_streaks(ranking: pd.DataFrame) -> pd.DataFrame:
    dates = (
        ranking.loc[:, ["fold", "bas_dd", "index_predicted"]]
        .drop_duplicates()
        .sort_values(["fold", "bas_dd"], kind="stable")
    )
    if dates.duplicated(["fold", "bas_dd"]).any():
        raise ValueError("같은 폴드·날짜의 KOSPI200 예측 방향이 서로 다릅니다.")

    rows = []
    for fold, group in dates.groupby("fold", sort=True):
        consecutive_up = 0
        consecutive_down = 0
        for row in group.itertuples(index=False):
            direction = int(row.index_predicted)
            consecutive_up = consecutive_up + 1 if direction == 1 else 0
            consecutive_down = consecutive_down + 1 if direction == -1 else 0
            rows.append(
                {
                    "fold": int(fold),
                    "bas_dd": str(row.bas_dd),
                    "consecutive_up": consecutive_up,
                    "consecutive_down": consecutive_down,
                }
            )
    return pd.DataFrame(rows)


def _prepare_rows(ranking: pd.DataFrame, top_n: int) -> pd.DataFrame:
    required = {
        "fold",
        "bas_dd",
        "code",
        "industry_index_name",
        "index_predicted",
        "index_direction_rank",
        "selected_probability",
        "entry_bas_dd",
        "exit_bas_dd",
        "entry_adj_open",
        "exit_adj_open",
        "p_down",
        "p_neutral",
        "p_up",
    }
    missing = required - set(ranking.columns)
    if missing:
        raise ValueError(f"개별종목 원장 입력 열이 없습니다: {sorted(missing)}")
    if top_n <= 0:
        raise ValueError("원장 Top-K는 양수여야 합니다.")

    selected = ranking.loc[ranking["index_direction_rank"].astype(int) <= top_n].copy()
    if selected.empty:
        raise ValueError(f"Top {top_n} 원장에 넣을 종목이 없습니다.")
    selected["bas_dd"] = selected["bas_dd"].astype("string")
    selected["entry_bas_dd"] = selected["entry_bas_dd"].astype("string")
    selected["exit_bas_dd"] = selected["exit_bas_dd"].astype("string")
    selected["code"] = selected["code"].astype("string").str.zfill(6)
    if selected.duplicated(["fold", "bas_dd", "code"]).any():
        raise ValueError("같은 폴드·날짜·종목 원장 행이 중복되었습니다.")

    entry = selected["entry_adj_open"].to_numpy(dtype=float)
    exit_ = selected["exit_adj_open"].to_numpy(dtype=float)
    if not np.isfinite(entry).all() or not np.isfinite(exit_).all():
        raise ValueError("원장 진입·청산 수정 시가에 결측 또는 무한대가 있습니다.")
    if (entry <= 0.0).any() or (exit_ <= 0.0).any():
        raise ValueError("원장 진입·청산 수정 시가는 양수여야 합니다.")
    selected["realized_return_5d"] = exit_ / entry - 1.0

    selected = selected.merge(
        _signal_streaks(selected),
        on=["fold", "bas_dd"],
        how="left",
        validate="many_to_one",
    )
    counts = selected.groupby(["fold", "bas_dd"])["code"].transform("size")
    selected["requested_trade_ratio"] = np.where(
        selected["index_predicted"].eq(1),
        1.0 / (HORIZON * counts),
        0.0,
    )
    selected["signal"] = selected["index_predicted"].map(LABEL_NAMES)
    if selected["signal"].isna().any():
        raise ValueError("KOSPI200 예측 방향은 -1·0·1만 허용합니다.")
    if "ranking_method" not in selected:
        selected["ranking_method"] = "model_probability"
    return selected


def build_stock_ledgers(
    ranking: pd.DataFrame,
    *,
    top_n: int,
    cost_rate: float,
    model_id: str,
    model_rev: str,
    run_id: str,
    holdout_start: str = HOLDOUT_START,
) -> StockLedgerResult:
    """Top-K 종목 신호와 실제 진입·청산을 공통 원장 두 장으로 만든다.

    금액·수량은 초기자산 1인 정규화 원장이다. 5슬리브 중 한 칸을 같은 날짜의 Top-K에
    동일 배분한다. 이 원장은 전략 성과 재계산용이 아니라 예측·체결 피드백 추적용이며,
    성과 곡선은 ``evaluation.stock_backtest``가 실제 일별 평가액으로 별도 계산한다.
    """

    if not 0.0 <= cost_rate < 1.0:
        raise ValueError("원장 비용률은 0 이상 1 미만이어야 합니다.")
    if not str(model_id).strip() or not str(model_rev).strip() or not str(run_id).strip():
        raise ValueError("model_id·model_rev·run_id는 비어 있을 수 없습니다.")

    rows = _prepare_rows(ranking, top_n)
    half_cost = cost_rate / 2.0
    rows["actual_trade_value"] = rows["requested_trade_ratio"] * (1.0 - half_cost)
    rows["position_ratio_after"] = rows["actual_trade_value"]
    rows["prediction_date"] = rows["bas_dd"]
    rows["execution_date"] = rows["entry_bas_dd"]
    rows["p_flat"] = rows["p_neutral"]
    rows["model_id"] = str(model_id)
    rows["model_rev"] = str(model_rev)
    rows["run_id"] = str(run_id)
    rows["cost_rate"] = float(cost_rate)
    rows["top_n"] = int(top_n)
    rows["exit_execution_date"] = rows["exit_bas_dd"]
    signal_log = rows.loc[:, [*SIGNAL_LOG_COLUMNS, *EXTRA_COLUMNS]].reset_index(drop=True)

    positioned = rows.loc[rows["index_predicted"].eq(1)].copy()
    buy_value = positioned["actual_trade_value"].to_numpy(dtype=float)
    quantity = buy_value / positioned["entry_adj_open"].to_numpy(dtype=float)
    buy = positioned.copy()
    buy["action"] = "buy"
    buy["actual_trade_ratio"] = buy["actual_trade_value"]
    buy["price"] = buy["entry_adj_open"]
    buy["quantity"] = quantity
    buy["trade_value"] = buy_value
    buy["cost"] = positioned["requested_trade_ratio"] * half_cost

    sell = positioned.copy()
    sell["execution_date"] = sell["exit_bas_dd"]
    sell["action"] = "sell"
    sell["requested_trade_ratio"] = -sell["requested_trade_ratio"]
    sell_value = quantity * sell["exit_adj_open"].to_numpy(dtype=float)
    sell["actual_trade_ratio"] = sell_value
    sell["price"] = sell["exit_adj_open"]
    sell["quantity"] = quantity
    sell["trade_value"] = sell_value
    sell["cost"] = sell_value * half_cost
    sell["position_ratio_after"] = 0.0

    if positioned.empty:
        trade_log = pd.DataFrame(columns=[*TRADE_LOG_COLUMNS, *EXTRA_COLUMNS])
    else:
        trade_log = (
            pd.concat([buy, sell], ignore_index=True)
            .sort_values(["execution_date", "fold", "code", "action"], kind="stable")
            .loc[:, [*TRADE_LOG_COLUMNS, *EXTRA_COLUMNS]]
            .reset_index(drop=True)
        )
    verification = verify_stock_ledgers(
        signal_log,
        trade_log,
        holdout_start=holdout_start,
    )
    problems = [
        *verification["signal"]["problems"],
        *verification["trade"]["problems"],
        *verification["stock_execution"]["problems"],
    ]
    if problems:
        raise ValueError(f"개별종목 원장 계약을 통과하지 못했습니다: {problems}")
    return StockLedgerResult(signal_log, trade_log, verification)


def verify_stock_ledgers(
    signal_log: pd.DataFrame,
    trade_log: pd.DataFrame,
    *,
    holdout_start: str = HOLDOUT_START,
) -> dict[str, object]:
    """공통 계약과 T+1~T+6 수정 시가 실행 계약을 함께 검사한다."""

    signal_result = verify_execution_log(signal_log, holdout_start=holdout_start)
    trade_result = verify_trade_log(trade_log, holdout_start=holdout_start)
    problems: list[str] = []
    required = {
        "prediction_date",
        "execution_date",
        "exit_execution_date",
        "entry_adj_open",
        "exit_adj_open",
        "realized_return_5d",
    }
    missing = required - set(signal_log.columns)
    if missing:
        problems.append(f"개별종목 실행 검산 열이 없습니다: {sorted(missing)}")
    elif len(signal_log):
        pred = pd.to_datetime(signal_log["prediction_date"], errors="coerce")
        entry_date = pd.to_datetime(signal_log["execution_date"], errors="coerce")
        exit_date = pd.to_datetime(signal_log["exit_execution_date"], errors="coerce")
        if pred.isna().any() or entry_date.isna().any() or exit_date.isna().any():
            problems.append("개별종목 원장 날짜를 읽지 못했습니다.")
        if ((entry_date <= pred) | (exit_date <= entry_date)).any():
            problems.append("개별종목 원장의 예측·진입·청산 날짜 순서가 틀렸습니다.")
        if (exit_date >= pd.Timestamp(str(holdout_start))).any():
            problems.append("개별종목 원장의 청산일이 홀드아웃에 들어갑니다.")
        expected = (
            signal_log["exit_adj_open"].to_numpy(dtype=float)
            / signal_log["entry_adj_open"].to_numpy(dtype=float)
            - 1.0
        )
        actual = signal_log["realized_return_5d"].to_numpy(dtype=float)
        if not np.allclose(actual, expected, atol=1e-12):
            problems.append("실현수익률이 T+1~T+6 수정 시가로 다시 계산한 값과 다릅니다.")

    return {
        "signal": signal_result,
        "trade": trade_result,
        "stock_execution": {
            "problems": problems,
            "return_basis": "T+1 adj_open to T+6 adj_open",
            "holdout_start": str(holdout_start),
        },
    }


__all__ = [
    "StockLedgerResult",
    "build_stock_ledgers",
    "verify_stock_ledgers",
]
