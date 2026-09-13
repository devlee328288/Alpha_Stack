import uuid
import warnings
from datetime import datetime
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore")

# backtest_strategies.py
# ============================================================
# KRX 실제 데이터 + 가상 예측(랜덤)을 이용한
# A / B / C 3가지 포지션 운용 전략 백테스트
#
# 지수(KOSPI200) / 개별종목(stocks30) 둘 다 지원.
# ============================================================

_REPO_ID = "qurious-quant/alphastack-krx-dev"
_KOSPI200_CSV = "small/features_labels_kospi200_dev.csv"
_STOCKS30_CSV = "small/features_labels_stocks30_dev.csv"


# ============================================================
# 0. 데이터 로드
# ============================================================


def load_data(ticker=None) -> pd.DataFrame:
    """
    Hugging Face 에서 KRX 데이터를 로드.

    Parameters
    ----------
    ticker : None | "KOSPI200" | "<code>"
        None / "KOSPI200" : KOSPI200 지수 (기본, 기존 동작)
        "<code>"          : stocks30 개별종목 (예 "005930")

    Returns
    -------
    pd.DataFrame
        날짜를 DatetimeIndex 로 갖는 정렬된 시장 데이터
    """
    if ticker is None or ticker == "KOSPI200":
        path = hf_hub_download(
            repo_id=_REPO_ID, filename=_KOSPI200_CSV, repo_type="dataset",
        )
        df = pd.read_csv(path)
    else:
        path = hf_hub_download(
            repo_id=_REPO_ID, filename=_STOCKS30_CSV, repo_type="dataset",
        )
        df_all = pd.read_csv(
            path, dtype={"code": str}, low_memory=False,
        )

        code_str = str(ticker).strip()
        candidates = [code_str]
        if code_str.isdigit():
            candidates.append(code_str.zfill(6))

        df = None
        for cand in candidates:
            sub = df_all[df_all["code"] == cand]
            if not sub.empty:
                df = sub.copy()
                break

        if df is None or df.empty:
            raise ValueError(f"종목코드 {ticker!r} 를 stocks30 에서 찾지 못했습니다.")

    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.set_index("date", inplace=True)

    required_columns = ["open", "close"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"데이터에 필요한 컬럼이 없습니다: {missing_columns}")

    return df


# ============================================================
# 1. 예측 함수
# ============================================================


def predict_5d_after(
    base_date: pd.Timestamp,
    market_data: pd.DataFrame,
) -> Tuple[str, Dict[str, float]]:
    """
    5영업일 후 방향을 예측 (구조 테스트용 랜덤).
    실제 AI 모델로 교체 시 이 함수만 바꾸면 됩니다.
    """
    np.random.seed(hash(base_date) % 2**32)

    labels = ["상승", "중립", "하락"]
    probs = [0.33, 0.34, 0.33]

    chosen_label = np.random.choice(labels, p=probs)

    prob_map = {
        "p_up": probs[labels.index("상승")],
        "p_flat": probs[labels.index("중립")],
        "p_down": probs[labels.index("하락")],
    }

    return chosen_label, prob_map


# ============================================================
# 2. A / B / C 전략의 매매 비율 계산
# ============================================================


def get_trade_ratio_a(signal: str, consecutive_up: int, consecutive_down: int) -> float:
    if signal == "상승":
        if consecutive_up == 1:
            return 0.20
        elif consecutive_up == 2:
            return 0.30
        else:
            return 0.50
    elif signal == "하락":
        if consecutive_down == 1:
            return -0.20
        elif consecutive_down == 2:
            return -0.30
        else:
            return -0.50
    else:
        return 0.0


def get_trade_ratio_b(signal: str) -> float:
    if signal == "상승":
        return 0.25
    elif signal == "하락":
        return -0.25
    else:
        return 0.0


def get_trade_ratio_c(signal: str) -> float:
    if signal == "상승":
        return 1.0
    elif signal == "하락":
        return -1.0
    else:
        return 0.0


# ============================================================
# 3. 전략별 매매 비율 선택
# ============================================================


def get_trade_ratio(
    strategy: str, signal: str, consecutive_up: int, consecutive_down: int
) -> float:
    if strategy == "A":
        return get_trade_ratio_a(signal, consecutive_up, consecutive_down)
    elif strategy == "B":
        return get_trade_ratio_b(signal)
    elif strategy == "C":
        return get_trade_ratio_c(signal)
    else:
        raise ValueError(f"알 수 없는 전략입니다: {strategy}")


# ============================================================
# 4. 연속 시그널 카운터 업데이트
# ============================================================


def update_signal_streak(signal: str, consecutive_up: int, consecutive_down: int):
    if signal == "상승":
        consecutive_up += 1
        consecutive_down = 0
    elif signal == "하락":
        consecutive_down += 1
        consecutive_up = 0
    elif signal == "중립":
        consecutive_up = 0
        consecutive_down = 0
    else:
        raise ValueError(f"알 수 없는 시그널: {signal}")

    return consecutive_up, consecutive_down


# ============================================================
# 5. 백테스트 실행
# ============================================================


def run_backtest(
    market_data: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    predict_func: Callable,
    strategy: str = "A",
    initial_cash: float = 100.0,
    trade_cost: float = 0.001,
    model_id: str = "baseline-v0",
    run_id: Optional[str] = None,
    asset_code: Optional[str] = None,  # 🆕 ticker 지원
) -> Dict:
    """
    A / B / C 전략 백테스트.

    Parameters
    ----------
    asset_code : str | None
        None 이면 "KOSPI200" 로 표기. 개별종목이면 종목코드 전달.
    """
    if strategy not in ["A", "B", "C"]:
        raise ValueError("strategy는 'A', 'B', 'C' 중 하나여야 합니다.")

    if run_id is None:
        run_id = (
            f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        )

    if asset_code is None:
        asset_code = "KOSPI200"

    # --------------------------------------------------------
    # 가격 데이터
    # --------------------------------------------------------
    open_prices = market_data["open"].copy()
    close_prices = market_data["close"].copy()

    all_days = close_prices.index

    trading_days = all_days[(all_days >= start_date) & (all_days <= end_date)]

    if len(trading_days) < 2:
        raise ValueError("백테스트를 수행하기 위한 영업일이 부족합니다.")

    # --------------------------------------------------------
    # 초기 상태
    # --------------------------------------------------------
    cash = float(initial_cash)
    position_quantity = 0.0
    position_ratio = 0.0
    consecutive_up = 0
    consecutive_down = 0

    portfolio_values = []
    daily_returns = []
    trade_log = []
    signal_log = []

    # --------------------------------------------------------
    # 날짜별 백테스트
    # --------------------------------------------------------
    for i, date in enumerate(trading_days):

        predict_result = predict_func(date, market_data)

        if isinstance(predict_result, tuple):
            signal, probs = predict_result
        else:
            signal = predict_result
            probs = {"p_up": 0.33, "p_flat": 0.34, "p_down": 0.33}

        consecutive_up, consecutive_down = update_signal_streak(
            signal, consecutive_up, consecutive_down
        )

        if i < len(trading_days) - 1:

            execution_date = trading_days[i + 1]
            execution_price = open_prices.loc[execution_date]

            trade_ratio = get_trade_ratio(
                strategy=strategy,
                signal=signal,
                consecutive_up=consecutive_up,
                consecutive_down=consecutive_down,
            )

            portfolio_value_before = cash + position_quantity * execution_price
            requested_trade_value = portfolio_value_before * trade_ratio

            actual_trade_value = 0.0
            action = "hold"
            cost = 0.0
            quantity = 0.0

            # 상승 → 매수
            if requested_trade_value > 0:
                current_investment = position_quantity * execution_price
                current_ratio = (
                    current_investment / portfolio_value_before
                    if portfolio_value_before > 0
                    else 0.0
                )
                max_additional_investment = portfolio_value_before * max(
                    0.0, 1.0 - current_ratio
                )
                actual_trade_value = min(
                    requested_trade_value, max_additional_investment
                )
                cost = actual_trade_value * trade_cost

                available_cash = cash
                if available_cash <= cost:
                    actual_trade_value = 0.0
                    cost = 0.0
                else:
                    max_trade_by_cash = available_cash / (1.0 + trade_cost)
                    actual_trade_value = min(actual_trade_value, max_trade_by_cash)
                    cost = actual_trade_value * trade_cost

                if actual_trade_value > 0:
                    quantity = actual_trade_value / execution_price
                    cash -= actual_trade_value + cost
                    position_quantity += quantity
                    action = "buy"
                else:
                    quantity = 0.0

            # 하락 → 매도
            elif requested_trade_value < 0:
                requested_sell_value = abs(requested_trade_value)
                current_investment = position_quantity * execution_price
                actual_trade_value = min(requested_sell_value, current_investment)
                cost = actual_trade_value * trade_cost

                if actual_trade_value > 0:
                    quantity = actual_trade_value / execution_price
                    position_quantity -= quantity
                    cash += actual_trade_value - cost
                    action = "sell"
                else:
                    quantity = 0.0

            # 중립
            else:
                actual_trade_value = 0.0
                cost = 0.0
                quantity = 0.0
                action = "hold"

            portfolio_value_after = cash + position_quantity * execution_price
            if portfolio_value_after > 0:
                position_ratio = (
                    position_quantity * execution_price / portfolio_value_after
                )
            else:
                position_ratio = 0.0

            # 5일 실현 수익률
            realized_return_5d = np.nan
            if i + 6 < len(trading_days):
                entry_open = open_prices.loc[trading_days[i + 1]]
                exit_open = open_prices.loc[trading_days[i + 6]]
                if entry_open != 0:
                    realized_return_5d = (exit_open - entry_open) / entry_open

            signal_log.append({
                "prediction_date": date,
                "execution_date": execution_date,
                "signal": signal,
                "consecutive_up": consecutive_up,
                "consecutive_down": consecutive_down,
                "requested_trade_ratio": trade_ratio,
                "actual_trade_value": actual_trade_value,
                "position_ratio_after": position_ratio,
                "code": asset_code,
                "p_up": probs["p_up"],
                "p_flat": probs["p_flat"],
                "p_down": probs["p_down"],
                "realized_return_5d": realized_return_5d,
                "model_id": model_id,
                "model_rev": "v0",
                "run_id": run_id,
                "cost_rate": trade_cost,
            })

            if action != "hold":
                trade_log.append({
                    "prediction_date": date,
                    "execution_date": execution_date,
                    "signal": signal,
                    "consecutive_up": consecutive_up,
                    "consecutive_down": consecutive_down,
                    "action": action,
                    "requested_trade_ratio": trade_ratio,
                    "actual_trade_ratio": (
                        actual_trade_value / portfolio_value_before
                        if portfolio_value_before > 0
                        else 0.0
                    ),
                    "price": execution_price,
                    "quantity": quantity,
                    "trade_value": actual_trade_value,
                    "cost": cost,
                    "position_ratio_after": position_ratio,
                    "code": asset_code,
                    "p_up": probs["p_up"],
                    "p_flat": probs["p_flat"],
                    "p_down": probs["p_down"],
                    "realized_return_5d": realized_return_5d,
                    "model_id": model_id,
                    "model_rev": "v0",
                    "run_id": run_id,
                    "cost_rate": trade_cost,
                })

        current_close = close_prices.loc[date]
        portfolio_value = cash + position_quantity * current_close
        portfolio_values.append(portfolio_value)

        if len(portfolio_values) >= 2:
            previous_value = portfolio_values[-2]
            if previous_value != 0:
                daily_return = (portfolio_values[-1] - previous_value) / previous_value
            else:
                daily_return = 0.0
            daily_returns.append(daily_return)

    # ========================================================
    # 시계열 생성
    # ========================================================
    portfolio_series = pd.Series(
        portfolio_values, index=trading_days[: len(portfolio_values)]
    )
    daily_returns_series = pd.Series(
        daily_returns, index=trading_days[1 : len(daily_returns) + 1]
    )

    if len(portfolio_series) == 0:
        raise ValueError("포트폴리오 시계열이 비어 있습니다.")

    total_return = (portfolio_series.iloc[-1] / initial_cash) - 1

    years = len(portfolio_series) / 252
    annual_return = (
        ((portfolio_series.iloc[-1] / initial_cash) ** (1 / years)) - 1
        if years > 0 else np.nan
    )

    volatility = (
        daily_returns_series.std(ddof=1) * np.sqrt(252)
        if len(daily_returns_series) > 1 else np.nan
    )

    if len(daily_returns_series) > 1 and daily_returns_series.std(ddof=1) != 0:
        sharpe = (
            daily_returns_series.mean() / daily_returns_series.std(ddof=1)
        ) * np.sqrt(252)
    else:
        sharpe = np.nan

    running_max = portfolio_series.cummax()
    drawdown = (portfolio_series / running_max) - 1
    max_drawdown = drawdown.min()

    positive_days = daily_returns_series > 0
    negative_days = daily_returns_series < 0
    neutral_days = daily_returns_series == 0

    num_winning_days = int(positive_days.sum())
    num_losing_days = int(negative_days.sum())
    num_neutral_days = int(neutral_days.sum())
    total_days = len(daily_returns_series)
    win_rate_daily = num_winning_days / total_days if total_days > 0 else 0.0

    avg_win = (
        daily_returns_series[positive_days].mean() if num_winning_days > 0 else 0.0
    )
    avg_loss = (
        daily_returns_series[negative_days].mean() if num_losing_days > 0 else 0.0
    )

    total_gain = (
        daily_returns_series[positive_days].sum() if num_winning_days > 0 else 0.0
    )
    total_loss = (
        abs(daily_returns_series[negative_days].sum()) if num_losing_days > 0 else 0.0
    )
    profit_factor = total_gain / total_loss if total_loss > 0 else np.inf

    def get_max_streak(condition: pd.Series) -> int:
        if len(condition) == 0:
            return 0
        groups = (condition != condition.shift()).cumsum()
        streaks = condition.astype(int).groupby(groups).sum()
        return int(streaks.max() if len(streaks) > 0 else 0)

    max_win_streak = get_max_streak(daily_returns_series > 0)
    max_loss_streak = get_max_streak(daily_returns_series < 0)

    num_trades = len(trade_log)
    buy_trades = sum(1 for trade in trade_log if trade["action"] == "buy")
    sell_trades = sum(1 for trade in trade_log if trade["action"] == "sell")
    total_transaction_cost = sum(trade["cost"] for trade in trade_log)

    return {
        "strategy": strategy,
        "asset_code": asset_code,
        "portfolio_series": portfolio_series,
        "daily_returns": daily_returns_series,
        "trade_log": pd.DataFrame(trade_log),
        "signal_log": pd.DataFrame(signal_log),
        "total_return": total_return,
        "annual_return": annual_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "final_portfolio_value": portfolio_series.iloc[-1],
        "final_cash": cash,
        "final_position_quantity": position_quantity,
        "final_position_ratio": position_ratio,
        "num_winning_days": num_winning_days,
        "num_losing_days": num_losing_days,
        "num_neutral_days": num_neutral_days,
        "win_rate_daily": win_rate_daily,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
        "num_trades": num_trades,
        "buy_trades": buy_trades,
        "sell_trades": sell_trades,
        "total_transaction_cost": total_transaction_cost,
    }


# ============================================================
# 12. 메인 실행
# ============================================================


if __name__ == "__main__":

    print("=" * 100)
    print("🚀 KRX 데이터 기반 A / B / C 전략 백테스트")
    print("=" * 100)

    print("\n[전략 구조]")
    print("A : 연속 시그널 기반 20% → 30% → 50%")
    print("B : 매 시그널마다 25%")
    print("C : 상승 100% / 하락 100%")
    print("\n공통:")
    print("  • t일 예측 → t+1 시가 매매")
    print("  • 예측 대상은 t+5 영업일")
    print("  • 만기 강제청산 없음")
    print("  • 중립 → 관망 + 연속성 초기화")
    print("  • 포지션 최대 100%")
    print("=" * 100)

    df = load_data()
    print(f"\n✅ 데이터 로드 완료: {len(df):,}개 행")

    start_date = pd.Timestamp("2023-01-01")
    end_date = pd.Timestamp("2024-08-22")

    strategies = ["A", "B", "C"]
    strategy_names = {
        "A": "A - 단계적 분할매매 (20-30-50%)",
        "B": "B - 고정 비중 (25%)",
        "C": "C - 올인/올아웃 (100%)",
    }

    results = {}

    for strategy in strategies:
        print(f"\n🔄 전략 실행 중: " f"{strategy_names[strategy]}")
        results[strategy] = run_backtest(
            market_data=df,
            start_date=start_date,
            end_date=end_date,
            predict_func=predict_5d_after,
            strategy=strategy,
            initial_cash=100.0,
            trade_cost=0.001,
            model_id="random-v0",
        )
        print(
            f"   ✅ 완료"
            f" | 거래 {results[strategy]['num_trades']}회"
            f" | 최종자산 "
            f"{results[strategy]['final_portfolio_value']:.2f}"
        )

    print("\n")
    print("=" * 110)
    print("📊 A / B / C 전략 백테스트 결과 비교")
    print("=" * 110)

    metric_defs = [
        ("최종 포트폴리오", "final_portfolio_value", "{:.2f}", False),
        ("총 수익률", "total_return", "{:.2f}%", True),
        ("연환산 수익률", "annual_return", "{:.2f}%", True),
        ("연환산 변동성", "volatility", "{:.2f}%", True),
        ("Sharpe Ratio", "sharpe_ratio", "{:.2f}", False),
        ("MDD", "max_drawdown", "{:.2f}%", True),
        ("일간 승률", "win_rate_daily", "{:.2f}%", True),
        ("평균 수익률", "avg_win", "{:.4f}%", True),
        ("평균 손실률", "avg_loss", "{:.4f}%", True),
        ("Profit Factor", "profit_factor", "{:.2f}", False),
        ("최대 연속 승리", "max_win_streak", "{:.0f}", False),
        ("최대 연속 패배", "max_loss_streak", "{:.0f}", False),
        ("총 거래 횟수", "num_trades", "{:.0f}", False),
        ("매수 거래", "buy_trades", "{:.0f}", False),
        ("매도 거래", "sell_trades", "{:.0f}", False),
        ("거래비용 합계", "total_transaction_cost", "{:.4f}", False),
        ("최종 투자비율", "final_position_ratio", "{:.2f}%", True),
    ]

    print(
        f"\n{'지표':<25} | " f"{'A 전략':>22} | " f"{'B 전략':>22} | " f"{'C 전략':>22}"
    )
    print("-" * 110)

    for label, key, fmt, is_pct in metric_defs:
        row = f"{label:<25} | "
        for strategy in strategies:
            value = results[strategy][key]
            if is_pct:
                value *= 100
            row += f"{fmt.format(value):>22} | "
        print(row)

    print("=" * 110)

    # 결과 업로드
    all_results = []
    for strategy in strategies:
        result = results[strategy]
        all_results.append({
            "strategy": strategy,
            "total_return": result["total_return"],
            "annual_return": result["annual_return"],
            "volatility": result["volatility"],
            "sharpe_ratio": result["sharpe_ratio"],
            "max_drawdown": result["max_drawdown"],
            "win_rate_daily": result["win_rate_daily"],
            "profit_factor": result["profit_factor"],
            "num_trades": result["num_trades"],
            "final_portfolio_value": result["final_portfolio_value"],
        })

    df_results = pd.DataFrame(all_results)
    dataset_results = Dataset.from_pandas(df_results)
    dataset_results.push_to_hub("qurious-quant/alphastack-backtest-results")
    print("\n✅ 백테스트 결과가 업로드되었습니다!")

    daily_returns_dict = {}
    for strategy in strategies:
        daily_returns_dict[f"{strategy}_daily_return"] = results[strategy]["daily_returns"]

    df_daily = pd.DataFrame(daily_returns_dict).reset_index()
    df_daily.rename(columns={"index": "date"}, inplace=True)

    dataset_daily = Dataset.from_pandas(df_daily)
    dataset_daily.push_to_hub("qurious-quant/alphastack-backtest-kospi200")
    print("✅ 일별 수익률 데이터도 업로드되었습니다!")