import warnings
from typing import Callable, Dict

import numpy as np
import pandas as pd
from datasets import Dataset
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore")

# backtest.py
# ============================================================
# KRX 실제 데이터 + 가상 예측(랜덤)을 이용한
# A / B / C 3가지 포지션 운용 전략 백테스트
#
# ============================================================
# [공통 전략 규칙]
#
# 1. 당일 t일 데이터를 이용하여 5영업일 후 방향을 예측
#    → 상승 / 중립 / 하락
#
# 2. 예측 결과는 다음 영업일 t+1 시가에 매매에 사용
#
# 3. 포지션에는 만기 청산이 없음
#    → 5영업일이라는 것은 "예측 시점과 예측 대상의 간격"
#    → 보유기간을 5일로 제한하는 것이 아님
#
# 4. 새로운 상승/하락 신호가 발생할 때마다
#    기존 포지션에 추가 매수/매도를 수행
#
# 5. 중립은 매매하지 않으며,
#    상승/하락 연속 횟수를 모두 초기화
#
# 6. 포지션은 전체 시그널을 하나의 포트폴리오로 합산하여 관리
#
# ============================================================
#
# [A 전략]
# 상승 연속:
#   1회 → +20%
#   2회 → +30%
#   3회 이상 → +50%
#
# 하락 연속:
#   1회 → -20%
#   2회 → -30%
#   3회 이상 → -50%
#
# 최대 투자비중: 100%
# 최소 투자비중: 0%
#
# ============================================================
#
# [B 전략]
# 상승 → 매번 +25%
# 하락 → 매번 -25%
#
# 최대 투자비중: 100%
# 최소 투자비중: 0%
#
# ============================================================
#
# [C 전략]
# 상승 → +100% (올인)
# 하락 → -100% (올아웃)
# 중립 → 관망
#
# 최대 투자비중: 100%
# 최소 투자비중: 0%
#
# ============================================================

# ============================================================
# 0. 데이터 로드
# ============================================================


def load_data() -> pd.DataFrame:
    """
    Hugging Face에서 KRX 데이터를 로드하고 정리합니다.

    필요한 컬럼:
        - date
        - open
        - close

    Returns
    -------
    pd.DataFrame
        날짜를 DatetimeIndex로 갖는 정렬된 시장 데이터
    """

    path = hf_hub_download(
        repo_id="qurious-quant/alphastack-krx-dev",
        filename="small/features_labels_kospi200_dev.csv",
        repo_type="dataset",
    )

    df = pd.read_csv(path)

    # 날짜 변환
    df["date"] = pd.to_datetime(df["date"])

    # 날짜순 정렬
    df.sort_values("date", inplace=True)

    # 날짜를 인덱스로 설정
    df.set_index("date", inplace=True)

    # 필수 컬럼 확인
    required_columns = ["open", "close"]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"데이터에 필요한 컬럼이 없습니다: {missing_columns}")

    return df


# ============================================================
# 1. 예측 함수
# ============================================================


def predict_5d_after(base_date: pd.Timestamp, market_data: pd.DataFrame) -> str:
    """
    5영업일 후 방향을 예측합니다.

    현재는 백테스트 구조 테스트를 위한 랜덤 예측입니다.
    실제 AI 모델을 사용할 경우 이 함수만 교체하면 됩니다.

    Parameters
    ----------
    base_date : pd.Timestamp
        예측 기준일 t

    market_data : pd.DataFrame
        시장 데이터

    Returns
    -------
    str
        "상승", "중립", "하락"
    """

    # 날짜마다 동일한 랜덤 결과가 나오도록 seed 고정
    np.random.seed(hash(base_date) % 2**32)

    labels = ["상승", "중립", "하락"]

    return np.random.choice(labels, p=[0.33, 0.34, 0.33])


# ============================================================
# 2. A / B / C 전략의 매매 비율 계산
# ============================================================


def get_trade_ratio_a(signal: str, consecutive_up: int, consecutive_down: int) -> float:
    """
    A 전략: 단계적 분할매매

    상승:
        1번째 → +20%
        2번째 → +30%
        3번째 이상 → +50%

    하락:
        1번째 → -20%
        2번째 → -30%
        3번째 이상 → -50%

    중립:
        0%

    주의
    ----
    여기서 반환하는 값은 '현재 목표 포지션'이 아니라
    '이번 신호에서 추가로 매매할 비율'입니다.

    예:
        현재 포지션 50%
        상승 1회
        → +20%
        → 최종 포지션 70%
    """

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
    """
    B 전략: 고정 25% 분할매매

    상승 → +25%
    하락 → -25%
    중립 → 0%

    연속 횟수와 관계없이 항상 동일한 비율로 거래합니다.
    """

    if signal == "상승":
        return 0.25

    elif signal == "하락":
        return -0.25

    else:
        return 0.0


def get_trade_ratio_c(signal: str) -> float:
    """
    C 전략: 올인 / 올아웃

    상승 → +100%
    하락 → -100%
    중립 → 0%

    현재 포지션보다 실제 매매 가능한 범위만큼만 실행합니다.
    """

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
    """
    선택된 전략에 따라 이번 신호의 매매 비율을 반환합니다.
    """

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
    """
    전체 시그널의 연속성을 관리합니다.

    규칙
    ----
    상승:
        상승 연속 횟수 +1
        하락 연속 횟수 = 0

    하락:
        하락 연속 횟수 +1
        상승 연속 횟수 = 0

    중립:
        두 카운트 모두 0

    예
    ----
    상승 → 상승 → 하락 → 상승

    결과:
        상승1
        상승2
        하락1
        상승1

    Returns
    -------
    tuple
        (new_consecutive_up, new_consecutive_down)
    """

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
) -> Dict:
    """
    A / B / C 전략 백테스트를 실행합니다.

    ------------------------------------------------------------
    핵심 시간 구조
    ------------------------------------------------------------

    t일
        ↓
    AI가 t+5 영업일 방향 예측
        ↓
    t+1일 시가
        ↓
    실제 매매
        ↓
    이후 포지션 유지
        ↓
    새로운 시그널 발생 시 추가 매매
        ↓
    만기 강제청산 없음

    ------------------------------------------------------------
    포지션 관리
    ------------------------------------------------------------

    position_ratio는 전체 자산 대비 주식 투자비율입니다.

        0.0 = 현금 100%
        0.5 = 주식 50%
        1.0 = 주식 100%

    공매도는 하지 않습니다.

    따라서:

        position_ratio < 0 → 0으로 제한
        position_ratio > 1 → 1로 제한

    ------------------------------------------------------------
    중요한 점
    ------------------------------------------------------------

    A/B/C 모두 '추가 매매' 방식입니다.

    예:

        현재 포지션 50%
        A 상승 1회
        → +20%
        → 70%

    현재 포지션이 90%인데 +20% 신호가 나오면:

        90% + 20% = 110%

    이므로 실제로는 10%만 매수하여
    최종 포지션을 100%로 제한합니다.

    반대로 포지션이 10%인데 -20% 매도 신호가 나오면:

        10% - 20% = -10%

    이므로 10%만 매도하여
    최종 포지션을 0%로 제한합니다.

    Parameters
    ----------
    market_data : pd.DataFrame
        KRX 시장 데이터

    start_date : pd.Timestamp
        백테스트 시작일

    end_date : pd.Timestamp
        백테스트 종료일

    predict_func : Callable
        t일 → t+5 방향을 반환하는 예측 함수

    strategy : str
        "A", "B", "C"

    initial_cash : float
        초기 투자금

    trade_cost : float
        거래비용 비율

    Returns
    -------
    Dict
        백테스트 결과
    """

    if strategy not in ["A", "B", "C"]:
        raise ValueError("strategy는 'A', 'B', 'C' 중 하나여야 합니다.")

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

    # 보유 주식 수량
    position_quantity = 0.0

    # 전체 자산 중 주식 투자비율
    position_ratio = 0.0

    # 연속 시그널
    consecutive_up = 0
    consecutive_down = 0

    # 결과 저장
    portfolio_values = []
    daily_returns = []
    trade_log = []
    signal_log = []

    # --------------------------------------------------------
    # 날짜별 백테스트
    # --------------------------------------------------------

    for i, date in enumerate(trading_days):

        # ====================================================
        # 1. 현재 날짜의 예측 생성
        # ====================================================

        signal = predict_func(date, market_data)

        # ----------------------------------------------------
        # 연속 시그널 업데이트
        # ----------------------------------------------------

        consecutive_up, consecutive_down = update_signal_streak(
            signal, consecutive_up, consecutive_down
        )

        # ====================================================
        # 2. 예측은 오늘, 매매는 다음날 시가
        # ====================================================

        # 마지막 거래일은 다음날 시가가 없으므로
        # 실제 매매를 실행할 수 없음
        if i < len(trading_days) - 1:

            execution_date = trading_days[i + 1]

            execution_price = open_prices.loc[execution_date]

            # -----------------------------------------------
            # 전략에 따른 추가 매매 비율
            # -----------------------------------------------

            trade_ratio = get_trade_ratio(
                strategy=strategy,
                signal=signal,
                consecutive_up=consecutive_up,
                consecutive_down=consecutive_down,
            )

            # -----------------------------------------------
            # 매매 전 포트폴리오 평가
            #
            # 다음날 시가 기준
            # -----------------------------------------------

            portfolio_value_before = cash + position_quantity * execution_price

            # -----------------------------------------------
            # 이번 매매에서 목표로 하는 금액
            #
            # 예:
            #   +0.20 → 전체 자산의 20% 추가 매수
            #   -0.20 → 전체 자산의 20% 매도
            # -----------------------------------------------

            requested_trade_value = portfolio_value_before * trade_ratio

            # 실제 거래 금액
            actual_trade_value = 0.0

            action = "hold"

            # =================================================
            # 상승 → 매수
            # =================================================

            if requested_trade_value > 0:

                # 현재 투자금
                current_investment = position_quantity * execution_price

                # 현재 투자비율
                current_ratio = (
                    current_investment / portfolio_value_before
                    if portfolio_value_before > 0
                    else 0.0
                )

                # 최대 투자 가능 금액
                max_additional_investment = portfolio_value_before * max(
                    0.0, 1.0 - current_ratio
                )

                # 실제 매수 금액
                actual_trade_value = min(
                    requested_trade_value, max_additional_investment
                )

                # 거래비용
                cost = actual_trade_value * trade_cost

                # 현금이 충분하지 않은 경우
                #
                # 여기서는 '현금 부족 시 매수 무시'가 아니라
                # 실제 가능한 범위까지만 매수하도록 처리합니다.
                #
                # 단, 전체 포트폴리오의 투자비율은
                # 최대 100%로 제한됩니다.
                available_cash = cash

                if available_cash <= cost:
                    actual_trade_value = 0.0
                    cost = 0.0

                else:
                    # 거래비용까지 고려하여
                    # 실제 사용할 수 있는 금액 제한
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

            # =================================================
            # 하락 → 매도
            # =================================================

            elif requested_trade_value < 0:

                requested_sell_value = abs(requested_trade_value)

                current_investment = position_quantity * execution_price

                # 보유하고 있는 투자금보다 많이 팔 수 없음
                actual_trade_value = min(requested_sell_value, current_investment)

                cost = actual_trade_value * trade_cost

                if actual_trade_value > 0:

                    quantity = actual_trade_value / execution_price

                    position_quantity -= quantity

                    cash += actual_trade_value - cost

                    action = "sell"

                else:
                    quantity = 0.0

            # =================================================
            # 중립
            # =================================================

            else:

                actual_trade_value = 0.0
                cost = 0.0
                quantity = 0.0
                action = "hold"

            # ------------------------------------------------
            # 매매 후 포지션 비율 계산
            # ------------------------------------------------

            portfolio_value_after = cash + position_quantity * execution_price

            if portfolio_value_after > 0:

                position_ratio = (
                    position_quantity * execution_price / portfolio_value_after
                )

            else:
                position_ratio = 0.0

            # ------------------------------------------------
            # 시그널 로그
            # ------------------------------------------------

            signal_log.append(
                {
                    "prediction_date": date,
                    "execution_date": execution_date,
                    "signal": signal,
                    "consecutive_up": consecutive_up,
                    "consecutive_down": consecutive_down,
                    "requested_trade_ratio": trade_ratio,
                    "actual_trade_value": actual_trade_value,
                    "position_ratio_after": position_ratio,
                }
            )

            # ------------------------------------------------
            # 거래 로그
            # ------------------------------------------------

            if action != "hold":

                trade_log.append(
                    {
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
                    }
                )

        # ====================================================
        # 3. 현재 날짜 종가 기준 포트폴리오 평가
        # ====================================================

        current_close = close_prices.loc[date]

        portfolio_value = cash + position_quantity * current_close

        portfolio_values.append(portfolio_value)

        # ====================================================
        # 4. 일간 수익률
        # ====================================================

        if len(portfolio_values) >= 2:

            previous_value = portfolio_values[-2]

            if previous_value != 0:

                daily_return = (portfolio_values[-1] - previous_value) / previous_value

            else:
                daily_return = 0.0

            daily_returns.append(daily_return)

    # ========================================================
    # 6. 시계열 생성
    # ========================================================

    portfolio_series = pd.Series(
        portfolio_values, index=trading_days[: len(portfolio_values)]
    )

    daily_returns_series = pd.Series(
        daily_returns, index=trading_days[1 : len(daily_returns) + 1]
    )

    # ========================================================
    # 7. 성과 지표
    # ========================================================

    if len(portfolio_series) == 0:
        raise ValueError("포트폴리오 시계열이 비어 있습니다.")

    # --------------------------------------------------------
    # 총 수익률
    # --------------------------------------------------------

    total_return = (portfolio_series.iloc[-1] / initial_cash) - 1

    # --------------------------------------------------------
    # 연환산 수익률
    # --------------------------------------------------------

    years = len(portfolio_series) / 252

    if years > 0:

        annual_return = ((portfolio_series.iloc[-1] / initial_cash) ** (1 / years)) - 1

    else:
        annual_return = np.nan

    # --------------------------------------------------------
    # 연환산 변동성
    # --------------------------------------------------------

    if len(daily_returns_series) > 1:

        volatility = daily_returns_series.std(ddof=1) * np.sqrt(252)

    else:
        volatility = np.nan

    # --------------------------------------------------------
    # Sharpe Ratio
    #
    # 무위험수익률은 현재 0% 가정
    # --------------------------------------------------------

    if len(daily_returns_series) > 1 and daily_returns_series.std(ddof=1) != 0:

        sharpe = (
            daily_returns_series.mean() / daily_returns_series.std(ddof=1)
        ) * np.sqrt(252)

    else:
        sharpe = np.nan

    # --------------------------------------------------------
    # MDD
    # --------------------------------------------------------

    running_max = portfolio_series.cummax()

    drawdown = (portfolio_series / running_max) - 1

    max_drawdown = drawdown.min()

    # ========================================================
    # 8. 일간 기본 통계
    # ========================================================

    positive_days = daily_returns_series > 0

    negative_days = daily_returns_series < 0

    neutral_days = daily_returns_series == 0

    num_winning_days = int(positive_days.sum())

    num_losing_days = int(negative_days.sum())

    num_neutral_days = int(neutral_days.sum())

    total_days = len(daily_returns_series)

    win_rate_daily = num_winning_days / total_days if total_days > 0 else 0.0

    # --------------------------------------------------------
    # 평균 수익 / 평균 손실
    # --------------------------------------------------------

    avg_win = (
        daily_returns_series[positive_days].mean() if num_winning_days > 0 else 0.0
    )

    avg_loss = (
        daily_returns_series[negative_days].mean() if num_losing_days > 0 else 0.0
    )

    # --------------------------------------------------------
    # Profit Factor
    # --------------------------------------------------------

    total_gain = (
        daily_returns_series[positive_days].sum() if num_winning_days > 0 else 0.0
    )

    total_loss = (
        abs(daily_returns_series[negative_days].sum()) if num_losing_days > 0 else 0.0
    )

    profit_factor = total_gain / total_loss if total_loss > 0 else np.inf

    # ========================================================
    # 9. 최대 연속 승리 / 패배
    # ========================================================

    def get_max_streak(condition: pd.Series) -> int:

        if len(condition) == 0:
            return 0

        groups = (condition != condition.shift()).cumsum()

        streaks = condition.astype(int).groupby(groups).sum()

        return int(streaks.max() if len(streaks) > 0 else 0)

    max_win_streak = get_max_streak(daily_returns_series > 0)

    max_loss_streak = get_max_streak(daily_returns_series < 0)

    # ========================================================
    # 10. 거래 통계
    # ========================================================

    num_trades = len(trade_log)

    buy_trades = sum(1 for trade in trade_log if trade["action"] == "buy")

    sell_trades = sum(1 for trade in trade_log if trade["action"] == "sell")

    total_transaction_cost = sum(trade["cost"] for trade in trade_log)

    # ========================================================
    # 11. 결과 반환
    # ========================================================

    return {
        # 전략
        "strategy": strategy,
        # 시계열
        "portfolio_series": portfolio_series,
        "daily_returns": daily_returns_series,
        # 로그
        "trade_log": pd.DataFrame(trade_log),
        "signal_log": pd.DataFrame(signal_log),
        # 성과
        "total_return": total_return,
        "annual_return": annual_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        # 최종 상태
        "final_portfolio_value": portfolio_series.iloc[-1],
        "final_cash": cash,
        "final_position_quantity": position_quantity,
        "final_position_ratio": position_ratio,
        # 기본 통계
        "num_winning_days": num_winning_days,
        "num_losing_days": num_losing_days,
        "num_neutral_days": num_neutral_days,
        "win_rate_daily": win_rate_daily,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
        # 거래 통계
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

    # ========================================================
    # 데이터 로드
    # ========================================================

    df = load_data()

    print(f"\n✅ 데이터 로드 완료: {len(df):,}개 행")

    # ========================================================
    # 백테스트 기간
    # ========================================================

    start_date = pd.Timestamp("2023-01-01")
    end_date = pd.Timestamp("2024-08-22")

    # ========================================================
    # 전략 정의
    # ========================================================

    strategies = ["A", "B", "C"]

    strategy_names = {
        "A": "A - 단계적 분할매매 (20-30-50%)",
        "B": "B - 고정 비중 (25%)",
        "C": "C - 올인/올아웃 (100%)",
    }

    results = {}

    # ========================================================
    # 각 전략 실행
    # ========================================================

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
        )

        print(
            f"   ✅ 완료"
            f" | 거래 {results[strategy]['num_trades']}회"
            f" | 최종자산 "
            f"{results[strategy]['final_portfolio_value']:.2f}"
        )

    # ========================================================
    # 13. 결과 비교 (콘솔 출력만)
    # ========================================================

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

    # ========================================================
    # 14. 전략 설명
    # ========================================================

    print("\n")
    print("=" * 110)
    print("📌 전략 해석")
    print("=" * 110)

    print("\n[A 전략 - 단계적 분할매매]")
    print("상승 연속: 20% → 30% → 50%")
    print("하락 연속: 20% → 30% → 50%")
    print("반대 방향 또는 중립이 나오면 해당 연속 카운트는 다시 1부터 시작합니다.")

    print("\n[B 전략 - 고정 비중]")
    print("상승마다 +25%, 하락마다 -25%를 적용합니다.")

    print("\n[C 전략 - 올인/올아웃]")
    print("상승 → 100% 투자")
    print("하락 → 0% 투자")
    print("중립 → 기존 포지션 유지")

    print("\n[공통]")
    print("• 예측일: t")
    print("• 예측 대상: t+5 영업일")
    print("• 실제 매매: t+1 시가")
    print("• 만기 강제청산: 없음")
    print("• 최대 투자비율: 100%")
    print("• 공매도: 없음")
    print("• 중립: 매매하지 않고 연속성 초기화")

    print("\n⚠️ 현재 predict_5d_after()는 랜덤 예측입니다.")
    print(
        "   실제 AI 모델로 교체하면 동일한 백테스트 구조를 그대로 사용할 수 있습니다."
    )

    # ========================================================
    # 각 전략의 결과를 데이터프레임으로 변환
    # ========================================================

    all_results = []

    for strategy in strategies:
        result = results[strategy]

        # 성과 지표를 딕셔너리로 추출
        result_dict = {
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
        }
        all_results.append(result_dict)

    # DataFrame으로 변환
    df_results = pd.DataFrame(all_results)

    # Dataset으로 변환하여 업로드
    dataset_results = Dataset.from_pandas(df_results)
    dataset_results.push_to_hub("qurious-quant/alphastack-backtest-results")

    print("\n✅ 백테스트 결과가 업로드되었습니다!")

    # ========================================================
    # (선택) 각 전략의 일별 수익률도 함께 업로드
    # ========================================================

    # 각 전략의 일별 수익률을 하나의 DataFrame으로 합치기
    daily_returns_dict = {}
    for strategy in strategies:
        daily_returns_dict[f"{strategy}_daily_return"] = results[strategy][
            "daily_returns"
        ]

    df_daily = pd.DataFrame(daily_returns_dict)

    # 날짜 인덱스 초기화 (업로드용)
    df_daily = df_daily.reset_index()
    df_daily.rename(columns={"index": "date"}, inplace=True)

    # Dataset으로 변환하여 업로드
    dataset_daily = Dataset.from_pandas(df_daily)
    dataset_daily.push_to_hub("qurious-quant/alphastack-backtest-kospi200")

    print("✅ 일별 수익률 데이터도 업로드되었습니다!")
