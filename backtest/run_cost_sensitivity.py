# backtest/run_cost_sensitivity.py
"""
거래비용 민감도 분석 (Cost Sensitivity Analysis)
- 동일한 전략(A/B/C)을 4가지 왕복비용으로 각각 백테스트
- 결과를 Hugging Face Dataset으로 업로드
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import warnings

warnings.filterwarnings("ignore")

# 1) 기존 백테스트 엔진 가져오기
from backtest_strategies import run_backtest, load_data, predict_5d_after

# 2) Hugging Face datasets
from datasets import Dataset

# 3) 비용 사전 정의 (이미지 기준)
COST_PRESETS = {
    "etf": 0.0005,        # 0.05% - ETF 왕복 (주 시나리오)
    "stock_min": 0.0023,  # 0.23% - 거래세 0.20 + 수수료 0.03
    "stock_slip": 0.0030, # 0.30% - 위 + 슬리피지 0.07 (잔차 추정)
    "stress": 0.0050,     # 0.50% - 보수 가정 (스트레스 테스트)
}

COST_LABELS = {
    "etf": "수수료 0.05% (ETF)",
    "stock_min": "거래세 0.20 + 수수료 0.03",
    "stock_slip": "0.23 + 슬리피지 0.07(잔차)",
    "stress": "보수 가정 0.50%",
}

STRATEGIES = ["A", "B", "C"]


def find_breakeven_cost(
    strategy: str,
    market_data: pd.DataFrame,
    predict_func,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    cost_min: float = 0.0001,
    cost_max: float = 0.02,
    tol: float = 0.0001,
) -> float:
    """
    이분 탐색으로 Sharpe Ratio = 0 이 되는 왕복비용(손익분기점)을 찾습니다.
    """
    if cost_min >= cost_max:
        return cost_min

    for _ in range(30):  # 최대 30회 반복 (0.01% 정밀도 보장)
        mid = (cost_min + cost_max) / 2
        result = run_backtest(
            market_data=market_data,
            start_date=start_date,
            end_date=end_date,
            predict_func=predict_func,
            strategy=strategy,
            initial_cash=100.0,
            trade_cost=mid,
        )
        sharpe = result["sharpe_ratio"]

        if sharpe > 0:
            cost_min = mid  # 더 높은 비용을 견딤
        else:
            cost_max = mid  # 더 낮은 비용에서 깨짐

        if (cost_max - cost_min) < tol:
            break

    return (cost_min + cost_max) / 2


def run_cost_sensitivity():
    print("=" * 80)
    print("📊 거래비용 민감도 분석 (Cost Sensitivity) 시작")
    print("=" * 80)

    # 데이터 로드
    df = load_data()
    start_date = pd.Timestamp("2023-01-01")
    end_date = pd.Timestamp("2024-08-22")

    all_results = []

    # 1) 4가지 비용 × 3개 전략 = 12회 백테스트 실행
    for cost_key, cost_rate in COST_PRESETS.items():
        cost_label = COST_LABELS[cost_key]

        for strategy in STRATEGIES:
            print(
                f"▶ 실행 중: 전략 {strategy} | 비용 {cost_label} ({cost_rate*100:.2f}%) ...",
                end="",
                flush=True,
            )

            result = run_backtest(
                market_data=df,
                start_date=start_date,
                end_date=end_date,
                predict_func=predict_5d_after,
                strategy=strategy,
                initial_cash=100.0,
                trade_cost=cost_rate,
            )

            # 결과 저장
            all_results.append(
                {
                    "strategy": strategy,
                    "cost_key": cost_key,
                    "cost_rate": cost_rate,
                    "cost_label": cost_label,
                    "final_value": result["final_portfolio_value"],
                    "total_return": result["total_return"],
                    "annual_return": result["annual_return"],
                    "volatility": result["volatility"],
                    "sharpe_ratio": result["sharpe_ratio"],
                    "max_drawdown": result["max_drawdown"],
                    "win_rate": result["win_rate_daily"],
                    "profit_factor": result["profit_factor"],
                    "num_trades": result["num_trades"],
                    "turnover": result.get("turnover", np.nan),  # 기존 코드에 turnover가 없으면 NaN
                }
            )
            print(" ✅ 완료")

    # 2) 결과를 DataFrame으로 변환
    df_results = pd.DataFrame(all_results)

    # 3) 각 전략별 손익분기점(BEP) 계산 (Sharpe=0이 되는 비용)
    print("\n" + "=" * 80)
    print("🔍 손익분기점(Breakeven Cost) 계산 중 (Sharpe Ratio = 0 지점)...")
    bep_results = []
    for strategy in STRATEGIES:
        bep = find_breakeven_cost(
            strategy=strategy,
            market_data=df,
            predict_func=predict_5d_after,
            start_date=start_date,
            end_date=end_date,
        )
        bep_results.append({"strategy": strategy, "breakeven_cost": bep})
        print(f"   전략 {strategy} → 손익분기 왕복비용: {bep*100:.2f}%")

    df_bep = pd.DataFrame(bep_results)

    # ============================================================
    # 4) Hugging Face Dataset으로 업로드
    # ============================================================
    # 결과 요약 (df_results) 업로드
    dataset_results = Dataset.from_pandas(df_results)
    dataset_results.push_to_hub("qurious-quant/alphastack-cost-sensitivity")

    # 손익분기점 (df_bep) 업로드
    dataset_bep = Dataset.from_pandas(df_bep)
    dataset_bep.push_to_hub("qurious-quant/alphastack-breakeven-cost")

    print("\n✅ Hugging Face 업로드 완료!")
    print(f"   - https://huggingface.co/datasets/qurious-quant/alphastack-cost-sensitivity")
    print(f"   - https://huggingface.co/datasets/qurious-quant/alphastack-breakeven-cost")

    # 5) 결과 요약 출력 (이미지 #4 스타일)
    print("\n" + "=" * 80)
    print("📋 비용 민감도 요약 (Sharpe Ratio 기준)")
    print("=" * 80)
    for strategy in STRATEGIES:
        print(f"\n[전략 {strategy}]")
        sub = df_results[df_results["strategy"] == strategy].sort_values("cost_rate")
        for _, row in sub.iterrows():
            print(
                f"  {row['cost_label']:>20} : Sharpe {row['sharpe_ratio']:.3f}  |  MDD {row['max_drawdown']*100:.2f}%  |  수익률 {row['total_return']*100:.2f}%"
            )
        bep = df_bep[df_bep["strategy"] == strategy]["breakeven_cost"].values[0]
        print(f"  👉 손익분기 왕복비용: {bep*100:.2f}% (Sharpe=0)")

    print("\n" + "=" * 80)
    print("✅ 분석 완료!")


if __name__ == "__main__":
    run_cost_sensitivity()