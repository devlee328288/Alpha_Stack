import cma
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

# 기존 피처 모듈 (compute_bands, load_data) - 실제 환경에 맞게 임포트
from step1_core_features import compute_bands, load_data
from tqdm import tqdm


# ============================================================
# 1. 실제 레이블(y_true) 생성 (평가용)
# ============================================================
def create_true_labels(df: pd.DataFrame, threshold: float = 0.005) -> np.ndarray:
    """당일 종가 수익률 기준 3개 클래스 레이블 (0:하락, 1:중립, 2:상승) -> NumPy 배열 반환"""
    ret = df["close"].pct_change()
    labels = np.where(ret > threshold, 2, np.where(ret < -threshold, 0, 1))
    return labels  # 이미 np.ndarray


# ============================================================
# 2. 파라미터 -> 예측 레이블 (y_pred) 생성 (4개 파라미터)
# ============================================================
def predict_labels(
    df: pd.DataFrame,
    alpha_up: float,
    alpha_down: float,
    beta_up: float,
    beta_down: float,
) -> np.ndarray:
    """
    주어진 파라미터로 기준선 계산 후 3개 클래스(0,1,2) 예측.
    base/vol/volume 기간은 고정 (SMA20, ATR14, LogRV20)
    """
    bands = compute_bands(
        df,
        base_type="SMA20",
        vol_type="ATR14",
        volume_type="LogRV20",
        asym=True,
        alpha_up=alpha_up,
        alpha_down=alpha_down,
        beta_up=beta_up,
        beta_down=beta_down,
    )
    close = df["close"].values
    upper = bands["upper"].values
    lower = bands["lower"].values
    pred = np.where(close > upper, 2, np.where(close < lower, 0, 1))
    return pred


# ============================================================
# 3. CMA-ES 목적 함수 (Sharpe + MDD 패널티, 최소화)
# ============================================================
def objective(params, df_train):
    """
    CMA-ES가 최소화하는 함수 (음의 피트니스).
    params = [alpha_up, alpha_down, beta_up, beta_down]
    """
    alpha_up, alpha_down, beta_up, beta_down = params
    try:
        y_pred = predict_labels(df_train, alpha_up, alpha_down, beta_up, beta_down)
        # 포지션: 상승=1, 중립=0, 하락=-1
        position = np.where(y_pred == 2, 1, np.where(y_pred == 0, -1, 0))

        # 익일 수익률에 적용 (look‑ahead 방지)
        market_ret = df_train["close"].pct_change().values
        pos_shifted = np.roll(position, 1)
        pos_shifted[0] = 0
        strategy_ret = pos_shifted * market_ret
        strategy_ret = strategy_ret[~np.isnan(strategy_ret)]

        if len(strategy_ret) < 20:
            return 1e6

        mean_ret = np.nanmean(strategy_ret)
        std_ret = np.nanstd(strategy_ret)
        if std_ret == 0 or np.isnan(std_ret):
            return 1e6

        sharpe = (mean_ret / std_ret) * np.sqrt(252)

        equity = (1 + strategy_ret).cumprod()
        peak = np.maximum.accumulate(equity)
        drawdown = equity / peak - 1
        mdd = abs(np.nanmin(drawdown))

        penalty = 2.0 * max(0, mdd - 0.20)
        fitness = sharpe - penalty

        return -fitness

    except Exception:
        return 1e6


# ============================================================
# 4. Walk-Forward 최적화 실행
# ============================================================
def run_walkforward_optimization(
    df: pd.DataFrame,
    train_years: int = 2,
    val_months: int = 3,
    step_months: int = 1,
    max_evals: int = 300,
) -> pd.DataFrame:
    train_days = train_years * 252
    val_days = val_months * 21
    step_days = step_months * 21

    # 🔥 수정: label 컬럼이 한글 문자열이면 to_list()로 안전하게 가져와서 매핑
    if "label" in df.columns:
        raw_labels = df["label"].tolist()  # Arrow 배열도 Python list로 변환됨
        map_dict = {"상승": 2, "중립": 1, "하락": 0}
        numeric_labels = [
            map_dict.get(x, 1) for x in raw_labels
        ]  # 매핑 안 되면 중립(1)
        y_true_all = np.array(numeric_labels, dtype=np.float64)
        print("✅ 'label' 컬럼 사용 (매핑 완료)")
    else:
        print("⚠️ 'label' 없음 → 당일 수익률(±0.5%)로 생성")
        y_true_all = create_true_labels(df, threshold=0.005)
        if not isinstance(y_true_all, np.ndarray):
            y_true_all = np.array(y_true_all, dtype=np.float64)

    results = []
    total_len = len(df)

    for start in tqdm(range(0, total_len - train_days - val_days + 1, step_days)):
        train_end = start + train_days
        val_end = train_end + val_days

        df_train = df.iloc[start:train_end]
        df_val = df.iloc[train_end:val_end]

        y_val = np.array(y_true_all[train_end:val_end], dtype=np.float64)

        # CMA-ES 설정
        x0 = [1.0, 1.0, 0.0, 0.0]
        sigma0 = 0.5
        bounds_low = [0.5, 0.5, -1.5, -1.5]
        bounds_high = [3.0, 3.0, 1.5, 1.5]

        es = cma.CMAEvolutionStrategy(
            x0,
            sigma0,
            {
                "maxfevals": max_evals,
                "bounds": [bounds_low, bounds_high],
                "verbose": -1,
                "CMA_diagonal": True,
            },
        )

        best_fitness = 1e6
        best_params = x0

        while not es.stop():
            solutions = es.ask()
            fitness = [objective(p, df_train) for p in solutions]
            es.tell(solutions, fitness)
            if es.result.fbest < best_fitness:
                best_fitness = es.result.fbest
                best_params = es.result.xbest

        # OOS 평가
        y_pred_val = predict_labels(df_val, *best_params)
        y_pred_val = np.array(y_pred_val, dtype=np.float64)

        # 🔥 pd.isna()로 Arrow 타입도 안전하게 처리
        valid_mask = ~(pd.isna(y_pred_val) | pd.isna(y_val))

        if valid_mask.sum() > 0:
            val_f1 = f1_score(
                y_val[valid_mask], y_pred_val[valid_mask], average="macro"
            )
            val_bal_acc = balanced_accuracy_score(
                y_val[valid_mask], y_pred_val[valid_mask]
            )
        else:
            val_f1 = np.nan
            val_bal_acc = np.nan

        # 전략 수익률
        position_val = np.where(y_pred_val == 2, 1, np.where(y_pred_val == 0, -1, 0))
        market_ret_val = df_val["close"].pct_change().values
        pos_shifted_val = np.roll(position_val, 1)
        pos_shifted_val[0] = 0
        strat_ret_val = pos_shifted_val * market_ret_val
        strat_ret_val = strat_ret_val[~np.isnan(strat_ret_val)]

        if len(strat_ret_val) >= 20:
            mean_ret = np.nanmean(strat_ret_val)
            std_ret = np.nanstd(strat_ret_val)
            if std_ret > 0 and not np.isnan(std_ret):
                sharpe_val = (mean_ret / std_ret) * np.sqrt(252)
            else:
                sharpe_val = 0.0
            equity_val = (1 + strat_ret_val).cumprod()
            peak_val = np.maximum.accumulate(equity_val)
            dd_val = equity_val / peak_val - 1
            mdd_val = abs(np.nanmin(dd_val))
            calmar_val = sharpe_val / mdd_val if mdd_val > 0 else np.nan
        else:
            sharpe_val = np.nan
            mdd_val = np.nan
            calmar_val = np.nan

        results.append(
            {
                "train_start": df.index[start],
                "train_end": df.index[train_end - 1],
                "val_start": df.index[train_end],
                "val_end": df.index[val_end - 1],
                "alpha_up": best_params[0],
                "alpha_down": best_params[1],
                "beta_up": best_params[2],
                "beta_down": best_params[3],
                "val_f1": val_f1,
                "val_bal_acc": val_bal_acc,
                "val_sharpe": sharpe_val,
                "val_mdd": mdd_val,
                "val_calmar": calmar_val,
            }
        )

        print(
            f"✅ 폴드 완료: Sharpe={sharpe_val:.3f}, MDD={mdd_val:.3f}, F1={val_f1:.3f}"
        )

    return pd.DataFrame(results)


# ============================================================
# 5. 메인 실행
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Walk‑Forward + CMA‑ES 최적화 (Sharpe + MDD 패널티)")
    print("=" * 60)

    df = load_data()
    print(f"📊 데이터: {df.shape[0]}일")

    result_df = run_walkforward_optimization(
        df,
        train_years=2,
        val_months=3,
        step_months=1,
        max_evals=300,
    )

    print("\n" + "=" * 60)
    print("📈 최적화 결과 요약")
    print("=" * 60)
    print(result_df.round(4).to_string())

    print("\n📊 평균 OOS 성능:")
    print(f"   Sharpe : {result_df['val_sharpe'].mean():.4f}")
    print(f"   MDD    : {result_df['val_mdd'].mean():.4f}")
    print(f"   Calmar : {result_df['val_calmar'].mean():.4f}")
    print(f"   F1     : {result_df['val_f1'].mean():.4f}")
    print(f"   BalAcc : {result_df['val_bal_acc'].mean():.4f}")

    median_params = result_df[
        ["alpha_up", "alpha_down", "beta_up", "beta_down"]
    ].median()
    print("\n📊 전체 폴드 파라미터 중앙값 (통계 참고용):")
    print(median_params.to_string())

    print("\n✅ 최적화 완료!")
    print("※ 각 폴드의 최적 파라미터는 해당 OOS에 이미 Rolling 적용됨.")
    print("※ 위 중앙값은 최종 모델 파라미터가 아닌 분포 확인용입니다.")
