import cma
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

# 기존 피처 모듈 임포트
from step1_core_features import compute_bands, load_data
from tqdm import tqdm


# ============================================================
# 1. 포지션 생성 함수 (상승=+1, 중립=0, 하락=-1)
# ============================================================
def get_positions(
    df: pd.DataFrame,
    alpha_up: float,
    alpha_down: float,
    beta_up: float,
    beta_down: float,
) -> np.ndarray:
    """기준선을 계산하고 3가지 포지션을 반환합니다."""
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
    positions = np.where(pred == 2, 1, np.where(pred == 0, -1, 0))
    return positions


# ============================================================
# 2. 전략 성과 지표 계산 (Sharpe, MDD, CAGR, Calmar)
# ============================================================
def calculate_metrics(returns: np.ndarray) -> dict:
    """일별 수익률 시계열을 받아 Sharpe, MDD, CAGR, Calmar를 반환합니다."""
    ann_factor = np.sqrt(252)
    mean_ret = np.nanmean(returns)
    std_ret = np.nanstd(returns)
    sharpe = (mean_ret / std_ret) * ann_factor if std_ret != 0 else 0.0

    cum_ret = np.nanprod(1 + returns)
    n_years = len(returns) / 252
    cagr = (cum_ret ** (1 / n_years)) - 1 if n_years > 0 else 0.0

    cum_wealth = np.nancumprod(1 + returns)
    peak = np.maximum.accumulate(cum_wealth)
    drawdown = (peak - cum_wealth) / peak
    mdd = np.nanmax(drawdown) if len(drawdown) > 0 else 0.0

    calmar = cagr / mdd if mdd > 0 else 0.0
    return {"sharpe": sharpe, "cagr": cagr, "mdd": mdd, "calmar": calmar}


# ============================================================
# 3. 3단계 목적 함수 (CMA-ES가 최적화할 대상)
# ============================================================
def objective(params, df_train, md_threshold=0.20, lambda_mdd=2.0):
    """Fitness = Sharpe - lambda_mdd * max(0, MDD - md_threshold)"""
    alpha_up, alpha_down, beta_up, beta_down = params

    positions = get_positions(df_train, alpha_up, alpha_down, beta_up, beta_down)

    if 'code' in df_train.columns:
        market_ret = df_train.groupby('code')['close'].pct_change().values
    else:
        market_ret = df_train['close'].pct_change().values

    pos_shifted = np.roll(positions, 1)
    pos_shifted[0] = 0
    strategy_ret = pos_shifted * market_ret

    valid_mask = ~(np.isnan(strategy_ret) | np.isnan(market_ret))
    if valid_mask.sum() < 10:
        return 1.0

    clean_ret = strategy_ret[valid_mask]
    metrics = calculate_metrics(clean_ret)
    sharpe = metrics["sharpe"]
    mdd = metrics["mdd"]

    excess_mdd = max(0, mdd - md_threshold)
    penalty = lambda_mdd * excess_mdd
    fitness = sharpe - penalty

    return -fitness  # 최소화


# ============================================================
# 4. OOS 평가 함수 (분류 지표 포함) - Arrow 타입 오류 수정
# ============================================================
def evaluate_oos(df_oos, params):
    alpha_up, alpha_down, beta_up, beta_down = params

    # 기준선 계산
    bands = compute_bands(
        df_oos,
        base_type="SMA20",
        vol_type="ATR14",
        volume_type="LogRV20",
        asym=True,
        alpha_up=alpha_up,
        alpha_down=alpha_down,
        beta_up=beta_up,
        beta_down=beta_down,
    )
    close = df_oos["close"].values
    upper = bands["upper"].values
    lower = bands["lower"].values

    # 예측 클래스 (2:상승, 1:중립, 0:하락)
    y_pred = np.where(close > upper, 2, np.where(close < lower, 0, 1))
    positions = np.where(y_pred == 2, 1, np.where(y_pred == 0, -1, 0))

    # 수익률 기반 지표
    if 'code' in df_oos.columns:
        market_ret = df_oos.groupby('code')['close'].pct_change().values
    else:
        market_ret = df_oos['close'].pct_change().values

    pos_shifted = np.roll(positions, 1)
    pos_shifted[0] = 0
    strategy_ret = pos_shifted * market_ret

    valid_mask = ~(np.isnan(strategy_ret) | np.isnan(market_ret))
    if valid_mask.sum() > 10:
        clean_ret = strategy_ret[valid_mask]
        perf = calculate_metrics(clean_ret)
    else:
        perf = {"sharpe": np.nan, "cagr": np.nan, "mdd": np.nan, "calmar": np.nan}

    # 분류 지표 (실제 레이블)
    # 🔥 Arrow 배열 문제 해결: pd.isna() 사용 + NumPy 변환
    if "label" in df_oos.columns:
        # 문자열 → 숫자 매핑 후 NumPy 배열로 변환
        raw_labels = df_oos["label"].tolist()
        map_dict = {"상승": 2, "중립": 1, "하락": 0}
        y_true = np.array([map_dict.get(x, 1) for x in raw_labels], dtype=np.float64)
    else:
        if 'code' in df_oos.columns:
            ret = df_oos.groupby('code')['close'].pct_change().values
        else:
            ret = df_oos['close'].pct_change().values

        # 수익률에 따른 레이블링 (상승: +0.5% 이상, 하락: -0.5% 이하, 중립: 그 외)
        y_true = np.where(ret > 0.005, 2, np.where(ret < -0.005, 0, 1)).astype(np.float64)

    # y_pred도 NumPy로 보장 (이미 np.ndarray)
    y_pred = np.array(y_pred, dtype=np.float64)

    # 🔥 pd.isna()로 안전하게 NaN 검사
    cls_mask = ~(pd.isna(y_pred) | pd.isna(y_true))
    if cls_mask.sum() > 0:
        y_true_clean = y_true[cls_mask]
        y_pred_clean = y_pred[cls_mask]
        perf["f1_macro"] = f1_score(y_true_clean, y_pred_clean, average="macro")
        perf["balanced_acc"] = balanced_accuracy_score(y_true_clean, y_pred_clean)

        unique, counts = np.unique(y_pred_clean, return_counts=True)
        class_ratio = dict(zip(unique, counts / len(y_pred_clean), strict=False))
        perf["ratio_up"] = class_ratio.get(2, 0.0)
        perf["ratio_neutral"] = class_ratio.get(1, 0.0)
        perf["ratio_down"] = class_ratio.get(0, 0.0)
    else:
        perf["f1_macro"] = np.nan
        perf["balanced_acc"] = np.nan
        perf["ratio_up"] = np.nan
        perf["ratio_neutral"] = np.nan
        perf["ratio_down"] = np.nan

    return perf


# ============================================================
# 5. Walk-Forward 최적화 실행 (3단계 버전) - WFO 설정 및 경계 수정
# ============================================================
def run_walkforward_optimization(
    df: pd.DataFrame,
    train_years: int = 2,  # 수정: 2년
    val_months: int = 3,  # 수정: 3개월
    step_months: int = 1,  # 수정: 1개월
    max_evals: int = 300,
    md_threshold: float = 0.20,
    lambda_mdd: float = 2.0,
) -> pd.DataFrame:
    train_days = train_years * 252
    val_days = val_months * 21
    step_days = step_months * 21

    results = []
    total_len = len(df)

    print(f"📐 학습: {train_days}일, 검증: {val_days}일, 이동: {step_days}일")

    for start in tqdm(range(0, total_len - train_days - val_days + 1, step_days)):
        train_end = start + train_days
        val_end = train_end + val_days

        df_train = df.iloc[start:train_end].copy()
        df_val = df.iloc[train_end:val_end].copy()

        # 🔥 CMA-ES 설정 수정: α 범위 0.5~3.0, β 범위 -1.5~1.5, 초기값 [1,1,0,0]
        x0 = [1.0, 1.0, 0.0, 0.0]
        sigma0 = 0.5
        bounds_low = [0.5, 0.5, -1.5, -1.5]
        bounds_high = [3.0, 3.0, 1.5, 1.5]

        def obj_func(p, df_train=df_train, md_threshold=md_threshold, lambda_mdd=lambda_mdd):
            return objective(p, df_train, md_threshold=md_threshold, lambda_mdd=lambda_mdd)

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

        best_fitness = np.inf
        best_params = x0
        while not es.stop():
            solutions = es.ask()
            fitness = [obj_func(p) for p in solutions]
            es.tell(solutions, fitness)
            if es.result.fbest < best_fitness:
                best_fitness = es.result.fbest
                best_params = es.result.xbest

        # OOS 평가
        oos_metrics = evaluate_oos(df_val, best_params)

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
                "is_sharpe": -best_fitness,
                "oos_sharpe": oos_metrics.get("sharpe", np.nan),
                "oos_mdd": oos_metrics.get("mdd", np.nan),
                "oos_calmar": oos_metrics.get("calmar", np.nan),
                "oos_cagr": oos_metrics.get("cagr", np.nan),
                "oos_f1_macro": oos_metrics.get("f1_macro", np.nan),
                "oos_bal_acc": oos_metrics.get("balanced_acc", np.nan),
                "ratio_up": oos_metrics.get("ratio_up", np.nan),
                "ratio_neutral": oos_metrics.get("ratio_neutral", np.nan),
                "ratio_down": oos_metrics.get("ratio_down", np.nan),
            }
        )

        print(
            f"✅ 폴드 완료 | OOS Sharpe: {oos_metrics.get('sharpe', 0):.3f} | "
            f"OOS MDD: {oos_metrics.get('mdd', 0):.3f}"
        )

    return pd.DataFrame(results)


# ============================================================
# 6. 메인 실행
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 3단계: Sharpe + MDD 패널티 기반 Walk-Forward 최적화 (수정 완료)")
    print("=" * 60)

    df = load_data()
    print(f"📊 데이터 로드 완료: {df.shape[0]}일")

    result_df = run_walkforward_optimization(
        df,
        train_years=2,  # 설계 반영
        val_months=3,  # 설계 반영
        step_months=1,  # 설계 반영
        max_evals=300,
        md_threshold=0.20,
        lambda_mdd=2.0,
    )

    print("\n" + "=" * 60)
    print("📈 3단계 최적화 결과 요약")
    print("=" * 60)
    print(result_df.round(4).to_string())

    print("\n📊 평균 Out-of-Sample 성능:")
    print(f"   평균 OOS Sharpe: {result_df['oos_sharpe'].mean():.4f}")
    print(f"   평균 OOS MDD:    {result_df['oos_mdd'].mean():.4f}")
    print(f"   평균 OOS Calmar: {result_df['oos_calmar'].mean():.4f}")
    print(f"   평균 OOS F1:     {result_df['oos_f1_macro'].mean():.4f}")
    print(f"   평균 중립 비율:   {result_df['ratio_neutral'].mean():.4f}")

    # 중앙값 (통계 참고용)
    median_params = result_df[
        ["alpha_up", "alpha_down", "beta_up", "beta_down"]
    ].median()
    print("\n📊 전체 폴드 파라미터 중앙값 (통계 참고용 - 최종 모델 아님):")
    print(median_params.to_string())

    # 최고 OOS Sharpe 폴드
    best_idx = result_df["oos_sharpe"].idxmax()
    print(
        f"\n🏆 최고 OOS Sharpe 폴드 ({result_df.loc[best_idx, 'val_start']}"
        f"~{result_df.loc[best_idx, 'val_end']}):"
    )
    print(
        f"   α_up={result_df.loc[best_idx, 'alpha_up']:.4f}, "
        f"α_down={result_df.loc[best_idx, 'alpha_down']:.4f}"
    )
    print(
        f"   β_up={result_df.loc[best_idx, 'beta_up']:.4f}, "
        f"β_down={result_df.loc[best_idx, 'beta_down']:.4f}"
    )

    print("\n✅ 3단계 최적화 완료!")
    print("※ 각 폴드의 최적 파라미터는 해당 OOS에 Rolling 적용됨.")
