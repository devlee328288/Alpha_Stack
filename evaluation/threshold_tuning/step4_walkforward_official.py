import warnings

import cma
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
from step1_core_features import compute_bands, load_data
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ============================================================
# 1. 포지션 생성 함수 (상승=+1, 중립=0, 하락=-1)
# ============================================================
def get_positions(
    df: pd.DataFrame,
    alpha_up: float,
    alpha_down: float,
    beta_up: float,
    beta_down: float,
) -> tuple:
    """기준선을 계산하고 3가지 포지션과 예측 클래스를 반환합니다."""
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

    # 예측 클래스: 상승=2, 중립=1, 하락=0
    pred = np.where(close > upper, 2, np.where(close < lower, 0, 1))
    # 포지션: 상승=+1, 중립=0, 하락=-1
    positions = np.where(pred == 2, 1, np.where(pred == 0, -1, 0))
    return positions, pred


# ============================================================
# 2. 전략 성과 지표 계산 (연결된 OOS용)
# ============================================================
def calculate_metrics(returns: np.ndarray) -> dict:
    """
    일별 수익률 시계열을 받아 Sharpe, MDD, CAGR, Calmar, 승률, PF를 반환합니다.
    """
    if len(returns) == 0 or np.all(np.isnan(returns)):
        return {
            "sharpe": np.nan,
            "cagr": np.nan,
            "mdd": np.nan,
            "calmar": np.nan,
            "win_rate": np.nan,
            "profit_factor": np.nan,
        }

    clean_ret = returns[~np.isnan(returns)]
    if len(clean_ret) == 0:
        return {
            "sharpe": np.nan,
            "cagr": np.nan,
            "mdd": np.nan,
            "calmar": np.nan,
            "win_rate": np.nan,
            "profit_factor": np.nan,
        }

    ann_factor = np.sqrt(252)
    mean_ret = np.nanmean(clean_ret)
    std_ret = np.nanstd(clean_ret)
    sharpe = (mean_ret / std_ret) * ann_factor if std_ret != 0 else 0.0

    cum_ret = np.nanprod(1 + clean_ret)
    n_years = len(clean_ret) / 252
    cagr = (cum_ret ** (1 / n_years)) - 1 if n_years > 0 else 0.0

    cum_wealth = np.nancumprod(1 + clean_ret)
    peak = np.maximum.accumulate(cum_wealth)
    drawdown = (peak - cum_wealth) / peak
    mdd = np.nanmax(drawdown) if len(drawdown) > 0 else 0.0

    calmar = cagr / mdd if mdd > 0 else 0.0
    win_rate = np.mean(clean_ret > 0) if len(clean_ret) > 0 else 0.0

    gains = clean_ret[clean_ret > 0].sum()
    losses = abs(clean_ret[clean_ret < 0].sum())
    profit_factor = gains / losses if losses > 0 else np.inf

    return {
        "sharpe": sharpe,
        "cagr": cagr,
        "mdd": mdd,
        "calmar": calmar,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
    }


# ============================================================
# 3. 3단계 목적 함수 (CMA-ES 최적화용) - IS(학습) 구간에서만 사용
# ============================================================
def objective(params, df_train, md_threshold=0.20, lambda_mdd=2.0):
    """
    CMA-ES가 최소화하는 함수 (음의 Fitness).
    Fitness = Sharpe - lambda_mdd * max(0, MDD - md_threshold)
    """
    alpha_up, alpha_down, beta_up, beta_down = params

    positions, _ = get_positions(df_train, alpha_up, alpha_down, beta_up, beta_down)
    market_ret = df_train["close"].pct_change().values

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
    fitness = sharpe - (lambda_mdd * excess_mdd)

    return -fitness  # CMA-ES는 최소화


# ============================================================
# 4. 메인 Walk-Forward 실행 (문서 사양: 2년 학습, 3개월 검증, 1개월 이동)
# ============================================================
def run_walkforward_official(
    df: pd.DataFrame,
    train_years: int = 2,
    val_months: int = 3,
    step_months: int = 1,
    max_evals: int = 200,
    md_threshold: float = 0.20,
    lambda_mdd: float = 2.0,
) -> dict:
    """
    문서의 4단계를 정확히 구현합니다.
    - 학습(In-Sample): 2년
    - 검증(Out-of-Sample): 3개월
    - 이동 단위: 1개월 (Rolling Window)
    - 모든 OOS 결과를 시간순으로 연결하여 최종 성과 평가
    """
    train_days = train_years * 252
    val_days = val_months * 21
    step_days = step_months * 21

    total_len = len(df)
    print(f"📅 전체 데이터: {total_len}일")
    print(
        f"📐 학습 구간: {train_days}일, 검증 구간: {val_days}일, 이동 간격: {step_days}일"
    )

    all_oos_returns = []
    all_oos_y_true = []
    all_oos_y_pred = []
    all_oos_dates = []

    fold_details = []
    total_folds = 0

    for start in tqdm(range(0, total_len - train_days - val_days + 1, step_days)):
        train_end = start + train_days
        val_end = train_end + val_days

        df_train = df.iloc[start:train_end].copy()
        df_val = df.iloc[train_end:val_end].copy()

        # ---- In-Sample 최적화 (CMA-ES) ----
        # 🔥 수정: α, β 범위 및 초기값 설계 반영
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

        alpha_up, alpha_down, beta_up, beta_down = best_params

        # ---- Out-of-Sample 적용 ----
        positions, preds = get_positions(
            df_val, alpha_up, alpha_down, beta_up, beta_down
        )

        # 🔥 Arrow 타입 오류 방지: y_true를 안전하게 NumPy 배열로 변환
        if "label" in df_val.columns:
            raw_labels = df_val["label"].tolist()
            # 문자열 레이블이면 숫자로 매핑
            if isinstance(raw_labels[0], str):
                map_dict = {"상승": 2, "중립": 1, "하락": 0}
                y_true = np.array(
                    [map_dict.get(x, 1) for x in raw_labels], dtype=np.float64
                )
            else:
                y_true = np.array(raw_labels, dtype=np.float64)
        else:
            ret = df_val["close"].pct_change().values
            y_true = np.where(ret > 0.005, 2, np.where(ret < -0.005, 0, 1)).astype(
                np.float64
            )

        # 전략 수익률 (Look-ahead 방지)
        market_ret = df_val["close"].pct_change().values
        pos_shifted = np.roll(positions, 1)
        pos_shifted[0] = 0
        strategy_ret = pos_shifted * market_ret

        # 🔥 유효 마스크: 모든 배열이 NumPy이므로 np.isnan 사용 가능
        valid_mask = ~(
            np.isnan(strategy_ret)
            | np.isnan(market_ret)
            | np.isnan(y_true)
            | np.isnan(preds)
        )

        if valid_mask.sum() > 0:
            all_oos_returns.extend(strategy_ret[valid_mask].tolist())
            all_oos_y_true.extend(y_true[valid_mask].tolist())
            all_oos_y_pred.extend(preds[valid_mask].tolist())
            all_oos_dates.extend(df_val.index[valid_mask].tolist())

        fold_details.append(
            {
                "train_start": df.index[start],
                "train_end": df.index[train_end - 1],
                "val_start": df.index[train_end],
                "val_end": df.index[val_end - 1],
                "alpha_up": alpha_up,
                "alpha_down": alpha_down,
                "beta_up": beta_up,
                "beta_down": beta_down,
                "is_fitness": -best_fitness,
                "oos_ret_mean": (
                    np.nanmean(strategy_ret[valid_mask])
                    if valid_mask.sum() > 0
                    else np.nan
                ),
                "oos_ret_std": (
                    np.nanstd(strategy_ret[valid_mask])
                    if valid_mask.sum() > 0
                    else np.nan
                ),
            }
        )
        total_folds += 1

        if total_folds % 10 == 0:
            current_oos_date = df.index[val_end-1].strftime('%Y-%m-%d')
            print(f"   → {total_folds}개 폴드 완료 "
                  f"(현재 OOS: {current_oos_date})")

    # ---- 연결된 OOS 결과로 최종 성능 평가 ----
    print("\n" + "=" * 60)
    print("📊 연결된 OOS 결과 분석 (전체 기간)")
    print("=" * 60)

    oos_returns = np.array(all_oos_returns)
    perf_metrics = calculate_metrics(oos_returns)

    y_true_arr = np.array(all_oos_y_true)
    y_pred_arr = np.array(all_oos_y_pred)

    cls_metrics = {}
    if len(y_true_arr) > 0:
        cls_metrics["f1_macro"] = f1_score(y_true_arr, y_pred_arr, average="macro")
        cls_metrics["balanced_acc"] = balanced_accuracy_score(y_true_arr, y_pred_arr)
        unique, counts = np.unique(y_pred_arr, return_counts=True)
        ratio_dict = dict(zip(unique, counts / len(y_pred_arr), strict=False))
        cls_metrics["ratio_up"] = ratio_dict.get(2, 0.0)
        cls_metrics["ratio_neutral"] = ratio_dict.get(1, 0.0)
        cls_metrics["ratio_down"] = ratio_dict.get(0, 0.0)
    else:
        cls_metrics["f1_macro"] = np.nan
        cls_metrics["balanced_acc"] = np.nan
        cls_metrics["ratio_up"] = np.nan
        cls_metrics["ratio_neutral"] = np.nan
        cls_metrics["ratio_down"] = np.nan

    final_results = {
        "total_folds": total_folds,
        "oos_dates": all_oos_dates,
        "perf_metrics": perf_metrics,
        "cls_metrics": cls_metrics,
        "fold_details": pd.DataFrame(fold_details),
        "params_median": pd.DataFrame(fold_details)[
            ["alpha_up", "alpha_down", "beta_up", "beta_down"]
        ]
        .median()
        .to_dict(),
    }
    return final_results


# ============================================================
# 5. 메인 실행
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 4단계: Walk-Forward 공식 루틴 (2년/3개월/1개월 롤링)")
    print("=" * 60)

    df = load_data()
    print(f"📊 데이터 로드 완료: {df.shape[0]}일")

    result = run_walkforward_official(
        df,
        train_years=2,
        val_months=3,
        step_months=1,
        max_evals=150,  # 빠른 테스트용, 최종 분석시 300~500 권장
        md_threshold=0.20,
        lambda_mdd=2.0,
    )

    perf = result["perf_metrics"]
    cls = result["cls_metrics"]

    print("\n" + "=" * 60)
    print("📈 최종 OOS 성능 (연결된 전체 기간 기준)")
    print("=" * 60)
    print(f"🔹 총 Walk-Forward 폴드 수: {result['total_folds']}")
    print(
        f"🔹 OOS 기간: {result['oos_dates'][0].strftime('%Y-%m-%d')} ~ "
        f"{result['oos_dates'][-1].strftime('%Y-%m-%d')}"
    )
    print("\n[수익률 기반 지표]")
    print(f"   Sharpe Ratio       : {perf['sharpe']:.4f}")
    print(f"   CAGR               : {perf['cagr']:.4%}")
    print(f"   MDD                : {perf['mdd']:.4%}")
    print(f"   Calmar Ratio       : {perf['calmar']:.4f}")
    print(f"   승률 (Win Rate)    : {perf['win_rate']:.4%}")
    print(f"   Profit Factor      : {perf['profit_factor']:.4f}")

    print("\n[분류 품질 지표]")
    print(f"   Macro-F1           : {cls['f1_macro']:.4f}")
    print(f"   Balanced Accuracy  : {cls['balanced_acc']:.4f}")
    print(f"   예측 상승 비율     : {cls['ratio_up']:.4%}")
    print(f"   예측 중립 비율     : {cls['ratio_neutral']:.4%}")
    print(f"   예측 하락 비율     : {cls['ratio_down']:.4%}")

    # 🔥 중앙값 출력 문구 수정 (최종 추천 → 통계 참고용)
    print("\n📊 전체 폴드 파라미터 중앙값 (통계 참고용 - 최종 모델 아님):")
    for k, v in result["params_median"].items():
        print(f"   {k} = {v:.4f}")

    best_fold = result["fold_details"].loc[
        result["fold_details"]["oos_ret_mean"].idxmax()
    ]
    print(
        f"\n🏆 최고 평균 수익률 폴드 ({best_fold['val_start'].strftime('%Y-%m')}~"
        f"{best_fold['val_end'].strftime('%Y-%m')}):"
    )
    print(f"   α_up={best_fold['alpha_up']:.4f}, α_down={best_fold['alpha_down']:.4f}")
    print(f"   β_up={best_fold['beta_up']:.4f}, β_down={best_fold['beta_down']:.4f}")

    print("\n✅ 4단계 Walk-Forward 검증 완료!")
    print("※ 각 폴드의 최적 파라미터는 해당 OOS에 Rolling 적용됨.")
