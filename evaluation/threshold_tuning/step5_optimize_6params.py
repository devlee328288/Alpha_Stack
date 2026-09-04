import warnings

import cma
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
from tqdm import tqdm

warnings.filterwarnings("ignore")

from step1_core_features import compute_atr, compute_base, compute_log_rv, load_data


# ============================================================
# 0. 기준선 계산 (유연한 기간 지원) - exp 제거, 선형 구조
# ============================================================
def compute_bands_flexible(
    df: pd.DataFrame,
    vol_period: int = 14,
    volume_period: int = 20,
    base_type: str = "SMA20",
    asym: bool = True,
    alpha_up: float = 1.0,
    alpha_down: float = 1.0,
    beta_up: float = 0.0,
    beta_down: float = 0.0,
) -> pd.DataFrame:
    """
    Width = ATR × max(0.05, alpha + beta × logRV)
    """
    base = compute_base(df, base_type)
    atr = compute_atr(df, period=vol_period)
    log_rv = compute_log_rv(df, period=volume_period)

    raw_up = alpha_up + beta_up * log_rv
    raw_down = alpha_down + beta_down * log_rv

    upper_width = atr * np.maximum(0.05, raw_up)
    lower_width = atr * np.maximum(0.05, raw_down)

    result = df.copy()
    result["base"] = base
    result["upper"] = base + upper_width
    result["lower"] = base - lower_width
    return result


# ============================================================
# 1. 포지션 생성 (6개 파라미터 버전)
# ============================================================
def get_positions_6params(
    df: pd.DataFrame,
    alpha_up: float,
    alpha_down: float,
    beta_up: float,
    beta_down: float,
    vol_period: int,
    volume_period: int,
) -> tuple:
    bands = compute_bands_flexible(
        df,
        vol_period=vol_period,
        volume_period=volume_period,
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
    return positions, pred


# ============================================================
# 2. 성과 지표 계산 (재사용)
# ============================================================
def calculate_metrics(returns: np.ndarray) -> dict:
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
# 3. 목적 함수 (CMA-ES용, 6개 파라미터) - Macro-F1 최대화 (유지)
# ============================================================
def objective_6params(
    params: list,
    df_train: pd.DataFrame,
) -> float:
    """
    params = [alpha_up, alpha_down, beta_up, beta_down, vol_period, volume_period]
    목표: Macro-F1 Score 최대화 (음수 반환)
    """
    alpha_up, alpha_down, beta_up, beta_down, vol_period_float, volume_period_float = (
        params
    )
    vol_period = int(round(vol_period_float))
    volume_period = int(round(volume_period_float))

    vol_period = np.clip(vol_period, 10, 30)
    volume_period = np.clip(volume_period, 10, 30)

    positions, preds = get_positions_6params(
        df_train, alpha_up, alpha_down, beta_up, beta_down, vol_period, volume_period
    )

    if "label" in df_train.columns:
        label_map = {"상승": 2, "중립": 1, "하락": 0}
        y_true_series = df_train["label"].map(label_map)
        y_true = y_true_series.fillna(1).astype(int).values
    else:
        ret = df_train["close"].pct_change().values
        y_true = np.where(ret > 0.005, 2, np.where(ret < -0.005, 0, 1))

    valid_mask = ~(np.isnan(preds) | np.isnan(y_true))
    if valid_mask.sum() < 10:
        return 1.0

    y_true_clean = y_true[valid_mask]
    y_pred_clean = preds[valid_mask]

    f1 = f1_score(y_true_clean, y_pred_clean, average="macro")

    # 중립 비율 패널티 (과매매 / 관망 방지)
    neutral_ratio = np.mean(y_pred_clean == 1)
    penalty = 0.0
    if neutral_ratio < 0.05:
        penalty = 0.5 * (0.05 - neutral_ratio)
    elif neutral_ratio > 0.85:
        penalty = 0.5 * (neutral_ratio - 0.85)

    return -(f1 - penalty)


# ============================================================
# 4. Walk-Forward 실행 (6개 파라미터) - 수정 사항 반영
# ============================================================
def run_walkforward_6params(
    df: pd.DataFrame,
    train_years: int = 2,
    val_months: int = 3,
    step_months: int = 1,
    max_evals: int = 300,
) -> dict:
    train_days = train_years * 252
    val_days = val_months * 21
    step_days = step_months * 21

    # 🔥 수정 1: OOS 지표 계산을 위한 Lookback 기간 (Vol/Volume 최대 30 + SMA20 + 여유)
    LOOKBACK_DAYS = 35

    total_len = len(df)
    print(f"📅 전체 데이터: {total_len}일")
    print(f"📐 학습: {train_days}일, 검증: {val_days}일, 이동: {step_days}일")
    print(f"📦 지표 계산용 Lookback: {LOOKBACK_DAYS}일 (OOS 이전 데이터 포함)")
    print("🔍 최적화 파라미터: α_up, α_down, β_up, β_down, Vol_Period, Volume_Period")

    all_oos_returns = []
    all_oos_y_true = []
    all_oos_y_pred = []
    fold_details = []

    # 🔥 수정 2: OOS 간 포지션 연속성을 위한 이전 포지션 저장 변수
    prev_last_position = 0

    total_folds = 0
    for start in tqdm(range(0, total_len - train_days - val_days + 1, step_days)):
        train_end = start + train_days
        val_end = train_end + val_days

        # 학습 데이터 (기존과 동일)
        df_train = df.iloc[start:train_end].copy()

        # 🔥 수정 1 적용: OOS + Lookback 데이터를 합쳐서 지표 계산
        calc_start = max(0, train_end - LOOKBACK_DAYS)
        df_calc = df.iloc[calc_start:val_end].copy()

        # ---- CMA-ES 설정 ----
        x0 = [0.2, 0.2, 0.0, 0.0, 14.0, 20.0]
        sigma0 = 0.5

        bounds_low = [0.05, 0.05, 0.0, 0.0, 10.0, 10.0]
        bounds_high = [0.35, 0.35, 1.5, 1.5, 30.0, 30.0]

        x0 = np.clip(x0, bounds_low, bounds_high).tolist()

        obj_func = lambda p: objective_6params(p, df_train)

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

        alpha_up, alpha_down, beta_up, beta_down, vol_p_float, volm_p_float = (
            best_params
        )
        vol_period = int(round(np.clip(vol_p_float, 10, 30)))
        volume_period = int(round(np.clip(volm_p_float, 10, 30)))

        # ---- OOS 적용 (Lookback이 포함된 df_calc에 계산 후 OOS만 슬라이싱) ----
        positions_full, preds_full = get_positions_6params(
            df_calc, alpha_up, alpha_down, beta_up, beta_down, vol_period, volume_period
        )

        # OOS 시작 위치에 맞게 자르기
        oos_offset = train_end - calc_start
        positions = positions_full[oos_offset:]  # OOS 구간만 추출
        preds = preds_full[oos_offset:]  # OOS 구간만 추출

        # OOS 데이터 (수익률 계산용)
        df_val = df.iloc[train_end:val_end].copy()

        if "label" in df_val.columns:
            label_map = {"상승": 2, "중립": 1, "하락": 0}
            y_true_series = df_val["label"].map(label_map)
            y_true = y_true_series.fillna(1).astype(int).values
        else:
            ret = df_val["close"].pct_change().values
            y_true = np.where(ret > 0.005, 2, np.where(ret < -0.005, 0, 1))

        market_ret = df_val["close"].pct_change().values

        # 🔥 수정 2: OOS 첫날 포지션을 이전 OOS 마지막 포지션으로 연결
        pos_shifted = np.roll(positions, 1)
        if len(pos_shifted) > 0:
            pos_shifted[0] = prev_last_position  # 직전 포지션 승계

        # OOS 마지막 포지션을 다음 루프를 위해 저장
        if len(positions) > 0:
            prev_last_position = positions[-1]

        strategy_ret = pos_shifted * market_ret
        valid_mask = ~(np.isnan(strategy_ret) | np.isnan(market_ret))

        if valid_mask.sum() > 0:
            all_oos_returns.extend(strategy_ret[valid_mask].tolist())
            all_oos_y_true.extend(y_true[valid_mask].tolist())
            all_oos_y_pred.extend(preds[valid_mask].tolist())

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
                "vol_period": vol_period,
                "volume_period": volume_period,
                "is_fitness": -best_fitness,
                "oos_ret_mean": (
                    np.nanmean(strategy_ret[valid_mask])
                    if valid_mask.sum() > 0
                    else np.nan
                ),
            }
        )
        total_folds += 1

        if total_folds % 10 == 0:
            print(
                f"   → {total_folds}개 폴드 완료 (현재 OOS: {df.index[val_end-1].strftime('%Y-%m-%d')})"
            )

    # ---- 연결된 OOS 최종 평가 ----
    oos_returns = np.array(all_oos_returns)
    perf_metrics = calculate_metrics(oos_returns)

    y_true_arr = np.array(all_oos_y_true)
    y_pred_arr = np.array(all_oos_y_pred)

    cls_metrics = {}
    if len(y_true_arr) > 0:
        cls_metrics["f1_macro"] = f1_score(y_true_arr, y_pred_arr, average="macro")
        cls_metrics["balanced_acc"] = balanced_accuracy_score(y_true_arr, y_pred_arr)
        unique, counts = np.unique(y_pred_arr, return_counts=True)
        ratio_dict = dict(zip(unique, counts / len(y_pred_arr)))
        cls_metrics["ratio_up"] = ratio_dict.get(2, 0.0)
        cls_metrics["ratio_neutral"] = ratio_dict.get(1, 0.0)
        cls_metrics["ratio_down"] = ratio_dict.get(0, 0.0)
    else:
        cls_metrics["f1_macro"] = np.nan
        cls_metrics["balanced_acc"] = np.nan
        cls_metrics["ratio_up"] = np.nan
        cls_metrics["ratio_neutral"] = np.nan
        cls_metrics["ratio_down"] = np.nan

    fold_df = pd.DataFrame(fold_details)
    params_median = (
        fold_df[
            [
                "alpha_up",
                "alpha_down",
                "beta_up",
                "beta_down",
                "vol_period",
                "volume_period",
            ]
        ]
        .median()
        .to_dict()
    )

    return {
        "total_folds": total_folds,
        "perf_metrics": perf_metrics,
        "cls_metrics": cls_metrics,
        "fold_details": fold_df,
        "params_median": params_median,
    }


# ============================================================
# 5. 메인 실행
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 5단계: 6개 파라미터 CMA-ES 최적화 (exp 제거, 선형 구조)")
    print("=" * 60)

    df = load_data()
    print(f"📊 데이터 로드 완료: {df.shape[0]}일")

    # 🔥 수정 4: 존재하지 않는 인자(md_threshold, lambda_mdd) 제거
    result = run_walkforward_6params(
        df,
        train_years=2,
        val_months=3,
        step_months=1,
        max_evals=250,
    )

    perf = result["perf_metrics"]
    cls = result["cls_metrics"]

    print("\n" + "=" * 60)
    print("📈 최종 OOS 성능 (6개 파라미터 최적화)")
    print("=" * 60)
    print(f"🔹 총 Walk-Forward 폴드 수: {result['total_folds']}")
    print("\n[수익률 기반 지표]")
    print(f"   Sharpe Ratio       : {perf['sharpe']:.4f}")
    print(f"   CAGR               : {perf['cagr']:.4%}")
    print(f"   MDD                : {perf['mdd']:.4%}")
    print(f"   Calmar Ratio       : {perf['calmar']:.4f}")
    print(f"   승률 (Win Rate)    : {perf['win_rate']:.4%}")
    print(f"   Profit Factor      : {perf['profit_factor']:.4f}")

    print("\n[분류 품질]")
    print(f"   Macro-F1           : {cls['f1_macro']:.4f}")
    print(f"   Balanced Accuracy  : {cls['balanced_acc']:.4f}")
    print(f"   예측 상승 비율     : {cls['ratio_up']:.4%}")
    print(f"   예측 중립 비율     : {cls['ratio_neutral']:.4%}")
    print(f"   예측 하락 비율     : {cls['ratio_down']:.4%}")

    # 🔥 수정 3: Median 파라미터를 "최종 추천"이 아닌 "안정성 분석용 통계"로 명확히 표시
    print("\n📊 전체 WF 파라미터 중앙값 (안정성 분석용 통계 - 최종 모델 아님):")
    for k, v in result["params_median"].items():
        if k in ["vol_period", "volume_period"]:
            print(f"   {k} = {int(v)}일")
        else:
            print(f"   {k} = {v:.4f}")

    print(
        "\n⚠️ 실전 적용 시: 위 중앙값이 아닌, 각 OOS 구간에 최적화된 파라미터가 Rolling 적용됩니다."
    )
    print("✅ 5단계 6개 파라미터 최적화 완료!")
