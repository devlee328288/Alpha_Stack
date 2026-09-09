import os
import sys
import warnings

import cma
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    recall_score,
)
from step1_core_features import compute_atr, compute_base, compute_log_rv, load_data
from tqdm import tqdm

from timeseries.models import fit_best

warnings.filterwarnings("ignore")


# ============================================================
# 0. 기준선 계산 (변경 없음)
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
# 1. 포지션 생성 (6개 파라미터) - 단일 돌파 (2일 확인 제거)
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
# 2. 성과 지표 계산 (변경 없음)
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


def objective_6params(
    params: list,
    df_train: pd.DataFrame,
) -> float:
    """
    🔥 [긴급 수정] Focal Loss + 안정화 조치 + Macro-F1 백업
    """
    (
        alpha_up,
        alpha_down,
        beta_up,
        beta_down,
        vol_period_float,
        volume_period_float,
        _,  # 더미
        _,  # 더미
    ) = params

    vol_period = int(round(vol_period_float))
    volume_period = int(round(volume_period_float))

    # 1) 기준선 계산
    bands = compute_bands_flexible(
        df_train,
        vol_period=vol_period,
        volume_period=volume_period,
        alpha_up=alpha_up,
        alpha_down=alpha_down,
        beta_up=beta_up,
        beta_down=beta_down,
    )

    close = df_train["close"].values
    upper = bands["upper"].values
    lower = bands["lower"].values
    _base = bands["base"].values

    # 2) 예측 클래스
    preds = np.where(close > upper, 2, np.where(close < lower, 0, 1))

    # 3) 실제 레이블 - 5일 후 수익률 ±1%
    if "label" in df_train.columns:
        label_map = {"상승": 2, "중립": 1, "하락": 0}
        y_true_series = df_train["label"].map(label_map)
        y_true = y_true_series.fillna(1).astype(int).values
    else:
        ret_5d = (df_train["close"].shift(-5) / df_train["close"] - 1).values
        y_true = np.where(ret_5d > 0.01, 2, np.where(ret_5d < -0.01, 0, 1))

    # 4) NaN 제거
    valid_mask = ~(np.isnan(preds) | np.isnan(y_true))
    if valid_mask.sum() < 10:
        return 1.0

    y_true_clean = y_true[valid_mask]
    y_pred_clean = preds[valid_mask]

    # ============================================================
    # 5) Macro-F1 계산 (안정적인 백업 목적 함수)
    # ============================================================
    f1_macro = f1_score(y_true_clean, y_pred_clean, average="macro")

    # ============================================================
    # 6) 중립 Recall 패널티 (가장 중요!)
    # ============================================================
    neutral_recall = recall_score(y_true_clean, y_pred_clean, labels=[1], average=None)[
        0
    ]
    recall_penalty = 0.0
    target_recall = 0.35
    if neutral_recall < target_recall:
        recall_penalty = 10.0 * (target_recall - neutral_recall)

    # ============================================================
    # 7) 최종 Fitness (CMA-ES는 최소화)
    # ============================================================
    # F1은 클수록 좋으므로 음수로 변환 (최소화 문제로 변환)
    # recall_penalty는 클수록 나쁨 (그대로 더함)
    fitness = -f1_macro + recall_penalty

    return fitness


# ============================================================
# 4. Walk-Forward 실행 - 🔥 Expanding (12폴드, Gap 5, 5일 ±1%)
# ============================================================
def run_walkforward_6params(
    df: pd.DataFrame,
    train_years: int = 2,
    val_months: int = 3,
    step_months: int = 1,
    max_evals: int = 300,
) -> dict:
    """
    🔥 Expanding Window (12폴드, Gap 5, 5일 ±1%)
    - MD v4.1 합의안 완벽 반영
    """
    # ===== 기본 설정 =====
    INITIAL_TRAIN = train_years * 252  # 504일
    VAL_DAYS = val_months * 21  # 60일 (3개월)
    GAP = 5  # Gap 5일 (고정)
    N_FOLDS = 12  # 12폴드 (고정)
    LOOKBACK_DAYS = 35

    total_len = len(df)

    first_train_end = INITIAL_TRAIN
    last_train_end = total_len - VAL_DAYS - GAP

    if last_train_end <= first_train_end:
        raise ValueError(
            f"데이터가 너무 짧습니다. "
            f"필요: 최소 {first_train_end + VAL_DAYS + GAP}일, "
            f"실제: {total_len}일"
        )

    # Expanding: 학습 종료일을 균등하게 분할
    train_ends = (
        np.linspace(first_train_end, last_train_end, N_FOLDS).astype(int).tolist()
    )
    train_ends = sorted(set(train_ends))

    print(f"📅 전체 데이터: {total_len}일")
    print("🔹 평가 방식: Expanding (전체 구간에 걸친 12폴드 균등 분배)")
    first_date = df.index[first_train_end - 1].strftime("%Y-%m-%d")
    print(f"   - 첫 학습 종료일: {first_train_end}일 (약 {first_date})")
    last_date = df.index[last_train_end - 1].strftime("%Y-%m-%d")
    print(f"   - 마지막 학습 종료일: {last_train_end}일 (약 {last_date})")
    print(f"🔹 Gap: {GAP}일, 검증(horizon): {VAL_DAYS}일")
    print("🔹 라벨: 5일 후 수익률 ±1.0% (고정)")
    print("🔍 최적화 파라미터: α_up, α_down, β_up, β_down, Vol_Period, Volume_Period")

    all_oos_returns = []
    all_oos_y_true = []
    all_oos_y_pred = []
    fold_details = []
    all_arima_accs = []

    prev_last_position = 0
    total_folds = 0

    for fold, train_end in enumerate(tqdm(train_ends, desc="Expanding 폴드 진행")):
        val_start = train_end + GAP
        val_end = val_start + VAL_DAYS

        if val_end + 5 > total_len:
            print(
                f"   ⚠️ 폴드 {fold+1}: 데이터 부족으로 중단 (필요:{val_end+5}, 실제:{total_len})"
            )
            break

        # 🔥 Expanding: 학습 데이터는 0부터 train_end까지
        df_train = df.iloc[0:train_end].copy()
        df_val = df.iloc[val_start:val_end].copy()

        calc_start = max(0, train_end - LOOKBACK_DAYS)
        df_calc = df.iloc[calc_start:val_end].copy()

        # ---- CMA-ES 설정 ----
        x0 = [0.50, 0.50, 0.0, 0.0, 14.0, 20.0]
        sigma0 = 0.5

        # α_up 상한 0.40, α_down 상한 0.30 (비대칭: 상승은 넓게, 하락은 좁게)
        bounds_low = [0.05, 0.05, 0.0, 0.0, 10.0, 10.0]
        bounds_high = [2.0, 2.0, 2.0, 2.0, 30.0, 30.0]

        def obj_func(p, df_train=df_train):
            p_8 = list(p) + [0.01, -0.01]
            return objective_6params(p_8, df_train)

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

        # ---- OOS 적용 (Lookback 포함) ----
        positions_full, preds_full = get_positions_6params(
            df_calc, alpha_up, alpha_down, beta_up, beta_down, vol_period, volume_period
        )

        oos_offset = val_start - calc_start
        positions = positions_full[oos_offset : oos_offset + VAL_DAYS]
        preds = preds_full[oos_offset : oos_offset + VAL_DAYS]

        ret_5d = (df_val["close"].shift(-5) / df_val["close"] - 1).values
        y_true = np.where(ret_5d > 0.01, 2, np.where(ret_5d < -0.01, 0, 1))

        # ============================================================
        # [ARIMA 기준선 평가 - 5일 누적 예측]
        # ============================================================
        train_ret = df_train["close"].pct_change().dropna().values

        if len(train_ret) > 10:
            try:
                arima_res = fit_best(train_ret, max_p=3, max_q=3)
                model_dict = arima_res.get("model")

                if (
                    model_dict is not None
                    and "levels" in model_dict
                    and "phi" in model_dict
                ):
                    phi = model_dict["phi"]
                    const = model_dict["const"]
                    history = list(model_dict["levels"])

                    arima_pred_5d_cum = []

                    for i in range(len(df_val)):
                        temp_hist = history.copy()
                        cum_ret = 1.0
                        for _step in range(5):
                            next_val = const
                            for j in range(len(phi)):
                                if j < len(temp_hist):
                                    next_val += phi[j] * temp_hist[-(j + 1)]
                            cum_ret = cum_ret * (1 + next_val)
                            temp_hist.append(next_val)

                        arima_pred_5d_cum.append(cum_ret - 1)

                        if i < len(df_val):
                            actual_ret = df_val["close"].pct_change().values[i]
                            if not np.isnan(actual_ret):
                                history.append(actual_ret)
                            else:
                                history.append(0.0)

                    arima_pred_5d_cum = np.array(arima_pred_5d_cum)

                    arima_preds = np.where(
                        arima_pred_5d_cum > 0.01,
                        2,
                        np.where(arima_pred_5d_cum < -0.01, 0, 1),
                    )

                    if len(arima_preds) == len(y_true):
                        fold_acc_arima = np.mean(arima_preds == y_true)
                        all_arima_accs.append(fold_acc_arima)
                        if total_folds == 0:
                            print(
                                f"   ✅ ARIMA 5일 누적 예측 성공! (정확도: {fold_acc_arima:.4f})"
                            )
                    else:
                        if total_folds == 0:
                            print(
                                f"   ❌ 길이 불일치: {len(arima_preds)} vs {len(y_true)}"
                            )
                else:
                    if total_folds == 0:
                        print("   ❌ model_dict에 'levels' 또는 'phi'가 없습니다.")
            except Exception as e:
                if total_folds == 0:
                    print(f"   ❌ ARIMA 실행 중 예외 발생: {e}")
        else:
            if total_folds == 0:
                print(f"   ❌ train_ret 길이 부족 (실제: {len(train_ret)})")

        # ============================================================
        # 포지션 수익률 계산 (명시적 Shift)
        # ============================================================
        market_ret = df_val["close"].pct_change().values

        pos_shifted = np.concatenate([[prev_last_position], positions[:-1]])
        if len(pos_shifted) > 0:
            pos_shifted[0] = prev_last_position

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
                "train_start": df.index[0],
                "train_end": df.index[train_end - 1],
                "val_start": df.index[val_start],
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

        if total_folds % 5 == 0:
            start_date = df.index[val_start].strftime("%Y-%m-%d")
            end_date = df.index[val_end - 1].strftime("%Y-%m-%d")
            print(f"   → {total_folds}개 폴드 완료 (OOS: {start_date} ~ {end_date})")

    # ---- 연결된 OOS 최종 평가 ----
    oos_returns = np.array(all_oos_returns)
    perf_metrics = calculate_metrics(oos_returns)

    if len(all_arima_accs) > 0:
        arima_mean_acc = np.mean(all_arima_accs)
        print(
            f"\n📊 [ARIMA 기준선 - 5일 누적] 평균 방향 적중률 (전체 {len(all_arima_accs)}개 폴드)"
            f": {arima_mean_acc:.4f} ({arima_mean_acc*100:.2f}%)"
        )
    else:
        print("⚠️ ARIMA 예측 실패 (데이터 부족)")

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
    print("🚀 5단계: Expanding + 5일 ±1% + ARIMA 누적 (분류 1순위)")
    print("=" * 60)

    df = load_data()
    print(f"📊 데이터 로드 완료: {df.shape[0]}일")

    result = run_walkforward_6params(
        df,
        train_years=2,
        val_months=3,
        step_months=1,
        max_evals=300,
    )

    perf = result["perf_metrics"]
    cls = result["cls_metrics"]

    print("\n" + "=" * 60)
    print("📈 최종 OOS 성능 (Expanding 12폴드, Gap 5, 5일 ±1%)")
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

    print("\n📊 전체 WF 파라미터 중앙값 (안정성 분석용):")
    for k, v in result["params_median"].items():
        if k in ["vol_period", "volume_period"]:
            print(f"   {k} = {int(v)}일")
        else:
            print(f"   {k} = {v:.4f}")

    print("\n✅ Expanding 기준 재측정 완료! (12폴드, Gap=5, 5일 ±1%)")
    print("   🔥 ARIMA와 동일한 조건에서 분류 성능을 비교합니다.")
