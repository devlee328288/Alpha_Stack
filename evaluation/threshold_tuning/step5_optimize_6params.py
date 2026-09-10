import os
import sys
import warnings

import cma
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
)
from step1_core_features import compute_atr, compute_base, compute_log_rv, load_data
from tqdm import tqdm

from timeseries.models import fit_best

warnings.filterwarnings("ignore")


# ============================================================
# ADR-AS-0002 공용 라벨 함수
# ------------------------------------------------------------
# - T일 종가까지의 정보로 신호 생성
# - T+1일 시가 체결
# - T+6일 시가 평가
# - fwd_ret = open.shift(-6) / open.shift(-1) - 1
# - |fwd_ret| <= threshold → 중립(1)
# - fwd_ret >  +threshold → 상승(2)
# - fwd_ret <  -threshold → 하락(0)
# - 미래값이 없는 마지막 6행은 NaN 유지 (평가에서 제외)
# ------------------------------------------------------------
# NOTE: focal_classifier.py 의 make_labels() 도 동일 로직으로 맞춰야 함.
# ============================================================
def make_labels(df: pd.DataFrame, threshold: float = 0.02) -> np.ndarray:
    """
    ADR-AS-0002 기준 라벨 생성 (공용).

    Returns
    -------
    np.ndarray (float), shape (len(df),)
        값: {0.0, 1.0, 2.0} 또는 np.nan
    """
    open_ = df["open"]
    fwd_ret = open_.shift(-6) / open_.shift(-1) - 1.0
    labels = np.where(
        fwd_ret > threshold,
        2.0,
        np.where(fwd_ret < -threshold, 0.0, 1.0),
    )
    # 미래값이 없는 행(마지막 6행)은 NaN 유지 → 중립으로 변환 금지
    labels[fwd_ret.isna().values] = np.nan
    return labels


def _regression_test_make_labels() -> None:
    """
    make_labels() 회귀 테스트:
      - 시간축: fwd_ret = open.shift(-6)/open.shift(-1) - 1
      - 마지막 6행은 NaN (중립으로 변환되지 않음)
      - 상승/하락/중립 라벨 부여
    """
    n = 30
    prices = np.arange(1, n + 1, dtype=float)
    df_test = pd.DataFrame({"open": prices})

    labels = make_labels(df_test, threshold=0.02)

    # 1) 마지막 6행 NaN
    assert np.all(np.isnan(labels[-6:])), (
        f"마지막 6행은 NaN이어야 함. 실제: {labels[-6:]}"
    )

    # 2) 시간축 정확성
    for i in [0, 1, 5, 10, n - 7]:
        fwd = prices[i + 6] / prices[i + 1] - 1.0
        if fwd > 0.02:
            expected = 2.0
        elif fwd < -0.02:
            expected = 0.0
        else:
            expected = 1.0
        assert labels[i] == expected, (
            f"labels[{i}]: expected {expected}, got {labels[i]} "
            f"(fwd_ret={fwd:.6f})"
        )

    # 3) 하락 추세 → 하락 라벨
    df_down = pd.DataFrame({"open": np.linspace(100.0, 50.0, n)})
    labels_down = make_labels(df_down, threshold=0.02)
    assert labels_down[0] == 0.0, f"하락 추세 라벨 오류: {labels_down[0]}"

    # 4) 평탄 구간 → 중립
    df_flat = pd.DataFrame({"open": np.full(n, 100.0)})
    labels_flat = make_labels(df_flat, threshold=0.02)
    assert labels_flat[0] == 1.0, f"중립 라벨 오류: {labels_flat[0]}"

    print("✅ make_labels 회귀 테스트 통과 (시간축 / 끝행 NaN / 중립)")


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
# 1. 포지션 생성 (6개 파라미터)
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

    # 상단 돌파 = 상승(2), 하단 돌파 = 하락(0)
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


# ============================================================
# 3. 목적 함수 (ADR-AS-0002 라벨 + 학습/검증 threshold 통일)
# ============================================================
def objective_6params(
    params: list,
    df_train: pd.DataFrame,
    threshold: float = 0.02,
) -> float:
    """
    ADR-AS-0002 기준 라벨 사용:
      - fwd_ret = open.shift(-6)/open.shift(-1) - 1
      - 학습 threshold == 검증 threshold
      - 미래값 없는 행(NaN)은 평가에서 제외
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

    # 2) 예측 클래스 (상단 돌파=상승(2), 하단 돌파=하락(0))
    preds = np.where(close > upper, 2, np.where(close < lower, 0, 1)).astype(float)
    nan_mask = np.isnan(close) | np.isnan(upper) | np.isnan(lower)
    preds[nan_mask] = np.nan

    # 3) 실제 레이블 — ADR-AS-0002 공용 함수 (T+1 시가 체결 → T+6 시가 평가)
    y_true = make_labels(df_train, threshold=threshold)

    # 4) NaN 제거 (미래값 없는 마지막 6행은 NaN으로 유지되어 자동 제외)
    valid_mask = ~(np.isnan(preds) | np.isnan(y_true))
    if valid_mask.sum() < 10:
        return 1.0

    y_true_clean = y_true[valid_mask].astype(int)
    y_pred_clean = preds[valid_mask].astype(int)

    # 5) Macro-F1
    f1_macro = f1_score(y_true_clean, y_pred_clean, average="macro")

    # 6) 중립 Recall 패널티
    neutral_recall = recall_score(
        y_true_clean,
        y_pred_clean,
        labels=[1],
        average=None,
        zero_division=0,
    )[0]
    recall_penalty = 0.0
    target_recall = 0.35
    if neutral_recall < target_recall:
        recall_penalty = 10.0 * (target_recall - neutral_recall)

    # 7) CMA-ES는 최소화
    fitness = -f1_macro + recall_penalty
    return fitness


# ============================================================
# 4. Walk-Forward 실행
# ============================================================
def run_walkforward_6params(
    df: pd.DataFrame,
    train_years: int = 2,
    val_months: int = 3,
    step_months: int = 1,
    max_evals: int = 300,
    threshold: float = 0.02,
) -> dict:
    """
    Expanding Window (12폴드)
    - 라벨: ADR-AS-0002 (open 5일 수익률, T+1 시가 → T+6 시가)
    - 학습/검증 동일 threshold
    - 미래값 없는 행 NaN 유지 & 평가 제외
    """
    # ===== 기본 설정 =====
    INITIAL_TRAIN = train_years * 252  # 504일
    VAL_DAYS = val_months * 21  # 63일
    GAP = 6
    N_FOLDS = 12
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
    print(f"🔹 라벨: ADR-AS-0002 (T+1 시가 → T+6 시가, ±{threshold*100:.0f}%)")
    print("🔹 신호: 상단돌파=상승(2), 하단돌파=하락(0)")
    print("🔍 최적화 파라미터: α_up, α_down, β_up, β_down, Vol_Period, Volume_Period")

    # ADR-AS-0002 라벨을 전체 df에 대해 1회 사전 계산 (마지막 6행은 NaN)
    labels_full = make_labels(df, threshold=threshold)

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

        # ADR-AS-0002: 마지막 행에서 T+6까지 필요 → val_end + 6 <= total_len
        if val_end + 6 > total_len:
            print(
                f"   ⚠️ 폴드 {fold+1}: 데이터 부족으로 중단 "
                f"(필요:{val_end+6}, 실제:{total_len})"
            )
            break

        df_train = df.iloc[0:train_end].copy()
        df_val = df.iloc[val_start:val_end].copy()

        calc_start = max(0, train_end - LOOKBACK_DAYS)
        df_calc = df.iloc[calc_start:val_end].copy()

        # ---- CMA-ES 설정 ----
        x0 = [0.50, 0.50, 0.0, 0.0, 14.0, 20.0]
        sigma0 = 0.5

        bounds_low = [0.05, 0.05, 0.0, 0.0, 10.0, 10.0]
        bounds_high = [2.0, 2.0, 2.0, 2.0, 30.0, 30.0]

        def obj_func(p, df_train=df_train, th=threshold):
            p_8 = list(p) + [0.01, -0.01]
            return objective_6params(p_8, df_train, threshold=th)

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
            df_calc,
            alpha_up,
            alpha_down,
            beta_up,
            beta_down,
            vol_period,
            volume_period,
        )

        oos_offset = val_start - calc_start
        positions = positions_full[oos_offset : oos_offset + VAL_DAYS]
        preds = preds_full[oos_offset : oos_offset + VAL_DAYS]

        # ADR-AS-0002 라벨 (전체 df 기준으로 계산 후 슬라이스)
        y_true = labels_full[val_start:val_end]

        # ============================================================
        # [ARIMA 기준선 평가 - ADR-AS-0002 시간축 정렬]
        #   T 종가 시점에서 6-step 예측 → step[1:] 누적이 T+1→T+6
        # ============================================================
        train_ret = df_train["open"].pct_change().dropna().values

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
                        # 현재 시점(i)까지의 실제 open 수익률을 history에 반영
                        actual_ret = df_val["open"].pct_change().values[i]
                        history.append(
                            actual_ret if not np.isnan(actual_ret) else 0.0
                        )

                        # i+1 ~ i+6 6-step 예측
                        temp_hist = history.copy()
                        step_returns = []
                        for _step in range(6):
                            next_val = const
                            for j in range(len(phi)):
                                if j < len(temp_hist):
                                    next_val += phi[j] * temp_hist[-(j + 1)]
                            step_returns.append(next_val)
                            temp_hist.append(next_val)

                        # T+1 → T+6 누적 = step[1:] (0-indexed 1..5)
                        cum_ret = 1.0
                        for r in step_returns[1:]:
                            cum_ret *= (1 + r)
                        arima_pred_5d_cum.append(cum_ret - 1)

                    arima_pred_5d_cum = np.array(arima_pred_5d_cum)

                    # ARIMA 예측 라벨도 동일 threshold
                    arima_preds = np.where(
                        arima_pred_5d_cum > threshold,
                        2,
                        np.where(arima_pred_5d_cum < -threshold, 0, 1),
                    )

                    valid_arima = ~np.isnan(y_true) & ~np.isnan(arima_pred_5d_cum)
                    if valid_arima.sum() > 0 and len(arima_preds) == len(y_true):
                        fold_acc_arima = np.mean(
                            arima_preds[valid_arima] == y_true[valid_arima]
                        )
                        all_arima_accs.append(fold_acc_arima)
                        if total_folds == 0:
                            print(
                                f"   ✅ ARIMA T+1→T+6 누적 예측 (정확도: {fold_acc_arima:.4f})"
                            )
                    else:
                        if total_folds == 0:
                            print(
                                f"   ❌ ARIMA 평가 불가 (valid={valid_arima.sum()}, "
                                f"len={len(arima_preds)} vs {len(y_true)})"
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
        # 포지션 수익률 계산 (ADR-AS-0002: T+1 시가 체결 → T+2 시가 청산)
        # signal[i] (close i 신호) → open i+1 체결 → open i+2 청산
        # strategy_ret[j] = positions[j-2] * market_ret[j]
        # ============================================================
        market_ret = df_val["open"].pct_change().values

        pos_shifted = np.empty(len(positions), dtype=float)
        if len(positions) >= 2:
            pos_shifted[:2] = prev_last_position
            pos_shifted[2:] = positions[:-2]
        else:
            pos_shifted[:] = prev_last_position

        if len(positions) > 0:
            prev_last_position = positions[-1]

        strategy_ret = pos_shifted * market_ret

        # 미래값 없는 행(NaN 라벨)도 평가에서 제외
        valid_mask = ~(
            np.isnan(strategy_ret) | np.isnan(market_ret) | np.isnan(y_true)
        )

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

    # ===== [진단] 라벨 축 수정 후 정확도 확인 =====
    y_true_arr = np.array(all_oos_y_true)
    y_pred_arr = np.array(all_oos_y_pred)

    nan_filter = ~(np.isnan(y_true_arr) | np.isnan(y_pred_arr))
    y_true_arr = y_true_arr[nan_filter]
    y_pred_arr = y_pred_arr[nan_filter]

    if len(y_true_arr) > 0:
        acc_current = accuracy_score(y_true_arr, y_pred_arr)
        print(f"\n🔍 [진단] ADR-AS-0002 라벨 기준 정확도: {acc_current:.4f}")
        invert_back = {0: 2, 1: 1, 2: 0}
        y_pred_reversed = np.array([invert_back[p] for p in y_pred_arr])
        acc_reversed = accuracy_score(y_true_arr, y_pred_reversed)
        print(f"   (참고) 만약 신호를 반전했다면: {acc_reversed:.4f}")

    # ---- 연결된 OOS 최종 평가 ----
    oos_returns = np.array(all_oos_returns)
    perf_metrics = calculate_metrics(oos_returns)

    if len(all_arima_accs) > 0:
        arima_mean_acc = np.mean(all_arima_accs)
        print(
            f"\n📊 [ARIMA 기준선 - T+1→T+6 누적] 평균 방향 적중률 "
            f"(전체 {len(all_arima_accs)}개 폴드): "
            f"{arima_mean_acc:.4f} ({arima_mean_acc*100:.2f}%)"
        )
    else:
        print("⚠️ ARIMA 예측 실패 (데이터 부족)")

    cls_metrics = {}
    if len(y_true_arr) > 0:
        cls_metrics["f1_macro"] = f1_score(
            y_true_arr, y_pred_arr, average="macro"
        )
        cls_metrics["balanced_acc"] = balanced_accuracy_score(
            y_true_arr, y_pred_arr
        )
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
# 5. 메인 실행 (1%와 2% 비교)
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 5단계: Expanding + ADR-AS-0002(T+1→T+6) + 1% vs 2% 비교")
    print("=" * 60)

    # 라벨 회귀 테스트 먼저 수행
    _regression_test_make_labels()

    df = load_data()
    print(f"📊 데이터 로드 완료: {df.shape[0]}일")

    results = {}

    for thresh in [0.01, 0.02]:
        print("\n" + "=" * 60)
        print(f"📌 [실험 시작] Threshold = {thresh*100:.0f}% (상승/하락 기준)")
        print("=" * 60)

        result = run_walkforward_6params(
            df,
            train_years=2,
            val_months=3,
            step_months=1,
            max_evals=300,
            threshold=thresh,
        )
        results[thresh] = result

    # ============================================================
    # 최종 비교 결과 출력
    # ============================================================
    print("\n\n" + "=" * 70)
    print("📊 [최종 비교] 1% 기준 vs 2% 기준 Walk-Forward 성능 (ADR-AS-0002)")
    print("=" * 70)

    compare_df = pd.DataFrame(
        {
            "Metric": [
                "Sharpe",
                "CAGR",
                "MDD",
                "Calmar",
                "Win Rate",
                "Macro-F1",
                "Balanced Acc",
            ],
            "1% Threshold": [
                results[0.01]["perf_metrics"]["sharpe"],
                results[0.01]["perf_metrics"]["cagr"],
                results[0.01]["perf_metrics"]["mdd"],
                results[0.01]["perf_metrics"]["calmar"],
                results[0.01]["perf_metrics"]["win_rate"],
                results[0.01]["cls_metrics"]["f1_macro"],
                results[0.01]["cls_metrics"]["balanced_acc"],
            ],
            "2% Threshold": [
                results[0.02]["perf_metrics"]["sharpe"],
                results[0.02]["perf_metrics"]["cagr"],
                results[0.02]["perf_metrics"]["mdd"],
                results[0.02]["perf_metrics"]["calmar"],
                results[0.02]["perf_metrics"]["win_rate"],
                results[0.02]["cls_metrics"]["f1_macro"],
                results[0.02]["cls_metrics"]["balanced_acc"],
            ],
        }
    )

    for col in ["1% Threshold", "2% Threshold"]:
        compare_df[col] = compare_df[col].apply(
            lambda x: f"{x:.4f}" if pd.notna(x) else "NaN"
        )

    print(compare_df.to_string(index=False))

    if (
        results[0.01]["perf_metrics"]["sharpe"]
        > results[0.02]["perf_metrics"]["sharpe"]
    ):
        print("\n🏆 1% 기준의 Sharpe Ratio가 더 높습니다.")
    else:
        print("\n🏆 2% 기준의 Sharpe Ratio가 더 높습니다.")

    print("\n✅ 전체 비교 실험 완료!")
