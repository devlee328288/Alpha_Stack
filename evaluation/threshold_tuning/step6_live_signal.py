import os
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
from step1_core_features import compute_atr, compute_base, compute_log_rv, load_data
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ============================================================
# 0. 기준선 계산
# ============================================================
def compute_bands_flexible(
    df: pd.DataFrame,
    vol_period: int = 14,
    volume_period: int = 20,
    base_type: str = "SMA20",
    alpha_up: float = 1.0,
    alpha_down: float = 1.0,
    beta_up: float = 0.0,
    beta_down: float = 0.0,
) -> pd.DataFrame:
    """🚀 step5와 완전히 동일한 선형 구조 (exp 제거, β≥0 적용)"""
    base = compute_base(df, base_type)
    atr = compute_atr(df, period=vol_period)
    log_rv = compute_log_rv(df, period=volume_period)

    # 🔥 step5와 동일: exp 없이 선형, 최소값 0.05로 클리핑
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
# 1. 단일 파라미터 세트로 신호 생성 (Look-ahead 방지 적용)
# ============================================================
def generate_signals_single(
    df: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """
    하나의 고정 파라미터 세트로 전체 데이터에 대해 신호를 생성합니다.
    params: alpha_up, alpha_down, beta_up, beta_down, vol_period, volume_period
    """
    bands = compute_bands_flexible(
        df,
        vol_period=params["vol_period"],
        volume_period=params["volume_period"],
        alpha_up=params["alpha_up"],
        alpha_down=params["alpha_down"],
        beta_up=params["beta_up"],
        beta_down=params["beta_down"],
    )

    close = df["close"].values
    upper = bands["upper"].values
    lower = bands["lower"].values

    # 신호 (0:하락, 1:중립, 2:상승)
    signal = np.where(close > upper, 2, np.where(close < lower, 0, 1))

    # 포지션 (상승=+1, 중립=0, 하락=-1)
    position = np.where(signal == 2, 1, np.where(signal == 0, -1, 0))

    result = bands.copy()
    result["signal"] = signal
    result["position"] = position

    # Look-ahead 방지: t일 포지션을 t+1일 수익률에 적용
    market_ret = df["close"].pct_change().values
    pos_shifted = np.roll(position, 1)
    pos_shifted[0] = 0
    result["strategy_return"] = pos_shifted * market_ret

    return result


# ============================================================
# 2. Rolling 파라미터 적용 (각 OOS 구간별 최적 파라미터 사용) - 🔥 Lookback 추가!
# ============================================================
def generate_signals_rolling(
    df: pd.DataFrame,
    fold_details: pd.DataFrame,
) -> pd.DataFrame:
    """
    Walk-Forward 폴드 정보를 기반으로 각 날짜에 해당하는 OOS 파라미터를 적용합니다.
    🔥 Step 5와 동일하게 Lookback(35일)을 포함하여 지표를 계산한 후 OOS만 슬라이싱합니다.
    fold_details: step5 또는 step4에서 반환된 데이터프레임
    """
    LOOKBACK_DAYS = 35  # Step 5와 동일한 Lookback 기간

    # 결과를 담을 빈 데이터프레임 생성
    result = pd.DataFrame(index=df.index)
    result["close"] = df["close"]
    result["signal"] = np.nan
    result["position"] = np.nan
    result["strategy_return"] = np.nan
    result["base"] = np.nan
    result["upper"] = np.nan
    result["lower"] = np.nan
    result["alpha_up"] = np.nan
    result["alpha_down"] = np.nan
    result["beta_up"] = np.nan
    result["beta_down"] = np.nan
    result["vol_period"] = np.nan
    result["volume_period"] = np.nan

    print(f"🔍 Rolling 파라미터 적용: 총 {len(fold_details)}개 폴드")
    print(f"📦 Lookback 기간: {LOOKBACK_DAYS}일 (OOS 이전 데이터 포함)")

    # 각 폴드의 인덱스 위치를 미리 계산 (성능 최적화)
    date_to_idx = {date: i for i, date in enumerate(df.index)}

    for _idx, row in tqdm(
        fold_details.iterrows(), total=len(fold_details), desc="OOS 구간 적용"
    ):
        val_start = row["val_start"]
        val_end = row["val_end"]

        # 해당 OOS 구간의 인덱스 위치
        start_idx = date_to_idx.get(val_start)
        end_idx = date_to_idx.get(val_end)

        if start_idx is None or end_idx is None:
            continue

        # 🔥 Lookback을 포함한 계산 구간 설정
        calc_start_idx = max(0, start_idx - LOOKBACK_DAYS)
        calc_end_idx = end_idx + 1  # 슬라이싱은 end_idx 미만이므로 +1

        # 계산용 데이터프레임 (Lookback 포함)
        df_calc = df.iloc[calc_start_idx:calc_end_idx].copy()

        # 해당 폴드의 파라미터
        params = {
            "alpha_up": row["alpha_up"],
            "alpha_down": row["alpha_down"],
            "beta_up": row["beta_up"],
            "beta_down": row["beta_down"],
            "vol_period": int(row["vol_period"]),
            "volume_period": int(row["volume_period"]),
        }

        # 신호 생성 (Step 5와 동일한 방식)
        bands = compute_bands_flexible(
            df_calc,
            vol_period=params["vol_period"],
            volume_period=params["volume_period"],
            alpha_up=params["alpha_up"],
            alpha_down=params["alpha_down"],
            beta_up=params["beta_up"],
            beta_down=params["beta_down"],
        )

        close = df_calc["close"].values
        upper = bands["upper"].values
        lower = bands["lower"].values

        # 전체 계산 구간에 대한 신호 생성
        signal_full = np.where(close > upper, 2, np.where(close < lower, 0, 1))
        position_full = np.where(signal_full == 2, 1, np.where(signal_full == 0, -1, 0))

        # 🔥 OOS 구간만 슬라이싱 (Step 5와 동일)
        oos_offset = start_idx - calc_start_idx  # OOS 시작 위치
        signal = signal_full[oos_offset:]
        position = position_full[oos_offset:]

        # OOS 데이터 (수익률 계산용)
        df_oos = df.iloc[start_idx : end_idx + 1].copy()
        market_ret = df_oos["close"].pct_change().values

        # Look-ahead 방지: 포지션 1일 Shift
        pos_shifted = np.roll(position, 1)
        if len(pos_shifted) > 0:
            pos_shifted[0] = 0  # 첫날은 포지션 없음
        strategy_ret = pos_shifted * market_ret

        # 결과 저장
        mask = (df.index >= val_start) & (df.index <= val_end)
        result.loc[mask, "signal"] = signal
        result.loc[mask, "position"] = position
        result.loc[mask, "strategy_return"] = strategy_ret

        # 기준선 값은 Lookback 포함된 전체 값에서 OOS만 슬라이싱하여 저장
        bands_oos = bands.iloc[oos_offset:]
        result.loc[mask, "base"] = bands_oos["base"].values
        result.loc[mask, "upper"] = bands_oos["upper"].values
        result.loc[mask, "lower"] = bands_oos["lower"].values

        result.loc[mask, "alpha_up"] = params["alpha_up"]
        result.loc[mask, "alpha_down"] = params["alpha_down"]
        result.loc[mask, "beta_up"] = params["beta_up"]
        result.loc[mask, "beta_down"] = params["beta_down"]
        result.loc[mask, "vol_period"] = params["vol_period"]
        result.loc[mask, "volume_period"] = params["volume_period"]

    return result


# ============================================================
# 3. 성과 평가 (연결된 OOS 기준)
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


def evaluate_signals(
    df_signals: pd.DataFrame,
    y_true: np.ndarray = None,
) -> dict:
    """신호 결과를 평가합니다."""
    # 수익률 지표
    ret = df_signals["strategy_return"].values
    perf = calculate_metrics(ret)

    # 분류 지표
    cls = {}
    if y_true is not None:
        valid_mask = ~(np.isnan(df_signals["signal"].values) | np.isnan(y_true))
        if valid_mask.sum() > 0:
            y_pred = df_signals["signal"].values[valid_mask].astype(int)
            y_true_clean = y_true[valid_mask].astype(int)
            cls["f1_macro"] = f1_score(y_true_clean, y_pred, average="macro")
            cls["balanced_acc"] = balanced_accuracy_score(y_true_clean, y_pred)
            unique, counts = np.unique(y_pred, return_counts=True)
            ratio_dict = dict(zip(unique, counts / len(y_pred),strict=False))
            cls["ratio_up"] = ratio_dict.get(2, 0.0)
            cls["ratio_neutral"] = ratio_dict.get(1, 0.0)
            cls["ratio_down"] = ratio_dict.get(0, 0.0)
        else:
            cls = {
                "f1_macro": np.nan,
                "balanced_acc": np.nan,
                "ratio_up": np.nan,
                "ratio_neutral": np.nan,
                "ratio_down": np.nan,
            }

    return {**perf, **cls}


# ============================================================
# 4. 메인 실행
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 6단계: 최종 기준선 산출 및 신호 생성 (실전 적용)")
    print("=" * 60)

    # 1) 데이터 로드
    df = load_data()
    print(f"📊 데이터 로드 완료: {df.shape[0]}일")

    # 2) 실제 레이블 생성 (평가용)
    ret = df["close"].pct_change().values
    y_true = np.where(ret > 0.005, 2, np.where(ret < -0.005, 0, 1))

    # ============================================================
    # 3-1) Rolling 파라미터 적용 (CSV 자동 로드)
    # ============================================================
    print("\n" + "=" * 60)
    print("📌 [모드 A] Rolling 파라미터 적용 (각 OOS 구간별 최적 파라미터)")
    print("=" * 60)

    csv_path = "wf_fold_details_6params.csv"

    if os.path.exists(csv_path):
        fold_details = pd.read_csv(csv_path, index_col=0)
        fold_details["val_start"] = pd.to_datetime(fold_details["val_start"])
        fold_details["val_end"] = pd.to_datetime(fold_details["val_end"])
        print(f"✅ step5 결과 CSV 로드 완료: {len(fold_details)}개 폴드")
        print(f"   (파일: {csv_path})")
    else:
        print(f"⚠️  '{csv_path}' 파일이 없습니다. 데모 모드로 실행합니다.")
        dates = df.index
        n_folds = min(20, (len(dates) - 504) // 21)
        demo_folds = []
        for i in range(n_folds):
            start_idx = i * 21
            train_end_idx = start_idx + 504
            val_end_idx = train_end_idx + 63
            if val_end_idx >= len(dates):
                break
            demo_folds.append(
                {
                    "val_start": dates[train_end_idx],
                    "val_end": dates[val_end_idx - 1],
                    "alpha_up": 0.8 + np.random.rand() * 0.6,
                    "alpha_down": 0.8 + np.random.rand() * 0.6,
                    "beta_up": -0.3 + np.random.rand() * 0.6,
                    "beta_down": -0.3 + np.random.rand() * 0.6,
                    "vol_period": np.random.randint(12, 18),
                    "volume_period": np.random.randint(15, 25),
                }
            )
        fold_details = pd.DataFrame(demo_folds)
        print(f"✅ 데모 폴드 생성 완료: {len(fold_details)}개")

    # Rolling 신호 생성
    df_signal_rolling = generate_signals_rolling(df, fold_details)
    perf_rolling = evaluate_signals(df_signal_rolling, y_true)

    print("\n📈 Rolling 파라미터 최종 성과:")
    print(f"   Sharpe Ratio       : {perf_rolling['sharpe']:.4f}")
    print(f"   CAGR               : {perf_rolling['cagr']:.4%}")
    print(f"   MDD                : {perf_rolling['mdd']:.4%}")
    print(f"   Calmar Ratio       : {perf_rolling['calmar']:.4f}")
    print(f"   승률 (Win Rate)    : {perf_rolling['win_rate']:.4%}")
    print(f"   Profit Factor      : {perf_rolling['profit_factor']:.4f}")
    print(f"   Macro-F1           : {perf_rolling.get('f1_macro', np.nan):.4f}")
    print(f"   예측 중립 비율     : {perf_rolling.get('ratio_neutral', np.nan):.4%}")

    # ============================================================
    # 3-2) Median 파라미터 적용
    # ============================================================
    print("\n" + "=" * 60)
    print("📌 [모드 B] Median 파라미터 적용 (전체 폴드 중앙값, 안정성 검증용)")
    print("=" * 60)

    median_params = {
        "alpha_up": fold_details["alpha_up"].median(),
        "alpha_down": fold_details["alpha_down"].median(),
        "beta_up": fold_details["beta_up"].median(),
        "beta_down": fold_details["beta_down"].median(),
        "vol_period": int(round(fold_details["vol_period"].median())),
        "volume_period": int(round(fold_details["volume_period"].median())),
    }
    print("Median 파라미터:")
    for k, v in median_params.items():
        if k in ["vol_period", "volume_period"]:
            print(f"   {k} = {v}일")
        else:
            print(f"   {k} = {v:.4f}")

    df_signal_median = generate_signals_single(df, median_params)
    perf_median = evaluate_signals(df_signal_median, y_true)

    print("\n📈 Median 파라미터 최종 성과:")
    print(f"   Sharpe Ratio       : {perf_median['sharpe']:.4f}")
    print(f"   CAGR               : {perf_median['cagr']:.4%}")
    print(f"   MDD                : {perf_median['mdd']:.4%}")
    print(f"   Calmar Ratio       : {perf_median['calmar']:.4f}")
    print(f"   승률 (Win Rate)    : {perf_median['win_rate']:.4%}")
    print(f"   Profit Factor      : {perf_median['profit_factor']:.4f}")
    print(f"   Macro-F1           : {perf_median.get('f1_macro', np.nan):.4f}")
    print(f"   예측 중립 비율     : {perf_median.get('ratio_neutral', np.nan):.4%}")

    # ============================================================
    # 3-3) 최종 비교
    # ============================================================
    print("\n" + "=" * 60)
    print("📊 Rolling vs Median 성능 비교")
    print("=" * 60)
    print(f"{'지표':<20} {'Rolling':<15} {'Median':<15} {'차이':<15}")
    print("-" * 65)

    metrics_to_compare = [
        "sharpe",
        "cagr",
        "mdd",
        "calmar",
        "win_rate",
        "profit_factor",
        "f1_macro",
    ]
    for m in metrics_to_compare:
        v1 = perf_rolling.get(m, np.nan)
        v2 = perf_median.get(m, np.nan)
        diff = v1 - v2 if not np.isnan(v1) and not np.isnan(v2) else np.nan
        fmt = "{:.4f}" if m not in ["cagr", "win_rate"] else "{:.4%}"
        print(
            f"{m:<20} {fmt.format(v1) if not np.isnan(v1) else 'NaN':<15} "
            f"{fmt.format(v2) if not np.isnan(v2) else 'NaN':<15} "
            f"{fmt.format(diff) if not np.isnan(diff) else 'NaN':<15}"
        )

    print("\n" + "=" * 60)
    print("✅ 6단계 실전 적용 완료!")
    print("\n📌 실전 운용 가이드:")
    print("   1. Rolling 파라미터를 기본 실전 모델로 사용하세요.")
    print("   2. Median 파라미터는 안정성 검증용 벤치마크로 활용하세요.")
    print(
        "   3. 매월 또는 분기별로 Walk-Forward를 재실행하여 파라미터를 업데이트하세요."
    )
    print("   4. 신호는 당일 종가 기준 생성 후, 익일 시가 또는 종가에 매매하세요.")
    print("=" * 60)
