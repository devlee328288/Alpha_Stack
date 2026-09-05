import warnings

import numpy as np
import step6_live_signal

# ============================================================
# 1. 필수 모듈 임포트 (core_features, step5, step6)
# ============================================================
from step1_core_features import load_data
from step5_optimize_6params import run_walkforward_6params
from step6_live_signal import (
    evaluate_signals,
    generate_signals_rolling,
    generate_signals_single,
)

warnings.filterwarnings("ignore")
"""
============================================================
🚀 KOSPI 동적 기준선(상승/중립/하락) Full Pipeline 실행기
============================================================
1~6단계를 순차적으로 실행하고 최종 결과를 한 번에 출력합니다.

Usage:
    python run_full_pipeline.py
    (QUICK_MODE = True로 설정하면 빠르게 테스트 가능)
============================================================
"""

print(f"🔥 step6 파일 경로: {step6_live_signal.__file__}")

# ============================================================
# 0. 설정 (Configuration)
# ============================================================
# 🔥 빠른 테스트: True로 하면 max_evals가 30으로 줄어들어 1분 안에 끝남
# 🔥 실제 분석: False로 하면 정확한 결과를 위해 10~20분 소요
QUICK_MODE = False

# Walk-Forward 기본 설정 (문서 기준: 2년 학습, 3개월 검증, 1개월 이동)
TRAIN_YEARS = 2
VAL_MONTHS = 3
STEP_MONTHS = 1
MD_THRESHOLD = 0.20  # 허용 MDD 20%
LAMBDA_MDD = 0.5  # MDD 패널티 강도

# ============================================================
# 2. 메인 파이프라인 실행 함수
# ============================================================
def run_full_pipeline():
    print("=" * 70)
    print("🚀 KOSPI 동적 기준선 Full Pipeline 실행 시작")
    print("=" * 70)

    # ---- 1) 데이터 로드 ----
    print("\n[1단계] 데이터 로드 중...")
    df = load_data()
    print(f"✅ 데이터 로드 완료: {df.shape[0]}일 ({df.index.min()} ~ {df.index.max()})")

    # 실제 레이블 생성 (평가용)
    ret = df["close"].pct_change().values
    y_true = np.where(ret > 0.005, 2, np.where(ret < -0.003, 0, 1))

    # ---- 2) 5단계 실행 (6개 파라미터 CMA-ES Walk-Forward) ----
    print("\n" + "=" * 70)
    print("[5단계] 6개 파라미터(α, β, Vol_Period, Volume_Period) Walk-Forward 최적화")
    print("=" * 70)

    if QUICK_MODE:
        max_evals = 30  # 빠른 테스트용 (실제는 300~500 권장)
        print("⚡ QUICK_MODE 활성화: max_evals=30 (결과의 정확도가 낮을 수 있음)")
    else:
        max_evals = 300  # 정확한 분석용
        print("🔬 정밀 모드: max_evals=300 (약 1~5분 소요 예상)")

    result_5 = run_walkforward_6params(
        df,
        train_years=TRAIN_YEARS,
        val_months=VAL_MONTHS,
        step_months=STEP_MONTHS,
        max_evals=max_evals,
    )

    fold_details = result_5["fold_details"]

    print("\n📌 [전달된 fold_details 샘플 (처음 5개)]")
    print(
        fold_details[
            [
                "alpha_up",
                "alpha_down",
                "beta_up",
                "beta_down",
                "vol_period",
                "volume_period",
            ]
        ].head()
    )

    print("\n" + "=" * 70)
    print("📊 [Rolling 파라미터 실태 진단 (143개 폴드 평균)]")
    print("=" * 70)

    alpha_up_mean = fold_details['alpha_up'].mean()
    alpha_up_max = fold_details['alpha_up'].max()
    print(f"   α_up  : 평균={alpha_up_mean:.3f}, 최대={alpha_up_max:.3f}")

    alpha_down_mean = fold_details['alpha_down'].mean()
    alpha_down_max = fold_details['alpha_down'].max()
    print(f"   α_down  : 평균={alpha_down_mean:.3f}, 최대={alpha_down_max:.3f}")

    beta_up_mean = fold_details['beta_up'].mean()
    beta_up_max = fold_details['beta_up'].max()
    print(f"   β_up  : 평균={beta_up_mean:.3f}, 최대={beta_up_max:.3f}")

    beta_down_mean = fold_details['beta_down'].mean()
    beta_down_max = fold_details['beta_down'].max()
    print(f"   β_down  : 평균={beta_down_mean:.3f}, 최대={beta_down_max:.3f}")

    print(f"   Vol_Period  : 중앙값={int(fold_details['vol_period'].median())}일")
    print(f"   Volume_Period: 중앙값={int(fold_details['volume_period'].median())}일")
    print("=" * 70)
    # ============================================================

    fold_details = result_5["fold_details"]
    print(f"\n✅ 5단계 완료: 총 {result_5['total_folds']}개 Walk-Forward 폴드 생성")
    print(f"   (최종 OOS Sharpe: {result_5['perf_metrics']['sharpe']:.4f})")

    # ---- 3) 6단계 실행 (Rolling 파라미터 적용) ----
    print("\n" + "=" * 70)
    print("[6단계] Rolling 파라미터로 실전 신호 생성 및 평가")
    print("=" * 70)

    # 6-1) Rolling 파라미터 적용 (각 OOS 구간별 최적 파라미터 사용)
    df_signals_rolling = generate_signals_rolling(df, fold_details)
    perf_rolling = evaluate_signals(df_signals_rolling, y_true)

    # ---- 3.5) 상세 분류 성능 분석 (Confusion Matrix / Per-class metrics) ----
    print("\n" + "=" * 70)
    print("📊 [상세 분류 성능] Rolling 예측 vs 실제 레이블")
    print("=" * 70)

    from sklearn.metrics import classification_report, confusion_matrix

    # Rolling 신호에 대한 유효 마스크 (NaN 제외)
    valid_mask = ~(np.isnan(df_signals_rolling["signal"].values) | np.isnan(y_true))
    if valid_mask.sum() > 0:
        y_true_valid = y_true[valid_mask].astype(int)
        y_pred_valid = df_signals_rolling["signal"].values[valid_mask].astype(int)

        # 1) 혼동 행렬 (Confusion Matrix)
        cm = confusion_matrix(y_true_valid, y_pred_valid, labels=[0, 1, 2])
        print("\n[혼동 행렬 (Confusion Matrix)]")
        print("(행: 실제 레이블, 열: 모델 예측)")
        print("            예측 하락(0)  예측 중립(1)  예측 상승(2)")
        print(f"실제 하락(0)     {cm[0][0]:>6}      {cm[0][1]:>6}      {cm[0][2]:>6}")
        print(f"실제 중립(1)     {cm[1][0]:>6}      {cm[1][1]:>6}      {cm[1][2]:>6}")
        print(f"실제 상승(2)     {cm[2][0]:>6}      {cm[2][1]:>6}      {cm[2][2]:>6}")

        # 2) 클래스별 Precision / Recall / F1
        print("\n[클래스별 성능 리포트 (Precision / Recall / F1-score)]")
        target_names = ['하락 (0)', '중립 (1)', '상승 (2)']
        print(classification_report(y_true_valid, y_pred_valid,
                                    target_names=target_names,
                                    digits=4
                                    ))

        # 3) 전체 정확도 (Accuracy)
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_true_valid, y_pred_valid)
        print(f"\n✅ 전체 정확도 (Accuracy): {acc:.4f} ({acc*100:.2f}%)")

    else:
        print("⚠️ 유효한 예측값이 없어 분류 성능을 출력할 수 없습니다.")

    # 6-2) Median 파라미터 적용 (안정성 검증용)
    median_params = {
        "alpha_up": fold_details["alpha_up"].median(),
        "alpha_down": fold_details["alpha_down"].median(),
        "beta_up": fold_details["beta_up"].median(),
        "beta_down": fold_details["beta_down"].median(),
        "vol_period": int(round(fold_details["vol_period"].median())),
        "volume_period": int(round(fold_details["volume_period"].median())),
    }
    df_signals_median = generate_signals_single(df, median_params)
    perf_median = evaluate_signals(df_signals_median, y_true)

    # ---- 4) 최종 결과 통합 출력 ----
    print("\n" + "=" * 70)
    print("📊 [최종 결과] Rolling vs Median 성능 비교")
    print("=" * 70)

    # 비교 테이블 생성
    metrics_keys = [
        "sharpe",
        "cagr",
        "mdd",
        "calmar",
        "win_rate",
        "profit_factor",
        "f1_macro",
        "ratio_neutral",
    ]
    labels = {
        "sharpe": "Sharpe Ratio",
        "cagr": "CAGR",
        "mdd": "MDD",
        "calmar": "Calmar Ratio",
        "win_rate": "승률",
        "profit_factor": "Profit Factor",
        "f1_macro": "Macro-F1",
        "ratio_neutral": "중립 비율",
    }
    formats = {
        "sharpe": "{:.4f}",
        "cagr": "{:.4%}",
        "mdd": "{:.4%}",
        "calmar": "{:.4f}",
        "win_rate": "{:.4%}",
        "profit_factor": "{:.4f}",
        "f1_macro": "{:.4f}",
        "ratio_neutral": "{:.4%}",
    }

    print(
        f"\n{'지표':<15} {'Rolling':<18} {'Median':<18} {'차이 (Rolling-Median)':<20}"
    )
    print("-" * 75)

    for key in metrics_keys:
        v_roll = perf_rolling.get(key, np.nan)
        v_med = perf_median.get(key, np.nan)
        diff = (
            v_roll - v_med if not np.isnan(v_roll) and not np.isnan(v_med) else np.nan
        )

        label = labels.get(key, key)
        fmt = formats.get(key, "{:.4f}")

        v_roll_str = fmt.format(v_roll) if not np.isnan(v_roll) else "NaN"
        v_med_str = fmt.format(v_med) if not np.isnan(v_med) else "NaN"
        diff_str = fmt.format(diff) if not np.isnan(diff) else "NaN"

        print(f"{label:<15} {v_roll_str:<18} {v_med_str:<18} {diff_str:<20}")

    # ---- 5) 추천 파라미터 및 최종 의견 ----
    print("\n" + "=" * 70)
    print("🎯 [실전 운용 가이드]")
    print("=" * 70)

    print("\n📌 1. 기본 실전 모델 (Rolling 적용):")
    print("   → 매월 최근 2년 데이터로 재최적화하여 파라미터를 업데이트하세요.")
    print(
        f"   → 최종 OOS Sharpe: {perf_rolling['sharpe']:.4f}, MDD: {perf_rolling['mdd']:.4%}"
    )

    print("\n📌 2. 안정성 벤치마크 (Median 파라미터):")
    print(
        f"   → α_up={median_params['alpha_up']:.3f}, α_down={median_params['alpha_down']:.3f}"
    )
    print(
        f"   → β_up={median_params['beta_up']:.3f}, β_down={median_params['beta_down']:.3f}"
    )
    print(
        f"   → Vol_Period={median_params['vol_period']}일, "
        f"Volume_Period={median_params['volume_period']}일"
    )
    print(f"   → Median OOS Sharpe: {perf_median['sharpe']:.4f}")

    print("\n📌 3. 최종 판단 기준:")

    if perf_rolling["sharpe"] > perf_median["sharpe"]:
        print(
            "   ✅ Rolling 파라미터가 Median보다 우수합니다. 시장 적응형 업데이트가 효과적입니다."
        )
    else:
        print(
            "   ⚠️ Median 파라미터가 Rolling보다 우수합니다. 과적합 가능성이 있습니다."
        )

    if perf_rolling.get("ratio_neutral", 0) < 0.15:
        print(
            "   ⚠️ 중립 비율이 15% 미만으로 낮습니다. 기준선이 좁아 매매가 잦을 수 있습니다."
        )
    elif perf_rolling.get("ratio_neutral", 0) > 0.60:
        print(
            "   ⚠️ 중립 비율이 60%를 초과합니다. 기준선이 너무 넓어 신호가 드물게 발생합니다."
        )
    else:
        print(
            "   ✅ 중립 비율이 적정 범위(15~60%)에 있습니다. 분류 밸런스가 양호합니다."
        )

    print("\n" + "=" * 70)
    print("✅ Full Pipeline 실행 완료!")
    print("=" * 70)

    return {
        "df_signals_rolling": df_signals_rolling,
        "df_signals_median": df_signals_median,
        "perf_rolling": perf_rolling,
        "perf_median": perf_median,
        "fold_details": fold_details,
        "median_params": median_params,
    }


# ============================================================
# 3. 메인 실행
# ============================================================
if __name__ == "__main__":
    result = run_full_pipeline()

    # (선택) 결과 CSV 저장
    # result["df_signals_rolling"].to_csv("signals_rolling.csv")
    # result["df_signals_median"].to_csv("signals_median.csv")
    # result["fold_details"].to_csv("fold_details_final.csv")

    print("\n💾 결과를 CSV로 저장하려면 코드 속 위 주석을 해제하세요.")
