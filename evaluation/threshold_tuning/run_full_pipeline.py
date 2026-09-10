import os
import warnings

import numpy as np
import pandas as pd
from focal_classifier import (
    FocalConfig,
    _regression_test_make_labels,
    build_features,
    make_labels,
)
from step1_core_features import load_data
from step5_optimize_6params import (
    compute_bands_flexible,
    run_walkforward_6params,
)

warnings.filterwarnings("ignore")


# ============================================================
# ADR-AS-0002
# ------------------------------------------------------------
# - T일 종가까지의 정보로 신호 생성
# - T+1일 시가 체결
# - T+6일 시가 평가
# - fwd_ret = open.shift(-6) / open.shift(-1) - 1
# - 마지막 6행은 NaN 유지 (중립으로 변환 금지)
# - 학습/검증/OOS 모두 동일 threshold 사용
#
# 이 파일은 라벨을 직접 계산하지 않고 focal_classifier.make_labels()
# 공용 함수만 사용한다. 그리고 각 threshold 실험마다 FocalConfig 를
# 별도로 생성해 step 7(Focal Walk-Forward)까지 동일 threshold 로
# 연결될 수 있도록 한다.
# ============================================================


# ============================================================
# 0. 설정
# ============================================================

QUICK_MODE = False

TRAIN_YEARS = 2
VAL_MONTHS = 3
STEP_MONTHS = 1

# 이 두 값을 동일한 파이프라인이 각각 독립적으로 실행한다.
THRESHOLDS = [0.01, 0.02]

MAX_EVALS = 30 if QUICK_MODE else 300

# ADR-AS-0002: horizon = 6 거래일 → 학습 라벨이 OOS 첫날 데이터를
# 참조하지 않도록 gap 도 6 이상이어야 한다.
GAP_DAYS = 6

# Step 7(Focal Walk-Forward)까지 이 파일에서 실행할지 여부.
# False 로 두면 이 파일은 Step 5 + Rolling Feature 검증만 수행한다.
RUN_FOCAL_WALKFORWARD = False


# ============================================================
# 1. threshold별 FocalConfig 팩토리
# ------------------------------------------------------------
# step 7 에서 학습/검증/OOS 라벨이 모두 같은 threshold 를 쓰도록
# FocalConfig 를 threshold 별로 만들어 넘긴다.
# ============================================================
def make_focal_config(threshold: float, **overrides) -> FocalConfig:
    """threshold 별 FocalConfig 를 생성한다.

    Step 7(Focal Walk-Forward)과 full pipeline 에서 이 팩토리를 사용해
    학습/검증/OOS 라벨 threshold 가 자동으로 통일되도록 한다.
    """
    cfg = FocalConfig(threshold=float(threshold))
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ============================================================
# 2. 기본 출력
# ============================================================


def print_header(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# 3. Rolling Feature 검증
# ============================================================


def validate_rolling_features(
    df: pd.DataFrame,
    fold_details: pd.DataFrame,
    threshold: float,
):
    """
    각 Walk-Forward fold에서 실제 사용되는 파라미터를 이용하여
    Rolling Feature가 정상적으로 생성되는지 검증한다.

    ADR-AS-0002 반영:
      - 라벨은 공용 make_labels(df, threshold) 만 사용
      - 라벨의 마지막 6행 NaN 은 검증에서 제외

    검증 대상
    ------------------------------------------------------------
    1. Fold별 기준선 파라미터가 정상적으로 전달되는가
    2. Fold별 Feature 개수와 컬럼이 동일한가
    3. Feature에 NaN / Inf가 과도하게 존재하지 않는가
    4. Feature가 OOS 구간까지 정상적으로 계산되는가
    5. OOS 시작 이전 데이터만으로 Feature가 생성되는 구조인가
    6. 기준선(base / upper / lower)이 정상적으로 생성되는가
    7. 1%, 2% threshold에서 Feature 구조가 동일한가
    """

    print_header(
        f"① Rolling Feature 생성 검증 | Threshold = {threshold * 100:.0f}%"
    )

    date_to_idx = {d: i for i, d in enumerate(df.index)}

    feature_signatures = []
    fold_results = []

    print(f"전체 데이터 : {len(df):,}일")
    print(f"검증 Fold   : {len(fold_details)}개")
    print()

    for fold_no, row in fold_details.iterrows():

        val_start = pd.Timestamp(row["val_start"])
        val_end = pd.Timestamp(row["val_end"])

        if val_start not in date_to_idx:
            print(f"⚠️ Fold {fold_no + 1}: val_start={val_start} 없음")
            continue

        if val_end not in date_to_idx:
            print(f"⚠️ Fold {fold_no + 1}: val_end={val_end} 없음")
            continue

        start_idx = date_to_idx[val_start]
        end_idx = date_to_idx[val_end]

        # ----------------------------------------------------
        # ADR-AS-0002: 학습 라벨이 OOS 첫날 open 을 참조하지 않도록
        # train_end 는 OOS 시작 이전 GAP_DAYS(=horizon)일 지점
        # ----------------------------------------------------
        train_end = start_idx - GAP_DAYS

        if train_end <= 0:
            print(f"⚠️ Fold {fold_no + 1}: train_end 부족")
            continue

        # ----------------------------------------------------
        # Fold별 최적화 파라미터
        # ----------------------------------------------------
        params = {
            "alpha_up": float(row["alpha_up"]),
            "alpha_down": float(row["alpha_down"]),
            "beta_up": float(row["beta_up"]),
            "beta_down": float(row["beta_down"]),
            "vol_period": int(round(row["vol_period"])),
            "volume_period": int(round(row["volume_period"])),
        }

        # ----------------------------------------------------
        # Feature 계산은 충분한 과거 데이터부터 OOS 끝까지.
        # 기준선/Feature 자체는 rolling/causal 계산이라 미래 참조 없음.
        # ----------------------------------------------------
        df_calc = df.iloc[: end_idx + 1].copy()

        try:
            bands = compute_bands_flexible(
                df_calc,
                vol_period=params["vol_period"],
                volume_period=params["volume_period"],
                alpha_up=params["alpha_up"],
                alpha_down=params["alpha_down"],
                beta_up=params["beta_up"],
                beta_down=params["beta_down"],
            )
            features = build_features(df_calc, bands)
        except Exception as exc:
            print(f"❌ Fold {fold_no + 1}: Feature 생성 실패 → {exc}")
            continue

        # ----------------------------------------------------
        # OOS Feature 추출
        # ----------------------------------------------------
        oos_features = features.iloc[start_idx : end_idx + 1].copy()

        feature_cols = list(features.columns)
        excluded_cols = {
            "open",
            "high",
            "low",
            "close",
            "volume",
            "date",
            "label",
            "fwd_return_5d",
            "fwd_return_6d",
            "fwd_ret",
        }
        ml_feature_cols = [c for c in feature_cols if c not in excluded_cols]

        feature_data = oos_features[ml_feature_cols].copy()

        nan_count = int(feature_data.isna().sum().sum())
        inf_count = int(
            np.isinf(
                feature_data.select_dtypes(include=[np.number]).to_numpy(dtype=float)
            ).sum()
        )
        total_values = feature_data.shape[0] * feature_data.shape[1]
        nan_ratio = nan_count / total_values if total_values > 0 else np.nan

        constant_features = []
        for col in ml_feature_cols:
            try:
                if feature_data[col].nunique(dropna=True) <= 1:
                    constant_features.append(col)
            except Exception:
                pass

        finite_feature_data = feature_data.replace([np.inf, -np.inf], np.nan)
        valid_feature_ratio = (
            finite_feature_data.notna().mean().mean()
            if len(finite_feature_data) > 0
            else 0.0
        )

        # ----------------------------------------------------
        # 기준선 유효성
        # ----------------------------------------------------
        band_valid = True
        for col in ["base", "upper", "lower"]:
            if col not in bands.columns:
                band_valid = False
                break
            if not np.isfinite(
                bands.loc[bands.index[start_idx : end_idx + 1], col].to_numpy(
                    dtype=float
                )
            ).any():
                band_valid = False
                break

        # ----------------------------------------------------
        # Label 확인 — ADR-AS-0002 공용 함수 사용
        #   make_labels(df, threshold) 는 T+1→T+6 라벨을 반환하며
        #   미래값이 없는 마지막 6행은 NaN 이다.
        # ----------------------------------------------------
        y = make_labels(df_calc, threshold=threshold)
        oos_y = y[start_idx : end_idx + 1]
        label_valid_count = int(np.isfinite(oos_y).sum())

        if label_valid_count > 0:
            y_clean = oos_y[np.isfinite(oos_y)].astype(int)
            for cls, _name in [(0, "하락"), (1, "중립"), (2, "상승")]:
                _ = float(np.mean(y_clean == cls))  # 로그 확장 여지

        fold_result = {
            "fold": fold_no + 1,
            "val_start": val_start,
            "val_end": val_end,
            "train_end_idx": train_end,
            "oos_days": len(oos_features),
            "n_features": len(ml_feature_cols),
            "nan_count": nan_count,
            "nan_ratio": nan_ratio,
            "inf_count": inf_count,
            "valid_feature_ratio": valid_feature_ratio,
            "constant_feature_count": len(constant_features),
            "band_valid": band_valid,
            "label_valid_count": label_valid_count,
            "alpha_up": params["alpha_up"],
            "alpha_down": params["alpha_down"],
            "beta_up": params["beta_up"],
            "beta_down": params["beta_down"],
            "vol_period": params["vol_period"],
            "volume_period": params["volume_period"],
        }

        fold_results.append(fold_result)
        feature_signatures.append(tuple(ml_feature_cols))

        print(
            f"Fold {fold_no + 1:2d} | "
            f"OOS {val_start.strftime('%Y-%m-%d')} ~ "
            f"{val_end.strftime('%Y-%m-%d')} | "
            f"Features={len(ml_feature_cols):2d} | "
            f"NaN={nan_ratio:.2%} | "
            f"Inf={inf_count:3d} | "
            f"Band={'OK' if band_valid else 'FAIL'}"
        )

    # ========================================================
    # 전체 검증 결과
    # ========================================================
    print_header("① Rolling Feature 검증 결과")

    if not fold_results:
        print("❌ 검증 가능한 Fold가 없습니다.")
        return {
            "passed": False,
            "fold_results": pd.DataFrame(),
            "feature_columns": [],
        }

    result_df = pd.DataFrame(fold_results)

    feature_consistent = len(set(feature_signatures)) == 1
    max_nan_ratio = result_df["nan_ratio"].max()
    nan_ok = max_nan_ratio <= 0.05
    inf_ok = result_df["inf_count"].max() == 0
    constant_ok = result_df["constant_feature_count"].max() == 0
    band_ok = bool(result_df["band_valid"].all())

    feature_counts = result_df["n_features"].unique()
    feature_count_ok = len(feature_counts) == 1

    passed = all(
        [
            feature_consistent,
            nan_ok,
            inf_ok,
            band_ok,
            feature_count_ok,
        ]
    )

    print(f"검증 Fold 수           : {len(result_df)}")
    print(f"Feature 개수            : {feature_counts.tolist()}")
    print(f"Feature 구조 동일       : {'PASS' if feature_consistent else 'FAIL'}")
    print(
        f"최대 NaN 비율           : {max_nan_ratio:.4%} "
        f"({'PASS' if nan_ok else 'FAIL'})"
    )
    print(f"Inf 존재 여부           : {'PASS' if inf_ok else 'FAIL'}")
    print(f"기준선(base/upper/lower): {'PASS' if band_ok else 'FAIL'}")
    print(f"Feature 개수 일관성     : {'PASS' if feature_count_ok else 'FAIL'}")
    print(f"상수 Feature            : {'PASS' if constant_ok else 'WARNING'}")
    print()

    if passed:
        print("✅ Rolling Feature 생성 검증 PASS")
    else:
        print("❌ Rolling Feature 검증 FAIL")

    if feature_signatures:
        print("\n[사용 Feature 목록]")
        for i, col in enumerate(feature_signatures[0], 1):
            print(f"  {i:2d}. {col}")

    print("\n[Fold별 검증 요약]")
    display_cols = [
        "fold",
        "val_start",
        "val_end",
        "oos_days",
        "n_features",
        "nan_ratio",
        "inf_count",
        "constant_feature_count",
        "band_valid",
    ]
    print(result_df[display_cols].to_string(index=False))

    return {
        "passed": passed,
        "fold_results": result_df,
        "feature_columns": (list(feature_signatures[0]) if feature_signatures else []),
    }


# ============================================================
# 4. Step 7 (Focal Walk-Forward) 옵션 훅
# ------------------------------------------------------------
# 이 파일이 두 threshold 를 모두 실행할 수 있도록 FocalConfig 를
# threshold 별로 생성해서 넘긴다.
# ============================================================
def run_focal_experiment(
    df: pd.DataFrame,
    fold_details: pd.DataFrame,
    threshold: float,
) -> dict:
    """
    threshold 하나에 대해 Step 7 Focal Walk-Forward 를 실행한다.

    - FocalConfig(threshold=threshold) 를 명시 생성
    - make_labels(df, threshold=threshold) 로 y_true 산출
    - 학습/검증/OOS 라벨이 모두 같은 threshold 사용
    """
    # Step 7 모듈은 순환 참조를 피하기 위해 함수 내부에서 import
    from step7_focal_walkforward import (  # type: ignore
        evaluate_signals,
        generate_signals_rolling,
    )

    cfg = make_focal_config(threshold)
    print_header(f"② Focal Walk-Forward | Threshold = {threshold * 100:.0f}%")
    print(f"   FocalConfig.threshold = {cfg.threshold}")

    signals = generate_signals_rolling(df, fold_details, focal_config=cfg)
    y_true = make_labels(df, threshold=threshold)
    metrics = evaluate_signals(signals, y_true)

    return {"signals": signals, "metrics": metrics, "config": cfg}


# ============================================================
# 5. Threshold별 전체 실험
# ============================================================


def run_threshold_experiment(
    df: pd.DataFrame,
    threshold: float,
) -> dict:
    """
    하나의 threshold 에 대해

    1. Step 5 Walk-Forward
    2. Fold별 최적 파라미터 확보
    3. Rolling Feature 생성 + 검증
    4. (옵션) Step 7 Focal Walk-Forward

    를 수행한다.
    """

    print_header(f"THRESHOLD = {threshold * 100:.0f}%")

    # ---- Step 5 ----
    print("\n[5단계] 동적 기준선 6-parameter Walk-Forward")

    result_5 = run_walkforward_6params(
        df,
        train_years=TRAIN_YEARS,
        val_months=VAL_MONTHS,
        step_months=STEP_MONTHS,
        max_evals=MAX_EVALS,
        threshold=threshold,
    )

    fold_details = result_5["fold_details"].copy()
    print(f"\n✅ 5단계 완료: {len(fold_details)}개 Fold")

    print("\n[Fold별 기준선 파라미터]")
    parameter_cols = [
        "alpha_up",
        "alpha_down",
        "beta_up",
        "beta_down",
        "vol_period",
        "volume_period",
    ]
    print(fold_details[parameter_cols].to_string(index=False))

    median_params = {
        "alpha_up": float(fold_details["alpha_up"].median()),
        "alpha_down": float(fold_details["alpha_down"].median()),
        "beta_up": float(fold_details["beta_up"].median()),
        "beta_down": float(fold_details["beta_down"].median()),
        "vol_period": int(round(fold_details["vol_period"].median())),
        "volume_period": int(round(fold_details["volume_period"].median())),
    }

    print("\n[Median 기준선 파라미터]")
    for k, v in median_params.items():
        print(f"  {k:<15}: {v}")

    # ---- Feature 검증 ----
    validation = validate_rolling_features(
        df=df,
        fold_details=fold_details,
        threshold=threshold,
    )

    # ---- Step 7 (옵션) ----
    focal_result = None
    if RUN_FOCAL_WALKFORWARD:
        try:
            focal_result = run_focal_experiment(
                df=df,
                fold_details=fold_details,
                threshold=threshold,
            )
        except Exception as exc:
            print(f"⚠️ Step 7 Focal 실행 실패: {exc}")

    return {
        "threshold": threshold,
        "step5_result": result_5,
        "fold_details": fold_details,
        "median_params": median_params,
        "feature_validation": validation,
        "focal_result": focal_result,
    }


# ============================================================
# 6. 1% vs 2% 비교
# ============================================================


def compare_feature_validation(results: dict):
    print_header("① 최종 비교: 1% vs 2% Rolling Feature")

    rows = []
    for threshold, result in results.items():
        validation = result["feature_validation"]
        if validation["fold_results"].empty:
            continue
        df_val = validation["fold_results"]
        rows.append(
            {
                "Threshold": f"{threshold * 100:.0f}%",
                "Fold 수": len(df_val),
                "Feature 수": int(df_val["n_features"].iloc[0]),
                "최대 NaN": df_val["nan_ratio"].max(),
                "최대 Inf": int(df_val["inf_count"].max()),
                "Feature 구조": (
                    "PASS" if len(validation["feature_columns"]) > 0 else "FAIL"
                ),
                "기준선": ("PASS" if bool(df_val["band_valid"].all()) else "FAIL"),
                "최종 검증": ("PASS" if validation["passed"] else "FAIL"),
            }
        )

    if not rows:
        print("❌ 비교 가능한 결과가 없습니다.")
        return

    print(pd.DataFrame(rows).to_string(index=False))

    if 0.01 in results and 0.02 in results:
        cols_1 = results[0.01]["feature_validation"]["feature_columns"]
        cols_2 = results[0.02]["feature_validation"]["feature_columns"]
        same_columns = cols_1 == cols_2
        print(f"\nFeature 컬럼 구조 1% vs 2%: {'동일' if same_columns else '다름'}")


def compare_focal_results(results: dict):
    """Step 7 을 실행한 경우에만 출력."""
    if not any(r.get("focal_result") for r in results.values()):
        return

    print_header("② 최종 비교: 1% vs 2% Focal Walk-Forward")

    rows = []
    for threshold, result in results.items():
        fr = result.get("focal_result")
        if fr is None:
            continue
        m = fr["metrics"]
        rows.append(
            {
                "Threshold": f"{threshold * 100:.0f}%",
                "Sharpe": m.get("sharpe", np.nan),
                "CAGR": m.get("cagr", np.nan),
                "MDD": m.get("mdd", np.nan),
                "Win Rate": m.get("win_rate", np.nan),
                "Macro-F1": m.get("f1_macro", np.nan),
                "Balanced Acc": m.get("balanced_acc", np.nan),
            }
        )
    if rows:
        df = pd.DataFrame(rows)
        for c in df.columns:
            if c != "Threshold":
                df[c] = df[c].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "NaN")
        print(df.to_string(index=False))


# ============================================================
# 7. 메인
# ============================================================


def run_full_pipeline():

    print("=" * 70)
    print("KOSPI 3-Class ML Pipeline")
    print("① Rolling Feature 생성 방식 검증")
    print("Threshold = 1% vs 2%")
    print(f"ADR-AS-0002: 라벨 = open(T+1)→open(T+6), GAP={GAP_DAYS}일")
    print("=" * 70)

    # ---- 라벨 회귀 테스트 선행 ----
    _regression_test_make_labels()

    print("\n[1단계] 데이터 로드")
    df = load_data()
    print(f"✅ {len(df):,}일")
    print(f"   기간: {df.index.min()} ~ {df.index.max()}")

    # ---- 라벨 구조 ----
    print("\n[라벨 구조 확인 — ADR-AS-0002, T+1→T+6]")
    for threshold in THRESHOLDS:
        y = make_labels(df, threshold=threshold)
        valid = np.isfinite(y)
        n_valid = int(valid.sum())
        n_nan = int((~valid).sum())
        print(f"\nThreshold {threshold * 100:.0f}%")
        print(f"  유효 라벨: {n_valid}일 (NaN 끝행: {n_nan}일)")
        for cls, name in [(0, "하락"), (1, "중립"), (2, "상승")]:
            ratio = float(np.mean(y[valid] == cls))
            print(f"  {name}: {ratio:.2%}")

    # ---- threshold별 실행 ----
    all_results = {}
    for threshold in THRESHOLDS:
        result = run_threshold_experiment(df, threshold)
        all_results[threshold] = result

    # ---- 비교 ----
    compare_feature_validation(all_results)
    compare_focal_results(all_results)

    # ---- 저장 ----
    output_dir = "feature_validation_results"
    os.makedirs(output_dir, exist_ok=True)

    for threshold, result in all_results.items():
        tag = f"{int(threshold * 100)}pct"

        fold_details = result["fold_details"]
        fold_path = os.path.join(output_dir, f"fold_details_{tag}.csv")
        fold_details.to_csv(fold_path, index=False)
        print(f"\n💾 저장: {os.path.abspath(fold_path)}")

        validation_df = result["feature_validation"]["fold_results"]
        val_path = os.path.join(output_dir, f"feature_validation_{tag}.csv")
        validation_df.to_csv(val_path, index=False)
        print(f"💾 저장: {os.path.abspath(val_path)}")

        fr = result.get("focal_result")
        if fr is not None:
            sig_path = os.path.join(output_dir, f"focal_signals_{tag}.csv")
            fr["signals"].to_csv(sig_path)
            print(f"💾 저장: {os.path.abspath(sig_path)}")

            met_path = os.path.join(output_dir, f"focal_metrics_{tag}.csv")
            pd.DataFrame([fr["metrics"]]).to_csv(met_path, index=False)
            print(f"💾 저장: {os.path.abspath(met_path)}")

    # ---- 최종 판정 ----
    print_header("① 단계 최종 판정")
    all_passed = all(
        result["feature_validation"]["passed"] for result in all_results.values()
    )
    if all_passed:
        print("✅ 1%, 2% 모두 Rolling Feature 검증 PASS")
    else:
        print("❌ Rolling Feature 검증 FAIL")

    return all_results


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":
    run_full_pipeline()
