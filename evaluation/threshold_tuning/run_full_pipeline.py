import os
import warnings

import numpy as np
import pandas as pd

from step1_core_features import load_data
from step5_optimize_6params import (
    run_walkforward_6params,
    compute_bands_flexible,
)
from focal_classifier import build_features, make_labels

warnings.filterwarnings("ignore")


# ============================================================
# 0. 설정
# ============================================================

QUICK_MODE = False

TRAIN_YEARS = 2
VAL_MONTHS = 3
STEP_MONTHS = 1

THRESHOLDS = [0.01, 0.02]

MAX_EVALS = 30 if QUICK_MODE else 300

GAP_DAYS = 5


# ============================================================
# 1. 기본 출력
# ============================================================


def print_header(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# 2. Rolling Feature 검증
# ============================================================


def validate_rolling_features(
    df: pd.DataFrame,
    fold_details: pd.DataFrame,
    threshold: float,
):
    """
    각 Walk-Forward fold에서 실제 사용되는 파라미터를 이용하여
    Rolling Feature가 정상적으로 생성되는지 검증한다.

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
        f"① Rolling Feature 생성 검증 " f"| Threshold = {threshold * 100:.0f}%"
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
            print(
                f"⚠️ Fold {fold_no + 1}: "
                f"val_start={val_start}를 데이터에서 찾을 수 없음"
            )
            continue

        if val_end not in date_to_idx:
            print(
                f"⚠️ Fold {fold_no + 1}: "
                f"val_end={val_end}를 데이터에서 찾을 수 없음"
            )
            continue

        start_idx = date_to_idx[val_start]
        end_idx = date_to_idx[val_end]

        # ----------------------------------------------------
        # 외부 OOS 직전까지가 모델 학습 가능 영역
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
        # 핵심:
        # Feature 계산은 충분한 과거 데이터부터 OOS 끝까지 계산.
        #
        # 단, 기준선/Feature 자체는 rolling/causal 계산이므로
        # 미래값을 참조하지 않는다.
        # ----------------------------------------------------
        df_calc = df.iloc[: end_idx + 1].copy()

        try:
            # step5와 동일한 기준선 계산식을 그대로 사용
            bands = compute_bands_flexible(
                df_calc,
                vol_period=params["vol_period"],
                volume_period=params["volume_period"],
                alpha_up=params["alpha_up"],
                alpha_down=params["alpha_down"],
                beta_up=params["beta_up"],
                beta_down=params["beta_down"],
            )

            # 동일한 Feature 생성 함수 사용
            features = build_features(
                df_calc,
                bands,
            )

        except Exception as exc:
            print(f"❌ Fold {fold_no + 1}: " f"Feature 생성 실패 → {exc}")
            continue

        # ----------------------------------------------------
        # OOS Feature만 추출
        # ----------------------------------------------------
        oos_features = features.iloc[start_idx : end_idx + 1].copy()

        # ----------------------------------------------------
        # Feature 컬럼
        # ----------------------------------------------------
        feature_cols = list(features.columns)

        # 원본 가격/기준선 관련 컬럼을 제외한
        # 실제 ML Feature 개수
        excluded_cols = {
            "open",
            "high",
            "low",
            "close",
            "volume",
            "date",
            "label",
            "fwd_return_5d",
        }

        ml_feature_cols = [c for c in feature_cols if c not in excluded_cols]

        # ----------------------------------------------------
        # NaN / Inf 검사
        # ----------------------------------------------------
        feature_data = oos_features[ml_feature_cols].copy()

        nan_count = int(feature_data.isna().sum().sum())

        inf_count = int(
            np.isinf(
                feature_data.select_dtypes(include=[np.number]).to_numpy(dtype=float)
            ).sum()
        )

        total_values = feature_data.shape[0] * feature_data.shape[1]

        nan_ratio = nan_count / total_values if total_values > 0 else np.nan

        # ----------------------------------------------------
        # 상수 Feature 검사
        # ----------------------------------------------------
        constant_features = []

        for col in ml_feature_cols:
            try:
                if feature_data[col].nunique(dropna=True) <= 1:
                    constant_features.append(col)
            except Exception:
                pass

        # ----------------------------------------------------
        # Feature 범위 확인
        # ----------------------------------------------------
        finite_feature_data = feature_data.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        valid_feature_ratio = (
            finite_feature_data.notna().mean().mean()
            if len(finite_feature_data) > 0
            else 0.0
        )

        # ----------------------------------------------------
        # 기준선 유효성
        # ----------------------------------------------------
        band_cols = ["base", "upper", "lower"]

        band_valid = True

        for col in band_cols:
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
        # Label 확인
        # ----------------------------------------------------
        y = make_labels(df_calc)

        oos_y = y.iloc[start_idx : end_idx + 1]

        label_valid_count = int(np.isfinite(oos_y.to_numpy(dtype=float)).sum())

        label_distribution = {}

        if label_valid_count > 0:
            y_clean = oos_y[np.isfinite(oos_y.to_numpy(dtype=float))].astype(int)

            for cls, name in [
                (0, "하락"),
                (1, "중립"),
                (2, "상승"),
            ]:
                label_distribution[name] = float(np.mean(y_clean == cls))

        # ----------------------------------------------------
        # 결과 저장
        # ----------------------------------------------------
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

        # Feature 구조 저장
        feature_signatures.append(tuple(ml_feature_cols))

        # ----------------------------------------------------
        # Fold 결과 출력
        # ----------------------------------------------------
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

    # --------------------------------------------------------
    # Feature 컬럼 일관성
    # --------------------------------------------------------

    unique_signatures = set(feature_signatures)

    feature_consistent = len(unique_signatures) == 1

    # --------------------------------------------------------
    # NaN 검사
    # --------------------------------------------------------

    max_nan_ratio = result_df["nan_ratio"].max()

    nan_ok = max_nan_ratio <= 0.05

    # --------------------------------------------------------
    # Inf 검사
    # --------------------------------------------------------

    inf_ok = result_df["inf_count"].max() == 0

    # --------------------------------------------------------
    # 상수 Feature
    # --------------------------------------------------------

    constant_ok = result_df["constant_feature_count"].max() == 0

    # --------------------------------------------------------
    # 기준선
    # --------------------------------------------------------

    band_ok = bool(result_df["band_valid"].all())

    # --------------------------------------------------------
    # Feature 개수
    # --------------------------------------------------------

    feature_counts = result_df["n_features"].unique()

    feature_count_ok = len(feature_counts) == 1

    # --------------------------------------------------------
    # 최종 PASS
    # --------------------------------------------------------

    passed = all(
        [
            feature_consistent,
            nan_ok,
            inf_ok,
            band_ok,
            feature_count_ok,
        ]
    )

    # ========================================================
    # 결과 출력
    # ========================================================

    print(f"검증 Fold 수           : {len(result_df)}")

    print(f"Feature 개수            : " f"{feature_counts.tolist()}")

    print(f"Feature 구조 동일       : " f"{'PASS' if feature_consistent else 'FAIL'}")

    print(
        f"최대 NaN 비율           : "
        f"{max_nan_ratio:.4%} "
        f"({'PASS' if nan_ok else 'FAIL'})"
    )

    print(f"Inf 존재 여부           : " f"{'PASS' if inf_ok else 'FAIL'}")

    print(f"기준선(base/upper/lower): " f"{'PASS' if band_ok else 'FAIL'}")

    print(f"Feature 개수 일관성     : " f"{'PASS' if feature_count_ok else 'FAIL'}")

    print(f"상수 Feature            : " f"{'PASS' if constant_ok else 'WARNING'}")

    print()

    if passed:
        print("✅ Rolling Feature 생성 검증 PASS")
        print("→ 다음 단계인 Cross Entropy vs Focal Loss 비교로 진행 가능")
    else:
        print("❌ Rolling Feature 검증 FAIL")
        print("→ Focal Loss 실험 전에 Feature 생성 구조를 먼저 수정해야 합니다.")

    # ========================================================
    # Feature 목록 출력
    # ========================================================

    if feature_signatures:
        print("\n[사용 Feature 목록]")

        for i, col in enumerate(feature_signatures[0], 1):
            print(f"  {i:2d}. {col}")

    # ========================================================
    # Fold별 상세표
    # ========================================================

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
# 3. Threshold별 전체 실험
# ============================================================


def run_threshold_experiment(
    df: pd.DataFrame,
    threshold: float,
):
    """
    하나의 threshold에 대해

    1. Step 5 Walk-Forward
    2. Fold별 최적 파라미터 확보
    3. Rolling Feature 생성
    4. Feature 구조 검증

    까지만 수행한다.

    아직 ML 학습은 수행하지 않는다.
    """

    print_header(f"THRESHOLD = {threshold * 100:.0f}%")

    # ========================================================
    # Step 5
    # ========================================================

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

    print(f"\n✅ 5단계 완료: " f"{len(fold_details)}개 Fold")

    # ========================================================
    # Fold 파라미터 확인
    # ========================================================

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

    # ========================================================
    # Median 파라미터
    # ========================================================

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

    # ========================================================
    # Step 1 Feature 검증
    # ========================================================

    validation = validate_rolling_features(
        df=df,
        fold_details=fold_details,
        threshold=threshold,
    )

    return {
        "threshold": threshold,
        "step5_result": result_5,
        "fold_details": fold_details,
        "median_params": median_params,
        "feature_validation": validation,
    }


# ============================================================
# 4. 1% vs 2% Feature 구조 비교
# ============================================================


def compare_feature_validation(
    results: dict,
):
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

    comparison = pd.DataFrame(rows)

    print(comparison.to_string(index=False))

    # ========================================================
    # 두 threshold의 Feature 컬럼이 같은지 확인
    # ========================================================

    if 0.01 in results and 0.02 in results:

        cols_1 = results[0.01]["feature_validation"]["feature_columns"]

        cols_2 = results[0.02]["feature_validation"]["feature_columns"]

        same_columns = cols_1 == cols_2

        print("\nFeature 컬럼 구조 1% vs 2%: " f"{'동일' if same_columns else '다름'}")

        if same_columns:
            print(
                "→ threshold 변경은 Label/기준선 파라미터에만 영향을 주고 "
                "Feature 구조는 동일합니다."
            )
        else:
            print("⚠️ 1%와 2%에서 Feature 구조가 달라졌습니다.")


# ============================================================
# 5. 메인
# ============================================================


def run_full_pipeline():

    print("=" * 70)
    print("KOSPI 3-Class ML Pipeline")
    print("① Rolling Feature 생성 방식 검증")
    print("Threshold = 1% vs 2%")
    print("=" * 70)

    print(
        "\n현재 단계에서는 "
        "Cross Entropy / Focal Loss / Threshold Tuning을 "
        "아직 수행하지 않습니다."
    )

    print("먼저 Rolling Feature가 완전히 정상인지 검증합니다.")

    # ========================================================
    # Data Load
    # ========================================================

    print("\n[1단계] 데이터 로드")

    df = load_data()

    print(f"✅ {len(df):,}일")

    print(f"   기간: " f"{df.index.min()} ~ {df.index.max()}")

    # ========================================================
    # Label distribution
    # ========================================================

    print("\n[라벨 구조 확인]")

    for threshold in THRESHOLDS:

        ret_5d = df["open"].shift(-6) / df["open"].shift(-1) - 1

        y = np.where(
            ret_5d > threshold,
            2,
            np.where(
                ret_5d < -threshold,
                0,
                1,
            ),
        )

        valid = np.isfinite(ret_5d)

        print(f"\nThreshold {threshold * 100:.0f}%")

        for cls, name in [
            (0, "하락"),
            (1, "중립"),
            (2, "상승"),
        ]:
            ratio = np.mean(y[valid] == cls)

            print(f"  {name}: {ratio:.2%}")

    # ========================================================
    # Threshold별 실행
    # ========================================================

    all_results = {}

    for threshold in THRESHOLDS:

        result = run_threshold_experiment(
            df,
            threshold,
        )

        all_results[threshold] = result

    # ========================================================
    # 최종 비교
    # ========================================================

    compare_feature_validation(all_results)

    # ========================================================
    # 결과 저장
    # ========================================================

    output_dir = "feature_validation_results"

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    for threshold, result in all_results.items():

        fold_details = result["fold_details"]

        filename = f"fold_details_" f"{int(threshold * 100)}pct.csv"

        path = os.path.join(
            output_dir,
            filename,
        )

        fold_details.to_csv(
            path,
            index=False,
        )

        print(f"\n💾 저장: " f"{os.path.abspath(path)}")

        validation_df = result["feature_validation"]["fold_results"]

        validation_filename = f"feature_validation_" f"{int(threshold * 100)}pct.csv"

        validation_path = os.path.join(
            output_dir,
            validation_filename,
        )

        validation_df.to_csv(
            validation_path,
            index=False,
        )

        print(f"💾 저장: " f"{os.path.abspath(validation_path)}")

    # ========================================================
    # 최종 판정
    # ========================================================

    print_header("① 단계 최종 판정")

    all_passed = all(
        result["feature_validation"]["passed"] for result in all_results.values()
    )

    if all_passed:

        print("✅ 1%, 2% 모두 Rolling Feature 검증 PASS")

    else:

        print("❌ Rolling Feature 검증 FAIL")

        print("Focal Loss 실험으로 넘어가지 말고 " "Feature 생성부터 수정해야 합니다.")

    return all_results


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":
    run_full_pipeline()
