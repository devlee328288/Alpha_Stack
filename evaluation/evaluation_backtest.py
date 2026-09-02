# 백테스트 평가 지표

from typing import Any, Dict, List, Optional

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    recall_score,
)

# ============================================================
# 백테스트 / AI 모델 성과평가 지표 계산 모듈
# ============================================================
#
# 이 파일의 목적
# ------------------------------------------------------------
# 회귀 모델과 분류 모델의 예측 결과를 받아 성과평가 지표를
# 계산하기 위한 함수들을 모아놓은 모듈이다.
#
# 현재 __main__ 부분에서는 가상의 데이터를 사용하지만,
# 실제 프로젝트에서는 해당 부분을 실제 모델의 예측 결과 또는
# 백테스트 결과로 교체하여 사용할 수 있다.
#
#
# [전체 구조]
#
# 1. 회귀 지표
#    - MAE
#    - RMSE
#    - Pearson IC
#    - Rank IC
#    - ICIR
#
# 2. 분류 지표
#    - Class Distribution
#    - Confusion Matrix
#    - Balanced Accuracy
#    - Multiclass MCC
#    - Macro Average Precision
#
# 3. 통합 계산 함수
#    - calculate_all_regression_metrics()
#    - calculate_all_classification_metrics()
#
# 4. 실행 예시
#    - 현재는 가상의 데이터 사용
#    - 실제 프로젝트에서는 이 부분의 데이터를 실제 결과로 교체
#
#
# [중요]
# ------------------------------------------------------------
# 아래 함수들은 "데이터를 어디에서 가져오는가"와
# "성과지표를 어떻게 계산하는가"를 분리해서 설계하였다.
#
# 따라서 실제 데이터가
#
#     CSV
#     → pandas DataFrame
#     → API
#     → ML/DL 모델 출력
#     → 백테스트 엔진
#
# 어디에서 들어오더라도 최종적으로 필요한 형태로 변환한 뒤
# 함수에 전달하면 동일하게 사용할 수 있다.
#
# 예:
#
# predictions = 실제 모델의 예측값
# returns     = 실제 수익률
#
# results = calculate_all_regression_metrics(
#     predictions,
#     returns
# )
#
# 처럼 사용할 수 있다.
#
#
# [주의]
# ------------------------------------------------------------
# 이 모듈의 Pearson IC / Rank IC 함수는
# "하나의 기간에서 계산하는 상관계수"를 계산한다.
#
# 전형적인 퀀트의 Cross-sectional IC는
#
#     특정 날짜의 여러 종목 예측값
#     vs
#     같은 날짜의 여러 종목 미래수익률
#
# 을 비교하여 날짜별 IC를 만든다.
#
# 따라서 여러 날짜의 Cross-sectional IC를 계산하려면
# 날짜별 IC를 먼저 만든 후 그 결과를 ic_series로 모아서
# ICIR을 계산하는 방식으로 사용하는 것이 적절하다.
#
# 반대로 KOSPI처럼 하나의 자산에 대한 시계열 예측에서는
# 시간에 따른 Pearson / Spearman 상관계수로 사용할 수 있지만,
# 이것은 Cross-sectional IC와는 개념적으로 다르다.
# ============================================================


# ============================================================
# 0. 입력값 검증 함수
# ============================================================


def _validate_same_length(
    array_a: np.ndarray,
    array_b: np.ndarray,
    name_a: str,
    name_b: str,
) -> None:
    """
    두 배열의 길이가 동일한지 확인한다.

    실제 모델 결과를 연결할 때
    예측값과 실제값의 행 수가 서로 다르면
    지표 계산 자체가 잘못될 수 있으므로
    계산 전에 명시적으로 검증한다.
    """

    if len(array_a) != len(array_b):
        raise ValueError(
            f"{name_a}와 {name_b}의 길이가 같아야 합니다. "
            f"현재 {name_a}={len(array_a)}, {name_b}={len(array_b)}"
        )


def _validate_non_empty(
    array: np.ndarray,
    name: str,
) -> None:
    """
    입력 배열이 비어 있는지 확인한다.
    """

    if len(array) == 0:
        raise ValueError(f"{name}는 비어 있을 수 없습니다.")


# ============================================================
# 1. 회귀 지표 (Regression Metrics)
# ============================================================


def mean_absolute_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    MAE (Mean Absolute Error)
    평균 절대 오차

    ------------------------------------------------------------
    [의미]
    ------------------------------------------------------------
    실제값과 예측값의 차이를 절댓값으로 만든 후 평균한다.

    예측값이 실제값에서 평균적으로 얼마나 떨어져 있는지를
    직관적으로 확인할 수 있다.

    값의 단위가 원래 데이터와 동일하다는 장점이 있다.

    예를 들어 실제 수익률과 예측 수익률을 입력하면
    MAE 역시 수익률 단위로 해석할 수 있다.


    [수식]

        MAE = (1/N) × Σ |y_i - ŷ_i|


    [입력]

    y_true
        실제값

    y_pred
        모델 예측값


    [출력]

    float
        평균 절대 오차
    ------------------------------------------------------------
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    _validate_same_length(y_true, y_pred, "y_true", "y_pred")
    _validate_non_empty(y_true, "y_true")

    # NaN이나 무한대가 포함된 경우 계산 결과가 왜곡될 수 있다.
    # 따라서 유효한 값만 사용한다.
    valid = np.isfinite(y_true) & np.isfinite(y_pred)

    if np.sum(valid) == 0:
        return np.nan

    return np.mean(np.abs(y_true[valid] - y_pred[valid]))


def root_mean_squared_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    RMSE (Root Mean Squared Error)
    평균 제곱근 오차

    ------------------------------------------------------------
    [의미]
    ------------------------------------------------------------
    실제값과 예측값의 차이를 제곱한 후 평균하고,
    마지막으로 제곱근을 취한다.

    큰 오차에 더 큰 패널티를 주기 때문에
    큰 예측 오류가 존재하는지를 확인하는 데 유용하다.


    [수식]

        RMSE = sqrt(
            (1/N) × Σ(y_i - ŷ_i)^2
        )


    [MAE와의 차이]

    MAE
        오차의 절댓값을 평균한다.

    RMSE
        오차를 제곱하기 때문에
        큰 오차의 영향이 상대적으로 더 커진다.


    [출력]

    float
        RMSE
    ------------------------------------------------------------
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    _validate_same_length(y_true, y_pred, "y_true", "y_pred")
    _validate_non_empty(y_true, "y_true")

    valid = np.isfinite(y_true) & np.isfinite(y_pred)

    if np.sum(valid) == 0:
        return np.nan

    mse = np.mean((y_true[valid] - y_pred[valid]) ** 2)

    return np.sqrt(mse)


def pearson_information_coefficient(
    predictions: np.ndarray,
    returns: np.ndarray,
) -> float:
    """
    Pearson IC
    Pearson Information Coefficient

    ------------------------------------------------------------
    [의미]
    ------------------------------------------------------------
    예측값과 실제 수익률 사이의
    선형적인 상관관계를 측정한다.

    Pearson 상관계수 자체의 수학적 계산은 일반적인
    Pearson correlation과 동일하다.

    퀀트에서는 특정 시점의 여러 종목을 대상으로
    예측값과 미래수익률의 횡단면 상관관계를 계산하여
    Cross-sectional IC로 사용하는 경우가 많다.


    [수식]

        IC = Cov(P, R) / (σ_P × σ_R)


    [해석]

        +1
        강한 양의 선형관계

         0
        선형적인 관계가 거의 없음

        -1
        강한 음의 선형관계


    [중요]

    이 함수는 "하나의 데이터 묶음에 대한 Pearson 상관계수"
    를 계산한다.

    따라서 여러 날짜에 대한 Cross-sectional IC를 계산하려면

        날짜 1 → IC
        날짜 2 → IC
        날짜 3 → IC
        ...

    형태로 각각 계산한 뒤 IC 시계열을 만들어야 한다.


    [주의]

    예측값과 실제값 중 하나가 모든 값이 동일하면
    상관계수를 정의할 수 없으므로 NaN을 반환한다.
    ------------------------------------------------------------
    """

    predictions = np.asarray(predictions, dtype=float)
    returns = np.asarray(returns, dtype=float)

    _validate_same_length(
        predictions,
        returns,
        "predictions",
        "returns",
    )
    _validate_non_empty(predictions, "predictions")

    # NaN / inf 제거
    valid = np.isfinite(predictions) & np.isfinite(returns)

    predictions = predictions[valid]
    returns = returns[valid]

    if len(predictions) < 2:
        return np.nan

    # 상관계수는 두 변수의 분산이 모두 존재해야 계산 가능하다.
    if np.std(predictions) == 0 or np.std(returns) == 0:
        return np.nan

    corr, _ = pearsonr(predictions, returns)

    return float(corr)


def rank_information_coefficient(
    predictions: np.ndarray,
    returns: np.ndarray,
) -> float:
    """
    Rank IC
    Spearman Rank Information Coefficient

    ------------------------------------------------------------
    [의미]
    ------------------------------------------------------------
    예측값의 순위와 실제 수익률의 순위가
    얼마나 비슷한지를 측정한다.

    Pearson IC가 값 자체의 선형적인 관계를 보는 반면,
    Rank IC는 값의 절대적인 크기보다
    순서 관계에 집중한다.

    퀀트에서는 여러 종목의

        예측 순위
        vs
        실제 미래수익률 순위

    를 비교하는 데 많이 사용된다.


    [수식]

    일반적으로 Spearman rank correlation을 사용한다.


    [해석]

        +1
        순위가 완전히 일치

         0
        순위 관계가 거의 없음

        -1
        순위가 완전히 반대


    [주의]

    이 함수 역시 하나의 데이터 묶음에 대한
    Spearman 상관계수를 계산한다.

    따라서 전형적인 Cross-sectional Rank IC를 사용하려면
    날짜별로 Rank IC를 계산하여 시계열로 관리하는 것이 좋다.
    ------------------------------------------------------------
    """

    predictions = np.asarray(predictions, dtype=float)
    returns = np.asarray(returns, dtype=float)

    _validate_same_length(
        predictions,
        returns,
        "predictions",
        "returns",
    )
    _validate_non_empty(predictions, "predictions")

    valid = np.isfinite(predictions) & np.isfinite(returns)

    predictions = predictions[valid]
    returns = returns[valid]

    if len(predictions) < 2:
        return np.nan

    # 모든 예측값이 동일한 경우
    # 순위 상관관계를 정의할 수 없다.
    if np.std(predictions) == 0 or np.std(returns) == 0:
        return np.nan

    corr, _ = spearmanr(predictions, returns)

    return float(corr)


def information_coefficient_ratio(
    ic_series: np.ndarray,
) -> float:
    """
    ICIR
    Information Coefficient Information Ratio

    ------------------------------------------------------------
    [의미]
    ------------------------------------------------------------
    여러 기간에 걸쳐 계산된 IC의 평균이
    IC 변동성에 비해 얼마나 큰지를 나타낸다.

    즉, 특정 한 시점에서 IC가 높은 것뿐만 아니라
    여러 기간에 걸쳐 일관되게 유지되는지를 평가한다.


    [수식]

        ICIR = Mean(IC) / Std(IC)


    여기서는 표본 표준편차(ddof=1)를 사용한다.


    [입력 예]

        [0.04, 0.06, 0.02, 0.08, 0.05, ...]


    실제 프로젝트에서는

        2025-01 → IC
        2025-02 → IC
        2025-03 → IC
        ...

    형태의 기간별 IC를 입력한다.


    [주의]

    ICIR은 단일 IC 값으로 계산할 수 없다.

    또한 ICIR 자체를 연환산하려면
    데이터의 주기와 사용 목적에 맞게
    추가적인 연환산 계수를 적용해야 한다.

    예를 들어 월별 ICIR을 연환산하는 경우
    일반적으로 sqrt(12)를 고려할 수 있다.
    다만 이는 IC의 독립성 및 사용 convention에 따라
    해석에 주의해야 한다.
    ------------------------------------------------------------
    """

    ic_series = np.asarray(ic_series, dtype=float)

    _validate_non_empty(ic_series, "ic_series")

    # NaN / inf 제거
    ic_series = ic_series[np.isfinite(ic_series)]

    if len(ic_series) < 2:
        return np.nan

    mean_ic = np.mean(ic_series)

    # 표본 표준편차
    std_ic = np.std(ic_series, ddof=1)

    if std_ic == 0:
        return np.nan

    return float(mean_ic / std_ic)


# ============================================================
# 2. 분류 지표
# ============================================================
#
# 다중 클래스 분류를 기본으로 한다.
#
# 예:
#
#   0 = 상승
#   1 = 하락
#   2 = 보합
#
# 실제 프로젝트에서는
#
#   y_true
#       실제 시장 상태
#
#   y_pred
#       AI가 예측한 시장 상태
#
#   y_pred_proba
#       AI가 출력한 각 클래스별 확률
#
# 을 입력한다.
#
# 주의:
# 숫자 0, 1, 2가 상승/하락/보합이라는 것은
# 코드의 필수 규칙이 아니다.
#
# 실제 프로젝트에서 정한 클래스 의미와
# labels 및 확률 열의 순서를 일관되게 유지해야 한다.
# ============================================================


def class_distribution(
    y_true: np.ndarray,
    labels: Optional[List] = None,
) -> Dict[Any, float]:
    """
    Class Distribution
    클래스 분포

    ------------------------------------------------------------
    [의미]
    ------------------------------------------------------------
    실제 데이터에서 각 클래스가 차지하는 비율을 계산한다.

    금융시장 방향성 분류에서는
    상승 / 하락 / 보합 데이터가 얼마나 불균형한지
    확인하는 데 사용할 수 있다.


    [수식]

        클래스 비율_k
        = 클래스 k의 샘플 수 / 전체 샘플 수


    [주의]

    이 함수는 "실제 클래스 분포"를 계산한다.

    모델의 예측 클래스 분포가 필요하다면
    y_pred를 입력하여 별도로 계산하면 된다.
    ------------------------------------------------------------
    """

    y_true = np.asarray(y_true)

    _validate_non_empty(y_true, "y_true")

    if labels is None:
        labels = np.unique(y_true)

    total = len(y_true)

    return {label: float(np.sum(y_true == label) / total) for label in labels}


def confusion_matrix_multiclass(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List] = None,
) -> np.ndarray:
    """
    Multiclass Confusion Matrix
    다중 클래스 혼동 행렬

    ------------------------------------------------------------
    [구조]

        행(row)
            실제 클래스

        열(column)
            예측 클래스


    예:

                    예측
                상승  하락  보합
        실제
        상승
        하락
        보합


    [활용]

    어떤 클래스를 어떤 클래스로 잘못 예측하는지
    구체적으로 확인할 수 있다.
    ------------------------------------------------------------
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    _validate_same_length(
        y_true,
        y_pred,
        "y_true",
        "y_pred",
    )
    _validate_non_empty(y_true, "y_true")

    if labels is None:
        labels = np.unique(y_true)

    return confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )


def balanced_accuracy_multiclass(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List] = None,
) -> float:
    """
    Balanced Accuracy
    균형 정확도

    ------------------------------------------------------------
    [의미]
    ------------------------------------------------------------
    각 클래스의 Recall을 계산한 후
    그 평균을 구한다.


    [수식]

        BA = (1/K) × Σ Recall_k


    각 클래스 Recall:

        Recall_k
        = TP_k / (TP_k + FN_k)


    [장점]

    클래스별 데이터 개수가 크게 다르더라도
    각 클래스를 동일한 비중으로 평가할 수 있다.


    [주의]

    Balanced Accuracy의 무작위 기준은 항상 0.5가 아니다.

    K개의 클래스를 균등하게 무작위 예측하는 경우
    일반적으로 기대되는 Balanced Accuracy는

        1 / K

    이다.

    따라서 3개 클래스라면
    단순 무작위 예측의 기준은 약 0.333이다.

    adjusted=True를 사용하는 sklearn의 조정된 Balanced Accuracy는
    무작위 성능을 0으로 맞춘다.
    ------------------------------------------------------------
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    _validate_same_length(
        y_true,
        y_pred,
        "y_true",
        "y_pred",
    )
    _validate_non_empty(y_true, "y_true")

    if labels is None:
        labels = np.unique(y_true)

    recalls = recall_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    return float(np.mean(recalls))


def mcc_multiclass(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Multiclass MCC
    다중 클래스 Matthews Correlation Coefficient

    ------------------------------------------------------------
    [의미]
    ------------------------------------------------------------
    Matthews Correlation Coefficient는
    예측값과 실제값의 상관관계적인 특성을 이용하여
    분류 성능을 평가하는 지표이다.

    클래스 불균형이 있는 분류 문제에서도
    Accuracy와 함께 사용할 수 있는 지표이다.


    [범위]

        -1 ~ +1


        +1
        완벽한 예측

         0
        무작위 예측에 가까운 수준

        -1
        예측과 실제가 강하게 반대되는 경우


    [중요]

    이전 코드에서는

        MCC Macro
        MCC Micro

    를 직접 One-vs-Rest 방식으로 구성했다.

    그러나 일반적인 단일-label 다중 클래스 문제에서는
    이런 방식의 MCC Macro / Micro를
    "표준 Multiclass MCC"라고 부르는 것은 적절하지 않다.

    따라서 여기서는 sklearn이 제공하는
    직접적인 Multiclass MCC를 사용한다.


    [출력]

    float
        Multiclass MCC
    ------------------------------------------------------------
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    _validate_same_length(
        y_true,
        y_pred,
        "y_true",
        "y_pred",
    )
    _validate_non_empty(y_true, "y_true")

    return float(
        matthews_corrcoef(
            y_true,
            y_pred,
        )
    )


def macro_average_precision(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    labels: Optional[List] = None,
) -> float:
    """
    Macro Average Precision
    Macro-averaged Average Precision

    ------------------------------------------------------------
    [의미]
    ------------------------------------------------------------
    각 클래스를 One-vs-Rest 방식의 이진 분류 문제로 보고
    클래스별 Average Precision(AP)을 계산한 후 평균한다.


    [중요]

    이 함수가 사용하는 sklearn의
    average_precision_score는
    일반적인 의미의 "사다리꼴 적분 방식 PR-AUC"와
    동일한 지표가 아니다.

    Average Precision은 Precision-Recall curve에서
    recall 변화량을 이용하여 계산한다.

    따라서 기존 코드의

        PR-AUC Macro

    라는 이름보다는

        Macro Average Precision

    이 더 정확한 명칭이다.


    [장점]

    클래스 불균형이 있는 문제에서
    각 클래스의 Positive 예측 성능을 평가하는 데
    유용하게 사용할 수 있다.


    [입력]

    y_true
        실제 클래스

    y_pred_proba
        각 샘플의 클래스별 예측 확률

        shape:

            (샘플 수, 클래스 수)


    labels
        y_pred_proba의 열 순서

        예:

            labels = [0, 1, 2]

            column 0 → 클래스 0 확률
            column 1 → 클래스 1 확률
            column 2 → 클래스 2 확률


    [중요]

    y_pred_proba의 열 순서는
    반드시 labels와 일치해야 한다.
    ------------------------------------------------------------
    """

    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(
        y_pred_proba,
        dtype=float,
    )

    _validate_non_empty(y_true, "y_true")

    if labels is None:
        labels = np.unique(y_true)

    # 확률 배열은 반드시 2차원이어야 한다.
    if y_pred_proba.ndim != 2:
        raise ValueError(
            "y_pred_proba는 " "(샘플 수, 클래스 수) 형태의 2차원 배열이어야 합니다."
        )

    # 샘플 수 확인
    if y_pred_proba.shape[0] != len(y_true):
        raise ValueError("y_true의 샘플 수와 y_pred_proba의 행 수가 같아야 합니다.")

    # 클래스 수 확인
    if y_pred_proba.shape[1] != len(labels):
        raise ValueError("y_pred_proba의 열 개수는 labels의 개수와 같아야 합니다.")

    # 확률에 NaN / inf가 있는 경우 계산 결과가 왜곡될 수 있다.
    if not np.all(np.isfinite(y_pred_proba)):
        raise ValueError("y_pred_proba에 NaN 또는 inf가 포함되어 있습니다.")

    # 일반적인 predict_proba / softmax 출력은
    # 각 행의 확률 합이 1이어야 한다.
    row_sums = np.sum(y_pred_proba, axis=1)

    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError("y_pred_proba의 각 행의 확률 합이 1이어야 합니다.")

    ap_list = []

    for i, cls in enumerate(labels):

        # 현재 클래스를 Positive(1),
        # 나머지 클래스를 Negative(0)으로 변환한다.
        y_true_cls = (y_true == cls).astype(int)

        # 해당 클래스가 실제 데이터에 전혀 존재하지 않는 경우
        # 해당 클래스의 AP는 의미 있게 계산할 수 없다.
        if np.sum(y_true_cls) == 0:
            continue

        # 현재 클래스의 예측 확률
        proba_cls = y_pred_proba[:, i]

        # One-vs-Rest Average Precision 계산
        ap = average_precision_score(
            y_true_cls,
            proba_cls,
        )

        if np.isfinite(ap):
            ap_list.append(ap)

    if not ap_list:
        return np.nan

    return float(np.mean(ap_list))


# ============================================================
# 3. 회귀 지표 통합 계산
# ============================================================


def calculate_all_regression_metrics(
    predictions: np.ndarray,
    returns: np.ndarray,
    ic_series: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    회귀 성과지표를 한 번에 계산한다.

    ------------------------------------------------------------
    [계산 지표]

    1. MAE
       예측값과 실제값의 평균 절대 오차

    2. RMSE
       큰 오차에 민감한 오차 지표

    3. Pearson IC
       예측값과 실제값의 Pearson 상관계수

    4. Rank IC
       예측 순위와 실제값 순위의 Spearman 상관계수

    5. ICIR
       기간별 IC 평균 / IC 표준편차


    [입력]

    predictions
        모델 예측값

    returns
        실제 미래 수익률

    ic_series
        여러 기간에서 계산한 IC 시계열

        예:

        [0.03, 0.05, 0.01, 0.07, ...]


    [중요]

    predictions와 returns가 하나의 긴 시계열이라면
    Pearson IC와 Rank IC는
    "전체 데이터에 대한 상관계수"가 된다.

    표준적인 Cross-sectional IC가 필요한 경우에는
    날짜별로 IC를 먼저 계산하여 ic_series를 구성해야 한다.


    [출력]

    dict
        계산된 모든 지표
    ------------------------------------------------------------
    """

    predictions = np.asarray(
        predictions,
        dtype=float,
    )

    returns = np.asarray(
        returns,
        dtype=float,
    )

    _validate_same_length(
        predictions,
        returns,
        "predictions",
        "returns",
    )

    results = {
        "MAE": mean_absolute_error(
            returns,
            predictions,
        ),
        "RMSE": root_mean_squared_error(
            returns,
            predictions,
        ),
        "Pearson IC": pearson_information_coefficient(
            predictions,
            returns,
        ),
        "Rank IC": rank_information_coefficient(
            predictions,
            returns,
        ),
    }

    # IC 시계열이 제공된 경우에만 ICIR 계산
    if ic_series is not None:

        results["ICIR"] = information_coefficient_ratio(ic_series)

    return results


# ============================================================
# 4. 분류 지표 통합 계산
# ============================================================


def calculate_all_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    labels: Optional[List] = None,
) -> Dict[str, Any]:
    """
    분류 성과지표를 한 번에 계산한다.

    ------------------------------------------------------------
    [기본 입력]

    y_true
        실제 정답 클래스

    y_pred
        모델의 최종 예측 클래스

    y_pred_proba
        모델의 클래스별 예측 확률

    labels
        클래스의 순서


    [예: 상승 / 하락 / 보합]

        0 = 상승
        1 = 하락
        2 = 보합


    [계산 지표]

    1. Confusion Matrix
    2. Balanced Accuracy
    3. Multiclass MCC
    4. Macro Average Precision
    5. Class Distribution


    [중요]

    이전 버전의

        MCC Macro
        MCC Micro

    는 제거하였다.

    단일-label 다중 클래스 분류에서는
    표준 Multiclass MCC를 직접 계산하는 것이
    더 적절하다.


    [PR 관련 지표]

    기존의 "PR-AUC Macro"는
    실제 구현상 sklearn의
    average_precision_score를 사용하고 있었기 때문에

        Macro Average Precision

    으로 명칭을 수정하였다.


    [출력]

    dict 형태

        {
            'Confusion Matrix': ...,
            'Balanced Accuracy': ...,
            'Multiclass MCC': ...,
            'Macro Average Precision': ...,
            'Class Distribution': ...
        }


    y_pred_proba가 제공되지 않으면
    Macro Average Precision은 계산하지 않는다.


    [실제 모델 연결]

        y_pred = model.predict(X_test)

        y_pred_proba = model.predict_proba(X_test)


    이후:

        results = calculate_all_classification_metrics(
            y_true=y_test,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            labels=[0, 1, 2]
        )

    형태로 사용할 수 있다.
    ------------------------------------------------------------
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    _validate_same_length(
        y_true,
        y_pred,
        "y_true",
        "y_pred",
    )
    _validate_non_empty(y_true, "y_true")

    if labels is None:
        labels = np.unique(y_true)

    # labels가 명시된 경우
    # 실제값과 예측값에 존재하는 클래스가
    # labels 안에 포함되어 있는지 확인한다.
    label_set = set(labels)

    unknown_true = set(np.unique(y_true)) - label_set
    unknown_pred = set(np.unique(y_pred)) - label_set

    if unknown_true:
        raise ValueError(
            f"y_true에 labels에 없는 클래스가 있습니다: " f"{unknown_true}"
        )

    if unknown_pred:
        raise ValueError(
            f"y_pred에 labels에 없는 클래스가 있습니다: " f"{unknown_pred}"
        )

    results = {
        # 실제 클래스와 예측 클래스의 관계
        "Confusion Matrix": confusion_matrix_multiclass(
            y_true,
            y_pred,
            labels,
        ),
        # 클래스별 Recall의 평균
        "Balanced Accuracy": balanced_accuracy_multiclass(
            y_true,
            y_pred,
            labels,
        ),
        # 표준 다중 클래스 MCC
        "Multiclass MCC": mcc_multiclass(
            y_true,
            y_pred,
        ),
        # 실제 데이터의 클래스 비율
        "Class Distribution": class_distribution(
            y_true,
            labels,
        ),
    }

    # 확률값이 제공된 경우
    # Macro Average Precision 계산
    if y_pred_proba is not None:

        results["Macro Average Precision"] = macro_average_precision(
            y_true,
            y_pred_proba,
            labels,
        )

    return results


# ============================================================
# 5. 실행 예시
# ============================================================
#
# 현재는 함수가 정상적으로 작동하는지 확인하기 위한
# 가상의 데이터를 사용한다.
#
# 실제 프로젝트에서는 이 부분을
# 실제 모델의 결과로 교체하면 된다.
# ============================================================


if __name__ == "__main__":

    print("성과 지표 계산 예시")
    print("=" * 60)

    # ========================================================
    # A. 회귀 모델 평가
    # ========================================================

    np.random.seed(42)

    n_samples = 100

    # --------------------------------------------------------
    # 실제 수익률을 가정한 가상 데이터
    # --------------------------------------------------------

    true_returns = np.random.normal(
        0.001,
        0.02,
        n_samples,
    )

    # --------------------------------------------------------
    # 가상의 예측값
    # --------------------------------------------------------
    #
    # 주의:
    # 여기서는 함수가 정상적으로 작동하는지 확인하기 위해
    # 실제 수익률에 노이즈를 추가하여 예측값을 만든다.
    #
    # 따라서 이것은 "모델의 실제 성능을 검증하는 백테스트"가
    # 아니라 단순한 함수 실행 예시이다.
    #
    # 실제 모델 평가에서는 반드시
    #
    #     X → 모델 → predictions
    #
    # 과정을 거친 실제 예측값을 사용해야 한다.
    # --------------------------------------------------------

    pred_returns = true_returns + np.random.normal(
        0,
        0.01,
        n_samples,
    )

    # --------------------------------------------------------
    # 기간별 IC 데이터
    # --------------------------------------------------------
    #
    # ICIR은 단일 IC가 아니라
    # 여러 기간의 IC가 필요하다.
    #
    # 현재는 12개의 가상 IC를 생성한다.
    #
    # 실제 사용 시:
    #
    #     ic_series = 실제 기간별 IC
    #
    # 로 교체한다.
    # --------------------------------------------------------

    ic_series = np.random.normal(
        0.05,
        0.03,
        12,
    )

    # --------------------------------------------------------
    # 회귀 지표 계산
    # --------------------------------------------------------

    reg_results = calculate_all_regression_metrics(
        predictions=pred_returns,
        returns=true_returns,
        ic_series=ic_series,
    )

    print("\n[회귀 지표]")
    print("-" * 60)

    for key, value in reg_results.items():
        print(f"{key:15s} : {value:.4f}")

    # ========================================================
    # B. 분류 모델 평가
    # ========================================================
    #
    # 예시:
    #
    # 0 = 상승
    # 1 = 하락
    # 2 = 보합
    #
    # 실제 프로젝트에서는
    #
    # y_true
    #     → 실제 시장 상태
    #
    # y_pred
    #     → 모델 예측 클래스
    #
    # y_proba
    #     → 모델 클래스별 예측 확률
    #
    # 로 교체한다.
    # ========================================================

    # 실제 클래스
    y_true = np.array(
        [
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1,
            2,
        ]
    )

    # 모델 예측 클래스
    y_pred = np.array(
        [
            0,
            1,
            2,
            0,
            0,
            2,
            1,
            1,
            2,
            0,
            1,
            2,
            0,
            1,
            2,
        ]
    )

    # --------------------------------------------------------
    # 모델의 클래스별 예측 확률
    # --------------------------------------------------------
    #
    # labels = [0, 1, 2] 기준
    #
    # column 0 → 클래스 0 확률
    # column 1 → 클래스 1 확률
    # column 2 → 클래스 2 확률
    #
    # 실제 sklearn 모델에서는:
    #
    #     y_proba = model.predict_proba(X_test)
    #
    # 로 얻을 수 있다.
    #
    # 딥러닝 모델에서는 일반적으로
    # softmax 출력값을 사용할 수 있다.
    # --------------------------------------------------------

    y_proba = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.2, 0.6],
            [0.6, 0.3, 0.1],
            [0.4, 0.4, 0.2],
            [0.2, 0.1, 0.7],
            [0.3, 0.5, 0.2],
            [0.1, 0.7, 0.2],
            [0.2, 0.3, 0.5],
            [0.5, 0.3, 0.2],
            [0.2, 0.6, 0.2],
            [0.1, 0.1, 0.8],
            [0.6, 0.2, 0.2],
            [0.1, 0.8, 0.1],
            [0.2, 0.2, 0.6],
        ]
    )

    # --------------------------------------------------------
    # 분류 지표 계산
    # --------------------------------------------------------

    cls_results = calculate_all_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_pred_proba=y_proba,
        labels=[0, 1, 2],
    )

    # --------------------------------------------------------
    # 분류 결과 출력
    # --------------------------------------------------------

    print("\n[분류 지표]")
    print("-" * 60)

    print("Confusion Matrix:")
    print(cls_results["Confusion Matrix"])

    print(f"Balanced Accuracy : " f"{cls_results['Balanced Accuracy']:.4f}")

    print(f"Multiclass MCC    : " f"{cls_results['Multiclass MCC']:.4f}")

    print(f"Macro Average Precision : " f"{cls_results['Macro Average Precision']:.4f}")

    print("Class Distribution:")

    for cls, ratio in cls_results["Class Distribution"].items():

        print(f"  클래스 {cls} : {ratio:.2%}")

    print("=" * 60)

    print("모든 성과 지표 계산 완료.")
