"""Logistic Regression 기준 모델의 동작을 확인하는 테스트."""

# 숫자 배열과 테스트 계산에 사용할 numpy를 가져옵니다.
import numpy as np

# 앞에서 만든 Logistic Regression 모델 생성 함수를 가져옵니다.
from models.logistic import build_logistic_baseline


def test_로지스틱_모델이_3개_클래스의_확률을_반환한다():
    """상승·중립·하락의 예측 확률이 정상적으로 나오는지 확인합니다."""

    # 세 클래스가 구분되도록 간단한 학습 데이터를 만듭니다.
    #
    # 각 행은 하루의 데이터라고 생각할 수 있습니다.
    # 각 열은 수익률, 이동평균 이격도 같은 피처라고 생각할 수 있습니다.
    x_train = np.array(
        [
            [-3.0, -2.5],
            [-2.0, -1.8],
            [-1.5, -1.0],
            [-0.2, 0.1],
            [0.0, 0.0],
            [0.2, -0.1],
            [1.5, 1.0],
            [2.0, 1.8],
            [3.0, 2.5],
        ],
        dtype=float,
    )

    # 각 학습 데이터에 해당하는 정답을 만듭니다.
    #
    # -1은 하락, 0은 중립, 1은 상승을 의미합니다.
    y_train = np.array(
        [-1, -1, -1, 0, 0, 0, 1, 1, 1],
        dtype=int,
    )

    # 학습에 사용하지 않을 검증용 데이터를 별도로 만듭니다.
    x_valid = np.array(
        [
            [-2.5, -2.0],
            [0.0, 0.1],
            [2.5, 2.0],
        ],
        dtype=float,
    )

    # Logistic Regression Pipeline을 생성합니다.
    model = build_logistic_baseline()

    # 학습 데이터와 정답만 사용하여 모델을 학습합니다.
    model.fit(x_train, y_train)

    # 검증 데이터의 상승·중립·하락 클래스를 예측합니다.
    predictions = model.predict(x_valid)

    # 검증 데이터가 각 클래스에 속할 확률을 계산합니다.
    probabilities = model.predict_proba(x_valid)

    # 검증 데이터가 3행이므로 예측 결과도 3개여야 합니다.
    assert predictions.shape == (3,)

    # 검증 데이터 3행마다 3개 클래스의 확률이 나와야 합니다.
    assert probabilities.shape == (3, 3)

    # 한 행에서 상승·중립·하락 확률의 합은 1이어야 합니다.
    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        np.ones(3),
    )

    # Pipeline 안에서 학습된 Logistic Regression 모델을 가져옵니다.
    classifier = model.named_steps["classifier"]

    # 모델이 학습한 클래스가 -1, 0, 1인지 확인합니다.
    np.testing.assert_array_equal(
        classifier.classes_,
        np.array([-1, 0, 1]),
    )


def test_스케일러가_학습_데이터만_사용한다():
    """스케일러가 학습 데이터의 평균을 사용했는지 확인합니다."""

    # 스케일러의 계산 결과를 확인하기 위한 학습 데이터를 만듭니다.
    x_train = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [5.0, 50.0],
            [6.0, 60.0],
        ],
        dtype=float,
    )

    # Logistic Regression 학습을 위해 두 개 클래스를 준비합니다.
    y_train = np.array(
        [-1, -1, -1, 1, 1, 1],
        dtype=int,
    )

    # Logistic Regression Pipeline을 생성합니다.
    model = build_logistic_baseline()

    # 준비한 학습 데이터로 모델을 학습합니다.
    model.fit(x_train, y_train)

    # Pipeline 안에서 학습된 StandardScaler를 가져옵니다.
    scaler = model.named_steps["scaler"]

    # StandardScaler가 저장한 평균값과 실제 학습 데이터 평균을 비교합니다.
    #
    # 두 값이 같으면 스케일러가 전달받은 학습 데이터로 정상 학습된 것입니다.
    np.testing.assert_allclose(
        scaler.mean_,
        x_train.mean(axis=0),
    )
