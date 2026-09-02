"""Logistic Regression 기준 모델의 동작을 확인하는 테스트."""

# 숫자 배열과 테스트 계산에 사용할 numpy를 가져옵니다.
import numpy as np

# 앞에서 만든 Logistic Regression 모델 생성 함수를 가져옵니다.
from models.logistic import build_logistic_baseline


def test_로지스틱_모델이_3개_클래스의_확률을_반환한다():
    """상승·중립·하락의 예측 확률이 정상적으로 나오는지 확인합니다."""

    # 테스트에서 사용할 가상의 학습 데이터를 만듭니다.
    #
    # 각 행은 특정 거래일에 확인할 수 있는 피처를 의미합니다.
    #
    # 첫 번째 열: 과거 5일 수익률
    # 두 번째 열: 현재 가격과 20일 이동평균의 이격률
    #
    # 예를 들어 첫 번째 행 [-0.030, -0.025]는 다음을 의미합니다.
    # 과거 5일 수익률은 -3.0%이고,
    # 현재 가격은 20일 이동평균보다 2.5% 아래에 있습니다.
    #
    # 실제 시장에서 가져온 값은 아니며 모델 동작 확인용 가상 데이터입니다.
    x_train = np.array(
        [
            [-0.030, -0.025],
            [-0.020, -0.018],
            [-0.015, -0.010],
            [-0.002, 0.001],
            [0.000, 0.000],
            [0.002, -0.001],
            [0.015, 0.010],
            [0.020, 0.018],
            [0.030, 0.025],
        ],
        dtype=float,
    )

    # 각 학습 행에 대응하는 미래 5거래일 방향을 가상 정답으로 만듭니다.
    #
    # -1: 하락
    #  0: 중립
    #  1: 상승
    #
    # 첫 번째부터 세 번째 행까지는 하락,
    # 네 번째부터 여섯 번째 행까지는 중립,
    # 일곱 번째부터 아홉 번째 행까지는 상승으로 설정했습니다.
    y_train = np.array(
        [-1, -1, -1, 0, 0, 0, 1, 1, 1],
        dtype=int,
    )

    # 모델이 학습하지 않은 가상의 검증 데이터를 만듭니다.
    #
    # 첫 번째 열: 과거 5일 수익률
    # 두 번째 열: 현재 가격과 20일 이동평균의 이격률
    #
    # 첫 번째 행은 음수 구간, 두 번째 행은 0 근처,
    # 세 번째 행은 양수 구간에 해당합니다.
    x_valid = np.array(
        [
            [-0.025, -0.020],
            [0.000, 0.001],
            [0.025, 0.020],
        ],
        dtype=float,
    )

    # StandardScaler와 LogisticRegression이 연결된 모델을 생성합니다.
    model = build_logistic_baseline()

    # 학습 데이터와 정답만 사용해 모델을 학습합니다.
    model.fit(x_train, y_train)

    # 검증 데이터의 하락·중립·상승 클래스를 예측합니다.
    predictions = model.predict(x_valid)

    # 검증 데이터가 각 클래스에 속할 확률을 계산합니다.
    probabilities = model.predict_proba(x_valid)

    # 검증 데이터가 3행이므로 예측 결과도 3개여야 합니다.
    assert predictions.shape == (3,)

    # 검증 데이터 3행마다 3개 클래스의 확률이 나와야 합니다.
    assert probabilities.shape == (3, 3)

    # 한 행에서 하락·중립·상승 확률의 합은 1이어야 합니다.
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
    """스케일러가 검증 데이터가 아닌 학습 데이터의 평균만 사용하는지 확인합니다."""

    # 학습에 사용할 가상의 금융 피처를 만듭니다.
    #
    # 첫 번째 열: 과거 5일 수익률
    # 두 번째 열: 현재 가격과 20일 이동평균의 이격률
    #
    # 모든 값은 소수로 표시합니다.
    # 예: -0.030은 -3.0%, 0.020은 2.0%입니다.
    x_train = np.array(
        [
            [-0.030, -0.025],
            [-0.020, -0.015],
            [-0.002, 0.001],
            [0.002, -0.001],
            [0.020, 0.015],
            [0.030, 0.025],
        ],
        dtype=float,
    )

    # 각 학습 행에 대응하는 미래 5거래일 방향을 만듭니다.
    #
    # -1: 하락
    #  0: 중립
    #  1: 상승
    y_train = np.array(
        [-1, -1, 0, 0, 1, 1],
        dtype=int,
    )

    # 검증 데이터에는 학습 데이터보다 의도적으로 큰 값을 넣습니다.
    #
    # 첫 번째 열: 과거 5일 수익률 50%
    # 두 번째 열: 현재 가격이 20일 이동평균보다 40% 위에 있는 상태
    #
    # 검증 데이터가 스케일러 학습에 잘못 들어가면 평균이 크게 달라지므로,
    # 데이터 누출 여부를 테스트에서 쉽게 확인할 수 있습니다.
    x_valid = np.array(
        [
            [0.500, 0.400],
        ],
        dtype=float,
    )

    # 학습 데이터만 사용한 열별 평균을 미리 계산합니다.
    expected_train_mean = x_train.mean(axis=0)

    # StandardScaler와 LogisticRegression이 연결된 모델을 생성합니다.
    model = build_logistic_baseline()

    # 스케일러와 모델에는 학습 데이터만 전달합니다.
    model.fit(x_train, y_train)

    # 검증 데이터는 모델 학습이 끝난 뒤 예측에만 사용합니다.
    #
    # predict()는 이미 학습된 스케일러로 검증 데이터를 변환할 뿐,
    # 검증 데이터로 스케일러를 다시 학습하지 않아야 합니다.
    model.predict(x_valid)

    # Pipeline 안에서 학습된 StandardScaler를 가져옵니다.
    scaler = model.named_steps["scaler"]

    # 스케일러가 저장한 평균이 학습 데이터의 평균과 같은지 확인합니다.
    #
    # 검증 데이터가 스케일러 학습에 섞였다면 이 검사가 실패합니다.
    np.testing.assert_allclose(
        scaler.mean_,
        expected_train_mean,
    )


def test_로지스틱_모델이_balanced_가중치를_선택적으로_받는다():
    """기존 기본값은 유지하고 balanced를 선택했을 때만 적용하는지 확인합니다."""

    default_model = build_logistic_baseline()
    balanced_model = build_logistic_baseline(class_weight="balanced")

    assert default_model.named_steps["classifier"].class_weight is None
    assert balanced_model.named_steps["classifier"].class_weight == "balanced"
