"""RandomForest·XGBoost·LightGBM 기준 모델의 공통 동작을 확인하는 테스트."""

# 테스트용 숫자 배열을 만들고 비교하기 위해 numpy를 가져옵니다.
import numpy as np

# 앞에서 만든 세 모델 생성 함수를 가져옵니다.
from models.lightgbm import build_lightgbm_baseline
from models.random_forest import build_random_forest_baseline
from models.xgboost import build_xgboost_baseline

# 모든 모델이 같은 조건에서 검사되도록 생성 함수를 한곳에 모읍니다.
MODEL_BUILDERS = (
    ("RandomForest", build_random_forest_baseline),
    ("XGBoost", build_xgboost_baseline),
    ("LightGBM", build_lightgbm_baseline),
)

# 프로젝트에서 사용하는 클래스 순서를 고정합니다.
EXPECTED_CLASSES = np.array([-1, 0, 1], dtype=int)


def _make_training_data() -> tuple[np.ndarray, np.ndarray]:
    """세 모델의 기본 동작을 확인할 가상 금융 데이터를 만듭니다."""

    # 하락 구간을 나타내는 가상 피처 24행을 만듭니다.
    #
    # 첫 번째 열: 과거 5일 수익률
    # 두 번째 열: 현재 가격과 20일 이동평균의 이격률
    downward = np.column_stack(
        (
            np.linspace(-0.050, -0.010, 24),
            np.linspace(-0.040, -0.008, 24),
        )
    )

    # 중립 구간을 나타내는 가상 피처 24행을 만듭니다.
    neutral = np.column_stack(
        (
            np.linspace(-0.005, 0.005, 24),
            np.linspace(-0.004, 0.004, 24),
        )
    )

    # 상승 구간을 나타내는 가상 피처 24행을 만듭니다.
    upward = np.column_stack(
        (
            np.linspace(0.010, 0.050, 24),
            np.linspace(0.008, 0.040, 24),
        )
    )

    # 하락·중립·상승 데이터를 하나의 학습 입력 X로 합칩니다.
    x_train = np.vstack((downward, neutral, upward))

    # 각 입력 행에 대응하는 가상 정답 y를 만듭니다.
    y_train = np.repeat(EXPECTED_CLASSES, 24)

    # 가상 학습 데이터와 정답을 반환합니다.
    return x_train, y_train


def _make_validation_data() -> np.ndarray:
    """모델이 학습하지 않은 가상 검증 데이터를 만듭니다."""

    # 음수 구간, 0 근처, 양수 구간의 검증 데이터를 한 행씩 만듭니다.
    return np.array(
        [
            [-0.030, -0.020],
            [0.000, 0.000],
            [0.030, 0.020],
        ],
        dtype=float,
    )


def test_트리_기준_모델들이_3개_클래스의_확률을_반환한다():
    """세 모델이 동일한 3분류 입력·출력 규칙을 지키는지 확인합니다."""

    # 모든 모델에 똑같이 전달할 가상 데이터를 만듭니다.
    x_train, y_train = _make_training_data()
    x_valid = _make_validation_data()

    # RandomForest·XGBoost·LightGBM을 같은 조건으로 검사합니다.
    for model_name, builder in MODEL_BUILDERS:
        # 아직 학습되지 않은 기준 모델을 생성합니다.
        model = builder()

        # 학습 데이터와 정답만 사용해 모델을 학습합니다.
        model.fit(x_train, y_train)

        # 검증 데이터의 방향과 클래스별 확률을 계산합니다.
        predictions = model.predict(x_valid)
        probabilities = model.predict_proba(x_valid)

        # 검증 데이터가 3행이므로 예측 결과도 3개여야 합니다.
        assert predictions.shape == (3,), model_name

        # 검증 데이터마다 하락·중립·상승 확률이 하나씩 나와야 합니다.
        assert probabilities.shape == (3, 3), model_name

        # 각 행의 하락·중립·상승 확률 합은 1이어야 합니다.
        np.testing.assert_allclose(
            probabilities.sum(axis=1),
            np.ones(3),
            err_msg=model_name,
        )

        # 모든 모델이 프로젝트 라벨 -1, 0, 1을 같은 순서로 사용해야 합니다.
        np.testing.assert_array_equal(
            model.classes_,
            EXPECTED_CLASSES,
            err_msg=model_name,
        )


def test_트리_기준_모델들이_같은_난수값에서_같은_결과를_반환한다():
    """같은 random_state를 사용하면 모델 결과가 재현되는지 확인합니다."""

    # 두 모델에 똑같이 전달할 가상 데이터를 만듭니다.
    x_train, y_train = _make_training_data()
    x_valid = _make_validation_data()

    # 세 모델의 재현성을 같은 방식으로 검사합니다.
    for model_name, builder in MODEL_BUILDERS:
        # 같은 기본 random_state를 사용하는 모델 두 개를 만듭니다.
        first_model = builder()
        second_model = builder()

        # 두 모델을 완전히 같은 데이터로 학습합니다.
        first_model.fit(x_train, y_train)
        second_model.fit(x_train, y_train)

        # 두 모델의 예측 결과가 같은지 확인합니다.
        np.testing.assert_array_equal(
            first_model.predict(x_valid),
            second_model.predict(x_valid),
            err_msg=model_name,
        )

        # 두 모델의 클래스별 확률도 같은지 확인합니다.
        np.testing.assert_allclose(
            first_model.predict_proba(x_valid),
            second_model.predict_proba(x_valid),
            err_msg=model_name,
        )


def test_트리_기준_모델들이_balanced_가중치를_선택적으로_받는다():
    """기존 기본값은 유지하고 balanced를 선택했을 때만 적용하는지 확인합니다."""

    for model_name, builder in MODEL_BUILDERS:
        default_model = builder()
        balanced_model = builder(class_weight="balanced")

        assert default_model.class_weight is None, model_name
        assert balanced_model.class_weight == "balanced", model_name


def test_balanced_트리_기준_모델들이_불균형_데이터를_학습한다():
    """XGBoost의 표본 가중치 변환까지 포함해 balanced 학습 경로를 검사합니다."""

    x_train, y_train = _make_training_data()
    x_valid = _make_validation_data()

    # 중립 클래스만 더 반복해 의도적으로 불균형한 학습 폴드를 만듭니다.
    neutral_rows = x_train[y_train == 0]
    x_imbalanced = np.vstack((x_train, neutral_rows, neutral_rows))
    y_imbalanced = np.concatenate((y_train, np.zeros(len(neutral_rows) * 2, dtype=int)))

    for model_name, builder in MODEL_BUILDERS:
        model = builder(class_weight="balanced")
        model.fit(x_imbalanced, y_imbalanced)

        assert model.predict(x_valid).shape == (3,), model_name
        assert model.predict_proba(x_valid).shape == (3, 3), model_name
