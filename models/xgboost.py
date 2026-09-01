"""XGBoost 기준 모델을 생성하는 모듈."""

# 숫자 배열과 라벨 변환에 사용할 numpy를 가져옵니다.
import numpy as np

# sklearn 모델 규칙을 따르는 사용자 정의 분류기를 만들기 위한 클래스를 가져옵니다.
from sklearn.base import BaseEstimator, ClassifierMixin

# predict() 전에 모델이 학습됐는지 확인하는 함수를 가져옵니다.
from sklearn.utils.validation import check_is_fitted

# XGBoost의 sklearn 호환 분류 모델을 가져옵니다.
from xgboost import XGBClassifier

# 프로젝트에서 사용하는 방향 라벨을 고정합니다.
DIRECTION_CLASSES = np.array([-1, 0, 1], dtype=int)

# 같은 데이터로 실행했을 때 같은 결과가 나오도록 난수값을 고정합니다.
DEFAULT_RANDOM_STATE = 42

# 순차적으로 학습할 트리의 개수를 설정합니다.
DEFAULT_N_ESTIMATORS = 100

# 각 결정트리의 최대 깊이를 설정합니다.
DEFAULT_MAX_DEPTH = 3

# 한 번에 모델이 학습하는 크기를 조절합니다.
DEFAULT_LEARNING_RATE = 0.05

# 사용할 수 있는 CPU 코어를 모두 사용합니다.
DEFAULT_N_JOBS = -1


class DirectionXGBClassifier(ClassifierMixin, BaseEstimator):
    """프로젝트 방향 라벨을 지원하는 XGBoost 분류기입니다.

    프로젝트는 하락=-1, 중립=0, 상승=1을 사용합니다.
    XGBoost 내부에서는 이를 0, 1, 2로 바꿔 학습하고,
    예측할 때 다시 -1, 0, 1로 되돌립니다.
    """

    def __init__(
        self,
        *,
        random_state: int = DEFAULT_RANDOM_STATE,
        n_estimators: int = DEFAULT_N_ESTIMATORS,
        max_depth: int = DEFAULT_MAX_DEPTH,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        n_jobs: int = DEFAULT_N_JOBS,
    ) -> None:
        """XGBoost 생성에 필요한 설정값을 저장합니다."""

        # sklearn의 clone()이 모델을 복제할 수 있도록 설정값을 그대로 저장합니다.
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_jobs = n_jobs

    def fit(self, x, y):
        """학습 데이터만 사용해 XGBoost 모델을 학습합니다."""

        # 전달받은 정답을 정수형 numpy 배열로 변환합니다.
        y_array = np.asarray(y, dtype=int)

        # 실제 정답에 들어 있는 클래스 종류를 확인합니다.
        actual_classes = np.unique(y_array)

        # 프로젝트가 정한 세 클래스가 모두 들어오지 않으면 즉시 알려줍니다.
        if not np.array_equal(actual_classes, DIRECTION_CLASSES):
            raise ValueError(
                "y에는 하락=-1, 중립=0, 상승=1의 세 클래스가 모두 필요합니다. "
                f"현재 클래스: {actual_classes.tolist()}"
            )

        # 외부에서 확인할 클래스 순서를 -1, 0, 1로 고정합니다.
        self.classes_ = DIRECTION_CLASSES.copy()

        # -1, 0, 1을 XGBoost가 사용하는 0, 1, 2로 변환합니다.
        encoded_y = np.searchsorted(self.classes_, y_array)

        # XGBoost 내부 분류 모델을 생성합니다.
        self.classifier_ = XGBClassifier(
            # 클래스별 확률을 출력하는 다중분류 목적함수를 사용합니다.
            objective="multi:softprob",

            # 하락·중립·상승의 세 클래스를 사용합니다.
            num_class=3,

            # 순차적으로 학습할 결정트리 개수를 설정합니다.
            n_estimators=self.n_estimators,

            # 하나의 결정트리가 지나치게 복잡해지지 않도록 깊이를 제한합니다.
            max_depth=self.max_depth,

            # 각 트리가 이전 오류를 보완하는 정도를 설정합니다.
            learning_rate=self.learning_rate,

            # 다중분류의 로그 손실을 평가 기준으로 사용합니다.
            eval_metric="mlogloss",

            # 같은 데이터에서 같은 결과가 나오도록 난수값을 고정합니다.
            random_state=self.random_state,

            # 로컬 CPU 코어를 활용해 학습합니다.
            n_jobs=self.n_jobs,

            # CPU에서 효율적으로 학습할 수 있는 히스토그램 방식을 사용합니다.
            tree_method="hist",

            # 불필요한 학습 메시지를 표시하지 않습니다.
            verbosity=0,
        )

        # XGBoost에는 변환된 학습 정답만 전달합니다.
        self.classifier_.fit(x, encoded_y)

        # sklearn 규칙에 맞게 학습이 끝난 자기 자신을 반환합니다.
        return self

    def predict(self, x) -> np.ndarray:
        """입력 데이터의 하락·중립·상승 방향을 예측합니다."""

        # fit()을 실행하지 않고 predict()를 호출하면 명확한 예외를 발생시킵니다.
        check_is_fitted(self, attributes=["classifier_", "classes_"])

        # XGBoost가 예측한 0, 1, 2 값을 정수 배열로 변환합니다.
        encoded_predictions = np.asarray(
            self.classifier_.predict(x),
            dtype=int,
        )

        # 내부 라벨 0, 1, 2를 프로젝트 라벨 -1, 0, 1로 되돌립니다.
        return self.classes_[encoded_predictions]

    def predict_proba(self, x) -> np.ndarray:
        """하락·중립·상승 클래스별 확률을 반환합니다."""

        # fit()을 실행하지 않고 확률을 요청하면 명확한 예외를 발생시킵니다.
        check_is_fitted(self, attributes=["classifier_", "classes_"])

        # 확률 열의 순서는 classes_와 같은 하락·중립·상승 순서입니다.
        return np.asarray(
            self.classifier_.predict_proba(x),
            dtype=float,
        )


def build_xgboost_baseline(
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    max_depth: int = DEFAULT_MAX_DEPTH,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    n_jobs: int = DEFAULT_N_JOBS,
) -> DirectionXGBClassifier:
    """3분류 방향 예측용 XGBoost 모델을 만듭니다.

    이 함수는 데이터를 직접 분할하지 않습니다.
    워크포워드에서 전달받은 학습 데이터만 fit()에 넣어야 합니다.
    """

    # 프로젝트 라벨 변환을 포함한 XGBoost 분류기를 생성합니다.
    model = DirectionXGBClassifier(
        random_state=random_state,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_jobs=n_jobs,
    )

    # 아직 학습되지 않은 XGBoost 모델을 반환합니다.
    return model
