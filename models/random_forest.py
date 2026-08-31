"""RandomForest 기준 모델을 생성하는 모듈."""

# RandomForest 분류 모델을 가져옵니다.
from sklearn.ensemble import RandomForestClassifier

# 같은 데이터로 실행했을 때 같은 결과가 나오도록 난수값을 고정합니다.
DEFAULT_RANDOM_STATE = 42

# 여러 개의 결정트리를 만들어 평균을 낼 트리 개수를 설정합니다.
DEFAULT_N_ESTIMATORS = 100

# 사용할 수 있는 CPU 코어를 모두 사용합니다.
DEFAULT_N_JOBS = -1


def build_random_forest_baseline(
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    n_jobs: int = DEFAULT_N_JOBS,
) -> RandomForestClassifier:
    """3분류 방향 예측용 RandomForest 모델을 만듭니다.

    이 함수는 학습 데이터와 검증 데이터를 직접 나누지 않습니다.
    워크포워드에서 전달받은 학습 데이터만 fit()에 넣어야 합니다.

    RandomForest는 피처의 크기보다 값을 나누는 기준을 학습하므로
    Logistic Regression과 달리 StandardScaler를 사용하지 않습니다.

    Args:
        random_state:
            같은 데이터로 다시 학습했을 때 결과를 재현하기 위한 난수값입니다.
        n_estimators:
            RandomForest를 구성할 결정트리의 개수입니다.
        n_jobs:
            학습에 사용할 CPU 코어 개수입니다. -1은 모든 코어를 의미합니다.

    Returns:
        아직 학습되지 않은 RandomForestClassifier입니다.
    """

    # 별도의 데이터 분할이나 전처리를 수행하지 않고 모델만 생성합니다.
    model = RandomForestClassifier(
        # 여러 결정트리의 결과를 종합할 수 있도록 트리 개수를 설정합니다.
        n_estimators=n_estimators,

        # 같은 데이터로 실행할 때 같은 결과가 나오도록 난수값을 고정합니다.
        random_state=random_state,

        # 로컬 CPU 코어를 활용해 여러 결정트리를 병렬로 학습합니다.
        n_jobs=n_jobs,
    )

    # 아직 학습되지 않은 RandomForest 모델을 반환합니다.
    return model
