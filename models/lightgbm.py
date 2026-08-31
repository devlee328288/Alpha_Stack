"""LightGBM 기준 모델을 생성하는 모듈."""

# LightGBM의 sklearn 호환 분류 모델을 가져옵니다.
from lightgbm import LGBMClassifier

# 같은 데이터로 실행했을 때 같은 결과가 나오도록 난수값을 고정합니다.
DEFAULT_RANDOM_STATE = 42

# 순차적으로 학습할 트리의 개수를 설정합니다.
DEFAULT_N_ESTIMATORS = 100

# 한 번에 모델이 학습하는 크기를 조절합니다.
DEFAULT_LEARNING_RATE = 0.05

# 사용할 수 있는 CPU 코어를 모두 사용합니다.
DEFAULT_N_JOBS = -1


def build_lightgbm_baseline(
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    n_jobs: int = DEFAULT_N_JOBS,
) -> LGBMClassifier:
    """3분류 방향 예측용 LightGBM 모델을 만듭니다.

    이 함수는 데이터를 직접 분할하지 않습니다.
    워크포워드에서 전달받은 학습 데이터만 fit()에 넣어야 합니다.

    LightGBM은 결정트리 기반 모델이므로 StandardScaler를 사용하지 않습니다.

    Args:
        random_state:
            같은 데이터로 다시 학습했을 때 결과를 재현하기 위한 난수값입니다.
        n_estimators:
            순차적으로 학습할 결정트리의 개수입니다.
        learning_rate:
            각 결정트리가 이전 오류를 보완하는 정도입니다.
        n_jobs:
            학습에 사용할 CPU 코어 개수입니다. -1은 모든 코어를 의미합니다.

    Returns:
        아직 학습되지 않은 LGBMClassifier입니다.
    """

    # 하락·중립·상승을 예측하는 다중분류 모델을 생성합니다.
    model = LGBMClassifier(
        # 세 개 클래스를 예측하는 다중분류 문제임을 명시합니다.
        objective="multiclass",

        # 하락·중립·상승의 세 클래스를 사용합니다.
        num_class=3,

        # 순차적으로 학습할 결정트리 개수를 설정합니다.
        n_estimators=n_estimators,

        # 각 트리가 이전 오류를 보완하는 정도를 설정합니다.
        learning_rate=learning_rate,

        # 같은 데이터에서 같은 결과가 나오도록 난수값을 고정합니다.
        random_state=random_state,

        # 로컬 CPU 코어를 활용해 학습합니다.
        n_jobs=n_jobs,

        # 단위 테스트와 실행 결과에 불필요한 학습 메시지를 표시하지 않습니다.
        verbosity=-1,
    )

    # 아직 학습되지 않은 LightGBM 모델을 반환합니다.
    return model
