"""Logistic Regression 기준 모델을 생성하는 모듈."""

# Logistic Regression 분류 모델을 가져옵니다.
from sklearn.linear_model import LogisticRegression

# 전처리와 모델을 하나로 연결하기 위한 Pipeline을 가져옵니다.
from sklearn.pipeline import Pipeline

# 피처마다 값의 크기를 비슷하게 맞추기 위한 StandardScaler를 가져옵니다.
from sklearn.preprocessing import StandardScaler

# 같은 데이터로 실행했을 때 같은 결과가 나오도록 난수값을 고정합니다.
DEFAULT_RANDOM_STATE = 42

# 학습이 너무 일찍 종료되지 않도록 최대 반복 횟수를 충분히 설정합니다.
DEFAULT_MAX_ITER = 1_000


def build_logistic_baseline(
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
    max_iter: int = DEFAULT_MAX_ITER,
) -> Pipeline:
    """3분류 방향 예측용 Logistic Regression 모델을 만듭니다.

    이 함수는 데이터를 학습 구간과 검증 구간으로 직접 나누지 않습니다.
    워크포워드에서 전달받은 학습 데이터만 model.fit()에 넣어야 합니다.

    Args:
        random_state:
            같은 데이터로 다시 학습했을 때 결과를 재현하기 위한 난수값입니다.
        max_iter:
            Logistic Regression이 반복해서 학습할 수 있는 최대 횟수입니다.

    Returns:
        StandardScaler와 LogisticRegression을 연결한 Pipeline입니다.
    """

    # 스케일러와 모델을 하나의 Pipeline으로 연결합니다.
    #
    # 스케일러를 Pipeline 안에 넣는 이유:
    # 학습 구간의 평균과 표준편차만 사용하도록 만들기 위해서입니다.
    # 전체 데이터를 먼저 스케일링하면 검증 구간의 정보가 학습에 섞일 수 있습니다.
    model = Pipeline(
        steps=[
            (
                # 첫 번째 단계에서 각 피처의 크기를 비슷하게 맞춥니다.
                "scaler",
                StandardScaler(),
            ),
            (
                # 두 번째 단계에서 상승·중립·하락 클래스를 학습합니다.
                "classifier",
                LogisticRegression(
                    # 모델 실행 결과를 재현할 수 있도록 난수값을 고정합니다.
                    random_state=random_state,

                    # 학습이 수렴할 수 있도록 최대 반복 횟수를 설정합니다.
                    max_iter=max_iter,
                ),
            ),
        ]
    )

    # 아직 학습되지 않은 Pipeline 모델을 반환합니다.
    return model
