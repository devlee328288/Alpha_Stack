"""기준선 — 우리 모델이 반드시 이겨야 할 상대들.

    "정확도 62%" 는 그 자체로 아무 말도 하지 않는다.

상승이 62% 나오는 구간에서는 **언제나 "오른다"고 답하기만 해도** 62% 가 나온다.
그래서 이 프로젝트의 모든 성능 보고에는 기준선이 나란히 실린다. 못 이기면 못 이겼다고
적는다. 주가는 효율시장에 가까워서 **못 이기는 것이 정상에 가깝고**, 그 사실을 감추면
보고서가 거짓말이 된다.

기준선 3종
---------
| 이름 | 무엇을 하나 | 무엇을 드러내나 |
|---|---|---|
| `always_up`        | 늘 상승이라고 답한다 | 자료의 상승 편향. 가장 흔한 착시의 원인 |
| `majority_class`   | 학습 구간의 다수 클래스로 답한다 | 편향을 학습에서 배운 경우 |
| `previous_direction` | 어제 방향이 오늘도 이어진다고 본다 | 모멘텀(추세 지속) 성분 |

⚠️ 세 기준선은 학습 구간만 본다
-----------------------------
`majority_class` 는 **학습 구간의** 다수 클래스를 쓴다. 전체 구간에서 세면 검증 구간의
정답 분포를 미리 본 것이 되어, 기준선이 실제보다 강해진다. 기준선을 부당하게 강하게
만들면 우리 모델이 억울하게 진다 — 반대 방향의 오류지만 똑같이 틀린 비교다.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

UP, DOWN = 1, -1


def always_up(n_valid: int) -> np.ndarray:
    """늘 상승(+1). 가장 단순하고, 가장 자주 우리를 부끄럽게 만드는 기준선."""
    return np.full(n_valid, UP, dtype=int)


def majority_class(y_train: Sequence[int], n_valid: int) -> np.ndarray:
    """학습 구간에서 더 많았던 방향으로 검증 구간 전체를 답한다.

    동수이면 상승으로 둔다 — 장기적으로 주가지수가 우상향해 온 경향을 반영한 임의의
    선택이고, 임의라는 사실을 여기 적어 둔다.
    """
    y = np.asarray(y_train)
    if y.size == 0:
        return always_up(n_valid)
    n_up = int(np.sum(y == UP))
    n_down = int(np.sum(y == DOWN))
    choice = UP if n_up >= n_down else DOWN
    return np.full(n_valid, choice, dtype=int)


def previous_direction(y_prev: Sequence[int]) -> np.ndarray:
    """직전 시점의 방향이 그대로 이어진다고 본다 (모멘텀 기준선).

    Args:
        y_prev: 검증 시점 **바로 앞** 시점들의 실제 방향.
                검증 구간이 [t+1 … t+h] 라면 [t … t+h-1] 의 실제 방향을 넣는다.

    ⚠️ 검증 구간의 정답을 그대로 넣지 않는다. 그러면 100% 가 나온다. 한 칸 앞선
       값이어야 한다 — 이 함수는 그 정렬을 검사할 방법이 없으니 호출부가 지킨다.
    """
    prev = np.asarray(y_prev, dtype=int)
    # 보합(0)은 방향이 없다. 직전 방향을 알 수 없으므로 상승으로 둔다.
    return np.where(prev == 0, UP, prev)


def evaluate_all(y_train: Sequence[int], y_valid: Sequence[int],
                 y_prev: Optional[Sequence[int]] = None) -> Dict[str, Optional[float]]:
    """기준선 3종의 적중률을 한 번에 낸다. 우리 모델 점수 옆에 그대로 놓는다.

    `y_prev` 가 없으면 `previous_direction` 은 `None` 으로 남는다 — 잴 수 없는 것을
    0 으로 채우지 않는다. 보고서에서 "0점"과 "측정 안 함"은 다른 말이다.
    """
    from evaluation.metrics import hit_rate  # 순환 import 를 피해 함수 안에서 부른다

    y_valid = np.asarray(y_valid, dtype=int)
    n = y_valid.size

    scores: Dict[str, Optional[float]] = {
        "always_up": hit_rate(always_up(n), y_valid),
        "majority_class": hit_rate(majority_class(y_train, n), y_valid),
        "previous_direction": None,
    }
    if y_prev is not None:
        scores["previous_direction"] = hit_rate(previous_direction(y_prev), y_valid)
    return scores
