"""워크포워드 분할 — 시간 순서를 지키는 학습·검증 구간 나누기.

**모델을 모른다.** 인덱스 쌍만 돌려준다. 그 인덱스로 무엇을 학습하든 상관없다.

왜 K-fold 를 쓰면 안 되는가
-------------------------
시계열을 무작위로 섞어 나누면 **미래로 과거를 예측**하게 된다. 8월 자료로 학습해 3월을
맞히는 셈이라, 그때는 알 수 없었던 정보가 학습에 새어 든다(look-ahead bias). 성능이
실제보다 훨씬 좋게 나오고, 실전에 올리면 그 성능이 사라진다.

`sklearn.model_selection.KFold` 를 이 프로젝트에서 쓰지 않는 이유다. `TimeSeriesSplit`
은 방향이 맞지만, 우리는 **갭(gap)** 과 **폴드별 보고**가 필요해 직접 짰다.

두 가지 창(window)
-----------------
    확장창 expanding   학습 [0 … t] → 검증 [t+1 … t+h]     자료를 계속 쌓는다
    롤링창 rolling     학습 [t-w … t] → 검증 [t+1 … t+h]   최근 w 개만 본다

    확장창  |=========|--|
            |============|--|
            |===============|--|

    롤링창  |=====|--|
              |=====|--|
                |=====|--|

어느 쪽이 옳은지는 자료가 정한다. 시장 성격이 변한다고 보면 롤링창이 맞고(옛 자료가
오히려 해롭다), 관계가 안정적이라고 보면 확장창이 맞다(자료가 많을수록 낫다).
**둘 다 돌려 보고 폴드별 분산을 비교하는 것**이 이 프로젝트의 기본 태도다.
"""

from __future__ import annotations

from typing import Iterator, List, Tuple

import numpy as np

# 첫 학습 구간의 최소 길이. 이보다 짧으면 어떤 모델이든 추정이 무의미하다.
# (원본 timeseries/backtest.py 가 ARIMA 기준으로 쓰던 값을 그대로 이어받았다)
MIN_TRAIN = 120

# 기본 폴드 수. 늘릴수록 추정이 안정되지만 매번 다시 학습하므로 그만큼 느려진다.
DEFAULT_FOLDS = 12

Split = Tuple[np.ndarray, np.ndarray]


def expanding_splits(n_samples: int, n_folds: int = DEFAULT_FOLDS,
                     min_train: int = MIN_TRAIN, horizon: int = 1,
                     gap: int = 0) -> List[Split]:
    """확장창 분할 — 학습 구간이 폴드마다 길어진다.

    Args:
        n_samples: 전체 표본 수
        n_folds:   폴드 수
        min_train: 첫 학습 구간의 최소 길이
        horizon:   한 폴드에서 검증할 시점 수
        gap:       학습 끝과 검증 시작 사이에 버릴 시점 수. **아래 설명을 읽는다**

    `gap` 을 왜 두는가
    -----------------
    레이블이 미래를 보고 만들어지는 경우가 있다. 예를 들어 "5일 뒤 상승" 레이블은
    t 시점 행이 t+5 의 가격을 알아야 만들어진다. 이때 학습 구간 마지막 행의 레이블이
    검증 구간 초반과 **겹친다** — 학습이 검증 답을 이미 본 셈이다.

    레이블이 k 일 앞을 보면 `gap=k` 로 둔다. 1일 등락 예측이면 `gap=0` 이 맞다.
    ⚠️ 이걸 빠뜨린 채 5일 레이블을 쓰면 성능이 조용히 부풀어 오른다.

    Returns:
        (학습 인덱스, 검증 인덱스) 쌍의 리스트. 시간 순으로 정렬돼 있다.
    """
    _validate(n_samples, n_folds, min_train, horizon, gap)

    splits: List[Split] = []
    # 마지막 폴드의 검증 구간이 표본 끝에 닿도록 시작점을 역산한다.
    last_train_end = n_samples - horizon
    first_train_end = min_train
    if n_folds == 1:
        train_ends = [last_train_end]
    else:
        step = (last_train_end - first_train_end) / (n_folds - 1)
        train_ends = [int(round(first_train_end + step * i)) for i in range(n_folds)]

    for train_end in train_ends:
        valid_start = train_end + gap
        valid_end = valid_start + horizon
        if valid_end > n_samples:
            # 표본 끝을 넘는 폴드는 만들지 않는다. 조용히 짧게 자르면
            # 폴드마다 검증 길이가 달라져 분산 비교가 뜻을 잃는다.
            continue
        splits.append((np.arange(0, train_end), np.arange(valid_start, valid_end)))
    return splits


def rolling_splits(n_samples: int, train_size: int, n_folds: int = DEFAULT_FOLDS,
                   horizon: int = 1, gap: int = 0) -> List[Split]:
    """롤링창 분할 — 학습 구간 길이를 고정하고 창을 앞으로 민다.

    옛 자료가 오히려 해롭다고 볼 때 쓴다. 시장 국면이 바뀌면 5년 전 관계는 지금과
    다르기 때문이다. `train_size` 를 얼마로 둘지가 이 방식의 핵심 가정이고,
    **그 값을 바꿔 가며 결과가 얼마나 흔들리는지 보는 것** 자체가 하나의 실험이다.
    """
    _validate(n_samples, n_folds, train_size, horizon, gap)

    splits: List[Split] = []
    last_train_end = n_samples - horizon - gap
    first_train_end = train_size
    if n_folds == 1:
        train_ends = [last_train_end]
    else:
        step = (last_train_end - first_train_end) / (n_folds - 1)
        train_ends = [int(round(first_train_end + step * i)) for i in range(n_folds)]

    for train_end in train_ends:
        train_start = train_end - train_size
        valid_start = train_end + gap
        valid_end = valid_start + horizon
        if train_start < 0 or valid_end > n_samples:
            continue
        splits.append((np.arange(train_start, train_end), np.arange(valid_start, valid_end)))
    return splits


def iter_splits(splits: List[Split]) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
    """폴드 번호를 붙여 훑는다. 폴드별 성능을 따로 기록할 때 쓴다.

    ⚠️ **폴드별 결과를 평균만 내고 버리지 않는다.** 평균이 좋아도 폴드 사이 분산이
       크면 그 전략은 시기에 따라 완전히 다르게 움직인다는 뜻이다. 그 분산이야말로
       "이 성능을 믿어도 되는가"에 대한 답이다.
    """
    for i, (train_idx, valid_idx) in enumerate(splits):
        yield i, train_idx, valid_idx


def _validate(n_samples: int, n_folds: int, min_train: int, horizon: int, gap: int) -> None:
    """분할이 성립하지 않는 조건을 **미리** 막는다.

    조용히 빈 리스트를 돌려주면 호출부는 "폴드가 0개"인 채로 계속 돌다가 한참 뒤에
    엉뚱한 곳에서 터진다. 여기서 무엇이 부족한지 말해 주는 편이 낫다.
    """
    if n_folds < 1:
        raise ValueError(f"n_folds 는 1 이상이어야 한다 (받은 값: {n_folds})")
    if horizon < 1:
        raise ValueError(f"horizon 은 1 이상이어야 한다 (받은 값: {horizon})")
    if gap < 0:
        raise ValueError(f"gap 은 0 이상이어야 한다 (받은 값: {gap})")
    needed = min_train + gap + horizon
    if n_samples < needed:
        raise ValueError(
            f"표본이 부족하다. {n_samples}개로는 분할할 수 없다.\n"
            f"  필요: 최소학습 {min_train} + 갭 {gap} + 검증 {horizon} = {needed}개\n"
            f"  해결: 자료 구간을 늘리거나(과거 백필), min_train 을 낮춘다."
        )
