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


class LeakageError(RuntimeError):
    """학습 구간과 검증 구간이 **레이블을 통해** 겹칠 때 던진다.

    인덱스가 겹치지 않아도 누수는 일어난다. 레이블이 미래를 보고 만들어지기 때문이다.
    t 행의 레이블이 t+5 가격으로 만들어졌다면, 학습 마지막 행 t 는 이미 t+5 를 알고
    있다. 검증이 t+1 에서 시작하면 학습은 검증 구간의 답을 본 셈이다.

    **이 오류는 조용히 넘어가면 안 된다.** 누수는 예외를 내지 않고 성능만 올린다.
    그래서 기본값을 안전한 쪽에 두고, 위험한 쪽을 선택할 때만 명시하게 만들었다.
    """


def _resolve_gap(gap: int | None, horizon: int, label_horizon: int | None,
                 allow_leakage: bool) -> Tuple[int, int]:
    """`gap` 을 확정하고, 누수가 되는 값이면 막는다.

    규칙은 셋뿐이다.

    1. `gap` 을 안 주면 **`label_horizon` 과 같게** 둔다 (그게 누수 없는 최솟값이다)
    2. `label_horizon` 을 안 주면 `horizon` 으로 본다 → 안 준 채 호출하면 `gap == horizon`
    3. `gap < label_horizon` 이면 `LeakageError`. `allow_leakage=True` 일 때만 통과

    ⚠️ **`horizon` 과 `label_horizon` 은 다른 것이다**
    ------------------------------------------------
    이 모듈에서 `horizon` 은 *한 폴드에서 검증할 시점 수*(검증창 길이)이고,
    `label_horizon` 은 *레이블이 앞을 보는 거리*다. 우리 프로젝트는 둘 다 5라
    (5거래일 레이블 · ADR-AS-0002) 구별하지 않아도 우연히 맞지만, 검증창을 넓히는
    순간(예: `horizon=60`) 둘은 갈라진다. 그때 `gap` 을 검증창 길이에 맞추면
    **60시점을 통째로 버리게 되고**, 반대로 규칙을 검증창 기준으로 검사하면
    올바른 호출(`horizon=60, gap=5`)이 억울하게 막힌다.

    그래서 누수 판정은 **언제나 `label_horizon` 기준**이다.

    Returns:
        (확정된 gap, 확정된 label_horizon)
    """
    lh = horizon if label_horizon is None else label_horizon
    if lh < 0:
        raise ValueError(f"label_horizon 은 0 이상이어야 한다 (받은 값: {lh})")

    if gap is None:
        return lh, lh

    if gap < 0:
        raise ValueError(f"gap 은 0 이상이어야 한다 (받은 값: {gap})")

    if gap < lh and not allow_leakage:
        # 줄을 리스트로 조립한다 — 긴 f-string 안에 이스케이프를 섞으면 읽기도 고치기도 나쁘다
        raise LeakageError(
            "\n".join([
                f"gap={gap} 은 레이블 앞보기 {lh} 시점보다 짧아 학습이 검증 답을 본다.",
                f"  왜 막나: 레이블이 t+{lh} 를 보고 만들어지므로 학습 마지막 "
                f"{lh - gap}개 행의 레이블이 검증 구간과 겹친다. "
                f"성능이 조용히 부풀고 예외는 나지 않는다.",
                f"  해결 ①: gap 을 빼고 호출한다 → 자동으로 {lh} 이 된다 (권장)",
                f"  해결 ②: gap={lh} 이상을 명시한다",
                "  해결 ③: 누수 폭을 '측정'하는 것이 목적이라면 allow_leakage=True 를 "
                "명시한다 (리포트에 누수 실험이라고 적을 것)",
            ])
        )
    return gap, lh


def expanding_splits(n_samples: int, n_folds: int = DEFAULT_FOLDS,
                     min_train: int = MIN_TRAIN, horizon: int = 1,
                     gap: int | None = None, *,
                     label_horizon: int | None = None,
                     allow_leakage: bool = False) -> List[Split]:
    """확장창 분할 — 학습 구간이 폴드마다 길어진다.

    Args:
        n_samples:     전체 표본 수
        n_folds:       폴드 수
        min_train:     첫 학습 구간의 최소 길이
        horizon:       **한 폴드에서 검증할 시점 수** (검증창 길이)
        gap:           학습 끝과 검증 시작 사이에 버릴 시점 수.
                       **주지 않으면 `label_horizon` 과 같아진다** — 그게 안전한 기본값이다
        label_horizon: 레이블이 앞을 보는 거리. 생략하면 `horizon` 으로 본다
        allow_leakage: `gap < label_horizon` 을 허용한다. **누수 폭을 측정할 때만** 쓴다

    `gap` 을 왜 두는가
    -----------------
    레이블이 미래를 보고 만들어지는 경우가 있다. 예를 들어 "5일 뒤 상승" 레이블은
    t 시점 행이 t+5 의 가격을 알아야 만들어진다. 이때 학습 구간 마지막 행의 레이블이
    검증 구간 초반과 **겹친다** — 학습이 검증 답을 이미 본 셈이다.

    ⚠️ **기본값이 0 이 아니다** (2026-08-26 변경 · 요구사항 F-09)
    ---------------------------------------------------------
    예전 기본값은 `gap=0` 이었다. 즉 **아무 생각 없이 부르면 누수되는** 설계였다.
    누수는 예외를 내지 않고 성능만 올리므로, 빠뜨린 사람은 끝까지 모른다.
    그래서 기본값을 안전한 쪽(`gap = label_horizon`)으로 뒤집고, 위험한 쪽을
    고를 때만 `allow_leakage=True` 를 **명시**하게 만들었다.

        expanding_splits(n, horizon=5)                    # gap=5 (자동)
        expanding_splits(n, horizon=60, label_horizon=5)  # gap=5 — 검증창은 60
        expanding_splits(n, horizon=5, gap=0)             # ❌ LeakageError
        expanding_splits(n, horizon=5, gap=0,
                         allow_leakage=True)              # ✅ 누수 폭 측정용

    Returns:
        (학습 인덱스, 검증 인덱스) 쌍의 리스트. 시간 순으로 정렬돼 있다.

    Raises:
        LeakageError: `gap` 이 `label_horizon` 보다 짧은데 `allow_leakage` 가 아닐 때
    """
    gap, _lh = _resolve_gap(gap, horizon, label_horizon, allow_leakage)
    _validate(n_samples, n_folds, min_train, horizon, gap)

    splits: List[Split] = []
    # 마지막 폴드의 검증 구간이 표본 끝에 닿도록 시작점을 역산한다.
    # ⚠️ gap 을 빼지 않으면 마지막 폴드가 표본 끝을 넘어 **조용히 버려진다.**
    #    폴드 수를 12 로 달라고 했는데 11 개가 나오는 원인이 이것이었다.
    last_train_end = n_samples - horizon - gap
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
                   horizon: int = 1, gap: int | None = None, *,
                   label_horizon: int | None = None,
                   allow_leakage: bool = False) -> List[Split]:
    """롤링창 분할 — 학습 구간 길이를 고정하고 창을 앞으로 민다.

    옛 자료가 오히려 해롭다고 볼 때 쓴다. 시장 국면이 바뀌면 5년 전 관계는 지금과
    다르기 때문이다. `train_size` 를 얼마로 둘지가 이 방식의 핵심 가정이고,
    **그 값을 바꿔 가며 결과가 얼마나 흔들리는지 보는 것** 자체가 하나의 실험이다.

    `gap` · `label_horizon` · `allow_leakage` 규칙은 `expanding_splits` 와 같다.
    기본값은 `gap = label_horizon` 이고, 더 짧게 주면 `LeakageError` 다.
    """
    gap, _lh = _resolve_gap(gap, horizon, label_horizon, allow_leakage)
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
