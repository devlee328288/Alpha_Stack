"""F-09 — `walk_forward` gap 잠금 테스트.

**왜 이 파일이 따로 있는가.** 누수(leakage)는 예외를 내지 않는다. 성능만 조용히
올린다. 그래서 "돌아가는가"를 보는 테스트로는 절대 잡히지 않고, **기본값이 안전한
쪽인지**를 직접 물어보는 테스트로만 잡힌다.

예전 기본값은 `gap=0` 이었다. 즉 아무 생각 없이 부르면 누수됐다. 이 파일은 그 기본값이
다시 0 으로 돌아가는 것을 막는 것이 목적이다.

    수용 기준 (docs/요구사항.md F-09)
    - gap 미지정 호출이 gap == horizon 으로 동작
    - gap=0 은 LeakageError, allow_leakage=True 일 때만 통과
    - 기존 테스트가 그대로 통과
"""

from __future__ import annotations

import numpy as np
import pytest

from evaluation import walk_forward
from evaluation.walk_forward import LeakageError

# ── 기본값이 안전한 쪽인가 ─────────────────────────────────────────────────

@pytest.mark.parametrize("horizon", [1, 3, 5, 10])
def test_gap_을_안_주면_horizon_과_같아진다(horizon):
    """F-09 핵심. 기본값이 0 이면 이 테스트가 깨진다."""
    splits = walk_forward.expanding_splits(n_samples=600, n_folds=4, min_train=200,
                                           horizon=horizon)
    for train_idx, valid_idx in splits:
        # train_idx.max() 는 학습 마지막 '인덱스', valid 시작은 그 다음 칸 + gap
        실제갭 = int(valid_idx.min()) - int(train_idx.max()) - 1
        assert 실제갭 == horizon


def test_롤링창도_같은_기본값을_쓴다():
    splits = walk_forward.rolling_splits(n_samples=600, train_size=200, n_folds=4,
                                         horizon=5)
    for train_idx, valid_idx in splits:
        assert int(valid_idx.min()) - int(train_idx.max()) - 1 == 5


# ── gap=0 을 막는가 ────────────────────────────────────────────────────────

def test_gap_0_은_LeakageError():
    with pytest.raises(LeakageError):
        walk_forward.expanding_splits(n_samples=600, n_folds=4, min_train=200,
                                      horizon=5, gap=0)


def test_롤링창_gap_0_도_LeakageError():
    with pytest.raises(LeakageError):
        walk_forward.rolling_splits(n_samples=600, train_size=200, n_folds=4,
                                    horizon=5, gap=0)


def test_gap_이_모자라기만_해도_막는다():
    """0 만 막으면 gap=3 · 레이블 5일 이 통과해 버린다. 부분 누수도 누수다."""
    with pytest.raises(LeakageError):
        walk_forward.expanding_splits(n_samples=600, n_folds=4, min_train=200,
                                      horizon=5, gap=3)


def test_LeakageError_는_무엇을_해야_하는지_알려_준다():
    """막다른 길로 만들지 않는다 — 해결책이 메시지에 있어야 한다."""
    with pytest.raises(LeakageError) as err:
        walk_forward.expanding_splits(n_samples=600, n_folds=4, min_train=200,
                                      horizon=5, gap=0)
    메시지 = str(err.value)
    assert "allow_leakage=True" in 메시지        # 빠져나갈 문이 적혀 있다
    assert "해결" in 메시지
    assert "gap=5" in 메시지                     # 얼마로 둬야 하는지 숫자로


# ── 누수를 '측정' 하려는 경우는 통과시키는가 ──────────────────────────────

def test_allow_leakage_면_gap_0_이_통과한다():
    """누수 폭(무작위 K-fold 대비 정확도 차이)을 재려면 일부러 누수시켜야 한다."""
    splits = walk_forward.expanding_splits(n_samples=600, n_folds=4, min_train=200,
                                           horizon=5, gap=0, allow_leakage=True)
    assert len(splits) == 4
    for train_idx, valid_idx in splits:
        assert int(valid_idx.min()) - int(train_idx.max()) - 1 == 0


# ── horizon(검증창) 과 label_horizon(레이블 앞보기) 을 구별하는가 ──────────

def test_검증창을_넓혀도_gap_은_레이블_기준이다():
    """`horizon` 은 검증창 길이, `label_horizon` 은 레이블 앞보기다. 섞이면 안 된다.

    검증창 60 · 레이블 5일이면 버릴 것은 5시점이지 60시점이 아니다.
    이걸 섞으면 검증 표본을 12배로 헛되이 버린다.
    """
    splits = walk_forward.expanding_splits(n_samples=1200, n_folds=4, min_train=400,
                                           horizon=60, label_horizon=5)
    assert splits
    for train_idx, valid_idx in splits:
        assert int(valid_idx.min()) - int(train_idx.max()) - 1 == 5
        assert len(valid_idx) == 60


def test_검증창이_넓어도_레이블보다_짧은_gap_은_막힌다():
    with pytest.raises(LeakageError):
        walk_forward.expanding_splits(n_samples=1200, n_folds=4, min_train=400,
                                      horizon=60, label_horizon=5, gap=2)


def test_검증창보다_짧은_gap_이라도_레이블을_넘으면_통과한다():
    """`gap=5 < horizon=60` 이지만 레이블 기준으로는 충분하다. 억울하게 막지 않는다."""
    splits = walk_forward.expanding_splits(n_samples=1200, n_folds=4, min_train=400,
                                           horizon=60, label_horizon=5, gap=5)
    assert len(splits) == 4


# ── 폴드가 조용히 사라지지 않는가 (회귀) ──────────────────────────────────

def test_gap_이_있어도_요청한_폴드_수가_그대로_나온다():
    """예전 expanding_splits 는 last_train_end 에서 gap 을 빼지 않아 마지막 폴드를
    조용히 버렸다. 12 개를 달라고 했는데 11 개가 나오면 폴드별 분산 비교가 어긋난다.
    """
    for horizon in (1, 5, 20):
        splits = walk_forward.expanding_splits(n_samples=2000, n_folds=12,
                                               min_train=500, horizon=horizon)
        assert len(splits) == 12, f"horizon={horizon} 에서 폴드가 사라졌다"


def test_마지막_폴드의_검증이_표본_끝을_넘지_않는다():
    n = 2000
    splits = walk_forward.expanding_splits(n_samples=n, n_folds=12, min_train=500,
                                           horizon=5)
    마지막검증 = splits[-1][1]
    assert int(마지막검증.max()) <= n - 1


# ── 시간 순서는 여전히 지켜지는가 ──────────────────────────────────────────

def test_gap_기본값_아래서도_학습이_검증보다_앞선다():
    for splits in (
        walk_forward.expanding_splits(n_samples=800, n_folds=6, min_train=200, horizon=5),
        walk_forward.rolling_splits(n_samples=800, train_size=200, n_folds=6, horizon=5),
    ):
        assert splits
        for train_idx, valid_idx in splits:
            assert int(train_idx.max()) < int(valid_idx.min())
            assert np.all(np.diff(train_idx) > 0)


# ── 잘못된 입력 ────────────────────────────────────────────────────────────

def test_음수_gap_은_ValueError():
    with pytest.raises(ValueError, match="gap"):
        walk_forward.expanding_splits(n_samples=600, n_folds=4, min_train=200,
                                      horizon=5, gap=-1)


def test_표본_부족_메시지가_갭을_포함해_계산된다():
    """gap 이 기본으로 붙으므로 필요 표본도 그만큼 늘어난다. 그 사실이 메시지에 나와야 한다."""
    with pytest.raises(ValueError, match="표본이 부족하다"):
        walk_forward.expanding_splits(n_samples=125, min_train=120, horizon=5)
