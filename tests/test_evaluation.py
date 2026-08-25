"""검증 엔진 테스트 — 이 코드가 틀리면 프로젝트 전체의 결론이 틀린다.

여기 있는 것은 "돌아가는가"가 아니라 **"틀린 답을 내지 않는가"** 를 본다.
특히 look-ahead 와 거래비용은 틀려도 예외가 나지 않아 테스트로만 잡힌다.
"""

from __future__ import annotations

import numpy as np
import pytest

from evaluation import baseline, metrics, walk_forward

# ── metrics ────────────────────────────────────────────────────────────────

def test_equity_curve_는_합이_아니라_곱으로_쌓인다():
    """-50% 뒤 +50% 는 본전이 아니라 -25% 다."""
    curve = metrics.equity_curve([-0.5, 0.5])
    assert curve[-1] == pytest.approx(0.75)


def test_max_drawdown_은_음수로_나온다():
    # 1.0 → 1.2 → 0.6 : 고점 1.2 대비 0.6 이므로 -50%
    mdd = metrics.max_drawdown([1.0, 1.2, 0.6, 0.9])
    assert mdd == pytest.approx(-0.5)


def test_max_drawdown_은_한번도_안_빠지면_0():
    assert metrics.max_drawdown([1.0, 1.1, 1.2]) == pytest.approx(0.0)


def test_sharpe_는_변동성이_0이면_None():
    """0.0 이 아니라 None 이어야 한다. '위험 대비 수익 없음'과 '잴 수 없음'은 다르다."""
    assert metrics.sharpe_ratio([0.0, 0.0, 0.0, 0.0]) is None


def test_sharpe_는_표본이_너무_짧으면_None():
    assert metrics.sharpe_ratio([0.01]) is None


def test_hit_rate_는_보합을_분모에서_뺀다():
    # actual 의 0 은 방향이 없어 맞고 틀림을 가릴 수 없다.
    # 유효한 것은 [1, -1] 두 개이고 예측이 [1, -1] 이므로 1.0
    assert metrics.hit_rate([1, 1, -1], [1, 0, -1]) == pytest.approx(1.0)


def test_hit_rate_는_길이가_다르면_에러():
    with pytest.raises(ValueError):
        metrics.hit_rate([1, 1], [1])


# ── 거래비용 — 여기가 조용히 틀리는 곳이다 ──────────────────────────────────

def test_거래비용은_포지션이_바뀔_때만_든다():
    """계속 들고 있으면 비용이 없다."""
    pos = [1.0, 1.0, 1.0]
    ret = [0.0, 0.0, 0.0]
    net = metrics.apply_cost(pos, ret, round_trip_cost=0.003)
    # 첫날 진입 비용만 든다: turnover 1 × 0.0015
    assert net[0] == pytest.approx(-0.0015)
    assert net[1] == pytest.approx(0.0)
    assert net[2] == pytest.approx(0.0)


def test_진입과_청산이_왕복_비용_한_번이_된다():
    """0 → +1 → 0 은 정확히 왕복 한 번이어야 한다."""
    pos = [1.0, 0.0]
    ret = [0.0, 0.0]
    net = metrics.apply_cost(pos, ret, round_trip_cost=0.003)
    assert float(np.sum(net)) == pytest.approx(-0.003)


def test_첫_시점_진입_비용을_빠뜨리지_않는다():
    """positions[0] 이 0 이 아니면 아무것도 없던 상태에서 잡은 것이므로 비용이 든다.

    이걸 빠뜨리면 폴드마다 진입이 공짜가 되어 성과가 조용히 부풀어 오른다.
    """
    net = metrics.apply_cost([1.0], [0.0], round_trip_cost=0.003)
    assert net[0] == pytest.approx(-0.0015)


def test_뒤집기는_왕복_한_번_값이_든다():
    """+1 → -1 은 turnover 2 이므로 왕복 한 번 값."""
    pos = [1.0, -1.0]
    ret = [0.0, 0.0]
    net = metrics.apply_cost(pos, ret, round_trip_cost=0.003)
    # 첫날 진입 0.0015 + 뒤집기 0.003
    assert float(np.sum(net)) == pytest.approx(-(0.0015 + 0.003))


def test_summarize_는_비용_전후를_모두_담는다():
    rng = np.random.default_rng(42)
    ret = rng.normal(0.0, 0.01, 100)
    pos = np.ones(100)
    out = metrics.summarize(pos, ret)
    assert out["total_return_gross"] > out["total_return_net"]   # 비용만큼 낮다
    assert out["assumptions"]["round_trip_cost"] == 0.003
    assert out["n_periods"] == 100


# ── walk_forward — 미래를 보지 않는가 ──────────────────────────────────────

def test_분할은_언제나_학습이_검증보다_앞선다():
    """이 성질이 깨지면 look-ahead 다. 모든 폴드에서 확인한다."""
    splits = walk_forward.expanding_splits(n_samples=300, n_folds=5, min_train=120)
    assert len(splits) == 5
    for train_idx, valid_idx in splits:
        assert train_idx.max() < valid_idx.min()


def test_확장창은_학습_구간이_점점_길어진다():
    splits = walk_forward.expanding_splits(n_samples=300, n_folds=5, min_train=120)
    lengths = [len(t) for t, _ in splits]
    assert lengths == sorted(lengths)
    assert lengths[0] < lengths[-1]


def test_롤링창은_학습_구간_길이가_일정하다():
    splits = walk_forward.rolling_splits(n_samples=300, train_size=100, n_folds=5)
    lengths = {len(t) for t, _ in splits}
    assert lengths == {100}


def test_gap_이_학습과_검증_사이를_벌린다():
    """레이블이 k 일 앞을 보면 gap=k 로 겹침을 막는다."""
    splits = walk_forward.expanding_splits(n_samples=300, n_folds=3, min_train=120,
                                           horizon=1, gap=5)
    for train_idx, valid_idx in splits:
        assert valid_idx.min() - train_idx.max() >= 5


def test_표본이_부족하면_무엇이_모자란지_말해_준다():
    """조용히 빈 리스트를 돌려주면 한참 뒤 엉뚱한 곳에서 터진다."""
    with pytest.raises(ValueError, match="표본이 부족하다"):
        walk_forward.expanding_splits(n_samples=50, min_train=120)


# ── baseline — 우리가 이겨야 할 상대 ───────────────────────────────────────

def test_always_up_은_상승_편향을_그대로_드러낸다():
    """상승이 70% 인 구간에서는 늘 상승이라 답해도 70% 가 나온다."""
    y = np.array([1] * 70 + [-1] * 30)
    assert metrics.hit_rate(baseline.always_up(len(y)), y) == pytest.approx(0.7)


def test_majority_class_는_학습_구간만_본다():
    """검증 구간 분포를 미리 보면 기준선이 부당하게 강해진다."""
    y_train = np.array([-1] * 80 + [1] * 20)   # 학습은 하락이 다수
    y_valid = np.array([1] * 90 + [-1] * 10)   # 검증은 상승이 다수
    pred = baseline.majority_class(y_train, len(y_valid))
    assert set(np.unique(pred)) == {-1}        # 학습 기준이므로 하락을 고른다


def test_evaluate_all_은_못_잰_것을_0으로_채우지_않는다():
    y_train = np.array([1, -1, 1, -1])
    y_valid = np.array([1, 1, -1])
    scores = baseline.evaluate_all(y_train, y_valid)          # y_prev 없음
    assert scores["previous_direction"] is None               # 0.0 이 아니다
    assert scores["always_up"] is not None
