import numpy as np
import pytest

from evaluation.overlapping import overlapping_long_only_returns


def test_상승_신호_한_개는_자본_20퍼센트만_5거래일_보유한다():
    opens = np.full(7, 100.0)

    result = overlapping_long_only_returns(opens, [0], [1])

    assert len(result.strategy_net) == 5
    expected_equity = 0.8 + 0.2 * (1.0 - 0.00025) ** 2
    assert np.prod(1.0 + result.strategy_net) == pytest.approx(expected_equity)


def test_중립과_하락_신호에는_진입하거나_비용을_내지_않는다():
    opens = np.array([100.0, 100.0, 110.0, 121.0, 133.1, 146.41, 161.051, 177.1561])

    result = overlapping_long_only_returns(opens, [0, 1], [0, -1])

    assert result.strategy_net == pytest.approx(np.zeros(6))


def test_매일_상승이면_다섯_슬리브가_순서대로_겹친다():
    opens = np.full(11, 100.0)

    result = overlapping_long_only_returns(opens, [0, 1, 2, 3, 4], [1, 1, 1, 1, 1])

    expected_equity = (1.0 - 0.00025) ** 2
    assert np.prod(1.0 + result.strategy_net) == pytest.approx(expected_equity)
    assert len(result.strategy_net) == 9


def test_마지막_신호를_청산할_미래_시가가_없으면_중단한다():
    with pytest.raises(ValueError, match="미래 시가"):
        overlapping_long_only_returns(np.full(6, 100.0), [0], [1])
