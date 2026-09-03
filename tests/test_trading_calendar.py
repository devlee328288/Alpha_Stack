"""거래일 달력 — `next_session()` 이 이분탐색으로 바뀐 뒤에도 답이 같은지 지킨다.

`next_session()` 은 예전에 달력 전체를 훑어 `min(...)` 을 구했다. `known_at`·`eff_dd`·
`next_business_day` 가 **행마다** 부르는 함수라, 만 단위 수집에서 그 비용이 그대로
시간이 된다. 그래서 정렬본 + `bisect` 로 바꿨다.

이 파일이 지키는 것은 두 가지다.

1. **답이 안 바뀌었나** — 기대값을 새 구현에서 뽑으면 항등식이라 아무것도 못 잡는다.
   그래서 옛 선형 구현을 이 파일 안에 그대로 두고 **다른 경로의 기대값**으로 쓴다.
2. **정렬본이 낡지 않나** — 정렬본은 집합에서 파생된 사본이라, 집합만 갈아 끼우면
   조용히 옛 달력으로 답할 수 있다. 테스트가 실제로 `_SESSION_CACHE` 를 monkeypatch
   하므로(`test_inbox_derive.py` 등) 이 경우가 실제로 일어난다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import trading_calendar as tc  # noqa: E402

#: 2021년 첫 주. 신정과 주말이 빠져 있다 — 주말 규칙만으로는 못 만드는 달력이다.
달력_가 = frozenset({
    "20201231", "20210104", "20210105", "20210106", "20210107", "20210108",
    "20210111", "20210112",
})

#: 크기는 같고 내용이 다른 달력. 크기 비교로 정렬본을 갱신하면 여기서 걸린다.
달력_나 = frozenset({
    "20201231", "20210104", "20210105", "20210106", "20210107", "20210108",
    "20210111", "20210115",          # ← 마지막 날만 다르다
})


def next_session_옛(bas_dd: str, days: frozenset) -> str:
    """전환 이전 구현. 기대값을 **다른 경로에서** 얻기 위해 남겨 둔다."""
    later = [d for d in days if d > bas_dd]
    if not later:
        raise tc.CalendarOutOfRange("범위 밖")
    return min(later)


@pytest.fixture
def 달력끼우기(monkeypatch):
    def 끼운다(days: frozenset):
        monkeypatch.setattr(tc, "_SESSION_CACHE", days)
        monkeypatch.setattr(tc, "_SESSION_SPAN", (min(days), max(days)))
    return 끼운다


# ==================================================
# 1. 답이 옛 구현과 같은가
# ==================================================
@pytest.mark.parametrize("물음", [
    "20201231", "20210101", "20210102", "20210103", "20210104",
    "20210105", "20210108", "20210109", "20210110", "20210111",
])
def test_이분탐색이_선형스캔과_같은_답을_준다(달력끼우기, 물음):
    달력끼우기(달력_가)
    assert tc.next_session(물음) == next_session_옛(물음, 달력_가)


def test_거래일을_inclusive로_물으면_그_날이_나온다(달력끼우기):
    달력끼우기(달력_가)
    for d in sorted(달력_가)[:-1]:
        assert tc.next_session(d, inclusive=True) == d


def test_휴장일을_inclusive로_물으면_다음_거래일이_나온다(달력끼우기):
    달력끼우기(달력_가)
    # 20210101 은 신정이라 달력에 없다. inclusive 여부와 무관하게 다음 거래일이 답이다.
    assert tc.next_session("20210101", inclusive=True) == "20210104"
    assert tc.next_session("20210101", inclusive=False) == "20210104"


# ==================================================
# 2. 정렬본이 낡지 않는가 — 회귀 방지
# ==================================================
def test_달력을_갈아_끼우면_새_달력으로_답한다(달력끼우기):
    """🔴 이게 이 파일의 핵심이다.

    정렬본을 갱신하는 책임을 갈아 끼우는 쪽에 맡기면, 여기서 **에러 없이** 옛 달력의
    답이 나온다. 조용히 틀리는 종류라 사람이 알아채기 어렵다.
    """
    달력끼우기(달력_가)
    assert tc.next_session("20210112", inclusive=True) == "20210112"

    달력끼우기(달력_나)          # 20210112 가 빠지고 20210115 가 들어온 달력
    assert tc.next_session("20210111") == "20210115"
    assert tc.next_session("20210112", inclusive=True) == "20210115"


def test_크기가_같아도_내용이_다르면_다시_정렬한다(달력끼우기):
    """두 달력은 원소가 8개로 같다. 크기만 보고 판단하면 이 시험이 깨진다."""
    assert len(달력_가) == len(달력_나)
    달력끼우기(달력_가)
    첫답 = tc.next_session("20210108")
    달력끼우기(달력_나)
    둘째답 = tc.next_session("20210108")
    assert 첫답 == "20210111"
    assert 둘째답 == "20210111"      # 여기까진 같고
    assert tc.next_session("20210111") == "20210115"   # 여기서 갈린다


# ==================================================
# 3. 달력 밖은 지어내지 않는다
# ==================================================
def test_달력보다_늦은_날은_세운다(달력끼우기):
    달력끼우기(달력_가)
    with pytest.raises(tc.CalendarOutOfRange):
        tc.next_session("20210112")          # 마지막 거래일 다음은 모른다


def test_달력보다_이른_날은_세운다(달력끼우기):
    달력끼우기(달력_가)
    with pytest.raises(tc.CalendarOutOfRange):
        tc.next_session("20201230")          # 첫 거래일보다 이르다


def test_세울_때_무엇을_해야_하는지까지_알려준다(달력끼우기):
    """막다른 길로 만들지 않는다 — 예외 문구에 할 일이 있어야 한다."""
    달력끼우기(달력_가)
    with pytest.raises(tc.CalendarOutOfRange) as 잡힌것:
        tc.next_session("20210112")
    assert "할 일" in str(잡힌것.value)
