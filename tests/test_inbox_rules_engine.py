"""규칙 해석기가 **실제로 잡는지** 확인한다 — 일부러 어긋난 행을 넣어 본다.

**왜 이 파일이 따로 있는가.** `test_inbox_schemas.py` 는 "규격이 우리 자료를 격리하지
않는다" 를 본다. 그런데 그것만으로는 검사가 도는지 알 수 없다 — 규칙이 늘 참을
돌려줘도 똑같이 통과한다. 여기서는 반대로, **틀린 행을 만들어 넣고 잡히는지** 본다.

특히 규격 note 에 "우리 자료에는 0건이라 한 번도 참이 된 적이 없다" 고 적어 둔
가지들(0 나눗셈 가드 같은 것)은 여기서만 검증된다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ingest.inbox.rules import (  # noqa: E402
    RuleSyntaxError,
    check_alias_invariant,
    evaluate_rule,
    referenced_columns,
)


def 판정(expr: str, **columns) -> list:
    """식 하나를 작은 표에 돌려 통과 여부를 목록으로 돌려준다."""
    frame = pd.DataFrame(columns)
    return evaluate_rule(expr, frame).tolist()


# ==================================================
# 1. 우선순위 — 문자열 치환이 틀렸던 바로 그 자리
# ==================================================
def test_비교와_논리연산의_우선순위가_뜻대로_읽힌다():
    """`a is null or b == 0` 이 `a is null or b` 를 먼저 묶으면 안 된다.

    문자열 치환(`or` → `|`)으로 옮기면 파이썬에서 `|` 가 `==` 보다 먼저 묶여
    `(a | b) == 0` 이 된다. 실제로 이 방식으로 920만 행을 재었더니 892만 행이
    가짜 위반으로 나왔다.
    """
    결과 = 판정(
        "high is null or high == 0 or (high >= open and high >= close)",
        high=[None, 0.0, 10.0, 5.0],
        open=[1.0, 3.0, 2.0, 9.0],
        close=[1.0, 3.0, 3.0, 1.0],
    )
    # 널 통과 · 0 통과 · 10>=2,3 통과 · 5<9 위반
    assert 결과 == [True, True, True, False]


def test_and_가_or_보다_먼저_묶인다():
    결과 = 판정("a == 1 or b == 1 and c == 1", a=[1, 0, 0], b=[0, 1, 1], c=[0, 0, 1])
    # a==1 → 참 / (b and c) 가 거짓 → 거짓 / 참
    assert 결과 == [True, False, True]


# ==================================================
# 2. 널 가드가 없으면 실제로 격리된다 (가드의 필요성 증명)
# ==================================================
def test_널_가드가_없으면_빈_값이_위반으로_잡힌다():
    """가드 없는 비교가 왜 위험한지 보인다 — 이것이 규격 초안의 결함이었다."""
    가드없음 = 판정("high >= low", high=[None, 3.0], low=[1.0, 1.0])
    assert 가드없음 == [False, True], "널이 위반으로 잡히는 것이 문제의 정체다"

    가드있음 = 판정("high is null or low is null or high >= low",
                    high=[None, 3.0], low=[1.0, 1.0])
    assert 가드있음 == [True, True]


# ==================================================
# 3. 규격 note 가 "한 번도 참이 된 적 없다" 고 적은 가지
# ==================================================
def test_전일종가가_0이면_나눗셈을_피한다():
    """지수 change_rate 검산의 0 나눗셈 가드.

    우리 자료에 `close - change == 0` 인 행이 0건이라 이 가지는 실제로 돈 적이 없다.
    규격 note 에 그렇게 적어 두었고, 여기서 만들어 넣어 확인한다.
    """
    expr = ("close is null or change is null or change_rate is null "
            "or (close - change) == 0 "
            "or abs(change_rate - change / (close - change) * 100) <= 0.05")
    결과 = 판정(expr, close=[5.0, 100.0], change=[5.0, 10.0], change_rate=[999.0, 11.11])
    # 첫 행은 전일종가 0 이라 가드로 통과해야 한다 (등락률이 말도 안 되는 값이어도)
    assert 결과[0] is True or 결과[0] == True    # noqa: E712 — numpy bool
    assert 결과[1] == True                        # noqa: E712 — 100/(100-10)... 11.11 ≈ 11.11


def test_한밤중_기사는_당일_배정을_통과하지_못한다():
    """뉴스 규격의 00:00 구멍. 가드가 없으면 00:00 < 08:30 이라 통과한다."""
    expr = ("eff_dd is null or pub_dt is null or eff_dd > date(pub_dt) "
            "or (time(pub_dt) != '000000' and time(pub_dt) < '083000')")
    결과 = 판정(
        expr,
        pub_dt=["2026-09-01 00:00:00", "2026-09-01 07:00:00", "2026-09-01 14:00:00"],
        eff_dd=["20260901", "20260901", "20260901"],
    )
    assert 결과 == [False, True, False], "0시 기사와 장중 기사는 당일 배정이 안 된다"

    가드없음 = ("eff_dd is null or pub_dt is null or eff_dd > date(pub_dt) "
                "or time(pub_dt) < '083000'")
    새는판정 = 판정(가드없음,
                    pub_dt=["2026-09-01 00:00:00"], eff_dd=["20260901"])
    assert 새는판정 == [True], "가드가 없으면 0시 기사가 통과한다 — 이것이 막으려던 구멍이다"


# ==================================================
# 4. 함수
# ==================================================
def test_날짜_조각을_문자열에서_뽑는다():
    assert 판정("year(d) == 2026", d=["20260901"]) == [True]
    assert 판정("month(d) == 9", d=["20260901"]) == [True]
    assert 판정("day(d) == 1", d=["20260901"]) == [True]


def test_분기_시작월만_통과한다():
    expr = ("freq != 'Q' or period_start is null or (day(period_start) == 1 "
            "and (month(period_start) == 1 or month(period_start) == 4 "
            "or month(period_start) == 7 or month(period_start) == 10))")
    결과 = 판정(expr,
                freq=["Q", "Q", "Q", "M"],
                period_start=["20260401", "20260501", "20260415", "20260501"])
    assert 결과 == [True, False, False, True]


def test_접수번호_앞자리가_접수일과_맞는지_본다():
    expr = "rcept_dt is null or rcept_no is null or starts_with(rcept_no, rcept_dt)"
    결과 = 판정(expr,
                rcept_dt=["20260901", "20260901", None],
                rcept_no=["20260901000001", "20260830000001", "20260901000001"])
    assert 결과 == [True, False, True]


def test_정규식과_길이와_포함검사():
    assert 판정("matches(p, '^[0-9]{4}Q[1-4]$')", p=["2026Q3", "2026-Q3"]) == [True, False]
    assert 판정("len(s) <= 5", s=["abc", "abcdef"]) == [True, False]
    assert 판정("not contains(t, '<b>')", t=["보통", "<b>강조</b>"]) == [True, False]


def test_요일을_본다():
    # 2026-09-01 은 화요일, 2026-09-05 는 토요일
    assert 판정("bas_dd is weekday", bas_dd=["20260901", "20260905"]) == [True, False]


# ==================================================
# 5. 문법을 벗어나면 조용히 넘기지 않고 세운다
# ==================================================
def test_허용하지_않은_함수는_거절한다():
    with pytest.raises(RuleSyntaxError, match="허용되지 않은 함수"):
        판정("eval('1')", a=[1])


def test_표에_없는_칸은_거절한다():
    with pytest.raises(RuleSyntaxError, match="표에 없는 칸"):
        판정("없는칸 > 0", a=[1])


def test_is_오른쪽에_이상한_이름이_오면_거절한다():
    with pytest.raises(RuleSyntaxError, match="`is` 오른쪽"):
        판정("a is 무언가", a=[1])


def test_파싱되지_않는_식은_세운다():
    with pytest.raises(RuleSyntaxError, match="파싱할 수 없다"):
        판정("a >>> b", a=[1], b=[1])


# ==================================================
# 6. 도우미
# ==================================================
def test_참조하는_칸을_모은다():
    assert referenced_columns("high is null or high >= low") == {"high", "low"}
    assert "null" not in referenced_columns("a is null")
    assert "abs" not in referenced_columns("abs(a) > 1")


def test_별칭이_겹치면_규격이_틀렸다고_세운다():
    나쁜규격 = {"x-alphastack": {"aliases": {"a": ["당기"], "b": ["당기"]}}}
    with pytest.raises(RuleSyntaxError, match="양쪽에 있다"):
        check_alias_invariant(나쁜규격)

    좋은규격 = {"x-alphastack": {"aliases": {"a": ["당기"], "b": ["전기"]}}}
    check_alias_invariant(좋은규격)      # 예외가 없으면 통과
