"""`ecos_data.known_at` — 거시지표를 **언제부터 알 수 있었나**의 시험대.

이 함수가 틀리면 예외가 나지 않고 성능만 올라간다. 그래서 여기서 막는다.

ECOS 는 월별 값을 **기준월 1일**로 준다 — 2026년 7월 물가가 `2026-07-01` 이다.
그대로 붙이면 7월 물가를 7월 1일에 아는 셈인데, 실제 발표는 8월 4일이었다.
경기지수는 더 벌어져서 7월분이 8월 31일에 나온다.

🔴 **항등식으로 시험하지 않는다.** 규칙으로 만든 값을 규칙으로 다시 만들어 비교하면
언제나 통과한다. 그래서 아래 `실제_공표일` 은 **바깥에서 조사한 사실**이다 —
국가데이터처·한국은행 공표일정에서 2026-09-02 에 옮겨 적었고, 코드와 무관하게 고정이다.
우리 값이 그보다 **늦기만 하면** 미래는 새지 않는다.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from ingest.clients.ecos_data import EcosError, known_at

# 공식 공표일정에서 옮겨 적은 **실제 발표일**. (지표, 기준월, 실제 공표일)
#   CPI  국가데이터처 공표일정
#   PPI  한국은행 통계공표일정 (06:00 공표)
#   경기 산업활동동향 — 2026년 7월분이 8.31 발표
실제_공표일 = [
    ("cpi", "202601", "20260203"),
    ("cpi", "202602", "20260306"),
    ("cpi", "202603", "20260402"),
    ("cpi", "202604", "20260506"),
    ("cpi", "202605", "20260602"),
    ("cpi", "202606", "20260702"),
    ("cpi", "202607", "20260804"),
    ("cpi", "202608", "20260902"),
    # 12월분은 연간과 함께 당월 말일에 나온다 — 유일하게 익월을 넘지 않는다.
    ("cpi", "202612", "20261231"),
    ("ppi", "202512", "20260120"),
    ("ppi", "202601", "20260224"),
    ("ppi", "202602", "20260324"),
    ("ppi", "202603", "20260422"),
    ("ppi", "202604", "20260521"),
    ("ppi", "202605", "20260619"),
    ("ppi", "202606", "20260722"),
    ("ppi", "202607", "20260821"),
    ("ppi", "202608", "20260918"),
    ("leading", "202607", "20260831"),
    ("coincident", "202607", "20260831"),
]


@pytest.mark.parametrize("지표, 기준월, 공표일", 실제_공표일)
def test_실제_공표일보다_늦게_잡는다(지표, 기준월, 공표일):
    """우리가 정한 시점이 실제 발표보다 앞서면 그만큼 미래가 학습에 샌다."""
    우리값 = known_at(지표, 기준월)
    벌어진일수 = (
        datetime.strptime(우리값, "%Y%m%d") - datetime.strptime(공표일, "%Y%m%d")
    ).days
    assert 벌어진일수 >= 0, (
        f"{지표} {기준월}: 우리는 {우리값} 부터 안다고 보는데 실제 발표는 {공표일} 이다. "
        f"{-벌어진일수}일치 미래가 샌다."
    )


def test_기준월을_그대로_쓰는_것보다_늦다():
    """ECOS 가 주는 기준월 1일을 그냥 붙이면 한 달~두 달치 미래가 들어간다."""
    막은일수 = {}
    for 지표 in ("base_rate", "cpi", "ppi", "leading", "coincident"):
        우리값 = datetime.strptime(known_at(지표, "202607"), "%Y%m%d")
        기준월그대로 = datetime(2026, 7, 1)
        막은일수[지표] = (우리값 - 기준월그대로).days

    # 한 달 미만이면 규칙이 사실상 없는 것과 같다.
    assert min(막은일수.values()) >= 28, 막은일수
    # 경기지수는 익월 말에 나오므로 두 달 가까이 막아야 한다.
    assert 막은일수["leading"] >= 60, 막은일수


def test_월별은_달을_넘겨도_어긋나지_않는다():
    assert known_at("cpi", "202612") == "20270110"
    assert known_at("base_rate", "202612") == "20270101"
    assert known_at("leading", "202611") == "20270105"
    assert known_at("leading", "202612") == "20270205"


def test_말일_규칙은_달마다_길이가_다르다():
    """`day="end"` 는 표를 두지 않고 계산한다 — 윤년을 손으로 관리하면 언젠가 틀린다."""
    assert known_at("ppi", "202601") == "20260228"      # 평년 2월
    assert known_at("ppi", "202401") == "20240229"      # 윤년 2월
    assert known_at("ppi", "202603") == "20260430"      # 30일 달
    assert known_at("ppi", "202611") == "20261231"      # 12월
    assert known_at("ppi", "202612") == "20270131"      # 해를 넘긴다


def test_일별은_하루를_더한다():
    assert known_at("usdkrw", "20260901") == "20260902"
    assert known_at("ktb3y", "20261231") == "20270101"   # 해 넘김
    assert known_at("ktb10y", "20240228") == "20240229"  # 윤년


def test_없는_지표는_무엇을_해야_하는지_알려준다():
    with pytest.raises(EcosError) as 잡힘:
        known_at("gdp", "202607")
    assert "없는 지표" in str(잡힘.value)
    # 쓸 수 있는 것을 함께 알려 줘야 막다른 길이 되지 않는다.
    assert "cpi" in str(잡힘.value)


def test_기간_형식이_어긋나면_조용히_넘어가지_않는다():
    """월별에 일별 형식을 주면 조용히 이상한 날짜를 만들지 말고 세워야 한다."""
    with pytest.raises(EcosError):
        known_at("cpi", "20260701")        # 월별인데 8자리
    with pytest.raises(EcosError):
        known_at("usdkrw", "202607")       # 일별인데 6자리
    with pytest.raises(EcosError):
        known_at("cpi", "2026-07")         # 숫자가 아니다


def test_아홉_지표_전부_규칙을_갖는다():
    """지표를 늘리고 규칙을 안 넣으면 그 지표만 조용히 시점 없이 담긴다."""
    from ingest.clients.ecos_data import INDICATORS, RELEASE_RULES

    빠진것 = [spec["id"] for spec in INDICATORS if spec["id"] not in RELEASE_RULES]
    assert not 빠진것, f"공표 시차 규칙이 없는 지표: {빠진것}"
