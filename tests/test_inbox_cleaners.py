"""정제기 20종이 **무엇을 무엇으로 바꾸는지** 하나씩 확인한다.

정제는 값을 고치는 일이라 조용히 틀리기 쉽다. `strip_percent` 가 값을 100 으로 나눠 버리거나
`zfill6` 이 ISIN 코드에 0 을 붙이면, 그 자료는 규격을 통과하고 학습까지 들어간 뒤에야 이상해
보인다. 그래서 **바꿔야 하는 것뿐 아니라 건드리면 안 되는 것도** 함께 재 둔다.

기록도 함께 본다. 정제기가 값만 고치고 기록을 안 남기면 나중에 *"출처가 그랬는지 우리가
그랬는지"* 를 되짚을 수 없다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ingest.inbox.cleaners import (  # noqa: E402
    CLEANERS,
    SAMPLE_LIMIT,
    CleanerError,
    apply_chain,
    normalize_missing,
)


def 돌린다(cleaner: str, values: list) -> list:
    """정제기 하나를 값 목록에 걸어 결과만 돌려준다."""
    outcome = CLEANERS[cleaner](pd.Series(values, dtype="object"))
    return outcome.values.tolist()


def 실패수(cleaner: str, values: list) -> int:
    return int(CLEANERS[cleaner](pd.Series(values, dtype="object")).failed.sum())


def 값들(series: pd.Series) -> list:
    """결측을 전부 None 으로 눕혀 비교한다.

    pandas 는 결측을 `None`·`nan`·`pd.NA` 세 가지로 담는데(dtype 에 따라 다르다) 셋 다
    "값이 없다" 는 같은 뜻이다. 테스트가 그 차이를 붙들면 dtype 이 바뀔 때마다 깨진다.
    """
    return [None if pd.isna(v) else v for v in series.tolist()]


# ==================================================
# 1. 문자열 다듬기
# ==================================================
def test_strip_은_앞뒤_공백만_뗀다():
    assert 돌린다("strip", ["  삼성전자 ", "\tSK\n"]) == ["삼성전자", "SK"]
    # 가운데 공백은 이름의 일부다 — "코스피 200" 은 "코스피200" 과 다른 지수다.
    assert 돌린다("strip", [" 코스피 200 "]) == ["코스피 200"]


def test_collapse_space_는_연속_공백만_줄인다():
    assert 돌린다("collapse_space", ["매출액    합계", "가\t나"]) == ["매출액 합계", "가 나"]


def test_strip_comma_는_천단위만_뗀다():
    assert 돌린다("strip_comma", ["1,234,567", "38,655,276"]) == ["1234567", "38655276"]


def test_strip_percent_는_기호만_떼고_100으로_나누지_않는다():
    # 🔴 여기서 나누면 규격의 minimum=-100 범위와 어긋나고, 등락률이 전부 100배 작아진다.
    assert 돌린다("strip_percent", ["2.47%", "-1.05%"]) == ["2.47", "-1.05"]


def test_paren_to_negative_는_회계_괄호를_음수로_바꾼다():
    assert 돌린다("paren_to_negative", ["(1234)", "1234"]) == ["-1234", "1234"]


def test_paren_to_negative_는_이미_부호가_있으면_건드리지_않는다():
    # "(-5)" 를 "--5" 로 만들면 숫자로 못 읽는다. 빈 괄호도 회계 표기가 아니다.
    assert 돌린다("paren_to_negative", ["(-5)", "()", "(+3)"]) == ["(-5)", "()", "(+3)"]


def test_zfill6_은_숫자일_때만_채운다():
    assert 돌린다("zfill6", ["5930", "660"]) == ["005930", "000660"]
    # 🔴 ISIN 에 0 을 붙이면 없던 코드가 생긴다. 이미 여섯 자리면 그대로 둔다.
    assert 돌린다("zfill6", ["KR7005930003", "005930"]) == ["KR7005930003", "005930"]


def test_zfill8_은_DART_고유번호를_되살린다():
    assert 돌린다("zfill8", ["126380", "00126380"]) == ["00126380", "00126380"]


def test_drop_suffix_ks_kq_는_야후_꼬리표만_뗀다():
    assert 돌린다("drop_suffix_ks_kq", ["005930.KS", "035720.kq", "123456.KN"]) == [
        "005930", "035720", "123456"]
    # 종목명에 든 마침표까지 지우면 안 된다.
    assert 돌린다("drop_suffix_ks_kq", ["한국.전력"]) == ["한국.전력"]


def test_upper_는_대문자로_올린다():
    assert 돌린다("upper", ["cfs", "ofs"]) == ["CFS", "OFS"]


# ==================================================
# 2. 숫자
# ==================================================
def test_to_int_는_소수부가_있으면_실패로_둔다():
    # 🔴 버림도 반올림도 하지 않는다. 거래량에 소수가 붙어 왔다면 고칠 값이 아니라 볼 값이다.
    outcome = CLEANERS["to_int"](pd.Series(["1234", "12.5"], dtype="object"))
    assert 값들(outcome.values) == [1234, None]
    assert outcome.failed.tolist() == [False, True]


def test_to_int_는_못_읽은_값과_원래_결측을_구별한다():
    outcome = CLEANERS["to_int"](pd.Series(["일이삼", None], dtype="object"))
    assert outcome.failed.tolist() == [True, False], "빈 칸은 실패가 아니다"


def test_to_float_는_소수를_그대로_읽는다():
    assert 돌린다("to_float", ["1.5", "-0.25"]) == [1.5, -0.25]


# ==================================================
# 3. 날짜·시각
# ==================================================
@pytest.mark.parametrize("given", [
    "20210104", "2021-01-04", "2021/01/04", "2021.01.04", "2021년 1월 4일",
    "2021-01-04 09:30:00",
])
def test_to_yyyymmdd_는_여러_모양을_한_모양으로_모은다(given):
    assert 돌린다("to_yyyymmdd", [given]) == ["20210104"]


def test_to_yyyymmdd_는_못_읽으면_실패로_남긴다():
    assert 실패수("to_yyyymmdd", ["어제"]) == 1


def test_to_kst_iso_는_오프셋을_읽어_옮긴다():
    # 🔴 이게 이 정제기의 존재 이유다. UTC 23:45 를 KST 로 착각하면 거래일이 하루 어긋나고,
    #    그 방향이 하필 미래를 당겨 보는 쪽이다.
    assert 돌린다("to_kst_iso", ["2021-01-03T23:45:00Z"]) == ["2021-01-04T08:45:00+09:00"]


def test_to_kst_iso_는_오프셋이_없으면_KST_로_본다():
    assert 돌린다("to_kst_iso", ["2021-01-04 09:30:00"]) == ["2021-01-04T09:30:00+09:00"]


def test_rfc1123_to_kst_는_요일이_붙은_RSS_시각을_읽는다():
    assert 돌린다("rfc1123_to_kst", ["Mon, 04 Jan 2021 09:30:00 +0900"]) == [
        "2021-01-04T09:30:00+09:00"]


# ==================================================
# 4. 텍스트
# ==================================================
def test_strip_html_tags_는_태그를_공백으로_바꾼다():
    # 🔴 빈 문자열로 바꾸면 "<p>가</p><p>나</p>" 가 "가나" 라는 없던 낱말이 된다.
    assert 돌린다("strip_html_tags", ["<p>가</p><p>나</p>"]) == [" 가  나 "]


def test_unescape_html_은_실체_참조를_되돌린다():
    assert 돌린다("unescape_html", ["삼성&amp;LG", "&quot;인용&quot;"]) == [
        '삼성&LG', '"인용"']


def test_규격이_정한_순서는_태그제거가_먼저다():
    """`unescape_html` 이 먼저 오면 `&lt;b&gt;` 가 진짜 태그가 되어 지워진다."""
    values, _, _, _ = apply_chain(
        pd.Series(["&lt;b&gt;진짜꺾쇠"]), ["strip_html_tags", "unescape_html"], column="title")
    assert values.tolist() == ["<b>진짜꺾쇠"], "사용자가 쓴 꺾쇠가 살아남아야 한다"

    뒤집힌, _, _, _ = apply_chain(
        pd.Series(["&lt;b&gt;진짜꺾쇠"]), ["unescape_html", "strip_html_tags"], column="title")
    # 뒤집으면 꺾쇠가 사라진다 — 그래서 규격이 순서를 정한다.
    assert 뒤집힌.tolist() == [" 진짜꺾쇠"]


def test_drop_tracking_params_는_추적_인자만_뗀다():
    assert 돌린다("drop_tracking_params", ["https://n.news/1?utm_source=naver"]) == [
        "https://n.news/1"]
    # 🔴 기사 번호가 질의 인자에 든 언론사가 실재한다. 다 지우면 목록 페이지를 가리킨다.
    assert 돌린다("drop_tracking_params", ["https://n.news/2?article_id=99&utm_medium=x"]) == [
        "https://n.news/2?article_id=99"]


# ==================================================
# 5. 값 맞추기
# ==================================================
def test_map_market_는_시장_이름을_규격_enum_으로_모은다():
    assert 돌린다("map_market", ["코스피", "유가증권", "KOSPI", "코스닥"]) == [
        "KOSPI", "KOSPI", "KOSPI", "KOSDAQ"]


def test_map_freq_는_주기를_한_글자로_모은다():
    assert 돌린다("map_freq", ["월", "monthly", "분기", "연간"]) == ["M", "M", "Q", "A"]


def test_map_source_는_출처를_규격_enum_으로_모은다():
    assert 돌린다("map_source", ["한국은행", "fred", "통계청"]) == ["ECOS", "FRED", "KOSIS"]


def test_모르는_값은_억지로_끼워_맞추지_않는다():
    """모르는 값이 왔다는 사실 자체가 알아야 할 정보다 — 추측하면 그 사실이 사라진다."""
    assert 돌린다("map_market", ["NASDAQ"]) == ["NASDAQ"], "규격의 enum 검사가 잡게 둔다"


# ==================================================
# 6. 기록
# ==================================================
def test_바꾼_것을_전부_세고_표본을_남긴다():
    values, log, failed, changed = apply_chain(
        pd.Series(["5930", "660", "005930"]), ["zfill6"], column="code")
    entry = log.entries[0]
    assert entry.changed == 2
    assert entry.samples == [("5930", "005930"), ("660", "000660")]
    assert changed.tolist() == [True, True, False]


def test_그릇만_바뀐_것은_변경으로_세지_않는다():
    """`"1234"` → `1234` 는 담는 형이 바뀐 것이지 값이 바뀐 것이 아니다.

    이걸 변경으로 세면 숫자 칸은 거의 모든 행이 변경이 되고, 표본 20칸이 `1234 → 1234`
    같은 것으로 다 차서 정작 봐야 할 `"1.5" → <NA>` 가 밀려난다.
    """
    _, log, _, _ = apply_chain(pd.Series(["1234", "5678"]), ["to_int"], column="volume")
    assert log.entries[0].changed == 0


def test_표본은_스무_건까지만_남긴다():
    """기록이 원본보다 커지지 않게 한다 — Great Expectations 의 기본값과 같은 선이다."""
    _, log, _, _ = apply_chain(
        pd.Series([str(i) for i in range(100)]), ["zfill6"], column="code")
    assert log.entries[0].changed == 100, "세는 것은 전량이다"
    assert len(log.entries[0].samples) == SAMPLE_LIMIT, "싣는 것은 20건이다"


def test_아무것도_안_한_정제기는_보고서에_싣지_않는다():
    _, log, _, _ = apply_chain(pd.Series(["005930"]), ["strip", "zfill6"], column="code")
    assert log.to_list() == [], "300줄짜리 무변화 목록에 정작 바뀐 것이 묻힌다"


def test_모르는_정제기_이름은_세운다():
    """규격의 오타를 조용히 넘기면 그 칸만 정제 없이 통과한다."""
    with pytest.raises(CleanerError, match="없는정제기"):
        apply_chain(pd.Series(["x"]), ["없는정제기"], column="code")


# ==================================================
# 7. 결측 표기
# ==================================================
def test_normalize_missing_은_출처의_결측_표기를_결측으로_바꾼다():
    out = normalize_missing(pd.Series(["-", "N/A", "83000"]), ["-", "N/A"])
    assert 값들(out) == [None, None, "83000"]


def test_normalize_missing_은_대소문자를_구별한다():
    """규격이 `"NA"` 와 `"nan"` 을 따로 적어 둔 이유 — 뭉개면 회사명 `"None"` 이 결측이 된다."""
    out = normalize_missing(pd.Series(["None", "none"]), ["None"])
    assert 값들(out) == [None, "none"]


def test_결측_표기를_먼저_눕히지_않으면_멀쩡한_행이_격리된다():
    """정제 순서가 왜 ③결측 → ④정제 인지를 재는 시험."""
    안_눕힌 = CLEANERS["to_int"](pd.Series(["-"], dtype="object"))
    assert 안_눕힌.failed.tolist() == [True], "출처가 결측을 그렇게 적었을 뿐인데 실패가 된다"

    눕힌 = CLEANERS["to_int"](normalize_missing(pd.Series(["-"]), ["-"]))
    assert 눕힌.failed.tolist() == [False]
