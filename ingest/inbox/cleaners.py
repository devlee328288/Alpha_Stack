"""규격의 `cleaners` 에 이름만 적혀 있던 정제기 20종을 실제로 구현한다.

**무엇을 무엇으로 바꿨는지 남기지 않는 정제는 정제가 아니라 훼손이다.** 팀원이 건네준
파일을 우리가 조용히 고쳐 담으면, 나중에 값이 이상할 때 *출처가 그랬는지 우리가 그랬는지*
알 방법이 없다. 그래서 모든 정제기는 값을 바꾸는 동시에 **바꿨다는 사실을 기록**한다.

기록의 상세도 — 왜 "전량 집계 + 표본 20건" 인가
-----------------------------------------------
Great Expectations 의 기본 결과 형식(`SUMMARY`)이 이 모양이다. 집계
(`element_count`·`missing_count`·`unexpected_count`·`unexpected_percent`)는 **전량**을 세고,
실제 값 목록(`partial_unexpected_list`·`partial_unexpected_counts`)은 **기본 20건**까지만
싣는다. 우리도 같은 선을 쓴다:

- **적용·변경·실패 건수는 전부 센다** — 몇 건이 바뀌었는지는 반올림하지 않는다.
- **before → after 실물은 최대 20건** — 값이 실제로 바뀐 것만 모은다.

전량 기록을 쓰지 않는 이유는 단순하다. 팀원이 몇십만 행짜리 파일을 주면 변경 이력이 원본보다
커진다. 반대로 건수만 남기면 *"code 칸 318건 변경"* 이 `zfill6` 때문인지 엉뚱한 절단 때문인지
사람이 확인할 길이 없다. 20건이면 눈으로 읽고 판단할 수 있다.

실패를 결측과 구별한다
----------------------
🔴 **값이 있었는데 못 읽은 것과 원래 비어 있던 것은 다르다.** `to_int("1,234")` 는 쉼표를
먼저 떼면 성공하지만 `to_int("일이삼")` 은 실패한다. 실패를 그냥 NaN 으로 두면 그 행은
*"거래량이 비어 있는 행"* 이 되어 조용히 통과하거나, required 검사에서 엉뚱한 이유로 걸린다.
그래서 `CleanOutcome.failed` 로 **실패한 자리를 따로 표시**해 부르는 쪽이 그 행을 격리할 수
있게 한다. 규격의 `missingValues` 에 있던 값은 실패가 아니다 — 그건 출처가 결측을 적는 방식이다.

쓰는 법
-------
    from ingest.inbox.cleaners import apply_chain

    values, log, failed, changed = apply_chain(frame["code"], ["strip", "zfill6"], column="code")
    frame["code"] = values
    log.entries[0].changed      # 318
    log.entries[0].samples      # [("5930", "005930"), ...]
    changed.sum()               # 이 칸에 손을 댄 행 수
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple

import pandas as pd

#: before → after 표본을 몇 건까지 남길지. Great Expectations 의
#: `partial_unexpected_list` 기본값과 같은 20 이다.
SAMPLE_LIMIT = 20


class CleanerError(ValueError):
    """규격이 우리가 모르는 정제기 이름을 가리킨다."""


@dataclass
class CleanOutcome:
    """정제기 하나의 결과.

    `values` 는 바뀐 값, `failed` 는 **값이 있었는데 못 읽은 자리**다.
    둘 다 원래 인덱스를 그대로 유지한다.
    """

    values: pd.Series
    failed: pd.Series


def _no_failure(series: pd.Series) -> pd.Series:
    """실패가 있을 수 없는 정제기가 쓰는 빈 실패 표시."""
    return pd.Series(False, index=series.index)


# ==================================================
# 1. 문자열 다듬기
# ==================================================
def _as_string(series: pd.Series) -> pd.Series:
    """문자열로 본다. 결측은 결측으로 둔다 — `"nan"` 이라는 글자로 만들지 않는다."""
    return series.astype("object").where(series.notna(), other=None)


def strip(series: pd.Series) -> CleanOutcome:
    """앞뒤 공백을 뗀다. 엑셀에서 온 값은 거의 항상 이게 붙어 있다."""
    text = _as_string(series)
    out = text.map(lambda v: v.strip() if isinstance(v, str) else v)
    return CleanOutcome(out, _no_failure(series))


def collapse_space(series: pd.Series) -> CleanOutcome:
    """연속된 공백을 하나로 줄인다.

    회사명·계정명이 `"매출액   "` 처럼 칸을 맞추려고 띄어져 오는 일이 잦다. 이걸 두면
    같은 계정이 서로 다른 이름으로 갈라져 열쇠가 어긋난다. 탭·개행도 공백으로 본다.
    """
    text = _as_string(series)
    out = text.map(lambda v: re.sub(r"\s+", " ", v).strip() if isinstance(v, str) else v)
    return CleanOutcome(out, _no_failure(series))


def upper(series: pd.Series) -> CleanOutcome:
    """대문자로 올린다. `cfs` 와 `CFS` 가 다른 값이 되지 않게."""
    text = _as_string(series)
    out = text.map(lambda v: v.upper() if isinstance(v, str) else v)
    return CleanOutcome(out, _no_failure(series))


def strip_comma(series: pd.Series) -> CleanOutcome:
    """천단위 쉼표를 뗀다 — `"1,234,567"` → `"1234567"`.

    ⚠️ 소수점 쉼표(유럽식 `"1,5"`)는 다루지 않는다. 우리가 받는 자료는 KRX·DART·ECOS·
    FRED 이고 넷 다 마침표를 소수점으로 쓴다. 유럽식을 지원하려 들면 `"1,234"` 가 천이백삼십사인지
    일점이삼사인지 값만 보고는 정할 수 없어, 반은 1000배 틀리게 된다.
    """
    text = _as_string(series)
    out = text.map(lambda v: v.replace(",", "") if isinstance(v, str) else v)
    return CleanOutcome(out, _no_failure(series))


def strip_percent(series: pd.Series) -> CleanOutcome:
    """퍼센트 기호를 뗀다 — `"1.23%"` → `"1.23"`.

    **값을 100 으로 나누지 않는다.** 우리 `change_rate` 는 등락률을 퍼센트 숫자 그대로
    담는다(1.23 이 1.23%). 여기서 나누면 규격의 `minimum=-100` 범위와 어긋난다.
    """
    text = _as_string(series)
    out = text.map(lambda v: v.replace("%", "").strip() if isinstance(v, str) else v)
    return CleanOutcome(out, _no_failure(series))


_PAREN_NEGATIVE = re.compile(r"^\((.+)\)$")


def paren_to_negative(series: pd.Series) -> CleanOutcome:
    """회계 표기의 괄호를 음수 부호로 바꾼다 — `"(1,234)"` → `"-1234"`.

    재무제표는 음수를 마이너스가 아니라 괄호로 적는 것이 관행이다. 이걸 모르고 숫자로 읽으면
    **부호가 통째로 뒤집힌다** — 당기순손실이 순이익이 된다. 규격에서 `financial` 의 금액
    세 칸(`thstrm_amount`·`frmtrm_amount`·`bfefrmtrm_amount`)에만 걸어 둔 이유가 이것이다.

    괄호 안이 비었거나(`"()"`) 이미 부호가 있으면(`"(-5)"`) 건드리지 않는다 — 그런 값은
    회계 표기가 아니라 다른 뜻일 수 있고, 추측해서 고치면 되돌릴 수 없다.
    """
    text = _as_string(series)

    def convert(value):
        if not isinstance(value, str):
            return value
        match = _PAREN_NEGATIVE.match(value.strip())
        if match is None:
            return value
        inner = match.group(1).strip()
        if not inner or inner.startswith(("-", "+")):
            return value
        return "-" + inner

    return CleanOutcome(text.map(convert), _no_failure(series))


def _zfill(series: pd.Series, width: int) -> CleanOutcome:
    text = _as_string(series)

    def pad(value):
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        # 숫자로만 이뤄졌고 자리가 모자랄 때만 채운다. `"KR7005930003"` 같은 값에
        # 0 을 붙이면 없던 코드가 생긴다.
        if stripped.isdigit() and len(stripped) < width:
            return stripped.zfill(width)
        return stripped

    return CleanOutcome(text.map(pad), _no_failure(series))


def zfill6(series: pd.Series) -> CleanOutcome:
    """여섯 자리 종목코드를 되살린다 — `"5930"` → `"005930"`.

    엑셀이 앞자리 0 을 지우는 것은 거의 규칙에 가깝다. 이걸 안 되살리면 삼성전자가
    통째로 격리된다(규격의 `pattern` 이 여섯 자리를 요구한다).
    """
    return _zfill(series, 6)


def zfill8(series: pd.Series) -> CleanOutcome:
    """여덟 자리 DART 고유번호를 되살린다 — `"126380"` → `"00126380"`."""
    return _zfill(series, 8)


_TICKER_SUFFIX = re.compile(r"\.(KS|KQ|KN)$", re.IGNORECASE)


def drop_suffix_ks_kq(series: pd.Series) -> CleanOutcome:
    """yfinance 식 꼬리표를 뗀다 — `"005930.KS"` → `"005930"`.

    팀원이 야후 파이낸스에서 받아 오면 이 꼬리가 붙어 온다. 코스닥은 `.KQ`, 코넥스는 `.KN` 이다.
    꼬리를 시장 정보로 살려 두지 않는 이유는 `market` 칸이 따로 있어서다 — 같은 사실을 두 군데
    적으면 언젠가 어긋난다.
    """
    text = _as_string(series)
    out = text.map(lambda v: _TICKER_SUFFIX.sub("", v.strip()) if isinstance(v, str) else v)
    return CleanOutcome(out, _no_failure(series))


# ==================================================
# 2. 숫자로 바꾸기
# ==================================================
def _to_number(series: pd.Series, *, integer: bool) -> CleanOutcome:
    """문자열을 숫자로 읽는다. **못 읽은 자리를 결측과 구별해 표시한다.**"""
    had_value = series.notna()
    if had_value.any() and series.dtype == object:
        # 빈 문자열은 값이 있는 것으로 치지 않는다 — `missingValues` 가 이미 걸렀어야 하지만
        # 정제 도중 생긴 빈 문자열(`"%".replace("%","")`)이 여기까지 올 수 있다.
        had_value = had_value & series.map(lambda v: not (isinstance(v, str) and not v.strip()))

    numbers = pd.to_numeric(series, errors="coerce")
    failed = had_value & numbers.isna()

    if integer:
        # 정수 칸이라도 소수점이 붙어 오는 일이 있다(`"1234.0"`). 반올림이 아니라 버림을
        # 쓰면 `1234.9` 가 1234 가 되어 조용히 틀리므로, **소수부가 있으면 실패로 둔다** —
        # 거래량에 소수가 붙어 왔다면 그건 우리가 고칠 값이 아니라 확인할 값이다.
        fractional = numbers.notna() & (numbers % 1 != 0)
        failed = failed | fractional
        numbers = numbers.where(~fractional)
        numbers = numbers.astype("Int64")

    return CleanOutcome(numbers, failed)


def to_int(series: pd.Series) -> CleanOutcome:
    """정수로 읽는다. 소수부가 붙어 있으면 고치지 않고 실패로 남긴다."""
    return _to_number(series, integer=True)


def to_float(series: pd.Series) -> CleanOutcome:
    """실수로 읽는다."""
    return _to_number(series, integer=False)


# ==================================================
# 3. 날짜·시각
# ==================================================
#: 날짜가 실제로 오는 모양들. **위에서부터 순서대로** 시도한다.
#: 🔴 `%Y%m%d` 를 맨 앞에 두는 이유: pandas 의 자동 추론에 `"20210102"` 를 주면 성공하지만
#:    `"01/02/2021"` 같은 값에서는 월·일 순서를 미국식으로 찍는다. 우리 자료는 한국식이라
#:    **추론에 맡기지 않고 형식을 명시**한다.
_DATE_FORMATS = (
    "%Y%m%d",
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y.%m.%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%Y년 %m월 %d일",
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
)


def _parse_dates(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """여러 형식을 차례로 대 본다. `(파싱된 Timestamp, 값이 있었는데 실패한 자리)`."""
    text = _as_string(series)
    had_value = text.notna() & text.map(lambda v: not (isinstance(v, str) and not v.strip()))

    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    remaining = had_value.copy()

    for fmt in _DATE_FORMATS:
        if not remaining.any():
            break
        subset = text[remaining]
        attempt = pd.to_datetime(subset, format=fmt, errors="coerce")
        got = attempt.notna()
        if got.any():
            parsed.loc[attempt.index[got]] = attempt[got]
            remaining.loc[attempt.index[got]] = False

    return parsed, remaining


def to_yyyymmdd(series: pd.Series) -> CleanOutcome:
    """어떤 모양으로 왔든 여덟 자리 `YYYYMMDD` 문자열로 통일한다.

    저장을 문자열로 하는 이유는 우리 `daily_price.bas_dd` 가 이미 그 모양이기 때문이다.
    여기서 날짜형으로 바꿔 두면 조인할 때마다 형변환이 끼고, 그 형변환은 시간대를 붙였다 뗐다
    하다가 하루씩 어긋난다.
    """
    parsed, failed = _parse_dates(series)
    out = parsed.dt.strftime("%Y%m%d").astype("object").where(parsed.notna(), other=None)
    return CleanOutcome(out, failed)


def to_kst_iso(series: pd.Series) -> CleanOutcome:
    """시각을 KST 로 옮겨 ISO 문자열로 만든다 — `"2024-01-02T09:30:00+09:00"`.

    🔴 **오프셋을 읽어 변환하지, `+0900` 이라고 가정하지 않는다.** 뉴스 규격이 이걸 못 박은
    이유는 `eff_dd` 배정이 08:30 을 경계로 갈리기 때문이다 — UTC 로 온 23:45 을 KST 로 착각하면
    거래일이 하루 어긋나고, 그 방향이 하필 **미래를 당겨 보는 쪽**이다.

    시간대가 안 붙어 온 값은 KST 로 본다. 우리가 받는 국내 자료의 관행이고, 이 가정이 틀리면
    `pub_dt` 규칙이 잡아낸다.
    """
    text = _as_string(series)
    had_value = text.notna() & text.map(lambda v: not (isinstance(v, str) and not v.strip()))

    # utc=True 로 읽으면 오프셋이 섞여 있어도 한 축으로 모인다. 오프셋이 없는 값은
    # UTC 로 읽히므로 그 자리만 따로 KST 로 붙여 준다.
    aware = pd.to_datetime(text, errors="coerce", utc=True, format="mixed")
    offset_tail = re.compile(r"(Z|[+-]\d{2}:?\d{2})$")
    has_offset = text.map(
        lambda v: bool(offset_tail.search(v.strip())) if isinstance(v, str) else False
    )

    naive = pd.to_datetime(text.where(~has_offset), errors="coerce", format="mixed")
    localized = naive.dt.tz_localize("Asia/Seoul", ambiguous="NaT", nonexistent="NaT")

    combined = aware.dt.tz_convert("Asia/Seoul")
    combined = combined.where(has_offset, localized)

    failed = had_value & combined.isna()
    out = combined.dt.strftime("%Y-%m-%dT%H:%M:%S%z").astype("object")
    out = out.where(combined.notna(), other=None)
    # `+0900` 을 `+09:00` 으로 — ISO 8601 은 콜론을 넣는다.
    out = out.map(lambda v: v[:-2] + ":" + v[-2:] if isinstance(v, str) else v)
    return CleanOutcome(out, failed)


def rfc1123_to_kst(series: pd.Series) -> CleanOutcome:
    """RSS 의 `pubDate` 를 KST ISO 로 옮긴다.

    네이버 뉴스 API 가 `"Mon, 02 Jan 2024 09:30:00 +0900"` 모양으로 준다. 이건 RFC 1123 이라
    일반 파서로도 읽히지만, **요일 이름이 로케일을 탄다** — 시스템 로케일이 한국어면 `"Mon"` 을
    못 읽는 파서가 있다. 그래서 요일을 먼저 떼고 나머지를 읽는다.
    """
    text = _as_string(series)
    without_weekday = text.map(
        lambda v: re.sub(r"^[A-Za-z]{3},\s*", "", v.strip()) if isinstance(v, str) else v
    )
    return to_kst_iso(without_weekday)


# ==================================================
# 4. 텍스트 씻기
# ==================================================
_HTML_TAG = re.compile(r"<[^>]+>")


def strip_html_tags(series: pd.Series) -> CleanOutcome:
    """태그를 뗀다 — 검색 API 가 `<b>` 로 감싼 하이라이트가 그대로 온다.

    태그를 빈 문자열이 아니라 **공백으로 바꾼다.** `"삼성<b>전자</b>"` 는 붙여도 되지만
    `"<p>가</p><p>나</p>"` 를 붙이면 `"가나"` 라는 없던 단어가 생긴다. 뒤에 `collapse_space` 를
    함께 걸어 두면 늘어난 공백은 정리된다.
    """
    text = _as_string(series)
    out = text.map(lambda v: _HTML_TAG.sub(" ", v) if isinstance(v, str) else v)
    return CleanOutcome(out, _no_failure(series))


def unescape_html(series: pd.Series) -> CleanOutcome:
    """실체 참조를 되돌린다 — `"&amp;"` → `"&"`, `"&quot;"` → `'"'`.

    ⚠️ **`strip_html_tags` 보다 뒤에 온다.** 순서를 뒤집으면 `"&lt;b&gt;"` 가 먼저 `"<b>"` 로
    풀리고, 그 다음 태그 제거기가 그것을 진짜 태그로 착각해 지운다 — 사용자가 쓴 꺾쇠가 사라진다.
    """
    text = _as_string(series)
    out = text.map(lambda v: html.unescape(v) if isinstance(v, str) else v)
    return CleanOutcome(out, _no_failure(series))


#: 추적용 질의 인자. 값이 달라도 같은 기사를 가리키므로 열쇠에서 빼야 한다.
_TRACKING_PREFIXES = ("utm_", "fbclid", "gclid", "igshid", "spm", "ref_src", "ref_url")


def drop_tracking_params(series: pd.Series) -> CleanOutcome:
    """URL 의 추적 인자를 뗀다.

    뉴스 규격의 열쇠가 `link` 하나라 이게 중요하다. `?utm_source=naver` 가 붙은 것과 안 붙은
    것이 **다른 기사로 세어지면** 같은 기사가 두 번 들어온다. 인자를 다 지우지 않고 추적용만
    지우는 이유는, 기사 번호가 질의 인자에 들어 있는 언론사가 실재하기 때문이다
    (`?article_id=...`) — 그걸 지우면 링크가 목록 페이지를 가리키게 된다.
    """
    text = _as_string(series)

    def clean(value):
        if not isinstance(value, str) or "?" not in value:
            return value
        head, _, query = value.partition("?")
        query, sep, fragment = query.partition("#")
        kept = [
            part for part in query.split("&")
            if part and not part.split("=")[0].lower().startswith(_TRACKING_PREFIXES)
        ]
        rebuilt = head + ("?" + "&".join(kept) if kept else "")
        return rebuilt + (sep + fragment if sep else "")

    return CleanOutcome(text.map(clean), _no_failure(series))


# ==================================================
# 5. 값 맞추기 — 규격의 enum 으로
# ==================================================
def _build_map(pairs: Dict[str, Sequence[str]]) -> Dict[str, str]:
    """`{정답: [변형들]}` 을 `{변형(소문자): 정답}` 으로 뒤집는다."""
    table: Dict[str, str] = {}
    for canonical, variants in pairs.items():
        table[canonical.lower()] = canonical
        for variant in variants:
            table[variant.lower()] = canonical
    return table


#: 시장 이름의 변형. `ohlcv_stock.market` (KOSPI·KOSDAQ·KONEX) 과
#: `ohlcv_index.index_class` (KOSPI·KOSDAQ) 이 함께 쓴다.
MARKET_MAP = _build_map({
    "KOSPI": ["코스피", "유가증권", "유가증권시장", "STK", "KRX", "KOSPI Market"],
    "KOSDAQ": ["코스닥", "코스닥시장", "KSQ", "KOSDAQ Market"],
    "KONEX": ["코넥스", "코넥스시장", "KNX"],
})

#: 주기 표기의 변형. 규격 `macro.freq` 의 enum 여섯 개로 모은다.
FREQ_MAP = _build_map({
    "D": ["일", "일별", "daily", "DD", "day"],
    "W": ["주", "주별", "주간", "weekly", "WW", "week"],
    "M": ["월", "월별", "monthly", "MM", "month"],
    "Q": ["분기", "분기별", "quarterly", "QQ", "quarter"],
    "H": ["반기", "반기별", "half", "half-yearly", "HH", "semiannual"],
    "A": ["년", "연", "연간", "annual", "yearly", "YY", "Y", "year"],
})

#: 출처 표기의 변형. 지연 추정이 출처마다 다르므로 이 칸이 어긋나면 `known_from` 이 틀린다.
SOURCE_MAP = _build_map({
    "ECOS": ["한국은행", "한은", "BOK", "ecos"],
    "FRED": ["연준", "FRB", "Federal Reserve", "St. Louis Fed", "fred"],
    "KOSIS": ["통계청", "KOSTAT", "국가통계포털", "kosis"],
})


def _map_values(series: pd.Series, table: Dict[str, str]) -> CleanOutcome:
    """표에 있는 변형만 바꾼다. **모르는 값은 건드리지 않는다.**

    억지로 끼워 맞추지 않는 이유는, 모르는 값이 왔다는 사실 자체가 알아야 할 정보이기
    때문이다. 그대로 두면 규격의 `enum` 검사가 그 행을 격리하고 보고서에 값이 남는다.
    추측해서 고치면 그 행은 조용히 통과하고, 무엇이 왔었는지는 영영 알 수 없다.
    """
    text = _as_string(series)
    out = text.map(
        lambda v: table.get(v.strip().lower(), v.strip()) if isinstance(v, str) else v
    )
    return CleanOutcome(out, _no_failure(series))


def map_market(series: pd.Series) -> CleanOutcome:
    """시장 이름을 규격의 enum 으로 맞춘다 — `"코스피"` → `"KOSPI"`."""
    return _map_values(series, MARKET_MAP)


def map_freq(series: pd.Series) -> CleanOutcome:
    """주기를 한 글자로 맞춘다 — `"월별"` → `"M"`."""
    return _map_values(series, FREQ_MAP)


def map_source(series: pd.Series) -> CleanOutcome:
    """출처를 규격의 enum 으로 맞춘다 — `"한국은행"` → `"ECOS"`."""
    return _map_values(series, SOURCE_MAP)


# ==================================================
# 6. 등록부
# ==================================================
#: 규격의 `cleaners` 가 이름으로 가리키는 것들. **여기 없는 이름은 세운다** —
#: 오타를 조용히 넘기면 그 칸만 정제 없이 통과해, 나중에 규격만 보고는 알 수 없다.
CLEANERS: Dict[str, Callable[[pd.Series], CleanOutcome]] = {
    "strip": strip,
    "collapse_space": collapse_space,
    "upper": upper,
    "strip_comma": strip_comma,
    "strip_percent": strip_percent,
    "paren_to_negative": paren_to_negative,
    "zfill6": zfill6,
    "zfill8": zfill8,
    "drop_suffix_ks_kq": drop_suffix_ks_kq,
    "to_int": to_int,
    "to_float": to_float,
    "to_yyyymmdd": to_yyyymmdd,
    "to_kst_iso": to_kst_iso,
    "rfc1123_to_kst": rfc1123_to_kst,
    "strip_html_tags": strip_html_tags,
    "unescape_html": unescape_html,
    "drop_tracking_params": drop_tracking_params,
    "map_market": map_market,
    "map_freq": map_freq,
    "map_source": map_source,
}


# ==================================================
# 7. 적용과 기록
# ==================================================
@dataclass
class CleanerEntry:
    """정제기 하나가 한 칸에 무엇을 했나."""

    column: str
    cleaner: str
    applied: int                                   # 값이 있어서 손을 댄 자리 수
    changed: int                                   # 그중 실제로 값이 달라진 수
    failed: int                                    # 값이 있었는데 못 읽은 수
    samples: List[Tuple[str, str]] = field(default_factory=list)   # before → after (최대 20)
    failed_samples: List[str] = field(default_factory=list)        # 못 읽은 값 (최대 20)

    def to_dict(self) -> dict:
        return {
            "column": self.column,
            "cleaner": self.cleaner,
            "applied": self.applied,
            "changed": self.changed,
            "failed": self.failed,
            "samples": [{"from": a, "to": b} for a, b in self.samples],
            "failed_samples": self.failed_samples,
        }


@dataclass
class CleanerLog:
    """한 파일에 적용된 정제 전부의 기록."""

    entries: List[CleanerEntry] = field(default_factory=list)

    @property
    def total_changed(self) -> int:
        return sum(entry.changed for entry in self.entries)

    @property
    def total_failed(self) -> int:
        return sum(entry.failed for entry in self.entries)

    def to_list(self) -> List[dict]:
        # 아무것도 안 한 정제기는 싣지 않는다. 20종 × 15칸이면 300줄인데 그중 대부분이
        # `applied N · changed 0` 이라, 다 실으면 정작 바뀐 것이 묻힌다.
        return [e.to_dict() for e in self.entries if e.changed or e.failed]


def _compare_key(value) -> str:
    """값이 "달라졌나" 를 재는 잣대 — 타입이 아니라 **표기**로 잰다.

    `"1234"` 와 `1234` 는 같은 값의 다른 담김새이므로 변경이 아니다. 반면 `"1.50"` → `1.5` 는
    표기가 달라졌으니 변경으로 센다 — 자릿수를 잃은 것이 맞고, 사람이 볼 만한 일이다.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)) or value is pd.NA:
        return "\x00"                      # 결측끼리는 서로 같다
    return str(value)


#: 값을 고치는 것이 아니라 **담는 그릇만 바꾸는** 정제기.
#: 이쪽은 변경 여부를 표기가 아니라 **값**으로 잰다 — 아래 `_changed_mask` 참고.
CASTING_CLEANERS = frozenset({"to_int", "to_float"})


def _changed_mask(before: pd.Series, after: pd.Series, *, casting: bool) -> pd.Series:
    """값이 실제로 달라진 자리를 표시한다.

    두 가지를 변경으로 세지 않는다.

    ① **둘 다 결측인 자리.** `NaN != NaN` 이라 그냥 비교하면 결측이 전부 변경이 된다.

    ② 🔴 **그릇만 바뀐 자리.** `to_int` 는 `"1234"`(문자열)를 `1234`(정수)로 만드는데, 이걸
       변경으로 세면 숫자 칸은 거의 모든 행이 변경이 되고 표본 20칸이 `1234 → 1234` 같은
       것으로 다 차 버린다. 정작 눈으로 봐야 할 `"1.5" → <NA>` 가 밀려난다.

    그래서 `to_int`·`to_float` 는 **숫자 값으로** 비교하고(`"1.50"` 과 `1.5` 는 같다),
    나머지는 **표기로** 비교한다. 표기 비교를 나머지에 남기는 것이 중요하다 — `zfill6` 의
    `"5930"` → `"005930"` 은 숫자로 보면 둘 다 5930 이라 **변경이 통째로 안 잡힌다.**
    """
    if casting:
        before_number = pd.to_numeric(before, errors="coerce")
        after_number = pd.to_numeric(after, errors="coerce")
        both_missing = before_number.isna() & after_number.isna()
        return ~both_missing & (before_number != after_number)

    # ⚠️ `.astype("object")` 를 먼저 건다. `Int64` 같은 확장 dtype 에 `.map()` 을 바로 걸면
    #    내부 float 표현이 넘어와 `1234` 가 `"1234.0"` 으로 읽힌다(실측 · pandas 3.0.5).
    both_missing = before.isna() & after.isna()
    keys_before = before.astype("object").map(_compare_key)
    keys_after = after.astype("object").map(_compare_key)
    return ~both_missing & (keys_before != keys_after)


def _render(value) -> str:
    """표본에 실을 문자열. 길면 자른다 — 뉴스 본문이 통째로 들어오는 것을 막는다."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value)
    return text if len(text) <= 120 else text[:117] + "..."


def apply_chain(series: pd.Series, chain: Sequence[str], *, column: str,
                sample_limit: int = SAMPLE_LIMIT) -> Tuple[pd.Series, CleanerLog,
                                                           pd.Series, pd.Series]:
    """정제기를 **적힌 순서대로** 걸고, 무엇이 바뀌었는지 기록한다.

    돌려주는 것은 `(정제된 값, 기록, 실패한 자리, 바뀐 자리)` 다. 뒤의 두 마스크는 정제기마다
    새로 생기므로 **논리합으로 누적**한다 — 앞에서 못 읽은 값을 뒤 정제기가 통과시켜도 실패였다는
    사실은 남는다.

    "바뀐 자리" 를 값이 아니라 **마스크로** 돌려주는 이유: 어느 행의 어느 칸에 손을 댔는지는
    행마다 남길 만큼 싸지만(불리언 한 칸), *무엇을 무엇으로* 바꿨는지까지 행마다 남기면 기록이
    원본보다 커진다. 그래서 행에는 **손 댄 칸 이름**만 남기고, 실제 before → after 는 위의
    표본 20건에서 본다.

    순서가 뜻을 갖는다는 점이 중요하다. `["strip", "strip_comma", "to_int"]` 는 공백을 떼고
    쉼표를 뗀 뒤 숫자로 읽는다는 뜻이고, 이 순서를 뒤집으면 `to_int("1,234")` 가 실패한다.
    """
    values = series
    log = CleanerLog()
    failed_any = pd.Series(False, index=series.index)
    changed_any = pd.Series(False, index=series.index)

    for name in chain:
        cleaner = CLEANERS.get(name)
        if cleaner is None:
            raise CleanerError(
                f"규격이 모르는 정제기를 가리킨다: {name!r} (칸 {column})\n"
                f"  아는 이름: {', '.join(sorted(CLEANERS))}\n"
                "  할 일: 규격의 오타를 고치거나, 새 정제기라면 ingest/inbox/cleaners.py 에 더한다."
            )

        before = values
        outcome = cleaner(values)
        values = outcome.values

        differs = _changed_mask(before, values, casting=name in CASTING_CLEANERS)

        applied = int(before.notna().sum())
        changed_index = differs[differs].index[:sample_limit]
        failed_index = outcome.failed[outcome.failed].index[:sample_limit]

        log.entries.append(CleanerEntry(
            column=column,
            cleaner=name,
            applied=applied,
            changed=int(differs.sum()),
            failed=int(outcome.failed.sum()),
            samples=[(_render(before[i]), _render(values[i])) for i in changed_index],
            failed_samples=[_render(before[i]) for i in failed_index],
        ))
        failed_any = failed_any | outcome.failed
        changed_any = changed_any | differs.fillna(False)

    return values, log, failed_any, changed_any


def normalize_missing(series: pd.Series, tokens: Sequence[str]) -> pd.Series:
    """규격의 `missingValues` 에 적힌 표기를 진짜 결측으로 바꾼다.

    **정제보다 먼저 돈다.** `"-"` 를 결측으로 보지 않은 채 `to_int` 에 넘기면 실패로 세어지고,
    그 행이 격리된다 — 출처가 결측을 그렇게 적었을 뿐인데.

    ⚠️ 대소문자를 구별한다. 규격이 `"NA"` 와 `"nan"` 을 **따로** 적어 둔 것은 그래서다.
    소문자로 뭉개면 `"None"` 이라는 회사명·계정명이 결측이 된다.
    """
    if not tokens:
        return series
    lookup = set(tokens)
    return series.map(
        lambda v: None if isinstance(v, str) and v.strip() in lookup else v
    )


__all__ = [
    "SAMPLE_LIMIT",
    "CleanerError",
    "CleanOutcome",
    "CleanerEntry",
    "CleanerLog",
    "CLEANERS",
    "MARKET_MAP",
    "FREQ_MAP",
    "SOURCE_MAP",
    "apply_chain",
    "normalize_missing",
]
