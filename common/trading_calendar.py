"""거래일·시각 유틸 (공통 계층)

`krx_store`(저장소)와 `market_data`(분석)가 함께 쓰는 작은 도구 모음이다.
어느 쪽에 넣어도 **순환 import** 가 생기기 때문에 별도 모듈로 분리했다.

    trading_calendar  ←  krx_store  ←  market_data
              ↖___________________________/

(위 그림처럼 화살표가 한 방향으로만 흐르면 순환이 생기지 않는다.)
"""

from __future__ import annotations

import math
from bisect import bisect_right
from datetime import date, datetime, timedelta, timezone
from typing import List, Optional, Tuple

# 한국 시간 기준으로 거래일을 계산한다.
# UTC 보다 9시간 빠른 고정 오프셋 — 한국은 서머타임이 없어 이렇게 단순히 표현할 수 있다.
KST = timezone(timedelta(hours=9))


def today_kst() -> date:
    """지금 이 순간의 한국 날짜(YYYY-MM-DD)를 반환한다."""
    # datetime.now(KST) 로 KST 기준 현재 시각을 얻고, .date() 로 시·분·초를 떼어낸다.
    return datetime.now(KST).date()


def now_kst_iso() -> str:
    """지금 이 순간을 KST 오프셋이 붙은 ISO8601 문자열로. 예: `2026-08-26T13:54:48+09:00`

    **왜 오프셋을 붙이나.** 기존 수집 대장(`fetch_log.fetched_at`)은 `datetime.now()` —
    타임존이 없는 naive 값이라, 다른 시간대에서 돌리면 같은 문자열이 다른 순간을 뜻한다.

    "언제 받았나" 는 두 곳이 기대는 값이라 모호하면 안 된다.
    하나는 **어디까지 받았는지 판단해 중단 지점부터 이어 받는 것**이고,
    다른 하나는 **그 시점에 알 수 있었던 자료만 골라 학습에 쓰는 것**이다.
    뒤엣것이 흔들리면 미래를 훔쳐본 모델이 나오고, 그건 성능이 좋아 보여서 더 위험하다.

    새 코드는 이 함수를 쓴다.
    """
    return datetime.now(KST).isoformat(timespec="seconds")


def round_half_up(value: float) -> int:
    """파이썬 round() 는 은행가 반올림이라 0.5 가 짝수 쪽으로 붙는다.

    사람이 기대하는 "0.5 는 올림"으로 맞추기 위해 0.5 를 더한 뒤 버린다.
    (2.5 → 3, 3.5 → 4. 파이썬 기본 round() 는 2, 4 를 준다.)
    """
    return math.floor(value + 0.5)


def trading_days(count: int, end: Optional[date] = None) -> List[date]:
    """end(기본: 오늘 KST)부터 거꾸로 주말을 건너뛰며 거래일 후보를 모은다.

    ⚠️ 공휴일은 반영하지 않는다. 실제 휴장일인지는 KRX 응답이 0건인지로 판별하며,
    그 결과는 `krx_store.fetch_log` 에 기록해 다시 요청하지 않는다.
    """
    day = end or today_kst()
    days: List[date] = []
    while len(days) < count:            # 원하는 개수를 채울 때까지 반복
        if day.weekday() < 5:           # 월(0) ~ 금(4)
            days.append(day)
        day -= timedelta(days=1)        # 하루 전으로 이동
    # 과거 → 현재 순으로 뒤집는다 (차트는 왼쪽이 과거이므로)
    return list(reversed(days))


def to_iso(bas_dd: str) -> str:
    """KRX 형식 `20260730` 을 화면·차트용 `2026-07-30` 으로 바꾼다."""
    return f"{bas_dd[:4]}-{bas_dd[4:6]}-{bas_dd[6:]}"


def to_krx(iso: str) -> str:
    """`2026-07-30` 을 KRX 형식 `20260730` 으로 바꾼다."""
    return iso.replace("-", "")


# ==================================================
# 실측 거래일 달력 — 계산이 아니라 기록으로
# ==================================================
# 🔴 **`trading_days()` 로 거래일을 판정하면 안 된다.** 그 함수는 주말만 건너뛴다.
#    개발구간(20100104~20210831)에서 재 보면 평일 3,042일 중 실제 거래일은 2,880일이라
#    **162일(5.3%)이 어긋난다.** 그 162일은 명절·공휴일이고, 하필 실적 발표와 뉴스가 몰리는
#    연휴 전후다. 뉴스의 `eff_dd` 배정이 이 달력 위에서 이뤄지므로 어긋나면 곧 미래참조다.
#
#    그래서 휴장일을 계산으로 맞히려 하지 않고 **우리가 실제로 받은 날을 그대로 쓴다** —
#    `daily_price.bas_dd` 에 있는 날이 거래일이다. 이건 추정이 아니라 기록이다.
#
# `trading_days()` 를 지우지 않는 이유: 그쪽은 *"받아 볼 후보 날짜"* 를 고르는 용도라
# 주말만 걸러도 맞다(휴장이면 0건으로 돌아오고 그 사실이 대장에 남는다). 쓰임이 다르다.

_SESSION_CACHE: Optional[frozenset] = None
_SESSION_SPAN: Optional[tuple] = None

#: 같은 날짜들을 **정렬한 사본**. 집합은 "그 날이 거래일인가" 에 빠르지만
#: "그 다음 거래일은 언제인가" 에는 못 쓴다 — 순서가 없기 때문이다.
#:
#: 예전에는 `min(d for d in days if d > bas_dd)` 로 매번 4,102개를 훑었다. 이 함수는
#: 행마다 불린다(재무의 `next_business_day`, 뉴스의 `eff_dd`, 새 수집원의 `known_at`).
#: 12,306행짜리 수집이면 5천만 번의 비교가 된다. 정렬해 두고 이분탐색하면 12번이다.
#:
#: 날짜는 `YYYYMMDD` 고정폭 문자열이라 **사전순 = 시간순** 이다. 그래서 문자열
#: 그대로 이분탐색해도 맞는다 (자릿수가 다르면 이 전제가 깨진다).
_SESSION_SORTED: Optional[Tuple[str, ...]] = None

#: `_SESSION_SORTED` 를 **어느 집합에서** 만들었는지. 정렬본이 낡았는지 판정하는 근거다.
#:
#: 🔴 테스트는 `_SESSION_CACHE` 를 monkeypatch 로 갈아 끼운다(`test_inbox_derive.py` ·
#:    `test_inbox_engine.py`). 정렬본을 갱신하는 책임을 갈아 끼우는 쪽에 맡기면, 새로
#:    끼우는 자리마다 잊을 수 있고 그때 **에러 없이 옛 달력으로 답한다.** 그래서
#:    "집합이 바뀌었으면 다시 정렬한다" 를 읽는 쪽 한 곳에서 지킨다.
_SORTED_FOR: Optional[frozenset] = None


class CalendarOutOfRange(LookupError):
    """물어본 날짜가 우리가 가진 달력 밖이다 — 답을 지어내지 않고 세운다."""


def _sorted_days(days: frozenset) -> Tuple[str, ...]:
    """`days` 를 정렬한 사본. 같은 집합이면 지난번 것을 그대로 준다.

    같은 집합인지는 **동일성(`is`)** 으로 본다. 내용 비교는 4,102개를 다시 훑는 것이라
    아끼려던 비용이 그대로 돌아오고, 크기 비교는 크기가 같으면서 내용이 다른 경우를
    놓친다.
    """
    global _SESSION_SORTED, _SORTED_FOR
    if _SORTED_FOR is not days or _SESSION_SORTED is None:
        _SESSION_SORTED = tuple(sorted(days))
        _SORTED_FOR = days
    return _SESSION_SORTED


def load_session_days(db_path=None, *, refresh: bool = False) -> frozenset:
    """실제 거래가 있었던 날의 집합을 `YYYYMMDD` 문자열로 돌려준다.

    한 번 읽어 캐시한다. 4,102개짜리 집합이라 메모리는 무시할 수 있고, 반입 검사가 행마다
    부르기 때문에 매번 DB 를 두드리면 느리다.

    **`trading_calendar` 표를 먼저 본다** (마이그레이션 v9). 없거나 비어 있으면
    `daily_price` 를 직접 센다 — 답은 같고 속도만 다르다:

        SELECT DISTINCT bas_dd FROM daily_price   9.2M 행을 훑는다   660ms
        SELECT bas_dd FROM trading_calendar       4,102행을 읽는다   ~1ms

    ⚠️ **폴백을 지우지 않는다.** 표는 `daily_price` 에서 파생된 것이라 시세를 더 받고
       `adj_price.rebuild_calendar()` 를 안 부르면 낡는다. 그때 표가 없다고 예외를 내면
       달력을 쓰는 기능이 전부 멈추고, 조용히 낡은 답을 주면 더 나쁘다. 느린 쪽이 낫다.
    """
    global _SESSION_CACHE, _SESSION_SPAN
    if _SESSION_CACHE is not None and not refresh:
        return _SESSION_CACHE

    import sqlite3

    from common.paths import krx_db_path

    path = db_path or krx_db_path()
    conn = sqlite3.connect(path)
    try:
        rows = []
        try:
            rows = conn.execute(
                "SELECT bas_dd FROM trading_calendar WHERE market = 'ALL'").fetchall()
        except sqlite3.OperationalError:
            pass                      # v9 이전 DB — 표가 없다. 아래에서 원본을 센다
        if not rows:
            rows = conn.execute("SELECT DISTINCT bas_dd FROM daily_price").fetchall()
    finally:
        conn.close()

    days = frozenset(str(row[0]) for row in rows)
    if not days:
        raise CalendarOutOfRange(
            "거래일 달력이 비어 있다 — daily_price 에 행이 없다.\n"
            "  할 일: python scripts/fetch_krx.py 로 시세를 먼저 받는다."
        )
    _SESSION_CACHE = days
    정렬본 = _sorted_days(days)          # 여기서 만들어 두면 첫 `next_session` 이 안 기다린다
    _SESSION_SPAN = (정렬본[0], 정렬본[-1])
    return days


def session_span(db_path=None) -> tuple:
    """달력이 덮는 구간 `(첫 거래일, 마지막 거래일)`."""
    load_session_days(db_path)
    return _SESSION_SPAN


def is_session(bas_dd: str, db_path=None) -> bool:
    """그 날이 거래일이었나. **달력 밖이면 세운다** — False 로 답하지 않는다.

    범위 밖에 False 를 주면 부르는 쪽이 *"휴장이었구나"* 로 읽는다. 2026-09-01 이 휴장인 것과
    우리가 아직 안 받은 것은 전혀 다른 사실이고, 뒤엣것을 앞엣것처럼 다루면 조용히 틀린다.
    """
    days = load_session_days(db_path)
    first, last = _SESSION_SPAN
    if bas_dd < first or bas_dd > last:
        raise CalendarOutOfRange(
            f"{bas_dd} 는 우리 거래일 달력({first}~{last}) 밖이다.\n"
            "  왜 세우나: 달력 밖을 '휴장' 으로 답하면 아직 안 받은 날과 구별되지 않는다.\n"
            "  할 일: 그 구간 시세를 받거나, 부르는 쪽에서 이 예외를 잡아 그 행을 격리한다."
        )
    return bas_dd in days


def next_session(bas_dd: str, db_path=None, *, inclusive: bool = False) -> str:
    """그 날(포함 여부는 `inclusive`) 이후 처음 오는 거래일.

    뉴스의 `eff_dd`, 재무의 `next_business_day(rcept_dt)` 가 이걸 쓴다.

    `inclusive=True` 면 그 날이 거래일일 때 그 날을 돌려준다 — 장 시작 전(08:30 미만) 기사가
    같은 날 시가에 반영되는 경우다. 기본값이 False 인 이유는 *"접수일 다음 거래일부터 유효"*
    처럼 당일을 배제하는 쪽이 더 흔하고, **늦는 방향이 안전한 방향**이기 때문이다.
    """
    days = load_session_days(db_path)
    first, last = _SESSION_SPAN

    if inclusive and bas_dd in days:
        return bas_dd

    # 정렬본에서 `bas_dd` 보다 **큰** 첫 자리를 이분탐색으로 찾는다. 예전에는 4,102개를
    # 전부 훑어 새 목록을 만들고 그 최솟값을 구했다 — 답은 같고 비교 횟수만 다르다.
    # 이 함수는 행마다 불린다(재무의 `next_business_day`, 뉴스의 `eff_dd`, 새 수집원의
    # `known_at`). 만 단위 수집이면 그 차이가 그대로 시간이 된다.
    정렬본 = _sorted_days(days)
    자리 = bisect_right(정렬본, bas_dd)
    if 자리 >= len(정렬본):
        raise CalendarOutOfRange(
            f"{bas_dd} 다음 거래일을 모른다 — 달력이 {last} 까지밖에 없다.\n"
            "  왜 세우나: 다음 거래일을 지어내면 아직 열리지 않은 장에 자료를 붙이게 된다.\n"
            "  할 일: 시세를 더 받아 달력을 넓히거나, 그 행을 격리한다."
        )
    if bas_dd < first:
        # 달력보다 이른 날은 "그 다음 거래일" 을 물어도 의미가 있다(첫 거래일). 다만 그
        # 사이에 우리가 모르는 거래일이 있었을 수 있으므로 사실대로 알린다.
        raise CalendarOutOfRange(
            f"{bas_dd} 는 달력 시작({first})보다 이르다 — 그 사이 거래일을 우리는 모른다.\n"
            "  할 일: 더 이른 구간을 받거나, 그 행을 격리한다."
        )
    return 정렬본[자리]
