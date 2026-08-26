"""거래일·시각 유틸 (공통 계층)

`krx_store`(저장소)와 `market_data`(분석)가 함께 쓰는 작은 도구 모음이다.
어느 쪽에 넣어도 **순환 import** 가 생기기 때문에 별도 모듈로 분리했다.

    trading_calendar  ←  krx_store  ←  market_data
              ↖___________________________/

(위 그림처럼 화살표가 한 방향으로만 흐르면 순환이 생기지 않는다.)
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone
from typing import List, Optional

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
