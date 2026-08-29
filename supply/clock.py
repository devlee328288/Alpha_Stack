"""언제부터 알 수 있었나 — `known_at` 을 정한다.

**이 파일이 답하는 질문.** 2010-01-04 의 코스피 200 종가를 우리는 **언제부터** 알 수
있었나. 답이 틀리면 미래를 훔쳐본 모델이 나오고, 그 모델은 **성능이 좋아 보이기 때문에**
가장 위험하다. 잘못된 성능은 예외를 던지지 않는다.

수집 시각은 답이 아니다
-----------------------
가장 흔한 실수는 *"우리가 받은 시각"* 을 그대로 쓰는 것이다. 우리는 2010년 자료를
2026년에 백필했다. 그 값을 그대로 쓰면 2010~2025년 전 구간이 *"2026년에야 알 수 있었던
것"* 이 되어 백테스트에 아무것도 못 쓴다.

반대로 **수집 시각을 무시하면** 반대 방향으로 틀린다. 그래서 자료 종류마다 갈린다.

    시세·지수    공표 시각이 규칙적이다 → **거래일로부터 계산한다** (이 파일)
    공시·재무    공표 시각이 불규칙하다 → **접수일시를 받아 적는다** (조인이 필요하다)
    뉴스         발행 시각이 그대로 답이다

시세를 계산으로 정하는 이유는, 4,097 거래일마다 공표 시각을 따로 받아 두는 것이
불가능하기도 하지만 **규칙이 있으면 계산이 기록보다 정확하기 때문**이다. 기록은 빠질 수
있고 빠진 자리를 나중에 채우면 그게 곧 미래 정보다.

얼마나 늦게 알 수 있었나
------------------------
실측 2026-08-26. 장 마감은 15:30 인데 **16:10 에 당일 자료를 요청하니 0행**이었다.
즉 마감 40분 뒤에도 아직 올라오지 않았다. 정확히 몇 시에 올라오는지는 **재지 않았다.**

재지 않은 값을 가정으로 쓰면 안 되므로 **하루를 통째로 미룬다** — 거래일 T 의 자료는
**T+1 의 0시부터** 알 수 있었다고 본다. 이 선택은 항상 진실보다 늦은 쪽이라, 틀리더라도
**성능을 부풀리는 방향으로는 틀리지 않는다.**

⚠️ 이 값을 앞당기려면 **실측하고 나서** 옮긴다. 앞당기는 방향은 곧 누수 방향이다.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Optional, Union

from common.trading_calendar import KST

#: `as_of` 로 받을 수 있는 것들. 문자열·날짜·시각 아무거나 받되 안에서 하나로 만든다.
AsOf = Union[str, date, datetime]


class AsOfRequired(TypeError):
    """`as_of` 없이 자료를 달라고 했다.

    기본값을 두지 않는 것이 이 설계의 핵심이다. 기본값이 "지금" 이면 **빠뜨려도
    돌아가고**, 빠뜨린 코드가 조용히 미래를 본다. 빠뜨리면 터지게 만들어야 막힌다.
    """


def to_kst(value: AsOf) -> datetime:
    """무엇으로 받았든 KST 가 붙은 시각 하나로 만든다.

    - `date` → 그날 **0시**. "2026-08-26 시점" 이라고 말했을 때 그날 하루치를 이미
      아는 것으로 치면 하루만큼 미래를 보게 된다.
    - 타임존이 없는 `datetime` → KST 로 본다. 이 프로젝트의 시각은 전부 KST 다.
    """
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(
                f"as_of 를 읽을 수 없다: {value!r}\n"
                "  쓸 수 있는 꼴: '2026-08-26' · '2026-08-26T15:30:00' · "
                "'2026-08-26T15:30:00+09:00'"
            ) from exc

    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=KST)
    if isinstance(value, date):
        return datetime.combine(value, time.min, tzinfo=KST)

    raise TypeError(f"as_of 로 쓸 수 없는 값이다: {type(value).__name__}")


def known_at(bas_dd: str) -> datetime:
    """거래일 `YYYYMMDD` 의 시세를 **언제부터 알 수 있었나.**

    T+1 의 0시(KST)를 돌려준다. 왜 하루를 통째로 미루는지는 이 파일 맨 위에 있다 —
    한 줄로 줄이면 *"실측하지 않은 값을 가정으로 쓰지 않는다"* 이다.
    """
    if len(bas_dd) != 8 or not bas_dd.isdigit():
        raise ValueError(f"거래일은 YYYYMMDD 여야 한다: {bas_dd!r}")
    day = date(int(bas_dd[:4]), int(bas_dd[4:6]), int(bas_dd[6:]))
    return datetime.combine(day + timedelta(days=1), time.min, tzinfo=KST)


def is_known(bas_dd: str, as_of: AsOf) -> bool:
    """`as_of` 시점에 그 거래일 자료를 알 수 있었는가."""
    return known_at(bas_dd) <= to_kst(as_of)


def as_bas_dd(value: Optional[AsOf]) -> Optional[str]:
    """날짜처럼 생긴 것을 **거래일 키(`YYYYMMDD`)** 하나로 맞춘다. `None` 은 그대로.

    🔴 **왜 필요한가 — 문자열 비교가 조용히 뒤집힌다.**

    거래일 경계를 정할 때 부르는 쪽이 준 `end` 와 `as_of` 가 만든 상한 중 **이른 쪽**을
    골라야 한다. 그런데 둘을 문자열로 그냥 비교하면 표기가 다를 때 답이 뒤집힌다.

        min('2026-08-21', '20260825') == '2026-08-21'

    `'-'`(0x2D)가 `'0'`(0x30)보다 작아서 **하이픈이 든 쪽이 언제나 작다.** 그래서
    `end='2026-08-21'` 처럼 ISO 로 주면 항상 그쪽이 이기고, 그 값이 `bas_dd <= ?` 에
    들어가면 `'20260821' <= '2026-08-21'` 이 거짓이라 **결과가 0행**이 된다.
    예외는 나지 않는다. 빈 표를 받은 쪽은 "그 구간에 자료가 없구나" 로 읽는다.

    표기를 하나로 맞추면 그 실수 자체가 불가능해진다.

    ⚠️ 여기서 **`as_of` 처럼 하루를 미루지 않는다.** 이 함수는 표기만 바꾼다.
       "언제부터 알 수 있었나" 를 정하는 것은 `latest_known_day` 다. 둘을 섞으면
       경계가 하루씩 어긋나는데 그 하루가 곧 누수다.
    """
    if value is None:
        return None
    if isinstance(value, str):
        digits = value.replace("-", "").replace("/", "").replace(".", "").strip()
        if len(digits) == 8 and digits.isdigit():
            return digits
        raise ValueError(
            f"거래일로 읽을 수 없다: {value!r}\n"
            "  쓸 수 있는 꼴: '20260821' · '2026-08-21'"
        )
    if isinstance(value, datetime):
        return value.date().strftime("%Y%m%d")
    if isinstance(value, date):
        return value.strftime("%Y%m%d")
    raise TypeError(f"거래일로 쓸 수 없는 값이다: {type(value).__name__}")


def latest_known_day(as_of: AsOf) -> str:
    """`as_of` 시점에 알 수 있었던 **가장 최근 거래일**(`YYYYMMDD`).

    SQL 에 `bas_dd <= ?` 로 바로 넣을 수 있게 문자열로 준다 — 4,097행을 파이썬으로
    거르는 대신 DB 가 인덱스로 자르게 하려는 것이다.

    T+1 0시부터 T 를 알 수 있으므로, 어떤 시각 X 에서 알 수 있는 마지막 거래일은
    **X 의 전날**이다. (X 가 정확히 0시여도 전날까지다 — 경계에서 하루를 더 주면
    그 하루가 곧 누수다.)
    """
    moment = to_kst(as_of)
    return (moment.date() - timedelta(days=1)).strftime("%Y%m%d")
