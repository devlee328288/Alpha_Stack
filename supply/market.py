"""시세를 **그 시점에 알 수 있었던 만큼만** 내보낸다.

이 모듈이 하는 일은 하나다 — `krx_index.series()` 가 주는 시계열에서 **`as_of` 시점에
아직 몰랐던 구간을 잘라내는 것**이다. 그게 전부인데, 그게 전부이기 때문에 중요하다.

왜 저장소를 직접 부르면 안 되나
------------------------------
`krx_index.series()` 는 표에 있는 것을 전부 준다. 표에는 **오늘까지** 들어 있다.
2020년 폴드를 학습하면서 그 함수를 그대로 부르면 2026년까지가 딸려 오고, 그 상태로
피처를 만들면 이동평균 한 줄에 미래가 섞인다. **예외는 안 난다.** 성능만 좋아진다.

그래서 저장소를 직접 부르는 길을 막고 이 문 하나만 열어 둔다. 문을 지나려면
`as_of` 를 내야 하고, **기본값이 없어서 빠뜨릴 수가 없다.**

    from supply import index_series

    rows = index_series(as_of="2020-06-30")     # 2020-06-29 까지만 나온다
    rows = index_series()                        # TypeError — 빠뜨릴 수 없다
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ingest.clients import krx_data as api
from ingest.store import krx_index
from supply.clock import AsOf, known_at, latest_known_day, to_kst

#: 1차 프로젝트의 예측 대상.
TARGET_INDEX = api.TARGET_INDEX


def index_series(index_name: str = TARGET_INDEX, *, as_of: AsOf,
                 start: Optional[str] = None, end: Optional[str] = None,
                 days: Optional[int] = None,
                 with_known_at: bool = False) -> List[Dict]:
    """`as_of` 시점에 알 수 있었던 지수 시계열을 **과거 → 현재 순**으로.

    `as_of` 는 키워드 전용이고 **기본값이 없다.** 빠뜨리면 `TypeError` 로 즉시 터진다 —
    기본값이 "지금" 이면 빠뜨린 코드가 조용히 미래를 보게 된다.

    ⚠️ 정렬 방향이 계약의 일부다(저장소와 같다). 부르는 쪽에서 다시 정렬하지 않는다.

    `with_known_at=True` 면 각 행에 **언제부터 알 수 있었는지**를 붙여 준다. 누수를
    의심할 때 눈으로 확인하는 용도다.
    """
    cutoff = latest_known_day(as_of)
    # 부르는 쪽이 준 end 와 "알 수 있었던 마지막 날" 중 **이른 쪽**을 쓴다.
    # 자르는 일은 DB 가 인덱스로 하게 둔다 — 4,097행을 파이썬으로 거르는 것보다 싸다.
    limit = min(end, cutoff) if end else cutoff

    rows = krx_index.series(index_name, days=days, start=start, end=limit)
    if with_known_at:
        for row in rows:
            # `series()` 는 날짜를 `2026-08-25` 꼴로 준다. 거래일 문자열로 되돌려 계산한다.
            bas_dd = row["date"].replace("-", "")
            row["known_at"] = known_at(bas_dd).isoformat(timespec="seconds")
    return rows


def as_of_bounds(as_of: AsOf) -> Dict[str, str]:
    """이 `as_of` 가 실제로 어디를 자르는지 알려 준다. 리포트와 사전등록에 적는 값이다.

    사전등록 문서에 *"홀드아웃은 언제부터"* 를 적을 때 사람이 손으로 계산하면 하루씩
    어긋난다. 코드가 답하게 한다.
    """
    moment = to_kst(as_of)
    return {
        "as_of": moment.isoformat(timespec="seconds"),
        "last_known_trading_day": latest_known_day(as_of),
    }
