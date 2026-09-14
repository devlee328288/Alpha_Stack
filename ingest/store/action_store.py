"""`corporate_action` — 자본 사건을 표로 세운다 (스키마 v16).

    python scripts/build_corporate_actions.py

## 왜 표인가

자본변동은 지금까지 **결과로만** 있었다. `adj_close` 는 이미 펴진 값이고 `adj_source`
는 그 값의 출처만 말한다. "무슨 일이 언제 얼마만큼 일어났나" 는 어디에도 없어서, 물을
때마다 `daily_price` 923만 행을 훑으며 `adjustment_factor` 로 역산해야 했다.

zipline 이 `splits`·`mergers`·`dividends` 를 따로 두는 이유가 이것이다 — 조정된 가격과
조정을 일으킨 사건은 다른 자료다. 배당은 v14 에서 이미 표가 됐으므로 여기서는 나머지를
세운다.

## 이 표는 파생물이다 — 원본이 늘면 낡는다

`daily_price` 와 `stock_base_info` 에서 전부 계산해 만든다. `trading_calendar` 와 같은
성질이라 **시세를 적재한 뒤에 다시 깐다.** 낡은 채로 조용히 틀리는 것이 제일 나쁘다.

## 무엇을 대조하나

행마다 두 값을 나란히 담는다.

    ratio        이 표가 **증거로** 세운 배율 (액면가 · 기준가 · 상장일)
    chain        `adj_price` 가 **실제로 쓴** 계수 (`factor_series`)

둘이 어긋난 자리를 표 안에 남기는 것이 목적이다. 밖에서 매번 다시 세면 아무도 안 센다.

🔴 **다만 `agrees` 를 아무 데나 매기지 않는다.** `rights_off`·`resume_revalue` 는 배율을
`adjustment_factor` 에서 얻으므로 그것을 다시 chain 과 대조하면 **검사가 대상과 같은 값을
쓴다** — 처음에 그렇게 두었더니 3,974건이 통째로 "맞음" 으로 나왔다. 초록이 아무것도
뜻하지 않는 자리다. chain 이 배율 계산에 쓰지 않는 축, 곧 **액면가와 상장일**만 잰다.

## 실측 (2026-09-10 · `daily_price` 9,231,938행 · 264초)

    사건            44,290건   2,946종 · 20100105~20260904
    배율을 예고        4,640건
    판정 가능            730건   액면가(708) + 상장일(22) 축
      맞음               722
      어긋남               8   split_merge 4 · market_transfer 4
    못 잼             43,560건   순환(3,974) + 예고 없음(39,586)
    자기모순              17건   **17/17 정리매매**

`source_conflict` 17건은 chain 을 안 쓰고 재는 축이다 — 감자로 주식수가 크게 줄었는데
KRX 가 기준가를 안 고쳐(기준가비가 정확히 1) 등락률이 가격제한폭 밖으로 나온다. 정리매매
구간에는 가격제한폭이 적용되지 않으므로 **원본 오류가 아니고**, 17건 전부
`is_liquidation` 이 덮어 `training_frame` 이 덜어낸다. **학습 표본으로 새는 것은 0건**
이라 값을 고치지 않고 해명과 함께 남긴다.
"""

from __future__ import annotations

import bisect
import sqlite3
from datetime import datetime
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from common.corporate_actions import (
    CorporateEvent,
    classify_event,
    event_agrees,
    factor_series,
    flag_series,
    is_series_restart,
    source_conflict,
)

#: `daily_price` 에서 읽을 칸. `adj_price.PRICE_COLUMNS` 에 `code` 와 `change_rate`
#: 를 얹는다 — 판정 함수가 두 표에서 같은 행을 보므로 칸 구성이 갈리면 안 된다.
#: `change_rate` 는 `source_conflict` 가 쓴다(`is_outlier` 가 가격제한폭과 견준다).
PRICE_COLUMNS = ("code", "bas_dd", "open", "high", "low", "close", "change",
                 "change_rate", "volume", "listed_shares")

#: ⚠️ `stock_base_info` 는 여기서 **전량을 읽지 않는다** (`base_steps` 참고).
#:    주식수는 `daily_price.listed_shares` 와 923만 행 전부 같으므로(실측
#:    2026-09-10 · 불일치 0) 다시 읽을 이유가 없고, 이 표가 홀로 주는 것은
#:    **액면가와 상장일** 둘뿐인데 그 둘은 계단이라 변경 자리만 있으면 된다.

#: 표에 넣는 칸 순서. `INSERT` 와 튜플 조립이 갈리지 않게 한 곳에서 정한다.
TABLE_COLUMNS = ("code", "ex_date", "event_type", "ratio_num", "ratio_den", "source",
                 "par_before", "par_after", "shares_before", "shares_after",
                 "basis_ratio", "chain_num", "chain_den", "agrees",
                 "source_conflict", "is_liquidation", "built_at")


def _rows_by_code(conn: sqlite3.Connection, table: str,
                  columns: Sequence[str]) -> Iterator[Tuple[str, List[Dict]]]:
    """`(종목코드, 그 종목의 전 구간 행)` 을 코드 순서로 흘려 준다.

    종목마다 따로 질의하면 3,677번 훑는다. 한 번 정렬해 받아 코드가 바뀌는 자리에서
    끊는 편이 훨씬 싸다 — 923만 행을 한 번만 지나간다.

    ⚠️ **그 종목의 전부**여야 한다. 잘라 주면 잘린 자리가 조정 하나로 보인다.
    """
    cur = conn.execute(f"SELECT {','.join(columns)} FROM {table} "
                       "ORDER BY code, bas_dd")
    names = [d[0] for d in cur.description]
    현재, 버퍼 = None, []
    for raw in cur:
        d = dict(zip(names, raw, strict=True))
        code = str(d["code"])
        if code != 현재:
            if 현재 is not None:
                yield 현재, 버퍼
            현재, 버퍼 = code, []
        버퍼.append(d)
    if 현재 is not None:
        yield 현재, 버퍼


def base_steps(conn: sqlite3.Connection) -> Dict[str, List[Tuple[str, object, object]]]:
    """`종목코드 → [(바뀐 날, 액면가, 상장일), …]` (날짜 오름차순).

    🔴 **`stock_base_info` 를 전부 읽지 않는다.** 액면가와 상장일은 **계단 함수**다 —
    한 번 정해지면 다음 변경까지 같은 값이라, 어느 날의 값은 *그 날 이하 마지막 변경*
    으로 정해진다. 923만 행 중 실제로 바뀌는 자리는 4,493행뿐이다(첫 행 3,677 +
    액면가 792 + 상장일 24, 실측 2026-09-10).

    처음에는 전량을 dict 로 올렸다가 **14분을 넘기고도 안 끝났다.** 920만 개짜리 dict
    가 메모리를 삼켜 스왑으로 갔다. 표가 크면 "필요한 것만" 이 최적화가 아니라 조건이다.

    ⚠️ 계단으로 되살리므로 **자료에 빠진 날이 있어도 값이 이어진다.** 그게 맞는
       가정이다 — 액면가는 날마다 새로 정해지는 값이 아니라 바뀔 때까지 그대로다.
    """
    out: Dict[str, List[Tuple[str, object, object]]] = {}
    for code, bas_dd, parval, list_dd in conn.execute(
        """
        SELECT code, bas_dd, parval, list_dd FROM (
          SELECT code, bas_dd, parval, list_dd,
                 LAG(parval)  OVER w AS 앞액면,
                 LAG(list_dd) OVER w AS 앞상장,
                 ROW_NUMBER() OVER w AS 순번
          FROM stock_base_info
          WINDOW w AS (PARTITION BY code ORDER BY bas_dd)
        )
        WHERE 순번 = 1 OR parval IS NOT 앞액면 OR list_dd IS NOT 앞상장
        ORDER BY code, bas_dd
        """
    ):
        out.setdefault(str(code), []).append((str(bas_dd), parval, list_dd))
    return out


def steps_as_rows(code: str,
                  steps: Sequence[Tuple[str, object, object]]) -> List[Dict]:
    """계단 목록을 `build_events` 가 받는 행 꼴로 바꾼다.

    행 꼴로 맞추는 까닭: 시험은 `stock_base_info` 전량을 넘기고 적재는 변경 자리만
    넘기는데, **둘이 같은 함수를 지나야** 판정이 갈리지 않는다.
    """
    return [{"code": code, "bas_dd": d, "parval": p, "list_dd": ld}
            for d, p, ld in steps]


def build_events(rows: Sequence[Mapping], base_rows: Sequence[Mapping], *,
                 calendar_index: Mapping[str, int],
                 listing_days: Sequence[str],
                 liquidation: Sequence[bool] = ()) -> List[Tuple[CorporateEvent,
                                                                 Optional[object],
                                                                 Optional[bool],
                                                                 bool, bool]]:
    """한 종목의 사건 전부. `(사건, chain, agrees, 자기모순, 정리매매)` 를 돌려준다.

    `rows` 는 `daily_price` 의 전 구간(`bas_dd` 오름차순)이고 `base_rows` 는 같은
    종목의 `stock_base_info` 다(`bas_dd` 오름차순).

    🔴 **`base_rows` 는 계단으로 읽는다** — 어느 날의 값은 *그 날 이하 마지막 행*이다.
    두 표는 날짜가 어긋난다. 시세에만 있는 날이 2,765행 있고, 반대로 액면가는 바뀔
    때까지 같은 값이라 변경 자리만 담아도 된다. 정확히 그 날 행을 찾는 방식으로 두면
    시세에만 있는 날에서 사건이 조용히 사라진다. 그래서 `base_rows` 에 전량을 줘도
    변경 자리만 줘도 **같은 답**이 나온다(`base_steps` 가 뒤쪽을 준다).

    `liquidation` 은 `flag_series` 가 매긴 정리매매 여부다. 길이가 `rows` 와 같아야
    하고, 비어 있으면 전부 False 로 본다 — 어긋남의 해명이 빠질 뿐 판정은 안 바뀐다.
    """
    if len(rows) < 2:
        return []
    단계 = [(str(b["bas_dd"]), b) for b in base_rows]
    날짜들 = [d for d, _ in 단계]

    def base_at(bas_dd: str):
        i = bisect.bisect_right(날짜들, bas_dd) - 1
        return 단계[i][1] if i >= 0 else None

    factors = factor_series(rows, calendar_index=calendar_index,
                            listing_days=listing_days)
    out = []
    for i in range(1, len(rows)):
        prev_row, row = rows[i - 1], rows[i]
        restart = is_series_restart(prev_row, row, calendar_index=calendar_index,
                                    listing_days=listing_days)
        event = classify_event(
            prev_row, row,
            base_at(str(prev_row["bas_dd"])),
            base_at(str(row["bas_dd"])),
            restart=restart,
        )
        if event is None:
            continue
        # `factor_series` 는 재시작 자리에서 1 을 낸다. 그 1 은 "조정이 없다" 가 아니라
        # "이어 붙일 것이 없다" 이므로, 배율로 적으면 뜻이 뒤집힌다.
        chain = None if restart else factors[i]
        if chain is not None and chain == 1:
            chain = None                     # 조정이 없었다 — 배율 1 을 굳이 적지 않는다
        정리매매 = bool(liquidation[i]) if i < len(liquidation) else False
        모순 = source_conflict(row, event.shares_before, event.shares_after)
        out.append((event, chain, event_agrees(event, chain), 모순, 정리매매))
    return out


def _to_row(event: CorporateEvent, chain, agrees: Optional[bool],
            모순: bool, 정리매매: bool, built_at: str) -> Tuple:
    """표 한 줄. 유리수는 분자·분모 두 칸으로 나눠 담는다 (반올림을 안 들이려고)."""
    ratio = event.ratio
    return (
        event.code, event.ex_date, event.event_type,
        None if ratio is None else ratio.numerator,
        None if ratio is None else ratio.denominator,
        event.source,
        event.par_before, event.par_after,
        event.shares_before, event.shares_after,
        event.basis_ratio,
        None if chain is None else chain.numerator,
        None if chain is None else chain.denominator,
        None if agrees is None else int(agrees),
        int(모순), int(정리매매), built_at,
    )


def rebuild(conn: sqlite3.Connection, *, calendar_index: Mapping[str, int],
            listing_days_by_code: Mapping[str, Sequence[str]],
            market_last_index: int, still_listed, collect_start: str,
            write_lock=None, progress=None) -> Dict[str, int]:
    """표를 통째로 다시 깐다. 무엇을 몇 건 넣었는지 돌려준다.

    `DELETE` 후 다시 넣는 이유는 `rebuild_calendar` 와 같다 — 사건이 사라졌을 때
    (원본을 다시 받아 값이 바뀌었을 때) **낡은 줄이 남지 않게** 하려는 것이다.
    4만 행이라 비용이 없다.

    달력·상장일·시장 마지막 날은 **부르는 쪽이 한 번 만들어** 넘긴다. 종목마다 다시
    만들면 한 시간이 넘는다 (`build_and_save` 와 같은 이유).
    """
    built_at = datetime.now().isoformat(timespec="seconds")
    단계 = base_steps(conn)

    묶음: List[Tuple] = []
    센것: Dict[str, int] = {}
    for code, rows in _rows_by_code(conn, "daily_price", PRICE_COLUMNS):
        listing = tuple(listing_days_by_code.get(code, ()))
        flags = flag_series(rows, calendar_index=calendar_index,
                            market_last_index=market_last_index,
                            still_listed=code in still_listed,
                            collect_start=collect_start, listing_days=listing)
        base_rows = steps_as_rows(code, 단계.get(code, ()))
        for event, chain, agrees, 모순, 정리매매 in build_events(
                rows, base_rows, calendar_index=calendar_index,
                listing_days=listing,
                liquidation=[f.liquidation for f in flags]):
            묶음.append(_to_row(event, chain, agrees, 모순, 정리매매, built_at))
            센것[event.event_type] = 센것.get(event.event_type, 0) + 1
            if agrees is False:
                센것["어긋남"] = 센것.get("어긋남", 0) + 1
            if 모순:
                센것["자기모순"] = 센것.get("자기모순", 0) + 1
        if progress is not None:
            progress(code, len(묶음))

    칸 = ",".join(TABLE_COLUMNS)
    자리 = ",".join("?" * len(TABLE_COLUMNS))
    잠금 = write_lock if write_lock is not None else _NoLock()
    with 잠금:
        conn.execute("DELETE FROM corporate_action")
        conn.executemany(f"INSERT INTO corporate_action ({칸}) VALUES ({자리})", 묶음)
    센것["행"] = len(묶음)
    return 센것


class _NoLock:
    """자물쇠를 안 준 부름 — 시험처럼 혼자 도는 자리."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
