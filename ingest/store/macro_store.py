"""한국은행 ECOS 거시지표를 받아 `macro_series` 에 채운다. (수집 → 저장 계층)

`scripts/fetch_macro.py` 가 이 모듈을 부른다. 시세의 `krx_store`, 재무의 `dart_store`
와 짝이고 같은 DB 파일의 다른 표에 담는다.

## 무엇을 받나

큐레이션 9종 — 금리 3(기준금리·국고채 3년·10년) · 환율 2 · 물가 2 · 경기 2.
지표 하나에 **1콜**이고, 한 번에 17년치가 통째로 온다. 그래서 전체 수집이 9콜이다.
900만 행을 4,348콜로 받은 시세와 견주면 사실상 공짜라, 부분 수집을 만들 이유가 없다.

실측 2026-09-02 — 요청 17년:

    일별 4종   각 4,207~4,208행  2009-08-27 ~ 2026-09-01
    월별 5종   각   204~  205행  2009-08-01 ~ 2026-07/08
    합계          약 17,851행

⚠️ **2010 으로 자르지 않는다.** 시세는 2010-01-04 부터지만 거시는 그보다 앞을 받아
   둔다. 거시는 계단식으로 쓰는 값이라(직전 발표치를 다음 발표 전까지 이어 쓴다),
   2009년 12월 물가가 없으면 **2010년 1월 초에 참조할 값이 사라진다.**

⚠️ **일별이 4,208행이고 `ecos_data.MAX_ROWS` 가 5,000 이다.** 연 250행씩 늘어나므로
   2029년경 상한에 닿는다. 그때 잘리면 `truncated` 로 알려 주지만 **행 수만 세는
   검사로는 안 잡힌다** — 끝이 잘리는 게 아니라 오래된 쪽이 사라지기 때문이다.
   `sync()` 가 `truncated` 를 오류로 올리는 이유다.

## 🔴 시점 기준은 `known_at` 하나뿐이다

ECOS 는 월별 값을 **기준월 1일**로 준다 — 2026년 7월 물가가 `2026-07-01` 이다.
그 날짜에 값을 붙이면 7월 물가를 7월 1일에 아는 셈인데, 실제 발표는 8월 4일이었다.
경기지수는 7월분이 8월 31일에 나오므로 61일이 벌어진다.

재무의 `rcept_dt` 가 하는 일을 여기서는 `ecos_data.known_at()` 이 한다. 다만 재무와
달리 **출처가 날짜를 주지 않아** 계산해서 붙인다 — ECOS 세 서비스를 전부 확인했고
어디에도 공표일이 없다(실측 2026-09-02). 규칙과 근거는 `ecos_data.RELEASE_RULES` 에 있다.

## 어디까지 받았는지 기억한다

수집 대장(`collect_log`)에 지표마다 한 줄을 남긴다. 다만 재무와 달리 **지표 하나가
곧 전 구간**이라, `empty` 는 "그 지표를 ECOS 가 안 준다" 는 뜻이다.

    ok      값이 들어왔다
    empty   받아봤는데 0건 — 통계표가 폐지됐거나 항목코드가 바뀐 것이다
    error   시도했는데 실패했다
"""

from __future__ import annotations

import sqlite3
from contextlib import nullcontext
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from common.trading_calendar import now_kst_iso
from ingest.clients import ecos_data
from ingest.store import collect_log
from ingest.store.krx_store import connect, init_db

#: 수집 대장에 적히는 출처 이름.
SOURCE = "macro_ecos"

#: 기본으로 받는 햇수. 2009년 8월부터 오므로 시세(2010-01-04)를 앞뒤로 덮는다.
DEFAULT_YEARS = 17

#: `macro_series` 의 칸 순서. 마이그레이션 v7 의 정의와 **같은 순서**여야 한다.
COLUMNS: Tuple[str, ...] = (
    "indicator_id", "period", "cycle", "value", "known_at",
    "stat_code", "item_code", "unit", "collected_at",
)

_INSERT = (
    f"INSERT OR REPLACE INTO macro_series ({', '.join(COLUMNS)}) "
    f"VALUES ({', '.join('?' * len(COLUMNS))})"
)


# ==================================================
# 1. 정규화
# ==================================================
def rows_from(series: Dict, collected_at: str) -> List[Tuple]:
    """`ecos_data.fetch_series` 의 응답을 `macro_series` 행으로 바꾼다.

    `periods` 를 쓰고 `dates` 를 쓰지 않는다. `dates` 는 화면용으로 `2026-07-01` 처럼
    바꾼 것이라 월별인지 일별인지가 지워져 있는데, 그 구분이 있어야 `known_at` 을
    계산할 수 있다. `periods` 는 ECOS 원문(`202607` · `20260901`)을 그대로 담고 있다.
    """
    indicator_id = series.get("id") or ""
    spec = ecos_data.INDICATOR_BY_ID.get(indicator_id)
    if not spec:
        raise ecos_data.EcosError(
            f"'{indicator_id}' 는 큐레이션에 없는 지표라 담을 수 없습니다.", status=502)

    periods = series.get("periods") or []
    values = series.get("values") or []
    if len(periods) != len(values):
        # 두 배열의 길이가 어긋나면 짝이 밀린다 — 값이 엉뚱한 기간에 붙는데
        # 행 수는 맞으므로 아무도 눈치채지 못한다.
        raise ecos_data.EcosError(
            f"'{indicator_id}' 의 기간({len(periods)})과 값({len(values)}) 개수가 다릅니다. "
            f"짝이 밀린 채로 담으면 값이 엉뚱한 시점에 붙습니다.", status=502)

    cycle = spec["cycle"]
    stat_code = spec["stat"]
    item_code = spec["items"][0] if spec.get("items") else None
    unit = series.get("unit") or spec.get("unit")

    rows: List[Tuple] = []
    # `strict=True` 는 위 길이 검사와 겹치지만 남겨 둔다 — 검사를 나중에 손대도
    # 짝이 밀린 채 조용히 담기는 일만은 막힌다.
    for period, value in zip(periods, values, strict=True):
        rows.append((
            indicator_id,
            str(period),
            cycle,
            value,                       # None 일 수 있다 — ECOS 가 '-' 를 주는 경우
            ecos_data.known_at(indicator_id, str(period)),
            stat_code,
            item_code,
            unit,
            collected_at,
        ))
    return rows


# ==================================================
# 2. 저장
# ==================================================
def save(series: Dict, conn: Optional[sqlite3.Connection] = None) -> int:
    """지표 하나의 시계열을 저장하고 **저장한 줄 수**를 돌려준다."""
    rows = rows_from(series, now_kst_iso())
    if not rows:
        return 0

    if conn is not None:
        conn.executemany(_INSERT, rows)
        return len(rows)

    with connect() as own:
        own.executemany(_INSERT, rows)
    return len(rows)


# ==================================================
# 3. 수집
# ==================================================
def target_of(indicator_id: str) -> str:
    """수집 대장의 대상 열쇠. 지표 하나가 곧 전 구간이라 지표 이름이 그대로 열쇠다."""
    return indicator_id


def sync(indicators: Optional[Sequence[str]] = None,
         years: int = DEFAULT_YEARS,
         progress: Optional[Callable] = None,
         conn: Optional[sqlite3.Connection] = None) -> Dict:
    """큐레이션 지표를 받아 채운다.

    돌려주는 것: `{"requested", "ok", "empty", "error", "rows", "details"}`

    재무의 `sync` 와 달리 **이미 받은 것을 건너뛰지 않는다.** 지표 하나가 1콜이고
    전체가 9콜뿐이라 아낄 것이 없고, 거시는 **과거 값이 나중에 개정된다** —
    건너뛰면 개정을 영원히 못 받는다. `INSERT OR REPLACE` 라 다시 받아도 늘지 않는다.
    """
    init_db()

    ids = list(indicators) if indicators else [s["id"] for s in ecos_data.INDICATORS]
    결과 = {"requested": len(ids), "ok": 0, "empty": 0, "error": 0, "rows": 0,
            "details": []}

    # 연결을 받았으면 그대로 쓰고, 아니면 새로 연다. `connect()` 는 블록이 끝날 때
    # 커밋하고 닫으므로 `with` 밖에서 붙잡고 있으면 안 된다.
    #
    # ⚠️ 아래 `collect_log.mark_*` 에 **이 연결을 넘긴다.** 안 넘기면 대장이 별도
    #    연결로 `BEGIN IMMEDIATE` 를 잡는데, 우리가 이미 쓰기 잠금을 쥐고 있어서
    #    `database is locked` 로 죽는다(실측). 넘기면 같은 트랜잭션에 얹혀
    #    적재와 대장이 함께 커밋된다 — 중간에 죽어도 둘이 어긋나지 않는다.
    with (nullcontext(conn) if conn is not None else connect()) as 사용연결:
        for indicator_id in ids:
            target = target_of(indicator_id)
            한줄 = {"id": indicator_id, "status": None, "rows": 0, "note": ""}
            try:
                series = ecos_data.fetch_series(indicator_id, years=years)

                # 🔴 빈 응답을 오류보다 **먼저** 본다. 순서를 뒤집으면 "그 지표를 ECOS 가
                #    안 주는 것" 이 "받다 실패한 것" 으로 남아 영원히 다시 부르게 된다.
                #    재무 수집에서 같은 순서 문제를 겪었다.
                if not (series.get("periods") or []):
                    collect_log.mark_empty(
                        SOURCE, target, note="ECOS 가 이 지표에 값을 주지 않는다",
                        conn=사용연결)
                    한줄["status"] = "empty"
                    결과["empty"] += 1
                    결과["details"].append(한줄)
                    if progress:
                        progress(한줄)
                    continue

                # 잘렸으면 담지 않는다. 잘림은 **오래된 쪽이 사라지는** 방식이라
                # 담아 버리면 앞부분이 조용히 비고, 행 수 검사로는 잡히지 않는다.
                if series.get("truncated"):
                    raise ecos_data.EcosError(
                        f"응답이 상한({ecos_data.MAX_ROWS}행)에서 잘렸습니다. "
                        f"앞쪽 기간이 사라진 채로 담기므로 멈춥니다.\n"
                        f"  할 일: ecos_data.MAX_ROWS 를 올리거나 years 를 나눠 받으세요.",
                        status=502)

                담은수 = save(series, conn=사용연결)
                collect_log.mark_ok(
                    SOURCE, target, rows=담은수,
                    cursor=(series.get("periods") or [None])[-1],
                    note=f"{series.get('cycle_name', '')} {series.get('label', '')}".strip(),
                    conn=사용연결)
                한줄["status"] = "ok"
                한줄["rows"] = 담은수
                결과["ok"] += 1
                결과["rows"] += 담은수

            except Exception as exc:                     # noqa: BLE001 — 한 지표의 실패가
                collect_log.mark_error(                  # 나머지를 막지 않게 한다
                    SOURCE, target, note=str(exc)[:400], conn=사용연결)
                한줄["status"] = "error"
                한줄["note"] = str(exc)[:200]
                결과["error"] += 1

            결과["details"].append(한줄)
            if progress:
                progress(한줄)

    return 결과


# ==================================================
# 4. 조회
# ==================================================
def as_of(indicator_id: str, as_of_date: str,
          conn: Optional[sqlite3.Connection] = None) -> Optional[Dict]:
    """`as_of_date`(YYYYMMDD) 시점에 **알 수 있었던** 가장 최근 값.

    거시를 쓰는 정문이다. `period` 가 아니라 `known_at` 으로 거르는 것이 요점 —
    기준월로 거르면 아직 발표되지 않은 값이 딸려 온다.
    """
    sql = (
        "SELECT indicator_id, period, cycle, value, known_at, unit "
        "FROM macro_series "
        "WHERE indicator_id = ? AND known_at <= ? AND value IS NOT NULL "
        "ORDER BY known_at DESC, period DESC LIMIT 1"
    )
    with (nullcontext(conn) if conn is not None else connect()) as 사용연결:
        row = 사용연결.execute(sql, (indicator_id, as_of_date)).fetchone()
    if not row:
        return None
    return {
        "indicator_id": row[0], "period": row[1], "cycle": row[2],
        "value": row[3], "known_at": row[4], "unit": row[5],
    }


def status(conn: Optional[sqlite3.Connection] = None) -> List[Dict]:
    """지표별로 무엇이 얼마나 들어 있는지. 수집 현황 화면과 품질 검사가 쓴다."""
    sql = (
        "SELECT indicator_id, cycle, COUNT(*), MIN(period), MAX(period), "
        "       MIN(known_at), MAX(known_at), SUM(value IS NULL) "
        "FROM macro_series GROUP BY indicator_id, cycle ORDER BY indicator_id"
    )
    with (nullcontext(conn) if conn is not None else connect()) as 사용연결:
        rows = 사용연결.execute(sql).fetchall()
    return [{
        "indicator_id": r[0], "cycle": r[1], "rows": r[2],
        "period_min": r[3], "period_max": r[4],
        "known_at_min": r[5], "known_at_max": r[6], "null_values": r[7],
    } for r in rows]


def all_indicators() -> Iterable[str]:
    """큐레이션 지표 이름들. 화면·CLI 가 선택지를 만들 때 쓴다."""
    return [spec["id"] for spec in ecos_data.INDICATORS]
