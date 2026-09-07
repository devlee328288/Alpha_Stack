"""KRX 종목기본정보를 `stock_base_info` 에 채운다. (수집 → 저장)

`scripts/fetch_base_info.py` 가 이 모듈을 부른다. 시세의 `krx_store`, 신원의
`identity_store`, 재무의 `dart_store` 와 짝이고 같은 DB 파일의 다른 표에 담는다.

## 무엇을 담나 — **보통주인지 우선주인지의 정본**

우리는 지금 종목명이 '우' 로 끝나는지로 우선주를 추측하고 있다. 그게 틀린다.
실측 2026-09-03, 세 시장 × 세 날짜(20150102 · 20200102 · 20260901) 전수 대조:

    미래에셋대우 · 연우 · 동우 · 신우 · 성우 · 에코글로우 · 이오플로우
    → 이름이 '우' 로 끝나는 **보통주 7종**을 우선주로 잘못 뺀다

006800(미래에셋대우)은 20200102 코스피 시총 **48위**다. 모델 파트가 쓰기로 한
"KOSPI 보통주 시가총액 상위 50" 후보에서 조용히 빠진다.

🔴 이 오류는 **이름이 바뀌는 구간에만** 나타난다 — 대우증권(정상) → 미래에셋대우(깨짐)
   → 미래에셋증권(정상). 오늘 유가 943종만 세면 어긋남이 **0건**이라 표본 검증으로는
   절대 안 잡힌다. 세 날짜로 넓혔더니 그제서야 7종이 나왔다.

## 🔴 이 표는 오늘 스냅샷이 아니라 **이력**이다

같은 엔드포인트를 다른 날짜로 부르면 다른 답이 온다 (실측 2026-09-03):

| 기준일 | 유가 | 코스닥 |
|---|---:|---:|
| 20150102 | 899 | 1,065 |
| 20200102 | 916 | 1,408 |
| 20260901 | 943 | 1,822 |

2015 에만 있고 지금은 없는 종목이 **159종**(상장폐지), 공통 740종 중 상장주식수가
다른 종목이 **534종**이다. 값까지 그날 것이다.

그래서 *"2026년에 확인한 값을 2015년에도 같았다고 적용"* 하지 않아도 된다 —
2015년 값을 2015년에 직접 물어볼 수 있다. `stock_identity` 가 겪는 문제(포털의
`basDt` 목록에 나중에 상장된 종목이 섞이는 것)가 여기서는 없다.

## 🔴 KONEX 는 받지 않는다

`daily_price` 에 KONEX 가 **한 행도 없다** (KOSPI 1,251종 · KOSDAQ 2,448종뿐 · 실측).
우리 유니버스는 `daily_price` 와의 교집합이라, KONEX 를 받아도 붙을 데가 없고
호출만 1/3 늘어난다. 필요해지면 `MARKETS` 에 한 줄 더하면 된다.

## 두 출처의 유니버스가 정확히 같다

20260901 실측 — 유가 시세 943 · 기본정보 943 · **차이 0**, 코스닥 1,822 · 1,822 ·
**차이 0**. 포털 목록과 달리 우선주도 외국기업도 다 들어 있어 조인이 깨끗하다.

## known_at — 보수적으로 다음 거래일

주권종류·상장일은 상장 공고에 이미 실리므로 당일에도 알 수 있었다고 볼 여지가 있다.
그래도 `bas_dd` 의 **다음 거래일**로 미룬다. 늦게 아는 것은 성능을 낮출 뿐이지만,
일찍 안 것으로 잘못 적으면 미래참조가 되고 그건 **에러 없이 성능만 좋아진다.**

`stock_identity` 와 같은 규칙이라 두 표를 같은 `as_of` 로 잘라도 어긋나지 않는다.
"""

from __future__ import annotations

import sqlite3
from contextlib import nullcontext
from typing import Dict, List, Optional, Sequence, Tuple

from common.paths import krx_db_path
from common.trading_calendar import CalendarOutOfRange, next_session, now_kst_iso
from ingest.clients import krx_data
from ingest.store import collect_log
from ingest.store.krx_store import connect

#: 수집 대장에 적히는 출처 이름. 시세·지수와 같은 KRX 예산을 쓴다.
SOURCE = "krx"

#: `known_at` 을 어떤 규칙으로 냈는지. 행마다 남겨 옛 규칙을 가려낼 수 있게 한다.
KNOWN_RULE = "basDd+1session"

#: 🔴 받을 시장. KONEX 를 뺀 이유는 모듈 문서 참조 — `daily_price` 에 한 행도 없다.
MARKETS: Tuple[str, ...] = ("KOSPI", "KOSDAQ")

#: 스키마를 이미 확인했나. 한 프로세스에서 마이그레이션을 매번 돌릴 이유가 없다.
_schema_ready = False

COLUMNS: Tuple[str, ...] = (
    "bas_dd", "code", "isin_cd", "isu_nm", "isu_abbrv", "isu_eng_nm",
    "list_dd", "market", "secugrp_nm", "sect_tp_nm", "kind_stkcert_tp_nm",
    "parval", "list_shrs", "known_at", "known_rule", "fetched_at",
)

#: `INSERT OR REPLACE` 를 쓰는 이유 — 같은 날짜를 다시 받아도 행이 늘지 않아야 한다.
#:
#: ⚠️ 그 대신 **덮어쓴다.** 행 수를 세는 검사로는 덮어쓰기를 못 잡는다. 값이 바뀌었는지
#:    보려면 `scripts/verify_base_info.py` 처럼 값을 맞대는 검사가 따로 있어야 한다.
_INSERT = (
    f"INSERT OR REPLACE INTO stock_base_info ({', '.join(COLUMNS)}) "
    f"VALUES ({', '.join('?' * len(COLUMNS))})"
)


class BaseInfoStoreError(RuntimeError):
    """담는 도중 세운다. 무엇을 해야 하는지까지 문구에 담는다."""


# ==================================================
# 1. known_at — 계산값이라는 것을 잊지 않는다
# ==================================================
def known_at_for(bas_dd: str, db_path=None) -> str:
    """`bas_dd` 다음 거래일.

    🔴 **날짜 계산으로 다음 날을 구하지 않는다.** 공휴일·임시휴장이 있어서 실측
       달력을 쓴다 (`stock_identity` 와 같은 이유·같은 함수).

    달력 밖이면 **세운다** — 지어내면 아직 열리지 않은 장에 자료를 붙이게 된다.
    """
    try:
        return next_session(bas_dd, db_path)
    except CalendarOutOfRange as exc:
        raise BaseInfoStoreError(
            f"{bas_dd} 의 다음 거래일을 몰라 known_at 을 정할 수 없다.\n"
            f"  {exc}\n"
            "  할 일: 그 구간 시세를 먼저 받아 달력을 넓히거나, 그 날짜를 건너뛴다."
        ) from exc


# ==================================================
# 2. 담기
# ==================================================
def save(rows: Sequence[Dict], conn: Optional[sqlite3.Connection] = None) -> int:
    """`stock_base_info` 에 담는다. 담은 행 수를 돌려준다.

    🔴 키가 될 값(`bas_dd`·`code`)이 없는 행은 **담지 않고 세운다.** 빈 키로 넣으면
       서로를 덮어써서, 행 수는 그럴듯한데 내용이 사라진다.
    """
    if not rows:
        return 0
    fetched = now_kst_iso()
    묶음 = []
    for r in rows:
        if not r.get("bas_dd") or not r.get("code"):
            raise BaseInfoStoreError(
                f"키가 빈 행이 있다: bas_dd={r.get('bas_dd')!r} code={r.get('code')!r}\n"
                "  흔한 원인: 응답의 ISU_SRT_CD 가 비어 왔다.\n"
                "  할 일: krx_data.normalize_base_info_row 를 확인하고, 그 행을 뺀다."
            )
        if not r.get("known_at"):
            raise BaseInfoStoreError(
                f"known_at 이 빈 행이 있다: bas_dd={r['bas_dd']} code={r['code']}\n"
                "  할 일: sync_day 가 known_at 을 채워 넘기는지 확인한다. "
                "빈 채로 담으면 as_of 가 그 행을 영원히 못 거른다."
            )
        묶음.append(tuple(
            {**r, "fetched_at": r.get("fetched_at") or fetched}.get(c)
            for c in COLUMNS))

    ctx = nullcontext(conn) if conn is not None else connect()
    with ctx as c:
        c.executemany(_INSERT, 묶음)
    return len(묶음)


# ==================================================
# 3. 하루치 받아 담기
# ==================================================
def sync_day(bas_dd: str, market: str = "KOSPI", *, db_path=None,
             conn: Optional[sqlite3.Connection] = None) -> Dict:
    """한 기준일·한 시장의 종목기본정보를 받아 담는다.

    돌려주는 것: `{"bas_dd", "market", "rows", "status"}`.

    `status` 는 수집 대장과 같은 말을 쓴다 — `ok` · `empty` · `error`.
    **`empty` 를 기록하는 이유**는 시세 수집과 같다: "받아 봤더니 없었다"(휴장)와
    "아직 안 받았다" 를 구별하지 않으면 휴장일마다 영원히 다시 요청한다.
    """
    대상 = f"base_info:{market}:{bas_dd}"

    known = known_at_for(bas_dd, db_path)
    try:
        행들 = krx_data.fetch_base_info(bas_dd, market)
    except krx_data.KrxQuotaExhausted:
        # 🔴 고장이 아니라 정상적인 하루의 끝이다. 대장에 남기고 그대로 올려보낸다 —
        #    부르는 쪽이 멈추고 내일 이어받아야 한다.
        collect_log.mark_quota_exhausted(SOURCE, 대상)
        raise
    except krx_data.KrxError as exc:
        collect_log.mark_error(SOURCE, 대상, note=str(exc)[:500])
        raise

    if not 행들:
        collect_log.mark_empty(SOURCE, 대상, note="0건 (휴장 또는 자료 없음)")
        return {"bas_dd": bas_dd, "market": market, "rows": 0, "status": "empty"}

    담을것 = [{**r, "known_at": known, "known_rule": KNOWN_RULE} for r in 행들]
    담은수 = save(담을것, conn)
    collect_log.mark_ok(SOURCE, 대상, rows=담은수, cursor=bas_dd)
    return {"bas_dd": bas_dd, "market": market, "rows": 담은수, "status": "ok"}


# ==================================================
# 4. 조회
# ==================================================
def pending_days(markets: Sequence[str] = MARKETS,
                 conn: Optional[sqlite3.Connection] = None) -> List[Tuple[str, str]]:
    """아직 안 받은 (기준일, 시장) 쌍. 이어 받을 때 쓴다.

    🔴 달력은 `trading_calendar` 에서 **DISTINCT 로** 꺼낸다. 그 표는 시장마다 한 줄씩
       있어 12,306행이지만 날짜는 4,102일이다. DISTINCT 를 빼면 같은 날을 세 번 받는다.

    이미 `ok` 든 `empty` 든 대장에 남은 대상은 건너뛴다 — `empty` 도 '받아 봤다' 이다.
    """
    ctx = nullcontext(conn) if conn is not None else connect()
    with ctx as c:
        날짜들 = [r[0] for r in c.execute(
            "SELECT DISTINCT bas_dd FROM trading_calendar ORDER BY bas_dd")]
        끝난것 = {r[0] for r in c.execute(
            "SELECT target FROM collect_log "
            "WHERE source = ? AND target LIKE 'base_info:%' "
            "AND status IN ('ok', 'empty')", (SOURCE,))}

    남은것 = []
    for d in 날짜들:
        for m in markets:
            if f"base_info:{m}:{d}" not in 끝난것:
                남은것.append((d, m))
    return 남은것


def universe_rows(bas_dd: str, *, market: str = "KOSPI", common_only: bool = True,
                  known_by: Optional[str] = None,
                  conn: Optional[sqlite3.Connection] = None) -> List[Dict]:
    """그 거래일의 종목 목록 + 시가총액. `as_of` 는 모른다 — `supply/` 가 씌운다.

    🔴 **`stock_base_info` 를 그날 목록으로 쓰지 않고 `daily_price` 와 교집합을 낸다.**
       시가총액이 시세에만 있어서이기도 하지만, 그보다 **거래된 종목만 남겨야** 하기
       때문이다. 기본정보는 '상장돼 있다' 를 말할 뿐 그날 거래됐는지는 말하지 않는다.

    🔴 **기본정보는 그날 것이 없으면 그 이전 가장 최근 것을 쓴다.**
       수집이 아직 다 안 됐거나 앞으로 주기를 성기게 바꿔도 답이 나와야 한다.
       주권종류는 실측상 15년간 바뀐 종목이 0건이라 이 대입이 안전하다. 다만
       `known_by` 로 **그때 알 수 있었던 행만** 보게 막는다 — 안 그러면 미래의
       스냅샷이 과거 판정에 쓰인다.

    `common_only=True` 면 보통주만. 무엇이 보통주인지는 `krx_data.COMMON_STOCK_KIND`
    하나로 정한다 — 판정 기준이 두 곳에 있으면 언젠가 갈라진다.
    """
    from ingest.clients.krx_data import COMMON_STOCK_KIND

    # 그 종목의, 이 날짜 이하의, (알 수 있었던) 가장 최근 기본정보를 고른다.
    # ⚠️ SQLite 는 lateral join 을 못 한다. JOIN 조건 안의 상관 서브쿼리로 쓰면
    #    `idx_base_info_code(code, bas_dd)` 를 그대로 타서 종목당 인덱스 조회 한 번이다.
    고른날 = ("SELECT MAX(b.bas_dd) FROM stock_base_info b "
              " WHERE b.code = p.code AND b.bas_dd <= ?")
    고른날값 = [bas_dd]
    if known_by:
        고른날 += " AND b.known_at <= ?"
        고른날값.append(known_by)

    sql = (
        "SELECT p.bas_dd, p.code, p.name, p.market, p.close, p.adj_close, "
        "       p.market_cap, p.listed_shares, "
        "       i.kind_stkcert_tp_nm, i.list_dd, i.isin_cd, i.isu_abbrv, "
        "       i.bas_dd AS info_bas_dd, i.known_at AS info_known_at "
        "  FROM daily_price p "
        "  LEFT JOIN stock_base_info i "
        "         ON i.code = p.code "
        f"       AND i.bas_dd = ({고른날}) "
        " WHERE p.bas_dd = ? AND p.market = ? "
        "   AND p.market_cap IS NOT NULL AND p.market_cap > 0"
    )
    # 파라미터는 SQL 에 나타난 순서다 — JOIN 안의 서브쿼리가 WHERE 보다 앞이다.
    params = 고른날값 + [bas_dd, market]

    ctx = nullcontext(conn) if conn is not None else connect()
    with ctx as c:
        c.row_factory = sqlite3.Row
        rows = [dict(r) for r in c.execute(sql, params).fetchall()]

    if common_only:
        rows = [r for r in rows
                if (r.get("kind_stkcert_tp_nm") or "").strip() == COMMON_STOCK_KIND]
    rows.sort(key=lambda r: -(r["market_cap"] or 0))
    return rows


def status(conn: Optional[sqlite3.Connection] = None) -> Dict:
    """지금 표에 무엇이 얼마나 있나."""
    ctx = nullcontext(conn) if conn is not None else connect()
    with ctx as c:
        기본 = c.execute(
            "SELECT COUNT(*), COUNT(DISTINCT bas_dd), COUNT(DISTINCT code), "
            "MIN(bas_dd), MAX(bas_dd) FROM stock_base_info").fetchone()
        종류 = dict(c.execute(
            "SELECT kind_stkcert_tp_nm, COUNT(DISTINCT code) "
            "FROM stock_base_info GROUP BY kind_stkcert_tp_nm").fetchall())
    return {
        "rows": 기본[0], "days": 기본[1], "codes": 기본[2],
        "first": 기본[3], "last": 기본[4], "kinds": 종류,
    }


def ensure_schema() -> None:
    """표가 없으면 만든다 (마이그레이션 v11). 한 프로세스에서 한 번만.

    🔴 `krx_store.init_db()` 로는 안 된다 — 그쪽 `SCHEMA` 는 시세 표만 만든다.
       `stock_base_info` 는 **마이그레이션이 만든다.** DDL 을 두 곳에 두면 언젠가
       갈라지기 때문이다 (`stock_identity` 와 같은 이유·같은 방식).
    """
    global _schema_ready
    if _schema_ready:
        return
    from ingest.store.migrations import migrate_path
    migrate_path(krx_db_path())
    _schema_ready = True
