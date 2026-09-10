"""공공데이터포털 주식배당정보를 `dividend` 표에 채운다. (수집 → 저장)

`scripts/fetch_data_go_kr.py --dividend` 가 이 모듈을 부른다. 시세의 `krx_store`,
재무의 `dart_store`, 거시의 `macro_store`, 신원의 `identity_store` 와 짝이다.

## 왜 배당을 받나 — 우리 수익률에 배당이 없다

학계 표준인 CRSP 는 `RET`(배당 포함)와 `RETX`(배당 제외)를 **나눠서** 준다. 우리가 가진
것은 `RETX` 뿐이다. FinanceDataReader·pykrx 의 수정주가는 액면분할·무상증자·주식배당까지만
펴고 **현금배당은 안 편다**(`ingest/store/adj_price.py` 와 같은 범위).

크기를 쟀다 — 배당락일의 평균 일간수익률이 평소보다 **1.4665%p 낮고 17년 전부 음수**다
(`docs/데이터파트/version4.2/퀀트기준_대조.md` §3.2). 개별종목 중립대가 ±2% 이므로,
5거래일 라벨 창에 배당락일이 들어오면 그 라벨이 조용히 아래로 밀린다.
**"하락" 으로 세어진 것 중 일부는 배당을 받은 것**이다.

⚠️ **이 숫자는 한 번 틀렸다.** 배당 자료가 없던 v4.1 까지는 *"KOSPI 시총 상위 100 ·
연말 마지막 거래일 대비 위치"* 라는 대용치로 쟀고 그 값이 **−0.455%p · 15년 중 11년
음수**였다. 실제 배당락일로 다시 재니 **3.2배**였다 — 연말만 봐서 분기배당을 놓치고,
상위 100 만 봐서 고배당 중소형주를 놓쳤기 때문이다. 대용치는 **크기를 줄이는 쪽으로
틀린다.** 원자료가 생기면 다시 잰다.

## 🔴 이 모듈의 존재 이유 — 기준일은 거래일이 아니다

포털이 주는 것은 **배당기준일**(`dvdnBasDt`)이다. 12월 결산법인이면 12월 31일이고 그날은
휴장이다. 실측하면 12월 기준일 중 거래일인 것은 **0.3%** 뿐이다. 값이 실제로 움직이는 날은
**배당락일**이고, 그것은 결제(T+2) 때문에 기준일에서 두 걸음 앞이다.

    ex_date = prev_session(prev_session(기준일, inclusive=True))

안쪽이 *"기준일까지 마지막으로 열린 날"*, 바깥이 *"그 하루 앞"* 이다. 두 걸음인 이유는
기준일 주주명부에 오르려면 마지막 거래일까지 **결제가 끝나야** 하기 때문이다 — 그러려면
마지막 거래일 이틀 전까지 사야 하고, 따라서 **마지막 거래일 전날부터는 사도 못 받는다.**
그날이 배당락일이다.

    2023-12-31(기준) → 12-28(폐장) → **12-27 배당락**   ← 실제와 일치
    2024-06-30(기준) → 06-28       → **06-27 배당락**

이 규칙은 계산값이므로 행마다 `ex_date_rule` 을 남긴다 — 규칙을 바꾸면 어느 행이 옛
규칙으로 계산됐는지 알 수 있어야 한다. 거시(`macro_series.known_at`)에서 같은 문제를 겪었다.

## 무엇을 담지 않나

- **금액을 지어내지 않는다.** 옛 행은 금액이 비고 배당률(액면 대비 %)만 있는데, 응답의
  액면가는 **적재 시점 값**이라(삼성전자 1987년 행에도 100원) 곱하면 틀린다. 빈 채로 둔다.
  2018년 이후로는 금액 채움이 98% 를 넘는다(연도별 실측은 `coverage()`).
- **우리 종목에 못 붙는 행도 담는다.** ISIN 이 우리 `stock_base_info` 에 없으면 `code` 가
  비지만 행 자체는 남긴다 — 나중에 상장 이력이 늘면 다시 붙일 수 있고, "없는 것" 과
  "안 받은 것" 을 구별하려면 원문이 있어야 한다.
"""

from __future__ import annotations

import sqlite3
from contextlib import nullcontext
from typing import Dict, List, Optional, Sequence, Tuple

from common.paths import krx_db_path
from common.trading_calendar import CalendarOutOfRange, now_kst_iso, prev_session
from ingest.clients import data_go_kr
from ingest.store import collect_log
from ingest.store.krx_store import connect

#: `dividend` 표의 칸 순서. INSERT 와 튜플을 만들 때 함께 쓴다.
DIVIDEND_COLUMNS: Tuple[str, ...] = (
    "isin_cd", "dvdn_bas_dt", "dvdn_rcd", "code", "crno", "corp_nm", "item_nm",
    "scrs_itms_kcd_nm", "stac_md", "dvdn_rcd_nm", "cash_pay_dt", "stck_hndv_dt",
    "genr_dvdn_amt", "grdn_dvdn_amt", "genr_cash_dvdn_rt", "genr_dvdn_rt",
    "cash_grdn_dvdn_rt", "grdn_dvdn_rt", "par_price_at_load",
    "ex_date", "ex_date_rule", "src_bas_dt", "fetched_at",
)

_INSERT = (
    f"INSERT OR REPLACE INTO dividend ({', '.join(DIVIDEND_COLUMNS)}) "
    f"VALUES ({', '.join('?' * len(DIVIDEND_COLUMNS))})"
)

#: 배당락일 계산 규칙의 이름. 규칙이 바뀌면 **이 문자열도 바꾼다** — 옛 규칙으로 계산된
#: 행을 골라내는 유일한 방법이다.
EX_DATE_RULE = "prev_session(prev_session(bas_dt, inclusive=True))"

#: 수집 대장에 남기는 이름.
COLLECT_SOURCE = "data_go_kr"
COLLECT_TARGET = "dividend"


class DividendStoreError(RuntimeError):
    """배당 적재가 멈춰야 하는 상황."""


# ==================================================
# 1. 배당락일 — 달력이 있어야 계산할 수 있다
# ==================================================
def ex_date_for(dvdn_bas_dt: str, db_path=None) -> Optional[str]:
    """배당기준일에서 배당락일을 낸다. 달력 밖이면 `None`.

    달력 밖에 `None` 을 주는 것은 `is_session` 의 규약(세운다)과 다르다. 이유가 있다 —
    배당 자료는 **1986년부터** 오는데 우리 거래일 달력은 2010년부터다. 옛 행마다 예외를
    세우면 전량 적재가 첫 행에서 멈춘다. 대신 `ex_date` 를 비우고 `ex_date_rule` 도 비워
    **"계산하지 못했다"** 를 남긴다. 지어낸 값보다 빈 칸이 눈에 띈다.
    """
    try:
        마지막거래일 = prev_session(dvdn_bas_dt, db_path, inclusive=True)
        return prev_session(마지막거래일, db_path)
    except CalendarOutOfRange:
        return None


# ==================================================
# 2. 종목코드 붙이기 — ISIN 은 우리 표에 있다
# ==================================================
def isin_to_code(conn: sqlite3.Connection) -> Dict[str, str]:
    """`stock_base_info` 에서 ISIN → 종목코드 지도를 만든다.

    🔴 **한 ISIN 이 두 코드에 붙는 경우가 있다**(2026-09-09 실측 · 3,699쌍 중 22쌍).
       그대로 조인하면 배당 한 줄이 두 줄로 불어난다 — 행 수는 늘고 아무도 안 본다.
       **가장 최근에 관측된 코드** 하나만 남긴다.
    """
    지도: Dict[str, str] = {}
    쿼리 = (
        "SELECT isin_cd, code, MAX(bas_dd) FROM stock_base_info "
        "WHERE isin_cd IS NOT NULL AND isin_cd <> '' GROUP BY isin_cd, code "
        "ORDER BY isin_cd, MAX(bas_dd)"
    )
    for isin, code, _ in conn.execute(쿼리):
        지도[isin] = code                     # 마지막 관측이 이긴다 (ORDER BY 로 보장)
    return 지도


# ==================================================
# 3. 담기
# ==================================================
def save(rows: Sequence[Dict], conn: Optional[sqlite3.Connection] = None,
         *, db_path=None) -> int:
    """`dividend` 에 담는다. 담은 행 수를 돌려준다.

    부르는 쪽이 준 행에 `code`·`ex_date` 가 없으면 **여기서 채운다.** 둘 다 저장소를
    알아야 만들 수 있는 값이라, 외부 연동 계층(`data_go_kr`)이 아니라 이쪽 일이다.

    🔴 키가 빈 행은 담지 않고 세운다. 빈 키로 넣으면 서로를 덮어써서 **행 수는 그럴듯한데
       내용이 사라진다** — 재무에서 `account_detail` 을 빠뜨려 6.4%가 조용히 사라진 적이 있다.
    """
    if not rows:
        return 0
    fetched = now_kst_iso()
    ctx = nullcontext(conn) if conn is not None else connect()
    with ctx as c:
        지도 = isin_to_code(c)
        묶음: List[Tuple] = []
        for r in rows:
            if not r.get("isin_cd") or not r.get("dvdn_bas_dt") or not r.get("dvdn_rcd"):
                raise DividendStoreError(
                    f"키가 빈 행이 있다: isin_cd={r.get('isin_cd')!r} "
                    f"dvdn_bas_dt={r.get('dvdn_bas_dt')!r} dvdn_rcd={r.get('dvdn_rcd')!r}\n"
                    "  할 일: data_go_kr.parse_dividend_row 는 이런 행에 None 을 준다 — "
                    "부르는 쪽에서 걸렀는지 확인한다."
                )
            채움 = dict(r)
            채움.setdefault("code", None)
            if 채움["code"] is None:
                채움["code"] = 지도.get(채움["isin_cd"])
            if "ex_date" not in 채움 or 채움["ex_date"] is None:
                채움["ex_date"] = ex_date_for(채움["dvdn_bas_dt"], db_path)
            채움["ex_date_rule"] = EX_DATE_RULE if 채움["ex_date"] else None
            채움["fetched_at"] = 채움.get("fetched_at") or fetched
            묶음.append(tuple(채움.get(칸) for 칸 in DIVIDEND_COLUMNS))
        c.executemany(_INSERT, 묶음)
    return len(묶음)


def sync_all(*, key: Optional[str] = None, db_path=None,
             conn: Optional[sqlite3.Connection] = None) -> Dict:
    """배당 전량을 받아 담는다. 한 번에 끝난다 — 날짜별로 돌 필요가 없다.

    2026-09-09 실측 **71,681행 · 72콜**. 시세처럼 4,000일을 도는 수집과는 자릿수가 다르다.
    `basDt` 가 이벤트 날짜가 아니라 적재일이라 날짜 축으로 훑을 이유가 없기 때문이다.
    """
    행들 = data_go_kr.fetch_dividends(key=key)
    if not 행들:
        collect_log.mark_empty(COLLECT_SOURCE, COLLECT_TARGET,
                               note="포털이 0건을 줬다", db_path=db_path)
        return {"rows": 0, "saved": 0}
    담김 = save(행들, conn, db_path=db_path)
    collect_log.mark_ok(COLLECT_SOURCE, COLLECT_TARGET, rows=담김,
                        cursor=max(r["dvdn_bas_dt"] for r in 행들), db_path=db_path)
    return {"rows": len(행들), "saved": 담김}


# ==================================================
# 4. 현황 — 받은 뒤 무엇을 얻었는지 센다
# ==================================================
def coverage(conn: Optional[sqlite3.Connection] = None, *,
             db_path=None, since: str = "20100101") -> Dict:
    """무엇이 들어왔는지 센다. 수집 뒤 보고와 시험이 함께 쓴다.

    행 수만 세지 않는다 — **금액이 실제로 있는 비율**과 **배당락일을 계산한 비율**을 함께
    본다. 둘 중 하나라도 낮으면 그 자료로는 총수익 축을 못 만든다.
    """
    경로 = db_path or krx_db_path()
    ctx = nullcontext(conn) if conn is not None else sqlite3.connect(
        f"file:{str(경로).replace(chr(92), '/')}?mode=ro", uri=True)
    with ctx as c:
        전체 = c.execute("SELECT COUNT(*) FROM dividend").fetchone()[0]
        if not 전체:
            return {"rows": 0}
        현금 = (
            "dvdn_rcd_nm IN ('현금배당', '동시배당') AND dvdn_bas_dt >= ?"
        )
        묶 = c.execute(
            f"SELECT COUNT(*), "
            f"       SUM(code IS NOT NULL), "
            f"       SUM(ex_date IS NOT NULL), "
            f"       SUM(genr_dvdn_amt > 0) "
            f"FROM dividend WHERE {현금}", (since,)
        ).fetchone()
        연도 = c.execute(
            f"SELECT substr(dvdn_bas_dt, 1, 4) AS y, COUNT(*), SUM(genr_dvdn_amt > 0) "
            f"FROM dividend WHERE {현금} AND code IS NOT NULL "
            f"GROUP BY y ORDER BY y", (since,)
        ).fetchall()
        구간 = c.execute(
            "SELECT MIN(dvdn_bas_dt), MAX(dvdn_bas_dt) FROM dividend"
        ).fetchone()
    셈, 코드있음, 락일있음, 금액있음 = 묶
    return {
        "rows": int(전체),
        "span": tuple(구간),
        "cash_rows_since": int(셈),
        "with_code": int(코드있음 or 0),
        "with_ex_date": int(락일있음 or 0),
        "with_amount": int(금액있음 or 0),
        "by_year": [(y, int(n), int(a or 0)) for y, n, a in 연도],
    }
