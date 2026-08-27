"""KRX 지수 일별시세 저장소 (저장소 계층)

`krx_store` 가 **종목**을 맡는다면 이 모듈은 **지수**를 맡는다.
1차 프로젝트의 예측 대상(KOSPI200)이 여기서 나온다 (ADR-AS-0003).

왜 모듈을 따로 두나
------------------
`krx_store` 는 이미 704줄이고 SQLite·Postgres 이중 읽기 경로를 안고 있다. 거기에 지수를
얹으면 "종목이냐 지수냐" × "SQLite냐 Postgres냐" 네 갈래가 한 파일에 생긴다.
지수는 읽기 경로가 하나뿐이므로(팀 Postgres 가 아직 없다 → 회의안건 A-1) 따로 둔다.

**DB 파일은 같이 쓴다.** `krx_store.DB_PATH` 하나에 표만 나눠 담는다 — 파일이 둘이면
백업·이동·스냅샷 해시가 둘로 갈라진다.

종목 저장소와 다른 점
--------------------
| | 종목 (`krx_store`) | 지수 (이 모듈) |
|---|---|---|
| 가격 타입 | **정수** (원 단위) | **실수** (지수 포인트, 예 1096.25) |
| 하루 행 수 | 약 2,700 | 약 91 (KOSPI 51 + KOSDAQ 40) |
| 날짜당 콜 | 2회 (시장 2개) | 1회/시장 |
| 키 | (날짜, 종목코드) | (날짜, 지수명) |

⚠️ **지수 가격을 정수로 깎으면 하루 등락이 통째로 사라진다.** 코스피 200 의 하루 변동은
   보통 소수점 아래 몇 자리다. 이 표가 REAL 인 이유다.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

from common.trading_calendar import to_iso, today_kst, trading_days
from ingest.clients import krx_data as api
from ingest.store import collect_log
from ingest.store.krx_store import (
    DB_PATH,  # 같은 DB 파일을 쓴다
    connect,
)
from ingest.store.sqlite_db import write_lock  # 같은 파일에 쓰므로 자물쇠도 같은 것

# 수집 대장에 남길 출처 이름. 종목 쪽과 갈라 둔다 — 같은 날짜라도 "종목은 받았고
# 지수는 안 받은" 상태가 실재하므로 한 이름으로 묶으면 서로의 진행을 지운다.
COLLECT_SOURCE = "krx_index"


def _target(bas_dd: str, market: str) -> str:
    """대장의 대상 이름. 시장과 날짜를 한 문자열로 묶는다 (표의 키가 둘이 아니라 하나다)."""
    return f"{market}/{bas_dd}"

# 지수를 받을 시장. **기본은 KOSPI 하나다.**
#
# ⚠️ 왜 KOSDAQ 을 기본에서 뺐나 — 하루 한도 때문이다. KRX 는 인증키당 1일 10,000회를
#    허용한다(이용약관 제8조 ④). 16년 백필은 시장당 4,343콜이라 둘을 켜면 8,686콜이고,
#    같은 날 종목 백필(8,686콜)까지 돌리면 한도를 넘는다.
#    1차 예측 대상은 KOSPI200 하나이므로 그것부터 받고, 코스닥 150 은 여유가 있을 때 켠다.
DEFAULT_MARKETS: Tuple[str, ...] = ("KOSPI",)

# 동시 호출 수. **지수 엔드포인트는 1 이다** — 종목 쪽(`krx_store.sync` 기본 6)과 다르다.
#
# ⚠️ 실측 2026-08-26. 지수 엔드포인트는 동시 호출을 하면 멀쩡한 키에도 401 을 뱉는다.
#    같은 날 같은 키로 잰 결과다:
#
#      워커 6 → 1,344일 받고 차단          워커 4 → 100일 받고 차단
#      워커 1 → 계속 진행 (1,362 → 1,933일)  순차 10회 연속 → 10/10 성공
#
#    KRX 가 429(Too Many Requests)가 아니라 **401 로 답하기 때문에** 겉보기로는
#    "인증키가 틀렸다" 로 보인다. 그래서 원인을 키에서 찾게 되고, 키를 아무리 확인해도
#    답이 안 나온다. 실제로 이 함정에 두 번 빠졌다.
#
#    지수는 날짜당 1콜이라(종목은 2콜) 순차로도 16년이 감당된다. 속도를 얻으려고
#    이 값을 올리면 **백필이 도중에 멈추고, 멈춘 이유가 인증 문제로 보인다.**
DEFAULT_WORKERS = 1

# 저장 컬럼 순서 (INSERT 와 SELECT 가 함께 쓴다)
COLUMNS = ("bas_dd", "index_name", "index_class",
           "open", "high", "low", "close", "change", "change_rate",
           "volume", "value", "market_cap")

SCHEMA = """
CREATE TABLE IF NOT EXISTS index_price (
  bas_dd       TEXT    NOT NULL,   -- 기준일자 YYYYMMDD
  index_name   TEXT    NOT NULL,   -- 지수명 "코스피 200" (띄어쓰기 포함)
  index_class  TEXT,               -- KOSPI / KOSDAQ
  open         REAL,               -- ⚠️ 지수는 실수다. INTEGER 로 두면 등락이 사라진다
  high         REAL,
  low          REAL,
  close        REAL,
  change       REAL,               -- 전일대비
  change_rate  REAL,               -- 등락률(%)
  volume       INTEGER,            -- 누적거래량
  value        INTEGER,            -- 누적거래대금
  market_cap   INTEGER,            -- 시가총액
  PRIMARY KEY (bas_dd, index_name) -- 같은 날 같은 지수가 두 번 들어가지 않도록
);

-- 지수 하나의 시계열을 뽑을 때 쓴다. 피처 생성이 이 경로만 탄다.
CREATE INDEX IF NOT EXISTS idx_index_name_date ON index_price(index_name, bas_dd);

-- 어떤 날짜를 이미 받아봤는지. **종목 쪽 fetch_log 와 표를 나눈다** —
-- 하나로 합치면 "종목은 받았고 지수는 안 받은 날"을 표현할 수 없어
-- 두 백필이 서로의 진행을 지운다.
CREATE TABLE IF NOT EXISTS index_fetch_log (
  bas_dd     TEXT NOT NULL,
  market     TEXT NOT NULL,
  rows       INTEGER,
  fetched_at TEXT,
  PRIMARY KEY (bas_dd, market)
);
"""

# 0건으로 받은 날짜를 "확정 휴장일"로 볼 때까지 기다리는 기간(일).
# 종목 쪽(`krx_store.ZERO_ROW_RETRY_DAYS`)과 같은 이유다 — 당일 자료는 장 마감 후에 올라온다.
ZERO_ROW_RETRY_DAYS = 7

# KRX Open API 가 지수를 주기 시작하는 날. 그 이전은 예외가 아니라 **0행**으로 돌아온다.
# 실측 2026-08-26: 20091230 → 0행 · 20100104 → 있음.
DATA_START = "20100104"


def init_db() -> None:
    """표와 인덱스를 만든다. 이미 있으면 아무 일도 하지 않는다."""
    with write_lock, connect() as conn:
        conn.executescript(SCHEMA)
    # 수집 대장은 이 파일의 SCHEMA 가 아니라 마이그레이션이 만든다 — 이미 900만 행이
    # 든 DB 에 칸을 얹으려면 버전 관리가 필요하고, DDL 을 두 곳에 두면 언젠가 갈라진다.
    collect_log._ensure_schema(DB_PATH)


# ==================================================
# 1. 수집 (KRX → DB)
# ==================================================
def fetched_dates(market: str) -> set:
    """그 시장에서 다시 받을 필요가 없는 날짜 집합.

    - 데이터가 있는 날짜(rows > 0)는 언제나 건너뛴다.
    - 0건인 날짜는 **오래된 것만** 건너뛴다. 휴장일은 영원히 0건이지만 오늘·어제의 0건은
      아직 장이 안 끝났을 수 있어 다시 확인해야 한다.
    """
    init_db()
    cutoff = (today_kst() - timedelta(days=ZERO_ROW_RETRY_DAYS)).strftime("%Y%m%d")
    with connect() as conn:
        rows = conn.execute(
            "SELECT bas_dd FROM index_fetch_log "
            "WHERE market = ? AND (rows > 0 OR bas_dd < ?)",
            (market, cutoff),
        ).fetchall()
    return {row[0] for row in rows}


def _save(bas_dd: str, market: str, items: List[Dict]) -> int:
    """정규화된 한 날짜치를 저장하고 저장 건수를 돌려준다.

    ⚠️ 가격이 비어 있는 지수 행(`close is None`)도 **버리지 않고 담는다.**
       "코스피 (외국주포함)" 처럼 거래량·시가총액만 오는 지수가 실재한다(실측).
       거르는 것은 피처를 만드는 쪽의 일이지 저장소의 일이 아니다 —
       여기서 버리면 왜 없는지를 나중에 알 수 없다.
    """
    rows = [tuple([bas_dd] + [item.get(col) for col in COLUMNS[1:]]) for item in items]
    placeholders = ",".join("?" * len(COLUMNS))

    with write_lock, connect() as conn:
        conn.executemany(
            f"INSERT OR REPLACE INTO index_price ({','.join(COLUMNS)}) VALUES ({placeholders})",
            rows,
        )
        conn.execute(
            "INSERT OR REPLACE INTO index_fetch_log (bas_dd, market, rows, fetched_at) "
            "VALUES (?,?,?,?)",
            (bas_dd, market, len(rows), datetime.now().isoformat(timespec="seconds")),
        )
        # 같은 트랜잭션 안에서 대장까지 남긴다. 따로 커밋하면 그 사이에 죽었을 때
        # "저장은 됐는데 대장에는 없는" 어긋난 상태가 남고, 그러면 다음 실행이
        # 이미 있는 날짜를 다시 받는다.
        #
        # ⚠️ 0행을 `mark_empty` 로 남기는 것이 요점이다. 휴장일은 **영원히 0행**이라
        #    실패로 기록하면 배치를 돌릴 때마다 같은 날짜에 호출을 태운다.
        if rows:
            collect_log.mark_ok(COLLECT_SOURCE, _target(bas_dd, market),
                                rows=len(rows), conn=conn)
        else:
            collect_log.mark_empty(COLLECT_SOURCE, _target(bas_dd, market),
                                   note="0행 — 휴장일로 본다.", conn=conn)
    return len(rows)


def fetch_date(bas_dd: str, market: str) -> int:
    """한 거래일의 지수 전부를 받아 저장한다."""
    return _save(bas_dd, market, api.fetch_index_snapshot(bas_dd, market))


def sync(days: int = 250, workers: int = DEFAULT_WORKERS, end: Optional[str] = None,
         markets: Sequence[str] = DEFAULT_MARKETS, progress=None) -> Dict:
    """최근 `days` 거래일 중 아직 없는 날짜만 받아 채운다.

    `krx_store.sync` 와 같은 모양이다 — 두 백필을 같은 방식으로 다루려는 것이다.

    ⚠️ **기본 워커가 1이다.** 종목 쪽(`krx_store.sync`)의 6과 다르다. 이유는
       `DEFAULT_WORKERS` 주석에 있다.
    """
    init_db()

    anchor = datetime.strptime(end, "%Y%m%d").date() if end else today_kst()
    wanted = [d.strftime("%Y%m%d") for d in trading_days(days, end=anchor)]
    # 제공 대상기간 밖은 아예 요청하지 않는다. 0행을 받아 휴장일로 기록해 두면
    # 나중에 KRX 가 과거를 열어도 우리가 다시 안 물어보게 된다.
    out_of_range = [d for d in wanted if d < DATA_START]
    wanted = [d for d in wanted if d >= DATA_START]

    result: Dict = {"requested": 0, "already": 0, "fetched": 0, "rows": 0,
                    "failed": [], "skipped_before_start": 0, "quota_exhausted": 0}

    # 왜 안 받았는지를 대장에 남긴다. 이게 없으면 나중에 "2009년이 왜 비어 있나"에
    # 아무도 답할 수 없고, 누군가 버그로 오해해 다시 받으려 든다.
    for bas_dd in out_of_range:
        for market in markets:
            collect_log.mark_out_of_range(
                COLLECT_SOURCE, _target(bas_dd, market),
                note=f"KRX 제공 시작일({DATA_START}) 이전이라 요청하지 않는다.",
                db_path=DB_PATH)
    result["skipped_before_start"] = len(out_of_range) * len(markets)

    todo: List[Tuple[str, str]] = []
    for market in markets:
        have = fetched_dates(market)
        # 최신 날짜부터 받는다 — 중간에 멈춰도 최근 구간이 먼저 채워진다
        days_todo = sorted((d for d in wanted if d not in have), reverse=True)
        result["requested"] += len(wanted)
        result["already"] += len(wanted) - len(days_todo)
        todo.extend((d, market) for d in days_todo)

    if not todo:
        return result

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def work(job: Tuple[str, str]) -> Tuple[str, str, int, Optional[str], bool]:
        bas_dd, market = job
        try:
            return bas_dd, market, fetch_date(bas_dd, market), None, False
        except api.KrxQuotaExhausted as error:
            # ⚠️ **실패가 아니다.** 예산이 마른 것은 이 날짜의 잘못이 아니므로 재시도
            #    횟수를 먹이면 안 된다 — 그러면 한도가 세 번 마르는 동안 멀쩡한 날짜가
            #    영영 버려진다. 내일 다시 돌리면 여기서부터 이어 받는다.
            collect_log.mark_quota_exhausted(COLLECT_SOURCE, _target(bas_dd, market),
                                             note=str(error), db_path=DB_PATH)
            return bas_dd, market, 0, str(error), True
        except Exception as error:      # 하루 실패가 전체를 멈추지 않게 한다
            collect_log.mark_error(COLLECT_SOURCE, _target(bas_dd, market),
                                   note=str(error), db_path=DB_PATH)
            return bas_dd, market, 0, str(error), False

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, job) for job in todo]
        for done, future in enumerate(as_completed(futures), start=1):
            bas_dd, market, rows, error, quota = future.result()
            if quota:
                result["quota_exhausted"] += 1
            elif error:
                result["failed"].append({"date": bas_dd, "market": market, "error": error})
            else:
                result["fetched"] += 1
                result["rows"] += rows
            if progress:
                progress(done, len(todo), f"{bas_dd} {market}", rows, error)

    return result


# ==================================================
# 2. 조회 (DB → 피처)
# ==================================================
def _rows_to_dicts(rows) -> List[Dict]:
    out = []
    for row in rows:
        item = dict(row)
        bas_dd = item.pop("bas_dd", "")
        item["date"] = to_iso(bas_dd) if bas_dd else None
        out.append(item)
    return out


def series(index_name: str = api.TARGET_INDEX, days: Optional[int] = None,
           start: Optional[str] = None, end: Optional[str] = None) -> List[Dict]:
    """지수 하나의 시계열을 **과거 → 현재 순**으로 돌려준다. 피처 생성이 쓰는 정문.

    ⚠️ 정렬 방향이 계약의 일부다. 최근순으로 돌려주면 이동평균·차분이 전부 뒤집힌 값을
       내는데 **예외가 나지 않는다.** 부르는 쪽에서 다시 정렬하지 말고 이 순서를 믿는다.

    ⚠️ 가격이 없는 행(`close is None`)은 **여기서 걸러 준다.** "코스피 (외국주포함)" 처럼
       거래량만 오는 지수가 있는데, 그 행이 시계열에 섞이면 수익률이 NaN 으로 오염된다.
    """
    init_db()
    sql = "SELECT * FROM index_price WHERE index_name = ? AND close IS NOT NULL"
    params: List = [index_name]
    if start:
        sql += " AND bas_dd >= ?"
        params.append(start)
    if end:
        sql += " AND bas_dd <= ?"
        params.append(end)
    sql += " ORDER BY bas_dd DESC"
    if days:
        sql += " LIMIT ?"
        params.append(days)

    with connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    # DESC 로 뽑아 LIMIT 을 최근 구간에 걸고, 돌려줄 때 과거→현재로 뒤집는다
    return list(reversed(_rows_to_dicts(rows)))


def available_indices(market: Optional[str] = None) -> List[Dict]:
    """어떤 지수가 얼마나 쌓여 있는지. 무엇을 피처로 쓸 수 있는지 고를 때 본다."""
    init_db()
    sql = ("SELECT index_name, index_class, COUNT(*) AS days, "
           "MIN(bas_dd) AS first_date, MAX(bas_dd) AS last_date "
           "FROM index_price WHERE close IS NOT NULL")
    params: List = []
    if market:
        sql += " AND index_class = ?"
        params.append(market)
    sql += " GROUP BY index_name, index_class ORDER BY days DESC, index_name"

    with connect() as conn:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def stats() -> Dict:
    """수집 현황 요약. `scripts/fetch_index.py --status` 가 쓴다."""
    init_db()
    with connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS rows, COUNT(DISTINCT bas_dd) AS days, "
            "COUNT(DISTINCT index_name) AS indices, "
            "MIN(bas_dd) AS first_date, MAX(bas_dd) AS last_date FROM index_price"
        ).fetchone()
        logged = conn.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(rows = 0), 0) AS closed FROM index_fetch_log"
        ).fetchone()

    return {
        "rows": row[0], "days": row[1], "indices": row[2],
        "first_date": to_iso(row[3]) if row[3] else None,
        "last_date": to_iso(row[4]) if row[4] else None,
        "logged_days": logged[0], "closed_days": logged[1],
        "db_path": str(DB_PATH),
    }
