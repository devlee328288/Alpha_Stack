"""KRX 일별 시세 디스크 캐시 (저장소 계층)

KRX OpenAPI 는 **하루치 전 종목 스냅샷**만 준다. 캔들 차트나 수익률 계산처럼
"한 종목의 여러 날"이 필요한 기능은 거래일 수만큼 호출해야 하는데, 1회에 2~3초가 걸린다.

그래서 한 번 받은 날짜는 SQLite 파일(`data/krx_cache.db`)에 쌓아 두고 다음부터는 DB 에서 읽는다.

| | 메모리 캐시 | SQLite 캐시 (이 모듈) |
|---|---|---|
| 서버 재시작 | 전부 사라짐 | 그대로 남음 |
| 250거래일 보관 | 1GB 이상 차지 | 필요한 행만 읽어 거의 0 |
| 매일 추가 비용 | 전체 재수집 | 새 거래일 1개(2회 호출) |

`sqlite3` 는 파이썬 표준 라이브러리라 별도 설치가 필요 없고, DB 파일 하나로 끝난다.
"""

from __future__ import annotations

import os  # 환경변수 · 쓰기 권한 확인
import sqlite3  # 파일 기반 DB (표준 라이브러리)
import tempfile  # 읽기 전용 환경에서 쓸 임시 폴더
import threading  # 쓰기 직렬화용 자물쇠
import time  # 라이브 조회 메모리 캐시 TTL
from contextlib import contextmanager  # 직접 만드는 with 블록
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from common import settings  # STORE_BACKEND — 어느 저장소에서 읽나 (ADR-DS-0015)
from common.trading_calendar import to_iso, today_kst, trading_days  # 거래일 계산 (공통 유틸)
from ingest.clients import krx_data as api  # KRX 호출·정규화 (외부 통신 담당)
from ingest.store import (
    krx_bundle,  # 배포용 축약본 (원본이 없을 때의 대타)
    krx_pg,  # Postgres 읽기 어댑터 (전환 S4 · ADR-DS-0015)
)

# 이 파일은 <루트>/ingest/store/ 안에 있으므로 parents[2] 가 프로젝트 루트다.
# (parents[0]=store, parents[1]=ingest, parents[2]=프로젝트 루트)
# 실행 위치(cwd)와 무관하게 항상 같은 DB 파일을 가리킨다.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_db_path() -> Path:
    """DB 파일 경로를 정한다. **쓸 수 있는 곳**이어야 한다.

    평소에는 `data/krx_cache.db` 지만, 서버리스(Vercel 등)에 올리면 배포된 파일이
    **읽기 전용**이라 그 자리에 DB 를 만들 수 없다. SQLite 는 파일을 열 때 없으면 만들려 하고,
    `PRAGMA journal_mode=WAL` 도 쓰기라서 곧바로 예외가 난다.

    그래서 쓰기가 막혀 있으면 임시 폴더(`/tmp`)로 옮긴다. 거기에 만들어진 DB 는 비어 있으므로
    `/krx`·`/quant` 화면은 "시세 캐시가 비어 있습니다" 안내(503)를 그대로 받는다.
    500 에러로 죽는 것보다, 무엇을 해야 하는지 알려 주는 편이 낫다.

    `KRX_DB_PATH` 환경변수로 직접 지정할 수도 있다 (배포 환경에서 경로를 바꾸고 싶을 때).
    """
    override = os.getenv("KRX_DB_PATH", "").strip()
    if override:
        path = Path(override)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    default = PROJECT_ROOT / "data" / "krx_cache.db"
    try:
        # 최초 실행 시 data/ 폴더가 없으면 sqlite3.connect 가 실패하므로 미리 만들어 둔다.
        default.parent.mkdir(parents=True, exist_ok=True)
        # 폴더가 있어도 쓰기 권한이 없을 수 있다. 실제로 쓸 수 있는지 확인한다.
        if os.access(default.parent, os.W_OK):
            return default
    except OSError:
        pass          # 폴더를 만들 수 없는 환경 (읽기 전용 배포)

    return Path(tempfile.gettempdir()) / "krx_cache.db"


DB_PATH = _resolve_db_path()

# 수집 대상 시장. KRX 는 시장마다 API 가 따로라 각각 호출해야 한다.
MARKETS = ("KOSPI", "KOSDAQ")

# DB 에 저장하는 컬럼 순서 (INSERT 와 SELECT 에서 함께 쓴다)
COLUMNS = ("bas_dd", "code", "name", "market", "sector",
           "open", "high", "low", "close", "change", "change_rate",
           "volume", "value", "market_cap", "listed_shares")

SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_price (
  bas_dd        TEXT    NOT NULL,   -- 기준일자 YYYYMMDD
  code          TEXT    NOT NULL,   -- 종목코드
  name          TEXT,               -- 종목명
  market        TEXT,               -- KOSPI / KOSDAQ
  sector        TEXT,               -- 소속부
  open          INTEGER,            -- 시가
  high          INTEGER,            -- 고가
  low           INTEGER,            -- 저가
  close         INTEGER,            -- 종가
  change        INTEGER,            -- 전일대비
  change_rate   REAL,               -- 등락률(%)
  volume        INTEGER,            -- 거래량
  value         INTEGER,            -- 거래대금
  market_cap    INTEGER,            -- 시가총액
  listed_shares INTEGER,            -- 상장주식수
  PRIMARY KEY (bas_dd, code)        -- 같은 날 같은 종목이 두 번 들어가지 않도록
);

-- 종목 하나의 시계열을 뽑을 때 쓰는 인덱스. 없으면 69만 행을 전부 훑는다.
CREATE INDEX IF NOT EXISTS idx_code_date ON daily_price(code, bas_dd);

-- 어떤 날짜를 이미 받아봤는지 기록한다.
-- 휴장일은 0건이 정상이라, 이 표가 없으면 매번 다시 요청하게 된다.
CREATE TABLE IF NOT EXISTS fetch_log (
  bas_dd     TEXT PRIMARY KEY,
  rows       INTEGER,
  fetched_at TEXT
);
"""


# ==================================================
# 1. 연결 · 초기화
# ==================================================
@contextmanager
def connect():
    """DB 연결을 열고, 블록이 끝나면 **커밋하고 반드시 닫는다.**

    주의: `with sqlite3.connect(...) as conn:` 은 커밋만 하고 **연결을 닫지 않는다.**
    수집처럼 여러 스레드가 동시에 쓰는 상황에서 연결이 남아 있으면
    잠금이 풀리지 않아 `database is locked` 가 발생한다. 그래서 직접 컨텍스트 매니저를 만든다.

    sqlite3 연결은 스레드 간에 공유하면 안 되고 FastAPI 는 요청을 여러 스레드에서 처리하므로,
    매번 새로 열고 닫는다 (파일 DB 라 연결 비용이 매우 작다).
    """
    conn = sqlite3.connect(DB_PATH, timeout=60)
    conn.row_factory = sqlite3.Row        # 결과를 컬럼명으로 꺼낼 수 있게 한다
    # WAL 모드: 읽기와 쓰기가 서로를 막지 않는다. 수집 중에도 화면 조회가 가능해진다.
    conn.execute("PRAGMA journal_mode=WAL")
    # 다른 연결이 쓰는 중이면 최대 60초까지 기다린다 (바로 포기하지 않도록)
    conn.execute("PRAGMA busy_timeout=60000")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()                      # 예외가 나도 반드시 닫는다


# SQLite 는 동시에 한 연결만 쓸 수 있다. 쓰기를 줄 세워 잠금 충돌 자체를 없앤다.
# (읽기는 WAL 덕분에 잠금과 무관하므로 이 자물쇠를 거치지 않는다.)
_write_lock = threading.Lock()


def init_db() -> None:
    """표와 인덱스를 만든다. 이미 있으면 아무 일도 하지 않는다."""
    with _write_lock, connect() as conn:
        conn.executescript(SCHEMA)


# ==================================================
# 2. 수집 (KRX → DB)
# ==================================================
# 0건으로 받은 날짜를 "확정 휴장일"로 볼 때까지 기다리는 기간(일).
# 당일 데이터는 장 마감 후에야 올라오므로, 최근 며칠간의 0건은 다시 확인해야 한다.
ZERO_ROW_RETRY_DAYS = 7


def fetched_dates() -> set:
    """다시 받을 필요가 없는 날짜 집합.

    - 데이터가 있는 날짜(rows > 0)는 언제나 건너뛴다.
    - 0건인 날짜는 **오래된 것만** 건너뛴다. 휴장일은 영원히 0건이지만,
      오늘·어제의 0건은 아직 장이 안 끝났거나 공시 전일 수 있어 다시 확인해야 한다.
    """
    init_db()
    cutoff = (today_kst() - timedelta(days=ZERO_ROW_RETRY_DAYS)).strftime("%Y%m%d")
    with connect() as conn:
        rows = conn.execute(
            "SELECT bas_dd, rows FROM fetch_log WHERE rows > 0 OR bas_dd < ?", (cutoff,)
        ).fetchall()
    return {row[0] for row in rows}


def _save(bas_dd: str, items: List[Dict]) -> int:
    """정규화된 한 날짜치를 DB 에 저장하고 저장 건수를 돌려준다."""
    rows = [
        tuple([bas_dd] + [item.get(col) for col in COLUMNS[1:]])
        for item in items
    ]
    placeholders = ",".join("?" * len(COLUMNS))

    # 쓰기는 한 번에 하나씩 — 6개 스레드가 동시에 INSERT 하면 잠금 충돌이 난다
    with _write_lock, connect() as conn:
        # INSERT OR REPLACE — 같은 날짜를 다시 받아도 중복 없이 덮어쓴다
        conn.executemany(
            f"INSERT OR REPLACE INTO daily_price ({','.join(COLUMNS)}) VALUES ({placeholders})",
            rows,
        )
        conn.execute(
            "INSERT OR REPLACE INTO fetch_log (bas_dd, rows, fetched_at) VALUES (?,?,?)",
            (bas_dd, len(rows), datetime.now().isoformat(timespec="seconds")),
        )
    return len(rows)


def fetch_date(bas_dd: str) -> int:
    """한 거래일을 시장별로 받아 DB 에 저장한다. 이미 받은 날짜면 0 을 돌려준다."""
    items: List[Dict] = []
    for market in MARKETS:
        items.extend(api.fetch_snapshot(bas_dd, market))
    return _save(bas_dd, items)


def sync(days: int = 250, workers: int = 6, end: Optional[str] = None,
         progress=None) -> Dict:
    """최근 `days` 거래일 중 아직 없는 날짜만 KRX 에서 받아 DB 에 채운다.

    KRX 1회 호출이 2~3초라 순서대로 받으면 250일 × 2시장 = 500회에 20분이 넘는다.
    스레드로 동시에 여러 날짜를 받아 시간을 1/`workers` 로 줄인다.
    (동시 호출 수를 너무 올리면 상대 서버에 부담이 되므로 기본 6개로 둔다.)
    """
    init_db()

    anchor = datetime.strptime(end, "%Y%m%d").date() if end else today_kst()
    wanted = [d.strftime("%Y%m%d") for d in trading_days(days, end=anchor)]
    have = fetched_dates()
    # 최신 날짜부터 받는다 — 수집이 끝나기 전에도 화면이 최근 구간을 먼저 보여줄 수 있다
    todo = sorted((d for d in wanted if d not in have), reverse=True)

    result = {"requested": len(wanted), "already": len(wanted) - len(todo),
              "fetched": 0, "rows": 0, "failed": []}
    if not todo:
        return result

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def work(bas_dd: str) -> Tuple[str, int, Optional[str]]:
        try:
            return bas_dd, fetch_date(bas_dd), None
        except Exception as error:            # 하루 실패가 전체 수집을 멈추지 않게 한다
            return bas_dd, 0, str(error)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, d) for d in todo]
        for done, future in enumerate(as_completed(futures), start=1):
            bas_dd, rows, error = future.result()
            if error:
                result["failed"].append({"date": bas_dd, "error": error})
            else:
                result["fetched"] += 1
                result["rows"] += rows
            if progress:
                progress(done, len(todo), bas_dd, rows, error)

    return result


# ==================================================
# 3. 조회 (DB → 서비스)
# ==================================================
# 원본 캐시가 비어 있으면 **배포용 축약본**(`krx_bundle`)에게 넘긴다.
#
# 왜 필요한가 — 배포 환경에는 123MB 원본을 올릴 수 없어 캐시가 늘 비어 있다.
# 예전에는 그 상태에서 `/quant` 4종 · `/krx` 캔들 · `/market` 시장의 폭이 전부 503·빈 화면이었다.
# 축약본(전종목 150거래일)을 읽기 전용으로 함께 실으면 같은 화면이 그대로 살아난다.
#
# 아래 조회 함수들은 전부 같은 규칙을 따른다.
#   1) 원본 캐시에 있으면 그것을 쓴다 (로컬 — 가장 정확하고 구간도 길다)
#   2) 없으면 축약본에게 묻는다 (배포본)
#   3) 축약본도 없으면 빈 결과 — 부르는 쪽이 라이브 조회나 안내로 넘어간다
#
# ⭐ **정본 저장소는 스위치가 정한다** (전환 S4 · ADR-DS-0015).
#
# 아래 아홉 함수가 `STORE_BACKEND` 를 보고 갈린다 — `_cache_is_empty` · `latest_date` ·
# `available_dates` · `snapshot_tiered` · `series_tiered` · `window` · `stats` ·
# **`lookup_security`**, 그리고 `tier()` 는 `_cache_is_empty()` 를 통해 따라온다.
# **아홉은 한 벌이다.** 하나만 남겨 두면 로컬 SQLite 를 지운 개발자 셸에서 Postgres 는
# 꽉 차 있는데 `tier()` 만 `bundle` 을 내는 어긋난 상태가 된다.
#
# ⭐ `lookup_security` 는 **S5 에 늘었다** (ADR-DS-0018). S4 때는 이 자리에 없었고
#   `stock_service` 가 읽기 표면을 우회해 SQLite 를 직접 열고 있었다 — 스위치가 안 닿는
#   경로가 하나 남아 있었다는 뜻이다. 기본값을 뒤집기 전에 그것부터 여기로 들였다.
#
# `snapshot()`·`series()`·`universe()`·`closes_matrix()`·`source_tag()` 는 **분기하지 않는다.**
# 전부 모듈 전역 이름으로 위 함수들을 부르므로 자동으로 따라온다. 거기까지 분기를 넣으면
# 이중 분기가 된다.
#
# ⚠️ 분기는 `init_db()` **앞**에 둔다. Postgres 에서 DDL 은 asyncpg 의 타입 캐시를
#    무효화하고, `init_db()` 는 애초에 SQLite 표를 만드는 함수라 그쪽에서는 뜻이 없다.
def _postgres() -> bool:
    """읽기를 Postgres 에서 하는가. 호출 시점에 다시 읽는다(설정이 상수가 아닌 이유)."""
    return settings.uses_postgres_store()


def _cache_is_empty() -> bool:
    """원본 캐시에 데이터가 한 줄이라도 있는지. (기본키 인덱스만 타므로 값싸다)"""
    if _postgres():
        return krx_pg.is_empty()
    init_db()
    with connect() as conn:
        row = conn.execute("SELECT MAX(bas_dd) FROM daily_price").fetchone()
    return not (row and row[0])


# 이 저장소가 대는 자료의 제공자. `source` 필드는 `<provider>-<tier>` 두 토막이다 (ADR-DS-0009).
PROVIDER = "krx"


def tier() -> str:
    """**저장소 전체**가 지금 어느 층에 서 있는지 — `db`(정본) · `bundle`(축약본) · `live`(둘 다 없음).

    화면 배지와 리포트가 "무엇을 근거로 말하고 있는지" 를 밝힐 때 쓴다.

    ⚠️ **조회 하나의 출처로 쓰면 틀린다.** 원본이 차 있어도 그 날짜·그 종목만 없으면
    `snapshot()`·`series()` 는 번들로 내려간다. 조회별 층은 `*_tiered()` 짝에게 물어야 한다.
    """
    if not _cache_is_empty():
        return "db"
    return "bundle" if krx_bundle.available() else "live"


def source_tag() -> str:
    """저장소 전체 상태를 `source` 필드 값(`krx-db` 등)으로 만들어 돌려준다 (ADR-DS-0009)."""
    return f"{PROVIDER}-{tier()}"


def _rows_to_dicts(rows: Iterable[sqlite3.Row]) -> List[Dict]:
    """sqlite3.Row 를 평범한 딕셔너리로 바꾸고 날짜 표기를 화면용으로 맞춘다."""
    out = []
    for row in rows:
        item = dict(row)
        bas_dd = item.pop("bas_dd", "")
        item["date"] = to_iso(bas_dd) if bas_dd else None
        out.append(item)
    return out


def latest_date() -> Optional[str]:
    """데이터가 있는 가장 최근 거래일 (YYYYMMDD). 원본·축약본 어디에도 없으면 None."""
    if _postgres():
        found = krx_pg.latest_date()
        return found or krx_bundle.latest_date()
    init_db()
    with connect() as conn:
        row = conn.execute("SELECT MAX(bas_dd) FROM daily_price").fetchone()
    if row and row[0]:
        return row[0]
    return krx_bundle.latest_date()


def available_dates(limit: int = 400) -> List[str]:
    """데이터가 있는 거래일 목록 (최근순). 화면의 날짜 선택 범위로 쓴다.

    원본이 비면 **파생 캘린더**(`krx_derived.json`)를 먼저 본다. 축약본 DB 는 150거래일뿐이지만
    파생 캘린더는 캐시 전 구간(282거래일)을 담고 있어, 전처리에 넘길 거래일 축이 더 길다.
    """
    if _postgres():
        found = krx_pg.available_dates(limit=limit)
        if found:
            return found
    else:
        init_db()
        with connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT bas_dd FROM daily_price ORDER BY bas_dd DESC LIMIT ?", (limit,)
            ).fetchall()
        if rows:
            return [r[0] for r in rows]

    # 파생 캘린더는 `YYYY-MM-DD` 오름차순이라 이 함수의 계약(`YYYYMMDD` 최근순)에 맞춰 돌려준다
    calendar = krx_bundle.trading_days(limit=limit)
    if calendar:
        return [d.replace("-", "") for d in reversed(calendar)]
    return krx_bundle.available_dates(limit=limit)


def snapshot_tiered(bas_dd: str, market: Optional[str] = None) -> Tuple[List[Dict], str]:
    """`snapshot()` 과 같되 **어느 층에서 나왔는지**를 함께 돌려준다 — `(행 목록, tier)`.

    출처를 응답에 싣는 호출자는 반드시 이쪽을 쓴다 (ADR-DS-0009 §5).
    `tier()` 를 따로 부르면 안 된다 — 그건 저장소 전체 상태라, 원본이 차 있는데
    **그 날짜만** 없어 번들로 내려간 경우를 `db` 라고 잘못 말한다.

    ⚠️ **사다리의 모양은 두 저장소에서 같다** (ADR-DS-0015 §2). 바뀌는 것은 맨 윗단의
    구현뿐이다 — 그래야 `STORE_BACKEND` 를 되돌리는 것만으로 화면이 원래대로 돌아온다.
    """
    if _postgres():
        rows = krx_pg.snapshot(bas_dd, market)
    else:
        init_db()
        sql = "SELECT * FROM daily_price WHERE bas_dd = ?"
        params: List = [bas_dd]
        if market:
            sql += " AND market = ?"
            params.append(market)
        with connect() as conn:
            rows = _rows_to_dicts(conn.execute(sql, params).fetchall())
    if rows:
        return rows, "db"
    fallback = _rows_to_dicts(krx_bundle.snapshot(bas_dd, market))
    # 축약본에도 없으면 "번들에서 왔다" 고 말할 근거가 없다. 저장소가 서 있는 층을 그대로 밝힌다.
    return fallback, "bundle" if fallback else tier()


def snapshot(bas_dd: str, market: Optional[str] = None) -> List[Dict]:
    """해당 거래일의 전 종목. `market` 을 주면 그 시장만 추린다."""
    return snapshot_tiered(bas_dd, market)[0]


# 라이브 조회 결과를 담아 두는 메모리 캐시. {(거래일, 시장): (저장시각, 행 목록)}
# 배포 환경(서버리스)은 DB 를 유지할 수 없어서, 같은 인스턴스가 살아 있는 동안만이라도
# KRX 를 다시 부르지 않도록 한다. KRX 호출은 1회에 2~3초가 걸린다.
_live_cache: Dict[Tuple, Tuple[float, List[Dict]]] = {}
_live_lock = threading.Lock()
LIVE_CACHE_TTL = 600            # 일별 데이터라 10분이면 충분하다


def snapshot_live(bas_dd: str = "", market: Optional[str] = None) -> Tuple[List[Dict], str, str]:
    """**DB 없이** KRX 를 직접 불러 전 종목 스냅샷을 돌려준다.

    `(행 목록, 실제 거래일, tier)` 를 돌려준다. tier 는 `live`(KRX 를 방금 호출) 또는
    `live-memo`(그 응답을 프로세스 메모리에서 재사용) 다 (ADR-DS-0009).

    왜 필요한가
    -----------
    배포 환경(Vercel 등 서버리스)에는 96MB 짜리 `krx_cache.db` 를 올릴 수 없고,
    파일을 써 봐야 인스턴스가 바뀌면 사라진다. 그렇다고 `/krx` 화면을 통째로 막아 두면
    "인증키는 멀쩡한데 화면은 죽어 있는" 이상한 상태가 된다.

    다행히 KRX 일별매매정보는 **하루치 전 종목**을 한 번에 주므로, DB 없이 그 자리에서
    받아 쓰면 된다. 집계·정렬·페이지는 어차피 메모리에서 하던 일이라 그대로 동작한다.

    `bas_dd` 를 비우면 **최근 거래일부터 거꾸로** 훑는다. KRX 는 휴장일에 빈 배열을 주므로,
    데이터가 나올 때까지 최대 7거래일을 시도한다 (연휴 대비).
    """
    markets = (market,) if market else MARKETS

    # 날짜를 지정했으면 그 날짜만, 아니면 최근 거래일부터 거슬러 올라가며 찾는다
    candidates = [bas_dd] if bas_dd else [d.strftime("%Y%m%d") for d in
                                          reversed(trading_days(7))]

    for day in candidates:
        key = (day, market or "ALL")
        now = time.monotonic()

        with _live_lock:
            hit = _live_cache.get(key)
            if hit and now - hit[0] < LIVE_CACHE_TTL:
                return hit[1], day, "live-memo"

        rows: List[Dict] = []
        for mkt in markets:
            # 인증 실패는 재시도해도 소용없으므로 그대로 올려보낸다 (차단기가 이미 걸린다)
            rows.extend(api.fetch_snapshot(day, mkt))

        if rows:
            with _live_lock:
                _live_cache[key] = (time.monotonic(), rows)
            return rows, day, "live"
        # 빈 배열이면 휴장일이다. 다음 후보 날짜로 넘어간다.

    return [], (bas_dd or ""), "live"


def clear_live_cache() -> int:
    """라이브 조회 메모리 캐시를 비운다. 버린 항목 수를 돌려준다.

    수집을 막 끝낸 직후에 부른다 (ADR-DS-0017). 그전에 원본이 비어 있어 라이브로 답한
    응답이 10분(`LIVE_CACHE_TTL`) 동안 남아 있으면, **캐시는 방금 찼는데 화면은 계속
    `krx-live-memo`** 라고 말한다. 값 자체는 같은 KRX 자료라 틀리지 않지만, 출처 배지가
    저장소 상태를 잘못 전한다 (ADR-DS-0009 — 배지는 "무엇을 근거로 말하는가" 다).
    """
    with _live_lock:
        dropped = len(_live_cache)
        _live_cache.clear()
    return dropped


def series_tiered(code: str, days: int = 250,
                  end: Optional[str] = None) -> Tuple[List[Dict], str]:
    """`series()` 와 같되 **어느 층에서 나왔는지**를 함께 돌려준다 — `(행 목록, tier)`.

    출처를 응답에 싣는 호출자는 반드시 이쪽을 쓴다 (ADR-DS-0009 §5).
    종목 단위로 갈리므로, 저장소 전체 상태인 `tier()` 로는 알 수 없다 —
    원본이 차 있어도 **그 종목만** 없으면 번들로 내려간다.
    """
    if _postgres():
        # 어댑터가 이미 오름차순으로 준다 (뒤집기까지 그쪽에서 끝낸다).
        ordered = krx_pg.series(code, days=days, end=end)
        if ordered:
            return ordered, "db"
        return _series_fallback(code, days=days, end=end)

    init_db()
    sql = "SELECT * FROM daily_price WHERE code = ?"
    params: List = [code]
    if end:
        sql += " AND bas_dd <= ?"
        params.append(end)
    # 최근 것부터 days 개를 가져온 뒤 파이썬에서 뒤집는다 (차트는 왼쪽이 과거)
    sql += " ORDER BY bas_dd DESC LIMIT ?"
    params.append(days)

    with connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    if rows:
        return list(reversed(_rows_to_dicts(rows))), "db"
    return _series_fallback(code, days=days, end=end)


def _series_fallback(code: str, days: int, end: Optional[str]) -> Tuple[List[Dict], str]:
    """정본 저장소에 그 종목이 없을 때의 아랫단. **두 저장소가 이것을 함께 쓴다.**

    사다리를 한 벌만 두는 것이 뜻을 가진다 — 백엔드마다 폴백이 갈리면
    `STORE_BACKEND` 를 되돌리는 일이 "환경변수 한 줄"이 아니라 "두 동작 중 고르기"가 된다.
    """
    # 축약본도 내림차순으로 주므로 같은 방식으로 뒤집는다 (차트는 왼쪽이 과거)
    fallback = list(reversed(_rows_to_dicts(krx_bundle.series(code, days=days, end=end))))
    # 축약본에도 없으면 "번들에서 왔다" 고 말할 근거가 없다. 저장소가 서 있는 층을 그대로 밝힌다.
    return fallback, "bundle" if fallback else tier()


def series(code: str, days: int = 250, end: Optional[str] = None) -> List[Dict]:
    """종목 하나의 일봉 시계열 (날짜 오름차순).

    인덱스(idx_code_date) 덕분에 69만 행 중 해당 종목만 곧바로 찾아낸다.
    """
    return series_tiered(code, days=days, end=end)[0]


def lookup_security(code_or_name: str) -> Optional[Dict]:
    """종목코드 또는 한글 종목명으로 종목 하나를 찾는다 — `{code, name, market}`. 없으면 None.

    **아홉 번째 이음매다** (전환 S5 · ADR-DS-0018). 앞선 여덟과 달리 이것은 S4 에 없었다 —
    `stock_service._lookup_krx()` 가 읽기 표면을 **우회해** `connect()` 로 SQLite 에 생 SQL
    세 개를 던지고 있었기 때문이다. 그대로 두면 `STORE_BACKEND=postgres` 를 켜도
    **한글 종목명 검색만** SQLite 를 계속 봐서, 두 저장소가 갈린 날 그 화면만 조용히 낡는다.

    ## 찾는 순서

    ① 6자리 숫자면 코드로 · ② 아니면 이름이 정확히 같은 것 · ③ 그것도 없으면 앞부분이 같은 것
    ("에코프로비" → "에코프로비엠"). ②③ 은 `거래일 DESC, 거래대금 DESC` 로 하나를 고른다 —
    "삼성" 처럼 여러 개가 걸릴 때 가장 대표적인 종목이 나오게 하려는 것이다.

    ⚠️ **예외를 삼키지 않는다.** 못 찾은 것(`None`)과 못 읽은 것(예외)은 다르다.
    폴백할지 말지는 부르는 쪽이 정한다 — `stock_service` 는 종목 마스터(JSON)로 내려간다.
    여기서 삼키면 DB 장애가 "그런 종목 없음" 으로 위장돼 야후 단독 경로로 조용히 강등된다.

    ⚠️ **축약본으로 내려가지 않는다.** 다른 이음매와 다른 점이다. 부르는 쪽의 아랫단이
    `stock_master.json`(전 종목 · 커밋된다)이고 축약본 DB(150거래일 · 커밋 안 된다)보다
    이 용도에는 넓다. 사다리를 여기에 또 만들면 두 벌이 된다.
    """
    needle = code_or_name.strip()
    if not needle:
        return None

    if _postgres():
        return krx_pg.lookup_security(needle)

    init_db()
    with connect() as conn:
        if krx_pg.CODE_PATTERN.fullmatch(needle):
            row = conn.execute(
                "SELECT code, name, market FROM daily_price WHERE code = ? "
                "ORDER BY bas_dd DESC LIMIT 1", (needle,)).fetchone()
        else:
            # 이름 검색 — 정확히 일치하는 것을 먼저 찾고, 없으면 앞부분이 같은 종목을 쓴다.
            row = conn.execute(
                "SELECT code, name, market FROM daily_price WHERE name = ? "
                "ORDER BY bas_dd DESC, value DESC LIMIT 1", (needle,)).fetchone()
            if row is None:
                row = conn.execute(
                    "SELECT code, name, market FROM daily_price WHERE name LIKE ? "
                    "ORDER BY bas_dd DESC, value DESC LIMIT 1", (f"{needle}%",)).fetchone()
    return dict(row) if row else None


def universe(bas_dd: Optional[str] = None, market: Optional[str] = None) -> List[Dict]:
    """가장 최근(또는 지정한) 거래일 기준 종목 목록. 거래대금 순으로 정렬해 돌려준다."""
    bas_dd = bas_dd or latest_date()
    if not bas_dd:
        return []
    rows = snapshot(bas_dd, market)
    return sorted(rows, key=lambda r: r.get("value") or 0, reverse=True)


def closes_matrix(codes: Sequence[str], days: int = 250) -> Dict[str, List[int]]:
    """여러 종목의 종가를 **공통 거래일**로 맞춰 돌려준다.

    상관계수·공분산처럼 **종목끼리 날짜를 맞춰야 하는** 계산에 쓴다.
    상장폐지·거래정지로 일부 날짜가 빠진 종목이 있으므로, 모든 종목에 존재하는 날짜만 남긴다.

    ⚠️ **종목별 수익률에는 쓰지 말 것.** 교집합이라 종목을 많이 넣을수록 축이 급격히 짧아진다.
    실측(2026-08) — 400종목이면 22일, 2,763종목 전부면 **5일**로 줄어든다.
    상장한 지 얼마 안 된 종목 하나가 전체 축을 자르기 때문이다.
    종목마다 독립적으로 계산할 때는 `window()` 로 한 번에 읽어 코드별로 나눠 쓴다
    (`scripts/build_market_snapshot.py` 의 `build_domestic` 이 그 예다).
    """
    per_code: Dict[str, Dict[str, int]] = {}
    for code in codes:
        per_code[code] = {
            row["date"]: row["close"] for row in series(code, days) if row.get("close")
        }

    if not per_code:
        return {}

    # 모든 종목이 공통으로 가진 날짜만 교집합으로 추린다
    common = set.intersection(*(set(d.keys()) for d in per_code.values())) if per_code else set()
    ordered = sorted(common)
    return {code: [per_code[code][d] for d in ordered] for code in codes}


def window(days: int = 60, columns: Sequence[str] = ("code", "bas_dd", "close", "value", "volume")) -> List[Dict]:
    """최근 `days` 거래일치를 필요한 컬럼만 골라 한 번에 읽어 온다.

    스크리닝·팩터·효율적 투자선은 모두 "전 종목 × 최근 N일" 을 훑어야 한다.
    종목마다 따로 조회하면 2,700번 질의하게 되므로, 한 방에 읽어 파이썬에서 묶는다.

    ⚠️ 정렬 기준이 성능을 좌우한다.
    `ORDER BY code, bas_dd` 로 쓰면 SQLite 가 `idx_code_date` 를 타면서 64만 건을 전부 훑고
    행마다 임의 접근을 하게 되어 **18초** 가 걸렸다. 기본키가 `(bas_dd, code)` 라
    `ORDER BY bas_dd` 는 이미 정렬된 순서라서 추가 비용이 없다 — **0.8초**.
    종목별 묶음은 파이썬에서 하고, 날짜 오름차순으로 읽으므로 각 묶음도 자동으로 날짜순이 된다.

    ⚠️ **Postgres 에서는 위 실측이 뒤집힌다** — 그쪽 PK 는 `(security_id, trade_date)` 라
    `ORDER BY trade_date` 가 공짜가 아니다. 어댑터가 하한을 서브질의로 걸어 그 문제를 푼다
    (`krx_pg.window()` docstring 참조).
    """
    if _postgres():
        rows = krx_pg.window(days=days, columns=columns)
        return rows if rows else krx_bundle.window(days=days, columns=columns)

    init_db()
    if _cache_is_empty():
        return krx_bundle.window(days=days, columns=columns)

    dates = available_dates(limit=days)
    if not dates:
        return []
    floor = min(dates)                       # 가장 오래된 대상 거래일

    cols = ",".join(columns)
    with connect() as conn:
        rows = conn.execute(
            f"SELECT {cols} FROM daily_price WHERE bas_dd >= ? ORDER BY bas_dd", (floor,)
        ).fetchall()
    return [dict(r) for r in rows]


def stats() -> Dict:
    """캐시 현황 — 화면 배지와 README 확인용.

    원본이 비어 있으면 **축약본의 현황**을 대신 돌려주고 `mode` 로 어느 쪽인지 밝힌다.
    화면이 "0거래일" 만 보고 고장으로 오해하지 않게 하기 위함이다.

    ⚠️ 그래서 `days`·`db_path`·`db_size_mb` 는 **`mode` 가 가리키는 층의 것**이다.
    `days > 0` 만 보고 "원본 캐시가 있다" 고 재계산하면 번들 숫자에 캐시 라벨이 붙는다
    (ADR-DS-0009 §4 — `mode` 는 이 함수가 정본이고 소비자는 그대로 쓴다).
    """
    if _postgres():
        found = krx_pg.stats()
        # 비어 있으면 축약본 가지로 내려간다 — SQLite 쪽과 같은 사다리다.
        return found if found["days"] else _stats_bundle()

    init_db()
    return _stats_sqlite()


def _stats_sqlite() -> Dict:
    """SQLite 원본의 현황. 비어 있으면 축약본 가지로 넘긴다."""
    with connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS rows, COUNT(DISTINCT bas_dd) AS days,"
            " COUNT(DISTINCT code) AS codes, MIN(bas_dd) AS first, MAX(bas_dd) AS last"
            " FROM daily_price"
        ).fetchone()

    if row["days"]:
        size = DB_PATH.stat().st_size if DB_PATH.exists() else 0
        return {
            "rows": row["rows"], "days": row["days"], "codes": row["codes"],
            "first_date": row["first"], "last_date": row["last"],
            "db_path": DB_PATH.name,
            "db_size_mb": round(size / 1024 / 1024, 1),
            "mode": "db",
            "notes": [],
        }
    return _stats_bundle()


def _stats_bundle() -> Dict:
    """축약본(또는 아무것도 없음)의 현황. **두 저장소가 함께 쓰는 아랫단이다.**"""
    bundle = krx_bundle.stats()
    calendar = krx_bundle.derived_stats()
    return {
        "rows": bundle["rows"], "days": bundle["days"], "codes": bundle["codes"],
        "first_date": bundle["first_date"], "last_date": bundle["last_date"],
        "db_path": bundle["path"] if bundle["available"] else DB_PATH.name,
        "db_size_mb": bundle["size_mb"],
        "mode": "bundle" if bundle["available"] else "live",
        "generated_at": bundle["generated_at"],
        "calendar_days": calendar["days"],
        "notes": bundle["notes"] if bundle["available"] else [
            "원본 캐시도 배포용 축약본도 없습니다. "
            "`python3 scripts/build_krx_bundle.py` 로 축약본을 만들 수 있습니다.",
        ],
    }
