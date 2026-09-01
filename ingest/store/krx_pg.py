"""KRX 일별 시세 — Postgres 읽기 어댑터 (저장계층 전환 S4 · ADR-DS-0015).

`krx_store` 가 SQLite `daily_price` 한 표에서 읽던 것을 Postgres `ohlcv` + `securities`
두 표에서 읽는다. **`krx_store` 의 읽기 함수와 같은 모양을 돌려주는 것이 이 모듈의 계약이다** —
키 이름·값 타입·정렬 순서까지 같아야 한다. 부르는 쪽 20여 곳은 어느 저장소인지 몰라야 한다.

## 세 가지를 경계에서 되돌린다

저장소가 바뀌면서 값의 **타입**이 셋 달라진다. 그것을 이 모듈이 전부 여기서 되돌린다.
안 되돌리면 대부분 예외 없이 조용히 틀린다 — 그게 이 절이 있는 이유다.

| 무엇 | Postgres 가 주는 것 | 되돌리는 값 | 안 되돌리면 |
|---|---|---|---|
| `change_rate` | `Decimal('26.8100')` | `float` | `tmp_cache.write()` 의 `json.dumps` 가 `TypeError` 를 내는데 그 함수가 **예외를 삼킨다**(tmp_cache.py:102). 캐시가 영원히 안 써지고 로그도 안 남는다 |
| `trade_date` | `datetime.date` | `date` 키는 `YYYY-MM-DD`, `bas_dd` 는 `YYYYMMDD` | `to_iso()` 가 문자열 슬라이싱이라 `TypeError`. pydantic `str` 필드는 `date` 를 coerce 하지 않아 **500** |
| `listed_shares` | 두 표에 다 있다 | **`ohlcv` 쪽** | `securities` 쪽은 최신값이라 액면분할 종목의 과거 회전율이 10배 틀린다 (ADR-DS-0010) |

⚠️ **`date` 와 `bas_dd` 는 서식이 다르다.** `snapshot()`·`series()` 는 `date`(`YYYY-MM-DD`),
`window()`·`latest_date()`·`available_dates()`·`stats()` 는 `YYYYMMDD` 다. `krx_store` 가
그렇게 하고 있고(`_rows_to_dicts` 가 `bas_dd` 를 pop 하고 `date` 를 붙인다), 화면·라우터가
그 차이에 기대고 있다.

## 조회는 두 걸음으로 나눈다 — JOIN 한 방이 127배 느리다

종목 하나의 시계열을 `JOIN securities ON code = ?` 로 한 번에 뽑으면 플래너가 파티션
14개를 전부 훑는다. `security_id` 를 먼저 풀고 그 값으로 `ohlcv` 를 보면 파티션마다
PK 를 곧장 탄다. 실측 (2026-08-25 · 005930 · 250행):

    JOIN 한 방        73.5 ms
    security_id 먼저   0.58 ms      ← 127배

## ⚠️ `NOT is_delisted` 를 반드시 건다

`securities_code_active_uq` 는 `WHERE NOT is_delisted` **부분** 유니크다. 상장폐지 뒤
코드 재사용을 허용하려고 그렇게 만들었다(01-schema.sql:57-58). 오늘은 `is_delisted` 가
전부 false 라 code 가 1:1 이지만, S8 이 폐지 종목을 채우기 시작하면 조건을 뺀 JOIN 은
**행을 두 배로 낸다.** 오늘 안 터진다고 빼 두면 그때 조용히 틀린다.

## DDL 을 발행하지 않는다

`krx_store` 는 조회마다 `init_db()` 로 `CREATE TABLE IF NOT EXISTS` 를 낸다. SQLite 에서는
무해했지만 Postgres 에서 DDL 은 asyncpg 의 타입 OID 캐시를 무효화한다. 스키마는
`sql/init/*.sql` 이 빈 볼륨에서 한 번 세운다. 이 모듈에는 DDL 이 한 줄도 없고
`tests/test_krx_pg.py` 가 그것을 검사한다.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import text

from common import codes, db

# `krx_store.snapshot()`/`series()` 가 돌려주는 키와 **순서**. `SELECT *` 를 쓰던 자리라
# 컬럼을 명시하지 않으면 키 집합이 조용히 달라진다.
ROW_COLUMNS = ("code", "name", "market", "sector",
               "open", "high", "low", "close", "change", "change_rate",
               "volume", "value", "market_cap", "listed_shares")

# `window(columns=...)` 이 받는 SQLite 컬럼명 → Postgres 표현식.
# ⚠️ **화이트리스트다.** `krx_store.window()` 는 인자를 f-string 으로 SQL 에 그대로 넣는데,
#    그 이름들은 SQLite 스키마 것이라 Postgres 에서는 `UndefinedColumn` 이 된다.
#    `krx_bundle.window()` 도 같은 이유로 화이트리스트를 두고 있다(krx_bundle.py:154-158).
WINDOW_COLUMNS: Dict[str, str] = {
    "bas_dd": "to_char(o.trade_date, 'YYYYMMDD')",
    "code": "s.code",
    "open": "o.open",
    "high": "o.high",
    "low": "o.low",
    "close": "o.close",
    "change": "o.change",
    "change_rate": "o.change_rate",
    "volume": "o.volume",
    "value": "o.value",
    "market_cap": "o.market_cap",
    # ⚠️ 그 거래일 값이다. `securities.listed_shares`(최신)가 아니다 — ADR-DS-0010.
    "listed_shares": "o.listed_shares",
}

# 코드 검색과 이름 검색은 질의가 아예 달라 먼저 가른다. 여기서 하는 일은 검증이 아니라
# **갈래 고르기**이므로 넓게 잡는다 — 한글 종목명과만 구분되면 되고, 못 찾으면 `None` 이다.
# ⚠️ 예전에는 `^\d{6}$` 였다. 그래서 `0009K0`(에임드바이오)·`0126Z0`(삼성에피스홀딩스)처럼
#    5·6번째 자리에 영문이 있는 84종이 코드 분기를 못 타고 이름 검색으로 새어, 예외도 로그도
#    없이 *"그런 종목 없음"* 으로 위장했다. 그 둘은 350종 핵심 유니버스 안에 있다.
# ⚠️ **정의는 `common/codes.py` 하나다.** `krx_store.lookup_security()` 도 이 값을 그대로 쓴다.
#    두 벌로 두면 한쪽만 고쳐진 채 오래 간다 — 이 레포가 갱신 사슬에서 이미 겪은 고장이다.
CODE_PATTERN = codes.CODE_LIKE_PATTERN

# 종목 속성은 `securities` 에, 시세는 `ohlcv` 에 있다. 둘을 잇는 조각을 한 곳에 둔다.
_JOIN = "ohlcv o JOIN securities s ON s.security_id = o.security_id AND NOT s.is_delisted"

# `snapshot`·`series` 가 함께 쓰는 SELECT 목록. `listed_shares` 는 **`o` 쪽**이다.
_ROW_SELECT = """
  s.code, s.name, s.market, s.sector,
  o.open, o.high, o.low, o.close, o.change, o.change_rate,
  o.volume, o.value, o.market_cap, o.listed_shares, o.trade_date
"""


# ==================================================
# 1. 값 되돌리기 — 순수 함수라 DB 없이 검사된다
# ==================================================
def to_trade_date(bas_dd: str) -> date:
    """`'20260731'` → `date(2026, 7, 31)`.

    ⚠️ **문자열을 그대로 바인딩하면 안 된다.** asyncpg 는 `date` 컬럼에 `str` 을 받으면
    `DataError: 'str' object has no attribute 'toordinal'` 로 **첫 질의부터** 죽는다.
    """
    return datetime.strptime(bas_dd, "%Y%m%d").date()


def to_bas_dd(value: date) -> str:
    """`date(2026, 7, 31)` → `'20260731'`. KRX 표기이자 이 레포의 날짜 열쇠다."""
    return value.strftime("%Y%m%d")


def to_float(value: Any) -> Optional[float]:
    """`Decimal` 을 `float` 으로 내린다. `None` 은 그대로 둔다.

    `change_rate` 만 해당한다 — 나머지 수치 컬럼은 정수라 asyncpg 가 `int` 로 준다.
    정밀도를 잃지만 **소비자 중 정밀도가 필요한 곳이 없다**(전부 비교·합·백분위).
    반대로 `Decimal` 이 그대로 흐르면 `float` 과 섞이는 산술에서 `TypeError` 가 나고,
    JSON 직렬화가 예외를 삼키는 자리(tmp_cache)에서는 **아무 말 없이** 캐시가 죽는다.
    """
    if isinstance(value, Decimal):
        return float(value)
    return value


def to_row(record: Any) -> Dict:
    """조회 결과 한 줄을 `krx_store` 와 같은 모양의 딕셔너리로 만든다.

    `krx_store._rows_to_dicts()` 와 같은 계약이다 — `bas_dd` 는 **빼고** `date` 를 **맨 뒤에**
    붙인다. 무심코 `trade_date` 를 남기면 응답에 없던 키가 하나 늘어나는데,
    pydantic 이 여분 필드를 잘라 내므로 화면만 봐서는 아무도 못 알아챈다.
    """
    # strict=True 가 뜻을 가진다 — SELECT 목록과 ROW_COLUMNS 가 어긋나면 **그 자리에서** 죽는다.
    row = dict(zip(ROW_COLUMNS, record[:len(ROW_COLUMNS)], strict=True))
    row["change_rate"] = to_float(row.get("change_rate"))
    trade_date = record[len(ROW_COLUMNS)]
    row["date"] = trade_date.isoformat() if trade_date else None
    return row


# ==================================================
# 2. 조회 — 전부 `krx_store` 의 같은 이름 함수와 계약이 같다
# ==================================================
def _fetch(sql: str, params: Optional[Dict] = None) -> List[Any]:
    """질의 하나를 동기 다리 위에서 돌리고 행 목록을 돌려준다.

    ⚠️ **예외를 삼키지 않는다.** 접속 실패를 빈 목록으로 바꾸면 DB 장애가 "그 날짜에 자료가
    없음"으로 위장되고, 부르는 쪽이 그것을 축약본 폴백 신호로 읽어 화면이 150거래일짜리로
    조용히 강등된다. 무엇이 폴백이고 무엇이 고장인지는 `krx_store` 가 정한다.
    """
    async def go():
        async with db.connect() as conn:
            result = await conn.execute(text(sql), params or {})
            return result.fetchall()

    try:
        return db.run_sync(go())
    except OSError as exc:
        # ⚠️ **삼키는 것이 아니라 길을 여는 것이다.** 예외는 그대로 올라간다 —
        #    바꾸는 것은 메시지뿐이다. S5 에서 로컬 기본값이 `postgres` 가 되면서
        #    "DB 를 안 띄우고 앱을 켠다" 가 새 clone 의 **첫 경험**이 됐다. 그때 화면에
        #    `[Errno 111] Connect call failed` 만 남으면 원인이 저장소 전환으로 안 보인다.
        #    (`OSError` 만 잡는다 — 인증 실패·DB 이름 오타는 asyncpg 가 더 정확히 말한다.)
        #    공통 처방은 `db.unreachable()` 한 곳에 있고, **되돌리는 법만** 여기서 얹는다.
        raise db.unreachable(
            exc, what="시세 저장소(Postgres)",
            extra=("SQLite 로 되돌린다: STORE_BACKEND=sqlite (S5 이전과 같아진다)",),
        ) from exc


def is_empty() -> bool:
    """시세가 한 줄이라도 있는가. `krx_store._cache_is_empty()` 의 짝이다."""
    rows = _fetch("SELECT max(trade_date) FROM ohlcv")
    return not (rows and rows[0][0])


def latest_date() -> Optional[str]:
    """시세가 있는 가장 최근 거래일 (`YYYYMMDD`). 없으면 None.

    파티션마다 `trade_date` 인덱스를 거꾸로 한 행씩만 읽는다 — 실측 0.8ms.
    """
    rows = _fetch("SELECT max(trade_date) FROM ohlcv")
    value = rows[0][0] if rows else None
    return to_bas_dd(value) if value else None


def available_dates(limit: int = 400) -> List[str]:
    """시세가 있는 거래일 목록 (`YYYYMMDD` · 최근순).

    ⚠️ **`ohlcv_sync_log` 가 아니라 `ohlcv` 를 본다.** 대장 쪽이 56배 빠르지만
    (45ms → 0.8ms 실측) 그것은 "받아 봤다"는 기록이고 이 함수의 계약은 "자료가 있다"이다.
    SQLite 쪽도 `daily_price` 를 본다. 두 표가 갈릴 수 있는 경로가 실재하므로
    (`ON CONFLICT DO NOTHING` 과 `DO UPDATE` 의 비대칭 — load_pg.py:462·468) 빠른 쪽을
    택하려면 갈림을 감시하는 자가 먼저 있어야 한다. 그것은 S9 의 일이다.
    """
    rows = _fetch(
        "SELECT DISTINCT trade_date FROM ohlcv ORDER BY trade_date DESC LIMIT :limit",
        {"limit": limit},
    )
    return [to_bas_dd(row[0]) for row in rows]


def snapshot(bas_dd: str, market: Optional[str] = None) -> List[Dict]:
    """해당 거래일의 전 종목. `market` 을 주면 그 시장만 추린다.

    `trade_date` 가 파티션 키라 프루닝이 되고, 그 안에서 `ohlcv_*_trade_date_idx` 를 탄다.
    """
    sql = f"SELECT {_ROW_SELECT} FROM {_JOIN} WHERE o.trade_date = :trade_date"
    params: Dict[str, Any] = {"trade_date": to_trade_date(bas_dd)}
    if market:
        sql += " AND s.market = :market"
        params["market"] = market
    return [to_row(row) for row in _fetch(sql, params)]


def series(code: str, days: int = 250, end: Optional[str] = None) -> List[Dict]:
    """종목 하나의 일봉 시계열 (날짜 **오름차순**).

    `security_id` 를 먼저 푸는 두 걸음이다 — 머리말의 실측표 참조.
    종목을 못 찾으면 질의를 한 번 덜 하고 빈 목록으로 끝난다.
    """
    found = _fetch(
        "SELECT security_id FROM securities WHERE code = :code AND NOT is_delisted LIMIT 1",
        {"code": code},
    )
    if not found:
        return []

    sql = f"""
        SELECT {_ROW_SELECT}
        FROM {_JOIN}
        WHERE o.security_id = :security_id
    """
    params: Dict[str, Any] = {"security_id": found[0][0], "days": days}
    if end:
        sql += " AND o.trade_date <= :end"
        params["end"] = to_trade_date(end)
    # 최근 것부터 days 개를 가져온 뒤 뒤집는다 (차트는 왼쪽이 과거) — SQLite 쪽과 같다.
    sql += " ORDER BY o.trade_date DESC LIMIT :days"

    return list(reversed([to_row(row) for row in _fetch(sql, params)]))


def window(days: int = 60,
           columns: Sequence[str] = ("code", "bas_dd", "close", "value", "volume")) -> List[Dict]:
    """최근 `days` 거래일치를 필요한 컬럼만 골라 한 번에 읽는다 (날짜 오름차순).

    ⚠️ **모르는 컬럼이면 예외다.** `krx_bundle.window()` 는 같은 경우에 빈 목록을 주는데,
    그쪽은 축약본에 **실제로 없는** 컬럼(`market_cap` 등)을 묻는 상황이라 "없다"가 맞는 답이다.
    이쪽은 열세 컬럼을 전부 갖고 있으므로 모르는 이름이 온다는 것은 자료가 없다는 뜻이 아니라
    **부르는 쪽의 오타**다. SQLite 경로도 그때 `OperationalError` 로 죽는다(실측) — 같이 죽는다.
    빈 목록으로 돌려주면 화면이 조용히 비고 원인이 저장소 전환으로 보인다.

    ⚠️ **SQLite 와 정렬 사정이 정반대다.** 저쪽은 PK 가 `(bas_dd, code)` 라
    `ORDER BY bas_dd` 가 공짜였다(그 실측이 `krx_store.window()` docstring 에 있다).
    이쪽 PK 는 `(security_id, trade_date)` 라 공짜가 아니다. 그래서 하한을 **서브질의로**
    걸어 파티션 프루닝을 태운다 — 그러면 `trade_date` 인덱스를 순서대로 읽게 되어
    플래너가 Sort 노드를 아예 만들지 않는다 (실측 166,057행 · 299ms · 디스크 정렬 없음).

    ⚠️ 날짜 오름차순은 **계약이다.** `market_data.py:71-76` 이 "SQL 이 날짜 오름차순으로
    준다"에 명시적으로 기대어 앞에서부터 담기만 한다.

    ⚠️ **날짜 안에서는 `code` 순이다. 이것도 계약이다.** SQLite 는 PK 가 `(bas_dd, code)` 라
    그 순서가 공짜로 따라왔고, 여기서는 정렬을 명시해야 나온다. 빼면 두 가지가 생긴다 —
    ① Postgres 쪽 순서가 **미정**이 된다(힙 순서라 VACUUM 한 번에 바뀔 수 있다),
    ② `market_data.screening_funnel()` 의 `sorted(key=score)` 가 **안정 정렬**이라
       점수가 같은 종목의 등수가 입력 순서를 그대로 따라간다. 실측으로 75.6점 동점인
       셀트리온·한국가스공사의 10·11위가 뒤바뀌었다.
    정렬 비용은 실측 299ms → 446ms 인데, 이 함수가 대신하는 SQLite 경로가 **800ms** 다
    (`krx_store.window()` docstring 의 실측). 즉 순서를 사도 여전히 더 빠르다.
    """
    unknown = [c for c in columns if c not in WINDOW_COLUMNS]
    if unknown:
        raise ValueError(
            f"window() 가 모르는 컬럼을 받았다: {unknown}. "
            f"쓸 수 있는 이름은 {', '.join(sorted(WINDOW_COLUMNS))} 다."
        )
    wanted = list(columns)

    select = ", ".join(f"{WINDOW_COLUMNS[c]} AS {c}" for c in wanted)
    rows = _fetch(
        f"""
        SELECT {select}
        FROM {_JOIN}
        WHERE o.trade_date >= (
            SELECT min(d) FROM (
                SELECT DISTINCT trade_date AS d FROM ohlcv ORDER BY d DESC LIMIT :days
            ) recent
        )
        ORDER BY o.trade_date, s.code
        """,
        {"days": days},
    )
    return [
        {col: to_float(value) if col == "change_rate" else value
         for col, value in zip(wanted, row, strict=True)}
        for row in rows
    ]


def stats() -> Dict:
    """시세 현황 — 화면 배지와 `/api/krx/status` 가 쓴다.

    `krx_store.stats()` 와 **같은 키**를 낸다. `db_path`·`db_size_mb` 는 파일 개념이라
    Postgres 에 대응물이 없는데 `CacheStats`(krx_router.py:106-116)가 **둘 다 필수**로
    요구한다. 없는 척하는 대신 있는 것을 정직하게 채운다 —
    경로 자리에는 DB 이름을, 크기 자리에는 **파티션까지 합한** 실제 크기를 MB 로 넣는다.
    화면(`krx.html:194-196`)이 그대로 "○○MB" 로 찍는데, 그 숫자가 실제로 그 자료의 크기다
    (실측 107MB — SQLite 원본 117.5MB 와 같은 자료다).
    """
    rows = _fetch("""
        SELECT count(*) AS rows, count(DISTINCT trade_date) AS days,
               count(DISTINCT security_id) AS codes,
               min(trade_date) AS first, max(trade_date) AS last,
               current_database() AS dbname,
               -- ⚠️ `pg_total_relation_size('ohlcv')` 만 쓰면 **0** 이 나온다. ohlcv 는
               -- 파티션 부모라 자기 자신은 비어 있고 자료는 14개 파티션에 들어 있다.
               -- 화면(krx.html:194-196)이 그 값을 "0.0MB" 로 찍어 고장처럼 보인다.
               (SELECT sum(pg_total_relation_size(relid))
                  FROM pg_partition_tree('ohlcv')) AS bytes
        FROM ohlcv
    """)
    row = rows[0]
    first, last = row[3], row[4]
    return {
        "rows": row[0], "days": row[1], "codes": row[2],
        "first_date": to_bas_dd(first) if first else None,
        "last_date": to_bas_dd(last) if last else None,
        "db_path": f"{row[5]}.ohlcv",
        "db_size_mb": round((row[6] or 0) / 1024 / 1024, 1),
        # ⚠️ `db` 그대로다. `postgres`·`pg` 를 새로 만들지 않는다 — 근거는 ADR-DS-0015 §3.
        "mode": "db",
        "notes": [],
    }


def lookup_security(code_or_name: str) -> Optional[Dict]:
    """종목코드 또는 한글 종목명으로 종목 하나를 찾는다 — `{code, name, market}`.

    `krx_store.lookup_security()` 의 짝이다 (전환 S5 · ADR-DS-0018).
    원래 `stock_service._lookup_krx()` 가 `store.connect()` 로 SQLite 에 생 SQL 세 개를
    던지던 자리다. 그 우회가 남아 있으면 스위치를 켜도 **한글 종목명 검색만** 계속
    옛 저장소를 봐서, 두 저장소가 갈린 날 그 화면만 조용히 낡는다.

    ## 찾는 순서 — SQLite 와 같아야 한다

    ① 6자리 숫자면 코드로 · ② 아니면 이름이 정확히 같은 것 · ③ 그것도 없으면 앞부분이 같은 것.
    ②③ 은 `ORDER BY 거래일 DESC, 거래대금 DESC` 로 하나를 고른다 — "삼성" 처럼 여러 개가
    걸리는 입력에서 가장 대표적인 종목이 나오게 하려는 것이다.

    ⚠️ **동점 처리에 `ohlcv` 가 필요하다.** SQLite 는 `daily_price` 한 표에 이름과 거래대금이
    같이 있어 정렬이 공짜였다. 여기서는 이름이 `securities`, 거래대금이 `ohlcv` 라 이어야 한다.
    이 두 줄을 빼고 `securities` 만 보면 **정렬 근거가 사라져** 힙 순서가 나온다 —
    "삼성" 이 삼성전자가 아니라 삼성공조를 가리키게 되고, 오류는 뜨지 않는다.

    ⚠️ **`JOIN LATERAL` 로 종목마다 최근 한 줄만 본다.** `WHERE s.name = :needle` 로 잇고
    통째로 정렬하면 후보 종목의 **전 구간**(297거래일)을 읽고 버린다. 이쪽은 머리말의
    "`security_id` 를 먼저 푼다" 와 같은 모양이라 파티션마다 PK 를 곧장 탄다.

    ⚠️ **`LIKE` 의 `%`·`_` 를 이스케이프하지 않는다.** SQLite 쪽도 안 한다 — 여기서 한쪽만
    바꾸면 같은 입력에 두 저장소가 다른 답을 낸다. 이 함수의 계약은 "같은 답" 이 먼저다.
    (한글 종목명에는 그 글자가 없어 실제로 갈리는 입력이 없다. 바꾸려면 양쪽을 같이 바꾼다.)
    """
    needle = code_or_name.strip()
    if not needle:
        return None

    # ⚠️ 코드 판정은 대문자로 올려서 하고, **찾을 때도 올린 값을 넘긴다.** 우리 `code` 는
    #    전부 대문자라 `0009k0` 을 그대로 넘기면 0행이 나온다. 반대로 이름 검색은 원본을
    #    쓴다 — 'iM금융지주' 처럼 소문자가 뜻을 갖는 종목명이 있어 올리면 못 찾는다.
    if codes.looks_like_code(needle):
        needle = needle.upper()
        rows = _fetch(
            "SELECT code, name, market FROM securities "
            "WHERE code = :code AND NOT is_delisted LIMIT 1",
            {"code": needle},
        )
        return _to_security(rows)

    # 정확히 일치 → 앞부분 일치. 앞엣것이 걸리면 뒤는 묻지 않는다 (SQLite 와 같다).
    for clause, value in (("s.name = :needle", needle), ("s.name LIKE :needle", f"{needle}%")):
        rows = _fetch(
            f"""
            SELECT c.code, c.name, c.market
            FROM (
                SELECT s.security_id, s.code, s.name, s.market
                FROM securities s
                WHERE {clause} AND NOT s.is_delisted
            ) c
            JOIN LATERAL (
                SELECT o.trade_date, o.value
                FROM ohlcv o
                WHERE o.security_id = c.security_id
                ORDER BY o.trade_date DESC
                LIMIT 1
            ) last ON true
            ORDER BY last.trade_date DESC, last.value DESC
            LIMIT 1
            """,
            {"needle": value},
        )
        if rows:
            return _to_security(rows)
    return None


def _to_security(rows: List[Any]) -> Optional[Dict]:
    """`(code, name, market)` 한 줄을 `krx_store` 와 같은 모양의 딕셔너리로."""
    if not rows:
        return None
    return {"code": rows[0][0], "name": rows[0][1], "market": rows[0][2]}
