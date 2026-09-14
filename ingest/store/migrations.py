"""SQLite 스키마를 버전으로 관리한다 — `PRAGMA user_version` 기반.

**왜 필요한가.** `data/krx_cache.db` 에는 이미 시세 900만 행이 들어 있다(1.65GB).
여기에 **표를 새로 얹어야** 하는데 — 어디까지 받았는지 기록하는 표, 하루 몇 번 불렀는지
세는 표 — 지금 스키마를 만드는 방법은 `krx_store.init_db()` 의
`CREATE TABLE IF NOT EXISTS` 뿐이라 *"이미 있는 표에 칸을 추가한다"* 를 표현할 길이 없다.
Postgres 쪽 `sql/init/*.sql` 은 **볼륨 최초 생성 시 1회만** 도는 init 스크립트라 더더욱
아니다(문서화된 유일한 갱신 방법이 `down -v`, 즉 볼륨을 통째로 지우는 것이다).

**왜 도구를 안 쓰나.** `alembic` 이 SQLite 에 주는 것은 batch(move-and-copy) 모드인데,
그건 표를 통째로 복사하는 절차다. 우리에게 필요한 `ADD COLUMN` 은 그럴 필요가 없다.
900만 행으로 직접 재 봤다(SQLite 3.53.2 · 727MB · WAL):

    ADD COLUMN (DEFAULT 없음)          0.72ms
    ADD COLUMN NOT NULL DEFAULT 0      0.16ms   ← DEFAULT 를 붙여도 공짜다
    RENAME COLUMN                      1.00ms
    ADD COLUMN + CHECK 제약        5,088ms      ← 약 3만 배
    DROP COLUMN                   19,163ms      ← 900만 행 전체 UPDATE 와 동급

`ADD COLUMN` 이 공짜인 이유는 SQLite 가 **`sqlite_schema` 의 SQL 텍스트만 고치고 행은
건드리지 않기** 때문이다(공식 문서). 즉 alembic 은 우리가 쓰지 않을 비싼 경로를 위해
의존성을 하나 늘리는 일이다. `yoyo`·`sqlite-utils` 도 같은 이유로 기각했다.

규약 네 가지 — 전부 실측으로 확인했다
-------------------------------------
1. **`BEGIN IMMEDIATE` 를 명시한다.** 파이썬 `sqlite3` 의 기본 모드는 **DDL 을 암묵
   트랜잭션에 넣지 않는다.** 이걸 빼면 마이그레이션이 원자적이지 않다.
2. **`executescript()` 를 쓰지 않는다.** 열린 트랜잭션을 **먼저 커밋해 버려서** 롤백이
   불가능해진다(실측: `ROLLBACK` 이 *"no transaction is active"* 로 실패하고 반쪽 스키마가
   그대로 남았다). 문장을 `execute()` 로 하나씩 돌린다.
3. **`PRAGMA user_version=N` 을 스키마 변경과 같은 트랜잭션에 넣는다.** *"헤더 값이라
   트랜잭션 밖"* 이라는 통설은 실측으로 반증됐다 — `ROLLBACK` 하면 이전 값으로 돌아온다.
   같이 묶으면 *"표는 바뀌었는데 버전이 안 올라간"* 중간 상태가 원천 차단된다.
4. **`PRAGMA user_version` 만 쓴다.** `schema_version` 은 SQLite 내부용이고 공식 문서가
   손대면 **DB 가 깨진다**고 경고한다.

스키마를 쓸 때 지킬 것
----------------------
- 기존 표에 칸을 더할 때 **CHECK 제약을 붙이지 않는다** (위 표: 3만 배).
  값 검증은 애플리케이션에서 한다. **단 `CREATE TABLE` 로 새로 만드는 빈 표는 예외다** —
  검사할 기존 행이 없어 비용이 0 이고, Postgres 쪽 스키마도 이미 그렇게 하고 있다.
- **`DROP COLUMN` 을 쓰지 않는다** (19초). 안 쓰는 칸은 그냥 둔다.
- `ADD COLUMN` 은 `DEFAULT CURRENT_TIMESTAMP`·`UNIQUE`·`PRIMARY KEY` 를 **거부한다**
  (공식 제약). 상수 DEFAULT 만 쓰고, 인덱스가 필요하면 뒤에 `CREATE INDEX` 를 따로 건다.

사용법
------
    from ingest.store.migrations import migrate_path
    applied = migrate_path()          # data/krx_cache.db 를 최신 버전으로
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from common.paths import krx_db_path  # DB 경로 — 유일한 정의


class MigrationError(RuntimeError):
    """마이그레이션이 실패했다. 무엇을 해야 하는지까지 메시지에 담는다."""


#: 마이그레이션 한 칸에 들어갈 수 있는 것 — 그냥 SQL 문자열이거나, **연결을 받아
#: SQL 을 만들어 주는 함수**다. 함수를 허용하는 이유는 둘이다:
#:
#:   ① `ALTER TABLE ... ADD COLUMN` 에는 `IF NOT EXISTS` 가 없다. 같은 이름이 이미 있으면
#:      **예외를 던진다.** 그런데 이 파일의 규약은 *"문장은 여러 번 돌려도 같은 결과"* 다.
#:      문자열만으로는 그 둘을 동시에 만족할 수 없어서, 칸이 있는지 보고 문장을 만들거나
#:      `None` 을 주는 함수를 받는다.
#:
#:   ② **표 재구성은 문장이 여럿인데 판단은 하나다.** SQLite 는 기본키를 `ALTER` 로 못
#:      바꿔서 새 표를 만들고 옮기고 지우고 이름을 바꾼다(v13). 이때 *"고칠 것이 있나"*
#:      를 문장마다 다시 재면 **첫 문장이 표를 바꾼 뒤 조건이 뒤집혀 나머지가 건너뛰어
#:      진다.** 그래서 한 번 판단하고 문장 **목록**을 통째로 돌려준다.
#:
#: `None` 을 돌려주면 그 문장은 **건너뛴다** — 할 일이 없다는 뜻이다. 빈 목록도 같다.
#: 목록의 문장들은 나머지와 **같은 트랜잭션**에서 차례로 돈다 — 가운데서 터지면 앞
#: 문장까지 함께 되돌아간다.
Statement = "str | Callable[[sqlite3.Connection], Optional[str | List[str]]]"


def _add_column(table: str, column: str, definition: str) -> Callable:
    """`ADD COLUMN` 을 **칸이 없을 때만** 내는 지연 문장을 만든다.

    `add_column_sql()` 은 연결이 있어야 판단할 수 있는데 `MIGRATIONS` 는 모듈이 읽힐 때
    만들어지므로 아직 연결이 없다. 그래서 판단을 실행 시점으로 미룬다.
    """
    return lambda conn: add_column_sql(conn, table, column, definition)


#: v13 이 만드는 `index_price` 의 모양. **`krx_index.SCHEMA` 와 같아야 한다.**
#:
#: 두 곳에 적는 이유는 `add_column_sql` 의 경고와 같다 — 표를 만드는 경로가 둘이다.
#: 마이그레이션이 먼저 돌 수도 있고(`migrate_path`), `init_db()` 가 먼저일 수도 있다.
#: 한쪽만 고치면 **그 DB 는 옛 모양으로 태어난 뒤 마이그레이션이 다시 돌지 않아**
#: 영영 안 고쳐진다. 여기서 문자열을 나눠 갖지 않고 각자 적되, 두 경로가 같은 기본키를
#: 만드는지 시험이 대조한다(`tests/test_krx_index.py`).
#:
#: 여기서 `krx_index` 를 import 하지 않는 것은 순환 때문이다 — 그쪽이 `collect_log` 를
#: 거쳐 이 파일을 부른다.
INDEX_PRICE_SCHEMA_V13 = """
CREATE TABLE {table} (
  bas_dd       TEXT    NOT NULL,   -- 기준일자 YYYYMMDD
  index_name   TEXT    NOT NULL,   -- 지수명 "코스피 200" (띄어쓰기 포함)
  index_class  TEXT    NOT NULL,   -- KOSPI / KOSDAQ 🔴 기본키의 일부다
  open         REAL,               -- ⚠️ 지수는 실수다. INTEGER 로 두면 등락이 사라진다
  high         REAL,
  low          REAL,
  close        REAL,
  change       REAL,               -- 전일대비
  change_rate  REAL,               -- 등락률(%)
  volume       INTEGER,            -- 누적거래량
  value        INTEGER,            -- 누적거래대금
  market_cap   INTEGER,            -- 시가총액
  -- 🔴 시장이 키에 있어야 한다. 두 시장이 같은 이름의 업종지수를 각각 갖는다.
  PRIMARY KEY (bas_dd, index_name, index_class)
)
"""

#: `index_price` 의 조회 인덱스. 표를 다시 만들면 인덱스도 함께 사라지므로 다시 만든다.
INDEX_PRICE_INDEX_SQL = ("CREATE INDEX IF NOT EXISTS idx_index_name_date "
                         "ON index_price(index_name, bas_dd)")


def _index_price_pk(conn: sqlite3.Connection) -> List[str]:
    """`index_price` 의 기본키 칸을 키 순서대로. 표가 없으면 빈 목록."""
    rows = conn.execute("PRAGMA table_info(index_price)").fetchall()
    return [r[1] for r in sorted((r for r in rows if r[5] > 0), key=lambda r: r[5])]


def _rebuild_index_price(conn: sqlite3.Connection) -> Optional[List[str]]:
    """🔴 `index_price` 의 기본키에 `index_class` 를 넣는다 (v13).

    ## 왜 — 2026-09-07 에 KOSPI 업종지수 40,324행을 잃었다

    옛 기본키가 `(bas_dd, index_name)` 이라 **시장이 없다.** 그런데 KOSPI 와 KOSDAQ 은
    `건설`·`금속`·`화학` 처럼 **같은 이름의 업종지수를 각각** 가진다. KOSDAQ 을 받자
    `INSERT OR REPLACE` 가 같은 이름의 KOSPI 행 17종 40,324행을 조용히 덮어썼다.

    🔴 **행 수로는 못 잡는다** — 오히려 늘었다(196,272 → 244,108). 수집기는 정상 종료했고
    오류도 없었다. `index_class` 별로 세어 보고서야 드러났다.

    그래서 막는 자리를 **"쓰기 전"** 으로 옮긴다. 검사는 덮어쓴 값을 되살리지 못한다.
    답하는 것은 경고가 아니라 **PRIMARY KEY** 다.

    ## 왜 표를 새로 만드나

    SQLite 는 `ALTER TABLE` 로 기본키를 못 바꾼다. 공식 문서가 권하는 절차 그대로 —
    새 표 → 복사 → 기존 삭제 → 이름 변경 → 인덱스 재생성. 다섯이 **한 트랜잭션**에
    있어야 중간에 죽어도 반쪽이 안 남는다(`migrate()` 가 감싼다).

    ## 세 경우

        표가 없다        → 새 기본키로 만든다. `init_db()` 보다 먼저 도는 DB 가 있다
        이미 새 기본키다 → 아무것도 하지 않는다 (`None`)
        옛 기본키다      → 재구성한다

    ## `index_class` 가 비어 있으면 세운다

    새 기본키는 `NOT NULL` 이라 빈 시장은 못 들어간다. 여기서 조용히 버리면 행이
    사라지고, 아무 말 없이 `NOT NULL` 위반으로 터지면 사람이 무엇을 해야 할지 모른다.
    **무엇을 해야 하는지까지 담아** 세운다. (실측 2026-09-07: 우리 DB 는 0행이다.)
    """
    현재 = _index_price_pk(conn)

    if not 현재:
        # 표가 없다 — 새 기본키로 태어나게 한다. `init_db()` 가 나중에 불려도
        # `CREATE TABLE IF NOT EXISTS` 라 이 모양을 덮지 않는다.
        return [INDEX_PRICE_SCHEMA_V13.format(table="index_price"), INDEX_PRICE_INDEX_SQL]

    if "index_class" in 현재:
        return None                                  # 이미 됐다

    빈시장 = conn.execute(
        "SELECT COUNT(*) FROM index_price WHERE index_class IS NULL OR index_class = ''"
    ).fetchone()[0]
    if 빈시장:
        raise MigrationError(
            f"index_price 에 index_class 가 빈 행이 {빈시장:,}개 있어 기본키에 넣을 수 없다.\n"
            "  왜 필요한가: 기본키에 시장이 없으면 KOSDAQ 업종지수가 같은 이름의 KOSPI\n"
            "               행을 덮어쓴다 (2026-09-07 에 40,324행을 잃었다).\n"
            "  할 일: 그 행들의 시장을 채우거나 지운 뒤 다시 실행한다.\n"
            "    SELECT bas_dd, index_name FROM index_price\n"
            "     WHERE index_class IS NULL OR index_class = '';"
        )

    칸 = ", ".join(column_names(conn, "index_price"))
    return [
        INDEX_PRICE_SCHEMA_V13.format(table="index_price_v13"),
        f"INSERT INTO index_price_v13 ({칸}) SELECT {칸} FROM index_price",
        "DROP TABLE index_price",
        "ALTER TABLE index_price_v13 RENAME TO index_price",
        INDEX_PRICE_INDEX_SQL,                       # 표와 함께 사라졌으므로 다시 만든다
    ]


# ==================================================
# 1. 마이그레이션 목록
# ==================================================
# 한 칸이 한 버전이다. **이미 배포된 항목은 절대 고치지 않는다** — 남의 DB 는 이미
# 그 버전을 지났으므로 고쳐 봐야 적용되지 않고, 새 DB 와 낡은 DB 만 갈라진다.
# 바꾸고 싶으면 **뒤에 새 버전을 더한다.**
#
# 문장은 전부 **여러 번 돌려도 같은 결과**여야 한다(`IF NOT EXISTS`). 이 파일이 도입되기
# 전에 만들어진 DB 는 `user_version` 이 0 이면서 기본 표는 이미 갖고 있기 때문이다.
#
# 🔴 **번호는 이름이 아니라 인덱스다.** `migrate()` 는 `range(현재, LATEST_VERSION)` 으로
#    도므로, 두 갈래에서 각자 "다음은 v5" 라고 붙인 채 합쳐지면 **이미 v5 를 적용한 DB 는
#    나중에 v5 자리에 들어온 항목을 영원히 건너뛴다.** 예외도 경고도 없다.
#
#    그래서 번호를 **미리 배정하고 그 순서대로 직렬로만 합친다.**
#
#      v5  반입 (inbox_batch · inbox_accepted · inbox_quarantine)   ← 2026-09-01 적용
#      v6  공시 시점정합 (dart_financial · dart_disclosure)          ← 2026-09-02 적용
#      v7  거시 통계 (macro_series)                                  ← 2026-09-02 적용
#      v8  수집 실행 기록 (ingest_run · ingest_run_stage)            ← 2026-09-02 적용
#      v9  수정주가 4칸 · 거래일 달력 (daily_price.adj_* · trading_calendar)
#                                                                   ← 2026-09-02 적용
#      v10 종목 신원 · 법인 개요 (stock_identity · corp_profile)      ← 2026-09-03 적용
#      v11 종목기본정보 (stock_base_info · 우선주 판별)               ← 2026-09-03 적용
#      v12 텍스트 신호 (text_signal — 공시 제목 감성 확률 3칸)  ← 2026-09-04 적용
#      v13 지수 기본키에 시장 (index_price PK + index_class)     ← 2026-09-07 적용
#      v14 배당 (dividend — 현금·주식배당과 배당락일)            ← 2026-09-09 적용
#      v15 총수익 (daily_price.adj_close_tr · adj_dividend)      ← 2026-09-10 적용
#      v16 기업행위 (corporate_action — 사건 표 · chain 대조)     ← 2026-09-10 적용
#      v17 다음 빈 번호
#
#    ⚠️ v5·v6 은 처음에 공시·거시로 **예약**돼 있었는데, 실제로 먼저 온 것은 반입이라
#       한 칸씩 밀었다. 밀 수 있었던 이유는 **그 번호를 적용한 DB 가 아직 없기 때문이다** —
#       예약은 자리를 비워 둔 것이지 배포된 것이 아니다. 이미 배포된 항목은 이렇게 못 민다.
#
#    배정표는 `ingest/store/sqlite_db.py` 에도 있다. 둘을 함께 고친다.
MIGRATIONS: Sequence[Tuple[str, Sequence[str]]] = (
    (
        "v1: 수집 대장 · 호출 예산",
        (
            # ── 수집 대장 ───────────────────────────────────────────────
            # "어느 출처의 어디까지를 언제 받았나" 를 한 줄씩 남긴다. 이게 있어야
            # 중간에 죽어도 받은 데를 건너뛰고 이어서 받을 수 있다.
            #
            # 기존 `fetch_log`·`index_fetch_log` 를 **일반화한 것**이다. 그 둘은 이미
            # 0건으로 받은 날에도 `rows=0` 행을 남겨서 **"받아 봤더니 없었다"(휴장)와
            # "아직 안 받았다"를 구별**하고 있었다. 이 구별이 없으면 휴장일마다 영원히
            # 다시 요청하게 된다. 여기서는 그 방식을 날짜·시장 축에 묶여 있던 것에서
            # 풀어 어떤 출처든 쓸 수 있게 넓힌다.
            #
            # ⚠️ 기존 두 표를 **지우거나 옮기지 않는다.** 900만 행 수집이 그 위에서
            #    돌고 있고, 옮기다 실패하면 16년치 백필을 다시 받아야 한다.
            #    새 코드가 이 표를 쓰고, 옛 표는 제자리에 둔 채 읽기로만 참조한다.
            """
            CREATE TABLE IF NOT EXISTS collect_log (
              source            TEXT    NOT NULL,
              target            TEXT    NOT NULL,
              status            TEXT    NOT NULL,
              rows              INTEGER NOT NULL DEFAULT 0,
              last_success_at   TEXT,
              last_attempted_at TEXT    NOT NULL,
              cursor            TEXT,
              note              TEXT,
              PRIMARY KEY (source, target),
              -- 새 표라 검사할 기존 행이 없다 → CHECK 가 공짜다.
              --   ok              받았고 행이 있다
              --   empty           받아봤는데 0건. **미수집과 다르다** (휴장 등)
              --   error           시도했는데 실패했다
              --   quota_exhausted 한도에 닿아 아껴 멈췄다. **실패가 아니다**
              --   out_of_range    출처가 제공하지 않는 구간이다 (예: KRX 지수 2010 이전)
              CHECK (status IN ('ok', 'empty', 'error', 'quota_exhausted', 'out_of_range'))
            )
            """,
            # 수집 현황 화면이 "출처별 마지막 성공 시각"을 뽑을 때 쓴다.
            "CREATE INDEX IF NOT EXISTS idx_collect_source_success "
            "ON collect_log(source, last_success_at)",
            # 품질 검사가 "실패한 것만" 훑을 때 쓴다.
            "CREATE INDEX IF NOT EXISTS idx_collect_status ON collect_log(source, status)",

            # ── 호출 예산 ───────────────────────────────────────────────
            # 출처마다 하루 호출 한도가 있다. 넘기 전에 스스로 멈추려면 오늘 몇 번
            # 불렀는지를 세어야 한다.
            #
            # `kst_date` 를 PK 에 넣어 **자정 리셋 로직 자체를 없앤다.** 날짜가 바뀌면
            # 그냥 다른 행이라 0 부터 시작한다 — 리셋을 "잊어버리는" 버그가 불가능해진다.
            #
            # 파일이 아니라 DB 에 두는 이유는 **프로세스가 여럿이기 때문**이다. 배치
            # (`scripts/fetch_*.py`)와 화면의 즉시수집 버튼이 따로 도는데, 모듈 전역
            # 변수로 세면 각자 0 부터 시작해 한도를 두 배로 쓴다.
            """
            CREATE TABLE IF NOT EXISTS call_budget (
              source       TEXT    NOT NULL,
              kst_date     TEXT    NOT NULL,
              used         INTEGER NOT NULL DEFAULT 0,
              daily_limit  INTEGER NOT NULL,
              warned_at    TEXT,
              PRIMARY KEY (source, kst_date)
            )
            """,
        ),
    ),
    (
        "v2: 수집 대장에 시도 횟수",
        (
            # 실패한 대상은 다시 받아야 하지만 **영원히 다시 받으면 안 된다.** 어떤
            # 날짜가 구조적으로 실패하면(출처가 그 날을 영영 안 주는 경우) 배치를 돌릴
            # 때마다 같은 자리에서 호출을 태운다. 몇 번 시도했는지를 세야 "이건 그만"
            # 이라고 판단할 수 있다.
            #
            # 횟수를 `note` 문자열에 적지 않는 이유는 그러면 읽을 때마다 파싱해야 하고,
            # 파싱은 언젠가 실패하기 때문이다. 칸으로 두면 SQL 이 직접 거른다.
            #
            # `ADD COLUMN ... DEFAULT` 는 SQLite 가 스키마 텍스트만 고치고 행은 건드리지
            # 않아 사실상 공짜다 (900만 행에서 0.16ms 실측 2026-08-26). 단 **CHECK 제약을
            # 함께 걸면 전체 스캔이 일어나 3만 배 느려진다** — 그래서 걸지 않는다.
            "ALTER TABLE collect_log ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0",
        ),
    ),
    (
        "v3: 응답 원문 보존",
        (
            # ── 응답 원문 ───────────────────────────────────────────────
            # **왜 정규화된 표만으로는 부족한가.** 정규화는 틀린다. 필드 이름을 잘못
            # 매핑하거나, 숫자 파싱이 어떤 값에서만 깨지거나, 나중에 필요해진 칸을
            # 그때는 안 담았거나. 그런데 원문이 없으면 **고치는 유일한 방법이 다시
            # 받는 것**이고, 16년치를 다시 받는 것은 며칠과 하루 한도를 쓰는 일이다.
            #
            # 원문을 남겨 두면 네트워크를 한 번도 안 타고 다시 정규화할 수 있다.
            #
            # 그리고 이 표에는 두 번째 쓸모가 있다 — `fetched_at` 이 **"우리가 이 사실을
            # 언제부터 알 수 있었나"의 근거**다. 그 시각을 정규화 표에만 적어 두면
            # 나중에 고쳐 적었는지 아닌지를 증명할 수 없다.
            #
            # ⚠️ 응답을 **바이트 그대로** 담는다. 문자열로 바꿔 담으면 그 순간 인코딩
            #    추측이 끼어들고(euc-kr 로 오는 곳이 실재한다), 잘못 디코딩한 원문은
            #    원문이 아니다.
            """
            CREATE TABLE IF NOT EXISTS raw_response (
              source       TEXT    NOT NULL,
              target       TEXT    NOT NULL,
              fetched_at   TEXT    NOT NULL,
              body         BLOB    NOT NULL,
              sha256       TEXT    NOT NULL,
              bytes        INTEGER NOT NULL,
              compression  TEXT    NOT NULL DEFAULT 'gzip',
              encoding     TEXT,
              note         TEXT,
              -- 같은 대상을 다시 받으면 **덮지 않고 한 줄 더 쌓는다.** 출처가 값을
              -- 정정하는 일이 실제로 있고, 그때 무엇이 어떻게 바뀌었는지가 증거다.
              PRIMARY KEY (source, target, fetched_at)
            )
            """,
            # 재정규화가 한 출처를 통째로 순회할 때 쓴다.
            "CREATE INDEX IF NOT EXISTS idx_raw_source ON raw_response(source, target)",
        ),
    ),
    (
        "v4: robots.txt 캐시",
        (
            # ── robots.txt ──────────────────────────────────────────────
            # 크롤링은 **요청 직전에** 허용 여부를 확인한다. 그렇다고 매 요청마다
            # `robots.txt` 를 받으면 그 자체가 상대 서버를 두드리는 일이 된다.
            # 그래서 캐시하되 하루가 지나면 다시 받는다.
            #
            # ⚠️ **표에 담는 것은 원문이 아니라 판정의 재료다.** 상태 코드를 함께
            #    남기는 이유는, 받지 못했다는 사실 자체가 판정에 쓰이기 때문이다 —
            #    5xx 는 "전면 차단"이고 4xx 는 "전면 허용"이라 정반대다.
            #    `status=0` 은 네트워크에 닿지도 못한 경우로, 5xx 와 같이 다룬다.
            """
            CREATE TABLE IF NOT EXISTS robots_cache (
              origin      TEXT    NOT NULL,
              status      INTEGER NOT NULL,
              body        TEXT,
              encoding    TEXT,
              fetched_at  TEXT    NOT NULL,
              PRIMARY KEY (origin)
            )
            """,
        ),
    ),
    (
        "v5: 반입 — 남의 자료를 들인 기록",
        (
            # ── 반입 묶음 ───────────────────────────────────────────────
            # 파일 하나를 검사한 것이 한 묶음이다. **행보다 파일이 먼저 있는 이유**는
            # 판정이 파일 단위로도 나기 때문이다 — 뉴스 본문 칸이 있으면 행을 보지도 않고
            # 통째로 돌려보낸다. 그 사실을 적을 자리가 행 표에는 없다.
            #
            # `src_sha256` 이 열쇠 노릇을 한다. `scripts/check_inbox.py` 가 세션마다 도는데,
            # 이미 들인 파일을 다시 넣지 않으려면 **내용으로** 같은지 봐야 한다 — 파일 이름은
            # 팀원이 바꿔 올리고, 수정 시각은 내려받을 때마다 달라진다.
            """
            CREATE TABLE IF NOT EXISTS inbox_batch (
              batch_id         TEXT    NOT NULL,
              kind             TEXT    NOT NULL,
              src_path         TEXT    NOT NULL,
              src_sha256       TEXT    NOT NULL,
              src_bytes        INTEGER NOT NULL,
              origin           TEXT    NOT NULL DEFAULT 'local',
              contributor      TEXT,
              schema_version   TEXT,
              rows_total       INTEGER NOT NULL DEFAULT 0,
              rows_accepted    INTEGER NOT NULL DEFAULT 0,
              rows_quarantined INTEGER NOT NULL DEFAULT 0,
              rejected         TEXT,
              report_path      TEXT,
              started_at       TEXT    NOT NULL,
              finished_at      TEXT    NOT NULL,
              PRIMARY KEY (batch_id),
              -- 새 표라 검사할 기존 행이 없다 → CHECK 가 공짜다 (v1 의 collect_log 와 같은 이유).
              CHECK (origin IN ('local', 'huggingface'))
            )
            """,
            # 같은 파일을 두 번 들이지 않기 위한 조회. UNIQUE 로 걸지 **않는** 이유는
            # 규격이 고쳐진 뒤 같은 파일을 일부러 다시 검사하는 일이 정당하기 때문이다 —
            # 그때 판정이 어떻게 달라졌는지가 곧 규격 개정의 근거가 된다.
            "CREATE INDEX IF NOT EXISTS idx_inbox_batch_sha ON inbox_batch(src_sha256)",
            "CREATE INDEX IF NOT EXISTS idx_inbox_batch_kind ON inbox_batch(kind, finished_at)",

            # ── 합격한 행 ───────────────────────────────────────────────
            # 🔴 **규격 5장의 칸이 서로 겹치지 않아 한 표에 펼 수 없다.** 종목은 15칸,
            #    재무는 20칸이고 이름이 같은 칸도 뜻이 다르다(`name` 은 종목명이지만 지수
            #    파일에서는 지수명일 수 있다). 칸을 다 펴면 71칸짜리 표에 대부분이 NULL 이 되고,
            #    종류를 하나 더 받을 때마다 마이그레이션이 붙는다.
            #
            #    그래서 **메타는 칸으로, 행 자체는 JSON 으로** 담는다. Airbyte 의 raw table
            #    (`_airbyte_raw_id` · `_airbyte_data` JSON · `_airbyte_meta` JSON)이 같은 모양이고,
            #    거기서도 `_airbyte_meta.changes` 가 "어느 칸을 왜 고쳤나" 를 배열로 담는다.
            #    조회는 `json_extract(payload, '$.code')` 로 한다.
            #
            # `extras` 를 따로 두는 이유: 규격 밖 칸을 **버리지 않는다.** 팀원이 애써 붙여 온
            # 것이고, 나중에 쓸모가 생겼을 때 원본을 다시 받는 것보다 싸다.
            """
            CREATE TABLE IF NOT EXISTS inbox_accepted (
              batch_id   TEXT    NOT NULL,
              row_no     INTEGER NOT NULL,
              kind       TEXT    NOT NULL,
              key_hash   TEXT,
              payload    TEXT    NOT NULL,
              extras     TEXT,
              changes    TEXT,
              warnings   TEXT,
              loaded_at  TEXT    NOT NULL,
              PRIMARY KEY (batch_id, row_no)
            )
            """,
            # 같은 열쇠가 두 번 들어왔는지 보는 조회 — 반입은 겹칠 수밖에 없다(팀원 둘이
            # 같은 구간을 받아 올 수 있다). 겹침을 막지 않고 **보이게** 둔다.
            "CREATE INDEX IF NOT EXISTS idx_inbox_accepted_key ON inbox_accepted(kind, key_hash)",

            # ── 격리된 행 ───────────────────────────────────────────────
            # ⚠️ 합격 표와 달리 `raw` 를 함께 담는다. 격리는 **되돌릴 수 있어야** 한다 —
            #    사람이 값을 고쳐 다시 넣으려면 우리가 정제하기 전 원본이 필요하고,
            #    정제 뒤 값만 남기면 무엇을 고쳐야 하는지 알 수 없다.
            """
            CREATE TABLE IF NOT EXISTS inbox_quarantine (
              batch_id   TEXT    NOT NULL,
              row_no     INTEGER NOT NULL,
              kind       TEXT    NOT NULL,
              payload    TEXT    NOT NULL,
              raw        TEXT    NOT NULL,
              extras     TEXT,
              changes    TEXT,
              violations TEXT    NOT NULL,
              loaded_at  TEXT    NOT NULL,
              PRIMARY KEY (batch_id, row_no)
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_inbox_quarantine_kind "
            "ON inbox_quarantine(kind, batch_id)",
        ),
    ),
    (
        "v6: 공시 시점정합 — 결산기가 아니라 알게 된 날로 세운다",
        (
            # ── 재무제표 ────────────────────────────────────────────────
            # 칸 이름과 기본키는 `ingest/inbox/schemas/financial.json` 을 **그대로**
            # 따른다. 그 규격이 스스로 이렇게 적어 두었다.
            #
            #     "compareWith": null,
            #     "대조할 표가 아직 없다. 재무 적재 표는 이 규격이 선 뒤에 만든다."
            #
            # 여기가 그 표다. 규격과 표가 갈라지면 우리가 만든 파일이 우리 정문에서
            # 격리되는 일이 다시 생긴다 — 이슈 #43 이 정확히 그것이었다.
            #
            # 🔴 시점 기준은 `rcept_dt`(공시 접수일) 하나뿐이다. `bsns_year` 는 결산기이지
            #    세상이 알게 된 날이 아니다. 2020년 4분기 실적은 2021년 3월에 나오므로,
            #    결산기에 값을 붙이면 **석 달치 미래**를 학습에 넣고도 예외는 나지 않고
            #    성능만 좋아진다. 그래서 규격이 `has_time_anchor` 를 error 로 걸어 두었다.
            #
            # ⚠️ `ord` 와 `account_detail` 을 기본키에 넣으면서 NOT NULL 로 못박는다.
            #    SQLite 는 rowid 표의 PRIMARY KEY 에 NULL 을 허용해서(역사적 호환),
            #    빠뜨리면 같은 계정이 여러 줄 쌓이는데 UNIQUE 위반도 안 난다.
            #
            # 🔴 **`account_detail` 이 기본키에 있어야 하는 이유 — 실측으로 찾았다.**
            #
            #    규격(`financial.json`)의 primaryKey 를 그대로 쓰면 자본변동표(SCE)에서
            #    행이 **조용히 사라진다.** 삼성전자 2023 연결 176줄을 넣었더니 135줄만
            #    남았다 (41줄 손실). 예외도 경고도 없다 — `INSERT OR REPLACE` 가 덮어썼다.
            #
            #    SCE 는 "자본금·주식발행초과금·이익잉여금·비지배지분…" 열마다 한 줄씩
            #    주는데, 그 열을 가리키는 칸이 `account_detail` 뿐이다. 계정명도 ord 도
            #    account_id 도 전부 같다.
            #
            #        '기초자본' ord=4 가 8줄이고 값이 전부 다르다
            #          연결재무제표 [member]        354,749,604 백만
            #          지배기업 소유주 지분          345,186,142 백만
            #          비지배지분                     9,563,462 백만
            #          이익잉여금                   337,946,407 백만
            #          자본금                           897,514 백만
            #
            #    무엇을 더해야 갈라지는지도 실측했다 — `account_id`·`thstrm_nm` 은
            #    소용이 없었고(135줄 그대로) `account_detail` 만 176줄을 지켰다.
            #
            #    ⚠️ 빈 값을 NULL 로 두면 안 된다. SQLite 는 PK 안의 NULL 을 서로 다른
            #       값으로 보므로 같은 계정이 여러 줄 쌓인다. 그래서 `NOT NULL DEFAULT ''`
            #       로 두고 저장하는 쪽도 None 이 아니라 빈 문자열을 넣는다.
            """
            CREATE TABLE IF NOT EXISTS dart_financial (
              corp_code        TEXT    NOT NULL,
              stock_code       TEXT,
              corp_name        TEXT,
              bsns_year        INTEGER NOT NULL,
              reprt_code       TEXT    NOT NULL,
              fs_div           TEXT    NOT NULL,
              sj_div           TEXT    NOT NULL,
              account_id       TEXT,
              account_nm       TEXT    NOT NULL,
              account_detail   TEXT    NOT NULL DEFAULT '',
              ord              INTEGER NOT NULL DEFAULT 0,
              currency         TEXT,
              thstrm_nm        TEXT,
              thstrm_amount    REAL,
              frmtrm_amount    REAL,
              bfefrmtrm_amount REAL,
              rcept_no         TEXT,
              rcept_dt         TEXT,
              report_nm        TEXT,
              rm               TEXT,
              collected_at     TEXT    NOT NULL,
              PRIMARY KEY (corp_code, bsns_year, reprt_code, fs_div, sj_div,
                           account_nm, ord, account_detail),
              -- 새 표라 검사할 기존 행이 없다 → CHECK 가 공짜다 (v1 과 같은 이유).
              -- 값 목록은 규격의 enum 을 그대로 옮긴 것이다.
              CHECK (reprt_code IN ('11011', '11012', '11013', '11014')),
              CHECK (fs_div IN ('CFS', 'OFS')),
              CHECK (sj_div IN ('BS', 'IS', 'CIS', 'CF', 'SCE'))
            )
            """,
            # as_of 조회가 "이 날짜에 알 수 있었던 재무" 를 고를 때 쓴다.
            # 접수일이 앞에 오는 이유는 그것이 **거르는 칸**이기 때문이다 — 회사를 먼저
            # 좁히는 것이 아니라 시점을 먼저 자른다.
            "CREATE INDEX IF NOT EXISTS idx_dart_fin_asof "
            "ON dart_financial(rcept_dt, corp_code)",
            # 한 회사의 연도별 추이를 뽑을 때.
            "CREATE INDEX IF NOT EXISTS idx_dart_fin_corp_year "
            "ON dart_financial(corp_code, bsns_year, reprt_code)",
            # 종목코드로 시세와 잇는 경로. 비상장 법인은 이 칸이 비어 있다.
            "CREATE INDEX IF NOT EXISTS idx_dart_fin_stock "
            "ON dart_financial(stock_code, bsns_year)",

            # ── 공시 목록 ───────────────────────────────────────────────
            # `list.json` 의 응답을 원본 이름 그대로 담는다.
            #
            # **왜 따로 두는가 — 콜을 아끼기 위해서다.** 재무제표 응답
            # (`fnlttSinglAcntAll`)에는 접수번호는 있어도 **접수일이 없다.** 그래서
            # 접수일을 되찾으려면 회사·보고서마다 `list.json` 을 한 번 더 불러야 한다
            # (350종 × 5개년이면 1,750콜). 그런데 그 응답에는 그 구간의 공시가 통째로
            # 딸려 온다 — 접수일 한 칸만 뽑고 버리면 같은 자료를 다음에 또 사야 한다.
            #
            # 🔴 접수번호 앞 8자리를 잘라 쓰지 않는다. 정기공시 4,800건 실측에서
            #    3건(0.062%)이 어긋났고 그중 둘은 접수일을 **3일 앞당겨** 읽는다.
            #    틀리는 방향이 전부 우리에게 유리한 쪽이면 잡음이 아니라 **편향**이고,
            #    예외는 나지 않는다.
            """
            CREATE TABLE IF NOT EXISTS dart_disclosure (
              rcept_no     TEXT NOT NULL,
              corp_code    TEXT NOT NULL,
              corp_name    TEXT,
              stock_code   TEXT,
              corp_cls     TEXT,
              report_nm    TEXT,
              flr_nm       TEXT,
              rcept_dt     TEXT NOT NULL,
              rm           TEXT,
              collected_at TEXT NOT NULL,
              PRIMARY KEY (rcept_no)
            )
            """,
            # 접수번호로 접수일을 되찾는 조회 (재무 수집기가 매번 쓴다).
            "CREATE INDEX IF NOT EXISTS idx_dart_disc_corp "
            "ON dart_disclosure(corp_code, rcept_dt)",
            # "이 날 무엇이 공시됐나" — 시점정합 검증과 뉴스 대조에 쓴다.
            "CREATE INDEX IF NOT EXISTS idx_dart_disc_date "
            "ON dart_disclosure(rcept_dt)",
        ),
    ),
    (
        "v7: 거시 통계 — 기준월이 아니라 공표된 날로 세운다",
        (
            # ── 거시 시계열 ─────────────────────────────────────────────
            # 한국은행 ECOS 에서 받는 국내 거시지표 9종을 **긴 형식(long)** 으로 담는다.
            # 지표마다 칸을 만들지 않는 이유는 주기가 섞여 있기 때문이다 — 일별 4종과
            # 월별 5종을 한 표에 넓은 형식으로 두면 월별 칸이 대부분 빈다.
            #
            # 🔴 **`known_at` 이 이 표의 존재 이유다.**
            #
            # ECOS 응답에는 **발표일이 없다.** 실측으로 확인했다 — `StatisticSearch` 의
            # 14칸(STAT_CODE·TIME·DATA_VALUE 등), `StatisticTableList` 의 6칸,
            # `StatisticMeta` 어디에도 "이 값이 언제 공개됐는지" 가 없다.
            #
            # 대신 월별 지표를 **기준월 1일**로 준다(2026년 7월 물가 → `2026-07-01`).
            # 그 날짜를 그대로 붙이면 **7월 물가를 7월 1일에 아는 셈**이 된다. 실제로는
            # 8월 4일에 발표됐다 — 34일치 미래다. 경기지수는 더 심해서 7월분이 8월 31일에
            # 나오므로 61일이 샌다. 그리고 이 오류는 **예외를 내지 않고 성능만 올린다.**
            #
            # 그래서 값과 별개로 "언제부터 알 수 있었나" 를 계산해 함께 담는다.
            # 규칙은 `ingest/clients/ecos_data.py` 의 `RELEASE_RULES` 에 있고,
            # 실제 공표일정에 안전 여유를 더한 값이다. 규칙이 바뀌면 이 표를 다시 채운다.
            #
            # `period` 는 ECOS 원문 그대로 둔다(일별 `20260901` · 월별 `202607`).
            # 주기마다 형식이 다른 것을 여기서 통일하지 않는 이유는, 통일하면 어느 것이
            # 원본이었는지 되찾을 수 없어 재수집 때 대조가 불가능해지기 때문이다.
            """
            CREATE TABLE IF NOT EXISTS macro_series (
              indicator_id TEXT NOT NULL,
              period       TEXT NOT NULL,
              cycle        TEXT NOT NULL,
              value        REAL,
              known_at     TEXT NOT NULL,
              stat_code    TEXT NOT NULL,
              item_code    TEXT,
              unit         TEXT,
              collected_at TEXT NOT NULL,
              -- 한 지표는 하나의 (통계표, 항목) 조합에 고정돼 있으므로 기간마다 값이
              -- 하나뿐이다. 그래서 두 칸으로 충분하다.
              -- ⚠️ 재무에서 `account_detail` 을 PK 에 빠뜨려 6.4%가 조용히 사라진 적이
              --    있다. 여기서도 "행 수가 맞으니 됐다" 로 넘기지 않고, 수집 뒤에
              --    지표×기간 조합이 실제로 유일한지 세어서 확인한다.
              PRIMARY KEY (indicator_id, period),
              -- 새 표라 검사할 기존 행이 없다 → CHECK 가 공짜다 (v1 과 같은 이유).
              CHECK (cycle IN ('D', 'M', 'Q', 'A')),
              -- 알게 된 날은 반드시 YYYYMMDD 여야 한다. 형식이 섞이면 시점 비교가
              -- 문자열 비교로 조용히 어긋난다.
              CHECK (length(known_at) = 8),
              -- 값이 없을 수는 있어도(ECOS 가 '-' 를 준다) 언제 알았는지는 늘 있어야 한다.
              CHECK (known_at <> '')
            )
            """,
            # as_of 조회가 "이 날짜에 알 수 있었던 거시" 를 고를 때 쓴다.
            # 시점을 앞에 두는 이유는 v6 과 같다 — 지표를 좁히기 전에 시점을 자른다.
            "CREATE INDEX IF NOT EXISTS idx_macro_asof "
            "ON macro_series(known_at, indicator_id)",
            # 한 지표의 시계열을 기간 순으로 훑을 때. PK 가 (지표, 기간) 이라 앞은
            # 겹치지만, 이쪽은 값까지 담아 표를 다시 읽지 않게 한다(커버링 인덱스).
            "CREATE INDEX IF NOT EXISTS idx_macro_series_value "
            "ON macro_series(indicator_id, period, value)",
        ),
    ),
    (
        "v8: 수집 실행 기록 — 지금 돌고 있는지 밖에서 볼 수 있게",
        (
            # ── 실행 ────────────────────────────────────────────────────
            # `python -m pipelines.ingest` 한 번이 한 줄이다.
            #
            # **왜 `collect_log` 로 부족한가.** 그 표는 *"무엇을 어디까지 받았나"* 를
            # 대상별로 담는다(종목·날짜·지표). 반면 여기는 *"언제 돌렸고 지금 어디쯤인가"* 다.
            # 대장만 보면 **지금 돌고 있는 중인지 죽은 것인지 구별할 수 없다** — 둘 다
            # "마지막 성공이 좀 됐다" 로 보인다. 대시보드가 알아야 하는 것이 그 구별이다.
            #
            # `args` 에 실행 인자를 그대로 남긴다. 나중에 "이 숫자가 어떤 조건에서
            # 나왔나" 를 답하려면 명령줄이 남아 있어야 한다.
            """
            CREATE TABLE IF NOT EXISTS ingest_run (
              run_id      TEXT NOT NULL,
              started_at  TEXT NOT NULL,
              finished_at TEXT,
              status      TEXT NOT NULL,
              args        TEXT,
              note        TEXT,
              PRIMARY KEY (run_id),
              -- 새 표라 검사할 기존 행이 없다 → CHECK 가 공짜다 (v1 과 같은 이유).
              --   running  돌고 있다. `finished_at` 이 비어 있다
              --   ok       전 단계가 끝났다
              --   partial  일부 단계가 실패했지만 나머지는 끝났다
              --   error    시작하자마자 못 돌았다
              --   dry_run  무엇을 할지만 보고 실제로 받지는 않았다
              CHECK (status IN ('running', 'ok', 'partial', 'error', 'dry_run'))
            )
            """,
            # 대시보드가 "가장 최근 실행" 을 뽑을 때. 시작 시각 역순이 곧 최신순이다.
            "CREATE INDEX IF NOT EXISTS idx_ingest_run_started "
            "ON ingest_run(started_at DESC)",

            # ── 단계 ────────────────────────────────────────────────────
            # 실행 하나 안의 price · financial · macro 각각.
            #
            # 실행 표와 나눈 이유는 **폴링 때문**이다. 한 표에 JSON 으로 뭉쳐 두면
            # 대시보드가 단계 하나의 진행을 보려고 매번 문자열을 파싱해야 하고,
            # 파싱은 언젠가 실패한다. 칸으로 두면 SQL 이 직접 고른다.
            """
            CREATE TABLE IF NOT EXISTS ingest_run_stage (
              run_id      TEXT    NOT NULL,
              stage       TEXT    NOT NULL,
              status      TEXT    NOT NULL,
              rows        INTEGER NOT NULL DEFAULT 0,
              started_at  TEXT    NOT NULL,
              finished_at TEXT,
              note        TEXT,
              PRIMARY KEY (run_id, stage),
              --   running  이 단계가 돌고 있다
              --   ok       끝났다
              --   error    실패했다. `note` 에 무엇이 실패했는지 남긴다
              --   skipped  `--only` 로 건너뛴 단계다. **실패가 아니다**
              --   dry_run  무엇을 할지만 셌다
              CHECK (status IN ('running', 'ok', 'error', 'skipped', 'dry_run'))
            )
            """,
            # 한 실행의 단계들을 순서대로 뽑을 때 (대시보드가 가장 자주 하는 조회).
            "CREATE INDEX IF NOT EXISTS idx_ingest_stage_run "
            "ON ingest_run_stage(run_id, started_at)",
        ),
    ),
    (
        "v9: 수정주가 4칸 · 실측 거래일 달력",
        (
            # ── 수정 OHLC ───────────────────────────────────────────────
            # `close` 는 **액면분할이 조정되지 않은 원가격**이다. 분할일에 가격이 그대로
            # 뚝 떨어지므로 수익률로 계산하면 삼성전자 2018-05-04 가 **-98.04%** 로 읽힌다
            # (실제 그날 등락은 -2.08%). 1,139건의 분할·병합이 806종(21.9%)에 걸쳐 있다.
            #
            # **원 가격 칸을 덮지 않고 옆에 4칸을 새로 둔다.** 세 가지 이유가 있다.
            #
            #   ① `market_cap = close × listed_shares` 는 **원가격**이어야 맞다. 덮으면 깨진다.
            #   ② 후방조정 값은 **다음 분할 때 과거 전체가 다시 바뀐다.** append-only 워터마크
            #      반입과 정면으로 충돌하므로, 원본을 덮으면 어제 결과를 재현할 수 없다.
            #   ③ Kronos 같은 OHLCV 6채널 모델은 종가만이 아니라 **수정 OHLC 전부**가 필요하다.
            #
            # ⚠️ **INTEGER 가 아니라 REAL 이다.** 후방조정 값은 정수가 아니다 — 삼성전자
            #    2010-01-04 은 원종가 809,000 이 아니라 16,180.00 이 되고, 분할이 잦았던
            #    종목은 1원 아래로 내려간다. INTEGER 로 두면 조용히 잘린다.
            _add_column("daily_price", "adj_open", "REAL"),
            _add_column("daily_price", "adj_high", "REAL"),
            _add_column("daily_price", "adj_low", "REAL"),
            _add_column("daily_price", "adj_close", "REAL"),
            # 이 행의 수정값이 **어디서 왔나**. 값을 믿을 범위가 둘이 다르다.
            #
            #   fdr    FinanceDataReader(네이버 fchart) 가 직접 준 값. 외부 실측이다.
            #   chain  우리가 조정계수를 곱해 **뒤로 이어 붙인** 값 (`common.corporate_actions`).
            #
            # 왜 두 가지인가: 네이버는 **최근 3,000거래일만** 준다(2014-06-13~). `count` 를
            # 6000·9000 으로 올려도 서버가 3,000 에서 자른다. 우리 달력은 4,102일이라
            # 2010-01-04~2014-06-12 의 1,103일(2,146,042행·23.3%)이 남고, 홀드아웃이
            # 20240901 이므로 그 구멍은 **전부 학습구간 안**이다. 그래서 겹치는 지점을
            # 앵커로 삼아 그 앞을 자체 계산으로 잇는다.
            #
            # 날짜로 유추할 수 없어서 칸으로 둔다 — 2012년에 상장폐지된 종목은 FDR 이
            # 아예 없어 전 구간이 `chain` 이고, 2020년 상장 종목은 전 구간이 `fdr` 다.
            _add_column("daily_price", "adj_source", "TEXT"),

            # ── 실측 거래일 달력 ────────────────────────────────────────
            # **거래일을 계산으로 맞히지 않는다.** 주말만 걸러 세면 개발구간 평일 3,042일
            # 중 162일(5.3%)이 어긋난다 — 그 162일은 명절·공휴일이고 하필 실적 발표와
            # 뉴스가 몰리는 연휴 전후다. 우리가 실제로 받은 날을 그대로 쓴다.
            #
            # 로직은 `common/trading_calendar.py` 에 이미 있었다. 표로 옮기는 이유는
            # **`SELECT DISTINCT bas_dd FROM daily_price` 가 9.2M 행을 훑어 660ms 걸리기
            # 때문이다.** 프로세스마다 한 번씩 무는 값이고, 반입 검사는 행마다 달력을
            # 부른다. 4,102행짜리 표로 옮기면 같은 답이 ~1ms 에 나온다.
            #
            # ⚠️ 이 표는 `daily_price` 에서 **파생된 것**이라 원본이 늘면 낡는다.
            #    그래서 `rebuild_calendar()` 를 시세 적재 뒤에 함께 부르고,
            #    `common.trading_calendar` 는 표가 비었거나 없으면 **원본으로 폴백**한다.
            #    표가 낡아서 조용히 틀리느니 느린 편이 낫다.
            """
            CREATE TABLE IF NOT EXISTS trading_calendar (
              bas_dd      TEXT    NOT NULL,
              market      TEXT    NOT NULL,
              stock_count INTEGER NOT NULL DEFAULT 0,
              built_at    TEXT    NOT NULL,
              -- 시장별로 한 줄. `ALL` 은 "어느 시장이든 열렸다" 를 뜻하는 합집합이다.
              -- 시장을 나눠 두는 이유: 한쪽만 열리는 날이 실제로 있고(코스닥 단독 개장),
              -- 합집합만 있으면 그 날을 코스피 거래일로 잘못 읽는다.
              PRIMARY KEY (bas_dd, market),
              -- 새 표라 검사할 기존 행이 없다 → CHECK 가 공짜다 (v1 과 같은 이유).
              CHECK (length(bas_dd) = 8),
              CHECK (market IN ('ALL', 'KOSPI', 'KOSDAQ')),
              -- 거래일인데 종목이 0종일 수는 없다. 0 이면 그건 휴장이고, 휴장일은
              -- 이 표에 **행이 없어야** 한다 — 0행과 미수집을 섞지 않기 위해서다.
              CHECK (stock_count > 0)
            )
            """,
            # 한 시장의 거래일을 날짜 순으로 훑을 때. PK 는 (날짜, 시장) 이라 시장으로
            # 먼저 좁히지 못한다.
            "CREATE INDEX IF NOT EXISTS idx_calendar_market_date "
            "ON trading_calendar(market, bas_dd)",
        ),
    ),
    (
        "v10: 종목 신원 · 법인 개요 (공공데이터포털 금융위)",
        (
            # ── 종목 신원 ───────────────────────────────────────────────
            # 우리는 종목을 **KRX 종목코드**로만 안다. 그 코드로는 DART(고유번호)·
            # 공공데이터포털(법인등록번호)·해외 자료(ISIN) 어디에도 못 붙는다.
            # 이 표가 그 **다리**다. 시세를 담으려는 게 아니다 — 시세는 KRX 가 정본이고
            # 이쪽은 20칸뿐이라 가져올 이유가 없다.
            #
            # 🔴 **기본키가 (bas_dd, code) 인 이유.** 종목명은 바뀌고(쓰리원 → UCI),
            #    코드는 재사용되기도 한다. 날짜를 키에서 빼면 "지금 이 코드가 누구인가"
            #    만 남고 **"그때는 누구였나"** 를 잃는다. 그러면 과거 시점 조인이 조용히
            #    현재 이름을 붙인다 — 행 수는 그대로라 어떤 검사에도 안 걸린다.
            #    재무 수집에서 `account_detail` 을 키에서 빼는 바람에 자본변동표의
            #    6.4%가 덮어써져 사라진 적이 있다. 같은 종류의 사고다.
            """
            CREATE TABLE IF NOT EXISTS stock_identity (
              bas_dd    TEXT NOT NULL,          -- 기준일 YYYYMMDD
              -- 🔴 응답의 `srtnCd` 는 'A000020' 처럼 **A 접두사**가 붙어 온다. 여기에는
              --    뗀 값을 넣는다. 안 떼면 `daily_price.code` 와 조인이 **0행**이 되는데,
              --    조인은 0행이어도 에러가 나지 않는다 — 그냥 아무것도 안 나온다.
              code      TEXT NOT NULL,
              isin_cd   TEXT,                   -- 국제 표준 식별자 (KR7000020008)
              crno      TEXT,                   -- 법인등록번호 — 기업기본정보 조회 키
              corp_nm   TEXT,                   -- 법인명. 종목명과 다를 수 있다
              item_nm   TEXT,                   -- 종목명
              market    TEXT,                   -- KOSPI / KOSDAQ / KONEX
              -- 이 사실을 **언제부터 알 수 있었나**. 포털은 발표 시각을 주지 않아
              -- `bas_dd` 의 다음 영업일로 계산한다. 계산값이라는 것을 잊으면 안 된다.
              known_at  TEXT NOT NULL,
              -- 어떤 규칙으로 `known_at` 을 냈는지. 규칙을 바꾸면 **재수집**해야 하는데,
              -- 무엇으로 계산했는지 안 남기면 어느 행이 옛 규칙인지 알 수 없다.
              known_rule TEXT NOT NULL,
              fetched_at TEXT NOT NULL,
              PRIMARY KEY (bas_dd, code),
              CHECK (length(bas_dd) = 8),
              -- 🔴 **코드 전체가 숫자라고 단정하지 않는다.** 5·6번째 자리에 영문이 오는
              --    종목이 84종 있다(`0001A0`·`00088K` 등 신형우선주 · 실측 2026-09-03,
              --    고유 3,677종 중). `code GLOB '[0-9]*'` 로 막으면 그 84종이 통째로
              --    격리되는데, 격리는 조용해서 한참 뒤에나 눈치챈다.
              --
              --    대신 **앞 4자리는 항상 숫자**다(3,677종 전수 확인). 이건 접두사를
              --    안 뗀 값을 잡아 준다 — `A000020` 을 6자로 자른 `A00002` 가 걸린다.
              --    접두사가 그대로면 7자라 아래 길이 검사에서 먼저 걸린다.
              CHECK (length(code) = 6),
              CHECK (code GLOB '[0-9][0-9][0-9][0-9]*')
            )
            """,
            # 코드로 "이 종목의 신원이 언제 어떻게 바뀌었나" 를 훑을 때. PK 가
            # (날짜, 코드) 라 코드로 먼저 좁히지 못한다.
            "CREATE INDEX IF NOT EXISTS idx_identity_code "
            "ON stock_identity(code, bas_dd)",
            # 법인등록번호로 `corp_profile` 에 붙일 때. NULL 인 행이 많아 부분 인덱스다.
            "CREATE INDEX IF NOT EXISTS idx_identity_crno "
            "ON stock_identity(crno) WHERE crno IS NOT NULL",
            # 시점 경계(`supply`)가 "그때 알 수 있었던 것" 만 고를 때.
            "CREATE INDEX IF NOT EXISTS idx_identity_known "
            "ON stock_identity(known_at)",

            # ── 법인 개요 ───────────────────────────────────────────────
            # 상장일·폐지일·감사의견·종업원수. 시세로는 알 수 없는 것들이다.
            #
            # **상장폐지일이 특히 값지다.** 지금 우리는 "어느 날부터 시세가 안 나온다"
            # 로 폐지를 추정하는데, 그건 장기 거래정지와 구별되지 않는다. 실측해 보니
            # 2020년 목록에 있다 2025년에 사라진 법인 8곳 **전부** `enpXchgLstgAbolDt`
            # 가 채워져 왔다 (메리츠화재 23/02/21 · DL건설 24/03/04 · 롯데푸드 22/07/20).
            #
            # 🔴 **기본키가 (crno, fst_opeg_dt) 인 이유 — 응답에 기준일이 없다.**
            #    설계 문서는 `(crno, bas_dd)` 로 잡았지만, 실제 응답에는 `basDt` 칸이
            #    아예 없다. 대신 `fstOpegDt`~`lastOpegDt` 가 **그 스냅샷이 유효한 구간**
            #    이고, 한 법인의 여러 행이 겹치지 않게 이어진다 (동화약품 17행 · 표본
            #    12개 법인 182행에서 겹침 0 · 빈값 0 · 실측 2026-09-03).
            #
            #    이게 우리에게 이득인 이유: `known_at` 이 **계산값이 아니라 관측값**이 된다.
            #    거시(ECOS)는 발표일을 안 줘서 "기준일의 다음 영업일" 로 계산할 수밖에
            #    없었고, 그래서 규칙을 바꾸면 재수집해야 하는 짐이 남았다. 여기서는
            #    출처가 직접 "이 값은 이 날부터 유효했다" 를 말해 준다.
            #
            #    부수 효과로 **날짜별 반복 호출이 필요 없다.** 한 번 부르면 그 법인의
            #    전 이력이 온다 — 법인 수만큼만 부르면 된다.
            """
            CREATE TABLE IF NOT EXISTS corp_profile (
              crno                TEXT NOT NULL,   -- 법인등록번호
              -- 이 스냅샷이 유효해진 날 (`fstOpegDt`). 기준일이 아니라 **유효 시작일**이다.
              fst_opeg_dt         TEXT NOT NULL,
              -- 유효 끝난 날 (`lastOpegDt`). 가장 최근 행은 계속 밀린다.
              last_opeg_dt        TEXT,
              corp_nm             TEXT,
              sic_nm              TEXT,            -- 표준산업분류 (대부분 비어 온다)
              estb_dt             TEXT,            -- 설립일 (YYYYMMDD 로 정규화한 값)
              stac_mm             TEXT,            -- 결산월 ('12' 같은 두 자리)
              xchg_lstg_dt        TEXT,            -- 유가증권시장 상장일 (정규화)
              xchg_lstg_abol_dt   TEXT,            -- 유가증권시장 상장폐지일 (정규화)
              kosdaq_lstg_dt      TEXT,
              kosdaq_lstg_abol_dt TEXT,
              audt_rpt_opnn       TEXT,            -- 감사의견 (`audtRptOpnnCtt`)
              actn_audpn          TEXT,            -- 회계감사인 (`actnAudpnNm`)
              empe_cnt            INTEGER,         -- 종업원수 (`enpEmpeCnt`)
              pn1_avg_slry_amt    INTEGER,         -- 1인평균급여 (`enpPn1AvgSlryAmt`)
              smenp_yn            TEXT,            -- 중소기업 여부 (대부분 비어 온다)
              -- 🔴 여기서는 **계산값이 아니다** — `fst_opeg_dt` 를 그대로 쓴다.
              --    출처가 "이 값은 이 날부터 유효했다" 를 직접 말해 주기 때문이다.
              known_at            TEXT NOT NULL,
              known_rule          TEXT NOT NULL,
              fetched_at          TEXT NOT NULL,
              PRIMARY KEY (crno, fst_opeg_dt),
              CHECK (length(fst_opeg_dt) = 8)
            )
            """,
            # 상장일·폐지일로 "이 구간에 살아 있던 법인" 을 고를 때.
            "CREATE INDEX IF NOT EXISTS idx_profile_lstg "
            "ON corp_profile(xchg_lstg_dt, xchg_lstg_abol_dt)",
            "CREATE INDEX IF NOT EXISTS idx_profile_known "
            "ON corp_profile(known_at)",
        ),
    ),
    (
        "v11: 종목기본정보 (KRX) — 우선주 판별의 정본",
        (
            # ── 종목기본정보 ───────────────────────────────────────────
            # 지금 우리는 **종목명이 '우' 로 끝나는지로 우선주를 추측**하고 있다.
            # 그게 보통주 7종을 우선주로 잘못 뺀다 (실측 2026-09-03 · 세 시장 × 세 날짜):
            #
            #   미래에셋대우 · 연우 · 동우 · 신우 · 성우 · 에코글로우 · 이오플로우
            #
            # 006800 은 20200102 코스피 시총 **48위**다. 모델 파트가 쓰기로 한
            # "KOSPI 보통주 시총 상위 50" 후보에서 조용히 빠진다.
            #
            # 🔴 이 오류는 **이름이 바뀌는 구간에만** 나타난다 — 대우증권(정상) →
            #    미래에셋대우(깨짐) → 미래에셋증권(정상). 오늘 유가 943종만 세면
            #    어긋남이 0건이라 **표본으로는 절대 안 잡힌다.** 전량으로만 보인다.
            #
            # 🔴 **기본키가 (bas_dd, code) 인 이유** — `stock_identity` 와 같다.
            #    이 응답은 오늘 스냅샷이 아니라 **그날의 사실**이다. 같은 엔드포인트를
            #    다른 날짜로 부르면 다른 답이 온다 (유가 20150102 899행 · 20200102
            #    916행 · 20260901 943행 / 2015 에만 있고 지금은 없는 종목 159종 /
            #    공통 740종 중 상장주식수가 다른 종목 534종 · 실측).
            #    날짜를 키에서 빼면 "지금 무엇인가" 만 남고 "그때는 무엇이었나" 를 잃는다.
            #
            # ⚠️ 그리고 이 성질 덕에 #92 에서 오준영님이 걱정한 문제가 여기서는
            #    안 생긴다 — *"2026년에 확인한 값을 2015년에도 같았다고 적용"* 하지
            #    않아도 된다. 2015년 값을 2015년에 직접 물어볼 수 있다.
            """
            CREATE TABLE IF NOT EXISTS stock_base_info (
              bas_dd    TEXT NOT NULL,          -- 기준일 YYYYMMDD (그날의 사실)
              -- 단축코드. `daily_price.code` 와 같은 축이다.
              -- 🔴 여기서도 코드 전체를 숫자로 단정하지 않는다 — 5·6번째 자리에 영문이
              --    오는 신형우선주가 있다(`0001A0`·`00088K`). 앞 4자리만 숫자다.
              code      TEXT NOT NULL,
              isin_cd   TEXT,                   -- 표준코드 KR7095570008
              isu_nm    TEXT,                   -- 정식명 (AJ네트웍스보통주)
              isu_abbrv TEXT,                   -- 한글약명 (AJ네트웍스)
              isu_eng_nm TEXT,                  -- 영문명
              -- 상장일. `corp_profile.xchg_lstg_dt` 와 **교차검증**할 수 있다.
              list_dd   TEXT,
              market    TEXT,                   -- KOSPI / KOSDAQ / KONEX
              secugrp_nm TEXT,                  -- 증권구분 (주권 · 외국주권 …)
              -- 🔴 소속부. **유가에서는 항상 빈 문자열이다** — 업종이 아니다(#92).
              --    받아 두지만 업종 매핑으로 쓸 수 없다.
              sect_tp_nm TEXT,
              -- ⭐ 이 표의 존재 이유. 보통주 · 구형우선주 · 신형우선주 · 종류주권.
              kind_stkcert_tp_nm TEXT,
              parval    TEXT,                   -- 액면가. '무액면' 이 와서 숫자가 아니다
              list_shrs INTEGER,                -- 상장주식수
              -- 이 사실을 **언제부터 알 수 있었나**. KRX 도 발표 시각을 주지 않아
              -- `bas_dd` 의 다음 거래일로 계산한다. `stock_identity` 와 같은 규칙이다.
              --
              -- ⚠️ 보수적인 선택이다. 주권종류·상장일은 상장 공고에 이미 실리므로
              --    당일에도 알 수 있었다고 볼 여지가 있다. 그래도 다음 거래일로 미루는
              --    쪽을 골랐다 — 늦게 아는 것은 성능을 낮출 뿐이고, 일찍 안 것으로
              --    잘못 적으면 미래참조가 되는데 그건 **에러 없이 성능만 좋아진다.**
              known_at  TEXT NOT NULL,
              -- 계산값이라 규칙을 바꾸면 재수집해야 한다. 어느 행이 옛 규칙인지 남긴다.
              known_rule TEXT NOT NULL,
              fetched_at TEXT NOT NULL,
              PRIMARY KEY (bas_dd, code),
              CHECK (length(bas_dd) = 8),
              CHECK (substr(code, 1, 4) GLOB '[0-9][0-9][0-9][0-9]')
            )
            """,
            # 종목 하나의 이력을 훑을 때 (언제 상장했고 언제부터 안 보이나).
            "CREATE INDEX IF NOT EXISTS idx_base_info_code "
            "ON stock_base_info(code, bas_dd)",
            # 🔴 유니버스를 만들 때 가장 많이 쓰는 축 — "그날 그 시장의 보통주".
            "CREATE INDEX IF NOT EXISTS idx_base_info_kind "
            "ON stock_base_info(bas_dd, market, kind_stkcert_tp_nm)",
            # as_of 로 자를 때.
            "CREATE INDEX IF NOT EXISTS idx_base_info_known "
            "ON stock_base_info(known_at)",
        ),
    ),
    (
        "v12: 텍스트 신호 — 제목이 아니라 **고유 제목**에 값을 매긴다",
        (
            # ── 공시 제목 감성 ──────────────────────────────────────────
            # 공시 제목에 감성 모델을 돌려 확률 3칸(`p_neg`·`p_neu`·`p_pos`)을 낸다.
            #
            # 🔴 **행마다 매기지 않고 고유 제목마다 매긴다.** 공시 제목은 정형 문구라
            #    155만 행의 고유 제목이 **18,600개(1.2%)** 뿐이다(실측 2026-09-04).
            #    행마다 추론하면 같은 문장을 83번씩 다시 읽는 셈이고, 시간이 83배 든다.
            #
            # **왜 `dart_disclosure` 에 칸을 붙이지 않았나.**
            #   ① 모델을 바꾸면 155만 행을 다시 써야 한다. 이 표는 18,600행만 다시 쓴다
            #   ② 모델 둘을 나란히 두고 비교할 수가 없다 — 칸이 하나뿐이라 덮어쓴다
            #   ③ **어느 모델·어느 리비전이 낸 값인지**를 행에 남길 자리가 없다.
            #      HF 모델은 주인이 가중치를 갈아 끼울 수 있어 리비전이 곧 재현성이다
            #
            # 붙일 때는 `report_nm` 으로 조인한다. 제목이 기본키가 아니라 `text_sha`
            # 를 쓰는 이유는 제목이 최대 172자라 인덱스가 두꺼워지기 때문이다.
            """
            CREATE TABLE IF NOT EXISTS text_signal (
              -- 제목의 SHA-256 앞 16자. 제목 자체를 키로 쓰면 인덱스가 두껍다.
              text_sha   TEXT NOT NULL,
              -- 원문을 함께 둔다. 해시만 있으면 사람이 확인할 수 없다.
              report_nm  TEXT NOT NULL,
              -- 🔴 어느 모델이 낸 값인가. 모델을 바꿔도 옛 값이 남아 비교가 된다.
              model_id   TEXT NOT NULL,
              -- HF 리비전(커밋 해시). 주인이 가중치를 갈아 끼울 수 있어 이게 재현성이다.
              revision   TEXT,
              p_neg      REAL NOT NULL,
              p_neu      REAL NOT NULL,
              p_pos      REAL NOT NULL,
              scored_at  TEXT NOT NULL,
              PRIMARY KEY (text_sha, model_id),
              -- 확률이므로 [0,1] 이고 합이 1 이어야 한다. 어긋나면 softmax 를 빠뜨린 것이다.
              CHECK (p_neg >= 0 AND p_neg <= 1),
              CHECK (p_neu >= 0 AND p_neu <= 1),
              CHECK (p_pos >= 0 AND p_pos <= 1),
              CHECK (ABS(p_neg + p_neu + p_pos - 1.0) < 0.01)
            )
            """,
            # 공시 목록에 붙일 때 쓰는 축 — `report_nm` 으로 조인한다.
            "CREATE INDEX IF NOT EXISTS idx_text_signal_nm "
            "ON text_signal(report_nm, model_id)",
        ),
    ),
    (
        "v13: 지수 기본키에 시장을 넣는다 — 같은 이름의 업종지수가 둘이다",
        (
            # ── index_price 재구성 ──────────────────────────────────────
            # 🔴 이 마이그레이션은 **자료를 잃고 나서** 만들었다. 자세한 것은
            #    `_rebuild_index_price` 의 설명에 있다.
            #
            # 문장이 아니라 **함수**인 이유: 표 재구성은 문장이 다섯인데 판단("옛
            # 기본키인가")은 하나다. 문장마다 다시 재면 첫 문장이 표를 바꾼 뒤 조건이
            # 뒤집혀 나머지가 건너뛰어진다. 한 번 보고 목록을 통째로 준다.
            _rebuild_index_price,
        ),
    ),
    (
        "v14: 배당 — 기준일은 달력이고, 값이 움직이는 날은 배당락일이다",
        (
            # ── 배당 ────────────────────────────────────────────────────
            # 공공데이터포털 금융위 **주식배당정보**(`GetStocDiviInfoService_V2`)를 담는다.
            # 전량 71,681행 · 72콜(2026-09-09 실측). 우리 개발구간(2010~)에 드는 것은
            # 현금·동시배당 26,008행이고 그중 우리 종목에 붙는 것이 21,846행이다.
            #
            # 🔴 **왜 필요한가 — 우리 수익률은 배당을 안 담고 있다.**
            #
            # FinanceDataReader·pykrx 의 수정주가는 액면분할·무상증자·주식배당까지만 펴고
            # **현금배당은 안 편다**(`ingest/store/adj_price.py` 와 같은 범위). 학계 표준인
            # CRSP 는 `RET`(배당 포함)와 `RETX`(배당 제외)를 **나눠서** 주는데, 우리가 가진
            # 것은 `RETX` 뿐이다. 실측으로 그 크기를 쟀다 — 배당락일의 평균 일간수익률이
            # 평소보다 **1.4665%p 낮고 17년 전부 음수**다. 개별종목 중립대가 ±2% 이므로
            # 5거래일 창에 배당락일이 들어오면 그 라벨이 조용히 아래로 밀린다.
            #
            # ⚠️ 이 표를 처음 만들 때는 **−0.455%p** 로 적었다. 배당 자료가 없어
            #    "연말 마지막 거래일 대비 위치" 라는 대용치로 쟀던 값인데, 실제
            #    배당락일로 다시 재니 **3.2배**였다. 대용치는 크기를 줄이는 쪽으로
            #    틀린다 — 연말만 봐서 분기배당을, 상위 100 만 봐서 고배당 중소형주를
            #    놓쳤다.
            #
            # 🔴 **기준일(`dvdn_bas_dt`)은 거의 거래일이 아니다.**
            #
            # 12월 결산법인의 기준일은 12월 31일인데 그날은 휴장이다. 실측하면 12월 기준일
            # 중 거래일인 것은 **0.3%** 뿐이다. 값이 실제로 움직이는 날은 **배당락일**이고,
            # 그것은 결제(T+2) 때문에 기준일에서 두 걸음 앞이다.
            #
            #     ex_date = (기준일 이하 마지막 거래일) 에서 1 거래일 앞
            #
            # 그래서 `ex_date` 를 **거래일 달력으로 계산해 함께 담는다**. 계산 규칙이 바뀌면
            # 이 칸을 다시 채운다 — 거시(`macro_series.known_at`)와 같은 성질이다.
            # 달력을 모르면 계산할 수 없으므로 채우는 것은 저장 계층(`dividend_store`)이고,
            # 외부 연동 계층(`data_go_kr`)은 원문만 옮긴다.
            #
            # 🔴 **액면가는 그날의 값이 아니다.**
            #
            # 응답의 `stckParPrc` 는 **적재 시점의 액면가**다(삼성전자 1987년 배당 행에도
            # 100원이 들어 있다 — 2018년 분할 뒤 값이다). 그래서 옛 행의 배당률(액면 대비 %)에
            # 이 액면가를 곱하면 틀린다. 금액이 빈 행은 **금액이 없는 채로 둔다.**
            # 2018년 이후로는 금액 채움이 98% 를 넘으므로 개발구간 대부분은 금액이 있다.
            """
            CREATE TABLE IF NOT EXISTS dividend (
              isin_cd        TEXT NOT NULL,
              dvdn_bas_dt    TEXT NOT NULL,
              dvdn_rcd       TEXT NOT NULL,
              code           TEXT,
              crno           TEXT,
              corp_nm        TEXT,
              item_nm        TEXT,
              scrs_itms_kcd_nm TEXT,
              stac_md        TEXT,
              dvdn_rcd_nm    TEXT,
              cash_pay_dt    TEXT,
              stck_hndv_dt   TEXT,
              genr_dvdn_amt  REAL,
              grdn_dvdn_amt  REAL,
              genr_cash_dvdn_rt REAL,
              genr_dvdn_rt   REAL,
              cash_grdn_dvdn_rt REAL,
              grdn_dvdn_rt   REAL,
              par_price_at_load REAL,
              ex_date        TEXT,
              ex_date_rule   TEXT,
              src_bas_dt     TEXT,
              fetched_at     TEXT NOT NULL,
              -- 한 종목(ISIN)·한 기준일에 배당구분마다 한 줄이다. 전량 71,681행에서
              -- 이 세 칸의 조합이 유일함을 확인하고 정했다(중복 0).
              -- ⚠️ 재무에서 `account_detail` 을 PK 에 빠뜨려 6.4%가 조용히 사라진 적이
              --    있다. 그래서 "행 수가 맞다" 로 넘기지 않고 조합을 직접 세었다.
              PRIMARY KEY (isin_cd, dvdn_bas_dt, dvdn_rcd),
              -- 새 표라 검사할 기존 행이 없다 → CHECK 가 공짜다 (v1 과 같은 이유).
              CHECK (length(dvdn_bas_dt) = 8),
              -- 배당락일은 못 구할 수 있다(달력 밖의 옛 기준일). 다만 있으면 8자리다.
              CHECK (ex_date IS NULL OR length(ex_date) = 8),
              -- 배당락일이 있으면 그것을 어떤 규칙으로 세웠는지도 반드시 있다.
              CHECK (ex_date IS NULL OR ex_date_rule IS NOT NULL)
            )
            """,
            # 종목별 시계열을 훑을 때. 배당락일이 앞인 이유는 조인의 축이 그쪽이기
            # 때문이다 — "이 종목의 이 거래일에 배당락이 있었나" 가 우리가 묻는 질문이다.
            "CREATE INDEX IF NOT EXISTS idx_dividend_ex "
            "ON dividend(code, ex_date)",
            # 기준일 축으로 훑을 때(연도별 집계·커버리지 판정).
            "CREATE INDEX IF NOT EXISTS idx_dividend_bas "
            "ON dividend(dvdn_bas_dt, code)",
        ),
    ),
    (
        "v15: 총수익 — 수정주가에 현금배당이 빠져 있다",
        (
            # 수정주가 옆에 **총수익 축**을 더한다. `adj_close` 는 그대로 둔다 —
            # 지우고 덮는 것이 아니라 나란히 놓는 것이다. 둘은 답하는 질문이 다르다.
            #
            #   adj_close     "분할·병합을 편 주가가 얼마였나"      (가격 축)
            #   adj_close_tr  "배당까지 재투자하면 얼마가 됐나"     (성과 축)
            #
            # 🔴 **왜 필요한가 — 우리 수익률에 배당이 빠져 있다.**
            #
            # FinanceDataReader 의 수정주가는 액면분할·무상증자·주식배당까지만 펴고
            # 현금배당은 안 편다. 학계 표준인 CRSP 가 `RET`(배당 포함)와 `RETX`(배당
            # 제외)를 나눠 주는데 우리가 가진 것은 `RETX` 쪽뿐이다. 배당락일의 평균
            # 일간수익률은 평소보다 **1.4665%p 낮고 17년 전부 음수**다 — 개별종목
            # 중립대가 ±2% 이므로 5거래일 창에 배당락일이 들어오면 라벨이 조용히
            # 아래로 밀린다. (v14 를 만들 때 적은 −0.455%p 는 배당 자료가 없어 쓴
            # 대용치였고, 실제 배당락일로 다시 재니 3.2배였다.)
            #
            # 🔴 **성과 축이지 피처가 아니다.**
            #
            # 배당 확정은 주주총회다. 그날 이전에는 금액을 알 수 없으므로 이 칸을
            # 입력 피처로 쓰면 미래참조가 된다. 라벨·성과 평가에만 쓴다.
            #
            # 🔴 **배당도 가격과 같은 배율로 조정해야 한다.**
            #
            # 배당금은 그날의 원(貨) 단위 절대금액이고 `adj_close` 는 분할 배율이
            # 곱해진 값이라, 원금액을 그대로 더하면 축이 어긋난다. 조인되는 20,248행
            # 중 `adj_close/close ≠ 1` 인 행이 6,245행(30.8%)이다 — 안 맞추면 그만큼
            # 틀린다. 그래서 `adj_dividend = 배당금 × (adj_close / close)` 로 적는다.
            #
            #     r_t = (adj_close_t + adj_dividend_t) / adj_close_{t-1} − 1
            #     adj_close_tr_t = adj_close_tr_{t-1} × (1 + r_t)
            #
            # 🔴 **조인 축은 배당락일이지 기준일이 아니다.**
            #
            # 기준일(`dvdn_bas_dt`)은 12월 31일 같은 휴장일이 대부분이다(12월 기준일
            # 중 거래일은 0.3%). 값이 실제로 움직이는 날은 배당락일이고, 그 칸은
            # 거래일 달력으로 계산해 `dividend.ex_date` 에 담아 두었다. `ex_date` 가
            # 있는 43,464행은 전부 거래일이고 `daily_price` 에도 있다.
            _add_column("daily_price", "adj_dividend", "REAL"),
            _add_column("daily_price", "adj_close_tr", "REAL"),
            # 원본 배당금의 단위 오류를 표시한다. **지우지 않고 깃발만 세운다** —
            # CRSP 가 `DlyDistRetFlg` 로 배분 종류를 남기는 것과 같은 이유로, 판정을
            # 되짚을 수 있어야 하기 때문이다. 총수익 계산에서만 그 배당을 0 으로 본다.
            #
            # 🔴 **크기로 자르지 않는다.** 배당이 종가보다 큰 행 넷 중 셋이 정당한
            # 청산분배금이었다 — 유전펀드 청산(종가 28원에 1,670원 분배)·리츠 분배가
            # 실제로 그렇게 크다. 극단값이 데이터 오류일 때만 처리하고 데이터 생성
            # 과정에서 나온 것은 그대로 두는 것이 표준이다(Leone et al.).
            #
            # 대신 **원본이 자기 정합적인가**로 가른다. `genr_cash_dvdn_rt` 는 액면가
            # 대비 배당률이라(현대차 2,500원 = 액면 5,000 × 50%) 금액을 그 비율로 나누면
            # 그날의 액면가가 되짚어진다. 되짚은 값이 표준 액면가에 붙으면 금액과
            # 배당률이 서로 맞는다는 뜻이므로 믿는다 — 조인 대상의 99.29%가 그렇다.
            # 배당률이 없어 되짚을 수 없는 33행만 배당수익률로 최후 판정하는데, 그 안에서
            # 윙입푸드홀딩스(132,722,717%)와 그다음(19.5%)의 간극이 680만 배다.
            _add_column("daily_price", "is_dividend_suspect", "INTEGER"),
        ),
    ),
    (
        "v16: 기업행위 — 결과만 있고 사건이 없었다",
        (
            # ── 기업행위 ────────────────────────────────────────────────
            # 지금까지 자본변동은 **결과로만** 존재했다. `adj_close` 는 이미 펴진 값이고
            # `adj_source` 는 그 값이 어디서 왔는지만 말한다. "무슨 일이 언제 얼마만큼
            # 일어났나" 는 어디에도 없어서, 물을 때마다 920만 행을 다시 훑으며
            # `common.corporate_actions.adjustment_factor` 로 역산해야 했다(3분 남짓).
            #
            # zipline 이 `splits`·`mergers`·`dividends` 세 표를 따로 두는 이유가 이것이다.
            # 조정된 가격과 **조정을 일으킨 사건**은 다른 자료다. 우리는 배당을 v14 에서
            # 이미 표로 세웠으므로, 남은 자본변동을 여기서 세운다.
            #
            # ⚠️ `supply/training.py` 의 `CORPORATE_ACTION_COLUMNS`(`is_liquidation` ·
            #    `is_halted` · `is_first_listing`)와 **이름이 겹치지만 다른 것**이다.
            #    그쪽은 "이 행의 가격을 믿어도 되나" 를 말하는 행 단위 깃발이고,
            #    이 표는 "무슨 사건이 있었나" 를 말하는 사건 단위 기록이다.
            #
            # 🔴 **왜 사건마다 증거를 함께 담나 — 출처 하나로는 사건이 안 갈린다.**
            #
            # 액면가만 보면 틀린다. 액면가 변경 792건을 자본금(`액면가 × 주식수`)으로
            # 가르면 성격이 셋으로 갈린다 (2026-09-10 실측).
            #
            #     순수 분할·병합 (자본금 불변)   666   가격이 정확히 액면가 비율만큼 움직인다
            #     동시사건 (자본금이 바뀜)         84   감자와 분할이 같은 날 — 액면가로 예측 불가
            #     액면가 감액 (주식수 그대로)      42   **가격이 안 움직인다** — 결손금을 턴 것뿐
            #
            # 액면가 감액 42건에 액면가 비율을 적용하면 멀쩡한 가격을 10배 망친다.
            # 그래서 `par_before/after` 와 `shares_before/after` 를 **둘 다** 담고,
            # 무엇으로 판정했는지를 `source` 에 남긴다.
            #
            # 🔴 **주식수가 바뀌었다고 다 조정 대상이 아니다.**
            #
            # 주식수 변경 41,051건 중 대부분이 유상증자다 — 주식수만 늘고 KRX 기준가는
            # 그대로라 **가격이 연속**이다. 그런 행은 `event_type='share_change'` 로 담되
            # `ratio_num/den` 을 비운다. 담는 이유는 희석이 그 자체로 신호이기 때문이고,
            # 비우는 이유는 여기에 배율을 적으면 쓰는 쪽이 곱해 버리기 때문이다.
            #
            # 🔴 **배율은 유리수 두 칸이다 — REAL 하나가 아니다.**
            #
            # 1/50 · 1/1500 같은 계수가 한 종목의 수천 행에 누적으로 곱해진다. REAL 로
            # 담으면 반올림이 다시 들어온다(`adjustment_factor` 가 `Fraction` 을 주는
            # 것과 같은 이유). 쓰는 쪽에서 **마지막에 한 번만** float 로 바꾼다.
            #
            # 🔴 **판정 결과까지 담는 까닭 — 검사가 대상과 같은 잘못을 공유하면 안 된다.**
            #
            # `chain_num/den` 은 `adj_price` 가 실제로 쓴 계수이고, `agrees` 는 이 표가
            # 증거로 세운 배율과 그것이 맞는지다. 어긋난 자리를 **표 안에 남겨야** 나중에
            # 새 어긋남이 생겼을 때 드러난다. 밖에서 매번 다시 세면 아무도 안 센다.
            #
            # 실측 (2026-09-10 · 전 종목 9,231,938행 · 적재 264초)
            #
            #     사건       44,290   2,946종 · 20100105~20260904
            #     배율을 예고   4,640
            #     판정 가능      730   액면가·상장일 축 (나머지는 아래 `agrees` 참고)
            #       어긋남         8
            #     자기모순        17   **17/17 정리매매**
            #
            # 자기모순 17건은 전부 그 종목의 **마지막 7거래일**에서 시작한다. 주식수가
            # 감자로 크게 줄었는데 KRX 기준가비가 정확히 1 이라(기준가를 안 고쳤다)
            # 등락률이 가격제한폭 밖으로 나온다(제일바이오 2026-02-09 +29,948%).
            # 정리매매 구간에는 가격제한폭이 적용되지 않으므로 **원본 오류가 아니다.**
            # 17건 전부 `is_liquidation` 이 이미 덮고 `training_frame` 이 덜어내므로
            # 학습 표본으로 새는 것은 0건이다. 그래서 값을 고치지 않고 표시만 한다.
            """
            CREATE TABLE IF NOT EXISTS corporate_action (
              code           TEXT    NOT NULL,
              -- 사건이 **가격에 반영된 첫 거래일**이다. 공시일도 결의일도 아니다 —
              -- 우리가 답해야 하는 질문이 "이 거래일의 가격이 앞날과 이어지나" 이므로
              -- 축을 거래일에 맞춘다. 배당의 `ex_date` 와 같은 뜻의 날짜다.
              ex_date        TEXT    NOT NULL,
              event_type     TEXT    NOT NULL,
              -- 가격 조정 배율. 액면분할 50:1 이면 1/50, 병합 5:1 이면 5.
              -- **가격이 연속인 사건(유상증자·이전상장)은 비운다.**
              ratio_num      INTEGER,
              ratio_den      INTEGER,
              -- 무엇을 보고 배율을 정했나. par(액면가) · shares(주식수) ·
              -- basis(KRX 기준가) · listing(상장일) · none(배율 없음)
              source         TEXT    NOT NULL,
              -- 증거 — 판정을 되짚을 수 있어야 한다.
              -- ⚠️ 액면가는 **숫자가 아닐 수 있다.** '무액면' 73,713행 · 주식예탁증권의
              --    '0' 7,616행 · 외국주 '.5' 같은 소수 표기가 실재한다. 그래서 TEXT 로
              --    받아 원문 그대로 둔다 — 숫자로 강제하면 그 행이 조용히 결측이 된다.
              par_before     TEXT,
              par_after      TEXT,
              shares_before  INTEGER,
              shares_after   INTEGER,
              -- KRX 기준가 / 전일종가. 재개일 판정의 근거라 판정값과 따로 남긴다.
              basis_ratio    REAL,
              -- `adj_price` 가 실제로 쓴 계수 (`common.corporate_actions.factor_series`).
              chain_num      INTEGER,
              chain_den      INTEGER,
              -- 증거로 세운 배율과 chain 계수가 맞나.
              --
              -- 🔴 **증거가 chain 과 같은 값이면 NULL 이다.** `rights_off` ·
              --    `resume_revalue` 는 배율을 `adjustment_factor` 에서 얻으므로
              --    그것을 다시 chain 과 대조하면 3,974건이 통째로 초록이 된다 —
              --    검사가 대상과 같은 판정을 공유하는 자리다. 잴 수 있는 것은
              --    액면가(`split_merge`·`par_reduction`)와 상장일
              --    (`market_transfer`)뿐이고, 그 730건에서 8건이 어긋난다.
              agrees         INTEGER,
              -- 원본 두 칸이 서로 모순인가 — **chain 을 안 쓰고 재는 축이다.**
              --
              -- 주식수가 1.5배 밖으로 바뀌었는데 그 날 KRX 등락률이 가격제한폭
              -- 밖이면, 같은 원본이 "주식수가 1,500분의 1 이 됐는데 가격은
              -- 이어진다" 고 말하는 셈이다. `agrees` 와 **다른 질문**이라 칸을
              -- 따로 둔다 — 한 칸에 섞으면 붉은불을 보고 무엇을 고쳐야 하는지
              -- 모른다.
              --
              -- 실측 17건 · **17건 전부 정리매매**이고 전부 기준가비가 정확히 1 이다
              -- (제일바이오 2026-02-09 은 주식수 ×1/1500 에 등락률 +29,948%).
              -- 정리매매 구간에는 가격제한폭이 적용되지 않으므로 원본 오류가 아니고,
              -- `is_liquidation` 이 이미 덮어 학습에 안 들어간다.
              source_conflict INTEGER NOT NULL DEFAULT 0,
              -- 어긋났을 때의 해명. 그 날이 정리매매 구간인가.
              is_liquidation INTEGER NOT NULL DEFAULT 0,
              built_at       TEXT    NOT NULL,
              -- 한 종목·한 거래일에 종류가 다른 사건이 겹칠 수 있다 — 액면분할과
              -- 유상증자가 같은 날 나는 자리가 실재한다. 그래서 종류까지 키에 넣는다.
              -- ⚠️ 재무에서 `account_detail` 을 키에 빠뜨려 6.4%가 조용히 사라진 적이
              --    있다. 실측으로 조합의 유일성을 확인하고 정했다(중복 0).
              PRIMARY KEY (code, ex_date, event_type),
              -- 새 표라 검사할 기존 행이 없다 → CHECK 가 공짜다 (v1 과 같은 이유).
              CHECK (length(ex_date) = 8),
              CHECK (event_type IN ('split_merge', 'par_reduction', 'resume_revalue',
                                    'rights_off', 'series_restart', 'market_transfer',
                                    'share_change')),
              CHECK (source IN ('par', 'shares', 'basis', 'listing', 'none')),
              -- 분모가 0 이면 배율이 아니라 오류다. 분자만 있고 분모가 없는 것도 마찬가지.
              CHECK ((ratio_num IS NULL) = (ratio_den IS NULL)),
              CHECK (ratio_den IS NULL OR ratio_den > 0),
              CHECK ((chain_num IS NULL) = (chain_den IS NULL)),
              CHECK (chain_den IS NULL OR chain_den > 0),
              CHECK (agrees IS NULL OR agrees IN (0, 1)),
              CHECK (source_conflict IN (0, 1)),
              CHECK (is_liquidation IN (0, 1))
            )
            """,
            # 종목의 시계열을 훑으며 "이 날 사건이 있었나" 를 묻는 것이 주된 질문이라
            # PK 가 이미 그 축이다. 여기서는 **날짜 축**과 **종류 축**을 따로 연다.
            #
            # 날짜 축: "2026-02-09 에 무슨 일이 있었나" (일별 점검·대시보드)
            "CREATE INDEX IF NOT EXISTS idx_ca_date "
            "ON corporate_action(ex_date, code)",
            # 종류 축: "어긋난 자리만 보여 달라" — 붉은불 조회가 전수 훑기가 되면
            # 아무도 안 본다.
            "CREATE INDEX IF NOT EXISTS idx_ca_type "
            "ON corporate_action(event_type, agrees)",
        ),
    ),
)

#: 이 코드가 아는 최신 스키마 버전.
LATEST_VERSION = len(MIGRATIONS)


# ==================================================
# 2. 도구
# ==================================================
def user_version(conn: sqlite3.Connection) -> int:
    """이 DB 가 어느 버전인지 돌려준다. 한 번도 마이그레이션하지 않았으면 0 이다."""
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def column_names(conn: sqlite3.Connection, table: str) -> List[str]:
    """표의 칸 이름 목록. 표가 없으면 빈 목록이다."""
    # PRAGMA 는 파라미터 바인딩을 받지 않아 이름을 문자열로 끼워 넣어야 한다.
    # `table` 은 **코드 안 상수만** 넘긴다 — 바깥 입력을 그대로 흘리면 SQL 주입이 된다.
    if not table.replace("_", "").isalnum():
        raise MigrationError(f"표 이름이 이상하다: {table!r}")
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]


def add_column_sql(conn: sqlite3.Connection, table: str, column: str,
                   definition: str) -> Optional[str]:
    """칸이 없을 때만 `ALTER TABLE ... ADD COLUMN` 문장을 만들어 준다.

    `ADD COLUMN` 은 **같은 이름이 이미 있으면 예외를 던진다** — `IF NOT EXISTS` 가 없다.
    마이그레이션은 여러 번 돌아도 안전해야 하므로 여기서 미리 걸러 준다.

    🔴 **표가 아직 없으면 `None` 이다 — 그리고 그건 정상이다.**
    -------------------------------------------------------
    `daily_price` 는 이 파일이 아니라 `krx_store.SCHEMA` 가 만든다. 그런데
    `migrate_path()` 는 예산·raw 저장소·robots·반입이 **각자 필요할 때 지연 호출**하므로,
    시세를 한 번도 받지 않은 DB 에서는 표가 없는 채로 v9 가 돈다. 거기서 예외를 던지면
    시세와 무관한 기능이 전부 못 뜬다.

    ⚠️ 대신 **`krx_store.SCHEMA` 에도 같은 칸을 넣어 둔다.** 여기서 건너뛰기만 하면
       그 DB 는 이미 v9 로 표시된 뒤에 칸 없는 `daily_price` 를 만들게 되고,
       마이그레이션은 다시 돌지 않으므로 **칸이 영영 안 생긴다.** 두 경로가 같은 모양으로
       모이게 하는 것이 요점이다 — 한쪽만 고치면 조용히 갈라진다.
    """
    if not column_names(conn, table):
        return None                    # 표 자체가 없다 — SCHEMA 쪽이 갖고 태어난다
    if column in column_names(conn, table):
        return None
    return f"ALTER TABLE {table} ADD COLUMN {column} {definition}"


# ==================================================
# 3. 실행
# ==================================================
def migrate(conn: sqlite3.Connection) -> int:
    """밀린 마이그레이션을 순서대로 적용하고, **적용한 개수**를 돌려준다.

    이미 최신이면 아무것도 하지 않고 0 을 돌려준다 (여러 번 불러도 안전하다).

    ⚠️ `conn` 은 **autocommit 모드**여야 한다 (`isolation_level=None`).
       파이썬 `sqlite3` 의 기본 모드는 우리가 `BEGIN` 을 쓰려 하면 *"cannot start a
       transaction within a transaction"* 으로 막는다. `connect_for_migration()` 을 쓰면
       알아서 맞춰 준다.
    """
    if conn.isolation_level is not None:
        raise MigrationError(
            "마이그레이션 연결은 autocommit 모드여야 한다.\n"
            "  할 일: sqlite3.connect(..., isolation_level=None) 으로 열거나,\n"
            "         ingest.store.migrations.connect_for_migration() 을 쓴다."
        )

    current = user_version(conn)
    if current > LATEST_VERSION:
        raise MigrationError(
            f"DB 가 코드보다 최신이다 (DB v{current} > 코드 v{LATEST_VERSION}).\n"
            "  왜 위험한가: 최신 코드가 만든 표를 옛 코드가 모른 채 쓰면 조용히 틀린다.\n"
            "  할 일: git pull 로 코드를 최신으로 맞춘다."
        )

    applied = 0
    for index in range(current, LATEST_VERSION):
        name, statements = MIGRATIONS[index]
        target = index + 1

        # ① BEGIN IMMEDIATE 를 명시한다 — 기본 모드는 DDL 을 트랜잭션에 넣지 않는다.
        conn.execute("BEGIN IMMEDIATE")
        try:
            # ② executescript() 를 쓰지 않는다 — 열린 트랜잭션을 커밋해 롤백을 막는다.
            for statement in statements:
                # 지연 문장(`_add_column`·`_rebuild_index_price`)은 지금 연결을 보고
                # 문장을 만든다. `None` 은 "이미 되어 있어 할 일이 없다" 이므로 건너뛴다.
                # 목록을 주면 그 안의 문장을 **이 트랜잭션에서** 차례로 돌린다.
                if callable(statement):
                    statement = statement(conn)
                    if statement is None:
                        continue
                for 문장 in ([statement] if isinstance(statement, str) else statement):
                    conn.execute(문장)
            # ③ 버전 표시를 같은 트랜잭션에 넣는다 — 중간 상태를 원천 차단한다.
            #    PRAGMA 는 파라미터 바인딩을 받지 않지만 `target` 은 우리가 만든 정수다.
            conn.execute(f"PRAGMA user_version={int(target)}")
            conn.execute("COMMIT")
        except Exception as exc:                       # noqa: BLE001 — 되살려 던진다
            conn.execute("ROLLBACK")
            raise MigrationError(
                f"마이그레이션 v{target}({name}) 이 실패해 되돌렸다: {exc}\n"
                f"  DB 는 v{index} 그대로다 — 반쪽 스키마가 남지 않았다.\n"
                "  할 일: 위 오류를 고친 뒤 다시 실행한다."
            ) from exc
        applied += 1

    return applied


def connect_for_migration(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """마이그레이션 전용 연결을 연다 — autocommit 모드로.

    `krx_store.connect()` 를 쓰지 않는 이유: 그쪽은 기본 isolation_level 이라
    `BEGIN IMMEDIATE` 를 쓸 수 없고, 블록이 끝날 때 자동으로 커밋한다.
    마이그레이션은 **커밋 시점을 우리가 정해야** 한다.
    """
    if db_path is None:
        db_path = krx_db_path()

    conn = sqlite3.connect(db_path, timeout=60, isolation_level=None)
    # ⚠️ **순서가 중요하다.** `journal_mode=WAL` 은 DB 가 아직 WAL 이 아닐 때 잠깐
    #    배타 잠금을 잡는데, `busy_timeout` 을 그 뒤에 걸면 기다릴 시간이 0 이라
    #    다른 연결이 쓰는 중이면 즉시 `database is locked` 로 죽는다.
    #    이미 WAL 인 DB 에서는 no-op 이라 잘 드러나지 않다가, **새 DB 를 여러 스레드가
    #    동시에 열 때만** 터진다.
    conn.execute("PRAGMA busy_timeout=60000")
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def migrate_path(db_path: Optional[Path] = None) -> int:
    """경로를 받아 열고·마이그레이션하고·닫는다. 적용한 개수를 돌려준다."""
    conn = connect_for_migration(db_path)
    try:
        return migrate(conn)
    finally:
        conn.close()
