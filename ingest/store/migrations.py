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
from typing import List, Optional, Sequence, Tuple

from common.paths import krx_db_path  # DB 경로 — 유일한 정의


class MigrationError(RuntimeError):
    """마이그레이션이 실패했다. 무엇을 해야 하는지까지 메시지에 담는다."""


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
#      v5  공시 시점정합 (dart_disclosure · dart_financial)   예약
#      v6  거시 통계 (macro_series)                            예약
#      v7  다음 빈 번호
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

    시세·공시 표에 *"이 행을 우리가 언제부터 알 수 있었나"* 칸을 얹을 때 쓸 자리다.
    지금은 아직 쓰이지 않는다.
    """
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
                conn.execute(statement)
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
