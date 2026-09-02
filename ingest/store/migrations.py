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
#: SQL 을 만들어 주는 함수**다. 함수를 허용하는 이유는 하나뿐이다:
#:
#:   `ALTER TABLE ... ADD COLUMN` 에는 `IF NOT EXISTS` 가 없다. 같은 이름이 이미 있으면
#:   **예외를 던진다.** 그런데 이 파일의 규약은 *"문장은 여러 번 돌려도 같은 결과"* 다.
#:   문자열만으로는 그 둘을 동시에 만족할 수 없어서, 칸이 있는지 보고 문장을 만들거나
#:   `None` 을 주는 함수를 받는다.
#:
#: `None` 을 돌려주면 그 문장은 **건너뛴다** — 할 일이 없다는 뜻이다.
Statement = "str | Callable[[sqlite3.Connection], Optional[str]]"


def _add_column(table: str, column: str, definition: str) -> Callable:
    """`ADD COLUMN` 을 **칸이 없을 때만** 내는 지연 문장을 만든다.

    `add_column_sql()` 은 연결이 있어야 판단할 수 있는데 `MIGRATIONS` 는 모듈이 읽힐 때
    만들어지므로 아직 연결이 없다. 그래서 판단을 실행 시점으로 미룬다.
    """
    return lambda conn: add_column_sql(conn, table, column, definition)


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
#      v10 다음 빈 번호
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
                # 지연 문장(`_add_column`)은 지금 연결을 보고 문장을 만든다.
                # `None` 은 "이미 되어 있어 할 일이 없다" 이므로 건너뛴다.
                if callable(statement):
                    statement = statement(conn)
                    if statement is None:
                        continue
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
