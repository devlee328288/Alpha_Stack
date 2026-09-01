"""저장 계층이 공유하는 것 — 쓰기 자물쇠와 모듈 작성 규약

## 왜 이 파일이 있나

**같은 DB 파일에 쓰는 모듈이 서로 다른 자물쇠를 쥐면 자물쇠가 없는 것과 같다.**

`krx_index.py` 는 지금까지 `krx_store` 의 **사적 이름** `_write_lock` 을 import 해서
썼다. 동작은 맞았지만 이름 앞의 밑줄이 *"밖에서 쓰지 말라"* 는 뜻이므로, 다음 사람이
그걸 보고 *"내 모듈은 내 자물쇠를 만들어야겠다"* 로 갈 수 있다. 그러면 **두 자물쇠가
서로를 모른 채 같은 파일에 동시에 쓴다** — 그리고 SQLite 는 `database is locked` 로
가끔만 터지므로 **재현이 안 되는 버그**가 된다.

공유해야 하는 것을 공개 이름으로 옮긴다.

    from ingest.store.sqlite_db import write_lock

    with write_lock, connect() as conn:
        ...

## 여기에 `connect()` 와 `DB_PATH` 는 옮기지 않는다

옮기고 싶어지지만 **옮기면 테스트가 조용히 무력화된다.**

`krx_store.connect()` 는 모듈 전역 `krx_store.DB_PATH` 를 읽고, 테스트 두 곳이
바로 그 이름을 `monkeypatch` 해서 임시 DB 로 갈아 끼운다. 연결 함수를 여기로 옮기면
그 monkeypatch 가 **아무 일도 하지 않게 되고, 테스트는 그대로 통과한다.**
2026-08-26 에 실제로 진짜 DB(1.6GB)에 20행이 들어간 사고가 그런 종류였다.

## 🔴 새 저장 모듈은 `DB_PATH` 모듈 전역을 만들지 않는다

`krx_store`·`krx_index` 는 모듈 전역 `DB_PATH` 를 쓴다. 그래서 테스트가 **모듈마다**
따로 갈아 끼워야 하고, **하나를 빠뜨려도 테스트는 통과한다.**

새로 만드는 저장 모듈은 `common/budget.py` · `common/raw_store.py` ·
`ingest/store/collect_log.py` 가 이미 쓰는 방식을 따른다 —

    def _connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
        conn = sqlite3.connect(db_path or krx_db_path(), timeout=60, isolation_level=None)
        conn.execute("PRAGMA busy_timeout=60000")
        conn.row_factory = sqlite3.Row
        return conn

    def 어떤함수(..., *, db_path: Optional[Path] = None) -> ...:

**경로를 인자로 받으면 갈아 끼울 자리가 하나뿐**이고, `tests/conftest.py` 가
프로세스 전체에서 진짜 경로를 막는 것과도 맞물린다.

## 마이그레이션 번호는 미리 배정한다

`ingest/store/migrations.py` 의 `MIGRATIONS` 는 **이름이 아니라 인덱스로** 돈다.
그래서 두 갈래에서 각자 *"다음은 v5"* 라고 붙인 채 병합되면, **이미 v5 를 적용한 DB 는
나중에 v5 자리에 들어온 항목을 영원히 건너뛴다.** 예외도 경고도 없다.

| 번호 | 무엇 | 상태 |
|---|---|---|
| v1~v4 | 기존 | 적용됨 |
| **v5** | 반입 (`inbox_batch` · `inbox_accepted` · `inbox_quarantine`) | **적용됨** (2026-09-01) |
| **v6** | 공시 시점정합 (`dart_disclosure` · `dart_financial`) | **예약** |
| **v7** | 거시 통계 (`macro_series`) | **예약** |
| v8 | 다음 빈 번호 | — |

v5·v6 은 처음에 공시·거시로 예약돼 있었으나 실제로 먼저 온 것이 반입이라 **한 칸씩 밀었다.**
밀 수 있었던 것은 그 번호를 적용한 DB 가 아직 없었기 때문이다 — 예약은 자리를 비워 둔 것이지
배포된 것이 아니다. **이미 배포된 항목은 이렇게 못 민다.**

**번호를 선점한 쪽이 먼저 머지한다.** 두 갈래는 직렬로만 합친다.
"""

from __future__ import annotations

import threading

#: 같은 DB 파일에 쓰는 모듈이 **공유하는** 자물쇠.
#:
#: SQLite 는 동시에 한 연결만 쓸 수 있다. 쓰기를 줄 세워 잠금 충돌 자체를 없앤다.
#: (읽기는 WAL 덕분에 잠금과 무관하므로 이 자물쇠를 거치지 않는다.)
#:
#: ⚠️ **모듈마다 새로 만들지 않는다.** 자물쇠가 둘이면 자물쇠가 없는 것과 같다.
write_lock = threading.Lock()

#: 다음에 쓸 마이그레이션 번호. 위 표를 함께 고친다.
#: 반입이 v5 를 가져가면서 공시·거시 예약이 v6·v7 로 밀렸으므로 빈 자리는 v8 이다.
NEXT_MIGRATION_VERSION = 8

__all__ = ["write_lock", "NEXT_MIGRATION_VERSION"]
