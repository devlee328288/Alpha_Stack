"""DB 엔진을 만드는 유일한 자리 (저장계층 전환 S2·S4 · ADR-DS-0011 · ADR-DS-0015).

`settings.py` 가 **어떻게 붙을지**를 정하고, 이 모듈이 그 값을 그대로 엔진에 옮긴다.
`paths.py` 가 경로의 기준점이고 `settings.py` 가 환경의 기준점인 것과 같은 자리다 —
이쪽은 **커넥션의 기준점**이다.

## 이제 어댑터가 이것을 쓴다 (S4, 2026-08-25)

S2 는 **아무도 import 하지 않는 상태**로 끝냈다. 엔진과 어댑터를 한 커밋에 섞으면 접속이
안 될 때 그것이 커넥션 설정 탓인지 질의 탓인지 가릴 수 없기 때문이었고, 그 판단이 값을
했다 — 배포본 전략에 손잡이 하나가 통째로 빠져 있었다는 것이 그때 드러났다(ADR-DS-0003 rev.2).

S4 에서 `ingest/store/krx_pg.py` 가 이 모듈을 부르기 시작했다. `tests/test_db.py` 가
얼려 두던 "app/ 안에서 아무도 import 하지 않는다" 검사는 그 표시로 지웠다(ADR-DS-0011 §5).

    실측 도구: `python3 scripts/check_db_connection.py`

⚠️ 어댑터는 async 가 아니라 **§2-1 의 동기 다리**를 거친다. 이 레포의 라우트 핸들러가
전부 `def` 라서다. 그 절에 실측표와 기각한 대안이 함께 있다.

## 왜 `settings.py` 와 나뉘어 있나

`settings.py` 는 SQLAlchemy 를 import 하지 않는다(ADR-DS-0003 §6). 설정을 읽는 것만으로
무거운 의존성이 딸려 오면 안 되기 때문이다. 그래서 그쪽은 `use_null_pool: bool` 같은
**사실**만 내놓고, 그 사실을 풀 클래스로 **해석**하는 일은 이 모듈이 한다.

## ⚠️ 준비구문 손잡이는 **셋**이고 한 벌이다

`connect_args` 로 넘어가는 값들은 성능 손잡이처럼 보이지만 아니다. transaction 모드
풀러(6543) 뒤에서 하나라도 빠지면 깨진다. 2026-08-23 에 pgbouncer 1.25.2 를
`pool_mode=transaction · max_prepared_statements=0` 으로 세워 실측했다.

**① 캐시 두 개** — 한쪽만 끄면 조합마다 결과가 다르다 (각 칸 200회):

| 손잡이 | NullPool | QueuePool(3) |
|---|---|---|
| 아무것도 안 끔 | 87건 실패 | 96건 실패 |
| `statement_cache_size` 만 0 | 0건 | **94건 실패** |
| `prepared_statement_cache_size` 만 0 | 87건 실패 | 7건 실패 |
| 둘 다 0 | 0건 | 0건 |

**② 이름** — ⭐ 위 표의 "둘 다 0"은 **갓 띄운 풀러에서만** 0건이다. asyncpg 0.30 은
캐시를 꺼도 준비구문에 이름을 붙이고, 그 이름은 커넥션마다 1부터 다시 세는 카운터다.
서버리스처럼 **호출마다 커넥션이 새로 열리면** 같은 이름을 계속 다시 쓰게 되고,
DEALLOCATE 가 다른 물리 커넥션으로 새면 이름이 남는다. 잔여물이 쌓인 뒤 다시 재면
(호출마다 엔진을 새로 만드는 서버리스 모양 · 24회):

| 조합 | 실패 |
|---|---|
| 둘 다 0 (캐시만) | **24건 (전부)** |
| 이름 함수만 (캐시는 켠 채) | 23건 |
| **둘 다 0 + 이름 함수** | **0건** |

실패는 `DuplicatePreparedStatementError: prepared statement "__asyncpg_stmt_1__"
already exists` 이고, SQLAlchemy 의 방언 초기화 질의(`select pg_catalog.version()`)에서부터
터진다 — 즉 **첫 요청부터 전부 죽는다.**

⚠️ **이 고장은 깨끗한 상대에서 재현되지 않는다.** 그래서 한 번 대 보고 "0건이니 됐다"고
결론내면 거짓 음성을 얻는다. 이 파일의 초안이 실제로 그렇게 틀렸다.
근거·버전별 차이·전체 표는 ADR-DS-0003 (rev.2) 에 있다.

## ⚠️ 이 계층은 DDL 을 발행하지 않는다

`krx_store` 는 조회 경로마다 `CREATE TABLE IF NOT EXISTS` 를 발행한다 — `init_db()`
(krx_store.py:146)를 **9곳**에서 부른다(167·214·265·306·320·342·425·501·528).
SQLite 에서는 무해했지만 Postgres 에서는 그러면 안 된다.
SQLAlchemy asyncpg 방언의 문서가 직접 경고한다 —

    "If DDL changes are made from other database engines and/or processes, a running
     application may encounter asyncpg exceptions ``InvalidCachedStatementError`` and/or
     ``InternalServerError("cache lookup failed for type <oid>")``"
    (sqlalchemy/dialects/postgresql/asyncpg.py 모듈 docstring)

스키마는 `sql/init/*.sql` 이 **빈 볼륨에서 한 번** 세운다. 어댑터는 읽고 쓸 뿐이다.
"""

from __future__ import annotations

import asyncio
import logging
import threading  # 동기 다리의 전용 루프 스레드 (§2-1)
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Sequence
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine
from sqlalchemy.pool import NullPool

from common import settings

log = logging.getLogger(__name__)

# ping 이 매달려 있지 않게 하는 상한(초). 로컬 DB 가 안 떠 있을 때 DNS·TCP 에서
# 수십 초를 기다리면 "안 뜬다"가 "느리다"로 보인다.
PING_TIMEOUT_SECONDS = 5.0

# 로컬에서만 켠다. 도커 DB 를 재시작하면 풀이 들고 있던 소켓이 죽어 있는데,
# 그걸 모른 채 꺼내 쓰면 첫 요청이 한 번 실패한다. 검사 왕복 한 번으로 막는다.
# 배포본은 NullPool 이라 매번 새 커넥션이므로 켤 이유가 없다(왕복만 는다).
LOCAL_POOL_PRE_PING = True


def unique_statement_name() -> str:
    """준비구문 이름을 **커넥션 사이에서 겹치지 않게** 만든다 (ADR-DS-0003 rev.2 §4).

    asyncpg 의 기본 이름은 `__asyncpg_stmt_1__` 처럼 커넥션마다 1부터 다시 세는
    카운터다. 직결이면 문제가 없지만, transaction 모드 풀러 뒤에서는 서로 다른 클라이언트가
    **같은 물리 커넥션에 같은 이름**을 쓰게 되어 `DuplicatePreparedStatementError` 가 난다.

    형태는 SQLAlchemy 문서의 PgBouncer 예제와 같게 둔다 — 접두사를 유지하면 서버에서
    `pg_prepared_statements` 를 볼 때 무엇이 남긴 것인지 알아볼 수 있다.

    `settings.py` 가 아니라 여기 있는 이유: 설정은 "유일해야 한다"는 **사실**만 말하고
    (`unique_statement_names`), 무엇으로 유일하게 만들지는 드라이버 사정이다.
    """
    return f"__asyncpg_{uuid4()}__"


# ==================================================
# 1. 엔진 인자 — 순수 함수라 DB 없이 검사된다
# ==================================================
def engine_kwargs(
    db: settings.DatabaseSettings | None = None, *, echo: bool = False
) -> dict[str, Any]:
    """`create_async_engine()` 에 넘길 인자를 만든다. **엔진을 만들지는 않는다.**

    엔진 생성과 분리해 둔 이유는 검사 때문이다. 이 함수는 DB 도 이벤트 루프도 필요 없어서
    `tests/test_db.py` 가 배포본 전략 한 벌(포트·풀·캐시 둘·이름 유일화)을 얼려 둘 수 있다.

    ⚠️ **두 캐시 값은 `connect_args` 안에 `int` 로 있어야 한다.** 옮길 자리가 셋처럼
    보이는데 나머지 둘은 각각 다른 방식으로 죽는다.

    | 어디에 두나 | 결과 |
    |---|---|
    | `connect_args={...: 0}` | ✅ 정답 |
    | `create_async_engine(url, prepared_statement_cache_size=0)` | ❌ `TypeError` — 방언 인자가 아니라 **DBAPI 인자**다 |
    | URL 쿼리 `?statement_cache_size=0` | ❌ 문자열 `"0"` 로 도착한다. asyncpg 가 `"0" < 0` 을 시도해 `TypeError` |

    URL 쿼리는 `prepared_statement_cache_size` 만 int 로 강제된다
    (asyncpg 방언의 `create_connect_args` 가 그 키에만 `coerce_kw_type` 을 건다).
    맨 `statement_cache_size` 는 강제 대상이 아니라 문자열인 채로 넘어간다.
    `tests/test_db.py` 가 자리와 타입을 함께 얼려 둔다 — "정리"로 옮기는 것을 막기 위해서다.
    """
    db = db or settings.database_settings()

    # MappingProxy 를 그대로 주면 SQLAlchemy 가 병합하면서 실패한다. 사본을 준다.
    connect_args: dict[str, Any] = dict(db.connect_args)
    if db.unique_statement_names:
        # 세 번째 손잡이. 캐시 둘만 끄면 갓 띄운 풀러에서만 통한다 — 모듈 docstring 의 ② 표.
        connect_args["prepared_statement_name_func"] = unique_statement_name

    kwargs: dict[str, Any] = {
        "url": db.url,
        "connect_args": connect_args,
        "echo": echo,
    }

    if db.use_null_pool:
        # 서버리스는 호출 사이에 얼었다 녹는다. 풀이 들고 있던 소켓은 다음 호출에서 이미
        # 죽어 있고, 앞단 Supavisor 가 이미 풀이라 그 위에 풀을 또 얹으면 회계가 어긋난다.
        kwargs["poolclass"] = NullPool
    else:
        kwargs["pool_pre_ping"] = LOCAL_POOL_PRE_PING

    return kwargs


def port_warning(db: settings.DatabaseSettings | None = None) -> str | None:
    """`DATABASE_URL` 의 포트가 이 환경의 기대값과 다르면 그 사실을 문장으로 돌려준다.

    **고치지 않는다** (ADR-DS-0003 §근거). 사용자가 적은 값을 말없이 바꾸면 설정 화면에
    적힌 값과 실제로 붙는 곳이 갈린다. 알려 주기만 하고 적힌 대로 붙는다.

    포트를 못 읽는 형태(`host/db` 처럼 생략된 경우)는 경고하지 않는다 — 기본 포트를
    쓰겠다는 정상적인 표기다.
    """
    db = db or settings.database_settings()
    actual = settings.url_port(db.url)
    if actual is None or actual == db.expected_port:
        return None

    if db.app_env == settings.VERCEL:
        why = (
            f"{settings.VERCEL_DB_PORT} 는 transaction 모드 풀러다. "
            f"{settings.LOCAL_DB_PORT} 로 붙으면 서버리스에서 커넥션이 남아돈다."
        )
    else:
        why = f"{settings.LOCAL_DB_PORT} 는 로컬 직결 포트다."

    return (
        f"DATABASE_URL 포트가 {actual} 인데 APP_ENV={db.app_env} 의 기대값은 "
        f"{db.expected_port} 다. {why}\n"
        f"  붙는 곳: {displayable_url(db.url)}\n"
        "  일부러 그렇게 적었다면 그대로 둔다 — 이 값을 자동으로 고치지 않는다."
    )


# ==================================================
# 2. 엔진 — 프로세스마다 하나
# ==================================================
_engine: AsyncEngine | None = None


def build_engine(
    db: settings.DatabaseSettings | None = None, *, echo: bool = False
) -> AsyncEngine:
    """**새** 엔진을 만든다. 캐시하지 않는다.

    `create_async_engine()` 은 이 시점에 접속하지 않는다 — 첫 `connect()` 에서 붙는다.
    그래서 이 함수는 DB 가 없어도 성공하고, 검사가 실 DB 없이 돈다.

    포트가 어긋나면 경고만 남기고 **적힌 대로** 붙는다.
    """
    db = db or settings.database_settings()
    warning = port_warning(db)
    if warning:
        log.warning("%s", warning)

    kwargs = engine_kwargs(db, echo=echo)
    url = kwargs.pop("url")
    return create_async_engine(url, **kwargs)


def get_engine() -> AsyncEngine:
    """이 프로세스의 엔진. 처음 부를 때 만들고 그 뒤로는 같은 것을 돌려준다.

    ⚠️ **엔진은 첫 접속 때 이벤트 루프에 묶인다.** (생성 시점이 아니다 — `build_engine()`
    은 붙지 않는다.) 로컬 풀은 커넥션을 들고 있으므로, 루프를 새로 여는 코드
    (`asyncio.run()` 을 여러 번 부르는 스크립트)는 **그 루프 안에서**
    `await dispose_engine()` 으로 닫고 나온다.

    `forget_engine()` 으로 대신하면 안 된다 — 참조만 버리므로 서버 쪽 세션이 살아남고,
    나중에 GC 가 죽은 루프에서 커넥션을 닫으려다 `RuntimeError: Event loop is closed` 를
    stderr 로 흘린다. Supabase 처럼 커넥션 수가 한도인 상대에서는 그대로 누수다.
    """
    global _engine
    if _engine is None:
        _engine = build_engine()
    return _engine


def forget_engine() -> None:
    """캐시된 엔진 참조만 버린다. **커넥션을 정리하지는 않는다.**

    ⚠️ **아직 한 번도 붙지 않은 엔진에만 쓴다.** 검사가 환경을 바꿔 가며 엔진을 다시 만들 때가
    그 경우다. 이미 붙은 엔진에 쓰면 서버 세션이 남고, GC 가 죽은 루프에서 그것을 닫으려다
    `RuntimeError: Event loop is closed` 를 흘린다. 그때는 `await dispose_engine()` 이다.
    """
    global _engine
    _engine = None


async def dispose_engine() -> None:
    """엔진이 들고 있는 커넥션을 닫고 캐시를 비운다.

    FastAPI lifespan 의 종료 쪽에서 부를 자리다(S4 에서 연결한다).
    """
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None


# ==================================================
# 2-1. 동기 다리 — sync 호출 경로에서 async 엔진을 쓴다 (ADR-DS-0015 · S4)
# ==================================================
# 이 레포의 라우트 핸들러 **60개가 전부 `def`** 다(`async def` 가 0개). FastAPI 는 그것을
# anyio 워커 스레드풀에서 돌리므로, 저장소 계층은 **돌고 있는 이벤트 루프가 없는 스레드**에서
# 불린다. 그런데 이 모듈의 엔진은 async 전용이다. 그 사이를 잇는 것이 이 절이다.
#
# ## 왜 `asyncio.run()` 을 호출마다 부르지 않는가 — 실측으로 기각했다
#
# 되는 것처럼 보인다. 워커 스레드에는 루프가 없으니 `asyncio.run()` 이 성립하고,
# `scripts/check_db_connection.py` 도 그 모양이다. 그러나 **엔진을 캐시한 채로는 깨진다.**
# 로컬 Postgres 에 스레드 8 × 6라운드로 대 봤다 (2026-08-25):
#
# | 방법 | APP_ENV=local (정상 풀) | APP_ENV=vercel (NullPool) |
# |---|---|---|
# | 호출마다 `asyncio.run()` | **15/48 실패** · 88ms | 0 실패 · 71ms |
# | 전용 루프 (이 절) | 0 실패 · 17ms | 0 실패 · 33ms |
# | 전용 루프 (풀이 데워진 뒤) | 0 실패 · **13ms** | 0 실패 · 31ms |
#
# 실패는 `RuntimeError: Task ... got Future ... attached to a different loop` 다.
# 풀이 들고 있던 asyncpg 커넥션은 **그것을 만든 루프**에 묶여 있는데, `asyncio.run()` 은
# 매번 새 루프를 열고 끝나면 닫는다. 다음 호출이 죽은 루프의 커넥션을 꺼내 쓰는 순간 터진다.
# NullPool 이면 매번 새 커넥션이라 안 나지만, 그 대신 왕복마다 TCP+인증을 다시 한다.
#
# 즉 `asyncio.run()` 방식은 **ADR-DS-0003 의 로컬 전략(정상 풀)을 포기해야만** 성립한다.
# 전용 루프는 커넥션이 전부 한 루프에 묶이므로 두 전략 어느 쪽에서도 돈다. 그래서 이쪽이다.
#
# ## 이 루프는 게으르게 뜬다
#
# import 만으로는 스레드가 생기지 않는다. `run_sync()` 를 처음 부를 때 뜬다.
# `STORE_BACKEND=sqlite`(기본)면 아무도 부르지 않으므로 **배포본에는 스레드가 없다** —
# S4 커밋이 기본 경로에 아무것도 더하지 않아야 한다는 요구(ADR-DS-0011 §2)가 여기서 지켜진다.
#
# ## ⚠️ 아는 값 — 루프가 하나라 조회가 직렬화된다
#
# 모든 조회가 이 루프 하나를 지난다. `window()` 처럼 큰 결과를 읽는 조회는
# `fetchall()` 이 166,057개 행 객체를 **루프 스레드에서** 만드는 동안(실측 300~450ms)
# 다른 DB 조회를 붙잡아 둔다. 화면 하나가 병렬 XHR 을 던지면 그만큼 밀린다.
#
# 지금 이대로 둔다 — 갈아야 할 이유가 실측으로 서기 전에 손잡이를 늘리면, 무엇이 느린지
# 모르는 채 복잡도만 는다. SQLite 경로도 쓰기를 전역 자물쇠로 줄 세우고 있었다(krx_store:147).
# 다만 **S5 에서 기본값을 뒤집을 때 이 값을 다시 잰다** — 그때가 화면 10개가 동시에
# 이 루프를 쓰는 첫 순간이다.

# 다리를 통과하는 한 번의 조회에 거는 상한(초). 넘으면 화면이 매달리는 대신 실패한다.
BRIDGE_TIMEOUT_SECONDS = 30.0

_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None
_loop_lock = threading.Lock()
# ⚠️ 접는 중에는 루프를 **새로 세우지 않는다.** 이 표식이 없으면 종료 중에 도착한 조회가
#    새 루프를 세우고, 아직 `dispose_engine()` 이 끝나지 않아 살아 있는 **옛 엔진**을
#    그 새 루프에서 쓰게 된다 — 이 절이 없애려던 바로 그 고장이 종료 경로로 되살아난다.
_closing = False


class BridgeClosing(RuntimeError):
    """다리를 접는 중에 조회가 들어왔다. 재시도할 값이 없으므로 그대로 올려보낸다."""


def _bridge_loop() -> asyncio.AbstractEventLoop:
    """조회용 이벤트 루프. 없으면 데몬 스레드에 하나 띄우고 그 뒤로는 같은 것을 쓴다.

    자물쇠는 **루프를 두 개 만들지 않기 위한** 것이다. 두 개가 되면 커넥션이 두 루프로
    갈려 위 표의 실패가 그대로 돌아온다.
    """
    global _loop, _loop_thread
    with _loop_lock:
        if _closing:
            raise BridgeClosing("다리를 접는 중이라 새 조회를 받지 않는다.")
        if _loop is not None and not _loop.is_closed():
            return _loop

        loop = asyncio.new_event_loop()
        thread = threading.Thread(
            target=_run_forever, args=(loop,), name="db-bridge", daemon=True
        )
        thread.start()
        _loop, _loop_thread = loop, thread
        return loop


def _run_forever(loop: asyncio.AbstractEventLoop) -> None:
    """스레드의 본체. 루프를 그 스레드 것으로 못 박고 계속 돈다."""
    asyncio.set_event_loop(loop)
    loop.run_forever()


def run_sync(coro, timeout: float = BRIDGE_TIMEOUT_SECONDS):
    """코루틴 하나를 다리 위에서 돌리고 결과를 **동기적으로** 돌려준다.

    저장소 어댑터가 이 모듈을 쓰는 **유일한 통로**다. 예외는 삼키지 않고 그대로 올려보낸다 —
    접속 실패를 빈 결과로 바꾸면 DB 장애가 "그 날짜에 자료가 없음"으로 위장되고,
    화면은 축약본으로 조용히 강등된다. 무엇이 폴백이고 무엇이 고장인지는 부르는 쪽이 정한다.

    ⚠️ **이미 이벤트 루프 안에서는 부르지 않는다.** `async def` 안에서 부르면 그 루프를
    막아 버린다. 이 레포의 라우트 핸들러는 전부 `def` 라 해당 사항이 없지만,
    `async def` 를 새로 쓰는 날에는 이 함수 대신 코루틴을 그대로 `await` 한다.
    """
    running = None
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        pass                              # 정상 — 워커 스레드에는 루프가 없다
    # ⚠️ 넘겨받은 코루틴은 **어느 갈래로 빠져나가든 닫는다.** 안 닫으면 파이썬이
    #    "coroutine ... was never awaited" 를 stderr 로 흘린다. 종료 중 거절 경로에서
    #    실제로 그 경고가 났다.
    if running is not None:
        coro.close()
        raise RuntimeError(
            "run_sync() 를 이벤트 루프 안에서 불렀다. `async def` 안에서는 코루틴을 "
            "그대로 await 한다 — 여기서 기다리면 그 루프가 통째로 막힌다."
        )

    try:
        loop = _bridge_loop()
    except BaseException:
        coro.close()
        raise

    return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout)


def shutdown_bridge() -> None:
    """다리를 접는다 — 엔진을 그 루프 안에서 닫고, 루프를 멈추고, 스레드를 거둔다.

    ⚠️ **엔진 정리가 반드시 루프 안에서 일어나야 한다.** 밖에서 `dispose()` 를 부르면
    죽은 루프의 커넥션을 닫으려다 `RuntimeError: Event loop is closed` 가 stderr 로 샌다.

    ⚠️ **`_closing` 을 먼저 세우고 마지막에 내린다.** 그 사이에 도착한 조회는
    `BridgeClosing` 을 받는다. 표식 없이 참조만 비우면, 아직 살아 있는 옛 엔진을
    새 루프가 집어 가는 창이 열린다(수 ms 지만 종료 중 요청에서 실제로 열린다).

    다 접고 나면 표식을 내려 **다시 쓸 수 있는 상태로** 되돌린다 — 검사가 매번 접고
    다시 세우기 때문이다.
    """
    global _loop, _loop_thread, _closing
    with _loop_lock:
        if _closing:
            return                        # 이미 다른 스레드가 접고 있다
        loop, thread = _loop, _loop_thread
        if loop is None or loop.is_closed():
            _loop, _loop_thread = None, None
            return
        _closing = True

    try:
        try:
            asyncio.run_coroutine_threadsafe(dispose_engine(), loop).result(
                BRIDGE_TIMEOUT_SECONDS
            )
        except Exception as error:        # 이미 죽은 루프 등 — 접는 것을 막지는 않는다
            log.debug("다리를 접는 중 엔진 정리에 실패했다: %s", error)

        loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=BRIDGE_TIMEOUT_SECONDS)
        loop.close()
    finally:
        with _loop_lock:
            _loop, _loop_thread, _closing = None, None, False


# ==================================================
# 3. 커넥션 획득 — 읽기와 쓰기를 이름으로 가른다
# ==================================================
@asynccontextmanager
async def connect() -> AsyncIterator[AsyncConnection]:
    """읽기용 커넥션. 블록이 끝나면 **롤백하고** 닫는다.

    SQLAlchemy 2.0 은 `connect()` 블록을 커밋하지 않는다("commit as you go"). 조회만
    하는 경로에서는 그게 맞다 — 읽기가 실수로 쓰기를 확정하는 일이 없다.
    """
    async with get_engine().connect() as conn:
        yield conn


@asynccontextmanager
async def begin() -> AsyncIterator[AsyncConnection]:
    """쓰기용 커넥션. 블록이 정상 종료하면 **커밋**하고, 예외면 롤백한다.

    적재기(S3)와 쓰기 경로(S8)가 쓸 자리다. 읽기에는 `connect()` 를 쓴다.
    """
    async with get_engine().begin() as conn:
        yield conn


# ==================================================
# 4. 진단 — 무엇에 어떻게 붙었는지 한 줄로 말한다
# ==================================================
@dataclass(frozen=True)
class Probe:
    """`ping()` 의 결과. **예외를 던지는 대신 사실을 담아 돌려준다.**

    호출하는 쪽이 스크립트일 수도 화면일 수도 있어서, 어느 쪽이든 그대로 찍을 수 있는
    형태가 낫다. 실패했을 때 `hint` 에 **무엇을 해야 하는지**가 들어간다.
    """

    ok: bool
    app_env: str
    safe_url: str                    # 비밀번호는 이미 가려져 있다
    pool: str                        # 실제로 붙은 풀 클래스 이름
    # 캐시 두 손잡이가 **둘 다** 0인가. ⚠️ 이것이 참이라고 "준비구문이 안 생긴다"는 뜻은
    # 아니다 — asyncpg 0.30 은 여전히 이름을 붙인다. 안전한지는 이름이 트랜잭션을 넘어
    # 사는가로 갈리고, 그건 scripts/check_db_connection.py 의 부하 검사가 직접 잰다.
    caches_disabled: bool
    server_version: str = ""
    elapsed_ms: float = 0.0
    error: str = ""
    hint: str = ""

    def as_lines(self) -> list[str]:
        """사람이 읽는 형태. 스크립트와 로그가 같은 문장을 쓰게 한다."""
        head = "붙었다" if self.ok else "못 붙었다"
        lines = [
            f"[{head}] APP_ENV={self.app_env}",
            f"  주소   : {self.safe_url}",
            f"  풀     : {self.pool}",
            f"  캐시   : {'두 손잡이 다 꺼짐' if self.caches_disabled else '켜짐 (직결 전용)'}",
        ]
        if self.ok:
            lines.append(f"  서버   : {self.server_version}")
            lines.append(f"  왕복   : {self.elapsed_ms:.0f} ms")
        else:
            lines.append(f"  오류   : {self.error}")
            if self.hint:
                lines.extend(f"  → {part}" for part in self.hint.splitlines())
        return lines


# 준비구문 충돌을 알아보는 표식. 붙기는 붙었는데 **전략이 상대와 안 맞는** 경우라,
# "DB 가 떠 있나" 같은 안내를 하면 엉뚱한 곳을 파게 된다.
_PREPARED_STATEMENT_MARKERS = (
    "prepared statement",
    "DuplicatePreparedStatement",
    "InvalidSQLStatementName",
)


def displayable_url(url: str) -> str:
    """화면·로그에 찍어도 되는 형태. `mask_url()` 보다 **한 겹 더 보수적이다.**

    `settings.mask_url()` 의 계약은 "`scheme://user:pw@host` 에서 비밀번호만 가린다"이고,
    그 모양이 아니면 **원문을 그대로 돌려준다** — 호스트·포트를 읽을 수 있어야 하니 맞는 계약이다.
    문제는 그 "모양이 아닌 값"이 오타 난 접속 문자열이나, 실수로 붙여넣은 비밀번호일 수
    있다는 것이다. 진단 출력은 그대로 화면·로그·이슈에 복사되므로 여기서 한 번 더 막는다.

    가릴 수 있으면 가리고, 못 알아보겠으면 **값 대신 모양만** 말한다.
    """
    if "://" not in url:
        return f"(접속 문자열 형식이 아니다 — {len(url)}자, 값은 찍지 않는다)"

    # ⚠️ **쿼리는 통째로 접는다.** 비밀번호가 `user:pw@` 에만 오는 것이 아니다 —
    #    `?password=…` · `?sslpassword=…` 처럼 쿼리에 실리는 형태가 실제로 쓰인다.
    #    `mask_url()` 은 자격증명 자리만 보므로 그쪽은 그대로 통과시킨다.
    body, sep, query = url.partition("?")
    body = body.partition("#")[0]

    masked = settings.mask_url(body)
    if masked == body and "@" in body:
        # `@` 는 있는데 가려지지 않았다 = 비밀번호 없는 형태이거나, 우리가 모르는 형태다.
        # 앞의 스킴만 남기고 나머지는 접는다.
        scheme = body.partition("://")[0]
        masked = f"{scheme}://… (가릴 수 없는 형태라 접었다)"

    if sep and query:
        return f"{masked}?… (쿼리 {len(query)}자는 접었다 — 비밀번호가 실릴 수 있다)"
    return masked


def _url_hint(db: settings.DatabaseSettings, error: str) -> str | None:
    """접속 문자열 자체가 잘못된 경우의 안내. 해당 없으면 None.

    ⚠️ **가장 흔한 실수는 드라이버 접미사 누락이다.** Supabase 대시보드는
    `postgresql://...` 를 준다. 그대로 넣으면 SQLAlchemy 가 기본 드라이버(psycopg2)를
    찾다가 `ModuleNotFoundError` 로 죽는데, 그 메시지는 원인에서 한참 떨어져 있다.
    ADR-DS-0011 의 S6(Supabase 연결)에서 바로 이 순간을 만나게 된다.
    """
    if "psycopg2" in error or "psycopg" in error:
        return (
            "접속 문자열에 **드라이버가 빠졌다.** 이 레포는 asyncpg 를 쓴다.\n"
            "  postgresql://...        ← 이렇게 적으면 psycopg2 를 찾는다 (설치돼 있지 않다)\n"
            "  postgresql+asyncpg://...  ← 이렇게 적는다\n"
            f"  지금 값: {displayable_url(db.url)}"
        )
    if "Could not parse" in error or "ArgumentError" in error:
        # ⚠️ **여기서는 값을 찍지 않는다.** `mask_url()` 은 `scheme://user:pw@host` 모양을
        #    알아볼 때만 가릴 수 있고, 못 알아본 문자열은 **원문 그대로** 돌려준다
        #    (그게 그 함수의 계약이다 — 호스트·포트를 읽을 수 있어야 하므로).
        #    파싱이 실패했다는 것은 곧 그 모양이 아니라는 뜻이라, 찍으면 비밀번호가 샌다.
        return (
            "DATABASE_URL 을 접속 문자열로 읽지 못했다. **값은 여기 찍지 않는다**"
            "(가릴 수 있는 모양이 아니라서 그대로 새어 나간다).\n"
            "  형식: postgresql+asyncpg://<사용자>:<비밀번호>@<호스트>:<포트>/<DB이름>\n"
            f"  길이 {len(db.url)}자 · `://` {'있음' if '://' in db.url else '없음'} · "
            f"`@` {'있음' if '@' in db.url else '없음'}"
        )
    return None


def _failure_hint(db: settings.DatabaseSettings, error: str = "") -> str:
    """붙지 못했을 때 **다음에 무엇을 할지**. 막다른 길로 만들지 않는다.

    원인마다 할 일이 다르다. 준비구문 충돌은 "DB 가 안 떴다"와 증상이 전혀 다른데,
    안내를 뭉뚱그리면 멀쩡한 컨테이너를 재시작하며 시간을 버리게 된다.
    """
    if any(marker in error for marker in _PREPARED_STATEMENT_MARKERS):
        head = (
            "붙기는 붙었고, **준비구문이 상대와 안 맞는다.** 앞단이 transaction 모드 풀러"
            "(pgbouncer·Supavisor)다.\n"
            f"  지금 값: APP_ENV={db.app_env} · 캐시={dict(db.connect_args)} · "
            f"이름 유일화={'켬' if db.unique_statement_names else '끔'}"
        )
        # 세 손잡이 중 무엇이 빠졌는지 **보고 말한다.** 이미 다 켜 놓은 사람에게
        # "켜라"고 하면 그게 막다른 길이다.
        missing = []
        if db.connect_args.get("statement_cache_size") != 0:
            missing.append("statement_cache_size=0")
        if db.connect_args.get("prepared_statement_cache_size") != 0:
            missing.append("prepared_statement_cache_size=0")
        if not db.unique_statement_names:
            missing.append("준비구문 이름 유일화")
        if not db.use_null_pool:
            missing.append("NullPool")

        if missing:
            return (
                f"{head}\n"
                f"  빠진 것: {' · '.join(missing)}\n"
                f"  APP_ENV={settings.VERCEL} 로 두면 넷이 한 벌로 켜진다."
            )
        # 전략은 이미 맞다 — 그렇다면 상대 쪽에 남은 잔여물이다.
        return (
            f"{head}\n"
            "  전략은 이미 맞게 켜져 있다. 그렇다면 **풀러 쪽에 남은 준비구문**이다 —\n"
            "  이전에 틀린 설정으로 붙은 커넥션이 서버에 이름을 남겨 두면 그 뒤로 계속 부딪힌다.\n"
            "  로컬 pgbouncer 라면:  RECONNECT;  (관리 콘솔) 또는 컨테이너 재시작\n"
            "  Supabase 라면 잠시 뒤 다시 시도한다(풀러가 서버 커넥션을 회수하면 사라진다)."
        )
    if db.app_env == settings.VERCEL:
        return (
            "Vercel 프로젝트 설정 → Environment Variables 의 DATABASE_URL 을 확인한다.\n"
            f"포트는 {settings.VERCEL_DB_PORT}(transaction 모드 풀러)여야 한다."
        )
    return (
        "로컬 DB 가 떠 있는지 본다:  docker compose --profile local-db up -d\n"
        "이미 떠 있다면 포트 충돌을 본다:  docker compose --profile local-db ps"
    )


def unreachable(error: BaseException, *, what: str,
                extra: Sequence[str] = ()) -> RuntimeError:
    """접속 실패를 **처방을 붙여** 다시 던질 예외로 만든다 (ADR-DS-0018 · ADR-DS-0019).

    ⚠️ **삼키는 것이 아니다.** 예외는 그대로 올라가고 바뀌는 것은 메시지뿐이다.
    빈 결과로 바꾸면 DB 장애가 "자료 없음" 으로 위장돼 화면이 조용히 강등된다.

    ⚠️ **여기 한 곳에만 있다.** 처음에는 `krx_pg._fetch()` 안에 있었는데, 자료 보관함이
    같은 처방을 필요로 하면서 두 벌이 될 뻔했다. 공통은 이쪽이 만들고 **되돌리는 법만**
    부르는 쪽이 `extra` 로 얹는다 — 그것은 저장소마다 다르다(시세는 SQLite 로 되돌아가고
    보관함은 되돌아갈 곳이 없다).

    ⚠️ 이 함수는 위층을 부르지 않는다. `extra` 가 문자열인 것이 그 경계다
    (`tests/test_db.py` §5 가 방향을 검사한다).
    """
    target = settings.database_settings()
    lines = [f"{what}에 못 붙었다: {error}", f"  붙는 곳: {target.safe_url()}"]
    if target.app_env == settings.VERCEL:
        # 배포본에는 `docker compose` 가 없다. 따를 수 없는 처방을 띄우면 시간만 버린다.
        lines.append("  Vercel 프로젝트 설정 → Environment Variables 의 DATABASE_URL 을 확인한다.")
        lines.append(f"  포트는 {settings.VERCEL_DB_PORT}(transaction 모드 풀러)여야 한다.")
    else:
        lines.append("  DB 를 띄운다      : docker compose --profile local-db up -d")
        lines.append("  호스트 셸이라면   : DATABASE_URL 의 @db:5432 를 @localhost:5432 로 바꾼다")
    lines += [f"  {line}" for line in extra]
    return RuntimeError("\n".join(lines))


async def ping(timeout: float = PING_TIMEOUT_SECONDS) -> Probe:
    """실제로 붙어 보고 무엇에 붙었는지 돌려준다. **예외를 밖으로 내지 않는다.**

    ADR-DS-0003 은 커넥션 전략을 *표명*만 하고 "값이 실제로 맞는지는 접속 코드를 쓰는
    순간 처음 검증된다"고 적어 두었다. 이 함수가 그 검증을 아무 때나 다시 할 수 있게 한다.

    엔진은 **캐시된 것을 쓰지 않는다.** 진단이 프로세스의 풀 상태를 바꾸면 안 되고,
    실패한 엔진이 캐시에 남아 다음 호출까지 오염시키면 진단의 뜻이 사라진다.

    ⚠️ **설정을 읽는 것과 엔진을 만드는 것까지 `try` 안이다.** 그 둘도 실패할 수 있다 —
    `DATABASE_URL` 에 `+asyncpg` 를 빠뜨리면(Supabase 대시보드가 주는 문자열이 그렇다)
    엔진 생성에서 `ModuleNotFoundError: psycopg2` 가 나고, `APP_ENV=vercel` 인데
    `DATABASE_URL` 이 비면 설정 읽기에서 `RuntimeError` 가 난다. 진단 도구가 진단 대신
    트레이스백으로 죽으면 존재 이유가 없다.
    """
    started = time.perf_counter()

    # 설정 읽기 자체가 실패할 수 있다. 이 단계에서는 아직 db 도 engine 도 없다.
    try:
        db = settings.database_settings()
    except Exception as exc:  # noqa: BLE001
        return Probe(
            ok=False,
            app_env=settings.env("APP_ENV") or "(미설정)",
            safe_url="(설정을 읽지 못했다)",
            pool="(없음)",
            caches_disabled=False,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            error=f"{type(exc).__name__}: {exc}",
            hint="APP_ENV 와 DATABASE_URL 을 확인한다. 위 메시지가 무엇을 넣어야 하는지 말한다.",
        )

    disabled = (
        db.connect_args.get("statement_cache_size") == 0
        and db.connect_args.get("prepared_statement_cache_size") == 0
    )

    engine = None
    try:
        engine = build_engine(db)
        async with asyncio.timeout(timeout):
            async with engine.connect() as conn:
                version = (await conn.execute(text("SELECT version()"))).scalar_one()
    except Exception as exc:  # noqa: BLE001 — 진단이므로 종류를 가리지 않고 문장으로 담는다
        message = f"{type(exc).__name__}: {exc}"
        return Probe(
            ok=False,
            app_env=db.app_env,
            safe_url=displayable_url(db.url),
            pool=type(engine.pool).__name__ if engine is not None else "(엔진 생성 실패)",
            caches_disabled=disabled,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            error=message,
            hint=_url_hint(db, message) or _failure_hint(db, message),
        )
    finally:
        if engine is not None:
            await engine.dispose()

    return Probe(
        ok=True,
        app_env=db.app_env,
        safe_url=displayable_url(db.url),
        pool=type(engine.pool).__name__,
        caches_disabled=disabled,
        server_version=str(version),
        elapsed_ms=(time.perf_counter() - started) * 1000,
    )
