"""실행 환경 설정을 한 곳에서 읽는다 (ADR-DS-0003).

`paths.py` 가 **경로**의 기준점인 것과 같은 자리다. 이 모듈은 **환경**의 기준점이다.
새 코드는 `os.getenv` 를 직접 부르지 않고 여기를 거친다.

## `secrets.py` 와 무엇이 다른가

이름이 둘 다 "설정"처럼 보이지만 **묻는 질문이 다르다.**

| | `secrets.py` | `settings.py` (이 파일) |
|---|---|---|
| 무엇을 읽나 | 외부 API 인증키 (KRX·DART·FRED…) | 이 프로세스가 **어디서 도는가**와 그로부터 갈리는 값 |
| 어디서 찾나 | 환경변수 → `.env` → `.key` | **환경변수만** |
| 왜 그 순서인가 | 강의 실습이 키를 파일에 두고 쓴다 | 실행 환경은 파일로 알 수 없다 — 같은 파일이 로컬과 배포본에 함께 실린다 |
| 없으면 | 그 API 만 503, 서버는 뜬다 | 기본값으로 떨어지거나 **즉시** 실패한다 |
| 값의 성격 | 전부 비밀 (`mask()` 로 가림) | 대부분 비밀이 아니다 — `DATABASE_URL` 만 예외 |

`DATABASE_URL` 은 비밀번호를 품고 있지만 여기 있다. **접속 전략의 일부**이고,
`.key` 처럼 파일로 흘리면 안 되기 때문이다. 로그·화면에 실을 때는 `safe_url()` 을 쓴다.

## 왜 상수가 아니라 함수인가

`paths.py` 는 모듈 상수(`PROJECT_ROOT` 등)다. 경로는 **파일 위치가 정하므로** import 시점에
확정되고 그 뒤로 변할 일이 없다. 환경은 다르다 — 프로세스마다 다르고, 테스트가
`monkeypatch.setenv` 로 바꿔 가며 분기를 확인한다. 상수로 두면 **첫 import 에 얼어붙어**
테스트가 두 갈래 중 한쪽만 보게 된다. 그래서 읽을 때마다 환경을 다시 본다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

# ==================================================
# 1. APP_ENV — 실행 환경 어휘
# ==================================================
# 둘뿐이다. staging 이 생기면 ADR-DS-0003 을 개정하고 여기에 더한다.
LOCAL = "local"
VERCEL = "vercel"
APP_ENVS: tuple[str, ...] = (LOCAL, VERCEL)

# 플랫폼이 항상 넣어 주는 표식. `APP_ENV` 를 빼먹어도 배포본을 배포본으로 알아본다.
# (원본 data-service 의 app/services/research/stages.py:287 이 같은 두 변수를 본다.
#  그 파일은 1차 범위 밖이라 옮기지 않았다.)
VERCEL_MARKERS: tuple[str, ...] = ("VERCEL", "VERCEL_ENV")

# ==================================================
# 2. 커넥션 전략 상수 (ADR-DS-0003 §4)
# ==================================================
# 로컬은 Postgres 에 직접 붙는다. 배포본은 Supabase 의 **transaction 모드 풀러**를 거친다.
# 포트가 곧 모드다 — 6543 으로 붙는 순간 prepared statement 를 쓸 수 없다.
LOCAL_DB_PORT = 5432
VERCEL_DB_PORT = 6543

# transaction 모드에서 준비구문 **캐시**를 끄는 두 손잡이. 둘 다 꺼야 한다.
#   statement_cache_size            asyncpg 자신의 LRU 캐시
#   prepared_statement_cache_size   SQLAlchemy asyncpg 방언이 그 위에 하나 더 두는 캐시
# 하나만 끄면 조합에 따라 결과가 전혀 다르고, asyncpg 버전에 따라서도 갈린다
# (실측표는 ADR-DS-0003 rev.2). 따로 켜고 끄지 않는다.
#
# ⚠️ **이 둘만으로는 부족하다.** 캐시를 꺼도 이름은 계속 붙는다 —
#    세 번째 손잡이는 아래 UNIQUE_STATEMENT_NAMES_ON_VERCEL 이다.
#
# ⚠️ 값은 파이썬 `int` 여야 한다. URL 쿼리(`?statement_cache_size=0`)로 옮기면
#    문자열 `"0"` 으로 도착해 asyncpg 가 `"0" < 0` 을 시도하다 죽는다.
VERCEL_CONNECT_ARGS: Mapping[str, int] = MappingProxyType({
    "statement_cache_size": 0,
    "prepared_statement_cache_size": 0,
})

# 로컬은 아무것도 끄지 않는다 — 직결이라 prepared statement 가 정상 동작하고, 그게 더 빠르다.
LOCAL_CONNECT_ARGS: Mapping[str, int] = MappingProxyType({})

# ⭐ 세 번째 손잡이 — **이름**이다 (ADR-DS-0003 rev.2, 2026-08-23 실측).
#
# 캐시를 둘 다 꺼도 asyncpg 0.30 은 준비구문에 **이름을 붙인다.** 그 이름은
# `__asyncpg_stmt_1__` 처럼 **커넥션마다 1부터 다시 세는 카운터**라, 서버리스에서 호출마다
# 새 커넥션이 열리면 매번 같은 이름을 다시 쓴다. transaction 모드 풀러 뒤에서는 DEALLOCATE 가
# 다른 물리 커넥션으로 갈 수 있어 이름이 남고, 그 뒤로는 **첫 질의부터** 전부 죽는다
# (`select pg_catalog.version()` 에서 `DuplicatePreparedStatementError`).
#
# ⚠️ 이 고장은 **갓 띄운 풀러에서는 재현되지 않는다.** 잔여물이 쌓인 뒤에만 나타나므로,
#    깨끗한 상대에 한 번 대 보고 "괜찮다"고 결론내면 거짓 음성을 얻는다.
#    실제로 rev.2 초안이 그 함정에 빠졌다 — 그래서 이 상수에 그 사실을 적어 둔다.
#
# 값이 아니라 **사실**만 둔다. 이름을 만드는 함수는 접속 계층(`common/db.py`)이 준다 —
# 이 모듈은 SQLAlchemy 를 몰라야 하고(§6), 무엇으로 이름을 만들지는 드라이버 사정이다.
UNIQUE_STATEMENT_NAMES_ON_VERCEL = True

# compose.yaml:22 의 기본값과 같은 문자열. 로컬은 이것만으로 뜬다.
DEFAULT_LOCAL_DATABASE_URL = "postgresql+asyncpg://postgres:postgres@db:5432/data_service"

# ==================================================
# 3. STORE_BACKEND — 시세를 어느 저장소에서 읽나 (ADR-DS-0015 · 전환 S4)
# ==================================================
# `DATABASE_URL` 이 **어디에 붙을지**를 말한다면 이 값은 **읽기 경로가 그것을 쓰는지**를 말한다.
# 둘은 따로다 — 적재기(`scripts/load_pg.py`)는 S3 부터 Postgres 에 붙어 있었지만
# 화면은 계속 SQLite 를 읽었다. 그 상태를 값 하나로 표현한 것이 이 스위치다.
SQLITE = "sqlite"
POSTGRES = "postgres"
STORE_BACKENDS: tuple[str, ...] = (SQLITE, POSTGRES)

# ⭐ **이제 두 환경이 같은 값이다 — 그것이 S6 의 정의다** (ADR-DS-0011 §1 · ADR-DS-0021).
#
#   로컬  → `postgres`   S5 가 뒤집었다 (ADR-DS-0018)
#   배포본 → `postgres`   S6 가 뒤집었다. Supabase 에 core 350종목이 서 있다
#
# ⚠️ **배포본에는 `DATABASE_URL` 이 반드시 있어야 한다.** 거기서 `database_url()` 은
#    기본값으로 대신하지 않고 **예외를 던진다**(§2). 그 환경변수가 없는 채로 이 기본값이
#    `postgres` 면 **화면 HTML 은 뜨고 시세 API 가 500** 이다 (실측 — ADR-DS-0021 §8).
#    겉보기로는 "화면은 열리는데 비어 있다" 라서 500 페이지보다 알아채기 어렵다.
#    S5 까지 이 값이 `sqlite` 였던 이유가 그것이고,
#    **S6 이 먼저 한 일이 환경변수를 채운 것**이다. 순서가 뜻을 가진다.
#
# ⚠️ **값이 같아졌다고 상수를 하나로 합치지 않는다.** 배포는 push 가 곧 배포라
#    (GitLab→Vercel) 배포본만 되돌려야 하는 순간이 온다. 둘로 두면 그 되돌림이
#    한 줄이고, 하나로 합치면 로컬까지 함께 끌려 내려간다. **되돌리는 단위가 값이다.**
#
# ⚠️ 그래도 가장 빠른 되돌림은 여전히 환경변수 한 줄이다 — `STORE_BACKEND=sqlite`.
#    사람이 적은 값이 언제나 이긴다(`app_env()` 와 같은 순서).
DEFAULT_STORE_BACKEND_LOCAL = POSTGRES
DEFAULT_STORE_BACKEND_VERCEL = POSTGRES


def default_store_backend() -> str:
    """`STORE_BACKEND` 가 없을 때 쓸 값. 환경이 정한다 — 위 표 참조."""
    return DEFAULT_STORE_BACKEND_VERCEL if is_vercel() else DEFAULT_STORE_BACKEND_LOCAL


def store_backend() -> str:
    """시세 읽기 경로가 쓸 저장소. 기본은 **양쪽 환경 다 `postgres`** 다 (S6 이후).

    **`app_env()` 와 같은 자리에 같은 모양으로 둔다** — 어휘 밖 값이면 예외를 던진다.
    오타(`postgre`·`pg`)를 조용히 기본값으로 떨어뜨리면 "스위치를 켰다고 믿었는데
    실은 SQLite 를 재고 있었다"가 된다. 그 거짓 음성이 이 전환에서 가장 비싼 실수다
    (ADR-DS-0011 근거 — "깨끗한 상대에 한 번 대 보는 것은 검증이 아니다").

    ⚠️ **상수가 아니라 함수인 것이 뜻을 가진다.** 모듈 상수로 두면 import 시점에 얼어붙어
    검사가 스위치를 뒤집을 방법이 없어진다. `krx_store.DB_PATH` 가 실제로 그렇게 굳어 있어
    `KRX_DB_PATH` 를 `monkeypatch.setenv` 해도 아무 효과가 없다 — 그 함정을 되풀이하지 않는다.
    """
    raw = env("STORE_BACKEND")
    if not raw:
        return default_store_backend()

    value = raw.lower()
    if value not in STORE_BACKENDS:
        # 막다른 길로 만들지 않는다 — 무엇을 해야 하는지까지 알려준다.
        raise ValueError(
            f"STORE_BACKEND 값 '{raw}' 을 모른다. 쓸 수 있는 값은 {', '.join(STORE_BACKENDS)} 다.\n"
            f"  SQLite 로 읽기   : STORE_BACKEND={SQLITE} (되돌릴 때 쓰는 값)\n"
            f"  Postgres 로 읽기 : STORE_BACKEND={POSTGRES} "
            f"(양쪽 환경의 기본값 — DATABASE_URL 과 붙을 수 있는 DB 가 함께 필요하다)"
        )
    return value


def uses_postgres_store() -> bool:
    """읽기 경로가 Postgres 를 보는가. 분기 조건을 한 낱말로 읽히게 한다."""
    return store_backend() == POSTGRES


# ==================================================
# 3-1. REFRESH_API — 화면에서 갱신을 실행할 수 있게 둘 것인가 (ADR-DS-0017)
# ==================================================
# `invoke refresh` 와 **같은 사슬**을 화면 버튼이 부른다. 그 버튼은 외부 API 를 부르고
# 파일을 고치므로, 켜고 끄는 손잡이가 하나 있어야 한다.
#
# ⚠️ **환경 분기와는 다른 축이다.** 배포본에서 못 도는 것은 이 값과 무관하게 능력의 문제고
#    (읽기 전용 파일시스템 · `scripts/` 부재), 이 값은 **돌 수 있는 곳에서 일부러 막는**
#    손잡이다. README §5 가 `--host 0.0.0.0` 을 안내하므로 LAN 에 열어 두는 경우가 실제로 있다.
ON = "on"
OFF = "off"
REFRESH_API_VALUES: tuple[str, ...] = (ON, OFF)

# 기본은 켜짐이다. 로컬 개발 도구이고, 꺼 두면 "버튼이 왜 없지" 를 먼저 만나기 때문이다.
DEFAULT_REFRESH_API = ON


def refresh_api() -> str:
    """화면 갱신 API 를 열어 둘 것인가. `on`(기본) 또는 `off`.

    `store_backend()` · `app_env()` 와 **같은 모양**이다 — 어휘 밖 값이면 예외다.
    `REFRESH_API=false` 를 조용히 기본값(`on`)으로 떨어뜨리면 "껐다고 믿었는데 열려 있는"
    상태가 되는데, 이 손잡이에서 그 거짓 음성은 방향이 나쁜 쪽이다.
    """
    raw = env("REFRESH_API")
    if not raw:
        return DEFAULT_REFRESH_API

    value = raw.lower()
    if value not in REFRESH_API_VALUES:
        raise ValueError(
            f"REFRESH_API 값 '{raw}' 을 모른다. 쓸 수 있는 값은 "
            f"{', '.join(REFRESH_API_VALUES)} 다.\n"
            f"  화면에서 갱신을 실행한다 : REFRESH_API={ON} (이 값이 기본이라 지워도 같다)\n"
            f"  실행 경로를 닫는다       : REFRESH_API={OFF} (상태 조회는 그대로 열려 있다)"
        )
    return value


def refresh_api_enabled() -> bool:
    """화면에서 갱신을 **실행**할 수 있는가. 상태 조회는 이 값과 무관하게 열려 있다."""
    return refresh_api() == ON


# ==================================================
# 3-2. COLLECT_API — 자동 수집을 실행할 수 있게 둘 것인가 (ADR-DS-0020)
# ==================================================
# `REFRESH_API` 와 **같은 모양이고 같은 이유**다 — 외부 API(DART)를 부르고 표에 쓰는
# 경로이므로 돌 수 있는 곳에서 일부러 막는 손잡이가 하나 있어야 한다.
#
# ⚠️ **호출 예산은 여기 두지 않는다.** `settings` 는 「환경이 정하는 것」의 집이고
#    (ADR-DS-0003) 예산은 환경이 아니라 이 서비스의 정책이다
#    (원본의 `app/services/dart_collector.py` — 옮기지 않았다).
#    환경변수로 뚫어 두면 "왜 오늘 한도가 찼지" 의
#    답이 셸 히스토리에 숨는다. 회차 단위 조정은 `--budget` 이다.
COLLECT_API_VALUES: tuple[str, ...] = (ON, OFF)
DEFAULT_COLLECT_API = ON


def collect_api() -> str:
    """자동 수집 실행 경로를 열어 둘 것인가. `on`(기본) 또는 `off`.

    ⚠️ 어휘 밖 값이면 **예외**다 — `refresh_api()`·`store_backend()`·`app_env()` 와 같다.
    `COLLECT_API=false` 를 조용히 `on` 으로 떨어뜨리면 "껐다고 믿었는데 열려 있는" 상태가
    되고, 이 손잡이에서 그 거짓 음성은 **외부 API 를 태우는 쪽**이라 방향이 나쁘다.
    """
    raw = env("COLLECT_API")
    if not raw:
        return DEFAULT_COLLECT_API

    value = raw.lower()
    if value not in COLLECT_API_VALUES:
        raise ValueError(
            f"COLLECT_API 값 '{raw}' 을 모른다. 쓸 수 있는 값은 "
            f"{', '.join(COLLECT_API_VALUES)} 다.\n"
            f"  수집을 실행한다   : COLLECT_API={ON} (이 값이 기본이라 지워도 같다)\n"
            f"  실행 경로를 닫는다 : COLLECT_API={OFF} (상태 조회는 그대로 열려 있다)"
        )
    return value


def collect_api_enabled() -> bool:
    """자동 수집을 **실행**할 수 있는가. 상태 조회는 이 값과 무관하게 열려 있다."""
    return collect_api() == ON


def env(name: str, default: str = "") -> str:
    """환경변수 한 개를 읽는다 — **새 코드가 환경을 만지는 유일한 통로.**

    앞뒤 공백을 떼고 돌려준다. 빈 문자열은 "없음"과 같게 본다 —
    compose 의 `${KRX_API_KEY:-}` 처럼 **키를 빈 값으로 넣는 구성**이 흔해서,
    "정의는 됐지만 비어 있다"를 따로 다루면 호출자마다 조건이 갈린다.

    인증키는 여기가 아니라 `common/secrets.py` 로 읽는다 (파일 폴백이 필요하다).
    """
    return os.getenv(name, default).strip()


def subprocess_env(**overrides: str) -> dict[str, str]:
    """자식 프로세스에 넘길 환경 한 벌 (ADR-DS-0017).

    **왜 여기 있나.** 이 모듈이 환경을 읽는 유일한 통로이고
    (`tests/test_settings.py` §5 가 그 목록을 얼려 둔다), 환경을 **자식에게 건네는 것**도
    같은 축의 일이다. 갱신 실행기가 `os.environ.copy()` 를 직접 부르면 통로가 하나 더 생긴다.

    - 지금 프로세스의 환경을 통째로 물려준다. 인증키(`KRX_API_KEY` 등)가 그대로 따라가야
      자식 스크립트가 뜬다 — 골라 담으면 키 하나가 빠졌을 때 자식이 **401 로** 죽는다.
    - `overrides` 는 위에 얹는다. **빈 값은 얹지 않는다** — 빈 문자열로 덮으면
      "정의는 됐는데 비어 있다"가 되어 `env()` 의 규약(빈 값 = 없음)과 어긋난다.
    """
    child = dict(os.environ)
    child.update({name: value for name, value in overrides.items() if value})
    return child


def app_env() -> str:
    """이 프로세스의 실행 환경. `local` 또는 `vercel`.

    **읽는 순서가 곧 결정이다** (ADR-DS-0003 §3).

    1. `APP_ENV` 가 명시돼 있으면 그것을 쓴다 — 사람이 적은 값이 언제나 이긴다.
    2. 없으면 `VERCEL`·`VERCEL_ENV` 를 보고 배포본인지 스스로 알아본다.
    3. 그래도 아니면 `local`.

    2번이 없으면 배포본에서 `APP_ENV` 를 한 번 빠뜨리는 것만으로 **로컬 전략으로 조용히**
    뜬다. 그 결과는 즉사가 아니라 산발적 실패라 원인을 찾기가 매우 어렵다.

    어휘 밖 값이면 예외를 던진다. 오타(`production`·`prod`)를 조용히 `local` 로
    떨어뜨리면 1번과 2번을 모두 무력화한다.
    """
    explicit = env("APP_ENV")
    if explicit:
        value = explicit.lower()
        if value not in APP_ENVS:
            # 막다른 길로 만들지 않는다 — 무엇을 해야 하는지까지 알려준다.
            raise ValueError(
                f"APP_ENV 값 '{explicit}' 을 모른다. 쓸 수 있는 값은 {', '.join(APP_ENVS)} 다.\n"
                f"  로컬 개발·도커  : APP_ENV={LOCAL}\n"
                f"  Vercel 배포본   : APP_ENV={VERCEL}\n"
                "APP_ENV 를 아예 지우면 VERCEL 환경변수를 보고 자동으로 고른다."
            )
        return value

    if any(env(marker) for marker in VERCEL_MARKERS):
        return VERCEL
    return LOCAL


def is_vercel() -> bool:
    """배포본(Vercel 서버리스)인가."""
    return app_env() == VERCEL


@dataclass(frozen=True)
class DatabaseSettings:
    """DB 접속에 필요한 값 묶음. **엔진을 만들지는 않는다** (ADR-DS-0003 §6).

    이 모듈은 SQLAlchemy 를 import 하지 않는다. 풀 클래스를 직접 들고 있으면
    설정을 읽는 것만으로 무거운 의존성이 딸려 오고, 아직 없는 계층에 이 파일이 묶인다.
    그래서 `use_null_pool` 이라는 **사실**만 내놓고 해석은 접속 코드에 맡긴다.
    """

    app_env: str
    url: str
    expected_port: int                  # 이 환경에서 정상인 포트. 검증·안내용이다
    use_null_pool: bool                 # True 면 접속 코드가 NullPool 을 쓴다
    connect_args: Mapping[str, int]     # asyncpg 로 그대로 넘어갈 값
    # True 면 접속 코드가 준비구문 이름을 **커넥션마다 겹치지 않게** 만들어 준다.
    # 함수가 아니라 사실만 둔다 — 위 UNIQUE_STATEMENT_NAMES_ON_VERCEL 주석 참조.
    unique_statement_names: bool = False

    def safe_url(self) -> str:
        """로그·화면에 실어도 되는 형태. **비밀번호를 가린다.**

        `/health` 같은 곳에 접속 문자열을 그대로 내보내는 사고가 흔하다.
        `postgresql+asyncpg://postgres:postgres@db:5432/x` → `...://postgres:***@db:5432/x`
        """
        return mask_url(self.url)


def mask_url(url: str) -> str:
    """접속 문자열에서 비밀번호만 `***` 로 바꾼다. 나머지는 그대로 둔다.

    호스트·포트·DB 이름은 남겨야 "어디에 붙으려 했는가"를 로그로 읽을 수 있다.
    """
    if "://" not in url:
        return url
    scheme, _, rest = url.partition("://")
    if "@" not in rest:                  # 자격증명이 없는 형태 (host:port/db)
        return url
    credentials, _, host_part = rest.rpartition("@")
    user, sep, _password = credentials.partition(":")
    if not sep:                          # 비밀번호 없이 사용자만 있는 형태
        return url
    return f"{scheme}://{user}:***@{host_part}"


def database_url() -> str:
    """`DATABASE_URL`. 로컬에서만 기본값으로 떨어진다.

    배포본에서 이 값이 비면 **기본값으로 때우지 않는다.** 로컬 기본값은 `@db` 라는
    compose 안에서만 뜻이 있는 호스트라, 배포본이 그걸 물고 뜨면 정체 모를
    이름 해석 실패가 된다. 무엇을 해야 하는지 말하고 멈추는 편이 낫다.
    """
    url = env("DATABASE_URL")
    if url:
        return url

    if is_vercel():
        raise RuntimeError(
            "DATABASE_URL 이 없다. 배포본에서는 기본값으로 대신하지 않는다.\n"
            "  Vercel 프로젝트 설정 → Environment Variables 에 DATABASE_URL 을 넣는다.\n"
            f"  포트는 {VERCEL_DB_PORT} (transaction 모드 풀러)여야 한다 — "
            f"{LOCAL_DB_PORT} 로 적으면 서버리스에서 커넥션이 남아돈다."
        )
    return DEFAULT_LOCAL_DATABASE_URL


def database_settings() -> DatabaseSettings:
    """지금 환경에 맞는 DB 설정 묶음 (ADR-DS-0003 §4).

    | | `local` | `vercel` |
    |---|---|---|
    | 포트 | 5432 직결 | 6543 transaction 풀러 |
    | 풀 | 정상 풀 | `NullPool` |
    | `statement_cache_size` | 그대로 | `0` |
    | `prepared_statement_cache_size` | 그대로 | `0` |
    | 준비구문 이름 | 기본(카운터) | **커넥션마다 유일** |

    ⚠️ **배포본 쪽은 한 벌이다.** 하나만 빠져도 prepared statement 충돌이 난다.
    ⚠️ 특히 **이름 손잡이를 빼면 갓 띄운 풀러에서는 멀쩡하다가**, 잔여물이 쌓인 뒤
    갑자기 전부 실패한다. 실측표는 ADR-DS-0003 rev.2 에 있다.
    """
    current = app_env()
    if current == VERCEL:
        return DatabaseSettings(
            app_env=current,
            url=database_url(),
            expected_port=VERCEL_DB_PORT,
            use_null_pool=True,           # 서버리스는 호출 사이에 얼었다 녹는다. 풀을 들고 있을 수 없다
            connect_args=VERCEL_CONNECT_ARGS,
            unique_statement_names=UNIQUE_STATEMENT_NAMES_ON_VERCEL,
        )
    return DatabaseSettings(
        app_env=current,
        url=database_url(),
        expected_port=LOCAL_DB_PORT,
        use_null_pool=False,              # 직결이라 풀이 그대로 이득이다
        connect_args=LOCAL_CONNECT_ARGS,
        unique_statement_names=False,     # 직결에는 이름 충돌이 없다. 기본 이름이 더 싸다
    )


def url_port(url: str) -> int | None:
    """접속 문자열에서 포트만 뽑는다. 못 찾으면 None.

    `database_settings().expected_port` 와 맞춰 보는 용도다. **자동으로 고치지 않는다** —
    포트를 말없이 바꾸면 사용자가 적은 값과 실제로 붙는 곳이 갈린다.
    """
    if "://" not in url:
        return None
    _, _, rest = url.partition("://")
    host_part = rest.rpartition("@")[2] if "@" in rest else rest
    # 뒤에 붙는 것을 순서대로 떼어낸다. **셋 다 떼야 한다** — `/dbname` 만 떼면
    # `host:6543?sslmode=require` 같은 형태에서 포트를 못 읽고 `None` 을 돌려주는데,
    # 그러면 port_warning() 이 **조용히 꺼져** 포트가 틀려도 아무 말을 하지 않는다.
    host_part = host_part.split("/", 1)[0]      # /dbname
    host_part = host_part.split("?", 1)[0]      # ?sslmode=...
    host_part = host_part.split("#", 1)[0]      # #fragment
    if host_part.startswith("["):               # IPv6 리터럴 [::1]:5432
        host_part = host_part.partition("]")[2]
    _, sep, port = host_part.rpartition(":")
    if not sep or not port.isdigit():
        return None
    return int(port)
