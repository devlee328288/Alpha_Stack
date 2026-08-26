"""`robots.txt` 를 요청 직전에 확인한다 — 그리고 **못 받았을 때를 옳게 다룬다.**

이 파일에서 가장 중요한 것은 판정 자체가 아니라 **실패했을 때의 방향**이다.

RFC 9309 가 정한 것
-------------------
| 응답 | 뜻 | 우리가 할 일 |
|---|---|---|
| 200 | 규칙을 받았다 | 규칙대로 판정한다 |
| **4xx** | robots.txt 가 **없다** | **전면 허용** — 규칙이 없으니 막을 근거도 없다 |
| **5xx** | 서버가 **답을 못 준다** | **전면 차단** — RFC 는 "MUST assume complete disallow" |
| 네트워크 오류 | 닿지도 못했다 | **전면 차단** (5xx 와 같다) |

**이 둘을 거꾸로 짜는 일이 흔하다.** "못 받았으면 없는 거니까 허용" 이라고 뭉뚱그리면
상대 서버가 잠깐 아픈 사이에 우리가 그 위를 긁는다. 반대로 4xx 를 차단으로 보면
robots.txt 를 두지 않은 사이트를 영영 못 읽는다. **두 실패는 뜻이 정반대다.**

표준 라이브러리를 쓰지 않는 이유
--------------------------------
`urllib.robotparser` 는 **명시적으로 금지된 URL 에 True 를 준다** (실측 2026-08-26).
와일드카드를 퍼센트 인코딩(`%2A`)해 버려 `Disallow: /item/*` 같은 규칙이 매칭되지 않는다.
차단 가드가 **허용 쪽으로** 틀리면 없느니만 못하다 — 안전하다는 착각을 주기 때문이다.
그래서 `protego` 를 쓴다. 0.6.1 이하에는 CVE-2026-55520 이 있어 **0.6.2 이상**을 쓴다.

그룹 단위로 읽어야 한다
-----------------------
`robots.txt` 는 `User-agent` 그룹 단위로 읽는다. 어느 그룹에 속한 규칙인지가 전부다.
2026-08-26 에 이걸 잘못 읽어 수집 계획 하나가 통째로 뒤집혔다 — `finance.naver.com` 의
`Allow: /item/board.naver` 는 **네이버 자체 크롤러(`yeti`) 그룹** 소속이었고, 제3자 봇이
매칭되는 `User-agent: *` 그룹은 `Disallow: /` 한 줄이었다.

인코딩
------
**UTF-8 을 가정하지 않는다.** 국내 사이트의 `robots.txt` 가 euc-kr 로 오는 경우가 있고,
잘못 디코딩하면 **예외 없이 규칙만 깨진다** — 그러면 그 규칙이 없는 것처럼 동작한다.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path
from typing import Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from common.paths import krx_db_path
from common.trading_calendar import KST, now_kst_iso

log = logging.getLogger(__name__)

#: 우리가 밝히는 이름. **바꾸면 판정이 달라진다** — 사이트가 이름별로 규칙을 나눠 두기
#: 때문이다. 실제 요청의 `User-Agent` 헤더와 **같은 값**을 써야 한다. 다르면 우리가
#: 확인한 규칙과 상대가 적용하는 규칙이 갈린다.
USER_AGENT = "AlphaStackBot"

#: 캐시를 몇 시간 쓰는가. 매 요청마다 받으면 그 자체가 상대 서버를 두드리는 일이 된다.
CACHE_HOURS = 24

#: `robots.txt` 를 기다리는 시간(초). 길게 잡을 이유가 없다 — 못 받으면 차단이고,
#: 차단은 기다린다고 달라지지 않는다.
FETCH_TIMEOUT = 10

#: 네트워크에 닿지도 못한 경우의 상태 코드. 5xx 와 **같이** 다룬다.
UNREACHABLE = 0


class RobotsBlocked(PermissionError):
    """`robots.txt` 가 막은 경로다. 요청을 보내지 않는다.

    예외로 만드는 이유는 **조용히 건너뛰면 안 되기 때문**이다. 차단된 것을 빈 결과로
    돌려주면 "수집했는데 없었다" 와 구별되지 않는다.
    """


def origin_of(url: str) -> str:
    """`https://finance.naver.com/item/board.naver?code=1` → `https://finance.naver.com`

    `robots.txt` 는 **스킴·호스트·포트마다 따로**다. http 와 https 를 같은 것으로 묶으면
    한쪽 규칙이 다른 쪽에 새어 든다.
    """
    parts = urlsplit(url)
    if not parts.scheme or not parts.netloc:
        raise ValueError(f"절대 URL 이어야 한다: {url!r}")
    return urlunsplit((parts.scheme, parts.netloc, "", "", ""))


# ==================================================
# 1. 받아 오기
# ==================================================
def _decode(body: bytes, header_charset: Optional[str]) -> Tuple[str, str]:
    """`robots.txt` 원문을 글자로 바꾸고, **무엇으로 읽었는지**도 함께 돌려준다.

    UTF-8 을 가정하지 않는다. 순서는 ① 헤더가 말한 것 ② UTF-8 ③ euc-kr 이다.
    전부 실패하면 깨진 글자를 대체 문자로 바꿔서라도 읽는다 — 규칙 한 줄이라도
    건지는 편이 통째로 못 읽는 것보다 낫다.
    """
    후보 = [c for c in (header_charset, "utf-8", "euc-kr") if c]
    for charset in 후보:
        try:
            return body.decode(charset), charset
        except (UnicodeDecodeError, LookupError):
            continue
    return body.decode("utf-8", errors="replace"), "utf-8(replace)"


def fetch(origin: str, *, user_agent: str = USER_AGENT) -> Tuple[int, Optional[str], str]:
    """`robots.txt` 를 받아 `(상태코드, 원문, 인코딩)` 을 돌려준다.

    **예외를 밖으로 내보내지 않는다.** 못 받았다는 사실 자체가 판정의 재료이고,
    그 판정은 부르는 쪽이 아니라 이 모듈이 한다.
    """
    url = f"{origin}/robots.txt"
    request = Request(url, headers={"User-Agent": user_agent}, method="GET")
    try:
        with urlopen(request, timeout=FETCH_TIMEOUT) as response:
            body = response.read()
            charset = response.headers.get_content_charset()
            text, used = _decode(body, charset)
            return response.status, text, used
    except HTTPError as error:
        # 4xx 와 5xx 는 **뜻이 정반대**다. 상태 코드를 그대로 올려 판정에 쓴다.
        return error.code, None, ""
    except (URLError, TimeoutError, OSError):
        # 닿지 못했다. RFC 9309 는 이 경우를 5xx 와 같이 다루라고 한다.
        return UNREACHABLE, None, ""


# ==================================================
# 2. 캐시
# ==================================================
def _connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path or krx_db_path(), timeout=60, isolation_level=None)
    conn.execute("PRAGMA busy_timeout=60000")
    conn.row_factory = sqlite3.Row
    return conn


_schema_ready: set = set()
_schema_lock = threading.Lock()


def _ensure_schema(db_path: Optional[Path] = None) -> None:
    key = str(db_path) if db_path is not None else "<기본>"
    if key in _schema_ready:
        return
    from ingest.store.migrations import migrate_path

    with _schema_lock:
        if key in _schema_ready:
            return
        migrate_path(db_path)
        _schema_ready.add(key)


def _fresh(fetched_at: str) -> bool:
    """캐시가 아직 쓸 만한가."""
    from datetime import datetime, timedelta

    try:
        stamp = datetime.fromisoformat(fetched_at)
    except ValueError:
        return False
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=KST)
    return datetime.now(KST) - stamp < timedelta(hours=CACHE_HOURS)


def _cached(origin: str, db_path: Optional[Path]) -> Optional[sqlite3.Row]:
    _ensure_schema(db_path)
    conn = _connect(db_path)
    try:
        row = conn.execute("SELECT * FROM robots_cache WHERE origin=?", (origin,)).fetchone()
    finally:
        conn.close()
    if row and _fresh(row["fetched_at"]):
        return row
    return None


def _store(origin: str, status: int, body: Optional[str], encoding: str,
           db_path: Optional[Path]) -> None:
    _ensure_schema(db_path)
    conn = _connect(db_path)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO robots_cache (origin, status, body, encoding, fetched_at) "
            "VALUES (?,?,?,?,?)",
            (origin, status, body, encoding, now_kst_iso()),
        )
    finally:
        conn.close()


def clear_cache(*, db_path: Optional[Path] = None) -> None:
    """캐시를 비운다. 테스트와 *"방금 robots 를 고쳤다"* 는 상황을 위한 것이다."""
    _ensure_schema(db_path)
    conn = _connect(db_path)
    try:
        conn.execute("DELETE FROM robots_cache")
    finally:
        conn.close()


# ==================================================
# 3. 판정
# ==================================================
def can_fetch(url: str, *, user_agent: str = USER_AGENT,
              db_path: Optional[Path] = None) -> bool:
    """이 URL 을 받아도 되는가. **요청을 보내기 직전에** 부른다.

    못 받았을 때의 방향이 이 함수의 핵심이다 — 4xx 는 허용, 5xx·네트워크 오류는 차단이다.
    자세한 근거는 이 파일 맨 위 표에 있다.
    """
    origin = origin_of(url)

    row = _cached(origin, db_path)
    if row is not None:
        status, body = row["status"], row["body"]
    else:
        status, body, encoding = fetch(origin, user_agent=user_agent)
        _store(origin, status, body, encoding, db_path)

    # ① 서버가 답을 못 준다 → 전면 차단. RFC 9309 가 MUST 로 정한 것이다.
    #    "잠깐 아픈 것뿐" 이라고 넘어가면 그 사이에 우리가 상대 위를 긁는다.
    if status == UNREACHABLE or 500 <= status <= 599:
        log.warning("[robots] %s 가 답을 주지 못했다 (status=%s) — 전면 차단으로 본다.",
                    origin, status)
        return False

    # ② robots.txt 가 없다 → 전면 허용. 규칙이 없으니 막을 근거도 없다.
    if 400 <= status <= 499:
        return True

    # ③ 규칙을 받았다 → 규칙대로. 그룹 단위로 읽는 일은 protego 가 한다.
    if not body:
        # 200 인데 본문이 비었다. 규칙이 없는 것과 같다.
        return True

    from protego import Protego

    return bool(Protego.parse(body).can_fetch(url, user_agent))


def require(url: str, *, user_agent: str = USER_AGENT,
            db_path: Optional[Path] = None) -> None:
    """막혀 있으면 **예외로 멈춘다.** 크롤러는 이쪽을 부른다.

    `can_fetch()` 가 돌려주는 `False` 를 무시하기는 너무 쉽다. 차단을 빈 결과로 넘기면
    *"수집했는데 없었다"* 와 구별되지 않고, 그 상태로 며칠이 지나면 아무도 눈치채지 못한다.
    """
    if not can_fetch(url, user_agent=user_agent, db_path=db_path):
        raise RobotsBlocked(
            f"robots.txt 가 막은 경로다: {url}\n"
            f"  확인한 이름: {user_agent}\n"
            "  왜: 규칙이 막았거나, robots.txt 를 받지 못했다(5xx·네트워크 오류는 차단이다).\n"
            "  할 일: 이 경로는 수집하지 않는다. 기술적으로 가능해도 하지 않는다."
        )
