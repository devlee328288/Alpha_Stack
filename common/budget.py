"""출처별 하루 호출 한도를 코드가 알고, 넘기 전에 스스로 멈춘다.

**무엇을 막는가.** 외부 API 는 하루에 몇 번까지만 부를 수 있다. 그걸 넘으면 그날 남은
수집이 통째로 막히고, 백필처럼 며칠에 걸친 작업은 진행이 어디서 끊겼는지도 모른 채
멈춘다. 그래서 한도를 **서버가 거절해서 아는 게 아니라 우리가 미리 세서** 알아야 한다.

세 가지를 지킨다
----------------
1. **80% 에서 한 번 경고한다.** 매번 찍으면 로그가 묻히므로 처음 넘을 때만 남긴다.
2. **100% 에서 멈추되 예외를 던지지 않는다.** 한도 소진은 고장이 아니라 정상적인 하루의
   끝이다. 예외로 만들면 배치가 실패로 기록되고, 그러면 "진짜 실패" 와 구별이 안 된다.
   `try_spend()` 가 `False` 를 돌려주고 호출자가 얌전히 접는다.
3. **서버가 한도 초과를 알려 오면 재시도하지 않는다.** 재시도는 남은 한도를 더 태울 뿐
   절대 성공하지 않는다. `mark_exhausted()` 로 그날치를 즉시 소진 처리한다.

왜 파일이 아니라 DB 인가
------------------------
카운터를 **프로세스가 여럿** 본다. 야간 배치와 화면의 즉시수집 버튼이 따로 도는데,
모듈 전역 변수로 세면 각자 0 부터 시작해 한도를 두 배로 쓴다. JSON 파일은 읽기-수정-쓰기
사이에 창이 있어 수집 스레드가 여럿일 때 실제로 어긋난다.

SQLite 는 이미 쓰고 있고 `BEGIN IMMEDIATE` 로 쓰기 잠금을 잡으면 프로세스 사이에서도
증가가 원자적이다. 표는 `(출처, KST 날짜)` 가 키라서 **자정 리셋 로직이 아예 필요 없다** —
날짜가 바뀌면 그냥 다른 행이라 0 부터 센다.

어디서 세는가
-------------
**실제로 네트워크에 나가는 자리 바로 앞**에서 센다. 재시도도 한 번의 호출이고 한도를
똑같이 먹기 때문이다. 상위 함수에서 "한 날짜 = 1콜" 로 세면 재시도분이 통째로 누락된다.

    if not budget.try_spend("krx"):
        break                     # 오늘은 여기까지. 실패가 아니다
    응답 = urlopen(...)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Dict, Optional

from common.paths import REPORTS_DIR, krx_db_path
from common.trading_calendar import now_kst_iso, today_kst

log = logging.getLogger(__name__)

#: 80% 를 넘으면 경고를 한 번 남긴다.
WARN_RATIO = 0.8

# 출처별 하루 한도.
#
# ⚠️ **이 숫자들의 근거는 서로 다르다.** 검증된 것처럼 쓰면 안 된다.
#
#   naver_search    25,000  ⚠️ **우리 키는 개발자센터가 아니라 NAVER API HUB 것이다**
#                           (2026-08-27 실호출 확인). HUB 는 **월 775,000건 · 월 단위**
#                           관리이고 우리는 **일 단위**로 센다. 보수적이라 넘길 위험은
#                           없지만 표기가 사실과 다르다는 것을 알고 쓴다.
#                           뉴스·카페글·블로그가 **한 통을 나눠 쓴다**. 단위는 계정이
#                           아니라 **애플리케이션(클라이언트 아이디)** 이다.
#   youtube_search     100  공식. 2026-06-01 부터 자체 통을 쓰고 콜당 1유닛이다.
#                           우리는 이 API 를 쓰지 않으므로 0 이어야 정상이다.
#   youtube         10,000  공식. 위를 뺀 나머지 전부가 이 통을 쓴다(유닛 단위).
#   dart            20,000  ⚠️ 공식 문구가 "일반적으로는 20,000건 이상" 이라며 스스로
#                           "요청 제한이 다르게 설정된 경우" 단서를 단다. 10,000 이라는
#                           2차 자료도 있다. 우리 키의 실제 값은 로그인 후 이용현황에서만
#                           확인된다 — 그때까지 이 값은 가정이다.
#   krx             10,000  ✅ **이용약관 제8조 ④ 명문이다** (2026-08-27 원문 확인):
#                           "하나의 키당 1일(매일 0시~24시) 10,000회 이하의 요청으로
#                           제한하며, 이를 초과할 경우 서비스가 중지될 수 있다."
#                           ⚠️ 단위가 계정이 아니라 **키**다 — 팀원이 각자 키를 받으면
#                           총량은 늘지만 한 사람이 몰아 돌리면 그 키만 막힌다.
#                           ⚠️ 제8조 ① 이 거래소에 횟수 재지정 재량을 준다. 거래소가
#                           낮추면 우리 카운터는 여유가 있다고 보고하는데 서버는 거절한다.
#                           openapi.krx.co.kr 이용약관
#
# 실제 한도가 이보다 낮으면 우리가 먼저 멈추는 게 아니라 서버가 먼저 거절한다.
# 그래서 확인 전까지는 보수적으로 보고, `mark_exhausted()` 가 안전망 역할을 한다.
LIMITS: Dict[str, int] = {
    "krx": 10_000,
    "dart": 20_000,
    "naver_search": 25_000,
    "youtube": 10_000,
    "youtube_search": 100,
}


class UnknownSource(KeyError):
    """한도를 모르는 출처다. 오타를 조용히 넘기지 않으려고 예외로 만든다."""


# ==================================================
# 1. 연결 · 표 준비
# ==================================================
#: 이 프로세스에서 표를 이미 확인한 DB 들. 매 호출마다 마이그레이션을 돌리면
#: 수집 스레드 6개가 같은 파일에 동시에 붙어 서로를 막는다.
_schema_ready: set = set()
#: 첫 확인이 스레드마다 동시에 시작되면 그것 자체가 잠금 충돌이 된다.
_schema_lock = threading.Lock()


def _connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """카운터용 연결. `BEGIN` 을 직접 쓰므로 autocommit 모드로 연다."""
    conn = sqlite3.connect(db_path or krx_db_path(), timeout=60, isolation_level=None)
    conn.execute("PRAGMA busy_timeout=60000")
    return conn


def _ensure_schema(db_path: Optional[Path] = None) -> None:
    """표가 없으면 만든다. 한 프로세스에서 한 번만 확인한다.

    ⚠️ 여기가 이 파일에서 **유일하게 위 계층(`ingest`)을 건드리는 곳**이다. 표를 만드는
    책임은 마이그레이션 한 곳에 있어야 하는데(두 곳에 DDL 을 두면 언젠가 갈라진다),
    이 모듈은 그보다 아래 계층이라 맨 위에서 import 하면 방향이 뒤집힌다.
    그래서 **함수 안에서 늦게** 가져온다.
    """
    key = str(db_path) if db_path is not None else "<기본>"
    if key in _schema_ready:
        return

    from ingest.store.migrations import migrate_path

    with _schema_lock:
        if key in _schema_ready:      # 기다리는 동안 다른 스레드가 끝냈을 수 있다
            return
        migrate_path(db_path)
        _schema_ready.add(key)


def _row(conn: sqlite3.Connection, source: str, kst_date: str,
         limit: int) -> sqlite3.Row:
    """오늘 행을 가져온다. 없으면 만든다. **호출자가 트랜잭션을 이미 열어 두어야 한다.**"""
    conn.execute(
        "INSERT OR IGNORE INTO call_budget (source, kst_date, used, daily_limit) "
        "VALUES (?,?,0,?)",
        (source, kst_date, limit),
    )
    return conn.execute(
        "SELECT used, daily_limit, warned_at FROM call_budget "
        "WHERE source=? AND kst_date=?",
        (source, kst_date),
    ).fetchone()


def _limit_for(source: str, override: Optional[int]) -> int:
    if override is not None:
        return override
    try:
        return LIMITS[source]
    except KeyError as exc:
        raise UnknownSource(
            f"한도를 모르는 출처다: {source!r}\n"
            f"  아는 출처: {', '.join(sorted(LIMITS))}\n"
            "  할 일: common/budget.py 의 LIMITS 에 한도를 근거와 함께 추가하거나,\n"
            "         try_spend(..., limit=<숫자>) 로 직접 넘긴다."
        ) from exc


# ==================================================
# 2. 쓰기
# ==================================================
def try_spend(source: str, n: int = 1, *, limit: Optional[int] = None,
              db_path: Optional[Path] = None) -> bool:
    """`n` 번 부를 여유가 있으면 **미리 세고** `True`, 없으면 아무것도 안 하고 `False`.

    `False` 는 실패가 아니라 *"오늘은 여기까지"* 다. 호출자는 예외를 기대하지 말고
    루프를 얌전히 빠져나가면 된다.

    ⚠️ **부르기 전에** 센다. 부르고 나서 세면, 응답을 못 받고 죽었을 때 이미 나간 호출이
    장부에 안 남아 한도를 넘겨 쓰게 된다. 세고 나서 실패하면 손해는 1콜뿐이다.
    """
    _ensure_schema(db_path)
    kst_date = today_kst().isoformat()
    ceiling = _limit_for(source, limit)

    conn = _connect(db_path)
    try:
        # 잠금을 먼저 잡는다 — 읽고 나서 쓰는 사이에 다른 프로세스가 끼어들면
        # 둘 다 "아직 여유 있다" 고 판단해 한도를 넘긴다.
        conn.execute("BEGIN IMMEDIATE")
        try:
            used, ceiling_db, warned_at = _row(conn, source, kst_date, ceiling)
            ceiling = ceiling_db          # 오늘 이미 정해진 한도를 따른다

            if used + n > ceiling:
                conn.execute("COMMIT")
                log.info("[예산] %s 한도 도달 — 오늘은 여기까지 (%d/%d)",
                         source, used, ceiling)
                return False

            after = used + n
            conn.execute(
                "UPDATE call_budget SET used=? WHERE source=? AND kst_date=?",
                (after, source, kst_date),
            )
            # 80% 를 처음 넘는 순간에만 경고를 남긴다. 매번 찍으면 로그에 묻힌다.
            crossed = warned_at is None and after >= ceiling * WARN_RATIO
            if crossed:
                conn.execute(
                    "UPDATE call_budget SET warned_at=? WHERE source=? AND kst_date=?",
                    (now_kst_iso(), source, kst_date),
                )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

        if crossed:
            log.warning("[예산] %s 가 한도의 %d%% 를 넘었다 (%d/%d)",
                        source, int(WARN_RATIO * 100), after, ceiling)
        return True
    finally:
        conn.close()


def mark_exhausted(source: str, *, note: str = "", limit: Optional[int] = None,
                   db_path: Optional[Path] = None) -> None:
    """서버가 한도 초과를 알려 왔다 — 그날치를 즉시 소진 처리한다.

    우리 계산과 서버의 계산이 어긋날 수 있다(한도가 실제로는 더 낮았거나, 다른 곳에서
    같은 키를 썼거나). 서버가 거절했으면 그쪽이 맞다. 남은 만큼을 채워 두면 이후
    `try_spend()` 가 전부 `False` 를 돌려주므로 **재시도 루프가 저절로 멈춘다.**
    """
    _ensure_schema(db_path)
    kst_date = today_kst().isoformat()
    ceiling = _limit_for(source, limit)

    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            _row(conn, source, kst_date, ceiling)
            conn.execute(
                "UPDATE call_budget SET used = daily_limit, warned_at = COALESCE(warned_at, ?) "
                "WHERE source=? AND kst_date=?",
                (now_kst_iso(), source, kst_date),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()

    log.warning("[예산] %s — 서버가 한도 초과를 알려 와 오늘치를 소진 처리했다. %s",
                source, note or "재시도하지 않는다.")


# ==================================================
# 3. 읽기 · 리포트
# ==================================================
def usage(source: Optional[str] = None, *, kst_date: Optional[str] = None,
          db_path: Optional[Path] = None) -> Dict[str, Dict]:
    """오늘(또는 지정한 날) 출처별 사용량. `{출처: {used, limit, ratio, warned_at}}`"""
    _ensure_schema(db_path)
    day = kst_date or today_kst().isoformat()

    sql = "SELECT source, used, daily_limit, warned_at FROM call_budget WHERE kst_date=?"
    params = [day]
    if source is not None:
        sql += " AND source=?"
        params.append(source)

    conn = _connect(db_path)
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()

    out: Dict[str, Dict] = {}
    for name, used, ceiling, warned_at in rows:
        out[name] = {
            "used": used,
            "limit": ceiling,
            # 한도가 0 이면 나눗셈이 터진다. 설정 실수를 예외로 키우지 않는다.
            "ratio": round(used / ceiling, 4) if ceiling else None,
            "warned_at": warned_at,
        }
    # 한 번도 안 부른 출처도 0 으로 보여 준다 — 화면에서 "빠진 것" 과 "안 쓴 것" 이
    # 구별돼야 한다.
    if source is None:
        for name, ceiling in LIMITS.items():
            out.setdefault(name, {"used": 0, "limit": ceiling, "ratio": 0.0,
                                  "warned_at": None})
    return out


def write_report(path: Optional[Path] = None, *, db_path: Optional[Path] = None) -> Path:
    """오늘 사용량을 JSON 으로 떨군다.

    정렬·들여쓰기를 고정한다 — 그래야 `git diff` 로 어제와 무엇이 달라졌는지 읽힌다.
    """
    target = path or (REPORTS_DIR / "quota_usage.json")
    target.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at_kst": now_kst_iso(),
        "kst_date": today_kst().isoformat(),
        "sources": usage(db_path=db_path),
    }
    text = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False)
    with open(target, "w", encoding="utf-8", newline="\n") as fp:
        fp.write(text + "\n")
    return target
