"""수집 대장 — 어느 출처의 어디까지를 언제 받았고, 왜 안 받았나.

**무엇을 막는가.** 수집은 중간에 죽는다. 네트워크가 끊기고, 한도가 차고, 출처가 그
구간을 아예 제공하지 않는다. 이때 *"받은 데까지"* 를 남기지 않으면 다시 돌릴 때
처음부터 받게 되고, 16년치 백필에서 그건 며칠을 날리는 일이다.

그런데 진짜 문제는 그게 아니다. **"안 받았다"에는 서로 다른 뜻이 여럿 섞여 있다.**

    받아봤는데 0건이었다      휴장일이다. 다시 물어봐야 아무것도 없다
    한도에 닿아 멈췄다        내일이면 받을 수 있다. 대상이 잘못된 게 아니다
    출처가 그 구간을 안 준다   영원히 없다. 물어보는 것 자체가 낭비다
    시도했는데 실패했다        다시 받아야 한다

이 넷을 하나로 뭉뚱그리면 두 방향으로 틀린다. 전부 재시도로 보면 휴장일마다 영원히
같은 호출을 태우고, 전부 완료로 보면 진짜 실패가 조용히 묻힌다. **그래서 상태를
칸으로 나눠 기록한다.**

기존 `fetch_log`·`index_fetch_log` 와의 관계
--------------------------------------------
그 둘이 이미 같은 일을 날짜·시장 축에 묶인 형태로 하고 있다. 900만 행 백필이 그 위에서
돌고 있으므로 **지우거나 옮기지 않는다.** 이 표는 그 방식을 어떤 출처에나 쓸 수 있게
넓힌 것이고, 새로 붙는 수집기가 이쪽을 쓴다.

저장과 같은 트랜잭션에 얹는 법
------------------------------
대장은 **적재와 같은 트랜잭션에서** 갱신돼야 한다. 따로 커밋하면 그 사이에 죽었을 때
"저장은 됐는데 대장에는 없는" 또는 그 반대의 어긋난 상태가 남는다. 그래서 이 모듈의
쓰기 함수는 전부 `conn` 을 받는다 — 넘기면 그 트랜잭션에 얹고, 안 넘기면 스스로 연다.

    with connect() as conn:
        conn.executemany("INSERT INTO index_price ...", rows)
        collect_log.mark_ok("krx_index", bas_dd, rows=len(rows), conn=conn)

읽는 쪽
-------
    if not collect_log.should_collect("krx_index", "20260826"):
        continue                      # 이미 답을 들었거나, 그만 물어볼 때가 됐다
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from common.paths import REPORTS_DIR, krx_db_path
from common.trading_calendar import now_kst_iso

log = logging.getLogger(__name__)


# ==================================================
# 1. 상태값
# ==================================================
#: 받았고 행이 있다.
OK = "ok"
#: 받아봤는데 0건이었다. **미수집과 다르다** — 휴장일이 여기 해당한다.
EMPTY = "empty"
#: 시도했는데 실패했다. 이것만 다시 받아야 한다.
ERROR = "error"
#: 하루 한도에 닿아 아껴 멈췄다. **실패가 아니다** — 내일이면 받을 수 있다.
QUOTA_EXHAUSTED = "quota_exhausted"
#: 출처가 제공하지 않는 구간이다. 물어보는 것 자체가 낭비다.
OUT_OF_RANGE = "out_of_range"

#: 표의 CHECK 제약과 같은 목록이다. 한쪽만 고치면 INSERT 가 터진다.
STATUSES = frozenset({OK, EMPTY, ERROR, QUOTA_EXHAUSTED, OUT_OF_RANGE})

#: **출처에서 이미 답을 들은** 상태. 다시 물어볼 이유가 없다.
#: `empty` 가 여기 있는 것이 이 표의 요점이다 — 0건도 답이다.
SETTLED = frozenset({OK, EMPTY, OUT_OF_RANGE})

#: 실패를 몇 번까지 다시 시도할 것인가.
#:
#: 무제한이면 구조적으로 실패하는 대상 하나가 배치를 돌릴 때마다 호출을 태운다.
#: 3 인 이유는 KRX 가 멀쩡한 키에도 20%가량 401 을 뱉기 때문이다(실측). 세 번이면
#: 일시적 흔들림은 거의 통과하고, 그래도 안 되면 사람이 볼 문제다.
DEFAULT_MAX_ATTEMPTS = 3


class UnknownStatus(ValueError):
    """표가 모르는 상태값이다. INSERT 가 터지기 전에 여기서 잡는다."""


# ==================================================
# 2. 연결 · 표 준비
# ==================================================
def _connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """대장 전용 연결. `BEGIN` 을 직접 쓰므로 autocommit 모드로 연다."""
    conn = sqlite3.connect(db_path or krx_db_path(), timeout=60, isolation_level=None)
    conn.execute("PRAGMA busy_timeout=60000")
    conn.row_factory = sqlite3.Row
    return conn


#: 이 프로세스에서 표를 이미 확인한 DB 들. 수집 루프가 날짜마다 부르는 자리라
#: 매번 마이그레이션을 돌리면 4,000번의 여닫기가 그대로 비용이 된다.
_schema_ready: set = set()
#: 첫 확인이 스레드마다 동시에 시작되면 그것 자체가 잠금 충돌이 된다.
_schema_lock = threading.Lock()


def _ensure_schema(db_path: Optional[Path] = None) -> None:
    """표가 없으면 만든다. 한 프로세스에서 한 번만 확인한다."""
    key = str(db_path) if db_path is not None else "<기본>"
    if key in _schema_ready:
        return

    from ingest.store.migrations import migrate_path

    with _schema_lock:
        if key in _schema_ready:      # 기다리는 동안 다른 스레드가 끝냈을 수 있다
            return
        migrate_path(db_path)
        _schema_ready.add(key)


# ==================================================
# 3. 쓰기
# ==================================================
#: 대장 한 줄을 넣거나 고친다.
#:
#: `last_success_at` 은 **덮어쓰지 않고 보존**한다. 어제 성공한 뒤 오늘 실패했다면
#: "마지막으로 성공한 때"는 여전히 어제다 — 그 값을 잃으면 얼마나 오래 막혀 있는지를
#: 알 수 없게 된다.
#:
#: `attempts` 는 실패할 때만 올라가고 성공하면 0 으로 돌아간다. 한도 소진은 대상의
#: 잘못이 아니므로 **올리지도 내리지도 않는다** — 예산이 세 번 마르면 멀쩡한 대상을
#: 영영 포기하게 되기 때문이다.
_UPSERT = f"""
INSERT INTO collect_log
  (source, target, status, rows, last_success_at, last_attempted_at, cursor, note, attempts)
VALUES (?,?,?,?,?,?,?,?,?)
ON CONFLICT(source, target) DO UPDATE SET
  status            = excluded.status,
  rows              = excluded.rows,
  last_success_at   = COALESCE(excluded.last_success_at, collect_log.last_success_at),
  last_attempted_at = excluded.last_attempted_at,
  cursor            = COALESCE(excluded.cursor, collect_log.cursor),
  note              = excluded.note,
  attempts          = CASE
                        WHEN excluded.status = '{ERROR}'  THEN collect_log.attempts + 1
                        WHEN excluded.status IN ('{OK}', '{EMPTY}', '{OUT_OF_RANGE}') THEN 0
                        ELSE collect_log.attempts
                      END
"""


def record(source: str, target: str, status: str, *, rows: int = 0,
           cursor: Optional[str] = None, note: str = "",
           conn: Optional[sqlite3.Connection] = None,
           db_path: Optional[Path] = None) -> None:
    """대장에 한 줄 남긴다. 상태별 함수(`mark_ok` 등)를 쓰는 편이 읽기 좋다.

    `conn` 을 넘기면 **그 트랜잭션에 얹는다** — 적재와 대장이 함께 커밋되거나 함께
    사라져야 중간에 죽어도 어긋나지 않는다.
    """
    if status not in STATUSES:
        raise UnknownStatus(
            f"모르는 상태값이다: {status!r}\n"
            f"  쓸 수 있는 값: {', '.join(sorted(STATUSES))}\n"
            "  할 일: 새 상태가 정말 필요하면 마이그레이션으로 표의 CHECK 부터 넓힌다."
        )

    now = now_kst_iso()
    # 출처에서 답을 들은 것만 "성공"으로 친다. 실패·한도소진은 성공 시각을 건드리지 않는다.
    success_at = now if status in (OK, EMPTY) else None
    # ⚠️ **첫 시도도 한 번이다.** 이 자리는 새 줄일 때만 쓰이는데, 여기에 0 을 넣으면
    #    맨 처음 실패가 세어지지 않아 재시도가 한 번씩 더 돈다. 이미 있는 줄은 아래
    #    `ON CONFLICT` 의 CASE 가 대신 센다.
    first_attempts = 1 if status == ERROR else 0
    params = (source, target, status, rows, success_at, now, cursor, note, first_attempts)

    if conn is not None:
        conn.execute(_UPSERT, params)
        return

    _ensure_schema(db_path)
    own = _connect(db_path)
    try:
        own.execute("BEGIN IMMEDIATE")
        try:
            own.execute(_UPSERT, params)
            own.execute("COMMIT")
        except Exception:
            own.execute("ROLLBACK")
            raise
    finally:
        own.close()


def mark_ok(source: str, target: str, *, rows: int, cursor: Optional[str] = None,
            note: str = "", conn: Optional[sqlite3.Connection] = None,
            db_path: Optional[Path] = None) -> None:
    """받았고 행이 있다."""
    record(source, target, OK, rows=rows, cursor=cursor, note=note,
           conn=conn, db_path=db_path)


def mark_empty(source: str, target: str, *, note: str = "",
               conn: Optional[sqlite3.Connection] = None,
               db_path: Optional[Path] = None) -> None:
    """받아봤는데 0건이었다 — 휴장일 등. **실패가 아니므로 다시 묻지 않는다.**

    ⚠️ 오늘·어제의 0건은 아직 장이 안 끝나서일 수 있다. 그 경우까지 확정으로 보고
       싶지 않다면 `should_collect(..., empty_recheck_days=7)` 로 다시 열어 준다.
    """
    record(source, target, EMPTY, rows=0, note=note, conn=conn, db_path=db_path)


def mark_error(source: str, target: str, *, note: str,
               conn: Optional[sqlite3.Connection] = None,
               db_path: Optional[Path] = None) -> None:
    """시도했는데 실패했다. `note` 에 **무엇이 실패했는지** 남긴다 — 비워 두지 않는다."""
    record(source, target, ERROR, rows=0, note=note, conn=conn, db_path=db_path)


def mark_quota_exhausted(source: str, target: str, *, note: str = "",
                         conn: Optional[sqlite3.Connection] = None,
                         db_path: Optional[Path] = None) -> None:
    """한도에 닿아 멈췄다. **실패로 세지 않는다** — 대상이 아니라 예산의 문제다."""
    record(source, target, QUOTA_EXHAUSTED, rows=0,
           note=note or "하루 한도 소진 — 다음 실행에서 이어 받는다.",
           conn=conn, db_path=db_path)


def mark_out_of_range(source: str, target: str, *, note: str = "",
                      conn: Optional[sqlite3.Connection] = None,
                      db_path: Optional[Path] = None) -> None:
    """출처가 제공하지 않는 구간이다. 다시 묻지 않는다.

    예: KRX Open API 는 2010-01-04 이전 지수를 주지 않는다. 그 이전 날짜는 오류가
    아니라 **0행**으로 조용히 돌아오므로, 이 상태로 남겨 두지 않으면 휴장일과 뒤섞인다.
    """
    record(source, target, OUT_OF_RANGE, rows=0, note=note, conn=conn, db_path=db_path)


# ==================================================
# 4. 읽기
# ==================================================
def entry(source: str, target: str, *, db_path: Optional[Path] = None) -> Optional[Dict]:
    """대장 한 줄. 아직 시도한 적이 없으면 `None`.

    **`None` 과 `status='empty'` 는 다르다.** 앞은 안 물어본 것이고 뒤는 물어봤는데
    없던 것이다. 이 구별이 이 표의 존재 이유다.
    """
    _ensure_schema(db_path)
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM collect_log WHERE source=? AND target=?", (source, target)
        ).fetchone()
    finally:
        conn.close()
    return dict(row) if row else None


def should_collect(source: str, target: str, *,
                   max_attempts: int = DEFAULT_MAX_ATTEMPTS,
                   empty_recheck_days: int = 0,
                   db_path: Optional[Path] = None) -> bool:
    """이 대상을 지금 받아야 하는가.

    - 한 번도 안 받았으면 → 받는다
    - 이미 답을 들었으면(`ok`·`empty`·`out_of_range`) → 안 받는다
    - 실패했으면 → `max_attempts` 까지만 다시 받는다
    - 한도 소진이면 → 받는다 (예산이 있는지는 `budget` 이 따로 답한다)

    `empty_recheck_days` 를 주면 **최근에 0건이었던 것만** 다시 연다. 장 마감 전에
    받은 0건을 확정 휴장일로 굳혀 버리는 사고를 막는 안전장치다.
    """
    row = entry(source, target, db_path=db_path)
    if row is None:
        return True

    status = row["status"]
    if status == EMPTY and empty_recheck_days > 0:
        if _days_since(row["last_attempted_at"]) < empty_recheck_days:
            return True
    if status in SETTLED:
        return False
    if status == ERROR:
        return int(row["attempts"] or 0) < max_attempts
    return True                                  # quota_exhausted — 예산이 풀리면 받는다


def _days_since(iso: Optional[str]) -> float:
    """ISO 문자열로부터 지금까지 며칠 지났나. 못 읽으면 아주 큰 값(= 오래됐다)."""
    from datetime import datetime

    from common.trading_calendar import KST

    if not iso:
        return float("inf")
    try:
        stamp = datetime.fromisoformat(iso)
    except ValueError:
        return float("inf")
    # 오래된 대장 행은 타임존이 없다(`datetime.now()` 로 적혔다). KST 로 간주한다 —
    # 그렇게 읽어야 하루 이내인지 판단이 어긋나지 않는다.
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=KST)
    return (datetime.now(KST) - stamp).total_seconds() / 86400


def pending(source: str, targets: Iterable[str], *,
            max_attempts: int = DEFAULT_MAX_ATTEMPTS,
            empty_recheck_days: int = 0,
            db_path: Optional[Path] = None) -> List[str]:
    """받아야 할 대상만 **순서를 지켜** 걸러 돌려준다.

    한 건씩 `should_collect()` 를 부르면 대상 수만큼 DB 를 여닫는다 — 4,097 거래일이면
    그것만으로 눈에 띄게 느리다. 여기서는 대장을 한 번에 읽어 메모리에서 판단한다.
    """
    _ensure_schema(db_path)
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT target, status, attempts, last_attempted_at FROM collect_log WHERE source=?",
            (source,),
        ).fetchall()
    finally:
        conn.close()

    known = {row["target"]: row for row in rows}
    out: List[str] = []
    for target in targets:
        row = known.get(target)
        if row is None:
            out.append(target)
            continue
        status = row["status"]
        if status == EMPTY and empty_recheck_days > 0 \
                and _days_since(row["last_attempted_at"]) < empty_recheck_days:
            out.append(target)
        elif status in SETTLED:
            continue
        elif status == ERROR:
            if int(row["attempts"] or 0) < max_attempts:
                out.append(target)
        else:                                    # quota_exhausted
            out.append(target)
    return out


def summary(source: Optional[str] = None, *,
            db_path: Optional[Path] = None) -> Dict[str, Dict]:
    """출처별 현황. 수집 현황 화면과 품질 검사가 읽는다.

    `{출처: {상태별 건수 · rows · last_success_at · stuck}}` 를 돌려준다.
    `stuck` 은 **재시도 한도까지 실패한 대상 수**다 — 사람이 봐야 할 유일한 숫자라
    따로 뽑아 둔다.
    """
    _ensure_schema(db_path)
    sql = ("SELECT source, status, COUNT(*) AS n, COALESCE(SUM(rows), 0) AS rows, "
           "MAX(last_success_at) AS last_success_at, "
           f"COALESCE(SUM(status='{ERROR}' AND attempts >= ?), 0) AS stuck "
           "FROM collect_log")
    params: List = [DEFAULT_MAX_ATTEMPTS]
    if source is not None:
        sql += " WHERE source=?"
        params.append(source)
    sql += " GROUP BY source, status"

    conn = _connect(db_path)
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()

    out: Dict[str, Dict] = {}
    for row in rows:
        bucket = out.setdefault(row["source"], {
            "targets": 0, "rows": 0, "last_success_at": None, "stuck": 0,
            **{status: 0 for status in sorted(STATUSES)},
        })
        bucket[row["status"]] = row["n"]
        bucket["targets"] += row["n"]
        bucket["rows"] += row["rows"]
        bucket["stuck"] += row["stuck"]
        if row["last_success_at"] and (bucket["last_success_at"] or "") < row["last_success_at"]:
            bucket["last_success_at"] = row["last_success_at"]
    return out


def stuck(source: Optional[str] = None, *,
          max_attempts: int = DEFAULT_MAX_ATTEMPTS,
          db_path: Optional[Path] = None) -> List[Dict]:
    """재시도를 다 쓰고도 실패한 대상 목록. **사람이 봐야 할 것만** 나온다."""
    _ensure_schema(db_path)
    sql = ("SELECT * FROM collect_log WHERE status=? AND attempts >= ?")
    params: List = [ERROR, max_attempts]
    if source is not None:
        sql += " AND source=?"
        params.append(source)
    sql += " ORDER BY source, target"

    conn = _connect(db_path)
    try:
        return [dict(row) for row in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


# ==================================================
# 5. 옛 대장 넘겨받기
# ==================================================
#: 옛 표 → (대장의 출처 이름, 시장 칸이 있는가)
#:
#: 종목 쪽 `fetch_log` 는 시장 칸이 없다. 날짜 하나로 KOSPI·KOSDAQ 을 함께 받아 왔기
#: 때문이다. 지수 쪽 `index_fetch_log` 는 시장별로 따로 받으므로 칸이 있다.
_LEGACY_TABLES = (
    ("fetch_log", "krx_stock", False),
    ("index_fetch_log", "krx_index", True),
)


def import_legacy(*, db_path: Optional[Path] = None) -> Dict[str, int]:
    """옛 수집 대장(`fetch_log`·`index_fetch_log`)의 이력을 이 표로 옮겨 온다.

    **왜 필요한가.** 이 표만 보고 "받아야 하는가"를 판단하기 시작하면, 옮기기 전까지는
    이미 받은 4,343 거래일이 전부 미수집으로 보인다. 그대로 배치를 돌리면 16년치를
    처음부터 다시 받는다 — 하루 한도를 통째로 태우는 사고다.

    **옛 표는 읽기만 한다.** 900만 행 백필이 아직 그 위에서 돌고 있어 지우거나 옮기면
    되돌릴 수 없다. 여기서는 복사만 하고 원본은 제자리에 둔다.

    이미 이 표에 있는 대상은 **건드리지 않는다.** 새 수집기가 남긴 최신 상태를 옛 값으로
    덮으면 방금 고친 실패가 되살아난다. 그래서 여러 번 돌려도 안전하다.
    """
    _ensure_schema(db_path)
    conn = _connect(db_path)
    moved: Dict[str, int] = {}
    try:
        for table, source, has_market in _LEGACY_TABLES:
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            if not exists:
                moved[source] = 0
                continue

            columns = "bas_dd, market, rows, fetched_at" if has_market \
                else "bas_dd, '', rows, fetched_at"
            rows = conn.execute(f"SELECT {columns} FROM {table}").fetchall()

            payload = []
            for bas_dd, market, count, fetched_at in rows:
                target = f"{market}/{bas_dd}" if has_market and market else bas_dd
                # 행이 있었으면 받은 것, 0이면 휴장일. 옛 표에는 이 둘밖에 없다 —
                # 실패는 애초에 기록되지 않았다(그래서 이 표가 필요했다).
                status = OK if (count or 0) > 0 else EMPTY
                # ⚠️ 옛 `fetched_at` 은 타임존이 없다(`datetime.now()` 로 적혔다).
                #    고쳐 적지 않고 **그대로** 옮긴다 — 없던 정밀도를 지어내면 나중에
                #    "이 시각이 어느 시간대인가"를 아무도 답할 수 없게 된다.
                payload.append((source, target, status, count or 0,
                                fetched_at, fetched_at, None, f"{table} 에서 넘겨받음", 0))

            conn.execute("BEGIN IMMEDIATE")
            try:
                before = conn.execute(
                    "SELECT COUNT(*) FROM collect_log WHERE source=?", (source,)
                ).fetchone()[0]
                # OR IGNORE — 이미 있는 대상은 그대로 둔다.
                conn.executemany(
                    "INSERT OR IGNORE INTO collect_log "
                    "(source, target, status, rows, last_success_at, last_attempted_at,"
                    " cursor, note, attempts) VALUES (?,?,?,?,?,?,?,?,?)",
                    payload,
                )
                # `conn.total_changes` 는 **연결이 열린 뒤의 누적**이라 이 문장이 넣은
                # 수가 아니다. 두 표를 잇달아 옮기면 두 번째가 첫 번째 것까지 세어
                # 로그가 조용히 부풀어 오른다.
                changed = conn.execute(
                    "SELECT COUNT(*) FROM collect_log WHERE source=?", (source,)
                ).fetchone()[0] - before
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
            moved[source] = len(payload)
            log.info("[대장] %s → %s %d건 확인 (새로 들어온 것 %d건)",
                     table, source, len(payload), changed)
    finally:
        conn.close()
    return moved


# ==================================================
# 6. 리포트
# ==================================================
def write_report(path: Optional[Path] = None, *,
                 db_path: Optional[Path] = None) -> Path:
    """수집 현황을 JSON 으로 떨군다.

    정렬·들여쓰기를 고정한다 — `git diff` 로 어제와 무엇이 달라졌는지 읽히게 하려는 것이다.

    ⚠️ **이 파일은 보고서지 상태의 정본이 아니다.** 정본은 DB 안의 표다. 이 파일을
       고쳐도 수집기의 판단은 달라지지 않는다.
    """
    target = path or (REPORTS_DIR / "collect_status.json")
    target.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at_kst": now_kst_iso(),
        "sources": summary(db_path=db_path),
        "stuck": stuck(db_path=db_path),
    }
    text = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False)
    with open(target, "w", encoding="utf-8", newline="\n") as fp:
        fp.write(text + "\n")
    return target
