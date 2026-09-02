"""수집 실행을 `ingest_run` · `ingest_run_stage` 에 남긴다. (기록 계층)

`pipelines/ingest.py` 가 이 모듈을 쓴다.

## `collect_log` 와 무엇이 다른가

`collect_log` 는 **"무엇을 어디까지 받았나"** 를 대상별로 담는다 — 종목·날짜·지표마다
한 줄이다. 여기는 **"언제 돌렸고 지금 어디쯤인가"** 다.

이 구별이 필요한 이유는 대장만 보면 **지금 돌고 있는 중인지 죽은 것인지 알 수 없기
때문**이다. 둘 다 "마지막 성공이 좀 됐다" 로 똑같이 보인다. 대시보드가 폴링해서
알고 싶은 것이 정확히 그 차이다.

## 중간에 죽어도 흔적이 남는다

단계를 **시작할 때 `running` 으로 한 줄 남기고** 끝날 때 갱신한다. 끝나고 한꺼번에
쓰면 죽었을 때 아무것도 안 남아서, 밖에서 보면 *"애초에 안 돌았다"* 와 구별되지 않는다.

    시작   ingest_run(status=running, finished_at=NULL)
    단계   ingest_run_stage(stage, status=running) → 끝나면 ok/error 로 갱신
    끝     ingest_run(status=ok|partial|error, finished_at=지금)

`finished_at` 이 비어 있는데 `started_at` 이 한참 전이면 **죽은 것**이다.
`stale_runs()` 가 그걸 찾아 준다.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from common.trading_calendar import now_kst_iso
from ingest.store.krx_store import connect

KST = timezone(timedelta(hours=9))

#: 실행 상태 — 마이그레이션 v8 의 CHECK 와 **같은 목록**이어야 한다.
RUN_STATUSES = ("running", "ok", "partial", "error", "dry_run")

#: 단계 상태.
STAGE_STATUSES = ("running", "ok", "error", "skipped", "dry_run")


def new_run_id() -> str:
    """실행 열쇠. 사람이 읽을 수 있게 KST 시각으로 만든다.

    UUID 를 쓰지 않는 이유는 대시보드와 로그를 눈으로 맞춰 보기 때문이다 —
    `20260902-143015` 는 알아볼 수 있지만 `a3f9...` 는 그렇지 않다.
    """
    return datetime.now(KST).strftime("%Y%m%d-%H%M%S")


def start_run(run_id: str, args: Optional[Dict] = None,
              conn: Optional[sqlite3.Connection] = None) -> None:
    """실행을 열고 `running` 으로 남긴다."""
    with (nullcontext(conn) if conn is not None else connect()) as own:
        own.execute(
            "INSERT OR REPLACE INTO ingest_run "
            "(run_id, started_at, finished_at, status, args, note) "
            "VALUES (?, ?, NULL, 'running', ?, NULL)",
            (run_id, now_kst_iso(),
             json.dumps(args or {}, ensure_ascii=False, sort_keys=True)),
        )


def finish_run(run_id: str, status: str, note: str = "",
               conn: Optional[sqlite3.Connection] = None) -> None:
    """실행을 닫는다. `status` 는 `ok` · `partial` · `error` · `dry_run`."""
    if status not in RUN_STATUSES:
        raise ValueError(
            f"모르는 실행 상태다: {status!r}\n"
            f"  쓸 수 있는 값: {', '.join(RUN_STATUSES)}\n"
            "  할 일: 새 상태가 필요하면 마이그레이션으로 표의 CHECK 부터 넓힌다."
        )
    with (nullcontext(conn) if conn is not None else connect()) as own:
        own.execute(
            "UPDATE ingest_run SET status = ?, finished_at = ?, note = ? "
            "WHERE run_id = ?",
            (status, now_kst_iso(), note or None, run_id),
        )


def start_stage(run_id: str, stage: str,
                conn: Optional[sqlite3.Connection] = None) -> None:
    """단계를 열고 `running` 으로 남긴다. **끝나기 전에 남기는 것이 요점이다.**"""
    with (nullcontext(conn) if conn is not None else connect()) as own:
        own.execute(
            "INSERT OR REPLACE INTO ingest_run_stage "
            "(run_id, stage, status, rows, started_at, finished_at, note) "
            "VALUES (?, ?, 'running', 0, ?, NULL, NULL)",
            (run_id, stage, now_kst_iso()),
        )


def finish_stage(run_id: str, stage: str, status: str, *, rows: int = 0,
                 note: str = "", conn: Optional[sqlite3.Connection] = None) -> None:
    """단계를 닫는다. `status` 는 `ok` · `error` · `skipped` · `dry_run`."""
    if status not in STAGE_STATUSES:
        raise ValueError(
            f"모르는 단계 상태다: {status!r}\n"
            f"  쓸 수 있는 값: {', '.join(STAGE_STATUSES)}"
        )
    with (nullcontext(conn) if conn is not None else connect()) as own:
        # 열지 않고 바로 닫는 경우(건너뛴 단계)도 있으므로 UPDATE 가 아니라 UPSERT 다.
        own.execute(
            "INSERT INTO ingest_run_stage "
            "(run_id, stage, status, rows, started_at, finished_at, note) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(run_id, stage) DO UPDATE SET "
            "  status = excluded.status, rows = excluded.rows, "
            "  finished_at = excluded.finished_at, note = excluded.note",
            (run_id, stage, status, rows, now_kst_iso(), now_kst_iso(), note or None),
        )


# ==================================================
# 조회 — 대시보드가 폴링한다
# ==================================================
def latest_runs(limit: int = 10,
                conn: Optional[sqlite3.Connection] = None) -> List[Dict]:
    """최근 실행들. 단계까지 함께 담아 준다."""
    with (nullcontext(conn) if conn is not None else connect()) as own:
        # `run_id` 를 두 번째 기준으로 둔다. `started_at` 은 초 단위라 같은 초에
        # 시작한 둘의 순서가 정해지지 않는데, `run_id` 도 시각이라 그 자리를 메운다.
        # (정렬이 흔들리면 대시보드가 "최근 실행" 으로 엉뚱한 것을 집는다.)
        runs = own.execute(
            "SELECT run_id, started_at, finished_at, status, args, note "
            "FROM ingest_run ORDER BY started_at DESC, run_id DESC LIMIT ?", (limit,)
        ).fetchall()
        결과 = []
        for r in runs:
            stages = own.execute(
                "SELECT stage, status, rows, started_at, finished_at, note "
                "FROM ingest_run_stage WHERE run_id = ? ORDER BY started_at",
                (r[0],)
            ).fetchall()
            결과.append({
                "run_id": r[0], "started_at": r[1], "finished_at": r[2],
                "status": r[3], "args": json.loads(r[4]) if r[4] else {},
                "note": r[5],
                "stages": [{
                    "stage": s[0], "status": s[1], "rows": s[2],
                    "started_at": s[3], "finished_at": s[4], "note": s[5],
                } for s in stages],
            })
    return 결과


def current_run(conn: Optional[sqlite3.Connection] = None) -> Optional[Dict]:
    """지금 돌고 있는 실행. 없으면 `None`.

    대시보드가 "수집 중" 배지를 켤 때 쓴다. `stale_runs()` 로 걸러지는 죽은 실행도
    여기서는 `running` 으로 보이므로, 오래됐는지는 부르는 쪽이 판단한다.
    """
    for run in latest_runs(limit=5, conn=conn):
        if run["status"] == "running":
            return run
    return None


def stale_runs(older_than_hours: float = 6.0,
               conn: Optional[sqlite3.Connection] = None) -> List[Dict]:
    """`running` 인 채로 오래 멈춰 있는 실행 — **죽은 것**이다.

    프로세스가 죽으면 `finish_run` 이 안 불린다. 그 흔적을 지우지 않고 남겨 두는
    이유는, 지우면 "안 돌았다" 와 "돌다 죽었다" 가 다시 같아지기 때문이다.
    """
    끊긴선 = (datetime.now(KST) - timedelta(hours=older_than_hours)).strftime(
        "%Y-%m-%d %H:%M:%S")
    return [
        run for run in latest_runs(limit=50, conn=conn)
        if run["status"] == "running" and run["started_at"][:19] < 끊긴선
    ]
