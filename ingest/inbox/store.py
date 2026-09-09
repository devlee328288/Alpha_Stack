"""검사 결과를 `krx_cache.db` 에 담는다 — `inbox_batch` · `inbox_accepted` · `inbox_quarantine`.

    from ingest.inbox.engine import inspect_file
    from ingest.inbox.store import load_result, already_ingested

    if not already_ingested(sha):
        result = inspect_file(path, kind="ohlcv_stock")
        batch_id = load_result(result, path)

🔴 **`daily_price` 를 건드리지 않는다.** 920만 행은 우리가 16년치를 받아 쌓은 것이고, 반입은
남이 준 자료다. 두 개를 한 표에 섞으면 *"이 값은 누가 어디서 가져왔나"* 를 되짚을 수 없게 된다.
합격분을 우리 시세와 맞대 보는 일(규격의 `target.compareWith`)은 **적재 뒤에 따로** 한다.

무엇을 열쇠로 쓰나
------------------
`batch_id` 는 `<끝난시각>-<파일지문 앞 12자>` 다. 시각만 쓰면 같은 초에 두 파일을 넣을 때
겹치고, 지문만 쓰면 **같은 파일을 규격 개정 뒤 다시 검사한 결과**가 앞의 것을 덮는다. 다시
검사하는 일은 정당하고, 판정이 어떻게 달라졌는지가 곧 규격 개정의 근거라 남겨야 한다.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import List, Optional

from common.paths import krx_db_path
from common.trading_calendar import now_kst_iso
from ingest.store.sqlite_db import write_lock

#: 한 번에 밀어 넣을 행 수. `executemany` 로 나눠 넣어 큰 파일에서도 메모리가 일정하다.
CHUNK = 1000


def _connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """반입 전용 연결. `BEGIN` 을 직접 쓰므로 autocommit 모드로 연다."""
    conn = sqlite3.connect(db_path or krx_db_path(), timeout=60, isolation_level=None)
    conn.execute("PRAGMA busy_timeout=60000")
    conn.row_factory = sqlite3.Row
    return conn


_schema_ready: set = set()
_schema_lock = threading.Lock()


def _ensure_schema(db_path: Optional[Path] = None) -> None:
    """표가 없으면 만든다. 한 프로세스에서 한 번만 확인한다."""
    key = str(db_path) if db_path is not None else "<기본>"
    if key in _schema_ready:
        return

    from ingest.store.migrations import migrate_path

    with _schema_lock:
        if key in _schema_ready:
            return
        migrate_path(db_path)
        _schema_ready.add(key)


# ==================================================
# 1. 파일 지문
# ==================================================
def file_sha256(path) -> str:
    """파일 내용의 지문. **이름이 아니라 내용으로** 같은 파일인지 판단하려고 쓴다.

    이름으로 판단하면 팀원이 `시세.csv` 를 고쳐 다시 올렸을 때 안 들이고 넘어간다.
    수정 시각도 못 쓴다 — 내려받을 때마다 새로 찍힌다.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def already_ingested(sha256: str, db_path: Optional[Path] = None) -> Optional[dict]:
    """그 지문의 파일을 이미 들였나. 들였으면 마지막 묶음 정보를, 아니면 None 을 돌려준다."""
    _ensure_schema(db_path)
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT batch_id, kind, rows_total, rows_accepted, rows_quarantined, "
            "       rejected, finished_at "
            "FROM inbox_batch WHERE src_sha256 = ? ORDER BY finished_at DESC LIMIT 1",
            (sha256,),
        ).fetchone()
    finally:
        conn.close()
    return dict(row) if row else None


def make_batch_id(sha256: str, finished_at: str) -> str:
    """`20260901T143022-a1b2c3d4e5f6` 꼴. 사람이 읽을 수 있고 겹치지 않는다."""
    stamp = finished_at.replace("-", "").replace(":", "").replace("+0900", "")
    stamp = stamp.split("+")[0].replace("T", "T")
    return f"{stamp}-{sha256[:12]}"


# ==================================================
# 2. 적재
# ==================================================
def _dump(value) -> Optional[str]:
    """JSON 문자열로. 빈 것은 NULL 로 둔다 — `{}` 와 "없다" 를 구별한다."""
    if value is None or (hasattr(value, "__len__") and len(value) == 0):
        return None
    return json.dumps(value, ensure_ascii=False)


def load_result(result, source_path, *, db_path: Optional[Path] = None,
                origin: str = "local", contributor: Optional[str] = None,
                started_at: Optional[str] = None, report_path: Optional[str] = None,
                sha256: Optional[str] = None) -> str:
    """검사 결과 하나를 통째로 담고 `batch_id` 를 돌려준다.

    **한 트랜잭션 안에서 셋을 다 쓴다.** 묶음만 남고 행이 안 들어가면 *"들였다고 적혀 있는데
    자료가 없는"* 상태가 되고, 그건 다음 실행이 "이미 들였다" 며 건너뛰기 때문에 조용히 굳는다.

    파일째 거부된 결과도 **담는다.** 무엇이 왜 되돌려졌는지가 팀원에게 줄 답이고, 그 기록이
    없으면 같은 파일이 다음 세션에 또 온다.
    """
    _ensure_schema(db_path)
    source_path = Path(source_path)
    digest = sha256 or file_sha256(source_path)
    finished_at = now_kst_iso()
    batch_id = make_batch_id(digest, finished_at)
    loaded_at = finished_at

    accepted_rows: List[tuple] = []
    for record in result.accepted.to_dict("records") if len(result.accepted) else []:
        accepted_rows.append((
            batch_id, record["row_no"], record["kind"], record.get("key_hash") or None,
            json.dumps(record["payload"], ensure_ascii=False),
            _dump(record.get("extras")), _dump(record.get("changes")),
            _dump(record.get("warnings")), loaded_at,
        ))

    quarantined_rows: List[tuple] = []
    for record in result.quarantined.to_dict("records") if len(result.quarantined) else []:
        quarantined_rows.append((
            batch_id, record["row_no"], record["kind"],
            json.dumps(record["payload"], ensure_ascii=False),
            json.dumps(record.get("raw") or {}, ensure_ascii=False),
            _dump(record.get("extras")), _dump(record.get("changes")),
            json.dumps(record.get("violations") or [], ensure_ascii=False), loaded_at,
        ))

    with write_lock:
        conn = _connect(db_path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT OR REPLACE INTO inbox_batch "
                "(batch_id, kind, src_path, src_sha256, src_bytes, origin, contributor, "
                " schema_version, rows_total, rows_accepted, rows_quarantined, rejected, "
                " report_path, started_at, finished_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (batch_id, result.kind, str(source_path), digest,
                 source_path.stat().st_size, origin, contributor,
                 result.report.get("schema_version"), result.rows_total,
                 len(result.accepted), len(result.quarantined), result.rejected,
                 report_path, started_at or finished_at, finished_at),
            )
            for start in range(0, len(accepted_rows), CHUNK):
                conn.executemany(
                    "INSERT OR REPLACE INTO inbox_accepted "
                    "(batch_id, row_no, kind, key_hash, payload, extras, changes, "
                    " warnings, loaded_at) VALUES (?,?,?,?,?,?,?,?,?)",
                    accepted_rows[start:start + CHUNK],
                )
            for start in range(0, len(quarantined_rows), CHUNK):
                conn.executemany(
                    "INSERT OR REPLACE INTO inbox_quarantine "
                    "(batch_id, row_no, kind, payload, raw, extras, changes, "
                    " violations, loaded_at) VALUES (?,?,?,?,?,?,?,?,?)",
                    quarantined_rows[start:start + CHUNK],
                )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    return batch_id


# ==================================================
# 3. 되읽기
# ==================================================
def list_batches(kind: Optional[str] = None, limit: int = 50,
                 db_path: Optional[Path] = None) -> List[dict]:
    """최근 반입 묶음 목록. 화면과 보고서가 함께 쓴다."""
    _ensure_schema(db_path)
    conn = _connect(db_path)
    try:
        if kind:
            rows = conn.execute(
                "SELECT * FROM inbox_batch WHERE kind = ? ORDER BY finished_at DESC LIMIT ?",
                (kind, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM inbox_batch ORDER BY finished_at DESC LIMIT ?", (limit,)
            ).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


#: 되읽은 표에 붙는 출처 칸. `_` 로 시작해 규격 칸과 섞이지 않는다.
#: dbt 의 `_dbt_source_relation` · Data Vault 의 `record_source` 가 앉는 자리다 —
#: "이 행이 어느 파일에서, 누구를 통해, 언제 들어왔나" 가 행에 붙어 있어야
#: 뒤에서 우리 시세와 맞대다 이상한 값을 만났을 때 원본까지 되짚을 수 있다.
ORIGIN_COLUMNS = ("_batch_id", "_row_no", "_loaded_at",
                  "_origin", "_contributor", "_src_sha256")


def accepted_frame(kind: str, batch_id: Optional[str] = None,
                   db_path: Optional[Path] = None):
    """합격분을 표로 되읽는다 — JSON payload 를 칸으로 편다.

    되읽기가 있어야 반입이 끝이 아니라 시작이 된다. `supply/` 가 이걸 받아 우리 시세와 맞댄다.

    🔴 **빈 결과에도 칸을 남긴다.** 칸 없는 빈 표를 주면 받는 쪽의 `df["close"]` 가
    `KeyError` 로 터진다. `supply/` 계층이 이걸 계약으로 못 박아 두었는데
    (`supply/market.py` 의 `to_frame`) 반입 쪽만 지키지 않고 있었다. 그리고
    `inbox_accepted` 가 0행인 동안에는 **100% 이 분기를 탄다** — 즉 지금까지
    이 함수를 부른 모든 코드가 그 표를 받았다.

    칸 목록은 **규격에서 가져온다.** 행에서 뽑으면 파일마다 담아 온 칸이 달라
    표의 모양이 그때그때 바뀐다.
    """
    import pandas as pd

    _ensure_schema(db_path)
    conn = _connect(db_path)
    try:
        # `inbox_batch` 를 왼쪽 조인한다 — 배치 기록이 지워졌어도 행은 나와야 한다.
        조건 = "a.kind = ?" + (" AND a.batch_id = ?" if batch_id else "")
        차례 = "a.row_no" if batch_id else "a.batch_id, a.row_no"
        인자 = (kind, batch_id) if batch_id else (kind,)
        rows = conn.execute(
            "SELECT a.batch_id, a.row_no, a.payload, a.extras, a.loaded_at, "
            "       b.origin, b.contributor, b.src_sha256 "
            "FROM inbox_accepted AS a "
            "LEFT JOIN inbox_batch AS b ON b.batch_id = a.batch_id "
            f"WHERE {조건} ORDER BY {차례}", 인자
        ).fetchall()
    finally:
        conn.close()

    from ingest.inbox import engine as _engine
    칸 = [f["name"] for f in _engine.load_spec(kind)["fields"]] + list(ORIGIN_COLUMNS)

    records = []
    for row in rows:
        payload = json.loads(row["payload"])
        payload["_batch_id"] = row["batch_id"]
        payload["_row_no"] = row["row_no"]
        payload["_loaded_at"] = row["loaded_at"]
        payload["_origin"] = row["origin"]
        payload["_contributor"] = row["contributor"]
        payload["_src_sha256"] = row["src_sha256"]
        records.append(payload)

    frame = pd.DataFrame(records, columns=칸) if records else pd.DataFrame(columns=칸)
    # 파일이 담아 오지 않은 규격 칸은 `reindex` 가 결측으로 채운다. 순서도 규격 순서로
    # 고정되므로, 어떤 파일을 들였든 받는 쪽이 보는 표의 모양이 같다.
    return frame.reindex(columns=칸)


def quarantine_summary(db_path: Optional[Path] = None) -> List[dict]:
    """무엇이 왜 격리됐나 — 규칙별 집계. 보고서의 첫 줄이 되는 값이다."""
    _ensure_schema(db_path)
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT kind, violations FROM inbox_quarantine"
        ).fetchall()
    finally:
        conn.close()

    counter: dict = {}
    for row in rows:
        for item in json.loads(row["violations"]):
            if item.get("severity") != "error":
                continue
            key = (row["kind"], item.get("rule"))
            counter[key] = counter.get(key, 0) + 1
    return [{"kind": kind, "rule": rule, "rows": count}
            for (kind, rule), count in sorted(counter.items(), key=lambda kv: -kv[1])]


__all__ = [
    "CHUNK",
    "file_sha256",
    "already_ingested",
    "make_batch_id",
    "load_result",
    "list_batches",
    "accepted_frame",
    "ORIGIN_COLUMNS",
    "quarantine_summary",
]
