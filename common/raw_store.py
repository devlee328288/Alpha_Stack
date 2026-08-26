"""응답 원문 보존 — 다시 받지 않고 다시 정규화할 수 있게.

**무엇을 막는가.** 정규화는 틀린다. 필드 이름을 잘못 매핑하고, 숫자 파싱이 어떤 값에서만
깨지고, 그때는 안 담은 칸이 나중에 필요해진다. 이 저장소에서만 이미 네 번 겪었다.

원문이 없으면 고치는 방법이 **다시 받는 것뿐**이다. 16년치를 다시 받는 것은 며칠과
하루 한도를 통째로 쓰는 일이고, 출처가 그사이에 과거 값을 정정했다면 **같은 자료를
다시 받을 수도 없다.**

원문을 남겨 두면 네트워크를 한 번도 안 타고 정규화만 다시 돌린다.

    python scripts/renormalize.py --source krx --dry-run

두 번째 쓸모 — 언제부터 알 수 있었나
------------------------------------
`fetched_at` 은 *"우리가 이 사실을 언제부터 알 수 있었나"* 의 근거다. 그 시각을 정규화
표에만 적어 두면 나중에 고쳐 적었는지 아닌지 증명할 방법이 없다. 미래를 훔쳐본 모델은
성능이 좋아 보이기 때문에 **증명할 수 없는 시각은 없는 것과 같다.**

무엇을 저장하고 무엇을 저장하지 않나
------------------------------------
**공공·금융 API 응답만 저장한다.** 크롤링으로 받은 문서 본문은 저장하지 않는다 —
기사 본문 전문 보관은 저작권 문제이고, 우리 수집 규칙이 이미 *"제목·요약·링크까지"* 로
선을 그어 두었다. 화이트리스트에 없는 출처는 조용히 넘어가지 않고 **예외**로 막는다.

    raw_store.save("krx", "idx/kospi_dd_trd/20260826", body)      # OK
    raw_store.save("naver_news_article", "...", body)             # SourceNotAllowed

⚠️ 응답을 **바이트 그대로** 담는다. 문자열로 바꿔 담으면 그 순간 인코딩 추측이 끼어들고
   (euc-kr 로 오는 곳이 실재한다), 잘못 디코딩한 원문은 더 이상 원문이 아니다.
"""

from __future__ import annotations

import gzip
import hashlib
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from common.paths import krx_db_path
from common.trading_calendar import now_kst_iso

log = logging.getLogger(__name__)


# ==================================================
# 1. 무엇을 저장해도 되는가
# ==================================================
#: 원문을 남겨도 되는 출처.
#:
#: 전부 **기관이 API 로 내려주는 공공·금융 데이터**다. 우리가 받은 것을 우리 DB 에
#: 캐시하는 것이고 재배포가 아니다.
#:
#: ⚠️ **크롤링으로 받은 문서 본문은 여기 넣지 않는다.** 뉴스 기사 전문 보관은 저작권
#:    문제이고, 수집 규칙이 이미 "제목·요약·링크까지"로 선을 그어 두었다. 뉴스 검색
#:    API 응답도 지금은 빼 둔다 — 요약문이 응답에 들어 있어 "원문 보관"과 "요약 저장"의
#:    경계가 흐려지기 때문이다. 필요해지면 팀이 정하고 나서 넣는다.
ALLOWED_SOURCES = frozenset({
    "krx",        # KRX Open API — 시세·지수
    "dart",       # 전자공시 — 공시목록·재무
    "ecos",       # 한국은행 경제통계
    "fred",       # 미 연준 경제지표
    "kosis",      # 국가통계포털
    "fss",        # 금융감독원
})

#: 한 응답이 이보다 크면 저장하지 않고 예외를 낸다.
#:
#: 원문 보존이 목적이지 DB 를 부풀리는 게 목적이 아니다. 예상 밖으로 큰 응답(잘못된
#: 파라미터로 전 구간을 받아 온 경우 등)이 들어오면 DB 가 순식간에 GB 단위로 큰다.
#:
#: 실측 2026-08-26 (KRX 하루치 응답 · 2026-08-25 기준일):
#:
#:     지수 KOSPI  51행     14,322B →  3,523B (24.6%)
#:     종목 KOSPI  944행   293,170B → 54,742B (18.7%)
#:
#: 즉 하루에 약 58KB, 1년이면 약 21MB 다. 상한 32MB 는 정상 응답의 500배가 넘으므로
#: 평소에 걸리지 않고, **걸린다면 그건 요청이 잘못된 것**이다.
MAX_BODY_BYTES = 32 * 1024 * 1024


class SourceNotAllowed(ValueError):
    """원문을 남겨도 되는 출처가 아니다."""


class RawTooLarge(ValueError):
    """응답이 보존 상한을 넘었다."""


def _check_source(source: str) -> None:
    if source not in ALLOWED_SOURCES:
        raise SourceNotAllowed(
            f"원문을 보존할 수 없는 출처다: {source!r}\n"
            f"  보존 가능: {', '.join(sorted(ALLOWED_SOURCES))}\n"
            "  왜 막나: 크롤링으로 받은 문서 본문 보관은 저작권 문제다.\n"
            "  할 일: 공공·금융 API 라면 common/raw_store.py 의 ALLOWED_SOURCES 에\n"
            "         근거와 함께 추가한다. 크롤링이라면 정규화 결과만 저장한다."
        )


# ==================================================
# 2. 연결 · 표 준비
# ==================================================
def _connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path or krx_db_path(), timeout=60, isolation_level=None)
    conn.execute("PRAGMA busy_timeout=60000")
    conn.row_factory = sqlite3.Row
    return conn


_schema_ready: set = set()
_schema_lock = threading.Lock()


def _ensure_schema(db_path: Optional[Path] = None) -> None:
    """표가 없으면 만든다. 한 프로세스에서 한 번만 확인한다.

    ⚠️ 여기가 이 파일에서 유일하게 위 계층(`ingest`)을 건드리는 곳이다. 표를 만드는
    책임은 마이그레이션 한 곳에 있어야 하는데, 이 모듈은 그보다 아래 계층이라 맨 위에서
    import 하면 방향이 뒤집힌다. 그래서 함수 안에서 늦게 가져온다.
    """
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
# 3. 쓰기
# ==================================================
def save(source: str, target: str, body: bytes, *,
         encoding: Optional[str] = None, note: str = "",
         fetched_at: Optional[str] = None,
         conn: Optional[sqlite3.Connection] = None,
         db_path: Optional[Path] = None) -> str:
    """응답 원문을 압축해 남기고 **압축 전 원문의 sha256** 을 돌려준다.

    지문을 압축 전 값으로 잡는 이유는, 압축 결과가 라이브러리 버전에 따라 달라질 수
    있기 때문이다. 버전에 매인 값을 지문으로 삼으면 언젠가 *"같은 자료인데 다르다"*
    가 나온다.

    `conn` 을 넘기면 그 트랜잭션에 얹는다 — 적재·대장·원문이 함께 커밋돼야 한다.
    """
    _check_source(source)
    if not isinstance(body, (bytes, bytearray)):
        raise TypeError(
            "원문은 bytes 여야 한다 — 문자열로 바꿔 담으면 인코딩 추측이 끼어든다.\n"
            "  할 일: response.read() 결과를 디코딩하기 **전에** 넘긴다."
        )
    if len(body) > MAX_BODY_BYTES:
        raise RawTooLarge(
            f"응답이 보존 상한을 넘었다: {len(body):,} > {MAX_BODY_BYTES:,} 바이트\n"
            "  할 일: 요청 범위를 좁히거나, 정말 필요하면 MAX_BODY_BYTES 를 근거와 함께 올린다."
        )

    digest = hashlib.sha256(body).hexdigest()
    # mtime 을 0 으로 고정한다 — 안 그러면 같은 원문을 같은 초에 두 번 압축해도
    # 바이트가 달라져 "바뀌었나?" 를 볼 때 헷갈린다.
    packed = gzip.compress(body, compresslevel=6, mtime=0)
    params = (source, target, fetched_at or now_kst_iso(), packed,
              digest, len(body), "gzip", encoding, note)

    sql = ("INSERT OR REPLACE INTO raw_response "
           "(source, target, fetched_at, body, sha256, bytes, compression, encoding, note) "
           "VALUES (?,?,?,?,?,?,?,?,?)")

    if conn is not None:
        conn.execute(sql, params)
        return digest

    _ensure_schema(db_path)
    own = _connect(db_path)
    try:
        own.execute("BEGIN IMMEDIATE")
        try:
            own.execute(sql, params)
            own.execute("COMMIT")
        except Exception:
            own.execute("ROLLBACK")
            raise
    finally:
        own.close()
    return digest


# ==================================================
# 4. 읽기
# ==================================================
def _unpack(row: sqlite3.Row) -> Dict:
    """한 줄을 원문까지 풀어서 돌려준다. 지문이 안 맞으면 **조용히 넘어가지 않는다.**"""
    body = gzip.decompress(row["body"]) if row["compression"] == "gzip" else row["body"]
    actual = hashlib.sha256(body).hexdigest()
    if actual != row["sha256"]:
        raise ValueError(
            f"보존된 원문이 손상됐다: {row['source']} {row['target']} {row['fetched_at']}\n"
            f"  기록된 지문 {row['sha256'][:16]}… · 실제 {actual[:16]}…\n"
            "  할 일: 이 줄을 지우고 그 대상을 다시 받는다."
        )
    return {
        "source": row["source"], "target": row["target"],
        "fetched_at": row["fetched_at"], "body": body,
        "sha256": row["sha256"], "bytes": row["bytes"],
        "encoding": row["encoding"], "note": row["note"],
    }


def load(source: str, target: str, *, fetched_at: Optional[str] = None,
         db_path: Optional[Path] = None) -> Optional[Dict]:
    """보존된 원문 하나. `fetched_at` 을 안 주면 **가장 최근에 받은 것**을 돌려준다."""
    _ensure_schema(db_path)
    conn = _connect(db_path)
    try:
        if fetched_at:
            row = conn.execute(
                "SELECT * FROM raw_response WHERE source=? AND target=? AND fetched_at=?",
                (source, target, fetched_at),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM raw_response WHERE source=? AND target=? "
                "ORDER BY fetched_at DESC LIMIT 1",
                (source, target),
            ).fetchone()
    finally:
        conn.close()
    return _unpack(row) if row else None


def iter_latest(source: str, *, prefix: str = "",
                db_path: Optional[Path] = None) -> Iterator[Dict]:
    """대상별 **가장 최근 원문**을 하나씩 돌려준다. 재정규화가 이 경로를 탄다.

    전부 메모리에 올리지 않는다 — 4,000건이 넘으면 그것만으로 수백 MB 다.
    """
    _ensure_schema(db_path)
    conn = _connect(db_path)
    try:
        sql = ("SELECT * FROM raw_response WHERE source=? "
               "AND (?='' OR target LIKE ?||'%') "
               # 같은 대상의 여러 판 중 가장 최근 것만 고른다.
               "AND fetched_at = (SELECT MAX(fetched_at) FROM raw_response r2 "
               "                  WHERE r2.source=raw_response.source "
               "                    AND r2.target=raw_response.target) "
               "ORDER BY target")
        for row in conn.execute(sql, (source, prefix, prefix)):
            yield _unpack(row)
    finally:
        conn.close()


def stats(db_path: Optional[Path] = None) -> Dict[str, Dict]:
    """출처별 보존 현황과 **실제 차지하는 용량**. 화면과 리포트가 읽는다."""
    _ensure_schema(db_path)
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT source, COUNT(*) AS n, COUNT(DISTINCT target) AS targets, "
            "COALESCE(SUM(bytes),0) AS raw_bytes, "
            "COALESCE(SUM(LENGTH(body)),0) AS stored_bytes, "
            "MIN(fetched_at) AS first_at, MAX(fetched_at) AS last_at "
            "FROM raw_response GROUP BY source"
        ).fetchall()
    finally:
        conn.close()

    out: Dict[str, Dict] = {}
    for row in rows:
        raw_bytes = row["raw_bytes"]
        stored = row["stored_bytes"]
        out[row["source"]] = {
            "responses": row["n"], "targets": row["targets"],
            "raw_bytes": raw_bytes, "stored_bytes": stored,
            # 압축이 실제로 얼마나 먹혔는지. 문서에 추측값을 적지 않으려고 함께 낸다.
            "ratio": round(stored / raw_bytes, 4) if raw_bytes else None,
            "first_at": row["first_at"], "last_at": row["last_at"],
        }
    return out


def targets(source: str, *, db_path: Optional[Path] = None) -> List[str]:
    """원문이 남아 있는 대상 목록."""
    _ensure_schema(db_path)
    conn = _connect(db_path)
    try:
        return [row[0] for row in conn.execute(
            "SELECT DISTINCT target FROM raw_response WHERE source=? ORDER BY target",
            (source,),
        )]
    finally:
        conn.close()
