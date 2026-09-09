"""dart_financial(접수일 포함)을 HF 반출용 parquet 로 내보낸다.

왜 따로 내보내나
----------------
HF `alphastack-dart` 의 기존 43개 parquet 는 접수일(`rcept_dt`)이 없어서
접수일 기준 시점 정합이 불가능하다 — 학습에 넣으면 석 달치 미래가 들어간다.
이 스크립트가 내보내는 것은 API 로 새로 받은 **접수일 포함** 수집분이고,
HF 저장소에서 `pit/`(point-in-time) 층으로 올라가 **학습용 정본**이 된다.
기존 묶음은 `bulk/` 로 옮겨 참조용으로만 남긴다 (scripts/reorganize_hf_dart.py).

🔴 DB 는 읽기 전용으로만 연다
-----------------------------
수집 세션이 같은 DB 에 쓰고 있을 수 있다. `mode=ro` URI 로만 열고,
어떤 경우에도 이 스크립트는 DB 에 쓰지 않는다.

출력
----
data/outbox/dart_pit_<오늘>/
  pit/<연도>_<보고서이름>.parquet   ← 기존 bulk 파일명 규칙(연도_보고서.parquet)과 통일
  MANIFEST.json                     ← krx-dev 와 같은 스키마(files[].sha256 등) + 수집 현황
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

#: 보고서 코드 → 파일명에 쓰는 이름. 기존 bulk 파일명(2015_사업보고서.parquet)과 맞춘다.
REPORT_NAMES = {
    "11011": "사업보고서",
    "11012": "반기보고서",
    "11013": "1분기보고서",
    "11014": "3분기보고서",
}

KST = timezone(timedelta(hours=9))


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="dart_financial 을 HF 반출용 parquet 로 내보낸다")
    parser.add_argument("--db", default="data/krx_cache.db")
    parser.add_argument("--out", default=None, help="출력 폴더 (기본: data/outbox/dart_pit_<오늘>)")
    args = parser.parse_args()

    db_path = Path(args.db).resolve()
    if not db_path.exists():
        print(f"DB 가 없다: {db_path}")
        return 1

    # 수집 세션과 동시에 돌 수 있으므로 반드시 읽기 전용으로 연다
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)

    today = datetime.now(KST).strftime("%Y%m%d")
    root = Path(args.out) if args.out else Path("data/outbox") / f"dart_pit_{today}"
    (root / "pit").mkdir(parents=True, exist_ok=True)

    # ── 연도×보고서 단위로 쪼개 내보낸다 ────────────────────────────
    groups = pd.read_sql_query(
        "SELECT bsns_year, reprt_code, COUNT(*) AS n FROM dart_financial "
        "GROUP BY bsns_year, reprt_code ORDER BY bsns_year, reprt_code",
        conn,
    )
    if groups.empty:
        print("dart_financial 이 비어 있다")
        return 1

    files: List[Dict] = []
    for _, g in groups.iterrows():
        year, code = int(g["bsns_year"]), str(g["reprt_code"])
        name = REPORT_NAMES.get(code, code)
        df = pd.read_sql_query(
            "SELECT * FROM dart_financial WHERE bsns_year = ? AND reprt_code = ?",
            conn,
            params=(year, code),
        )
        path = root / "pit" / f"{year}_{name}.parquet"
        df.to_parquet(path, index=False, compression="zstd")
        files.append({
            "path": f"pit/{path.name}",
            "rows": int(len(df)),
            "size_mb": round(path.stat().st_size / 1024 / 1024, 2),
            "sha256": sha256_of(path),
            "note": f"사업연도 {year} {name} — 접수일(rcept_dt) 포함, 학습용 정본",
            "columns": list(df.columns),
        })
        print(f"  ✅ {path.name:28s} {len(df):>8,}행")

    # ── 수집 현황을 dart_financial 자체에서 계산한다 ─────────────────
    # (수집 대장 표에 의존하지 않는다 — 스키마가 수집 세션 소관이라 바뀔 수 있다)
    stats = pd.read_sql_query(
        "SELECT COUNT(*) AS rows, COUNT(DISTINCT corp_code) AS corps, "
        "MIN(bsns_year) AS y0, MAX(bsns_year) AS y1, "
        "MIN(rcept_dt) AS r0, MAX(rcept_dt) AS r1 FROM dart_financial",
        conn,
    ).iloc[0]
    conn.close()

    manifest = {
        "generated_at": datetime.now(KST).isoformat(timespec="seconds"),
        "source": "OpenDART 단일회사 전체 재무제표 API (접수일 포함 재수집분)",
        "layer": "pit",
        "why_pit": "결산기가 아니라 접수일(rcept_dt) 기준으로 붙여야 미래 누출이 없다. "
                   "bulk/ 층에는 접수일이 없어 학습에 쓰면 안 된다.",
        "coverage": {
            "rows": int(stats["rows"]),
            "companies": int(stats["corps"]),
            "bsns_year": [int(stats["y0"]), int(stats["y1"])],
            "rcept_dt": [str(stats["r0"]), str(stats["r1"])],
        },
        "files": files,
    }
    (root / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    total_mb = sum(f["size_mb"] for f in files)
    print()
    print(f"MANIFEST.json 기록 — 파일 {len(files)}개 · 합계 {total_mb:,.1f} MB → {root}")
    print("다음 단계: scripts/reorganize_hf_dart.py 로 bulk/ 이동 + pit/ 업로드 (사용자 확인 후)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
