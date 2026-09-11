"""봉인 홀드아웃 반출 — 개발본과 **같은 함수**로, 개봉 기록과 함께만 만든다 (이슈 #240).

    python scripts/export_holdout_dataset.py --dry-run          # 가짜 홀드아웃 (기록 없음)
    python scripts/unseal_holdout.py open ...                   # 진짜는 여기서만

## 왜 CLI 로는 진짜를 못 만드나

진짜 홀드아웃 반출은 곧 **개봉**이다. 반출만 따로 돌릴 수 있으면 기록 없이 한 번 열린다.
그래서 이 파일의 CLI 는 드라이런만 받고, 진짜는 `scripts/unseal_holdout.py open` 이 사전
조건(사전등록 커밋 · 깨끗한 작업 트리 · 데이터 게이트)을 확인한 뒤 `export_holdout()` 을
부르고 곧바로 `reports/unseal.log` 에 한 줄을 쓴다.

## 무엇을 내나 (`data/sealed/<이름>/` · gitignore)

    full/daily_price_all.parquet   개발구간 + 홀드아웃 · 개발본 반출과 같은 칸 (파생 14칸 포함)
    full/index_price_all.parquet   개발구간 + 홀드아웃 전 지수
    MANIFEST_holdout.json          구간 · 행 수 · 칸 · 결측 · 이상치 · SHA-256 · 커밋 · 품질 원장

개발구간을 함께 싣는 이유: 피처는 롤링 창이라 홀드아웃 첫날에도 **그 앞 행**이 필요하다
(`hv_regime` 준비 구간 269행). 품질·기업행위 판정도 종목의 앞뒤 행을 보므로 잘라 읽으면
경계에서 답이 달라진다 — 그래서 `build_full_daily` 는 전 구간을 읽는다.

## 드라이런

가짜 홀드아웃 시작(기본: 개발구간 끝에서 60거래일 전 — 오준영 님 드라이런 준비기와 같은 길이)을
삼고 **개발구간 끝(`DEV_END`)까지만** 읽는다. 진짜 봉인 구간은 한 줄도 읽지 않는다 — 반출 뒤
`verify_no_holdout` 으로 파일을 다시 열어 확인한다. 범위 검사는 **DB 를 읽기 전에** 한다.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from common.trading_calendar import load_session_days  # noqa: E402
from evaluation.horizon import HOLDOUT_START  # noqa: E402
from ingest.store import krx_store  # noqa: E402
from scripts.export_team_dataset import (  # noqa: E402
    DEV_END,
    _utcnow,
    _write_parquet,
    build_full_daily,
    read_full_index,
    verify_no_holdout,
)
from supply.adj_quality import FLAG_COLUMNS  # noqa: E402
from supply.holdout import HOLDOUT_DAILY, HOLDOUT_INDEX, HOLDOUT_MANIFEST  # noqa: E402
from supply.training import CORPORATE_ACTION_COLUMNS, market_context  # noqa: E402

#: 봉인 반출본을 두는 곳. `.gitignore` 의 `data/sealed/` 가 통째로 막는다.
DEFAULT_SEALED_ROOT = ROOT / "data" / "sealed"

#: 드라이런의 가짜 홀드아웃 길이. `scripts/prepare_final_holdout_dry_run.py` 와 같은 60거래일.
DRY_RUN_SESSIONS = 60

#: 대장에 결측을 세어 적는 칸.
MISSING_COLUMNS = ("adj_open", "adj_high", "adj_low", "adj_close", "industry",
                   "kind_stkcert_tp_nm")


def git_commit() -> str:
    """반출한 코드의 커밋. git 이 없으면 'unknown' — 반출은 막지 않고 대장에 모른다고 적는다."""
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def db_schema_version() -> Optional[int]:
    with krx_store.connect() as conn:
        return int(conn.execute("PRAGMA user_version").fetchone()[0])


def quality_ledger_ref() -> Optional[Dict[str, object]]:
    """가장 최근 품질 원장 이력의 머리 — 이 반출이 어느 품질 판정 위에 섰는지 잇는다."""
    try:
        from supply.quality_ledger import read_last_history
        last = read_last_history()
    except Exception:                                   # noqa: BLE001 — 잇기 실패로 반출을 막지 않는다
        return None
    if not isinstance(last, dict):
        return None
    return {k: last.get(k) for k in ("run_id", "status", "generated_at") if k in last}


def dry_run_start(sessions: int = DRY_RUN_SESSIONS) -> str:
    """개발구간 끝에서 `sessions` 거래일 전 — 드라이런의 가짜 홀드아웃 시작."""
    days = sorted(d for d in load_session_days() if d <= DEV_END)
    if len(days) < sessions:
        raise ValueError(f"개발구간 거래일이 {len(days)}일뿐이라 {sessions}일을 뗄 수 없다.")
    return days[-sessions]


def _day_before(day: str) -> str:
    return (datetime.strptime(day, "%Y%m%d") - timedelta(days=1)).strftime("%Y%m%d")


def validate_range(*, holdout_start: str, end: str, dry_run: bool) -> None:
    """범위를 **DB 를 읽기 전에** 막는다 — 드라이런은 봉인에 안 닿게, 진짜는 정본대로."""
    for 이름, 값 in (("holdout_start", holdout_start), ("end", end)):
        if len(str(값)) != 8 or not str(값).isdigit():
            raise ValueError(f"{이름} 은 YYYYMMDD 여야 한다: {값!r}")
    # 봉인 경계를 먼저 본다 — 시작·끝 순서 오류보다 "봉인에 닿는다" 가 더 급한 소식이다.
    if dry_run and holdout_start >= HOLDOUT_START:
        raise ValueError(
            f"🔴 드라이런의 가짜 홀드아웃 시작({holdout_start})은 진짜 봉인 시작"
            f"({HOLDOUT_START})보다 앞이어야 한다.")
    if holdout_start > end:
        raise ValueError(f"holdout_start({holdout_start}) 가 end({end}) 보다 늦다.")
    if dry_run:
        if end > DEV_END:
            raise ValueError(f"🔴 드라이런은 개발구간 끝({DEV_END})까지만 읽는다 — end={end}")
    elif holdout_start != HOLDOUT_START:
        raise ValueError(
            f"진짜 반출의 홀드아웃 시작({holdout_start})이 코드 정본({HOLDOUT_START})과 다르다.")


def segment_quality(daily: pd.DataFrame, *, start: str, end: str) -> Dict[str, object]:
    """구간(홀드아웃 또는 가짜 홀드아웃)의 행 수·결측·이상치 — 대장에 적는 값."""
    dates = daily["bas_dd"].astype(str)
    seg = daily.loc[(dates >= start) & (dates <= end)]
    return {
        "rows": int(len(seg)),
        "trading_days": int(seg["bas_dd"].nunique()),
        "codes": int(seg["code"].nunique()),
        "first_bas_dd": str(seg["bas_dd"].min()) if len(seg) else None,
        "last_bas_dd": str(seg["bas_dd"].max()) if len(seg) else None,
        "missing": {c: int(seg[c].isna().sum()) for c in MISSING_COLUMNS if c in seg.columns},
        "flags": {c: int(seg[c].fillna(False).astype(bool).sum())
                  for c in (*FLAG_COLUMNS, *CORPORATE_ACTION_COLUMNS) if c in seg.columns},
    }


def export_holdout(root: Path, *, holdout_start: str, end: str, dry_run: bool, as_of: str,
                   note: str = "") -> Dict[str, object]:
    """봉인 반출본을 만들고 대장을 돌려준다. 이미 무언가 든 폴더에는 쓰지 않는다."""
    validate_range(holdout_start=holdout_start, end=end, dry_run=dry_run)
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            f"{root} 가 비어 있지 않다 — 봉인 반출본은 덮어쓰지 않는다 "
            "(다시 만들면 지문이 달라져 개봉 기록과 어긋난다).")

    print(f"[봉인 반출] {'드라이런' if dry_run else '🔴 진짜'} · 구간 {holdout_start} ~ {end}")
    daily, stats = build_full_daily(end=end, as_of=as_of, ctx=market_context())
    구간 = segment_quality(daily, start=holdout_start, end=end)
    if 구간["rows"] == 0:
        raise RuntimeError(f"{holdout_start} ~ {end} 에 시세 행이 없다 — 수집부터 확인한다.")

    files: List[Dict] = []
    _write_parquet(daily, root / HOLDOUT_DAILY, files,
                   "개발구간 + 홀드아웃 전 종목 시세 · 개발본 반출과 같은 칸(build_full_daily)")
    files[-1]["path"] = HOLDOUT_DAILY
    del daily
    idx = read_full_index(end=end)
    구간["index_rows"] = int((idx["bas_dd"].astype(str) >= holdout_start).sum())
    _write_parquet(idx, root / HOLDOUT_INDEX, files, "개발구간 + 홀드아웃 전 지수")
    files[-1]["path"] = HOLDOUT_INDEX
    del idx

    if dry_run:
        검사수 = verify_no_holdout(root)
        print(f"  ✅ 드라이런 파일 {검사수}개를 다시 열어 확인 — 진짜 봉인 구간 없음")

    manifest = {
        "schema_version": 1,
        "kind": "holdout_snapshot",
        "dry_run": bool(dry_run),
        "generated_at": _utcnow(),
        "holdout_start": holdout_start,
        "sealed_holdout_start": HOLDOUT_START,
        "dev_end": _day_before(holdout_start),
        "end": end,
        "as_of": as_of,
        "git_commit": git_commit(),
        "db_schema_version": db_schema_version(),
        "files": files,
        "holdout_segment": 구간,
        "stats": stats,
        "quality_ledger": quality_ledger_ref(),
        "note": note,
    }
    (root / HOLDOUT_MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    print(f"  ✅ {HOLDOUT_MANIFEST} · 구간 {구간['rows']:,}행 · {구간['trading_days']}거래일 · "
          f"{구간['codes']:,}종")
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="봉인 홀드아웃 반출 — CLI 는 드라이런만 받는다")
    p.add_argument("--dry-run", action="store_true", help="가짜 홀드아웃으로 반출한다 (필수)")
    p.add_argument("--dry-run-start", default=None,
                   help=f"가짜 홀드아웃 시작 YYYYMMDD "
                        f"(기본: 개발구간 끝 {DRY_RUN_SESSIONS}거래일 전)")
    p.add_argument("--out", default=None, help="출력 폴더 (기본 data/sealed/dryrun_<시작>_<끝>)")
    args = p.parse_args(argv)

    if not args.dry_run:
        print("🔴 진짜 홀드아웃 반출은 이 명령으로 하지 않는다 — "
              "반출이 곧 개봉이라 기록과 함께여야 한다.")
        print("   할 일: python scripts/unseal_holdout.py open "
              "--requested-by <이름> --reason <사유>")
        return 2

    start = args.dry_run_start or dry_run_start()
    root = Path(args.out) if args.out else DEFAULT_SEALED_ROOT / f"dryrun_{start}_{DEV_END}"
    export_holdout(root, holdout_start=start, end=DEV_END, dry_run=True,
                   as_of=datetime.now().strftime("%Y%m%d"),
                   note="CLI 드라이런 — 개봉 기록 없음")
    print(f"저장: {root}")
    print("  이 반출본은 기록이 없어 load_holdout_snapshot 으로 열리지 않는다.")
    print("  통합 드라이런은 python scripts/unseal_holdout.py open --dry-run 으로 한다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
