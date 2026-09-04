"""신원·법인·재무·거시·달력을 **종류별 폴더**로 반출한다 — HF 참조 자료의 정본 스크립트.

    python scripts/export_reference_dataset.py                    # data/outbox/reference_<오늘>/
    python scripts/export_reference_dataset.py --upload --dry-run # 무엇을 올릴지만 본다
    python scripts/export_reference_dataset.py --upload           # 검증기 통과 → HF 루트에 덧붙임

왜 이 스크립트가 생겼나
----------------------
HF(`qurious-quant/alphastack-krx-dev`)의 `identity/`·`financial/`·`macro/`·`calendar/` 는
2026-09-03 에 **스크래치패드의 임시 스크립트**로 만들어 올렸다. 정본이 저장소에 없으면
다음에 같은 것을 다시 만들 수 없고, 무엇을 어떻게 잘랐는지도 MANIFEST 의 SHA 로만 남는다.
그래서 그 스크립트를 여기로 옮기고 시험을 붙였다.

폴더를 종류로 나누는 이유
------------------------
한 폴더에 쏟으면 파일 이름으로만 구분해야 하고, `hf_hub_download(filename=...)` 을 쓰는
쪽에서 무엇이 한 벌인지 알 수 없다. 폴더가 곧 "이건 무슨 자료인가" 의 답이 되게 한다.

    identity/    종목 신원 · 법인 개요   — 종목코드 ↔ ISIN ↔ 법인등록번호를 잇는 다리
    financial/   DART 재무
    macro/       거시 지표
    calendar/    거래일 달력

`full/`·`small/` 은 **건드리지 않는다.** 팀원 코드가 그 경로를 그대로 쓴다.

🔴 홀드아웃은 나가지 않는다
--------------------------
경계는 `evaluation.horizon.HOLDOUT_START` **하나만** 본다. 2026-09-02 의 DART 반출본이
자기 상수(20210901)를 들고 있다가 정본(20240901)과 갈라져, 개발구간 재무 498,059행 중
300,991행만 나갔다. 197,068행이 조용히 빠졌다. 그래서 여기서는 상수를 새로 정의하지 않고
가져다 쓰고, 다 만든 뒤 **파일을 다시 열어** 경계 넘은 행이 0인지 확인한다.

🔴 DB 는 읽기 전용으로만 연다
---------------------------
수집 세션이 같은 DB 에 쓰고 있을 수 있다. `mode=ro` 로만 열고 어떤 경우에도 쓰지 않는다.

🔴 업로드 — `upload_to_hf.py` 를 그대로 쓰지 않는 이유
--------------------------------------------------
그 스크립트는 루트에 `MANIFEST.json` 이 있어야 진행하고, 카드도 시세 반출을 전제로 새로
만든다. 여기서 올리는 것은 성격이 다른 자료라 둘 다 안 맞는다. 대신 **그쪽 안전장치는
그대로 가져온다** — ① private 을 서버에 다시 물어 확인 ② 반출 전 검증기 전수(붉으면
안 올라간다) ③ `--dry-run` ④ 올린 뒤 서버 목록을 다시 받아 확인. 그리고 하나 더,
**이름 충돌 검사** — 루트로 올라가므로 `--replace` 없이는 기존 파일을 덮지 않는다.

파일명은 `MANIFEST_reference.json` 이다. `MANIFEST.json` 으로 두면 시세 반출의 이력을
덮는다 — 행 수로는 안 잡히는 사고다.

카드는 여기서 만지지 않는다. `scripts/hf_card_reference_block.md` 를 고치고
`upload_to_hf.py` 가 카드를 재생성한다 (파일만 올리면 팀원이 옛 설명을 본다).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from common.paths import krx_db_path  # noqa: E402
from evaluation.horizon import HOLDOUT_START  # noqa: E402

KST = timezone(timedelta(hours=9))
DEFAULT_REPO = "qurious-quant/alphastack-krx-dev"
MANIFEST_NAME = "MANIFEST_reference.json"

#: 수집 시각 칸. 팀원에게 쓸모가 없고, 다시 돌릴 때마다 값이 달라져 해시가 흔들린다.
JUNK_COLUMNS = ("fetched_at", "collected_at", "built_at")

#: (폴더, 파일명, 표, 시간 칸, 설명, 출처). 시간 칸으로 `< HOLDOUT_START` 를 자른다.
#:
#: ⚠️ `corp_profile` 의 시간 칸이 `known_at`(= `fst_opeg_dt`) 인 이유 — 그 표에는 `bas_dd`
#:    가 없다. 유효구간이 시작된 날이 곧 "언제부터 알 수 있었나" 다.
SPECS: Tuple[Tuple[str, str, str, str, str, str], ...] = (
    ("identity", "stock_identity_dev.parquet", "stock_identity", "bas_dd",
     "종목 신원 — 기준일마다 종목코드·ISIN·법인등록번호·법인명·시장. "
     "종목코드와 DART 를 잇는 다리다.",
     "공공데이터포털 금융위원회_KRX상장종목정보 (공공누리 1유형)"),
    ("identity", "corp_profile_dev.parquet", "corp_profile", "known_at",
     "법인 개요 — 설립일·상장일·감사의견·종업원수. "
     "유효구간이 (fst_opeg_dt, last_opeg_dt) 라 이력으로 쌓인다.",
     "공공데이터포털 금융위원회_기업기본정보 (공공누리 1유형)"),
    ("financial", "dart_financial_dev.parquet", "dart_financial", "rcept_dt",
     "DART 재무 — 접수일 기준. 기본키에 account_detail 이 들어간다.",
     "금융감독원 DART OpenAPI"),
    ("macro", "macro_series_dev.parquet", "macro_series", "known_at",
     "거시 지표 — 기준금리·환율 등. known_at 은 발표일을 우리가 계산한 값이다.",
     "한국은행 ECOS · FRED · KOSIS"),
    ("calendar", "trading_calendar_dev.parquet", "trading_calendar", "bas_dd",
     "거래일 달력 — 실측이다. 시세가 있는 날을 그대로 모았다.",
     "KRX OpenAPI 응답에서 파생"),
)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _connect_ro(db_path: Path) -> sqlite3.Connection:
    """읽기 전용. 쓰려 하면 sqlite 가 거절한다 — 실수로도 못 쓰게."""
    return sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True, timeout=30)


# ==================================================
# 1. 반출
# ==================================================
def export(db_path: Path, out: Path, *, holdout_start: str = HOLDOUT_START,
           specs: Sequence[Tuple[str, str, str, str, str, str]] = SPECS) -> List[Dict]:
    """표마다 `시간칸 < holdout_start` 로 잘라 parquet 으로 쓰고, MANIFEST 항목 목록을 준다."""
    conn = _connect_ro(db_path)
    항목들: List[Dict] = []
    try:
        for folder, fname, table, tcol, note, source in specs:
            dest = out / folder
            dest.mkdir(parents=True, exist_ok=True)
            path = dest / fname

            # 표·칸 이름은 이 파일 안 상수뿐이라 문자열로 끼워도 주입이 안 된다.
            df = pd.read_sql_query(f"SELECT * FROM {table} WHERE {tcol} < ?", conn,
                                   params=(holdout_start,))
            df = df.drop(columns=[c for c in JUNK_COLUMNS if c in df.columns])
            df.to_parquet(path, index=False, compression="zstd")

            항목들.append({
                "path": f"{folder}/{fname}",
                "table": table,
                "time_column": tcol,
                "rows": int(len(df)),
                "columns": list(df.columns),
                "range": [str(df[tcol].min()), str(df[tcol].max())] if len(df) else None,
                "bytes": path.stat().st_size,
                "size_mb": round(path.stat().st_size / 1e6, 3),
                "sha256": sha256_of(path),
                "note": note,
                "source": source,
            })
            print(f"  ✅ {folder}/{fname:32s} {len(df):>9,}행 "
                  f"{path.stat().st_size / 1e6:>8.2f}MB  "
                  f"{항목들[-1]['range'][0] if 항목들[-1]['range'] else '-'}~"
                  f"{항목들[-1]['range'][1] if 항목들[-1]['range'] else '-'}")
    finally:
        conn.close()
    return 항목들


def recheck_holdout(out: Path, entries: Sequence[Dict], *,
                    holdout_start: str = HOLDOUT_START) -> int:
    """🔴 **파일을 다시 열어** 경계 넘은 행을 센다. 만드는 쪽을 믿지 않는다."""
    나쁜행 = 0
    for e in entries:
        df = pd.read_parquet(out / e["path"])
        넘은것 = int((df[e["time_column"]].astype(str) >= holdout_start).sum())
        print(f"  {'✅' if 넘은것 == 0 else '🔴'} {e['path']:44s} 경계 넘은 행 {넘은것}")
        나쁜행 += 넘은것
    return 나쁜행


def write_manifest(out: Path, entries: Sequence[Dict], *, db_path: Path,
                   holdout_start: str = HOLDOUT_START) -> Path:
    manifest = {
        "generated_at": datetime.now(KST).isoformat(timespec="seconds"),
        "holdout_start": holdout_start,
        "holdout_start_authority": "evaluation/horizon.py HOLDOUT_START",
        "db": str(db_path),
        "files": list(entries),
    }
    path = out / MANIFEST_NAME
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")
    return path


# ==================================================
# 2. 업로드 — 안전장치는 upload_to_hf.py 와 같다
# ==================================================
def _run_verifiers():
    """`scripts/upload_to_hf.py` 의 검증기 게이트를 **빌려 쓴다.** 목록의 정본은 그쪽이다.

    `scripts/` 는 패키지가 아니라 파일 경로로 읽는다. 검증기 목록을 여기 복사하면 둘이
    갈라지고, 갈라진 쪽은 붉은 자료를 통과시킨다.
    """
    spec = importlib.util.spec_from_file_location(
        "upload_to_hf", ROOT / "scripts" / "upload_to_hf.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_verifiers()


def upload(root: Path, *, repo: str = DEFAULT_REPO, dry_run: bool = False,
           replace: bool = False, skip_verify: bool = False, note: str = "") -> int:
    from ingest.clients import hf_data

    files = [p for p in sorted(root.rglob("*")) if p.is_file()]
    if not files:
        print(f"{root} 에 올릴 파일이 없다")
        return 1
    token, source = hf_data.load_hf_key()
    if not token:
        print("HUGGINGFACE_ACCESS_TOKEN 이 없다 (.env 확인)")
        return 1
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    print(f"토큰 출처 {source} · 레포 {repo}")

    # ① private — 만들었다고 믿지 않고 서버에 묻는다
    info = api.repo_info(repo_id=repo, repo_type="dataset")
    if not info.private:
        print("🔴 중단 — public 이다. KRX 원자료가 든 레포는 private 이어야 한다.")
        return 1
    print(f"✅ private 확인 · 현재 sha {info.sha[:12]}")

    # ② 반출 전 검증기 (규약 v3.3 §7.1) — 고친 것과 나간 것은 다르다
    if skip_verify:
        print("⚠️ --skip-verify — 검증기를 건너뛴다. 카드에 그 사실을 적을 것")
    else:
        print("── 반출 전 검사 ──")
        붉은것 = []
        for 경로, 설명, 코드, 마지막 in _run_verifiers():
            print(f"  {'✅' if 코드 == 0 else '🔴'} {설명:<36} {마지막[:70]}")
            if 코드 != 0:
                붉은것.append((경로, 설명))
        if 붉은것:
            print(f"🔴 중단 — 검사 {len(붉은것)}건이 붉다. 고치고 다시 실행한다.")
            return 1
        print("✅ 검사 전부 통과")

    # ③ 이름 충돌 — 루트로 올리므로 겹치면 조용히 덮어쓴다
    있는파일 = set(api.list_repo_files(repo_id=repo, repo_type="dataset"))
    올릴것 = [p.relative_to(root).as_posix() for p in files]
    충돌 = sorted(set(올릴것) & 있는파일)
    print(f"\n올릴 파일 {len(files)}개 · 합계 {sum(p.stat().st_size for p in files) / 1e6:.1f}MB")
    for p in files:
        rel = p.relative_to(root).as_posix()
        표시 = "갱신(덮어씀)" if rel in 있는파일 else "신규"
        print(f"   {rel:44s} {p.stat().st_size / 1e6:>8.2f}MB  {표시}")
    if 충돌 and not replace:
        print(f"\n🔴 중단 — 기존 파일 {len(충돌)}개와 이름이 겹친다. "
              "덮어쓸 의도면 --replace 를 준다.")
        return 1
    if dry_run:
        print("\n--dry-run 이라 여기서 멈춘다")
        return 0

    메시지 = f"참조 자료 반출 {root.name} (신원·법인·재무·거시·달력)"
    if note:
        메시지 += f" — {note}"
    print(f"\n올리는 중… ({메시지})")
    api.upload_folder(folder_path=str(root), repo_id=repo, repo_type="dataset",
                      commit_message=메시지)

    # ④ 올라갔는지 서버에 다시 묻는다
    뒤 = set(api.list_repo_files(repo_id=repo, repo_type="dataset"))
    빠진것 = [f for f in 올릴것 if f not in 뒤]
    if 빠진것:
        print(f"🔴 안 올라간 것 {len(빠진것)}: {빠진것}")
        return 1
    print(f"✅ 올린 파일 전부 서버에서 확인됨 · 서버 파일 {len(뒤)}개")
    print(f"   https://huggingface.co/datasets/{repo}")
    print("   할 일: scripts/hf_card_reference_block.md 를 고쳤다면 "
          "upload_to_hf.py 로 카드도 갱신한다")
    return 0


# ==================================================
# 3. 진입점
# ==================================================
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="참조 자료(신원·법인·재무·거시·달력) 종류별 반출")
    ap.add_argument("--db", default=None, help="기본 data/krx_cache.db (읽기 전용으로 연다)")
    ap.add_argument("--out", default=None, help="기본 data/outbox/reference_<오늘>")
    ap.add_argument("--upload", action="store_true", help="반출 뒤 HF 루트에 덧붙인다")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--dry-run", action="store_true", help="올리지 않고 무엇을 올릴지만")
    ap.add_argument("--replace", action="store_true", help="이름이 겹치는 기존 파일을 덮어쓴다")
    ap.add_argument("--skip-verify", action="store_true",
                    help="🔴 검증기를 건너뛴다. 되도록 쓰지 않는다")
    ap.add_argument("--note", default="", help="커밋 메시지에 덧붙일 한 줄")
    args = ap.parse_args(argv)

    db = Path(args.db) if args.db else krx_db_path()
    out = Path(args.out) if args.out else (
        ROOT / "data" / "outbox" / f"reference_{datetime.now(KST):%Y%m%d}")
    print(f"경계 HOLDOUT_START = {HOLDOUT_START} (evaluation/horizon.py)")
    print(f"DB {db} (읽기 전용) → {out}\n")

    entries = export(db, out)
    print("\n홀드아웃 재검사 (파일을 다시 읽는다)")
    if recheck_holdout(out, entries):
        print("\n🔴 홀드아웃이 섞였다. 올리지 않는다.")
        return 1
    manifest = write_manifest(out, entries, db_path=db)
    print(f"\n{manifest.name} 씀 · 파일 {len(entries)}개 · "
          f"합계 {sum(e['bytes'] for e in entries) / 1e6:.1f}MB")

    if not args.upload:
        return 0
    print()
    return upload(out, repo=args.repo, dry_run=args.dry_run, replace=args.replace,
                  skip_verify=args.skip_verify, note=args.note)


if __name__ == "__main__":
    raise SystemExit(main())
