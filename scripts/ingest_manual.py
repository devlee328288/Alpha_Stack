"""손으로 받아 정리한 파일을 반입 엔진에 들인다 — `data/manual/normalized/<종류>/`.

    python scripts/ingest_manual.py --kind sector             # 업종분류 현황 스냅샷 전부
    python scripts/ingest_manual.py --kind sector --dry-run   # 검사만, DB 에 안 담는다
    python scripts/ingest_manual.py --kind sector --force     # 이미 들인 것도 다시

`scripts/check_inbox.py` 는 **팀원이 건네준 파일**(`data/inbox/` · HuggingFace)을 훑는다.
팀장이 사이트에서 클릭해 받은 파일은 자리도 다르고(`data/manual/`) 출처 표기도 달라야
한다 — 값이 우리 것과 다를 때 **누구에게 물어야 하는지**가 그 표기에서 갈린다. 그래서
검사·저장은 같은 `handle()` 을 쓰되 입구를 따로 둔다.

🔴 `inbox_batch.origin` 은 CHECK 제약으로 `local`·`huggingface` 둘뿐이다 — 시험이 잡았다.
   손으로 받은 것은 `origin="local"` 로 들이고 **출처는 `contributor` 에 적는다.**
   `data/manual/README.md` 가 처음부터 그렇게 정해 두었다 (표: "contributor 에 출처를 적는다").

원본이 아니라 **변환본**을 들인다. 원본은 cp949 이고 업종분류 현황은 기준일 칸이 없어
`scripts/normalize_manual.py` 가 종가로 되짚어 붙인 변환본이 반입 규격에 맞는 모양이다.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ingest.inbox.engine import available_kinds  # noqa: E402

NORMALIZED_DIR = ROOT / "data" / "manual" / "normalized"

#: `inbox_batch.origin`. 제약이 local·huggingface 둘뿐이라 손으로 받은 것도 local 이다.
ORIGIN = "local"
#: 출처는 여기에 — 같은 자료가 다르면 이 화면으로 돌아가 다시 받는다.
#: `huggingface` 배치는 contributor 가 팀원 이름이라, 이 값이 곧 "팀장이 화면에서" 라는 표식이다.
CONTRIBUTOR = "이동원 · KRX Data Marketplace 화면 다운로드 (data/manual)"


def _load_check_inbox():
    """`scripts/check_inbox.py` 의 `handle()` 을 빌려 쓴다 (scripts 는 패키지가 아니다)."""
    spec = importlib.util.spec_from_file_location(
        "check_inbox", Path(__file__).resolve().parent / "check_inbox.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def target_files(kind: str) -> List[Path]:
    folder = NORMALIZED_DIR / kind
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.glob("*.csv") if p.is_file())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="손으로 받아 정리한 파일을 반입 엔진에 들인다")
    parser.add_argument("--kind", required=True, choices=available_kinds(), help="규격 이름")
    parser.add_argument("--dry-run", action="store_true", help="검사만 하고 DB 에 담지 않는다")
    parser.add_argument("--force", action="store_true", help="이미 들인 파일도 다시 검사한다")
    args = parser.parse_args(argv)

    files = target_files(args.kind)
    if not files:
        print(f"들일 파일이 없습니다: {NORMALIZED_DIR / args.kind}")
        print("  할 일: data/manual/<종류>/ 에 원본을 두고 "
              "python scripts/normalize_manual.py 를 먼저 돌린다.")
        return 1

    check_inbox = _load_check_inbox()
    print(f"── {args.kind} · {len(files)}개 ({'검사만' if args.dry_run else '검사 후 저장'}) ──")

    tally = {"ingested": 0, "skipped": 0, "rejected": 0, "error": 0}
    rows_accepted = rows_quarantined = 0
    for path in files:
        out = check_inbox.handle(path, kind=args.kind, origin=ORIGIN, contributor=CONTRIBUTOR,
                                 dry_run=args.dry_run, force=args.force)
        status = out["status"]
        tally[status] = tally.get(status, 0) + 1
        if status == "skipped":
            print(f"  · {path.name}  건너뜀 — {out['note']}")
            continue
        if status in ("error", "rejected"):
            print(f"  🔴 {path.name}  {status} — {out.get('note') or out.get('rejected')}")
            continue
        rows_accepted += out["rows_accepted"]
        rows_quarantined += out["rows_quarantined"]
        표시 = "✅" if out["rows_quarantined"] == 0 else "⚠️"
        꼬리 = "" if args.dry_run else f"  → {out['batch_id']}"
        print(f"  {표시} {path.name}  통과 {out['rows_accepted']:,} · "
              f"격리 {out['rows_quarantined']:,}{꼬리}")
        for q in out.get("questions") or []:
            print(f"      ❓ {q}")

    print(f"\n완료 — 들임 {tally['ingested']} · 건너뜀 {tally['skipped']} · "
          f"거부 {tally['rejected']} · 오류 {tally['error']}"
          f"  (행 통과 {rows_accepted:,} · 격리 {rows_quarantined:,})")
    if rows_quarantined:
        print("  격리 사유는 data/inbox/_reports/ 의 보고서에 있다.")
    return 1 if (tally["rejected"] or tally["error"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
