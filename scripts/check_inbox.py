"""세션마다 한 번 — 팀원이 새로 올린 파일을 찾아 검사하고 들인다.

    python scripts/check_inbox.py                # 로컬 + HuggingFace 를 훑고 새 것만 들인다
    python scripts/check_inbox.py --dry-run      # 검사만 하고 DB 에는 안 담는다
    python scripts/check_inbox.py --local-only   # HuggingFace 는 보지 않는다
    python scripts/check_inbox.py --force        # 이미 들인 파일도 다시 검사한다
    python scripts/check_inbox.py --path 어떤.csv --kind ohlcv_stock   # 한 파일만

두 곳을 한꺼번에 훑는다
-----------------------
| 어디 | 경로 | 종류를 어떻게 아나 |
|---|---|---|
| 로컬 | `data/inbox/<종류>/*.csv` | **폴더 이름이 알려 준다** |
| HuggingFace | `inbox/<이름>/*.csv` | 폴더에 종류가 없다 → **규격 5장에 대 보고 잰다** |

HF 쪽에 종류 폴더가 없는 것은 팀원 가이드가 `inbox/<본인이름>/` 으로 안내했기 때문이다.
이름으로 찍지 않고 `engine.guess_kind()` 로 재며, **애매하면 정하지 않고 물어본다** —
틀린 규격으로 검사하면 멀쩡한 파일이 통째로 격리되고, 팀원은 자기 자료가 잘못됐다고 오해한다.

같은 파일을 두 번 들이지 않는다
-------------------------------
판단은 **내용 지문(SHA-256)** 으로 한다. 이름은 팀원이 바꿔 올리고, 수정 시각은 내려받을
때마다 새로 찍힌다. 규격을 고친 뒤 일부러 다시 검사하고 싶으면 `--force` 를 쓴다 — 그때
판정이 어떻게 달라졌는지가 곧 규격 개정의 근거다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# 저장소 어디서 실행해도 import 가 되게 한다.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest.inbox import report as report_mod  # noqa: E402
from ingest.inbox import store  # noqa: E402
from ingest.inbox.engine import (  # noqa: E402
    InboxError,
    available_kinds,
    guess_kind,
    inspect_file,
    read_table,
)

#: 로컬에서 훑는 자리 **둘**. 축이 다르다 — 누가 가져온 것인지가 갈린다.
#:   data/inbox/   팀원 3명이 HuggingFace 로 보낸 것
#:   data/manual/  팀장이 사이트에서 손으로 받아 온 것
#: 한 폴더에 섞여 있으면 값이 우리 것과 다를 때 누구에게 물어야 하는지 알 수 없다.
LOCAL_ROOT = Path("data/inbox")
MANUAL_ROOT = Path("data/manual")
LOCAL_ROOTS = (LOCAL_ROOT, MANUAL_ROOT)

HF_REPO = "qurious-quant/alphastack-krx-dev"
HF_PREFIX = "inbox/"

#: 검사 대상 확장자. 그 밖의 파일(README·메모)은 조용히 건너뛴다.
#: ⚠️ `.json` 은 직접 수집 때문에 넣었다 — ECOS·FRED·DART 화면 다운로드가 전부 JSON 이라,
#:    없으면 파일을 넣고 검사를 돌려도 **아무 말 없이 0건**이 나온다.
#: ⚠️ `.zip` 은 넣지 않는다. 풀어서 넣어야 무엇이 들었는지 사람이 보고 판단한다.
DATA_SUFFIXES = {".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".json"}


# ==================================================
# 1. 찾기
# ==================================================
def find_local(root=None) -> List[dict]:
    """로컬의 `<뿌리>/<종류>/` 아래 자료 파일을 모은다.

    뿌리를 안 주면 `data/inbox/` 와 `data/manual/` 둘 다 훑는다. 하나만 보고 싶으면
    경로를 하나 주면 된다 (테스트가 그렇게 쓴다).

    ⚠️ **HF 캐시를 건너뛴다.** `--cache-dir` 기본값이 `data/inbox/_hf` 라 스캔 뿌리
    안에 있고, `rglob` 이 그것까지 훑으면 같은 파일이 로컬로 한 번 더 잡힌다.
    """
    if root is None:
        roots = LOCAL_ROOTS
    elif isinstance(root, (str, Path)):
        roots = (Path(root),)
    else:
        roots = tuple(root)

    found: List[dict] = []
    kinds = set(available_kinds())

    for base in roots:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in DATA_SUFFIXES:
                continue
            # 밑줄로 시작하는 폴더는 우리가 만든 작업 공간이다 (`_hf` 캐시 등).
            if any(part.startswith("_") for part in path.parts):
                continue
            # 경로 어딘가에 종류 이름이 있으면 그것을 쓴다. 명시가 추측보다 언제나 낫다.
            kind = next((part for part in path.parts if part in kinds), None)
            # 어느 뿌리에서 왔는지를 기여자 칸에 남긴다 — origin 은 CHECK 제약이 걸려 있어
            # 'manual' 을 넣으려면 마이그레이션이 필요한데, 그만한 값은 아직 없다.
            manual = base == MANUAL_ROOT
            found.append({
                "path": path,
                "kind": kind,
                "origin": "local",
                "contributor": "직접수집" if manual else None,
            })
    return found


def find_huggingface(token_names=("HUGGINGFACE_ACCESS_TOKEN", "HF_TOKEN",
                                  "HUGGINGFACE_API_KEY")) -> tuple:
    """HF 데이터셋의 `inbox/` 를 훑는다. `(파일 목록, 안내 문구)`.

    ⚠️ **토큰이 없거나 네트워크가 막히면 막다른 길로 만들지 않는다.** 로컬만 훑고 그 사실을
    말한다 — 반입 검사가 네트워크 때문에 통째로 실패하면 손에 든 파일조차 못 들인다.
    """
    try:
        from huggingface_hub import HfApi

        from common import secrets
    except ImportError as error:
        return [], f"HuggingFace 를 보지 못했다 ({error}). 로컬만 훑는다."

    token, _ = secrets.load_key(token_names)
    if not token:
        return [], ("HuggingFace 토큰이 없어 로컬만 훑는다.\n"
                    "  넣으려면: .env 또는 .key 에 HUGGINGFACE_ACCESS_TOKEN=hf_... 한 줄.")

    try:
        api = HfApi(token=token)
        names = api.list_repo_files(repo_id=HF_REPO, repo_type="dataset")
    except Exception as error:                       # noqa: BLE001 — 원인을 그대로 전한다
        return [], f"HuggingFace 목록을 못 받았다 ({error}). 로컬만 훑는다."

    found: List[dict] = []
    for name in sorted(names):
        if not name.startswith(HF_PREFIX):
            continue
        if Path(name).suffix.lower() not in DATA_SUFFIXES:
            continue
        parts = name.split("/")
        # inbox/<이름>/... 에서 가운데가 보낸 사람이다.
        contributor = parts[1] if len(parts) > 2 else None
        found.append({"repo_file": name, "kind": None, "origin": "huggingface",
                      "contributor": contributor, "api": api})
    return found, ""


def download(entry: dict, cache_dir: Path) -> Path:
    """HF 파일을 내려받아 로컬 경로를 돌려준다."""
    from huggingface_hub import hf_hub_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=HF_REPO, repo_type="dataset", filename=entry["repo_file"],
        local_dir=str(cache_dir), token=entry["api"].token,
    )
    return Path(path)


# ==================================================
# 2. 한 파일 다루기
# ==================================================
def handle(path: Path, *, kind: Optional[str], origin: str, contributor: Optional[str],
           dry_run: bool, force: bool, db_path=None) -> dict:
    """파일 하나를 검사하고(필요하면) 담는다. 무슨 일이 있었는지 요약을 돌려준다."""
    digest = store.file_sha256(path)

    if not force:
        seen = store.already_ingested(digest, db_path)
        if seen:
            return {"status": "skipped", "path": str(path), "kind": seen["kind"],
                    "note": f"이미 들였다 ({seen['finished_at']})", "sha256": digest}

    if kind is None:
        try:
            frame = read_table(path)
        except InboxError as error:
            return {"status": "error", "path": str(path), "note": str(error).splitlines()[0],
                    "sha256": digest}
        kind, scores = guess_kind(frame)
        if kind is None:
            top = " · ".join(f"{s['kind']} {s['score']:.2f}" for s in scores[:3])
            return {"status": "unknown_kind", "path": str(path), "sha256": digest,
                    "note": f"어느 규격인지 정할 수 없다 (점수: {top})",
                    "hint": "--kind 로 알려 주거나 data/inbox/<종류>/ 아래에 둔다"}

    started_at = report_mod.datetime.now(report_mod.KST).isoformat(timespec="seconds")
    try:
        result = inspect_file(path, kind=kind)
    except InboxError as error:
        return {"status": "error", "path": str(path), "kind": kind, "sha256": digest,
                "note": str(error).splitlines()[0]}

    paths = report_mod.write_report(result, contributor=contributor)

    batch_id = None
    if not dry_run:
        batch_id = store.load_result(
            result, path, db_path=db_path, origin=origin, contributor=contributor,
            started_at=started_at, report_path=paths["markdown"], sha256=digest,
        )
        # 묶음 번호가 붙은 뒤 보고서를 한 번 더 써서 둘을 잇는다.
        paths = report_mod.write_report(result, batch_id=batch_id, contributor=contributor)

    return {
        "status": "rejected" if result.rejected else "ingested",
        "path": str(path), "source": str(path), "kind": kind, "sha256": digest,
        "contributor": contributor, "batch_id": batch_id,
        "rows_total": result.rows_total,
        "rows_accepted": len(result.accepted),
        "rows_quarantined": len(result.quarantined),
        "rejected": result.rejected,
        "questions": result.questions,
        "report": paths["markdown"],
    }


# ==================================================
# 3. 실행
# ==================================================
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="팀원이 올린 파일을 찾아 검사하고 들인다.")
    parser.add_argument("--path", help="이 파일 하나만 검사한다")
    parser.add_argument("--kind", choices=available_kinds(), help="종류를 직접 지정한다")
    parser.add_argument("--dry-run", action="store_true", help="검사만 하고 DB 에 담지 않는다")
    parser.add_argument("--local-only", action="store_true", help="HuggingFace 를 보지 않는다")
    parser.add_argument("--force", action="store_true", help="이미 들인 파일도 다시 검사한다")
    parser.add_argument("--cache-dir", default="data/inbox/_hf",
                        help="HuggingFace 파일을 내려받을 자리")
    args = parser.parse_args(argv)

    entries: List[dict] = []
    notices: List[str] = []

    if args.path:
        entries.append({"path": Path(args.path), "kind": args.kind,
                        "origin": "local", "contributor": None})
    else:
        entries.extend(find_local())
        if not args.local_only:
            remote, notice = find_huggingface()
            if notice:
                notices.append(notice)
            entries.extend(remote)

    print("── 반입 확인 ──────────────────────────────")
    for notice in notices:
        print(f"  ℹ️ {notice}")
    local_count = sum(1 for e in entries if e["origin"] == "local")
    remote_count = len(entries) - local_count
    print(f"  로컬 {local_count}개 · HuggingFace {remote_count}개")
    if not entries:
        print("  들일 파일이 없다.")
        print(f"  ▸ 요약: {report_mod.write_summary([])}")
        return 0
    print()

    results: List[dict] = []
    for entry in entries:
        if entry["origin"] == "huggingface":
            try:
                path = download(entry, Path(args.cache_dir))
            except Exception as error:              # noqa: BLE001 — 한 파일 때문에 멈추지 않는다
                print(f"  ❌ {entry['repo_file']} — 내려받지 못했다: {error}")
                continue
        else:
            path = entry["path"]

        if not path.exists():
            print(f"  ❌ {path} — 파일이 없다")
            continue

        outcome = handle(
            path, kind=args.kind or entry["kind"], origin=entry["origin"],
            contributor=entry["contributor"], dry_run=args.dry_run, force=args.force,
        )
        results.append(outcome)
        _print_outcome(outcome)

    ingested = [r for r in results if r["status"] in ("ingested", "rejected")]
    summary_path = report_mod.write_summary(ingested)

    print()
    print("── 정리 ───────────────────────────────────")
    print(f"  들임 {sum(1 for r in results if r['status'] == 'ingested')}"
          f" · 되돌림 {sum(1 for r in results if r['status'] == 'rejected')}"
          f" · 건너뜀 {sum(1 for r in results if r['status'] == 'skipped')}"
          f" · 종류 불명 {sum(1 for r in results if r['status'] == 'unknown_kind')}"
          f" · 오류 {sum(1 for r in results if r['status'] == 'error')}")
    if args.dry_run:
        print("  ⚠️ --dry-run 이라 DB 에는 담지 않았다.")
    print(f"  ▸ 요약: {summary_path}")

    questions = [(r["path"], q) for r in ingested for q in (r.get("questions") or [])]
    if questions:
        print()
        print("  ❓ 물어볼 것 — 이름만 보고는 정할 수 없어 남겨 뒀다")
        for path, question in questions:
            candidates = " 또는 ".join(question["candidates"]) or "—"
            print(f"     {Path(path).name} · {question['column']} → {candidates}")

    return 0


def _print_outcome(outcome: dict) -> None:
    name = Path(outcome["path"]).name
    status = outcome["status"]
    if status == "skipped":
        print(f"  ⏭️  {name} — {outcome['note']}")
    elif status == "unknown_kind":
        print(f"  ❓ {name} — {outcome['note']}")
        print(f"      할 일: {outcome['hint']}")
    elif status == "error":
        print(f"  ❌ {name} — {outcome['note']}")
    elif status == "rejected":
        print(f"  ↩️  {name} [{outcome['kind']}] — 파일째 되돌림")
        print(f"      {(outcome['rejected'] or '').splitlines()[0]}")
        print(f"      ▸ {outcome['report']}")
    else:
        total = outcome["rows_total"]
        accepted = outcome["rows_accepted"]
        quarantined = outcome["rows_quarantined"]
        mark = "✅" if quarantined == 0 else "⚠️"
        print(f"  {mark} {name} [{outcome['kind']}] — {total:,}행 중 "
              f"{accepted:,} 들임 · {quarantined:,} 격리")
        print(f"      ▸ {outcome['report']}")


if __name__ == "__main__":
    raise SystemExit(main())
