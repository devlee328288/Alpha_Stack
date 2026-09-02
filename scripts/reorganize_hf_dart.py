"""HF `alphastack-dart` 를 bulk/(참조) + pit/(학습 정본) 2층으로 재구성한다.

무엇을 하나
-----------
1. 루트에 흩어져 있는 기존 parquet 43개를 `bulk/` 아래로 **서버 쪽에서 이동**한다.
   `CommitOperationCopy` 는 LFS 포인터만 복사하므로 443MB 를 다시 올리지 않는다.
2. `scripts/export_dart_dataset.py` 가 만든 반출 폴더의 `pit/` + MANIFEST 를 올린다.
3. README 를 2층 구조에 맞게 다시 쓴다 — 어느 층을 언제 쓰는지 표로.

왜 2층인가 (2026-09-02 팀장 확정)
---------------------------------
- `bulk/` — 재무정보 일괄다운로드 원본 묶음. 2015~2026 전 상장사, 분기·반기 포함.
  **접수일이 없어 학습 금지.** EDA·대조·넓은 참조용으로만.
- `pit/`  — API 재수집분. 접수일(`rcept_dt`)·`account_detail` 포함. **학습용 정본.**

🔴 실행 전 확인 두 가지
-----------------------
- 수집 세션의 DART 수집이 끝났는가 (오류 배치 정리까지)
- 사용자 확인을 받았는가 — HF 저장소 동시 push 금지 규칙
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest.clients import hf_data  # noqa: E402

REPO = "qurious-quant/alphastack-dart"


def build_readme(manifest: dict) -> str:
    cov = manifest["coverage"]
    pit_row = (
        f"| `pit/` | OpenDART API 재수집분 ({cov['companies']}사 · "
        f"사업연도 {cov['bsns_year'][0]}~{cov['bsns_year'][1]} · {cov['rows']:,}행) "
        "| ✅ `rcept_dt` | **학습용 정본** — 접수일 기준 as-of join |"
    )
    bulk_row = (
        "| `bulk/` | 재무정보 일괄다운로드 묶음 (2015~2026 전 상장사, 분기·반기 포함) "
        "| ❌ 없음 | EDA·대조·참조만. **학습 금지** |"
    )
    return f"""---
license: other
language: [ko]
tags: [finance, korea, dart, financial-statements]
---

# Alpha_Stack — DART 재무제표 (private)

> 🔴 **private 데이터셋입니다.** 조직(`qurious-quant`) 밖으로 내보내지 마세요.

## 층이 두 개입니다 — 학습에는 pit/ 만 쓰세요

| 층 | 무엇 | 접수일 | 용도 |
|---|---|---|---|
{pit_row}
{bulk_row}

## 🔴 왜 bulk/ 를 학습에 쓰면 안 되나

`bulk/` 에는 공시 접수일이 없습니다. `결산기준일`은 결산기일 뿐, 세상이 그 숫자를
알게 된 날이 아닙니다. 둘은 석 달까지 벌어지므로, 결산기에 값을 붙이면 **석 달치
미래가 학습에 들어가고 예외는 나지 않습니다.**

## 바로 쓰기 (pit/)

```python
from huggingface_hub import hf_hub_download
import pandas as pd

path = hf_hub_download(
    repo_id="{REPO}",
    filename="pit/2023_사업보고서.parquet",
    repo_type="dataset",
)
df = pd.read_parquet(path)
# 시점 정합: 거래일 t 에는 rcept_dt <= t 인 가장 최근 보고서만 붙인다
```

⚠️ 재무 표의 기본키에는 `account_detail` 이 반드시 들어갑니다 — 빼면 자본변동표에서
행이 조용히 사라집니다 (삼성전자 2023 실측: 176줄 → 135줄).

## 무결성

루트 `MANIFEST.json` 에 pit/ 파일별 SHA-256 · 행 수 · 수집 범위가 있습니다.

생성 시각 `{manifest['generated_at']}` · 만든 방법 `scripts/export_dart_dataset.py` +
`scripts/reorganize_hf_dart.py` (저장소 `devlee328288/Alpha_Stack`)
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="alphastack-dart 를 bulk/ + pit/ 로 재구성")
    parser.add_argument("--outbox", required=True,
                        help="export_dart_dataset.py 가 만든 폴더 (data/outbox/dart_pit_*)")
    parser.add_argument("--dry-run", action="store_true",
                        help="무엇을 옮기고 올릴지만 보여준다 (기본으로 먼저 실행할 것)")
    args = parser.parse_args()

    root = Path(args.outbox)
    manifest_path = root / "MANIFEST.json"
    if not manifest_path.exists():
        print(f"{manifest_path} 가 없다. 먼저 scripts/export_dart_dataset.py 를 돌릴 것")
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    token, source = hf_data.load_hf_key()
    if not token:
        print("HUGGINGFACE_ACCESS_TOKEN 이 없다 (.env 확인)")
        return 1
    print(f"토큰 출처: {source}")

    from huggingface_hub import CommitOperationCopy, CommitOperationDelete, HfApi

    api = HfApi(token=token)

    # 🔴 private 이중 확인 — upload_to_hf.py 와 같은 이유
    info = api.repo_info(repo_id=REPO, repo_type="dataset")
    if not info.private:
        print("🔴 중단 — 이 데이터셋이 public 이다.")
        return 1
    print("✅ private 확인")

    # ── 1. 루트 parquet → bulk/ 이동 계획 ───────────────────────────
    existing = api.list_repo_files(repo_id=REPO, repo_type="dataset")
    to_move = [f for f in existing
               if f.endswith(".parquet") and "/" not in f]
    already = [f for f in existing if f.startswith("bulk/")]
    print(f"루트 parquet {len(to_move)}개 → bulk/ 이동 예정 (이미 bulk/ 에 {len(already)}개)")
    for f in to_move:
        print(f"   {f}  →  bulk/{f}")

    # ── 2. pit/ 업로드 계획 ─────────────────────────────────────────
    pit_files = sorted((root / "pit").glob("*.parquet"))
    total_mb = sum(p.stat().st_size for p in pit_files) / 1024 / 1024
    print(f"pit/ 업로드 예정 {len(pit_files)}개 · {total_mb:,.1f} MB + MANIFEST.json + README.md")

    if args.dry_run:
        print()
        print("--dry-run 이라 여기서 멈춘다")
        return 0

    # ── 실행 1: 이동 (복사+삭제 한 커밋 — LFS 재업로드 없음) ────────
    if to_move:
        ops = []
        for f in to_move:
            ops.append(CommitOperationCopy(src_path_in_repo=f, path_in_repo=f"bulk/{f}"))
        for f in to_move:
            ops.append(CommitOperationDelete(path_in_repo=f))
        api.create_commit(
            repo_id=REPO, repo_type="dataset", operations=ops,
            commit_message=f"일괄다운로드 묶음 {len(to_move)}개를 bulk/ 로 이동",
        )
        print(f"✅ bulk/ 이동 완료 ({len(to_move)}개)")

    # ── 실행 2: pit/ + MANIFEST + README 업로드 ─────────────────────
    (root / "README.md").write_text(build_readme(manifest), encoding="utf-8")
    api.upload_folder(
        folder_path=str(root), repo_id=REPO, repo_type="dataset",
        commit_message=f"pit/ 반출 {root.name} — 접수일 포함 학습용 정본",
    )
    now = api.list_repo_files(repo_id=REPO, repo_type="dataset")
    print(f"✅ 업로드 완료 — 서버에 파일 {len(now)}개")
    print(f"   https://huggingface.co/datasets/{REPO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
