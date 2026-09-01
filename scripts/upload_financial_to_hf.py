"""손으로 받아 묶은 재무 parquet 을 HuggingFace private 데이터셋으로 올린다.

왜 따로 있나
-----------
`upload_to_hf.py` 는 `data/outbox/<날짜>/` 반출본 전용이고 `MANIFEST.json` 을 요구한다.
이쪽은 `scripts/pack_manual_financial.py` 가 만든 재무 묶음을 올린다.
**자료마다 레포를 나누는** 방침(2026-09-01 합의)에 따라 대상도 다르다.

    qurious-quant/alphastack-krx-dev     시세 반출본  (upload_to_hf.py)
    qurious-quant/alphastack-dart        재무 묶음    (이 스크립트)

왜 올리나
--------
로컬 디스크가 90% 넘게 차 있어 원본 TSV 7.5GB 를 지우려는데, 지우기 전에 **어딘가에는
남겨 두어야** 한다. parquet 443MB 를 여기 올려 두면 나중에 다시 받을 수 있다.

약관
----
🔴 **private 로만 올린다.** OpenDART 이용약관에는 재배포를 금지하는 조문이 **없지만**
(제19조②는 인증키 공유 금지일 뿐), 확인된 것은 "금지 조문이 없다" 이지 "허용된다" 가
아니다. 그리고 이 저장소는 Public 이라 습관을 하나로 통일하는 편이 안전하다.
`upload_to_hf.py` 와 **같은 이중 확인**을 한다 — 만들 때 private 을 주고, 만든 뒤
서버에 다시 물어 아니면 한 파일도 올리지 않는다.

사용법
------
    python scripts/upload_financial_to_hf.py --dry-run   # 무엇을 올릴지만
    python scripts/upload_financial_to_hf.py             # 실제 업로드
    python scripts/upload_financial_to_hf.py --repo 조직/이름
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pyarrow.parquet as pq  # noqa: E402

from ingest.clients import hf_data  # noqa: E402

#: 자료별 레포 분리 방침에 따른 대상.
DEFAULT_REPO = "qurious-quant/alphastack-dart"
SRC = Path("data/manual/financial_packed")


def _api(token: str):
    """`huggingface_hub` 을 늦게 부른다 — 없을 때 무엇을 하라고 알려주기 위해서다."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub 가 없다. 아래를 실행할 것:")
        print("    uv pip install huggingface_hub --python .venv/Scripts/python.exe")
        raise SystemExit(2) from None
    return HfApi(token=token)


def build_card(files: list[Path], repo_id: str) -> str:
    총행 = sum(pq.read_metadata(f).num_rows for f in files)
    총MB = sum(f.stat().st_size for f in files) / 1024 / 1024
    칸 = list(pq.read_schema(files[0]).names)

    표 = "\n".join(
        f"| `{f.name}` | {pq.read_metadata(f).num_rows:,} | "
        f"{f.stat().st_size / 1024 / 1024:.1f} MB |"
        for f in files
    )

    return f"""---
license: other
language: [ko]
tags: [finance, korea, dart, financial-statements]
---

# Alpha_Stack — DART 재무제표 묶음 (private)

OpenDART **재무정보 일괄다운로드**로 받은 전 상장사 재무제표를 parquet 으로 묶은 것입니다.

> 🔴 **private 데이터셋입니다.** 조직(`qurious-quant`) 밖으로 내보내지 마세요.
> OpenDART 약관에 재배포 금지 조문은 없지만, 확인된 것은 *"금지 조문이 없다"* 이지
> *"허용된다"* 가 아닙니다.

## 규모

- 파일 **{len(files)}개** · **{총MB:,.0f} MB** · **{총행:,}행**
- 원본 TSV **830개 · 7,603 MB** 를 묶은 것입니다 (**17.2배** 압축, zstd)
- 기간 **2015 ~ 2026** · 사업/반기/1분기/3분기 보고서

## 칸 {len(칸)}개

원본 TSV 15칸을 **하나도 버리지 않았고**, 파일명에만 있던 정보를 칸으로 옮겼습니다.

```
{", ".join(칸)}
```

| 칸 | 무엇 |
|---|---|
| `업종` | 표준산업분류 **코드** (`262` 등) — 원본 |
| `업종명` | 산업 이름 (`전자부품 제조업`) — 원본 |
| `업종구분` | OpenDART 다운로드 화면의 구분 (`일반`·`금융기타`·`보험`·`은행`·`증권`) |
| `연결여부` | `연결` / `별도` — 파일명에서 |
| `받은날짜` | 🔴 **공시 접수일이 아니라 내려받은 날**입니다 |
| `원본파일` | 원래 TSV 이름 |

## 🔴 시점 주의 — 그대로 학습에 넣지 마세요

이 자료에는 **공시 접수일(`rcept_dt`)이 없습니다.** `결산기준일`은 결산기일 뿐,
세상이 그 숫자를 알게 된 날이 아닙니다. 둘은 석 달까지 벌어집니다.

결산기에 값을 붙이면 **석 달치 미래가 학습에 들어가고 예외는 나지 않습니다.**

접수일은 DART `list.json` 으로 되찾을 수 있습니다 — 회사당 1콜, 약 2,512콜
(하루 한도 20,000의 12.6%). 접수일은 불변이라 한 번 받아 캐시하면 됩니다.

## 파일

| 파일 | 행 | 크기 |
|---|---|---|
{표}

## 바로 쓰기

```python
from huggingface_hub import hf_hub_download
import pandas as pd

path = hf_hub_download(
    repo_id="{repo_id}",
    filename="{files[0].name}",
    repo_type="dataset",
)
df = pd.read_parquet(path)

# 연결재무제표의 자산총계만
mask = (df["연결여부"] == "연결") & (df["항목명"].str.strip() == "자산총계")
df.loc[mask, ["종목코드", "회사명", "당기"]]
```

⚠️ 금액에 콤마가 들어 있습니다(`24,203,421,673`). 종목코드에는 대괄호가 붙어 있습니다
(`[060310]`). **원문 그대로** 담았습니다 — 정제는 쓰는 쪽에서 합니다.

## 만든 방법

`scripts/pack_manual_financial.py` (저장소 `devlee328288/Alpha_Stack`).
같은 TSV 로 다시 돌리면 같은 파일이 나옵니다.
"""


def main() -> int:
    p = argparse.ArgumentParser(description="재무 parquet 을 HF private 로 올린다")
    p.add_argument("--repo", default=DEFAULT_REPO, help=f"기본 {DEFAULT_REPO}")
    p.add_argument("--dry-run", action="store_true", help="올리지 않고 보여만 준다")
    a = p.parse_args()

    files = sorted(SRC.glob("*.parquet"))
    if not files:
        print(f"🔴 {SRC} 에 parquet 이 없다. 먼저 scripts/pack_manual_financial.py 를 돌릴 것")
        return 1

    총MB = sum(f.stat().st_size for f in files) / 1024 / 1024
    총행 = sum(pq.read_metadata(f).num_rows for f in files)
    print(f"올릴 것 : {SRC}")
    print(f"          {len(files)}개 · {총MB:,.1f} MB · {총행:,}행")
    print(f"대상    : {a.repo}")
    print()

    token, source = hf_data.load_hf_key()
    if not token:
        print("🔴 HUGGINGFACE_ACCESS_TOKEN 이 없다 (.env 확인)")
        return 1
    print(f"토큰 출처: {source}")

    if a.dry_run:
        for f in files:
            print(f"   {f.name:<28} {f.stat().st_size / 1024 / 1024:>7.1f} MB "
                  f"{pq.read_metadata(f).num_rows:>10,}행")
        print("\n--dry-run 이라 여기서 멈춘다")
        return 0

    api = _api(token)
    api.create_repo(repo_id=a.repo, repo_type="dataset", private=True, exist_ok=True)

    # 🔴 만들었다고 믿지 않는다. 서버에 다시 물어본다.
    #    이미 있는 레포에 exist_ok=True 로 붙으면 그것이 public 이어도 그냥 성공한다.
    info = api.repo_info(repo_id=a.repo, repo_type="dataset")
    if not info.private:
        print("🔴 중단 — 이 데이터셋이 public 이다. 한 파일도 올리지 않는다.")
        print(f"   https://huggingface.co/datasets/{a.repo}/settings 에서")
        print("   private 으로 바꾼 뒤 다시 실행할 것.")
        return 1
    print("✅ private 확인")

    card = SRC / "README.md"
    card.write_text(build_card(files, a.repo), encoding="utf-8")
    print("✅ 데이터셋 카드 생성")

    print()
    print(f"업로드 중… ({총MB:,.0f} MB — 몇 분 걸린다)")
    api.upload_folder(
        folder_path=str(SRC),
        repo_id=a.repo,
        repo_type="dataset",
        commit_message=f"재무 묶음 {len(files)}개 · {총행:,}행 (TSV 830개를 17.2배로)",
    )
    print(f"✅ 완료 — https://huggingface.co/datasets/{a.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
