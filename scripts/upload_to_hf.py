"""반출 폴더를 HuggingFace private 데이터셋으로 올린다.

왜 드라이브가 아니라 HF 인가
--------------------------
팀원이 `hf_hub_download()` 한 줄로 받아 쓰고, 올린 이력이 커밋으로 남아 "누가 언제
무엇을 바꿨나" 를 되짚을 수 있기 때문이다. 드라이브는 링크를 눌러 받는 것까지는 쉽지만
코드에서 받으려면 인증이 번거롭고, 파일이 바뀌어도 무엇이 바뀌었는지 알 수 없다.

🔴 public 이 되면 그 순간 약관 위반이다
------------------------------------
올리는 것은 KRX 원자료다. 이용약관 제11조 ②가 제3자 제공을 금지하므로 이 데이터셋은
**반드시 private** 여야 하고, 조직(`qurious-quant`) 멤버 밖으로 나가면 안 된다.
그래서 이 스크립트는

  1. 레포를 만들 때 `private=True` 를 준다
  2. 만든 뒤 **서버에 다시 물어 private 인지 확인**하고, 아니면 **한 파일도 올리지 않고 멈춘다**

두 번 확인하는 이유는 1번이 실패해도 예외가 안 날 수 있기 때문이다 — 이미 있는 레포에
`exist_ok=True` 로 붙으면 그 레포가 public 이어도 그냥 성공한다.

토큰
----
`.env` 의 `HUGGINGFACE_ACCESS_TOKEN`. 조직에 `repo.write` 가 있어야 한다.
**팀원에게 이 토큰을 주지 않는다.** 팀원은 조직 멤버로 초대하고 각자 read 토큰을 만든다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest.clients import hf_data  # noqa: E402

#: 기본 대상. 조직 이름을 앞에 두면 개인 계정 것과 섞이지 않는다.
DEFAULT_REPO = "qurious-quant/alphastack-krx-dev"


def _api(token: str):
    """`huggingface_hub` 을 늦게 부른다 — 없을 때 무엇을 하라고 알려주기 위해서다.

    맨 위에서 import 하면 `ModuleNotFoundError` 한 줄만 나오고, 받는 사람은 무엇을
    깔아야 하는지 모른 채 검색을 시작한다. 막다른 길로 만들지 않는다.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub 가 없다. 아래를 실행할 것:")
        print("    uv pip install huggingface_hub --python .venv/Scripts/python.exe")
        print("  또는")
        print("    .venv/Scripts/python.exe -m pip install -e .[dev]")
        raise SystemExit(2) from None
    return HfApi(token=token)


def build_dataset_card(root: Path, repo_id: str) -> str:
    """HF 가 데이터셋 첫 화면에 띄우는 카드(README.md)를 만든다.

    팀원이 레포에 들어와 가장 먼저 보는 글이라, **경고를 맨 위에** 둔다.
    """
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    files: List[Dict] = manifest["files"]

    표 = ["| 파일 | 행 | 크기(MB) | 무엇인가 |", "|---|---:|---:|---|"]
    for f in sorted(files, key=lambda x: x["path"]):
        표.append(f"| `{f['path']}` | {f['rows']:,} | {f['size_mb']:.2f} | {f['note']} |")

    지수분포 = manifest["stats"].get("kospi200", {}).get("distribution", {})
    종목분포 = manifest["stats"].get("stocks30", {}).get("distribution", {})

    def 비율(d: Dict[str, int]) -> str:
        총 = sum(d.values()) or 1
        return " · ".join(f"{k} {v / 총:.2%}" for k, v in d.items())

    return f"""---
license: other
license_name: krx-terms
license_link: https://data.krx.co.kr
language:
  - ko
tags:
  - finance
  - korean-stock
  - time-series
pretty_name: AlphaStack KRX 개발구간
---

# AlphaStack — KRX 개발구간 데이터셋

> 🔴 **이 저장소는 private 이며, 조직 `qurious-quant` 밖으로 나가면 안 됩니다.**
> 담긴 것은 KRX 원자료이고 이용약관 제11조 ②가 제3자 제공을 금지합니다.
> 파일을 다른 곳에 다시 올리거나 공개 저장소에 커밋하지 마세요.

> 🔴 **홀드아웃(`{manifest['holdout_start']}` 이후)은 여기 없습니다. 찾지도 마세요.**
> 이 데이터셋은 `{manifest['dev_end']}` 까지만 담습니다. 봉인 구간을 미리 보면
> "미리 정해 두고 딱 한 번 열어본다" 는 검증 설계가 그 자리에서 무너지고,
> 되돌릴 방법이 없습니다.

생성 시각 `{manifest['generated_at']}`

## 무엇이 들어 있나

{chr(10).join(표)}

## 예측 대상

- 진입 **t+1 시가** → 청산 **t+6 시가** ({manifest['horizon']}거래일)
- 3분류 중립 밴드: 지수 ±{manifest['band']['index']:.1%} · 종목 ±{manifest['band']['stock']:.1%}
- 지수 라벨 분포: {비율(지수분포)}
- 표본 종목 라벨 분포: {비율(종목분포)}
  ⚠️ 표본 30종목의 분포입니다. 거래정지·상장폐지 사례를 **일부러 섞어** 뽑았기 때문에
  전 종목 분포(30.12/37.90/31.98)와 다릅니다. 전체 통계로 인용하지 마세요.

## 바로 쓰기

```python
from huggingface_hub import hf_hub_download
import pandas as pd

path = hf_hub_download(
    repo_id="{repo_id}",
    filename="small/features_labels_kospi200_dev.csv",
    repo_type="dataset",
)
df = pd.read_csv(path)

FEATURES = [c for c in df.columns if c not in
            ("bas_dd", "date", "index_name", "index_class",
             "open", "high", "low", "close", "change", "change_rate",
             "volume", "value", "market_cap", "fwd_return_5d", "label")]
X, y = df[FEATURES], df["label"]
```

⚠️ `X` 와 `y` 를 무작위로 섞어 나누지 마세요. 시계열이라 **시간 순서로** 잘라야 합니다.

## 무결성 확인

`MANIFEST.json` 에 파일마다 SHA-256 이 있습니다. 받은 파일이 보낸 것과 같은지
맞춰 볼 수 있습니다.

## 만든 방법

`scripts/export_team_dataset.py` (저장소 `devlee328288/Alpha_Stack`). 같은 커밋에서
다시 돌리면 같은 파일이 나옵니다.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="반출 폴더를 HF private 데이터셋으로 올린다")
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"기본 {DEFAULT_REPO}")
    parser.add_argument("--path", default=None,
                        help="올릴 폴더 (기본: data/outbox 의 가장 최근 날짜)")
    parser.add_argument("--dry-run", action="store_true",
                        help="올리지 않고 무엇을 올릴지만 보여준다")
    args = parser.parse_args()

    if args.path:
        root = Path(args.path)
    else:
        후보 = sorted(Path("data/outbox").glob("*"))
        if not 후보:
            print("data/outbox 가 비어 있다. 먼저 scripts/export_team_dataset.py 를 돌릴 것")
            return 1
        root = 후보[-1]

    if not (root / "MANIFEST.json").exists():
        print(f"{root}/MANIFEST.json 이 없다. 반출이 끝나지 않았다")
        return 1

    token, source = hf_data.load_hf_key()
    if not token:
        print("HUGGINGFACE_ACCESS_TOKEN 이 없다 (.env 확인)")
        return 1
    print(f"토큰 출처: {source}")
    print(f"올릴 폴더: {root}")
    print(f"대상 레포: {args.repo}")
    print()

    api = _api(token)

    # ── 레포 준비 ────────────────────────────────────────────────────
    api.create_repo(repo_id=args.repo, repo_type="dataset", private=True, exist_ok=True)

    # 🔴 만들었다고 믿지 않는다. 서버에 다시 물어본다.
    info = api.repo_info(repo_id=args.repo, repo_type="dataset")
    if not info.private:
        print("🔴 중단 — 이 데이터셋이 public 이다. KRX 원자료를 올릴 수 없다.")
        print(f"   https://huggingface.co/datasets/{args.repo}/settings 에서")
        print("   private 으로 바꾼 뒤 다시 실행할 것.")
        return 1
    print("✅ private 확인")

    # ── 데이터셋 카드 ────────────────────────────────────────────────
    card = build_dataset_card(root, args.repo)
    (root / "README.md").write_text(card, encoding="utf-8")
    print("✅ 데이터셋 카드 생성")

    올릴것 = [p for p in sorted(root.rglob("*")) if p.is_file()]
    총MB = sum(p.stat().st_size for p in 올릴것) / 1024 / 1024
    print()
    print(f"올릴 파일 {len(올릴것)}개 · 합계 {총MB:,.1f} MB")
    for p in 올릴것:
        print(f"   {p.relative_to(root).as_posix():46s} {p.stat().st_size / 1024 / 1024:>8.2f} MB")

    if args.dry_run:
        print()
        print("--dry-run 이라 여기서 멈춘다")
        return 0

    print()
    print("업로드 중… (142MB parquet 이 있어 몇 분 걸린다)")
    api.upload_folder(
        folder_path=str(root),
        repo_id=args.repo,
        repo_type="dataset",
        commit_message=f"데이터 반출 {root.name} — 개발구간만",
    )

    올라간 = api.list_repo_files(repo_id=args.repo, repo_type="dataset")
    print()
    print(f"✅ 업로드 완료 — 서버에 파일 {len(올라간)}개")
    print(f"   https://huggingface.co/datasets/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
