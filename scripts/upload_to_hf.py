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

    dev = manifest["dev_end"]
    dev_iso = f"{dev[:4]}-{dev[4:6]}-{dev[6:]}"

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

## 🔴 가장 먼저 알아야 할 것 두 가지

### ① 수익률은 `close` 가 아니라 `adj_close` 로 계산하세요

`open`·`high`·`low`·`close` 는 **KRX 원문 그대로라 액면분할이 조정돼 있지 않습니다.**
그대로 수익률을 내면 분할일이 폭락으로 읽힙니다.

```
삼성전자 2018-05-04 (50:1 액면분할)
  close     로 계산 →  -98.04%   ← 틀렸다
  adj_close 로 계산 →   -2.08%   ← 맞다 (KRX 공시 등락률과 같다)
```

분할·병합 **1,139건이 806종(21.9%)** 에 걸쳐 있습니다. 그래서 조정된 값을 옆에 붙였습니다.

| 무엇을 하려나 | 어느 칸을 쓰나 |
|---|---|
| 수익률 · 라벨 · 모멘텀 | **`adj_close`** |
| 변동성 · 고저 폭 (ATR · Parkinson) | **`adj_high`·`adj_low`·`adj_open`** |
| 시가총액 | **`close`** — `market_cap = close × listed_shares` 라서 |
| KRX 원문과 대조 | **`close`** |

**원 칸을 지우지 않은 이유**: 수정주가는 *현재 가격 기준으로 과거를 눌러 놓은 값*이라
**새 분할이 생기면 과거 값이 전부 바뀝니다.** 원문을 남겨 두어야 언제든 되돌아갈 수 있습니다.

`adj_source` 칸이 그 행의 값이 어디서 왔는지 알려 줍니다.

| 값 | 뜻 | 비중 |
|---|---|---|
| `fdr` | FinanceDataReader(네이버) 외부 실측 | 78.4% |
| `chain` | 우리가 조정계수로 이어 붙인 값 | 21.6% |

`chain` 이 있는 이유: FDR 은 **최근 3,000거래일만** 줍니다. 우리 자료는 2010년부터라
그 앞이 비어서, FDR 이 닿는 가장 이른 날을 기준점으로 삼아 과거로 이어 붙였습니다.
그 계산의 오차는 **최대 0.39%** 입니다 (기준점 하나만 남기고 2,999일을 우리 계산으로
채운 뒤 진짜 값과 대조해 실측). 정지일은 시·고·저가가 비어 있고 종가만 있습니다.

### ② 한글이 깨져 보이면 파일이 아니라 **읽는 방법**입니다

파일은 전부 UTF-8 입니다. 한국어 Windows 의 파이썬 기본 인코딩은 `cp949` 라서,
`encoding=` 을 안 주면 UTF-8 파일을 cp949 로 해석해 `ê¸°ì¤ì¼` 같은 글자가 나옵니다.

```python
# 🔴 이렇게 하면 깨집니다
meta = json.load(open(path))

# ✅ 이렇게 하세요
with open(path, encoding="utf-8") as f:
    meta = json.load(f)

# CSV 는 pandas 가 기본 UTF-8 이라 그냥 읽힙니다
df = pd.read_csv(path)               # ✅
# parquet 은 UTF-8 이 내장이라 애초에 안 깨집니다
```

콘솔 출력까지 깨진다면 — PowerShell `$env:PYTHONUTF8=1` · cmd `chcp 65001`.

## 예측 대상

- 진입 **t+1 시가** → 청산 **t+6 시가** ({manifest['horizon']}거래일)
- 3분류 중립 밴드: 지수 ±{manifest['band']['index']:.1%} · 종목 ±{manifest['band']['stock']:.1%}
- 지수 라벨 분포: {비율(지수분포)}
- 표본 종목 라벨 분포: {비율(종목분포)}
  ⚠️ 표본 30종목의 분포입니다. 거래정지·상장폐지 사례를 **일부러 섞어** 뽑았기 때문에
  전 종목 분포와 다릅니다. 전체 통계로 인용하지 마세요.

## 개발구간이 어디까지인가 — 오해가 잦은 곳

**개발구간에 하한은 없습니다.** `{manifest['dev_end']}` 는 **끝**이지 시작이 아닙니다.

```
{dev_iso} 로 경계가 바뀐 것은 "끝이 뒤로 밀린 것"입니다.
시작은 여전히 2010-01-04 입니다.

  2010-01-04 ─────────────────────── {dev_iso}  │  봉인 {manifest['holdout_start']} ~
             ↑ 여기부터 전부 쓸 수 있습니다        │  (여기 없습니다)
```

늘어난 구간은 기존 구간을 **대체하는 것이 아니라 더해지는 것**입니다.
뒤쪽 몇 년만 잘라 쓰면 학습 자료의 대부분을 버리게 됩니다.

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

## 칸 설명 — `full/daily_price_dev.parquet`

| 칸 | 뜻 | 함정 |
|---|---|---|
| `bas_dd` | 거래일 `YYYYMMDD` | **문자열**입니다. 사전순 = 날짜순이라 `<=` 비교가 그대로 통합니다 |
| `code` | 종목코드 | 🔴 **숫자가 아닙니다** — 5·6번째에 영문이 오는 종목 84종(`0001B0`) |
| `name` | 종목명 | 같은 코드도 이름이 바뀝니다. 코드로 잇고 이름으로 잇지 마세요 |
| `market` | `KOSPI` / `KOSDAQ` | |
| `sector` | KRX 업종 | WICS 와 다릅니다. KOSPI 는 빈 값이 많습니다 |
| `open` `high` `low` `close` | 시·고·저·종가 (원) | 🔴 **미조정 원가격** — 위 ① 참고 |
| `change` `change_rate` | 전일대비 · 등락률(%) | `change_rate` 는 **분할이 반영된 값** |
| `volume` `value` | 거래량 · 거래대금 | 거래정지일은 0 |
| `market_cap` | 시가총액 (원) | 1.8경이라 float64 유효숫자를 넘습니다 |
| `listed_shares` | 상장주식수 | 증자·감자·분할로 바뀝니다 |
| `adj_open` `adj_high` `adj_low` `adj_close` | **수정 OHLC** | ✅ **수익률은 이쪽** |
| `adj_source` | `fdr` / `chain` | 그 행의 수정값 출처 |

⚠️ **거래정지일**(`open=high=low=0`)에는 `adj_open`·`adj_high`·`adj_low` 가 **비어 있고**
`adj_close` 만 있습니다. 그 날은 체결이 없었으므로 시·고·저가가 존재하지 않습니다.
`0` 으로 채우면 "그 날 가격이 0원" 이 되니 그대로 두거나 걸러 내세요.

## 무결성 확인

`MANIFEST.json` 에 파일마다 SHA-256 이 있습니다. 받은 파일이 보낸 것과 같은지
맞춰 볼 수 있습니다.

```python
import hashlib, json

with open("MANIFEST.json", encoding="utf-8") as f:      # encoding 필수
    manifest = json.load(f)
for item in manifest["files"]:
    print(item["path"], item["sha256"][:16], f'{{item["rows"]:,}}행')
```

## 만든 방법

`scripts/export_team_dataset.py` → `scripts/upload_to_hf.py`
(저장소 `devlee328288/Alpha_Stack`). 같은 커밋에서 다시 돌리면 같은 파일이 나옵니다.

이 카드는 **업로드할 때마다 `MANIFEST.json` 에서 자동으로 다시 만들어집니다** —
파일 목록·행 수·홀드아웃 경계·라벨 분포는 항상 지금 올라간 자료의 값입니다.

문제가 보이면 저장소에 이슈로 남겨 주세요. 데이터 파트(이동원)가 봅니다.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="반출 폴더를 HF private 데이터셋으로 올린다")
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"기본 {DEFAULT_REPO}")
    parser.add_argument("--path", default=None,
                        help="올릴 폴더 (기본: data/outbox 의 가장 최근 날짜)")
    parser.add_argument("--path-in-repo", default="",
                        help="레포 안의 하위 폴더 (기본: 루트). "
                             "🔴 같은 레포에 성격이 다른 자료를 함께 둘 때 반드시 준다 — "
                             "루트에 올리면 기존 README.md·MANIFEST.json 을 덮어쓴다")
    parser.add_argument("--no-card", action="store_true",
                        help="데이터셋 카드를 만들지 않는다. 🔴 build_dataset_card 는 "
                             "시세 반출(MANIFEST 의 stats.kospi200)을 전제하므로 "
                             "다른 종류의 반출은 자기 README.md 를 들고 와야 한다")
    parser.add_argument("--note", default="",
                        help="커밋 메시지에 덧붙일 한 줄")
    parser.add_argument("--dry-run", action="store_true",
                        help="올리지 않고 무엇을 올릴지만 보여준다")
    args = parser.parse_args()

    if args.path:
        root = Path(args.path)
    else:
        # 🔴 **날짜 모양(YYYY-MM-DD)인 폴더만 본다.**
        #
        # 예전에는 `glob("*")` 을 그냥 정렬해 마지막을 골랐다. 그런데 `data/outbox` 에는
        # 성격이 다른 반출도 함께 산다(`dart_20260902` 등). 사전순으로 정렬하면
        # `'d' > '2'` 라서 **`dart_20260902` 가 `2026-09-02` 를 이긴다.**
        #
        # 실제로 2026-09-02 에 이 일이 났다. 시세 반출을 올리려는데 기본값이 DART 폴더를
        # 골랐고, 그대로 갔으면 **DART 자료가 시세 레포의 루트를 덮을 뻔했다**
        # (README.md·MANIFEST.json 까지). `--dry-run` 이 잡아서 막았다.
        날짜모양 = "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]"
        후보 = sorted(p for p in Path("data/outbox").glob(날짜모양) if p.is_dir())
        if not 후보:
            print("data/outbox 에 날짜(YYYY-MM-DD) 폴더가 없다.")
            print("  할 일: python scripts/export_team_dataset.py 를 먼저 돌린다.")
            print("  다른 종류의 반출을 올리려면 --path 로 폴더를 직접 지정한다.")
            return 1
        root = 후보[-1]
        print(f"(날짜 폴더 {len(후보)}개 중 가장 최근을 골랐다. 다른 것을 올리려면 --path)")

    if not (root / "MANIFEST.json").exists():
        print(f"{root}/MANIFEST.json 이 없다. 반출이 끝나지 않았다")
        return 1

    token, source = hf_data.load_hf_key()
    if not token:
        print("HUGGINGFACE_ACCESS_TOKEN 이 없다 (.env 확인)")
        return 1
    print(f"토큰 출처: {source}")
    print(f"올릴 폴더: {root}")
    print(f"대상 레포: {args.repo}"
          + (f"  (하위 폴더 {args.path_in_repo}/)" if args.path_in_repo else "  (루트)"))
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
    if args.no_card:
        있나 = (root / "README.md").exists()
        print(f"✅ 카드 생성 건너뜀 (반출본의 README.md {'있음' if 있나 else '없음'})")
    else:
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
    메시지 = f"데이터 반출 {root.name}"
    if args.note:
        메시지 += f" — {args.note}"
    api.upload_folder(
        folder_path=str(root),
        repo_id=args.repo,
        repo_type="dataset",
        path_in_repo=args.path_in_repo or None,
        commit_message=메시지,
    )

    올라간 = api.list_repo_files(repo_id=args.repo, repo_type="dataset")
    print()
    print(f"✅ 업로드 완료 — 서버에 파일 {len(올라간)}개")
    print(f"   https://huggingface.co/datasets/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
