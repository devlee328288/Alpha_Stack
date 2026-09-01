# 이 폴더는 비어 있습니다 — 자료는 옆 폴더와 HuggingFace 에 있습니다

2026-09-01 에 원본 TSV **830개(7.5GB)** 를 지웠습니다. 없어진 것이 아니라 **옮긴 것**입니다.

## 어디로 갔나

| 어디 | 무엇 | 크기 |
|---|---|---|
| `data/manual/financial_packed/` | parquet 43개 (연도 × 보고서) | **443 MB** |
| `qurious-quant/alphastack-dart` (HF private) | 같은 parquet 43개 + 데이터셋 카드 | 443 MB |

**칸을 하나도 버리지 않았습니다.** 원본 TSV 15칸이 그대로 있고, 파일명에만 있던 정보
(연도·보고서·재무제표·업종구분·연결여부·받은날짜·원본파일)가 칸으로 들어갔습니다.

```
7,603 MB → 443 MB   (17.2배 · zstd)
27,233,430행 · 2015~2026 · 사업/반기/1분기/3분기
```

## 왜 지웠나

로컬 디스크가 **90.6% (933G 중 845G)** 차 있었습니다. parquet 이 로컬에 남아 있어
작업은 그대로 이어지고, 원본이 다시 필요하면 아래처럼 되찾습니다.

## 되찾는 법

**① 로컬 parquet 을 그냥 씁니다** (대부분 이걸로 충분합니다)

```python
import pandas as pd
df = pd.read_parquet("data/manual/financial_packed/2023_사업보고서.parquet")
```

**② HuggingFace 에서 받습니다** (로컬이 날아갔을 때)

```python
from huggingface_hub import hf_hub_download
import pandas as pd

path = hf_hub_download(
    repo_id="qurious-quant/alphastack-dart",
    filename="2023_사업보고서.parquet",
    repo_type="dataset",
)
df = pd.read_parquet(path)
```

**③ TSV 모양이 꼭 필요하면** parquet 에서 되돌립니다 — 칸이 다 있습니다.

```python
df[df["원본파일"] == "2023_사업보고서_01_재무상태표_연결_20260814.tsv"] \
  .to_csv("복원.tsv", sep="\t", index=False, encoding="utf-8")
```

**④ OpenDART 에서 다시 받습니다** (최후 수단 — 시간이 꽤 걸립니다)
→ [`docs/데이터파트/version2.4/직접수집_가이드.md`](../../../docs/데이터파트/version2.4/직접수집_가이드.md) §4.1

## ⚠️ 쓰기 전에 알아야 할 것

- **금액에 콤마**가 있습니다 (`24,203,421,673`) · **종목코드에 대괄호**가 있습니다 (`[060310]`)
  · **항목명 앞에 공백 들여쓰기**가 있습니다. 원문 그대로 담았습니다 — 정제는 쓰는 쪽에서.
- 🔴 **공시 접수일(`rcept_dt`)이 없습니다.** `결산기준일` 은 결산기일 뿐이고 세상이 그 숫자를
  알게 된 날이 아닙니다. 둘은 석 달까지 벌어집니다. 결산기에 값을 붙이면 **석 달치 미래가
  학습에 들어가고 예외는 나지 않습니다.**
  → 회사당 1콜(약 2,512콜)로 되찾을 수 있습니다. 가이드 §5.2 참고.
- `업종` 은 **표준산업분류 코드**(`262`), `업종구분` 은 **OpenDART 다운로드 화면의 구분**
  (`일반`·`금융기타`·`보험`·`은행`·`증권`)입니다. 서로 다른 값입니다.

## 만든 방법

```bash
python scripts/pack_manual_financial.py            # TSV → parquet
python scripts/upload_financial_to_hf.py           # → HF private
```
