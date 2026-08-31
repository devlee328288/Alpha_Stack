# 팀원용 — HuggingFace 데이터 받기 · 쓰기 · 올리기

> 대상: Qurious 팀원 4명
> 데이터셋: `qurious-quant/alphastack-krx-dev` (**private**)
> 마지막 갱신: 2026-08-31 (이동원)

---

## 0. 왜 저장소가 아니라 HuggingFace 인가

데이터 파일을 GitHub 에 커밋하지 않습니다. 이유가 둘입니다.

1. **약관** — 담긴 것은 KRX 원자료이고, 이용약관 제11조 ②가 제3자 제공을 금지합니다.
   우리 저장소는 **PUBLIC** 이라 커밋하는 순간 전 세계에 배포됩니다.
2. **크기** — 개발구간 전량이 parquet 으로도 142MB 입니다. git 이 감당할 크기가 아닙니다.

그래서 **파일은 HF private 에, 만드는 방법은 GitHub 에** 둡니다.
`scripts/export_team_dataset.py` 를 같은 커밋에서 다시 돌리면 같은 파일이 나옵니다.

> 🔴 **받은 파일을 다른 곳에 다시 올리지 마세요.** 개인 드라이브 공유 링크, 공개 저장소,
> 블로그 첨부 전부 안 됩니다. 조직 `qurious-quant` 안에서만 씁니다.

---

## 1. 준비 (한 번만)

### 1-1. HuggingFace 계정 만들고 팀장에게 아이디 알려주기

https://huggingface.co/join 에서 가입한 뒤, **HF 아이디**를 이동원에게 알려주세요.
GitHub 닉과 다를 수 있어서 따로 받아야 합니다. 조직에 초대해 드립니다.

초대를 수락하면 https://huggingface.co/qurious-quant 가 보입니다.

### 1-2. 액세스 토큰 만들기

https://huggingface.co/settings/tokens → `New token`

| 무엇을 할 건가 | 고를 역할 |
|---|---|
| 데이터를 **받기만** 한다 | `Read` |
| 내가 모은 데이터를 **올리기도** 한다 | `Write` |

> 🔴 **팀장의 토큰을 받아 쓰지 마세요.** 각자 자기 토큰을 만듭니다.
> 토큰은 비밀번호와 같아서, 남의 것을 쓰면 누가 무엇을 했는지 구분이 안 되고
> 하나가 새면 전부 다시 만들어야 합니다.

### 1-3. 토큰을 `.env` 에 넣기

프로젝트 루트의 `.env` 파일에 한 줄 추가합니다.

```
HUGGINGFACE_ACCESS_TOKEN=hf_여기에_본인_토큰
```

`.env` 는 `.gitignore` 에 있어서 커밋되지 않습니다. **절대 코드 안에 토큰을 적지 마세요.**

### 1-4. 패키지 설치

```bash
uv pip install huggingface_hub --python .venv/Scripts/python.exe
```

또는 `pip install -e .[dev]` 로 한 번에 (pyproject 의 dev 에 들어 있습니다).

---

## 2. 받기

### 2-1. 파일 하나만

```python
from huggingface_hub import hf_hub_download
import pandas as pd

path = hf_hub_download(
    repo_id="qurious-quant/alphastack-krx-dev",
    filename="small/features_labels_kospi200_dev.csv",
    repo_type="dataset",          # ← 빠뜨리면 모델 저장소를 찾다가 404 가 납니다
)
df = pd.read_csv(path)
```

토큰은 따로 넘기지 않아도 됩니다 — `.env` 를 읽는 우리 코드를 쓰거나,
`huggingface-cli login` 을 한 번 해두면 자동으로 붙습니다. 명시하고 싶으면:

```python
from common import secrets
token, _ = secrets.load_key(("HUGGINGFACE_ACCESS_TOKEN",))
path = hf_hub_download(..., token=token)
```

### 2-2. 전량 parquet

```python
path = hf_hub_download(
    repo_id="qurious-quant/alphastack-krx-dev",
    filename="full/daily_price_dev.parquet",
    repo_type="dataset",
)
df = pd.read_parquet(path)        # 599만 행 · 메모리 약 920MB
```

> ⚠️ 메모리가 부족하면 필요한 칸만 읽으세요.
> `pd.read_parquet(path, columns=["bas_dd", "code", "close", "volume"])`

### 2-3. 통째로

```python
from huggingface_hub import snapshot_download
folder = snapshot_download(
    repo_id="qurious-quant/alphastack-krx-dev",
    repo_type="dataset",
)
```

받은 파일은 캐시에 남아 두 번째부터는 바로 열립니다.

### 2-4. 무엇이 들어 있나

| 파일 | 행 | 누구에게 |
|---|---:|---|
| `small/index_kospi200_dev.csv` | 2,880 | 예측 대상 그 자체 |
| `small/index_all_dev.csv` | 135,879 | 지수 51종 |
| `small/features_labels_kospi200_dev.csv` | 2,815 | **오준영** — 바로 `fit` 됩니다 |
| `small/features_labels_stocks30_dev.csv` | 56,706 | **오준영** — 종목 단위 |
| `small/stocks_sample30_raw_dev.csv` | 63,746 | **신장환** — 정제 전 원본 |
| `small/stocks_sample30_train_dev.csv` | 58,791 | **신장환** — 정제 후 |
| `full/daily_price_dev.parquet` | 5,989,308 | 최종 학습용 전량 |
| `full/index_price_dev.parquet` | 135,879 | 지수 전량 |

`MANIFEST.json` 에 파일마다 SHA-256 이 있습니다. 받은 것이 보낸 것과 같은지 맞춰 볼 수 있습니다.

---

## 3. 쓰기

### 3-1. 바로 학습시키기

```python
import pandas as pd
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="qurious-quant/alphastack-krx-dev",
    filename="small/features_labels_kospi200_dev.csv",
    repo_type="dataset",
)
df = pd.read_csv(path)

# 피처 22칸 — 시세 원본 칸과 라벨 칸을 뺀 나머지가 전부 피처입니다
NOT_FEATURE = {
    "bas_dd", "date", "index_name", "index_class",
    "open", "high", "low", "close", "change", "change_rate",
    "volume", "value", "market_cap", "fwd_return_5d", "label",
}
FEATURES = [c for c in df.columns if c not in NOT_FEATURE]

X, y = df[FEATURES], df["label"]
```

`models/logistic.py` 의 `build_logistic_baseline()` 에 그대로 넣으면 됩니다.

### 3-2. 🔴 반드시 지킬 것 셋

**① 무작위로 섞어 나누지 마세요.**

```python
# ❌ 이러면 미래로 과거를 맞히는 셈이 됩니다
train_test_split(X, y, shuffle=True)

# ✅ 시간 순서로 자릅니다
cut = int(len(df) * 0.8)
X_train, X_test = X[:cut], X[cut:]
y_train, y_test = y[:cut], y[cut:]
```

**② 스케일러를 전체 데이터에 `fit` 하지 마세요.**

`StandardScaler().fit(X)` 를 전체에 하면 검증 구간의 평균·표준편차가 학습에 새어 듭니다.
`build_logistic_baseline()` 은 `Pipeline` 안에 스케일러를 넣어 이걸 막아 뒀습니다 —
그 함수를 쓰면 신경 쓸 것이 없습니다.

**③ 홀드아웃을 찾지 마세요.**

이 데이터셋은 `20210831` 까지만 담겨 있습니다. `20210901` 이후는 **봉인 구간**이고,
미리 보면 "미리 정해 두고 딱 한 번 열어본다" 는 우리 검증 설계가 그 자리에서 무너집니다.
되돌릴 방법이 없어서, 실수로도 안 되게 아예 반출하지 않았습니다.

### 3-3. 라벨이 무엇인가

- 진입 **t+1 시가** → 청산 **t+6 시가** (5거래일)
- 3분류: 지수 ±1.0% · 종목 ±2.0% 밖이면 상승/하락, 안이면 중립
- 지수 분포 **상승 34.03% · 중립 38.66% · 하락 27.31%**
- 표본 종목 분포 상승 32.56% · 중립 33.30% · 하락 34.14%

> ⚠️ 표본 종목 분포는 **30종목만의 값**입니다. 거래정지·상장폐지 사례를 일부러 섞어
> 뽑았기 때문에 전 종목 분포(30.12/37.90/31.98)와 다릅니다. **전체 통계로 인용하지 마세요.**

기준선은 **개발구간 52.64%** 이고, 유의미하다고 말하려면 중첩 보정 기준 **57.22%** 를
넘겨야 합니다. 53% 가 나왔다고 "이겼다" 고 적으면 안 됩니다.

---

## 4. 올리기 — 내가 모은 데이터를 팀에 넘길 때

### 4-1. 어디에 올리나

같은 데이터셋의 **`inbox/<본인이름>/`** 아래에 둡니다.

```
inbox/
  강민석/  2026-08-31_거래대금상위.csv
  신장환/  2026-08-31_업종분류.csv
  오준영/  2026-08-31_뉴스헤드라인.csv
```

> 🔴 **`small/` 과 `full/` 은 건드리지 마세요.** 그 둘은 스크립트가 만들어 내는 자리라
> 손으로 올린 파일은 다음 반출 때 덮어씌워집니다.

### 4-2. 올리는 코드

```python
from huggingface_hub import HfApi
from common import secrets

token, _ = secrets.load_key(("HUGGINGFACE_ACCESS_TOKEN",))
api = HfApi(token=token)

api.upload_file(
    path_or_fileobj="C:/내려받은/거래대금상위.csv",
    path_in_repo="inbox/강민석/2026-08-31_거래대금상위.csv",
    repo_id="qurious-quant/alphastack-krx-dev",
    repo_type="dataset",
    commit_message="거래대금 상위 종목 수집분 — 2021년 개발구간",
)
```

`Write` 권한 토큰이 필요합니다 (1-2 참고).

### 4-3. 같이 적어 주실 것

파일 옆에 같은 이름의 `.md` 를 하나 두거나, 커밋 메시지에 아래를 적어 주세요.
**이게 없으면 제가 받아서 무엇을 어떻게 검사할지 알 수 없습니다.**

| 항목 | 예 |
|---|---|
| 어디서 받았나 | KRX 정보데이터시스템 / 네이버금융 / 직접 크롤링 |
| 언제 받았나 | 2026-08-30 |
| 무슨 기간인가 | 2015-01-01 ~ 2021-08-31 |
| 칸이 무슨 뜻인가 | `거래대금` = 원 단위 누적, `순위` = 그날 기준 |
| 손댄 것이 있나 | 엑셀에서 쉼표 지움 / 헤더 한 줄 지움 |

### 4-4. 올린 뒤

이동원에게 알려 주세요. 받아서

1. **검증** — 칸·타입·날짜 범위·중복·결측·이상치를 검사하고
2. **정제** — 날짜 형식 통일, 종목코드 6자리 0채움, 쉼표 제거 등을 자동으로 맞추고
3. **분리 적재** — 통과한 행과 격리된 행을 나눠 DB 에 넣습니다

한 뒤 결과를 알려 드립니다. 규격과 검사기는 `ingest/inbox/` 에서 만들고 있습니다.

### 4-5. 🔴 올리면 안 되는 것

- **개인정보** — 이름·연락처·계좌번호가 든 파일
- **API 키·비밀번호** — `.env`, `secrets.json`, 캡처 이미지 포함
- **유료 데이터** — 재배포가 금지된 유료 구독 자료
- **홀드아웃 구간** — `20210901` 이후 시세. 봉인이 깨집니다

---

## 5. 자주 막히는 곳

| 증상 | 원인과 해결 |
|---|---|
| `401 Unauthorized` | 토큰이 없거나 만료. `.env` 확인, 조직 초대 수락했는지 확인 |
| `404 Not Found` | `repo_type="dataset"` 을 빠뜨렸습니다. 없으면 모델 저장소를 찾습니다 |
| `403 Forbidden` (올릴 때) | `Read` 토큰입니다. `Write` 로 새로 만드세요 |
| 조직이 안 보임 | 초대 메일의 수락 링크를 아직 안 누른 경우가 대부분입니다 |
| parquet 이 안 열림 | `pyarrow` 가 필요합니다. `pip install -e .[dev]` |
| 메모리 부족 | `pd.read_parquet(path, columns=[...])` 로 필요한 칸만 |

---

## 관련 문서

- [명세서](명세서.md) — 데이터 파트 전체 규격
- `scripts/export_team_dataset.py` — 이 데이터셋을 만드는 코드
- `scripts/upload_to_hf.py` — 올리는 코드
- `ingest/inbox/schemas/` — 반입 규격 (작성 중)
