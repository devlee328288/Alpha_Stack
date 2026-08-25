# AlphaStack — 주가지수 등락 방향 예측과 성과 검증

> **"맞히는 것보다 어려운 건, 맞혔다고 말해도 되는지 아는 것"**

팀 **적층(積層)** 1차 프로젝트 · 2026-09-01(화) ~ 09-15(화) · 4인
발표: koreaIT 노원 B강의실

| | |
|---|---|
| **주제** | 주가지수 데이터 활용 머신러닝·딥러닝 |
| **GitHub** | https://github.com/devlee328288/Alpha_Stack (PUBLIC) |
| **GitLab** | https://gitlab.com/dev-dongwon05253/alpha_stack (미러 · main 만) |
| **상태** | 🟡 저장소 이관 완료 · 킥오프 대기 |

---

## 🚀 빠른 시작 — 5분 안에 돌려 보기

팀원이 클론한 뒤 **이 순서 그대로** 하면 됩니다. 막히면 그 자리에서 물어보세요.

```bash
git clone https://github.com/devlee328288/Alpha_Stack.git
cd Alpha_Stack

# 1) 파이썬 환경 (Python 3.12 필요)
#    uv 가 없으면: pip install uv
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -e ".[dev]"
#    macOS/Linux 는 .venv/bin/python

# 2) 검증 엔진이 살아 있는지 확인 (외부 연결·API 키 없이 돕니다)
.venv/Scripts/python.exe -m pytest tests/ -q
#    → 20 passed 가 나오면 성공입니다

# 3) 코드 스타일 확인
.venv/Scripts/python.exe -m ruff check evaluation/ tests/
#    → All checks passed!
```

**API 키 없이도 여기까지 됩니다.** 실제 시세 수집은 `.env` 가 필요한데,
그건 킥오프에서 팀 Supabase 를 정한 뒤입니다 ([회의안건 A-1](docs/회의안건.md)).

```bash
# 4) (선택) 자격증명 — 아직 안 해도 됩니다
cp .env.example .env      # 값은 팀장에게 요청
```

> ⚠️ **`.env` 를 절대 커밋하지 마세요.** 이 저장소는 PUBLIC 입니다.
> `.gitignore` 가 막고 있지만, push 전에 `git status --short` 로 눈으로 확인합니다.

---

## Why — 왜 이 문제인가

### 문제: "정확도 62%" 는 아무 말도 하지 않는다

주가 예측 프로젝트는 대개 정확도 몇 %로 끝난다. 그런데 그 숫자는 혼자서는 아무것도
증명하지 못한다.

- 상승이 62% 나오는 구간에서는 **언제나 "오른다"고 답하기만 해도** 62%가 나온다
- 시계열을 무작위로 섞어 나누면 미래로 과거를 예측하게 되어(look-ahead) 성능이 부풀어 오른다
- 거래비용을 빼지 않은 수익률은 **실현할 수 없는 숫자**다. 일간 매매는 왕복 비용이
  수익을 통째로 먹는 일이 흔하다

그래서 이 프로젝트는 "잘 맞히는 모델"보다 **"맞혔다고 말해도 되는지 판별하는 장치"** 를
먼저 만든다.

### 왜 검증 엔진을 따로 떼는가 — 팀 이름이 적층(積層)인 이유

세 차수의 주제가 이미 정해져 있다.

| 차수 | 주제 |
|---|---|
| 1차 | 주가지수 데이터 활용 머신러닝·딥러닝 |
| 2차 | 나만의 로보 어드바이저 개발 **및 성과 검증** |
| 3차 | 나만의 투자 인디케이터 개발 **및 성과 검증** |

셋 다 "개발 및 성과 검증"으로 끝난다. 차수마다 새로 만드는 것은 **앞쪽**(신호 →
포트폴리오 → 지표)이고, **재는 방법은 바뀌지 않는다.** MDD 를 재는 법은 무엇이 그
수익률을 만들었든 같다.

검증 코드를 노트북 셀에 흩어 놓으면 1차는 통과하고 **2차에 전부 다시 짜게 된다.**
그 비용을 지금 없애는 것이 이 저장소 구조의 이유다.

---

## What — 무엇을 하는가

### 필수 범위 — 반드시 닫는다

| # | 항목 | 완료 조건 |
|---|---|---|
| ① | 기술적 지표 기반 등락 방향 예측 모델 | 예측 결과가 재현된다 |
| ② | ML 모델 성능 비교·평가 | 동일 조건 비교표가 나온다 |
| ③ | 롤링(walk-forward) 백테스팅 | 폴드별 성능과 **분산**이 나온다 |
| ④ | 재현 가능한 단일 파이프라인 | 한 명령으로 같은 결과 |
| ⑤ | **성과 검증 엔진 분리** (MDD·Sharpe·승률·거래비용) | 2·3차가 그대로 가져다 쓴다 |

### 선택 범위 — 관심사에 따라

⑥ 딥러닝(LSTM/GRU) 동일 조건 비교 · ⑦ 뉴스·공시 제목 감성 피처(무료 인코더만) ·
⑧ 유니버스 확장 350종목 횡단면 랭킹 · ⑨ ADF 정상성 검정(**기구현**) · ⑩~⑫ 인프라·팀원 제안

---

## Who — 누가 무엇을 맡는가

| 영역 | 담당 | 주 작업 위치 |
|---|---|---|
| 백테스팅 · 성과지표 · 대시보드 | 팀원 A | [evaluation/](evaluation/) |
| 크롤링 검수 · 개선 | 팀원 B | [ingest/clients/](ingest/clients/) |
| 수집 · 저장 · 검증엔진 기반 | 팀장 | [ingest/store/](ingest/store/) · [evaluation/](evaluation/) |
| 피처 · 모델 | **미정** | [features/](features/) · [models/](models/) |

> 역할 확정은 [회의안건 C-3](docs/회의안건.md) 입니다.

### 이 저장소를 만든 사람

- 코드 기반: 팀장 개인 프로젝트 `data-service` 에서 필요한 것만 골라 이관 후 재구성
- 이관 세션: 2026-08-25 · 9,031줄 이관 + 734줄 신규 작성

---

## How — 어떻게 만들었는가

### 계층 구조 — 의존은 아래로만 흐른다

```
common/       설정·경로·자격증명·거래일 계산. 아무것도 import 하지 않는다
  ↑
ingest/       바깥 세상과 통신하고 시세를 저장한다
  ├ clients/  외부 API 9종 (KRX·DART·ECOS·FRED·yfinance…)
  └ store/    수집한 것을 Postgres·SQLite 에 넣고 꺼낸다
  ↑
features/     시세 → 기술적 지표 → 학습 가능한 표
  ↑
models/       표 → 등락 방향 분류기 (RF·XGBoost·LightGBM)
  ↑
evaluation/   ★ 예측 → 믿어도 되는가 (워크포워드·MDD·Sharpe·거래비용)
  ↑
api/          위의 결과를 HTTP 로 내보낸다 (최소 계약)

timeseries/   ARIMA·ADF·ACF/PACF. numpy 로 손수 짠 시계열 도구 한 벌
              위 사슬과 나란히 선다 — 분류기와 겨룰 통계적 기준선
```

**`evaluation/` 은 `models/` 를 import 하지 않는다.** 받는 것은 포지션 배열과 수익률
배열뿐이다. 무엇이 그 포지션을 만들었는지 모르게 짜 두면 2·3차가 그대로 가져다 쓴다.
이 무지(無知)가 재사용의 조건이다.

> `evaluation` 안에서 models 를 import 하고 싶어지면 설계가 틀어진 신호다.
> import 를 추가하지 말고 **함수 인자를 늘린다.**

### 왜 원본을 통째로 복제하지 않았는가

원본 `data-service` 는 리서치 리포트 생성까지 하는 더 큰 서비스다(라우터 17파일 ·
엔드포인트 64개 · research 24파일 11,241줄). **1차 범위에 필요한 것만 골라 가져오고
구조는 목적에 맞게 다시 짰다.**

| 가져온 것 | 줄수 | 이유 |
|---|---|---|
| `common/` ← `app/core/` | 1,360 | 설정·경로·자격증명·거래일 |
| `ingest/clients/` ← `app/clients/` | 3,978 | 외부 API 9종. 팀원 B 검수 대상 |
| `ingest/store/` ← `app/repositories/` | 1,612 | krx 계열만. clip·report·user 는 뺐다 |
| `timeseries/` ← `app/services/timeseries/` | 2,063 | ARIMA·ADF·워크포워드 |
| `scripts/` | 532 | 수집·유니버스 구축 |
| `sql/init/` | 250 | 스키마 (clip 제외) |

**가져오지 않은 것**: `app/services/research/`(리포트 생성기 · 1차 범위 밖),
`app/routers/`(엔드포인트 64개), `app/main.py`

### 기술 스택 — 버전은 추측이 아니라 실측이다

```
Python 3.12.13
numpy 2.5.1 · pandas 3.0.5 · scipy 1.18.1
scikit-learn 1.9.0 · LightGBM 4.7.0 · XGBoost 3.4.1 · joblib 1.5.3
FastAPI 0.141.1 · uvicorn 0.52.0 · pydantic 2.13.4 · SQLAlchemy 2.0.52
```

> ⚠️ numpy 2.5 / pandas 3.0 은 상당히 앞선 버전이라 ML 라이브러리와 ABI 가 어긋날
> 수 있습니다. **2026-08-25 에 실제로 설치해 7종 전부 import 되는 것을 확인한 뒤**
> 이 조합을 고정했습니다. 임의로 올리지 마세요.

### 학습을 어디서 돌리나 — 로컬이 기본이다

| 작업 | 실행처 | 근거 |
|---|---|---|
| **LightGBM 학습·그리드** | **로컬** | 배포판이 GPU 빌드가 아니다. 코어 수만 좌우한다 |
| sklearn RF · 보정 · bootstrap | 로컬 | 전부 CPU 경로 |
| 전처리 · 피처 · EDA | 로컬 | I/O 바운드이고 원자료가 로컬에 있다 |
| 딥러닝(⑥ LSTM/GRU) | Colab Pro | GPU 이득이 실재한다 |

> **"Colab 이니까 빠르겠지"로 판단하지 않습니다.** 로컬이 22코어라 CPU 바운드
> 작업은 Colab 런타임보다 **빠릅니다.**

---

## Impact — 무엇을 숫자로 잴 것인가

> 🟡 **1차 프로젝트는 이제 시작합니다. 아래는 아직 결과가 아니라 "무엇을 잴 것인가"입니다.**
> 여기에 추측값을 적지 않습니다 — 측정한 뒤 채웁니다.

### 이관 세션에서 실제로 나온 것 (2026-08-25 실측)

| 항목 | 값 |
|---|---|
| 이관한 코드 | 9,031줄 (원본 약 15,000줄 후보 중 선별) |
| 신규 작성 | 734줄 (검증 엔진 478 + 테스트 159 + 계약 97) |
| 모듈 import 검증 | **32/32 성공** |
| 검증 엔진 테스트 | **20/20 통과** (0.23초) |
| 린트 (새 코드) | **0건** — 남은 36건은 전부 원본에서 물려받은 것 |
| 사용 가능한 유니버스 | KOSPI200 200 + KOSDAQ150 150 = **350종목** (2026-08-25 기준) |

### 1차 종료 시 보고할 숫자

- **① 예측 성능**: Accuracy · F1 · 방향 적중률 — **기준선 3종과 나란히**
- **② 모델 비교**: RF vs XGBoost vs LightGBM, 동일 폴드·동일 피처·동일 레이블
- **③ 폴드별 분산**: 평균이 아니라 **흔들림**. 이게 "믿어도 되는가"의 답이다
- **⑤ 성과 지표**: MDD · Sharpe · 승률 · **거래비용 차감 후** 수익률

> ⚠️ 기준선을 못 이기면 **못 이겼다고 적습니다.** 주가는 효율시장에 가까워서
> 못 이기는 것이 정상에 가깝고, 그 사실을 감추면 보고서가 거짓말이 됩니다.

---

## Proof — 깊이 볼 만한 것

팀원들이 코드 리뷰할 때 여기부터 보면 좋습니다.

### 1. 거래비용은 조용히 틀린다 — [evaluation/metrics.py](evaluation/metrics.py)

포지션이 바뀔 때만 비용이 든다. 그런데 **첫 시점의 진입 비용**을 빠뜨리기 쉽다.
`positions[0]` 이 0 이 아니면 아무것도 없던 상태에서 잡은 것이므로 비용이 드는데,
`np.diff` 로 짜면 첫 원소가 사라져 매 폴드마다 진입이 공짜가 된다.

```python
previous = np.concatenate(([0.0], pos[:-1]))   # 맨 앞에 0 을 덧대는 이유가 이것
turnover = np.abs(pos - previous)
```

→ [tests/test_evaluation.py](tests/test_evaluation.py) 의
`test_첫_시점_진입_비용을_빠뜨리지_않는다` 가 이걸 잡습니다.

### 2. gap 이 없으면 5일 레이블이 학습에 샌다 — [evaluation/walk_forward.py](evaluation/walk_forward.py)

"5일 뒤 상승" 레이블은 t 행이 t+5 가격을 알아야 만들어진다. 학습 구간 마지막 행의
레이블이 검증 구간 초반과 **겹친다** — 학습이 검증 답을 이미 본 셈이다.

레이블이 k 일 앞을 보면 `gap=k` 로 둔다. **이걸 빠뜨려도 에러가 나지 않는다.**
성능이 조용히 부풀어 오를 뿐이다.

### 3. 포지션 정렬 한 줄 — [evaluation/metrics.py](evaluation/metrics.py) 모듈 docstring

```python
signal   = model.predict(features)   # t 일 종가까지 보고 낸 신호
position = np.roll(signal, 1)        # t+1 일 수익률에 적용  ← 이 한 줄
position[0] = 0
```

이걸 빠뜨리면 정확도 60% 짜리 모델이 연 300% 를 벌어 준다.
**그런 결과가 나오면 기뻐하기 전에 이 정렬부터 의심합니다.**

### 4. 상장폐지 종목을 지우지 않는 이유 — [sql/init/01-schema.sql](sql/init/01-schema.sql)

```sql
is_delisted boolean NOT NULL DEFAULT false,
```

지우면 **생존 편향**이 생긴다. 망한 종목이 빠진 유니버스로 백테스트하면 실제보다
훨씬 좋은 결과가 나온다.

### 5. 원본이 numpy 로 손수 짠 ARIMA — [timeseries/](timeseries/)

`scipy` · `statsmodels` 없이 ADF(MacKinnon 임계값) · ACF(Bartlett) ·
PACF(Durbin-Levinson) · ARIMA(Hannan-Rissanen)를 직접 구현해 둔 2,063줄.
**⑨ 정상성 검정은 이미 구현되어 있고**, ARIMA 는 우리 트리 모델이 겨룰 통계적 기준선입니다.

---

## Learn — 배운 것 / 개선할 것

> 🟡 프로젝트 회고는 1차 종료 후 채웁니다. 아래는 **이관 세션에서 실제로 겪은 것**입니다.

### 겪은 것 ① — 경로 깊이는 조용히 깨진다

원본 `app/clients/krx_data.py` 의 `parents[2]` 는 `app` 을 건너뛰어 루트를 가리키도록
센 숫자였다. 폴더 구조를 바꾸자 **루트가 아닌 곳을 가리켰지만 import 는 멀쩡히
통과했다.** 파일을 실제로 읽는 순간에야 드러난다.

→ 그래서 [common/paths.py](common/paths.py) 에 기준점을 모았습니다. 새 코드는
깊이를 세지 말고 `DATA_DIR` · `ARTIFACTS_DIR` 을 씁니다.

### 겪은 것 ② — 상대 import 는 일괄 치환에서 빠진다

`from ..repositories import tmp_cache` 는 절대 경로 치환 규칙에 걸리지 않았다.
import 검증을 돌리지 않았다면 팀원이 9/1에 `ModuleNotFoundError` 를 봤을 것이다.

→ **"옮겼다"의 완료 조건은 "import 가 된다"입니다.** 32개 모듈 전부 확인했습니다.

### 겪은 것 ③ — "설치된다"와 "import 된다"는 다르다

`uv pip compile` 은 numpy 2.5.1 + sklearn 조합을 문제없이 해석했다. 하지만 의존성
해석 성공이 ABI 호환을 보장하지 않는다. **실제로 설치해서 import 해 봐야** 안다.

### 개선하고 싶은 것

- `common/settings.py`(437줄)에 1차가 쓰지 않는 원본 설정이 섞여 있다 → [회의안건 C-1](docs/회의안건.md)
- `ruff` 36건이 남아 있다 (전부 이관 코드의 한국어 주석 줄 길이)
- `features/` · `models/` 는 아직 계약만 있고 구현이 없다

---

## 📊 현재 상태 — 무엇이 되고 무엇이 안 되는가

### ✅ 완성 (지금 바로 쓸 수 있다)

| 기능 | 위치 | 검증 |
|---|---|---|
| **성과 지표** MDD·Sharpe·승률·거래비용 | [evaluation/metrics.py](evaluation/metrics.py) | 테스트 12개 통과 |
| **워크포워드 분할** 확장창·롤링창·gap | [evaluation/walk_forward.py](evaluation/walk_forward.py) | 테스트 5개 통과 |
| **기준선 3종** always_up·majority·momentum | [evaluation/baseline.py](evaluation/baseline.py) | 테스트 3개 통과 |
| **시계열 도구** ADF·ACF/PACF·ARIMA·확률보행 | [timeseries/](timeseries/) | import 확인 (원본 검증본) |
| **외부 API 클라이언트** 9종 | [ingest/clients/](ingest/clients/) | import 확인 |
| **시세 저장·조회** | [ingest/store/](ingest/store/) | import 확인 |
| **DB 스키마** securities·ohlcv·calendar·sync_log·watermark | [sql/init/](sql/init/) | 원본에서 운영 중 |
| **유니버스** KOSPI200+KOSDAQ150 350종목 | [data/universe_core.json](data/universe_core.json) | 2026-08-25 생성 |

### 🟡 구현 중 / 계약만 있음

| 기능 | 위치 | 막고 있는 것 |
|---|---|---|
| **피처 엔지니어링** | [features/](features/) | 예측 대상·레이블 정의 미정 → [회의안건 A-2](docs/회의안건.md) |
| **모델 학습·비교** | [models/](models/) | 위와 동일 |
| **최소 API·Swagger** | [api/](api/) | 팀 Supabase·대시보드 방식 미정 → [회의안건 A-1·B-3](docs/회의안건.md) |
| **재현 파이프라인** | [pipelines/](pipelines/) | 위 셋이 정해진 뒤 |

### ⬜ 미착수

- 과거 구간 데이터 백필 (현재 297거래일 → 몇 년치로 늘릴지 미정)
- 대시보드
- 배포 (**방침: 9/1 이후**. 그전까지 전원 로컬 실행)

---

## 🗣️ 킥오프에서 정해야 할 것

**코드보다 먼저 정해야 하는 것들이 있습니다.** 특히 아래 셋은 정해지기 전에는
`features/` 가 첫 줄도 못 나갑니다.

1. **팀 Supabase** — 팔 것인가, 용량은 되는가, 권한은 어떻게
2. **예측 대상과 레이블** — 지수인가 종목인가 / 종가 대비인가 / 보합을 뺄 것인가
3. **데이터 백필** — 297거래일은 얇다. 몇 년치를 언제 누가

👉 **전체 목록: [docs/회의안건.md](docs/회의안건.md)**

---

## 📁 문서

| 문서 | 내용 |
|---|---|
| [docs/회의안건.md](docs/회의안건.md) | 킥오프에서 정할 것 (미결 사항 정본) |
| [docs/아키텍처.md](docs/아키텍처.md) | 계층 구조와 설계 결정 |
| [docs/ERD.md](docs/ERD.md) | 데이터 모델 |
| [docs/decisions/](docs/decisions/) | ADR — 결정 이력 |
| [AGENTS.md](AGENTS.md) | 팀 협업 규약 (Git·브랜치·코드 스타일) |
| [docs/AlphaStack_팀프로젝트_소개.html](docs/AlphaStack_팀프로젝트_소개.html) | 모집용 소개 (범위·일정 정본) |

---

## ⚠️ 이 저장소의 규칙 (짧게)

- **PUBLIC 입니다.** `.env` · API 키 · 데이터 원본을 커밋하지 않습니다
- **PR 로 올립니다.** `main` 직커밋 금지. 머지는 **사람이 GitHub 웹에서** 직접
- **`gh pr merge` 금지** — 계정 정지 이력이 있습니다
- 파이썬은 `snake_case` · 4-space · line-length 100 · 주석과 문서는 한국어
- push 전 `git status --short` 로 PDF·ZIP·데이터·시크릿 혼입 확인

전문: [AGENTS.md](AGENTS.md)
