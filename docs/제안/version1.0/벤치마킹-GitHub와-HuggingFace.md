# 벤치마킹 — GitHub 레포와 HuggingFace 모델

> 2026-08-29 · 회람용
> 스타 수·라이선스·최종 커밋은 **GitHub·HuggingFace API 를 직접 호출해** 받아 적었습니다.
> 가장 결과가 센 3건은 따로 대조했고 **전부 일치**했습니다.

---

## 읽는 법 — 두 가지를 구분합니다

| | 뜻 |
|---|---|
| **설치** | `requirements` 에 넣는다. 의존성이 늘고 라이선스가 우리에게 옮는다 |
| **참고** | 코드를 읽고 **구조만 배운다.** 설치하지 않는다 |

2주 프로젝트에서 의존성 하나는 생각보다 비쌉니다. 설치는 아껴야 합니다.

---

## 1. GitHub — 무엇을 설치하고 무엇을 읽을 것인가

### ✅ 설치할 것 — 3개뿐

| 레포 | ★ | 라이선스 | 최종 커밋 | 우리가 취할 것 한 가지 |
|---|---:|---|---|---|
| **[bukosabino/ta](https://github.com/bukosabino/ta)** | 5,180 | MIT | 2026-03-18 | **기술적 지표.** 순수 pandas/numpy 라 컴파일이 없다 |
| **[FinanceDataReader](https://github.com/FinanceData/FinanceDataReader)** | 1,536 | MIT | 2026-05-13 | **섹터·종목 메타.** 수업에서도 쓴다 |
| **[OpenDartReader](https://github.com/FinanceData/OpenDartReader)** | 473 | MIT | 2026-08-10 | DART 공시 (선택 범위일 때만) |

> ⚠️ **`ta` 를 설치해도 지표는 직접 구현합니다.** 수업 방식(numpy 직접)이 설명 가능하고,
> 손계산 테스트를 붙일 수 있습니다. `ta` 는 **우리 구현을 대조할 정답지**로 씁니다 —
> 이게 훨씬 값싼 검증입니다.

### 🔴 `pandas-ta` 는 쓰면 안 됩니다 — 저장소가 사라졌습니다

```
api.github.com/repos/twopirllc/pandas-ta  →  HTTP 404 Not Found
```

PyPI 에는 `0.4.71b0`(2025-09-14)이 남아 있지만 Repository URL 이 **끊긴 링크**를 가리킵니다.
블로그·유튜브에 예제가 아직 많이 돌아다니니 **팀원이 무심코 설치하지 않도록** 공유합니다.

### 📖 읽기만 할 것

| 레포 | ★ | 왜 읽나 |
|---|---:|---|
| **[machine-learning-for-trading](https://github.com/stefan-jansen/machine-learning-for-trading)** | 20,710 | ⭐ **가장 추천.** MIT · 2026-08-28 커밋(가장 활발) · 27개 챕터 노트북. 우리가 하려는 것의 교과서 |
| **[microsoft/qlib](https://github.com/microsoft/qlib)** | 48,018 | MIT. **구조만** 본다 — `data/model/strategy/backtest/workflow` 계층 분리가 우리 설계와 같다. 설치는 과하다 |
| **[awesome-quant](https://github.com/wilsonfreitas/awesome-quant)** | 29,267 | 큐레이션 목록. 필요할 때 찾아보는 용도 |

### ❌ 설치하면 안 되는 것 — 대부분 라이선스 때문입니다

| 레포 | ★ | 문제 |
|---|---:|---|
| **backtesting.py** | 8,907 | 🔴 **AGPL-3.0.** 활발히 관리되지만 라이선스가 강하게 전염된다 |
| **vectorbt** | 8,888 | 🔴 **Apache-2.0 + Commons Clause** — "Sell" 권리를 주지 않는다. 오픈소스가 아니다 |
| **mlfinlab** | 4,915 | 🔴 라이선스가 **제3자 제공·재배포·소스 공개를 금지**. 게다가 **2023-10-02 이후 방치**(약 3년) |
| **backtrader** | 23,011 | GPL-3.0 + **2024-08-19 이후 실질 중단** |
| **pyfolio · alphalens · empyrical** | 6,410 / 4,434 / 1,509 | Apache-2.0 이지만 전부 **2023~2024 에 멈췄다**. quantopian 폐업 |
| **quantstats** | 7,598 | Apache-2.0 · 활발. 그런데 **우리 검증 엔진과 숫자가 갈린다** — 화면에 Sharpe 두 개가 뜨고 "왜 다르냐"에 답을 못 한다 |
| **TA-Lib** | 12,218 | BSD-2. 0.6.5부터 wheel 이 있지만 **Windows 설치 난이도**가 여전히 리스크 |

> **라이선스는 발표에서 설명할 수 있는 판단입니다.** *"AGPL·Commons Clause 때문에 뺐습니다"*
> 한 문장이 *"그냥 안 썼습니다"* 보다 훨씬 낫습니다.

### 📌 범위 밖 (2·3차용)

FinRL(16,130★ · 강화학습) · PyPortfolioOpt(5,992★) · Riskfolio-Lib — **2차 자산배분**에서 봅니다.

---

## 2. HuggingFace — 한국어 금융 텍스트

### 🔴 먼저 — 이건 **선택 범위**입니다

커리큘럼에 `transformers` 가 **한 번도 안 나옵니다**(`from_pretrained` 0회).
강사님이 텍스트 분류로 보여주신 것은 **TF-IDF + LogisticRegression** 과 **Keras Embedding+LSTM**
입니다. 팀 전원에게 처음이므로 **필수 범위에 넣지 않았습니다.**

넣는다면 뉴스 감성이 **선택 ⑪** 로 들어가고, 그때 아래를 씁니다.

### ✅ 바로 쓸 수 있는 것

| 모델 | 30일 다운로드 | 라이선스 | 무엇 |
|---|---:|---|---|
| **[snunlp/KR-FinBert-SC](https://huggingface.co/snunlp/KR-FinBert-SC)** | 75,323 | ⚠️ **표기 없음** | 서울대 **금융 특화** 한국어 감성분류. 우리 용도에 가장 정확히 맞는다 |
| **[FISA-conclave/klue-roberta-news-sentiment](https://huggingface.co/FISA-conclave/klue-roberta-news-sentiment)** | 16,934 | **Apache-2.0** | 뉴스 3분류(부정/중립/긍정) · 110M · 2026-04-06. **라이선스가 깨끗한 유일한 감성 모델** |
| **[jhgan/ko-sroberta-multitask](https://huggingface.co/jhgan/ko-sroberta-multitask)** | **956,517** | ⚠️ 표기 없음 | 한국어 문장 임베딩 768차원. KorSTS 84.77/85.60. **중복 기사 제거·같은 사건 묶기**에 쓴다 |

### 🔴 라이선스 주의 — 다섯 개가 표기가 없습니다

`KR-FinBert-SC` · `klue/roberta-base` · `ko-sroberta-multitask` · `KoELECTRA-modu-ner` ·
`KPF-bert-ner` 는 **HuggingFace 카드에 라이선스 필드가 아예 없습니다.**

> 문서에는 **"라이선스 미표기"** 라고 그대로 적습니다. 임의로 "오픈소스"라고 쓰지 않습니다.
> 부트캠프 프로젝트라 실무적으로 문제될 일은 적지만, **모르는 것을 안다고 쓰지 않는 것**이
> 이 프로젝트의 논지와 맞습니다.

### 📌 필요해지면 볼 것

| 모델 | 30일 | 라이선스 | 용도 |
|---|---:|---|---|
| [kakaobank/kf-deberta-base](https://huggingface.co/kakaobank/kf-deberta-base) | 31,001 | MIT | **금융 도메인** 사전학습. 파인튜닝 베이스 |
| [klue/roberta-base](https://huggingface.co/klue/roberta-base) | 41,456 | ⚠️ (상류 CC-BY-SA-4.0) | 한국어 범용 베이스 |
| [nlpai-lab/KURE-v1](https://huggingface.co/nlpai-lab/KURE-v1) | 346,081 | MIT | 임베딩 1024차원 · max 8192 · 2026-08-26 (최신) |
| [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | **34,924,513** | MIT | 다국어 임베딩. 긴 문서 |
| [mDeBERTa-xnli](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7) | 826,793 | MIT | **제로샷 분류** — 학습 없이 라벨만 주면 된다 |
| [KPF/KPF-bert-ner](https://huggingface.co/KPF/KPF-bert-ner) | 24,446 | ⚠️ 표기 없음 | 한국언론진흥재단 뉴스 NER. 기사에서 기업명 추출 |
| [amphora/korfin-asc](https://huggingface.co/datasets/amphora/korfin-asc) | 89 | CC-BY-SA-4.0 | 한국어 **금융 속성기반 감성** 데이터셋 |

### ❌ 시계열 파운데이션 모델은 이번엔 아닙니다

| 모델 | 왜 |
|---|---|
| [amazon/chronos-bolt-small](https://huggingface.co/amazon/chronos-bolt-small) | Apache-2.0 · 30일 608만. 성능은 좋지만 **회귀 예측**이라 우리 3분류와 목적이 다르다 |
| [google/timesfm-2.0-500m](https://huggingface.co/google/timesfm-2.0-500m-pytorch) | 500M 파라미터 — 2주에 과하다 |
| [Salesforce/moirai-1.1-R-small](https://huggingface.co/Salesforce/moirai-1.1-R-small) | 🔴 **CC-BY-NC-4.0 — 비상업 전용.** 부트캠프라도 라이선스를 지켜야 한다 |

---

## 3. 우리 프로젝트에 실제로 옮길 것 — 요약

| 순위 | 무엇 | 어디서 | 담당 |
|---|---|---|---|
| 1 | **계층 분리 구조** (data → model → strategy → backtest) | qlib **읽기** | 이미 우리 구조가 그렇다 ✅ |
| 2 | **워크포워드·라벨링 노트북** | machine-learning-for-trading **읽기** | 오준영 |
| 3 | **지표 대조 정답지** | `ta` 설치 | 신장환 |
| 4 | **섹터 매핑** | FinanceDataReader + 수업 `06-data-crawling/7` | 이동원 |
| 5 | (선택) 뉴스 감성 | KR-FinBert-SC 또는 klue-roberta-news-sentiment | 이동원 |
| 6 | (선택) 중복 기사 제거 | ko-sroberta-multitask + MinHash LSH | 이동원 |

**설치 목록 최종안**: `ta` · `FinanceDataReader` — **둘뿐입니다.**
(선택 범위가 열리면 `transformers` · `OpenDartReader` 추가)

---

## 확인 방법과 한계

- 스타 수·라이선스·최종 커밋은 `api.github.com/repos/...` 를, 다운로드·파라미터 수는
  `huggingface.co/api/models/...` 를 **직접 호출**해 받았습니다
- 직접 대조한 3건: `pandas-ta` 404 / `backtesting.py` 8,907★·AGPL-3.0 / `KR-FinBert-SC`
  존재·라이선스 없음·75,323 — **전부 일치**했습니다
- **모델 성능은 카드에 적힌 값**입니다. 우리 데이터로 재 본 것이 아닙니다.
  쓰기로 하면 반드시 우리 뉴스 표본으로 다시 재야 합니다
