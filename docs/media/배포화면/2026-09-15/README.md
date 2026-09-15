# 배포 화면 캡처 — 2026-09-15 (HF Space)

> 강사님 서류용 · 캡처 이동원 · 대시보드 코드는 강민석 님(PR #261 · #262 · #269 · #279 · #287 · #289 · #300 · #301)
> 주소 https://data-student-alphastack-qurious.hf.space/ (Space `data-student/alphastack-qurious` · cpu-basic)
> 빌드 확인 — Space 빌드 시각은 공개 API 로 알 수 없어 **화면 글자로** 확인했다: Model Lab 머리 `COMBINATION C` · 요약 칸 `BEST HARMONIC MODEL`(#300) · Backtest `RUN ALL (A~F)` · `EXPERIMENT CONFIG · SSOT`(#301). main 의 `dashboard/` · `backtest/` 는 #301 머지(`9cf3f2c`) 뒤 `cff6715` 까지 바뀌지 않았다(#302 는 발표 파일 하나)
> 찍은 방법 — puppeteer 25.3.0 headless · 창 너비 1600 · 본문 높이만큼 창을 늘려 한 장 · 사이드바를 눌러 한 세션 안에서 이동 · **설정은 전부 화면 기본값** · 스코프 MARKET(KOSPI200) · 2026-09-15 09:44:48 ~ 09:47:40 KST
>
> 이전 판 [2026-09-14](../2026-09-14/README.md) 는 Model Lab 이 조합 E 이던 시점의 기록으로 남긴다.

## 화면 목록

순서는 찍은 순서가 아니라 사이드바 순서입니다. Overview 는 다른 페이지의 결과를 모아 보여 주므로 맨 마지막에 찍었습니다.

| # | 파일 | 페이지 | 상태 | RUN 걸린 시간 | 찍은 시각 (KST) | 화면에서 읽을 것 |
|---|---|---|---|---:|---|---|
| 00 | [00_Overview.png](00_Overview.png) | Overview | RUN 없음 — 네 단계 결과를 모아 보여 준다 | — | 09:47:40 | PIPELINE 4 STAGES 모두 ready · Baseline Sharpe 0.0815 · BEST MODEL XGBoost(아래 주의 ①) · 전략 6개 표 · 손익분기 비용 C 0.014% · 최근 20일 정확도 30.0% |
| 01 | [01_Baseline.png](01_Baseline.png) | Baseline | ▶ RUN BASELINE (Threshold 1% · CMA-ES QUICK 30) | 9초 이하(주의 ③) | 09:45:17 | 12폴드 · Sharpe 0.0815 · CAGR −0.08% · MDD −28.65% · Macro F1 0.2973 · 판정 비율 상승 18.68% · 중립 38.71% · 하락 42.61% |
| 02 | [02_Model_Lab.png](02_Model_Lab.png) | Model Lab | ▶ RUN (4모델 · 조합 C · 라벨 fwd_return ±1%) | 9초 이하(주의 ③) | 09:45:36 | BEST HARMONIC MODEL XGBoost · Harmonic 0.3447 · Accuracy 0.3861 · LogisticRegression Accuracy 0.3736(주의 ①②) |
| 03 | [03_Comparison.png](03_Comparison.png) | Comparison | RUN 없음 — Baseline · Model Lab 결과를 읽는다 | — | 09:45:46 | 라벨 분포 fwd_return 상승 33.63% · 중립 38.46% · 하락 27.91%(N 3,547) · Adaptive 18.68% · 38.71% · 42.61%(N 756) · 노란 안내는 Adaptive 라벨로 학습한 모델이 없다는 기본 안내 |
| 04 | [04_Backtest.png](04_Backtest.png) | Backtest | ▶ RUN ALL (A~F) · RandomForest × fwd_return · 2010-01-01 ~ 2025-01-01 · Trade Cost 0.0010 | 9초 이하(주의 ③) | 09:46:05 | 전략 C 총수익 −0.2242 · Sharpe −0.1808 · MDD −0.3502 · 전략 D(A + 5거래일 락) 총수익 +0.5419 · Sharpe 0.4032 |
| 05 | [05_Risk.png](05_Risk.png) | Risk | RUN 없음 · 탭 5개 중 OVERVIEW | — | 09:46:15 | 전략 A · MDD −23.53% · Sharpe −0.2211(무위험 2%) |
| 06 | [06_Cost_Sensitivity.png](06_Cost_Sensitivity.png) | Cost Sensitivity | ▶ RUN Cost Grid (Backtest 설정 그대로 · 비용 프리셋 4 × 전략 6) | 33초 | 09:46:59 | 손익분기 비용(Sharpe = 0) A 0.162% · B 0.061% · C 0.014% · D 1.996% · E 1.996% · F 0.208%(주의 ④) |
| 07 | [07_System.png](07_System.png) | System | RUN 없음 | — | 09:47:09 | 엔진 4/4 · streamlit 1.63.0 · exchange-calendars 미설치 · 결과 키 5 · 머리 시각 `2026-09-15 00:46:59` 는 UTC |
| 08 | [08_Universe.png](08_Universe.png) | Universe | **RUN 하지 않음** — 설정 화면 | — | 09:47:19 | 머리 `COMBINATION K · 14 FEAT` · 화면 안내상 30종목 RUN 예상이 기준선 3분 · 모델 37분/개라 이번 캡처에서 뺐다 |
| 09 | [09_QFRS.png](09_QFRS.png) | QFRS | RUN 없음 · RISK 탭 | — | 09:47:29 | 전략 A 점수 16 / 100 POOR |

## 2026-09-14 판과 달라진 것

| 페이지 | 2026-09-14 | 2026-09-15 | 바꾼 PR |
|---|---|---|---|
| Model Lab | 조합 E · `BEST MODEL` LightGBM | 조합 C(MARKET) · K(STOCK · UNIVERSE) · `BEST HARMONIC MODEL` XGBoost | #287 · #289 · #300 |
| Backtest | 전략 A · B · C | 전략 A ~ F — D · E · F 는 A · B · C 에 5거래일 락(거래가 난 뒤 5거래일 동안 신규 · 청산 금지) · 설정 표 `EXPERIMENT CONFIG · SSOT` | #301 |
| Cost Sensitivity | Predictor · 기간을 이 페이지에서 따로 고른다 · 전략 3 | Backtest 에서 RUN 한 설정을 그대로 받아 비용만 바꾼다(Backtest 를 먼저 돌리지 않으면 `NO CONFIG`) · 전략 6 | #301 |
| Baseline | Sharpe 0.2679 · Macro F1 0.3083 | Sharpe 0.0815 · Macro F1 0.2973 | #300 이 이 페이지가 쓰는 판정 함수 `make_band_labels` 를 미래 5일 수익률 기준으로 바꿨다 — 숫자 차이가 그 때문인지는 따로 재지 않았다 |
| Universe | 조합 E | 조합 K | #287 |

## 읽을 때 주의

1. ⚠️ **BEST HARMONIC MODEL XGBoost 는 프로젝트의 공식 최종 모델이 아닙니다.** 피처 조합은 공식 모델과 같아졌지만(MARKET 조합 C), 화면은 네 모델을 정확도 · Macro F1 · 하락 재현율의 **조화평균**으로 줄 세웁니다(`dashboard/services/comparison_service.py:41`). 공식 최종 모델은 **KOSPI200 조합 C · 개별종목 조합 K 의 LogisticRegression** 이고, 고르는 규칙은 [ADR 0007](../../../decisions/0007-최종-모델-선정과-홀드아웃.md) 입니다. Overview 패널은 아직 `BEST MODEL` 로 적혀 있습니다(`dashboard/app.py:168`).
2. 화면의 LogisticRegression 정확도 **0.3736** 은 공식 개발구간 결과 **0.4542**(같은 조합 C · 12폴드 · 표본 밖 720행)와 다릅니다. 원인은 판정하지 않았습니다. 공식 결과는 [루트 README](../../../../README.md) 의 홀드아웃 표와 [모델파트 v1.3](../../../모델파트/version1.3/최종결과-2년과-마지막-구간.md) 에 있습니다.
3. 화면 안의 시각은 **UTC** 입니다(KST 는 9시간을 더합니다 · [이슈 #276](https://github.com/devlee328288/Alpha_Stack/issues/276)). Overview 에 적힌 결과 시각 `2026-09-14 23:21:49`(Baseline) · `23:22:25`(Model Lab)는 KST 09-15 08:21 · 08:22 로, **이번 RUN(09:44 KST)보다 앞섭니다.** 결과 캐시가 서버 메모리에 있어 모든 접속자가 공유하므로, 이번 RUN 은 먼저 계산돼 있던 결과를 받았습니다. RUN 시간이 짧은 이유입니다(대기 루프가 2초 간격 세 번 조용함을 확인하는 데 9초가 걸린다).
4. **비용.** 프리셋 이름과 주석은 왕복 비용인데(`backtest/run_cost_sensitivity.py:21-24`), 엔진은 매수할 때와 매도할 때 `trade_cost` 를 **각각** 곱합니다(`backtest/backtest_strategies.py:353` · `:377`). [이슈 #265](https://github.com/devlee328288/Alpha_Stack/issues/265) 는 2026-09-15 에 닫혔고 이 부분 코드는 그대로입니다. 손익분기 D · E **1.996%** 는 탐색 상한 2%(`run_cost_sensitivity.py:44` `cost_max=0.02`) 바로 아래라, 상한까지 Sharpe 가 0 을 넘었다는 뜻입니다. 비용 표에서 D · E 는 비용이 0.05% → 0.50% 로 오를 때 Sharpe 가 오른 칸이 있습니다(D 0.4123 → 0.4145) — 판정하지 않았습니다.
5. Overview 의 `PIPELINE · Backtest ready A/B/C` 와 `BACKTEST · A/B/C` 머리 글자는 그대로이고, 표에는 전략 6개가 나옵니다.
6. Backtest · Cost Sensitivity 의 기간(2010 ~ 2025)과 비용 입력은 화면 기본값입니다. 공식 홀드아웃(2024-09-02 ~ 2026-09-01)의 비용 차감 성과가 아닙니다.
7. 페이지별 계산 엔진과 배포 기록은 [평가파트 문서](../../../평가파트/README.md) 의 최신 판에 있습니다.
