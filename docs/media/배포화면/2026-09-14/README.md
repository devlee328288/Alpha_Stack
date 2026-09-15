# 배포 화면 캡처 — 2026-09-14 (HF Space)

> 강사님 서류용 · 캡처 이동원 · 대시보드 코드는 강민석 님(PR #261 · #262 · #269 · #279)
> 주소 https://data-student-alphastack-qurious.hf.space/ (Space `data-student/alphastack-qurious` · cpu-basic) — main `5e70ee1` 을 Factory rebuild 한 빌드
> 찍은 방법 — puppeteer 25.3.0 headless · 창 너비 1600 · 본문 높이만큼 창을 늘려 한 장 · 사이드바를 눌러 한 세션 안에서 이동 · **설정은 전부 화면 기본값** · 스코프 MARKET(KOSPI200)
>
> ⚠️ **Model Lab 이 조합 E 이던 시점의 화면입니다.** 강민석 님이 공식 조합(KOSPI200 C · 개별종목 K)으로 고치겠다고 답했습니다([#280](https://github.com/devlee328288/Alpha_Stack/issues/280) · 2026-09-14 14:17 KST).
> 🔁 **2026-09-15 — 수정 PR(#287 · #289 · #300 · #301) 머지와 Space rebuild 뒤 다시 찍은 화면은 [2026-09-15](../2026-09-15/README.md) 에 있습니다.** 이 폴더는 조합 E 시점의 기록으로 남깁니다.

## 화면 목록

순서는 찍은 순서가 아니라 사이드바 순서입니다. Overview 는 다른 페이지의 결과를 모아 보여 주므로 맨 마지막에 찍었습니다.

| # | 파일 | 페이지 | 상태 | RUN 걸린 시간 | 찍은 시각 (KST) | 화면에서 읽을 것 |
|---|---|---|---|---:|---|---|
| 00 | [00_Overview.png](00_Overview.png) | Overview | RUN 없음 — 네 단계 결과를 모아 보여 준다 | — | 14:13:05 | PIPELINE 4 STAGES 모두 ready · Baseline Sharpe 0.2679 · BEST MODEL LightGBM(아래 주의 ①) · 전략 C 손익분기 비용 0.379% |
| 01 | [01_Baseline.png](01_Baseline.png) | Baseline | ▶ RUN BASELINE (Threshold 1% · CMA-ES QUICK 30) | 11초 | 14:10:21 | 12폴드 · Sharpe 0.2679 · CAGR 3.23% · MDD −26.46% · Macro F1 0.3083 |
| 02 | [02_Model_Lab.png](02_Model_Lab.png) | Model Lab | ▶ RUN (4모델 · 조합 E · 라벨 fwd_return ±1%) | 35초 | 14:11:08 | BEST MODEL LightGBM · Harmonic 0.3371 · Accuracy 0.3389(아래 주의 ①) |
| 03 | [03_Comparison.png](03_Comparison.png) | Comparison | RUN 없음 — Baseline · Model Lab 결과를 읽는다 | — | 14:11:19 | 라벨 분포 상승 fwd_return 33.6% · Adaptive 17.7% · 노란 안내는 Adaptive 라벨로 학습한 모델이 없다는 기본 안내 |
| 04 | [04_Backtest.png](04_Backtest.png) | Backtest | ▶ RUN ALL (A/B/C) · RandomForest × fwd_return · 2010-01-01 ~ 2025-01-01 · Trade Cost 0.0010 | 7초 | 14:11:37 | 전략 C 총수익 0.2899 · Sharpe 0.2147 · MDD −0.3655 |
| 05 | [05_Risk.png](05_Risk.png) | Risk | RUN 없음 · 탭 5개 중 OVERVIEW | — | 14:11:48 | 전략 A · MDD −38.57% · Sharpe −0.1925(무위험 2%) |
| 06 | [06_Cost_Sensitivity.png](06_Cost_Sensitivity.png) | Cost Sensitivity | ▶ RUN (RandomForest · 비용 프리셋 4 × 전략 3) | 21초 | 14:12:21 | 손익분기 비용(Sharpe = 0) A 0.092% · B 0.340% · C 0.379% |
| 07 | [07_System.png](07_System.png) | System | RUN 없음 | — | 14:12:32 | 엔진 4/4 · streamlit 1.63.0 · exchange-calendars 미설치 · 결과 키 5 |
| 08 | [08_Universe.png](08_Universe.png) | Universe | **RUN 하지 않음** — 설정 화면 | — | 14:12:43 | 화면 안내상 30종목 RUN 예상이 기준선 3분 · 모델 37분/개라 이번 캡처에서 뺐다 |
| 09 | [09_QFRS.png](09_QFRS.png) | QFRS | RUN 없음 · RISK 탭 | — | 14:12:54 | 전략 A 점수 11 / 100 POOR |

## 읽을 때 주의

1. ⚠️ **Model Lab · Overview 의 BEST MODEL LightGBM 은 프로젝트의 공식 최종 모델이 아닙니다.** 화면은 조합 E 를 조화평균(정확도 · Macro F1 · 하락 재현율)으로 줄 세운 별도 실험이고, 공식 최종 모델은 **KOSPI200 조합 C · 개별종목 조합 K 의 LogisticRegression** 입니다([이슈 #280](https://github.com/devlee328288/Alpha_Stack/issues/280) · [ADR 0009](../../../decisions/0009-최종-홀드아웃-실행설정.md)). 공식 2년 홀드아웃 결과는 [루트 README](../../../../README.md) 의 홀드아웃 표에 있습니다.
2. 화면 안의 시각(예: `2026-09-14 05:10:13`)은 **UTC** 입니다. KST 는 9시간을 더합니다([이슈 #276](https://github.com/devlee328288/Alpha_Stack/issues/276)).
3. 이 대시보드는 버튼을 누르면 **서버에서 학습**합니다. 결과 캐시는 서버 메모리에 있어 모든 접속자가 공유하고, 재시작하면 사라집니다. 같은 설정으로 다시 돌렸을 때 숫자가 같은지는 확인하지 않았습니다.
4. Backtest · Cost Sensitivity 의 기간(2010 ~ 2025)과 비용 입력은 화면 기본값입니다. 비용이 편도인지 왕복인지는 [이슈 #265](https://github.com/devlee328288/Alpha_Stack/issues/265) 에서 논의 중입니다.
5. 페이지별 계산 엔진과 배포 기록은 [평가파트 v1.3 — 대시보드 9페이지와 배포](../../../평가파트/version1.3/대시보드_9페이지와_배포.md) 에 있습니다.
