# 05-백테스트·성과

# 📊 성과 및 백테스트 평가 지표 가이드

이 문서는 퀀트/알고리즘 트레이딩 백테스트 결과를 평가하기 위한 **성과 지표(Performance Metrics)** 및 **머신러닝 평가 지표(Machine Learning Metrics)**를 정의하고 설명합니다. 
해당 문서는 프로젝트의 백테스트 및 모델 평가 시 참고 자료로 사용됩니다.

---

## 🚀 1. 성과 지표 (Performance Metrics)

포트폴리오/전략의 수익성, 위험, 그리고 위험 대비 효율성을 측정합니다.

| 지표 | 설명 | 핵심 포인트 |
| :--- | :--- | :--- |
| **MDD** (Maximum Drawdown) | 분석 기간 중 고점 대비 최대 손실률 (최대 낙폭) | **분모의 기본 단위**. 리스크의 극단적인 크기를 측정합니다. |
| **Sharpe Ratio** (샤프 비율) | (수익률 - 무위험 수익률) / 표준편차 | **변동성 대비 초과 수익률**. 가장 보편적인 리스크 조정 지표입니다. |
| **Sortino Ratio** (소티노 비율) | (수익률 - MAR) / **하방 변동성 (Downside Deviation)** | **하락 변동성만 분모로 사용**. MAR(최소 요구 수익률) 미만의 음수 수익률만 위험으로 간주합니다. |
| **Sterling Ratio** (스털링 비율) | CAGR / (**평균 MDD** + 10%) | **상위 N개 MDD의 평균**을 분모로 사용. Calmar보다 이상치(Outlier)에 덜 민감합니다. |
| **Calmar Ratio** (칼마 비율) | CAGR / (**최대 MDD**) | **단일 최대 MDD**를 분모로 사용. 가장 보수적인 하방 리스크 지표입니다. |
| **Deflated Sharpe Ratio** (디플레이트 샤프) | 표준정규 CDF 기반 통계량 | **백테스트 횟수(N)와 왜도/첨도를 보정**. 수많은 파라미터 튜닝(오버피팅)으로 인해 우연히 높아진 샤프 비율을 통계적으로 무효화(Deflate)합니다. (값은 0~1 사이의 신뢰 확률로 해석) |

---

## 📈 2. 백테스트 회귀 태스크 지표 (Regression Metrics)

팩터 점수나 예측값을 통해 **연속적인 수치(미래 수익률 등)를 예측**했을 때의 정확도를 평가합니다.

| 지표 | 설명 | 수식/특징 |
| :--- | :--- | :--- |
| **MAE** (Mean Absolute Error) | 예측값과 실제값 간 **절대 오차의 평균** | `(1/n) * Σ |y - ŷ|` <br>이상치에 강건(Robust)하며 직관적인 해석이 가능합니다. |
| **RMSE** (Root Mean Square Error) | 오차를 제곱하여 평균낸 뒤 **제곱근**을 취한 값 | `sqrt((1/n) * Σ (y - ŷ)^2)` <br>큰 오차(이상치)에 매우 민감하며, 예측의 극단적 실패 리스크를 평가할 때 필수입니다. |
| **IC** (Information Coefficient) | 예측값(Predicted)과 실제값(Actual) 간의 **상관관계(Correlation)** | 팩터의 예측 방향성과 강도를 측정합니다. (-1 ~ +1). 실무에서는 이상치에 강한 **Rank IC (스피어만 순위 상관)**를 주로 사용합니다. |
| **ICIR** (IC Information Ratio) | **IC의 평균 / IC의 표준편차** | IC의 **일관성(Consistency)**을 측정합니다. IC의 샤프 비율(Sharpe Ratio) 버전으로, 값이 높을수록 예측력이 매 기간 안정적으로 유지됨을 의미합니다. |

---

## 🎯 3. 백테스트 분류 태스크 지표 (Classification Metrics)

**상승/하락(방향성)** 또는 **이상 거래 여부(불균형 데이터)**를 예측했을 때의 성능을 평가합니다.

### 📋 3.1. 기초 혼동 행렬 (Confusion Matrix)
모든 분류 지표의 근간이 되는 2x2 행렬입니다.

| | 예측: Positive | 예측: Negative |
| :--- | :--- | :--- |
| **실제: Positive** | **TP** (진짜 양성) - 정답 | **FN** (거짓 음성) - **2종 오류** (기회 상실) |
| **실제: Negative** | **FP** (거짓 양성) - **1종 오류** (허위 경보/손실) | **TN** (진짜 음성) - 정답 |

### 📊 3.2. 분류 평가 지표 목록

| 지표 | 설명 | 핵심 포인트 (불균형 데이터 대응) |
| :--- | :--- | :--- |
| **클래스 비율 (Class Ratio)** | 전체 데이터 중 **양성(Positive)의 비율** | **PR-AUC의 무작위 기준선**이자, Accuracy 함정을 파악하는 첫 단계입니다. (예: 양성 1% vs 음성 99%) |
| **MCC** (Matthews Correlation Coef.) | 혼동 행렬의 4개 요소(TP,TN,FP,FN)를 모두 고려한 **상관계수** | **-1 ~ +1**. 데이터 불균형이 심해도 신뢰할 수 있는 단일 지표입니다. (0.3 이상이면 실무적 의미 있음) |
| **PR-AUC** (Precision-Recall AUC) | **정밀도(Precision)**와 **재현율(Recall)** 곡선의 아래 면적 | **양성 클래스의 탐지 성능**에 집중합니다. ROC-AUC와 달리 불균형이 심할수록 값이 급감하여 진짜 성능을 보여줍니다. |
| **Balanced Accuracy** | (민감도(TPR) + 특이도(TNR)) / 2 | **클래스 비율과 무관하게 항상 0.5가 무작위 기준선**입니다. 다수 클래스만 잘 맞추는 모델을 즉시 필터링합니다. |

---

## 📁 관련 문서 (Documentation)

해당 지표들에 대한 자세한 수식과 계산 예시는 아래 HTML 문서들을 참조하세요.

### 성과 지표 (Performance Metrics)
- `MDD.html`
- `Sharpe_Ratio.html`
- `Sortino_Ratio.html`
- `Sterling_Ratio.html`
- `Calmar_Ratio.html`
- `Deflated_Sharpe_Ratio.html`

### 백테스트 평가 지표 (Backtest Metrics)
- **회귀 (Regression)**
  - `MAE.html`
  - `RMSE.html`
  - `IC.html`
  - `ICIR.html`
- **분류 (Classification)**
  - `MCC.html`
  - `PR-AUC.html`
  - `Balanced_Accuracy.html`
  - `Confusion_Matrix.html`
  - `Class_Ratio.html`

---
*해당 문서는 프로젝트 내 백테스트 엔진 및 머신러닝 평가 파이프라인의 기준으로 활용됩니다.*