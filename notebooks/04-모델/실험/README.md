# 04-모델 실험

HF 개발구간 데이터로 KOSPI200의 미래 5거래일 상승·중립·하락을 예측합니다.
홀드아웃 시작일인 `20240901` 이후 데이터는 읽거나 학습에 사용하지 않습니다.

## 예측값과 데이터

- 원천: HF `qurious-quant/alphastack-krx-dev`
- A~F 입력: `full/index_price_dev.parquet`의 `코스피 200`
- 라벨 수익률: 신호일 다음 거래일 시가 `open(t+1)`부터 `open(t+6)`까지
- 분류: 수익률 `+1%` 초과는 상승, `-1%` 미만은 하락, 나머지는 중립
- 마지막 모델 실행의 HF 커밋: `cf3759afebf73a59c8f6c9aa7265cccb56a38f27`

## 피처 뜻

| 피처 | 간단한 설명 |
|---|---|
| `rsi_14` | 최근 14일의 상승·하락 강도 |
| `bb_bandwidth` | 볼린저밴드 폭으로 본 변동성 크기 |
| `hv_20` | 최근 20일 로그수익률의 연환산 표준편차 |
| `vol_ratio_20` | 당일 거래량을 최근 20일 평균 거래량으로 나눈 값 |
| `sma_gap_5_20` | 5일 이동평균이 20일 이동평균에서 떨어진 정도인 단기 추세 |
| `sma_gap_20_60` | 20일 이동평균이 60일 이동평균에서 떨어진 정도인 중기 추세 |
| `macd_hist_ratio` | MACD 히스토그램을 종가로 나눠 가격 단위를 없앤 모멘텀 |
| `macd_hist_atr` | MACD 히스토그램을 ATR로 나눈 변동성 대비 모멘텀 |
| `bb_position` | 현재 종가가 볼린저밴드 하단과 상단 사이 어디에 있는지 표시 |
| `atr_ratio` | 14일 ATR을 종가로 나눈 상대 변동성 |
| `hv_regime` | 현재 20일 변동성을 최근 250일 평균 변동성과 비교한 시장 국면 |
| `obv_slope_20` | 최근 20일 OBV 변화를 평균 거래량으로 정규화한 거래량 방향 |

## A~F 조합

| 조합 | 사용 피처 | 의도 |
|---|---|---|
| A | `rsi_14`, `bb_bandwidth`, `hv_20`, `vol_ratio_20` | 강도·변동성·거래량을 고르게 넣은 기본 조합 |
| B | `sma_gap_5_20`, `macd_hist_ratio`, `rsi_14`, `hv_20` | 단기 추세와 모멘텀 중심 조합 |
| C | B + `bb_position`, `vol_ratio_20` | B에 가격 위치와 거래량 확인 신호를 추가 |
| D | `sma_gap_20_60`, `rsi_14`, `atr_ratio`, `bb_bandwidth`, `hv_regime`, `obv_slope_20` | 중기 추세와 변동성 국면 중심 조합 |
| E | `atr_ratio`, `bb_bandwidth`, `hv_regime` | 방향성 피처를 줄이고 변동성만 본 최소 조합 |
| F | D + `sma_gap_5_20`, `macd_hist_atr` | 단기·중기 추세, 모멘텀, 변동성, 거래량을 모두 넣은 최대 조합 |

각 조합에는 다음 수익률 변형을 별도로 실행했습니다.

| 변형 | 뜻 |
|---|---|
| 기본 | 조합 고유 피처만 사용 |
| `Daily_Return` | 전일 종가 대비 당일 종가 수익률 추가 |
| `5Day_Return` | 5거래일 전 종가 대비 당일 종가 수익률 추가 |
| `Daily_Return + 5Day_Return` | 두 수익률을 모두 추가 |

지수에는 액면분할이 없으므로 A~F 수익률은 KOSPI200 지수 종가로 계산합니다. 개별 종목
데이터를 사용하는 추가 실험에서는 액면분할 왜곡을 막기 위해 반드시 `adj_close`를 씁니다.

## 공통 실험 조건

- expanding walk-forward 12폴드
- 최초 학습 750거래일
- 폴드별 검증 60거래일
- 학습과 검증 경계 직전 5거래일 제거
- 외부 폴드마다 학습구간 안의 마지막 60거래일로 `None`과 `balanced`를 다시 비교
- 클래스 가중치와 최우수 모델 선정 기준: Accuracy·Macro F1·하락 Recall의 조화평균
- 매매 평가: 상승 예측만 다음 시가 진입, 5슬리브로 5거래일 보유, 왕복비용 0.05%
- 상승 예측이 전혀 없는 폴드는 평가용 전략 Sharpe를 0으로 기록

학습창 후보 비교에서는 폴드별 순비용 ΔSharpe 중앙값이 가장 높았던 `expanding`을 모든
조합에 공통 적용했습니다.

## 조합별 최우수 결과

각 조합 안에서 네 모델과 네 수익률 변형 중 핵심지표 조화평균이 가장 높은 결과입니다.

| 조합 | 모델 | 변형 | Accuracy | 보합 기준선 | Macro F1 | 하락 Recall | 조화평균 | ΔSharpe 중앙값 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| A | LightGBM | Daily Return | 0.3944 | 0.4250 | 0.3839 | 0.3436 | 0.3726 | 0.2024 |
| B | XGBoost | 5Day Return | 0.3778 | 0.4222 | 0.3719 | 0.3918 | 0.3803 | -0.1252 |
| C | XGBoost | 5Day Return | 0.3958 | 0.4222 | 0.3847 | 0.3557 | 0.3780 | -0.1648 |
| D | RandomForest | 기본 | 0.3681 | 0.4181 | 0.3548 | 0.2472 | 0.3131 | 0.0182 |
| **E** | **RandomForest** | **5Day Return** | **0.3764** | **0.4181** | **0.3743** | **0.4101** | **0.3863** | **1.0723** |
| F | Logistic Regression | 5Day Return | 0.4000 | 0.4181 | 0.3764 | 0.2809 | 0.3442 | 0.0799 |

전체 최우수는 조화평균이 `0.3863`인 조합 E의 RandomForest + 5Day Return입니다.
다만 Accuracy는 각 조합의 전부 보합 기준선보다 낮습니다. 피처별 워밍업 길이가 달라
A~C와 D~F의 OOS 날짜 및 보합 기준선도 조금 다릅니다. 이 결과는 개발구간 OOS 비교이며
봉인 홀드아웃의 최종 성능이 아닙니다.

모델별 선정 이유와 결과는 `조합별 best result/`에 있으며, 전체
96개 실측값은 [`reports/model_sweep.json`](../../../reports/model_sweep.json)에 있습니다.

## HF 전 종목 시장 내부 피처 추가 실험

`full/daily_price_dev.parquet`의 그날 존재한 KOSPI 종목을 집계해 다음 피처를 시험했습니다.

- 상승 종목 수와 하락 종목 수의 차이인 `AD Percent`
- 200일 이동평균 위에 있는 종목 비율
- 상승·하락 종목 수와 거래량을 함께 보는 `log TRIN`
- 종목별 일간수익률의 단면 분산
- 가격이 변하지 않은 종목 비율

시장 내부 피처만 사용했을 때 최고 Accuracy는 LightGBM의 `0.3389`였습니다. 기존
F + 5Day Return에 결합했을 때는 Logistic Regression의 Accuracy `0.3917`이 가장 높았지만,
같은 OOS의 전부 보합 기준선 `0.4181`을 넘지 못했습니다. 따라서 이 피처는 A~F 정식 조합에
추가하지 않았습니다.

상세 결과는 [`market_feature_experiment.json`](../../../reports/market_feature_experiment.json)과
[`combined_market_feature_experiment.json`](../../../reports/combined_market_feature_experiment.json)에
기록했습니다.
