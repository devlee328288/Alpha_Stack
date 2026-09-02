# 04-모델

학습·예측·해석

주로 쓰는 사람: **오준영**

파일 이름은 `NN.주제.ipynb` 입니다.
규칙은 [상위 README](../README.md) 를 보세요.

## 현재 상태

`실험/조합A_rsi14_bb_bandwidth_hv20_vol_ratio20/`에서 같은 피처와 같은 12폴드
워크포워드 조건으로 모델 네 종을 비교합니다.

| 모델 | 클래스 가중치 | 전체 OOS Accuracy | 전체 OOS Macro F1 | 하락 Recall |
|---|---|---:|---:|---:|
| Logistic Regression | balanced | 0.4028 | 0.3570 | 0.1176 |
| RandomForest | None | 0.3736 | 0.3585 | 0.2353 |
| XGBoost | balanced | 0.4111 | 0.3958 | 0.2500 |
| LightGBM | balanced | 0.3653 | 0.3569 | 0.2745 |

숫자는 KOSPI200 개발구간 720개 OOS 표본의 실측값입니다. RandomForest는 balanced에서
주요 지표가 악화되어 기본 가중치를 유지했고, 나머지 세 모델은 balanced를
적용했습니다. Accuracy와 Macro F1은 XGBoost가 가장 높고, 하락 Recall은
LightGBM이 가장 높습니다. 네 모델 모두 최빈 클래스 기준선 Accuracy 0.4250을
넘지 못했습니다.

표·클래스별 Recall·혼동행렬을 한눈에 보는 해석은
[`05.모델비교.ipynb`](실험/조합A_rsi14_bb_bandwidth_hv20_vol_ratio20/05.모델비교.ipynb)에
정리했습니다.

### 조합 B

`실험/조합B_sma_gap_5_20_macd_hist_ratio_rsi14_hv20/`에서 네 모델을 모두
`class_weight="balanced"`로 비교합니다. XGBoost는 같은 의미의 표본 가중치로
변환해 학습합니다.

| 모델 | 전체 OOS Accuracy | 전체 OOS Macro F1 | 하락 Recall |
|---|---:|---:|---:|
| Logistic Regression | **0.3792** | 0.3258 | 0.0784 |
| RandomForest | 0.3514 | 0.3392 | 0.2598 |
| XGBoost | 0.3708 | 0.3556 | 0.2500 |
| LightGBM | 0.3681 | **0.3570** | **0.3039** |

Accuracy는 Logistic Regression이 가장 높고, Macro F1과 하락 Recall은 LightGBM이
가장 높습니다. 하지만 네 모델 모두 최빈 클래스 기준선 Accuracy 0.4250을
넘지 못했습니다. 표·클래스별 Recall·혼동행렬은
[조합 B `05.모델비교.ipynb`](실험/조합B_sma_gap_5_20_macd_hist_ratio_rsi14_hv20/05.모델비교.ipynb)에
정리했습니다.

### 조합 C

`실험/조합C_sma_gap_macd_rsi_bb_hv_volume/`에서 조합 B의 네 피처에
`bb_position`과 `vol_ratio_20`을 추가한 여섯 피처로 네 모델을 비교합니다.
네 모델 모두 `class_weight="balanced"`를 사용합니다.

| 모델 | 전체 OOS Accuracy | 전체 OOS Macro F1 | 하락 Recall |
|---|---:|---:|---:|
| Logistic Regression | **0.3722** | 0.3421 | 0.1471 |
| RandomForest | 0.3597 | 0.3464 | 0.2696 |
| XGBoost | **0.3722** | 0.3582 | 0.2647 |
| LightGBM | 0.3708 | **0.3599** | **0.2990** |

Accuracy는 Logistic Regression과 XGBoost가 공동 1위고, Macro F1과 하락 Recall은
LightGBM이 가장 높습니다. 네 모델 모두 최빈 클래스 기준선 Accuracy 0.4250을
넘지 못했습니다. 표·클래스별 Recall·혼동행렬은
[조합 C `05.모델비교.ipynb`](실험/조합C_sma_gap_macd_rsi_bb_hv_volume/05.모델비교.ipynb)에
정리했습니다.

### 조합 D — 이슈 #37의 2안

`실험/조합D_option2_volatility_focus/`에서 장기 추세와 단기·장기 변동성 국면,
거래량 보정 OBV 기울기를 함께 사용하는 여섯 피처로 네 모델을 비교합니다.
기본 가중치와 비교한 결과 네 모델 모두 `class_weight="balanced"`를 사용합니다.

| 모델 | 전체 OOS Accuracy | 전체 OOS Macro F1 | 하락 Recall |
|---|---:|---:|---:|
| Logistic Regression | 0.3806 | 0.3565 | 0.2206 |
| RandomForest | 0.3806 | 0.3711 | 0.2647 |
| XGBoost | 0.3611 | 0.3542 | 0.3039 |
| LightGBM | **0.3833** | **0.3794** | **0.3137** |

Accuracy, Macro F1, 하락 Recall 모두 LightGBM이 가장 높습니다. 다만 네 모델 모두
최빈 클래스 기준선 Accuracy 0.4250을 넘지 못했습니다. 표·클래스별 Recall·혼동행렬은
[조합 D `05.모델비교.ipynb`](실험/조합D_option2_volatility_focus/05.모델비교.ipynb)에
정리했습니다.

### 조합 E — 이슈 #37의 5안

`실험/조합E_option5_volatility_only/`에서 `atr_ratio`, `bb_bandwidth`,
`hv_regime`의 변동성 피처 세 개만으로 네 모델을 비교합니다. 기본값과 balanced를
비교해 Logistic Regression과 XGBoost는 기본값, RandomForest와 LightGBM은
balanced를 최종 설정으로 선택했습니다.

| 모델 | 클래스 가중치 | 전체 OOS Accuracy | 전체 OOS Macro F1 | 하락 Recall |
|---|---|---:|---:|---:|
| Logistic Regression | None | **0.4111** | 0.3086 | 0.1029 |
| RandomForest | balanced | 0.3972 | **0.3928** | 0.3824 |
| XGBoost | None | 0.3792 | 0.3473 | 0.2549 |
| LightGBM | balanced | 0.3750 | 0.3736 | **0.3971** |

Accuracy는 Logistic Regression, Macro F1은 RandomForest, 하락 Recall은 LightGBM이
가장 높습니다. 네 모델 모두 최빈 클래스 기준선 Accuracy 0.4250을 넘지 못했습니다.
표·클래스별 Recall·혼동행렬은
[조합 E `05.모델비교.ipynb`](실험/조합E_option5_volatility_only/05.모델비교.ipynb)에
정리했습니다.

### 조합 F — 이슈 #37의 6안

`실험/조합F_option6_maximum/`에서 조합 D의 여섯 피처에 `sma_gap_5_20`과
`macd_hist_atr`를 추가한 최대 8개 피처로 네 모델을 비교합니다. 기본값과 balanced를
비교해 Logistic Regression과 XGBoost는 기본값, RandomForest와 LightGBM은
balanced를 최종 설정으로 선택했습니다.

| 모델 | 클래스 가중치 | 전체 OOS Accuracy | 전체 OOS Macro F1 | 하락 Recall |
|---|---|---:|---:|---:|
| Logistic Regression | None | **0.3931** | 0.3417 | 0.1569 |
| RandomForest | balanced | 0.3917 | **0.3840** | 0.3039 |
| XGBoost | None | 0.3889 | 0.3602 | 0.1961 |
| LightGBM | balanced | 0.3861 | 0.3831 | **0.3431** |

Accuracy는 Logistic Regression, Macro F1은 RandomForest, 하락 Recall은 LightGBM이
가장 높습니다. 네 모델 모두 최빈 클래스 기준선 Accuracy 0.4250을 넘지 못했습니다.
표·클래스별 Recall·혼동행렬은
[조합 F `05.모델비교.ipynb`](실험/조합F_option6_maximum/05.모델비교.ipynb)에
정리했습니다.
