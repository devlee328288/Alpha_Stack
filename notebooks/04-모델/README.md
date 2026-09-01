# 04-모델

학습·예측·해석

주로 쓰는 사람: **오준영**

파일 이름은 `NN.주제.ipynb` 입니다.
규칙은 [상위 README](../README.md) 를 보세요.

## 현재 상태

`실험/조합A_rsi14_bb_bandwidth_hv20_vol_ratio20/`에서 같은 피처와 같은 12폴드
워크포워드 조건으로 모델 네 종을 비교합니다.

| 모델 | 전체 OOS Accuracy | 전체 OOS Macro F1 | 하락 Recall |
|---|---:|---:|---:|
| Logistic Regression | 0.3958 | 0.2971 | 0.0000 |
| RandomForest | 0.3736 | 0.3585 | 0.2353 |
| XGBoost | 0.4083 | 0.3446 | 0.0637 |
| LightGBM | 0.3667 | 0.3471 | 0.2010 |

숫자는 KOSPI200 개발구간 720개 OOS 표본의 실측값입니다. Accuracy만 보면 XGBoost가
가장 높고, 세 클래스를 같은 비중으로 보는 Macro F1과 하락 Recall은 RandomForest가
가장 높습니다. 네 모델 모두 최빈 클래스 기준선 Accuracy 0.4250을 넘지 못했습니다.

표·클래스별 Recall·혼동행렬을 한눈에 보는 해석은
[`05.모델비교.md`](실험/조합A_rsi14_bb_bandwidth_hv20_vol_ratio20/05.모델비교.md)에
정리했습니다.
