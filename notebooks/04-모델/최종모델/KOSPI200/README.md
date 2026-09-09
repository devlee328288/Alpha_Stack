<!-- 이 파일은 scripts/update_final_model_docs.py가 생성합니다. 직접 수정하지 마세요. -->
# KOSPI200 현재 잠정 모델

## 상태

현재 운영 중인 `Accuracy·Macro F1·하락 Recall` 조화평균 기준의 잠정 1위입니다.
long-only `{0, +1}` 목적에 맞는 최종 선정 기준은 이슈 #203에서 확인 중이므로 아직 최종
확정으로 표시하지 않습니다.

| 항목 | 값 |
|---|---|
| 조합 | E + 5Day Return |
| 모델 | RandomForest |
| 피처 수 | 4 |
| 선정 지표 | core_harmonic_mean |
| 선정값 | 0.3863 |

## 사용 피처

- `atr_ratio`
- `bb_bandwidth`
- `hv_regime`
- `five_day_return`

## 개발구간 OOS 결과

| 지표 | 값 |
|---|---:|
| Accuracy | 0.3764 |
| Macro F1 | 0.3743 |
| 하락 Recall | 0.4101 |
| 3지표 조화평균 | 0.3863 |
| Balanced Accuracy | 0.3811 |
| MCC | 0.0686 |
| 상승 Recall | 0.3776 |
| 상승 PR-AUC | 0.4084 |

공통 조건은 expanding 12폴드, 최초 학습 750거래일, 검증 60거래일, gap 5입니다.
OOS 예측은 720행이고 예측 분포는 하락 254·보합
236·상승 230입니다.

## 선정 기준별 잠정 1위

| 기준 | 조합·모델 | Accuracy | Macro F1 | 하락 Recall | 예측 분포 하/보/상 |
|---|---|---:|---:|---:|---:|
| Accuracy·Macro F1·하락 Recall 조화평균 | 조합 E 5Day Return·RandomForest | 0.3764 | 0.3743 | 0.4101 | 254/236/230 |
| Accuracy 우선, Macro F1 차순 | 조합 E 기본·LogisticRegression | 0.4486 | 0.3425 | 0.0506 | 32/521/167 |
| Accuracy·Macro F1 동일 가중 조화평균 | 조합 D 기본·LogisticRegression | 0.4389 | 0.3901 | 0.1742 | 92/383/245 |
| Accuracy-보고서 majority_accuracy 우선, Macro F1 차순 | 조합 E 기본·LogisticRegression | 0.4486 | 0.3425 | 0.0506 | 32/521/167 |

Accuracy 우선 후보의 보합 예측 비중처럼 클래스 편향을 함께 확인한 뒤 기준을 확정해야 합니다.
`majority_accuracy` 사용 행은 현재 KOSPI200 보고서에 저장된 값의 단순 비교이며, 개별종목
ADR 0007의 폴드별 학습구간 최빈 기준선과 같은 값이라고 간주하지 않습니다.

정책별 전체 실측값은
[`reports/index_model_selection_comparison.json`](../../../../reports/index_model_selection_comparison.json)에
기록합니다. 지수 파일 SHA-256은 `376c66434db688f42972d2011baee330641421f6b2ca32a8074a96db85ebc13a`입니다.
