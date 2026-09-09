<!-- 이 파일은 scripts/update_final_model_docs.py가 생성합니다. 직접 수정하지 마세요. -->
# 개별종목 최종 모델

ADR 0007의 `학습 최빈 기준선 대비 Accuracy → Macro F1 → 기준선 승리 폴드 수` 순서로
선택한 개발구간 1위입니다.

| 항목 | 값 |
|---|---|
| 조합 | K |
| 모델 | LogisticRegression |
| 피처 수 | 14 |
| 기준선 승리 | 10/12 |

## 사용 피처

- `atr_ratio`
- `bb_bandwidth`
- `hv_regime`
- `five_day_return`
- `relative_ret_5_market`
- `sma_gap_5_20`
- `sma_gap_20_60`
- `rsi_14`
- `macd_hist_ratio`
- `bb_position`
- `hv_20`
- `vol_ratio_20`
- `obv_slope_20`
- `daily_return`

## 개발구간 OOS 결과

| 지표 | 값 |
|---|---:|
| Accuracy | 0.4211 |
| 학습 최빈 기준선 Accuracy | 0.3969 |
| 기준선 대비 Accuracy | +0.0243 |
| Macro F1 | 0.3781 |
| 하락 Recall | 0.3058 |
| 기존 3지표 조화평균 | 0.3479 |
| Balanced Accuracy | 0.3920 |
| MCC | 0.0994 |
| Macro PR-AUC | 0.4018 |
| 기준선 승리 폴드 | 10/12 |

공통 조건은 날짜 그룹 expanding 12폴드, 최초 학습 750거래일, 검증 60거래일,
gap 5입니다. 중립대는 `±2.0%`, 라벨은 T일 정보로 예측한
`T+1 adj_open → T+6 adj_open` 수익률입니다.

## 기본 출력과 매수 조건

매 거래일 업종 시가총액 상위 10개 × 업종별 보통주 시가총액 상위 5개, 최대 50종목을
전부 출력합니다. 종목별 예측·확률·실제 라벨·적중 여부와 업종별·전체 적중률을 남깁니다.
실제 매수는 KOSPI200과 개별종목이 모두 상승으로 예측된 경우만 허용합니다.

공통 OOS 720거래일의 기본 출력은 34,649행이며 전체 적중률은 0.4216, 상승 Precision은 0.3692, 실제 매수 신호는 3,258건입니다.

Top 1·3·5는 기본 후보를 줄이는 규칙이 아니라 부가 실험입니다.
