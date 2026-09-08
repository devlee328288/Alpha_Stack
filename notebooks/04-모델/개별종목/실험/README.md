# 개별종목 실험

KOSPI200 실험과 같은 방식으로 피처 조합별 폴더 안에 모델 4개, 모델 비교 노트북,
`피처선정.md`를 둡니다. A~H는 공통 `(bas_dd, code)` 표본에서 피처만 바꿉니다.

## 조합

| 조합 | 폴더 | 내용 |
|---|---|---|
| A | `조합A_trend_momentum_volatility_volume_returns` | 추세·모멘텀·변동성·거래량·수익률 12개 |
| B | `조합B_trend_momentum_high_distance` | 5거래일 단기 반전·고점 거리 7개 |
| C | `조합C_short_reversal_intraday` | 단기 반전·장중 위치 7개 |
| D | `조합D_volatility_liquidity` | 변동성·유동성 8개 |
| E | `조합E_sector_market_relative_strength` | 업종·시장 상대강도 7개 |
| F | `조합F_cross_sectional_ranks` | 당일 후보군 횡단면 순위 9개 |
| G | `조합G_direction_magnitude_interaction` | 단기 반전 방향축·변동성 크기축 6개 |
| H | `조합H_volatility_regime_interaction` | 조합 G의 `hv_20`을 `hv_regime`으로 교체한 6개 |

## 공통 조건

- HF `full/daily_price_dev.parquet`, `full/index_price_dev.parquet`만 사용
- 업종지수 시가총액 상위 10개 × 업종별 KOSPI 보통주 시가총액 상위 5개
- T일 판단 → T+1 `adj_open` 진입 → T+6 `adj_open` 평가
- 종목 수익률 ±2%로 상승·중립·하락 분류
- `±2%` 근거와 평가 기준선은 `docs/decisions/0006-개별종목-중립대와-평가기준선.md`에 기록
- 날짜 그룹 expanding 12폴드, 최초 학습 750거래일, 검증 60거래일, gap 5
- 각 외부 폴드 안에서 `class_weight=None`과 `balanced`를 다시 비교
- class weight는 Accuracy·Macro F1·하락 Recall 조화평균으로 내부 선택
- 최종 조합·모델은 기준선 대비 Accuracy → Macro F1 → 기준선 승리 폴드 수로 선택
- MCC·Balanced Accuracy·클래스별 및 Macro PR-AUC·혼동행렬 함께 기록
- 폴드별 학습 최빈 Accuracy 기준선과 모델 Accuracy의 차이를 함께 기록
- 검증 최빈 비율은 정답을 본 `oracle` 참고값으로만 표시
- A~H 공통 159,936개 `(bas_dd, code)` 행·3,343거래일에서 비교
- `hv_regime`은 269거래일 준비구간이 필요하므로 H를 포함한 공통 표본은 A~G만 비교할 때보다 짧음
- `|수정종가 수익률| > 100%` 자동 제거는 사용하지 않음
- KRX 등락률과 수정주가 수익률이 1%p 넘게 어긋난 `is_adj_suspect`만 제외
- 최신 후보 176,705행의 의심 행은 0행, 보존한 실제 극단 사건은 5행

## 조합별 1위

| 전체 순위 | 조합 | 모델 | 기준선 대비 Accuracy | 기준선 승리 |
|---:|---|---|---:|---:|
| 1 | A | LogisticRegression | **+0.0172** | 8/12 |
| 2 | D | XGBoost | **+0.0148** | 9/12 |
| 3 | G | LogisticRegression | **+0.0138** | 7/12 |
| 4 | H | LogisticRegression | **+0.0102** | 6/12 |
| 5 | B | LogisticRegression | **+0.0092** | 6/12 |
| 6 | E | XGBoost | **+0.0033** | 5/12 |
| 7 | F | LogisticRegression | **-0.0024** | 4/12 |
| 8 | C | XGBoost | **-0.0052** | 4/12 |

조합별 4모델 최선 결과는 `조합별 best result/`에 정리합니다.

`기본모델/`은 `models/`의 네 생성 함수를 확인하는 얇은 실행 노트북입니다. 각 조합
노트북은 같은 생성 함수를 명시적으로 import하고 피처 목록만 지정하므로, Pylance가
동적 `%run` 변수 때문에 내던 미정의 경고 없이 단독 실행할 수 있습니다. 실제 1,152회
fit은 공통 실행기에서 한 번 수행하고, 노트북은 보존된 실측 리포트를 읽습니다.
