# 개별종목 실험

KOSPI200 실험과 같은 방식으로 피처 조합별 폴더 안에 모델 4개와 모델 비교 노트북을
둡니다. 현재 구현된 첫 번째 구성이 조합A입니다.

## 조합

| 조합 | 폴더 | 내용 |
|---|---|---|
| A | `조합A_trend_momentum_volatility_volume_returns` | 추세·모멘텀·변동성·거래량·수익률 12개 |

## 공통 조건

- HF `full/daily_price_dev.parquet`, `full/index_price_dev.parquet`만 사용
- 업종지수 시가총액 상위 10개 × 업종별 KOSPI 보통주 시가총액 상위 5개
- T일 판단 → T+1 `adj_open` 진입 → T+6 `adj_open` 평가
- 종목 수익률 ±2%로 상승·중립·하락 분류
- 날짜 그룹 expanding 12폴드, 최초 학습 750거래일, 검증 60거래일, gap 5
- 각 외부 폴드 안에서 `class_weight=None`과 `balanced`를 다시 비교
- Accuracy·Macro F1·하락 Recall 조화평균으로 선택

조합별 4모델 최선 결과는 `조합별 best result/`에 정리합니다.

`기본모델/`은 `models/`의 네 생성 함수를 확인하는 얇은 실행 노트북입니다. 각 조합
노트북은 같은 생성 함수를 명시적으로 import하고 피처 목록만 지정하므로, Pylance가
동적 `%run` 변수 때문에 내던 미정의 경고 없이 단독 실행할 수 있습니다.
