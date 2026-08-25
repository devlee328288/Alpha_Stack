"""시계열 엔진 — numpy 로 손수 짠 고전 시계열 도구 한 벌.

`scipy` · `statsmodels` 없이 ADF·ACF/PACF·ARIMA 를 직접 구현해 둔 것이다.
원본 개인 프로젝트(data-service)에서 가져왔다.

    import timeseries as ts

    ts.stationarity.adf(x)                  정상성 검정
    ts.correlogram.acf(x) / .pacf(x)        상관도
    ts.models.fit_best(prices)              차수 선택 + 적합 + 확률보행 벤치마크
    ts.forecast.interval(model, 20)         점예측 + 신뢰구간
    ts.backtest.walk_forward(prices)        확장창 검증

| 모듈 | 하는 일 |
|---|---|
| `numerics`     | 정규·카이제곱 분포 함수 (scipy 대체) |
| `transform`    | 로그수익률 · 차분 · 역차분 · 이동평균 |
| `decompose`    | 추세 · 계절 · 잔차 (중심이동평균) |
| `stationarity` | ADF (MacKinnon 임계값 하드코딩) |
| `correlogram`  | ACF(Bartlett) · PACF(Durbin-Levinson) |
| `models`       | AR(OLS) · ARIMA(Hannan-Rissanen) · 확률보행 · AIC 격자탐색 |
| `forecast`     | 점예측 · ψ-weight 신뢰구간 · 상승확률 |
| `backtest`     | walk-forward (rmse · mae · 적중률 · 확률보행 대비) |

1차 프로젝트에서 이 패키지의 자리
------------------------------
ML 사슬(features → models → evaluation)과 **나란히** 선다. 두 가지 몫이 있다.

1. **⑨ 정상성 검정** — 기구현 상태다. 로그수익률이 정상인지 확인하는 근거.
2. **통계적 기준선** — ARIMA 가 우리 분류기와 겨룰 상대다. 트리 모델이 ARIMA 를
   못 이기면 그 사실이 결과의 일부다.

⚠️ `backtest.walk_forward` 를 `evaluation/` 과 혼동하지 않는다
-----------------------------------------------------------
이름이 비슷하지만 하는 일이 다르다.

    timeseries.backtest   ARIMA **전용**. 가격을 받아 rmse·mae 를 잰다 (회귀)
    evaluation.walk_forward  모델 **불문**. 분할만 만들어 준다 (분류·회귀 공용)
    evaluation.metrics       수익률을 받아 MDD·Sharpe·거래비용을 잰다

`timeseries.backtest` 는 ARIMA 를 다시 적합하는 코드가 안에 박혀 있어 트리 모델에
그대로 쓸 수 없다. 그래서 `evaluation/` 을 따로 두되, 그 설계 원칙 — 확장창을 쓸 것,
기준선과 반드시 함께 잴 것 — 은 이 파일에서 그대로 물려받았다.
(→ `evaluation/__init__.py`)
"""

from timeseries import (
    backtest,
    correlogram,
    decompose,
    forecast,
    models,
    numerics,
    stationarity,
    transform,
)

__all__ = ["backtest", "correlogram", "decompose", "forecast",
           "models", "numerics", "stationarity", "transform"]
