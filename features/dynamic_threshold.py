"""유동 임계값 — 변동성에 비례해 레이블 경계값을 매 시점 다시 잰다
(피처 엔지니어링 계층, 계획서 v3.0 필수 ⑤ · 신장환 담당)

`features/` 계약(기능명세 §3.3)은 `indicators.py`와 같다 — 자세한 설명은 그쪽 모듈
docstring 참고, 여기서는 요약만 적는다.

    t 행은 t 시점까지의 자료만 쓴다 (look-ahead 금지)
    ------------------------------------------------
    `volatility.historical_volatility`가 t 시점까지의 롤링 창만 보므로, 그 위에 상수를
    곱하기만 하는 이 모듈도 같은 성질을 그대로 물려받는다.

⚠️ 이 모듈은 `volatility.py`를 import 한다(`historical_volatility`) — `indicators.py`·
`volume.py`·`volatility.py` 세 파일이 서로 import 하지 않기로 한 원칙과는 다르다. 그
셋은 같은 층(원자 지표)이라 독립성을 지켰지만, 이 모듈은 그 위층(파생/조합)이다.
`indicators.percent_b`·`sma_gap`이 `sma`·`bollinger_bands`를 그대로 갖다 쓴 것과 같은
이유로, 변동성 계산 로직을 다시 짜지 않고 재사용한다 — 원자 함수가 나중에 고쳐지면
이 값도 자동으로 같이 고쳐진다.

주 레이블(±1.0% 고정, ADR-AS-0002)을 대체하지 않는다 — 사전등록(ADR-AS-0004 §9
"무엇을 포기했나 ①")이 이미 "보조 탐색(exploratory)"으로 여러 임계값을 함께 보는
것을 허용해 뒀다. 이 모듈이 만드는 것도 그 자리다: 고정 밴드와 나란히 도는 부 실험용
임계값이지, 봉인을 다시 여는 것이 아니다.

공식과 배율의 출처
------------------
    threshold_t = k · σ_5d(t)
    σ_5d(t)     = historical_volatility(prices, window, ddof, annualize=False)[t] · √horizon

- `k=0.40` — 새로 고른 값이 아니라 ADR-AS-0002가 이미 실측해 둔 "±1.0%는 약 ±0.40σ
  (5일 σ≈2.53%)"를 그대로 배율로 쓴다.
- `√horizon` — 일별 변동성을 5거래일 지평으로 스케일하는 근사. ADR-AS-0002가 손익분기
  계산에 쓴 것과 같은 근사다(일간 0.904%×√5 ≈ 5일 2.53%).
- `window=20` — 새 결정이 아니라 `historical_volatility`의 기존 기본값을 그대로 쓴다.

개별 종목(③번)에도 이 공식을 그대로 쓸지, 결과를 `evaluation/` 리포트 어디에
얹을지는 아직 정해지지 않았다 — 이 함수 바깥의 일이라 이 모듈은 관여하지 않는다.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from features.volatility import historical_volatility

#: ADR-AS-0002 실측값("±1.0%는 약 ±0.40σ")을 그대로 쓰는 배율 — 새로 고른 값이 아니다.
DEFAULT_K = 0.40

#: 레이블 지평(거래일). ADR-AS-0002가 고정한 5거래일과 같다.
DEFAULT_HORIZON = 5


def dynamic_threshold(
    prices: Sequence,
    window: int = 20,
    k: float = DEFAULT_K,
    horizon: int = DEFAULT_HORIZON,
    ddof: int = 1,
) -> np.ndarray:
    """변동성에 비례하는 유동 임계값 — 고정 ±1.0% 대신 매 시점 다시 잰 경계값.
    공식과 배율의 출처는 모듈 docstring 참고.

    `historical_volatility`가 이미 t 시점까지의 정보만 쓰므로(look-ahead 없음), 이 함수는
    그 결과에 상수를 곱하는 것 말고는 아무것도 하지 않는다 — 앞쪽 `window`행이 `nan`인
    것도, 결측이 창에 섞이면 그 창 전체가 `nan`인 것도 `historical_volatility`의 동작을
    그대로 물려받는다.

    3분류에 쓰려면 `evaluation.horizon.classify_3`처럼 `ret_5(t)`를
    `[-threshold_t, +threshold_t]`와 비교하면 된다 — 다만 그 비교 함수는 지금
    스칼라 `band`를 받게 돼 있어 시점마다 다른 배열을 바로는 못 받는다. 개별 종목에도
    같은 공식을 쓸지, 라벨링·리포트 어디에 연결할지는 아직 정해지지 않은, 이 함수
    바깥의 일이다.
    """
    daily_vol = historical_volatility(prices, window=window, ddof=ddof, annualize=False)
    return k * daily_vol * np.sqrt(horizon)
