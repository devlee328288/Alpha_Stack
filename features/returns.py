"""수익률 계열 — 종가 기반 n일 수익률 (피처 엔지니어링 계층, 이슈 #38)

`features/` 계약(기능명세 §3.3)은 `indicators.py`와 같다 — 자세한 설명은 그쪽 모듈
docstring 참고, 여기서는 요약만 적는다.

    t 행은 t 시점까지의 자료만 쓴다 (look-ahead 금지)
    ------------------------------------------------
    `ret_t = close_t / close_{t-window} - 1` 은 `window`일 **전** 값과 비교하므로
    과거만 본다. `shift(-1)` 에 해당하는 어떤 연산도 하지 않는다.

⚠️ 왜 이 파일이 새로 생겼나 — 기존 22개 피처(이동평균·모멘텀·볼린저·변동성·거래량)는
전부 한 계열(KOSPI200 OHLCV)의 파생값인데, 그중 어느 것도 "최근 며칠 동안 얼마나
움직였나"를 직접 담지 않는다. 이동평균 차이는 **평활된** 변화라 급변이 뭉개지고,
`vol_roc_5`는 거래량 변화율이지 가격 수익률이 아니다. 실측(#38)으로 `ret_5`가
살아있는 신호(3분류 검정 p=4.3e-04)임을 확인한 뒤 추가했다.

⚠️ `indicators.py`·`volume.py`·`volatility.py`와 마찬가지로 numpy 로 직접 구현하고
(pandas 아님), `_to_array` 도 이 파일 안에 로컬로 둔다 — 다른 `features/` 모듈을
import 하지 않는다(모듈 간 결합을 최소로 둔다).

⚠️ 액면분할·무상증자 등 기업행위가 조정 안 된 종가를 넣으면 그 시점에서 거짓 폭락
(또는 폭등)이 그대로 수익률로 잡힌다 — 실측(2026-09-01)으로 확인했다: 삼성전자
2018-05-04 50:1 액면분할 구간에서 `ret_5` 가 -98% 로 나온다. 이건 이 함수의 버그가
아니라 **입력 데이터가 액면분할 조정이 안 된 것**이다. 이 함수는 받은 값을 정직하게
계산할 뿐이고, 조정은 데이터 계층(`supply`/`common/corporate_actions.py`)의 책임이다.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _to_array(values: Sequence) -> np.ndarray:
    """목록을 실수 배열로. `None` 은 `nan` 이 된다 (계산에서 자연히 전파된다)."""
    return np.asarray([np.nan if v is None else float(v) for v in values], dtype=float)


def n_day_return(prices: Sequence, window: int) -> np.ndarray:
    """`window` 일 전 대비 등락률.

        ret_t = close_t / close_{t-window} - 1

    앞쪽 `window` 자리는 그만큼 전 값이 없으므로 `nan` 이다. `close.shift(window)`
    처럼 과거만 보는 나눗셈이라 look-ahead 가 아니다 — 미래 종가를 요구하지 않는다.

    분모(과거 종가)가 0 이하이면(데이터 오류) `nan` 으로 막는다 — 나눗셈이 `inf` 를
    조용히 흘리지 않도록 `historical_volatility` 와 같은 방식으로 방어한다.
    """
    x = _to_array(prices)
    window = max(1, int(window))
    n = x.size
    out = np.full(n, np.nan)
    if n <= window:
        return out

    with np.errstate(divide="ignore", invalid="ignore"):
        base = np.where(x[:-window] > 0, x[:-window], np.nan)
        out[window:] = x[window:] / base - 1.0
    return out
