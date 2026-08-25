"""변환 — 로그수익률 · 차분 (시계열 엔진) — 명세서 §4.1 · 05-6 · 05-10

원계열(주가)을 **모델이 가정하는 모양**으로 바꾸는 일을 한다.

왜 주가를 그대로 모델에 넣으면 안 되는가
------------------------------------
ARIMA·AR 은 계열이 **정상(stationary)** 이라고 가정한다. 평균과 분산이 시간에 따라
변하지 않아야 한다는 뜻이다. 주가는 둘 다 어긴다 — 우상향하면 평균이 계속 올라가고,
가격이 10배가 되면 하루 변동폭도 대략 10배가 된다.

    로그수익률   ln(P_t / P_{t-1})   가격 수준과 무관한 **비율** 이 된다 (분산 안정)
    차분         y_t − y_{t-1}       추세를 걷어낸다 (평균 안정)

로그를 쓰는 이유가 하나 더 있다. **더할 수 있다.**
ln(P₂/P₀) = ln(P₂/P₁) + ln(P₁/P₀) 이라 여러 날 수익률이 그냥 합이 된다.
단순수익률은 곱해야 해서 h일 예측을 누적할 때마다 식이 지저분해진다.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


def to_array(values: Sequence) -> np.ndarray:
    """목록을 실수 배열로. `None` 은 `nan` 이 된다 (계산에서 자연히 전파된다)."""
    return np.asarray([np.nan if v is None else float(v) for v in values], dtype=float)


def log_returns(prices: Sequence) -> np.ndarray:
    """로그수익률 r_t = ln(P_t / P_{t-1}). 길이는 입력보다 **하나 짧다.**

    가격이 0 이하인 자리는 `nan` 으로 둔다. 로그를 취할 수 없는 값을 0 이나 직전 값으로
    슬쩍 바꾸면, 그 자리가 "변동이 없던 날" 로 둔갑해 변동성이 낮게 잡힌다.
    """
    px = to_array(prices)
    if px.size < 2:
        return np.empty(0, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        # 0 이하를 nan 으로 먼저 막고 나눈다 (음수 로그 경고를 만들지 않는다)
        safe = np.where(px > 0, px, np.nan)
        return np.log(safe[1:] / safe[:-1])


def simple_returns(prices: Sequence) -> np.ndarray:
    """단순수익률 (P_t − P_{t−1}) / P_{t−1}. 화면 표기용 — 모델에는 로그수익률을 쓴다."""
    px = to_array(prices)
    if px.size < 2:
        return np.empty(0, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        safe = np.where(px != 0, px, np.nan)
        return (px[1:] - safe[:-1]) / safe[:-1]


def difference(values: Sequence, d: int = 1) -> np.ndarray:
    """`d` 차 차분. 길이는 `d` 만큼 줄어든다.

    d=1 이면 y_t − y_{t−1}, d=2 면 그것을 한 번 더 한다.
    **실무에서 d 가 2를 넘는 일은 거의 없다** — 차분할수록 잡음이 증폭되기 때문이다.
    """
    x = to_array(values)
    for _ in range(max(0, int(d))):
        if x.size < 2:
            return np.empty(0, dtype=float)
        x = np.diff(x)
    return x


def integrate(diffs: Sequence, anchors: Sequence) -> np.ndarray:
    """`difference` 를 되돌린다. 예측값을 **원래 단위(가격)로 돌려놓을 때** 쓴다.

    차분한 채로 예측을 내보내면 독자가 "0.003" 을 보게 된다. 사람이 판단할 수 있는 단위는
    원(₩)이다. 그래서 되돌리는 일이 반드시 필요하다.

    `anchors` 의 계약 — **차수별 마지막 값**이지 "마지막 값 d개"가 아니다
    ------------------------------------------------------------------
    되돌릴 구간 **직전 시점 T** 에서, 차분 차수마다 하나씩 필요하다.

        anchors[0] = y_T            원계열의 마지막 값
        anchors[1] = Δy_T           1차 차분의 마지막 값  (= y_T − y_{T−1})
        anchors[2] = Δ²y_T          2차 차분의 마지막 값
        …                           d개 (인덱스 0 … d−1)

    흔한 실수는 `[y_{T−1}, y_T]` 처럼 **원계열 값 두 개**를 넣는 것이다. 그러면 d=2 에서
    결과가 통째로 어긋난다. 앵커는 `[y_T, y_T − y_{T−1}]` 이어야 한다.

        d=1:  ŷ_{T+h} = y_T + Σ Δ̂
        d=2:  Δ̂ 를 Δy_T 에서 누적해 만든 뒤, 그것을 y_T 에서 다시 누적한다

    편하게 쓰라고 `anchors_for` 를 함께 둔다 — 원계열을 주면 이 목록을 만들어 준다.
    """
    values = to_array(diffs)
    tail = to_array(anchors)
    d = tail.size

    # 안쪽(가장 높은 차수)부터 바깥(원계열)으로 한 겹씩 되돌린다.
    # level = d−1 → … → 0 이므로 `tail` 을 뒤에서부터 꺼내 쓴다.
    for level in range(d):
        values = tail[d - 1 - level] + np.cumsum(values)
    return values


def anchors_for(values: Sequence, d: int) -> List[float]:
    """`integrate` 에 넘길 앵커를 원계열에서 만들어 준다. → `[y_T, Δy_T, Δ²y_T, …]` (d개)

    d=0 이면 빈 목록이다 (되돌릴 것이 없다).
    """
    x = to_array(values)
    x = x[np.isfinite(x)]
    out: List[float] = []
    for _ in range(max(0, int(d))):
        if x.size == 0:
            break
        out.append(float(x[-1]))
        x = np.diff(x)
    return out


def summarize(values: Sequence) -> Dict:
    """계열의 기본 통계 — 화면 타일과 리포트 `limitation` 에 쓴다.

    표본표준편차(ddof=1)를 쓴다. 우리가 가진 것은 모집단이 아니라 표본이다.
    """
    x = to_array(values)
    clean = x[np.isfinite(x)]
    if clean.size == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None,
                "annualized_vol": None}

    std = float(np.std(clean, ddof=1)) if clean.size > 1 else 0.0
    return {
        "count": int(clean.size),
        "mean": float(np.mean(clean)),
        "std": std,
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
        # 일간 변동성 → 연율. 국내 증시 연간 거래일은 대략 252일이다.
        "annualized_vol": std * float(np.sqrt(252.0)),
    }


def moving_average(values: Sequence, window: int) -> List[Optional[float]]:
    """단순이동평균. 앞쪽 `window-1` 자리는 계산할 수 없으므로 `None` 으로 둔다.

    시나리오 트리거("20일선 상향 돌파")가 이 값을 본다.
    앞자리를 0 이나 첫 값으로 채우면 차트 왼쪽 끝에 없는 추세가 생긴다.
    """
    x = to_array(values)
    window = max(1, int(window))
    if x.size < window:
        return [None] * int(x.size)

    # 누적합의 차이로 한 번에 구한다 (반복문보다 빠르고 부동소수 오차도 작다)
    cumulative = np.concatenate(([0.0], np.nancumsum(x)))
    sums = cumulative[window:] - cumulative[:-window]
    averages = sums / window
    return [None] * (window - 1) + [float(v) if np.isfinite(v) else None for v in averages]
