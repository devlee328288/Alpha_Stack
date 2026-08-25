"""성과 지표 — MDD · Sharpe · 승률 · 거래비용 차감 수익.

이 모듈은 **무엇이 그 수익률을 만들었는지 모른다.** 포지션 배열과 자산 수익률 배열만
받는다. 그래서 1차의 LightGBM 이든 2차의 자산배분이든 3차의 커스텀 지표든 같은 함수가
돈다 (→ `evaluation/__init__.py`).

    positions   각 시점의 포지션. +1 매수 · 0 현금 · -1 매도(공매도)
    returns     각 시점 **자산 자체의** 수익률. 전략 수익률이 아니다

⚠️ 두 배열의 시점을 어긋나게 두지 않는다
--------------------------------------
`positions[t]` 는 **`returns[t]` 를 벌기 전에 이미 잡고 있던** 포지션이어야 한다.
t 시점 수익률을 보고 t 시점 포지션을 정하면 미래를 본 것이다(look-ahead). 모델
예측으로 포지션을 만들 때는 반드시 한 칸 밀어 둔다.

    signal   = model.predict(features)   # t 일 종가까지 보고 낸 신호
    position = np.roll(signal, 1)        # t+1 일 수익률에 적용
    position[0] = 0                      # 첫날은 잡을 포지션이 없다

이 한 줄을 빠뜨리면 정확도 60% 짜리 모델이 연 300% 를 벌어 준다. 그런 결과가 나오면
기뻐하기 전에 이 정렬부터 의심한다.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

# ── 기본 가정값 ────────────────────────────────────────────────────────────
# 왕복 거래비용. 증권거래세 + 위탁수수료 + 슬리피지를 뭉뚱그린 보수적 가정치다.
#
# ⚠️ **이 숫자는 실측이 아니라 가정이다.** 실제 세율·수수료는 상품(주식·ETF·선물)과
#    증권사에 따라 다르고 해마다 바뀐다. 팀이 확인해서 확정하기 전까지 이 값으로
#    두되, 보고서에는 반드시 "왕복 0.3% 가정" 이라고 함께 적는다.
#    → docs/회의안건.md 의 미결 항목
DEFAULT_ROUND_TRIP_COST = 0.003

# 연율화 계수. 한국 주식시장의 연간 거래일 수 근사치다.
TRADING_DAYS_PER_YEAR = 252


def equity_curve(returns: Sequence[float], initial: float = 1.0) -> np.ndarray:
    """수익률 배열을 누적 자산 곡선으로 바꾼다.

    단순 합이 아니라 곱으로 쌓는다. -50% 뒤 +50% 는 본전이 아니라 -25% 이기 때문이다.
    이 차이가 MDD 계산에서 그대로 드러난다.
    """
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return np.array([], dtype=float)
    return initial * np.cumprod(1.0 + r)


def max_drawdown(equity: Sequence[float]) -> float:
    """최대 낙폭(MDD) — 고점 대비 가장 크게 빠진 비율. **음수로 돌려준다.**

    "얼마를 벌었나"가 아니라 **"버틸 수 있었나"** 를 재는 숫자다. 연 30% 를 벌어도
    중간에 -60% 를 겪었다면 대부분의 사람은 그 전에 손을 뗀다. 그래서 수익률보다
    이쪽을 먼저 본다.

    부호에 주의한다. -0.35 는 35% 하락이고, 0.0 은 한 번도 고점을 밑돌지 않았다는 뜻이다.
    """
    e = np.asarray(equity, dtype=float)
    if e.size == 0:
        return 0.0
    running_peak = np.maximum.accumulate(e)
    # 고점이 0 이하면 비율이 뜻을 잃는다. 그런 구간은 낙폭 0 으로 둔다.
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdown = np.where(running_peak > 0, e / running_peak - 1.0, 0.0)
    return float(np.min(drawdown))


def sharpe_ratio(returns: Sequence[float], risk_free_rate: float = 0.0,
                 periods_per_year: int = TRADING_DAYS_PER_YEAR) -> Optional[float]:
    """샤프 지수 — 위험 한 단위당 초과수익. 연율화해서 돌려준다.

    `risk_free_rate` 는 **연율**로 받아 내부에서 기간 단위로 나눈다.

    ⚠️ 변동성이 0 이면 `None` 을 돌려준다 (0.0 이 아니다).
       포지션을 한 번도 잡지 않아 수익률이 전부 0 인 경우가 그렇다. 이때 분모가 0 이라
       샤프는 정의되지 않는다. 여기서 0.0 을 돌려주면 "위험 대비 수익이 없다"로 읽혀
       "잴 수 없다"와 구별되지 않는다. 보고서에서 그 둘은 전혀 다른 말이다.

    ⚠️ 표본이 짧으면 이 숫자는 매우 불안정하다. 30개 남짓한 수익률로 낸 샤프를
       연율화하면 그럴듯한 크기의 무의미한 값이 나온다. 폴드별 분산을 함께 본다.
    """
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return None
    excess = r - risk_free_rate / periods_per_year
    # 표본표준편차(ddof=1). 모표준편차를 쓰면 짧은 표본에서 샤프가 과대평가된다.
    sd = float(np.std(excess, ddof=1))
    if sd == 0.0 or not np.isfinite(sd):
        return None
    return float(np.mean(excess) / sd * np.sqrt(periods_per_year))


def hit_rate(predicted: Sequence[int], actual: Sequence[int]) -> Optional[float]:
    """방향 적중률 — 예측한 등락 방향이 맞은 비율.

    ⚠️ **이 숫자 하나만 보고하지 않는다.** 상승이 55% 나오는 구간에서 언제나 "오른다"고
       답해도 적중률 55% 가 나온다. 그래서 `baseline.always_up` 과 나란히 두고 본다.
       이기지 못하면 그 사실이 결과의 일부다 (→ `evaluation/baseline.py`).

    방향이 0(보합)인 시점은 맞고 틀림을 가릴 수 없어 분모에서 뺀다.
    """
    p = np.asarray(predicted)
    a = np.asarray(actual)
    if p.size != a.size:
        raise ValueError(f"길이가 다르다: predicted={p.size} actual={a.size}")
    mask = a != 0
    if not np.any(mask):
        return None
    return float(np.mean(p[mask] == a[mask]))


def apply_cost(positions: Sequence[float], returns: Sequence[float],
               round_trip_cost: float = DEFAULT_ROUND_TRIP_COST) -> np.ndarray:
    """포지션·자산수익률 → **거래비용을 뺀** 전략 수익률.

    비용은 포지션이 **바뀔 때만** 든다. 계속 들고 있으면 들지 않는다.

        turnover_t = |position_t - position_{t-1}|      한 시점의 거래량(편도 기준)
        cost_t     = turnover_t × round_trip_cost / 2   편도는 왕복의 절반

    0 → +1 → 0 (진입 후 청산)이면 turnover 합이 2 이므로 왕복 비용이 정확히 한 번
    부과된다. +1 → -1 (매수에서 공매도로 뒤집기)은 turnover 2 라 왕복 한 번 값이 든다.

    ⚠️ 첫 시점의 진입 비용을 빠뜨리지 않는다. `positions[0]` 이 0 이 아니면 아무것도
       없던 상태에서 그 포지션을 잡은 것이므로 비용이 든다. 아래에서 앞에 0 을 덧대는
       이유가 이것이다 — 이걸 빠뜨리면 매 폴드마다 진입 비용이 공짜가 된다.
    """
    pos = np.asarray(positions, dtype=float)
    r = np.asarray(returns, dtype=float)
    if pos.size != r.size:
        raise ValueError(f"길이가 다르다: positions={pos.size} returns={r.size}")
    if pos.size == 0:
        return np.array([], dtype=float)

    # 직전 포지션. 맨 앞은 "아무것도 안 들고 있던 상태"이므로 0 이다.
    previous = np.concatenate(([0.0], pos[:-1]))
    turnover = np.abs(pos - previous)
    cost = turnover * (round_trip_cost / 2.0)
    return pos * r - cost


def summarize(positions: Sequence[float], returns: Sequence[float],
              round_trip_cost: float = DEFAULT_ROUND_TRIP_COST,
              risk_free_rate: float = 0.0,
              periods_per_year: int = TRADING_DAYS_PER_YEAR) -> Dict[str, object]:
    """한 전략의 성과를 한 번에 잰다. 보고서에 그대로 실을 수 있는 형태로 돌려준다.

    비용 **전**과 **후** 를 둘 다 담는다. 둘의 차이가 "이 전략이 얼마나 자주 손을
    대는가"를 말해 준다. 차이가 크면 신호가 과하게 흔들린다는 뜻이고, 그건 모델을
    고칠 이유가 된다.
    """
    pos = np.asarray(positions, dtype=float)
    r = np.asarray(returns, dtype=float)

    gross = pos * r                                          # 비용 전
    net = apply_cost(pos, r, round_trip_cost)                # 비용 후
    net_equity = equity_curve(net)

    previous = np.concatenate(([0.0], pos[:-1])) if pos.size else pos
    turnover_total = float(np.sum(np.abs(pos - previous))) if pos.size else 0.0

    return {
        "n_periods": int(r.size),
        "total_return_gross": float(np.prod(1.0 + gross) - 1.0) if gross.size else 0.0,
        "total_return_net": float(np.prod(1.0 + net) - 1.0) if net.size else 0.0,
        "max_drawdown": max_drawdown(net_equity),
        "sharpe": sharpe_ratio(net, risk_free_rate, periods_per_year),
        # 이긴 날의 비율. 승률이 높아도 한 번의 큰 손실이 전부를 지울 수 있어
        # MDD 와 반드시 함께 읽는다.
        "win_rate": float(np.mean(net > 0)) if net.size else None,
        "turnover": turnover_total,
        "cost_drag": (float(np.prod(1.0 + gross) - np.prod(1.0 + net))
                      if gross.size else 0.0),
        "assumptions": {
            "round_trip_cost": round_trip_cost,
            "risk_free_rate": risk_free_rate,
            "periods_per_year": periods_per_year,
            # ⚠️ 보고할 때 이 항목을 함께 싣는다. 비용 가정을 감춘 수익률은
            #    비교할 수 없는 숫자다.
            "note": "거래비용은 실측이 아니라 가정이다 (docs/회의안건.md 참고)",
        },
    }
