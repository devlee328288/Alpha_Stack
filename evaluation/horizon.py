"""예측 지평 계산 — 기준선 · 손익분기 · 클래스 균형

**왜 이 모듈이 따로 있나.** 기준선을 두 곳에서 각자 계산하다가 실제로 갈라졌다.
`scripts/check_data.py`(당시 이름 `check_index_data.py`)는 KRX 가 반올림해 준 등락률로
세어 **52.72%** 를 인쇄했고,
지평 실측 스크립트는 원값으로 세어 **52.64%**(개발구간)를 냈다. 같은 것을 두 벌로 두면
언젠가 갈라지고, **갈라져도 에러는 안 난다.** 그래서 계산을 여기 한 벌로 모은다.

## 🔴 기준선은 KRX 원값(`change`)으로 센다

`change_rate` 는 KRX 가 **소수 2자리로 반올림**해 준 값이라, 실제로는 오르내린 날이
`0.00` 으로 찍혀 보합으로 빠진다. 전구간에서 15일이 그렇고 **그중 7일이 실제 상승일**이다.
"항상 상승" 은 그 7일에 실제로 돈을 번다.

반올림 기준을 쓰면 기준선이 0.17%p **낮아지는데**, 낮은 기준선은 **우리가 이기기 쉬워지는
방향**이다. 오차가 작아도 **부호가 한쪽으로 쏠리면 그건 잡음이 아니라 편향**이다.

## 이 모듈은 DB 를 모른다

행 목록(`List[Dict]`)을 받아 숫자만 돌려준다. 자료를 어디서 어떻게 읽는지는 부르는 쪽의
일이다. `evaluation/` 은 저장소를 직접 부르지 않는다는 경계
(`tests/test_supply_boundary.py`)가 여기에도 적용된다.

재현:
    python scripts/measure_horizon.py
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

#: 봉인 홀드아웃 시작일. 이 앞이 개발구간이다.
HOLDOUT_START = "20240901"

#: 3분류 중립 밴드. 예측 대상 ADR 이 못 박은 값이다.
NEUTRAL_BAND = 0.01

#: 왕복 거래비용 가정 (KOSPI200 ETF).
#: ⚠️ **실측이 아니라 가정값**이다. 문서에도 그렇게 적는다.
ROUND_TRIP_COST = 0.0005

#: 연간 거래일 수. 회전 비용을 연율로 환산할 때 쓴다.
TRADING_DAYS_PER_YEAR = 245

#: 3분류가 통과해야 하는 클래스 비율 범위. 벗어나면 임계값을 조정하고
#: **그 조정 1회를 시도 횟수 장부에 기록**해야 한다.
CLASS_BALANCE_RANGE = (0.15, 0.45)


def daily_baseline(rows: List[Dict]) -> Tuple[float, int, int, int]:
    """KRX 원값 `change` 로 센 "항상 상승" 기준선.

    반환: `(상승 비율, 상승, 하락, 보합)`

    ⚠️ `change_rate` 를 쓰지 않는다. 이 모듈 맨 위에 그 이유가 있다.
    """
    up = sum(1 for r in rows if r.get("change") is not None and r["change"] > 0)
    down = sum(1 for r in rows if r.get("change") is not None and r["change"] < 0)
    flat = len(rows) - up - down
    return (up / len(rows) if rows else float("nan")), up, down, flat


def significance_threshold(baseline: float, n: int, z: float = 1.645) -> float:
    """기준선을 단측 유의수준 5% 로 넘으려면 필요한 적중률.

    정규근사다. `n` 이 1,000 단위면 이항분포와 거의 같다.
    """
    if n <= 0:
        return float("nan")
    return baseline + z * math.sqrt(baseline * (1.0 - baseline) / n)


def overlap_vif(horizon: int) -> float:
    """중첩 레이블의 분산팽창계수 — 이론값.

    🔴 **왜 필요한가.** 매일 `horizon` 일 앞을 보는 레이블을 만들면 이웃한 관측이
       `horizon-1` 일을 공유한다. 적중 여부가 자기상관을 갖고, iid 를 가정한
       표준오차는 **너무 작게** 나온다. 즉 유의하지 않은 것을 유의하다고 말하게 된다.

    부호 지시자의 자기상관은 가우시안 가정에서 닫힌 형태로 나온다:

        rho_k = (2/pi) * arcsin(1 - k/horizon)      (k = 1 .. horizon-1)
        VIF   = 1 + 2 * sum(rho_k)

    `horizon=5` 면 rho = 0.590 / 0.410 / 0.262 / 0.128 이고 **VIF = 3.78** 이다.
    실효 표본이 1/3.78 로 줄어든다는 뜻이다.

    ⚠️ 이건 **이론 앵커**다. 실제 적중 계열은 예측의 지속성까지 얹히므로 보통 더 크다.
       실측이 가능하면 `empirical_vif` 를 쓰고, 이 값은 대조용으로만 본다.
    """
    if horizon <= 1:
        return 1.0
    rho = [(2.0 / math.pi) * math.asin(1.0 - k / horizon) for k in range(1, horizon)]
    return 1.0 + 2.0 * sum(rho)


def empirical_vif(hits: Sequence[float], horizon: int) -> float:
    """적중 계열에서 직접 잰 분산팽창계수.

    `hits` 는 0/1 적중 지시자다. `horizon-1` 시차까지의 자기상관을 더한다.

    ⚠️ 음의 자기상관이 나와 VIF 가 1 아래로 내려가면 1 로 자른다. 표준오차를
       **줄이는** 방향으로는 보정하지 않는다 — 우리에게 유리한 쪽으로 기우는 보정은
       하지 않는다는 뜻이다.
    """
    n = len(hits)
    if n < 2 or horizon <= 1:
        return 1.0
    mean = sum(hits) / n
    dev = [h - mean for h in hits]
    denom = sum(d * d for d in dev)
    if denom <= 0:
        return 1.0
    rho = [
        sum(dev[i] * dev[i + k] for i in range(n - k)) / denom
        for k in range(1, min(horizon, n))
    ]
    return max(1.0, 1.0 + 2.0 * sum(rho))


def significance_threshold_overlapping(
    baseline: float,
    n: int,
    horizon: int,
    z: float = 1.645,
    vif: float | None = None,
) -> float:
    """중첩 레이블을 감안한 유의 임계.

    `significance_threshold` 는 관측이 서로 독립이라고 본다. 매일 예측하는
    `horizon` 일 레이블에서는 그 가정이 깨지고 임계가 **너무 낮게** 나온다.

    우리 숫자로: 기준선 52.64% · N=1,217 · horizon=5 면
    iid 로는 **54.99%** 지만 중첩을 반영하면 **57.2%** 다. 2.2%p 나 느슨했다.

    ⚠️ ADR-AS-0004 §5 는 방향정확도를 **임계 없는 보조 지표**로 정했다. 이 함수는
       그 보조 지표를 정직하게 읽기 위한 것이지, 기각 판정에 쓰라는 것이 아니다.
       주 검정은 일간 초과수익 `d_t` 의 평균이고 **그쪽은 중첩 문제가 없다**
       (같은 날의 손익을 두 번 세지 않는다).
    """
    if n <= 0:
        return float("nan")
    factor = overlap_vif(horizon) if vif is None else vif
    se = math.sqrt(baseline * (1.0 - baseline) / n * factor)
    return baseline + z * se


def mean_abs(values: Sequence[float]) -> float:
    """기대 절대수익률. 손익분기 계산의 분모다."""
    return sum(abs(v) for v in values) / len(values) if values else float("nan")


def breakeven_accuracy(cost: float, expected_move: float) -> float:
    """손익분기 방향정확도 — `0.5 + 왕복비용 / (2 × E|수익|)`.

    ⚠️ *"방향 적중 여부와 수익 크기가 독립"* 이라는 **가정** 위에 있다. 큰 변동일이
       예측하기 어렵다면 실제 손익분기는 **더 높다.**
    """
    if expected_move <= 0:
        return float("nan")
    return 0.5 + cost / (2.0 * expected_move)


def annual_turnover_cost(hold_days: int, cost: float = ROUND_TRIP_COST) -> float:
    """`hold_days` 마다 갈아탈 때의 연간 회전비용."""
    if hold_days <= 0:
        return float("nan")
    return cost * TRADING_DAYS_PER_YEAR / hold_days


def returns_1d_close(rows: List[Dict]) -> List[float]:
    """1거래일 · 종가→종가.

    ⚠️ 이 형태는 **실행할 수 없다** — 종가를 보고 판단했으면 빨라야 다음 날 시가에
       들어간다. 비교용으로만 쓰고 체결 가정에는 쓰지 않는다.
    """
    return [rows[i + 1]["close"] / rows[i]["close"] - 1.0 for i in range(len(rows) - 1)]


def returns_1d_open(rows: List[Dict]) -> List[float]:
    """1거래일 · 시가(t+1)→시가(t+2). 우리 체결 규칙에 맞춘 형태다."""
    return [rows[i + 2]["open"] / rows[i + 1]["open"] - 1.0 for i in range(len(rows) - 2)]


def returns_5d_open(rows: List[Dict]) -> List[float]:
    """5거래일 · 시가(t+1)→시가(t+6). 예측 대상 ADR 이 못 박은 형태다.

    ⚠️ 시가(t)→시가(t+5) 로 잡으면 **t 일 장중 수익률이 라벨에 들어간다.** 조사에서
       그 오염이 상관을 **10배** 부풀리는 것을 확인했다(+0.1709 대 +0.0171).
       진입은 반드시 `t+1` 시가다.
    """
    return [rows[i + 6]["open"] / rows[i + 1]["open"] - 1.0 for i in range(len(rows) - 6)]


def trading_day_index(all_days: Sequence[str]) -> Dict[str, int]:
    """거래일 문자열을 0,1,2… 순번으로 바꾼 표.

    시장 전체의 거래일 달력이다. 종목별 행에는 **구멍이 있다**(거래정지·상장 전·
    상장폐지 후). 그래서 종목의 행 번호로 "5거래일 뒤"를 세면 안 되고,
    이 달력의 순번으로 세야 한다.
    """
    return {d: i for i, d in enumerate(sorted(set(all_days)))}


def returns_5d_open_gapless(
    rows: List[Dict],
    day_index: Dict[str, int],
    horizon: int = 5,
) -> List[float]:
    """종목용 5거래일 · 시가(t+1)→시가(t+6). **구멍을 건너뛰지 않는다.**

    🔴 **왜 `returns_5d_open` 을 그대로 쓰면 안 되나.** 그 함수는 행 번호로 센다.
       지수는 휴장일 말고 구멍이 없어서 맞지만, **종목은 거래정지가 있다.**
       3개월 정지된 종목이면 `rows[i+6]` 이 3개월 뒤 행이고, 그걸 "5일 수익률" 로
       세면 수익 크기가 통째로 부풀려진다. **에러는 나지 않는다.**

       실제로 이 저장소에는 중도 소멸 종목이 910개 있고(상장폐지·합병), 정지 구간을
       가진 종목은 그보다 많다.

    이 함수는 시장 거래일 달력(`day_index`)으로 거리를 재서, 진입일과 청산일이
    **정확히 `horizon` 거래일** 떨어진 경우만 남긴다.
    """
    out: List[float] = []
    for i in range(len(rows) - horizon - 1):
        entry, exit_ = rows[i + 1], rows[i + 1 + horizon]
        ei, xi = day_index.get(entry["bas_dd"]), day_index.get(exit_["bas_dd"])
        if ei is None or xi is None or xi - ei != horizon:
            continue
        if not entry.get("open"):
            continue
        out.append(exit_["open"] / entry["open"] - 1.0)
    return out


def classify_3(returns: Sequence[float], band: float = NEUTRAL_BAND) -> Dict[str, int]:
    """±`band` 3분류 분포."""
    dist = {"상승": 0, "중립": 0, "하락": 0}
    for r in returns:
        if r > band:
            dist["상승"] += 1
        elif r < -band:
            dist["하락"] += 1
        else:
            dist["중립"] += 1
    return dist


def class_balance_ok(dist: Dict[str, int]) -> bool:
    """세 클래스가 전부 15~45% 안에 드는가.

    벗어나면 임계값을 조정해야 하고, **그 조정 1회도 시도 횟수**다.
    """
    total = sum(dist.values())
    if not total:
        return False
    low, high = CLASS_BALANCE_RANGE
    return all(low <= n / total <= high for n in dist.values())


def split_dev(rows: List[Dict], holdout_start: str = HOLDOUT_START) -> List[Dict]:
    """개발구간만 남긴다.

    **기본값이 안전한 쪽이다** — 부르는 쪽이 아무것도 안 정하면 봉인 구간이 빠진다.
    홀드아웃을 보려면 명시적으로 `split_holdout` 을 불러야 한다.
    """
    return [r for r in rows if r["bas_dd"] < holdout_start]


def split_holdout(rows: List[Dict], holdout_start: str = HOLDOUT_START) -> List[Dict]:
    """봉인 홀드아웃 구간.

    🔴 **여기서 본 값으로 설계를 고치면 봉인이 하는 일이 없어진다.**
       무엇을 왜 봤는지 문서에 남기고 쓴다.
    """
    return [r for r in rows if r["bas_dd"] >= holdout_start]
