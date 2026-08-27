"""예측 지평 비교 실측 — 1거래일 이진 vs 5거래일 3분류

**왜 이 스크립트가 있나.** 팀원이 *"다음 거래일 상승/하락 이진 분류"* 를 제안했고,
우리 문서는 *"시가(t+1)→시가(t+6) · ±1% 3분류"* 로 닫혀 있다. 어느 쪽이 맞다고
**의견으로** 말하면 합의가 안 된다. 같은 자료로 양쪽을 재서 숫자로 놓는다.

**양쪽 다 공정하게 잰다.** 1거래일 지평에는 실제로 유리한 점이 있다 — 표본이 5배라
검정력이 높다. 그것도 함께 인쇄한다. 유리한 쪽만 재면 그건 측정이 아니라 변론이다.

## ⚠️ 기본 구간은 개발구간이다

봉인 홀드아웃(2021-09-01~)의 통계는 **기본으로 재지 않는다.** 레이블 분포나 기준선도
홀드아웃의 성질이고, 그걸 보고 설계를 고치면 봉인이 하는 일이 없어진다. 홀드아웃을
보려면 `--peek` 을 명시해야 하고, 그때 경고를 찍는다.

> 📌 2026-08-27 에 `--peek` 없이 홀드아웃을 한 번 쟀다. 그 사실과 본 값은
> `docs/조사/예측지평.md` 에 적어 두었다. 지우지 않고 남긴다.

## 계산은 여기 없다

기준선·손익분기·클래스 균형 계산은 **`evaluation/horizon.py` 한 벌**이다. 이 파일은
자료를 읽어 그 함수에 넣고 인쇄만 한다. 두 벌로 두면 갈라지고, **갈라져도 에러는 안 난다** —
실제로 이 저장소에서 기준선이 52.72% 와 52.64% 로 갈라진 적이 있다.

실행:
    python scripts/measure_horizon.py            # 개발구간 (기본)
    python scripts/measure_horizon.py --peek     # 홀드아웃까지 (봉인을 연다)
"""

from __future__ import annotations

import argparse
import math
import sqlite3
from typing import Dict, List

from common.paths import krx_db_path
from evaluation.horizon import (
    CLASS_BALANCE_RANGE,
    HOLDOUT_START,
    NEUTRAL_BAND,
    ROUND_TRIP_COST,
    annual_turnover_cost,
    breakeven_accuracy,
    class_balance_ok,
    classify_3,
    daily_baseline,
    mean_abs,
    returns_1d_close,
    returns_1d_open,
    returns_5d_open,
    significance_threshold,
    split_dev,
    split_holdout,
)

#: 예측 대상. 지수명은 KRX 가 주는 그대로다 (띄어쓰기 포함).
TARGET_INDEX = "코스피 200"


def load_rows(index_name: str = TARGET_INDEX) -> List[Dict]:
    """시가·종가·전일대비를 날짜 오름차순으로 읽는다.

    ⚠️ 이 함수는 **여기 있어야 한다.** `evaluation/` 로 옮기면 그 패키지가 저장소를
       직접 부르게 되고, 경계 테스트(`tests/test_supply_boundary.py`)가 그 자리에서
       깨진다. 깨지는 게 맞다 — 쓰는 계층은 `supply/` 를 지나야 한다.

    ⚠️ 시가가 없는 행은 버린다. 체결 시점이 시가라 시가가 없으면 그 날은 매매를
       가정할 수 없다. 0 으로 채우면 수익률이 조용히 -100% 가 된다.
    """
    conn = sqlite3.connect(krx_db_path())
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT bas_dd, open, close, change FROM index_price "
            "WHERE index_name = ? AND open IS NOT NULL AND close IS NOT NULL "
            "AND open > 0 AND close > 0 ORDER BY bas_dd",
            (index_name,),
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def report(rows: List[Dict], label: str) -> None:
    print(f"━━ {label} · {len(rows):,}거래일 "
          f"({rows[0]['bas_dd']} ~ {rows[-1]['bas_dd']}) ━━")
    print()

    base, up, down, flat = daily_baseline(rows)
    print("기준선 '항상 상승' (KRX 원값 change 기준)")
    print(f"  상승 {up:,} · 하락 {down:,} · 보합 {flat:,}  →  {base:.2%}")
    print(f"  유의 임계 (단측 5%, n={len(rows):,}): "
          f"{significance_threshold(base, len(rows)):.2%}")
    print()

    print("① 1거래일 지평 — 팀원 제안 (상승=1, 하락·보합=0)")
    ea1 = mean_abs(returns_1d_close(rows))
    for name, rets in (("종가→종가", returns_1d_close(rows)),
                       ("시가→시가", returns_1d_open(rows))):
        ea = mean_abs(rets)
        print(f"  {name}  표본 {len(rets):5,}  E|수익률| {ea:6.3%}  "
              f"비용/신호 {ROUND_TRIP_COST / ea:5.1%}")
    print()

    print(f"② 5거래일 지평 — 시가(t+1)→시가(t+6) · ±{NEUTRAL_BAND:.0%} 3분류")
    rets5 = returns_5d_open(rows)
    dist = classify_3(rets5)
    n5 = len(rets5)
    for name in ("상승", "중립", "하락"):
        print(f"  {name}  {dist[name]:5,}건  {dist[name] / n5:6.2%}")
    top = max(dist, key=lambda k: dist[k])
    base5 = dist[top] / n5
    ea5 = mean_abs(rets5)
    print(f"  최빈 '{top}' 를 항상 찍는 기준선 {base5:.2%}  "
          f"유의 임계 {significance_threshold(base5, n5):.2%}")
    print(f"  표본 {n5:,}  E|수익률| {ea5:.3%}  비용/신호 {ROUND_TRIP_COST / ea5:.1%}")
    print()

    # 예측 대상 ADR 이 5일 손익분기를 "증분이 독립" 이라는 √5 근사로 계산해 두고
    # **미측정**이라고 적어 두었다. 실제 5일 절대수익을 쟀으니 여기서 대조한다.
    approx5 = ea1 * math.sqrt(5)
    print("③ 손익분기 방향정확도 — 0.5 + 왕복비용 / (2 × E|수익|)")
    print(f"  √5 근사 5일 E|수익| {approx5:.3%}  vs  실측 {ea5:.3%}  "
          f"→ 근사가 {approx5 / ea5 - 1:+.1%} 어긋난다")
    print(f"  {'왕복비용':>8}  {'1일 지평':>9}  {'5일 지평':>9}")
    for cost in (0.0005, 0.0023, 0.0030, 0.0050):
        print(f"  {cost:>8.2%}  {breakeven_accuracy(cost, ea1):>9.2%}  "
              f"{breakeven_accuracy(cost, ea5):>9.2%}")
    print("  ※ 문헌상 방향정확도 상한은 55%대다. 1일 지평은 그걸 맞혀도 비용에 진다.")
    print()

    저, 고 = CLASS_BALANCE_RANGE
    print(f"④ 3분류 클래스 균형 — 세 클래스가 각 {저:.0%}~{고:.0%} 를 요구한다")
    판정 = ("통과 — 임계값을 조정할 이유가 없다" if class_balance_ok(dist)
            else f"미달 — ±{NEUTRAL_BAND:.0%} 를 조정해야 한다")
    print(f"  판정: {판정}")
    print()

    print("⑤ 회전 비용을 연율로")
    for name, hold in (("1거래일마다 교체", 1), ("5거래일마다 교체", 5)):
        print(f"  {name}: 연 {annual_turnover_cost(hold):.2%}")
    print()

    print("⑥ 1거래일 지평이 유리한 점 — 표본이 많아 검정력이 높다")
    n1 = len(returns_1d_open(rows))
    # 5거래일은 겹치는 표본이라 독립 표본 수가 약 1/5 로 줄어든다.
    n5_eff = max(n5 // 5, 1)
    print(f"  1거래일 {n1:,}  vs  5거래일 겹치지 않는 표본 약 {n5_eff:,}  "
          f"({n1 / n5_eff:.1f}배)")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="예측 지평을 실측으로 비교한다")
    parser.add_argument("--peek", action="store_true",
                        help="봉인 홀드아웃 구간까지 잰다 (기본은 개발구간만)")
    args = parser.parse_args()

    rows = load_rows()
    if not rows:
        print("지수 자료가 비어 있습니다 — python scripts/fetch_index.py 로 먼저 채우세요.")
        return 1

    report(split_dev(rows), f"개발구간 (~{HOLDOUT_START} 이전)")

    if args.peek:
        print("⚠️ " + "─" * 68)
        print("⚠️ 봉인 홀드아웃을 엽니다. 여기서 본 값으로 설계를 고치면 봉인이")
        print("⚠️ 하는 일이 없어집니다. 무엇을 왜 봤는지 문서에 남기세요.")
        print("⚠️ " + "─" * 68)
        print()
        report(split_holdout(rows), "봉인 홀드아웃")
    else:
        held = len(split_holdout(rows))
        print(f"봉인 홀드아웃: {held:,}거래일 — 표본 수만 셉니다. "
              "분포를 보려면 --peek (권장하지 않음)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
