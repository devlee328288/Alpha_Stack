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
> 데이터파트 v2.0 변경사항에 적어 두었다. 지우지 않고 남긴다.

## 기준선은 KRX 원값(`change`)으로 센다

`change_rate` 는 KRX 가 **소수 2자리로 반올림**해 준 값이라, 실제로는 오르내린 날이
`0.00` 으로 찍혀 보합으로 빠진다. 전구간에서 15일이 그렇고 **그중 7일이 실제 상승일**이다.
"항상 상승" 은 그 7일에 실제로 돈을 번다. 반올림 기준을 쓰면 기준선이 낮아지는데,
**낮은 기준선은 우리가 이기기 쉬워지는 방향**이라 쓰면 안 된다.

실행:
    python scripts/measure_horizon.py            # 개발구간 (기본)
    python scripts/measure_horizon.py --peek     # 홀드아웃까지 (봉인을 연다)
"""

from __future__ import annotations

import argparse
import math
import sqlite3
from typing import Dict, List, Sequence, Tuple

from common.paths import krx_db_path

#: 예측 대상. 지수명은 KRX 가 주는 그대로다 (띄어쓰기 포함).
TARGET_INDEX = "코스피 200"

#: 봉인 홀드아웃 시작일. 이 앞이 개발구간이다.
HOLDOUT_START = "20210901"

#: 3분류 중립 밴드. ADR-AS-0002 가 못 박은 값이다.
NEUTRAL_BAND = 0.01

#: 왕복 거래비용 가정. ⚠️ **실측이 아니라 가정값**이다. 문서에 그렇게 적는다.
ROUND_TRIP_COST = 0.0005

#: 연간 거래일 수. 회전 비용을 연율로 환산할 때 쓴다.
TRADING_DAYS_PER_YEAR = 245


def load_rows(index_name: str = TARGET_INDEX) -> List[dict]:
    """시가·종가·전일대비를 날짜 오름차순으로 읽는다.

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


def significance_threshold(baseline: float, n: int, z: float = 1.645) -> float:
    """기준선을 단측 유의수준 5% 로 넘으려면 필요한 적중률. 정규근사."""
    if n <= 0:
        return float("nan")
    return baseline + z * math.sqrt(baseline * (1.0 - baseline) / n)


def mean_abs(values: Sequence[float]) -> float:
    return sum(abs(v) for v in values) / len(values) if values else float("nan")


def daily_baseline(rows: List[dict]) -> Tuple[float, int, int, int]:
    """KRX 원값 `change` 로 센 "항상 상승" 기준선.

    반환: (상승 비율, 상승, 하락, 보합)
    """
    up = sum(1 for r in rows if r["change"] is not None and r["change"] > 0)
    down = sum(1 for r in rows if r["change"] is not None and r["change"] < 0)
    flat = len(rows) - up - down
    return (up / len(rows) if rows else float("nan")), up, down, flat


def returns_1d_close(rows: List[dict]) -> List[float]:
    """1거래일 · 종가→종가. 팀원 제안이 말한 형태다."""
    return [rows[i + 1]["close"] / rows[i]["close"] - 1.0 for i in range(len(rows) - 1)]


def returns_1d_open(rows: List[dict]) -> List[float]:
    """1거래일 · 시가(t+1)→시가(t+2). 우리 체결 규칙에 맞춘 형태다.

    종가→종가와 갈라지는 이유: **종가에 사서 다음 종가에 파는 것은 실행할 수 없다.**
    종가를 보고 판단했으면 빨라야 다음 날 시가에 들어간다.
    """
    return [rows[i + 2]["open"] / rows[i + 1]["open"] - 1.0 for i in range(len(rows) - 2)]


def returns_5d_open(rows: List[dict]) -> List[float]:
    """5거래일 · 시가(t+1)→시가(t+6). ADR-AS-0002 가 못 박은 형태다.

    ⚠️ 시가(t)→시가(t+5) 로 잡으면 **t 일 장중 수익률이 라벨에 들어간다.** 조사에서
       그 오염이 상관을 10배 부풀리는 것을 확인했다. 진입은 반드시 t+1 시가다.
    """
    return [rows[i + 6]["open"] / rows[i + 1]["open"] - 1.0 for i in range(len(rows) - 6)]


def classify_3(returns: Sequence[float]) -> Dict[str, int]:
    dist = {"상승": 0, "중립": 0, "하락": 0}
    for r in returns:
        if r > NEUTRAL_BAND:
            dist["상승"] += 1
        elif r < -NEUTRAL_BAND:
            dist["하락"] += 1
        else:
            dist["중립"] += 1
    return dist


def report(rows: List[dict], label: str) -> None:
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
    for name, rets in (("종가→종가", returns_1d_close(rows)),
                       ("시가→시가", returns_1d_open(rows))):
        ea = mean_abs(rets)
        print(f"  {name}  표본 {len(rets):5,}  E|수익률| {ea:6.3%}  "
              f"비용/신호 {ROUND_TRIP_COST / ea:5.1%}")
    print()

    print("② 5거래일 지평 — ADR-AS-0002 (시가(t+1)→시가(t+6) · ±1% 3분류)")
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

    # ADR-AS-0002 가 5일 손익분기를 "증분이 독립" 이라는 √5 근사로 계산해 두고
    # **미측정**이라고 적어 두었다. 실제 5일 절대수익을 쟀으니 여기서 대조한다.
    ea1 = mean_abs(returns_1d_close(rows))
    approx5 = ea1 * math.sqrt(5)
    print("③ 손익분기 방향정확도 — 0.5 + 왕복비용 / (2 × E|수익|)")
    print(f"  √5 근사 5일 E|수익| {approx5:.3%}  vs  실측 {ea5:.3%}  "
          f"→ 근사가 {approx5 / ea5 - 1:+.1%} 어긋난다")
    print(f"  {'왕복비용':>8}  {'1일 지평':>9}  {'5일 지평':>9}")
    for cost in (0.0005, 0.0023, 0.0030, 0.0050):
        print(f"  {cost:>8.2%}  {0.5 + cost / (2 * ea1):>9.2%}  "
              f"{0.5 + cost / (2 * ea5):>9.2%}")
    print("  ※ 문헌상 방향정확도 상한은 55%대다. 1일 지평은 그걸 맞혀도 비용에 진다.")
    print()

    print("④ 3분류 클래스 균형 — ADR-AS-0002 은 세 클래스가 각 15~45% 를 요구한다")
    ok = all(0.15 <= dist[k] / n5 <= 0.45 for k in dist)
    verdict = "통과 — 임계값을 조정할 이유가 없다" if ok else "미달 — ±1.0% 를 조정해야 한다"
    print(f"  판정: {verdict}")
    print()

    print("⑤ 회전 비용을 연율로")
    for name, hold in (("1거래일마다 교체", 1), ("5거래일마다 교체", 5)):
        print(f"  {name}: 연 {ROUND_TRIP_COST * TRADING_DAYS_PER_YEAR / hold:.2%}")
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

    dev = [r for r in rows if r["bas_dd"] < HOLDOUT_START]
    report(dev, f"개발구간 (~{HOLDOUT_START} 이전)")

    if args.peek:
        print("⚠️ " + "─" * 68)
        print("⚠️ 봉인 홀드아웃을 엽니다. 여기서 본 값으로 설계를 고치면 봉인이")
        print("⚠️ 하는 일이 없어집니다. 무엇을 왜 봤는지 문서에 남기세요.")
        print("⚠️ " + "─" * 68)
        print()
        report([r for r in rows if r["bas_dd"] >= HOLDOUT_START], "봉인 홀드아웃")
    else:
        held = sum(1 for r in rows if r["bas_dd"] >= HOLDOUT_START)
        print(f"봉인 홀드아웃: {held:,}거래일 — 표본 수만 셉니다. "
              "분포를 보려면 --peek (권장하지 않음)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
