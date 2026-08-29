"""종목 트랙 실측 — 5일 수익 크기 · 손익분기 · 3분류 밴드 · 생존 편향

**왜 이 스크립트가 있나.** 계획서가 *"지수 방향 예측과 종목 랭킹을 둘 다 한다"* 로
범위를 넓혔다. 그런데 종목은 지수와 **거래비용이 다르다** — 지수는 ETF 로 왕복 0.05%
지만 개별주는 증권거래세만 0.23% 다. "비용이 5배면 못 이기는 것 아닌가" 라는 물음에
의견으로 답하면 합의가 안 된다. 같은 자료로 재서 숫자로 놓는다.

그리고 **3분류 밴드를 지수와 같은 ±1.0% 로 쓰면 안 된다.** 종목은 훨씬 크게 움직여서
같은 밴드를 쓰면 중립이 얇아지고 분포가 한쪽으로 쏠린다. 밴드도 여기서 고른다.

## 🔴 이 스크립트에는 `--peek` 이 없다

`measure_horizon.py` 에는 홀드아웃을 여는 `--peek` 이 있다. 여기에는 **일부러 넣지
않았다.** 종목 트랙의 홀드아웃 통계는 **아직 한 번도 보지 않았고**, 그 상태를 유지한다.
문을 만들어 두면 언젠가 열린다. 지수 쪽은 2026-08-27 에 실제로 그렇게 한 번 열렸다
(`docs/조사/예측지평.md` 에 기록).

## 🔴 종목에는 구멍이 있다

지수는 휴장일 말고 구멍이 없지만 **종목은 거래정지가 있다.** 행 번호로 "5거래일 뒤" 를
세면 3개월 정지된 종목의 정지 전후 가격 차이가 "5일 수익률" 로 둔갑한다. 그래서
시장 거래일 달력으로 거리를 재는 `returns_5d_open_gapless` 를 쓴다.
**이걸 틀려도 에러는 안 나고, 수익 크기가 부풀려져 손익분기가 낮게 나온다** —
즉 우리가 이기기 쉬워 보이는 방향으로 틀린다.

## 계산은 여기 없다

손익분기·클래스 분포 계산은 `evaluation/horizon.py` 한 벌이다. 이 파일은 자료를 읽어
그 함수에 넣고 인쇄만 한다.

실행:
    python scripts/measure_stock_horizon.py                  # KOSPI (기본)
    python scripts/measure_stock_horizon.py --market KOSDAQ
    python scripts/measure_stock_horizon.py --market ALL
    python scripts/measure_stock_horizon.py --bands 0.01 0.02 0.03
"""

from __future__ import annotations

import argparse
import sqlite3
from itertools import groupby
from statistics import median
from typing import Dict, Iterator, List, Tuple

from common.paths import krx_db_path
from evaluation.horizon import (
    HOLDOUT_START,
    breakeven_accuracy,
    class_balance_ok,
    classify_3,
    mean_abs,
    returns_5d_open_gapless,
    split_dev,
    trading_day_index,
)

#: 재 볼 왕복 거래비용. 출처는 `docs/시장조사/version2.0/시장조사.md` §3.
#: ⚠️ 0.40% 는 스프레드를 얹은 **가정값**이다. 실측이 아니다.
COSTS: List[Tuple[float, str]] = [
    (0.0005, "ETF 왕복 (거래세 면제)"),
    (0.0023, "개별주 거래세 2021~"),
    (0.0030, "개별주 + 수수료"),
    (0.0040, "개별주 + 스프레드 (가정)"),
]

#: 후보 밴드. 세 클래스가 전부 15~45% 에 드는 것을 고른다.
DEFAULT_BANDS: List[float] = [0.010, 0.015, 0.020, 0.025, 0.030]


def market_days(con: sqlite3.Connection, market: str) -> List[str]:
    """시장 전체 거래일. 종목 구멍을 판정할 달력이 된다."""
    if market == "ALL":
        sql, args = "SELECT DISTINCT bas_dd FROM daily_price", ()
    else:
        sql, args = "SELECT DISTINCT bas_dd FROM daily_price WHERE market = ?", (market,)
    return [r[0] for r in con.execute(sql, args)]


def iter_codes(con: sqlite3.Connection, market: str) -> Iterator[Tuple[str, List[Dict]]]:
    """종목 하나씩 흘려보낸다.

    ⚠️ 전부 메모리에 올리지 않는다. KOSPI 개발구간만 260만 행이고 KOSDAQ 까지
       합치면 두 배가 넘는다. 종목 단위로 처리하면 한 번에 수천 행만 든다.
    """
    where = "" if market == "ALL" else "WHERE market = ?"
    args = () if market == "ALL" else (market,)
    cur = con.execute(
        f"SELECT code, bas_dd, open, volume FROM daily_price {where} ORDER BY code, bas_dd",
        args,
    )
    for code, rows in groupby(cur, key=lambda r: r[0]):
        yield code, [
            {"bas_dd": d, "open": o, "volume": v}
            for _, d, o, v in rows
            if o and v  # 시가 0 · 거래량 0 은 체결을 가정할 수 없다
        ]


def survivorship(con: sqlite3.Connection) -> Tuple[int, int, Dict[str, int]]:
    """중도 소멸 종목 수. **전 구간** 기준이다 (자료의 성질이지 모델의 성질이 아니다).

    이 숫자가 0 이면 생존 편향이 들어와 있다는 뜻이다 — 지금 상장된 종목만으로
    과거를 조회했다는 말이기 때문이다.
    """
    rows = list(con.execute("SELECT code, MAX(bas_dd) FROM daily_price GROUP BY code"))
    last_day = max(d for _, d in rows)
    gone = [d for _, d in rows if d != last_day]
    by_year: Dict[str, int] = {}
    for d in gone:
        by_year[d[:4]] = by_year.get(d[:4], 0) + 1
    return len(rows), len(gone), by_year


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--market", default="KOSPI", choices=["KOSPI", "KOSDAQ", "ALL"])
    ap.add_argument("--bands", type=float, nargs="+", default=DEFAULT_BANDS)
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{krx_db_path()}?mode=ro", uri=True)

    print(f"── 종목 트랙 실측 — {args.market} · 개발구간(~{HOLDOUT_START} 직전) ──")
    print("   봉인 홀드아웃은 열지 않습니다. 이 스크립트에는 여는 방법이 없습니다.\n")

    # 달력도 개발구간으로 자른다. 홀드아웃 날짜가 달력에 남아 있으면 개발구간 마지막
    # 진입일의 청산일이 봉인 구간으로 넘어가고, 그 값이 조용히 섞인다.
    장날 = split_dev([{"bas_dd": d} for d in market_days(con, args.market)])
    달력 = trading_day_index([r["bas_dd"] for r in 장날])

    수익: List[float] = []
    종목수 = 0
    for _code, rows in iter_codes(con, args.market):
        개발 = split_dev(rows)
        if len(개발) < 7:
            continue
        종목수 += 1
        수익.extend(returns_5d_open_gapless(개발, 달력))

    if not 수익:
        print("🔴 표본이 없습니다.")
        return 1

    E = mean_abs(수익)
    print(f"표본(종목×날짜)  {len(수익):,}")
    print(f"종목 수          {종목수:,}")
    print(f"거래일 수        {len(달력):,}")
    print(f"E|5일수익|       {E * 100:.4f}%")
    print(f"중앙값|5일수익|  {median(abs(r) for r in 수익) * 100:.4f}%\n")

    print("손익분기 방향정확도  (0.5 + 왕복비용 / (2 × E|수익|))")
    for cost, label in COSTS:
        print(f"  {cost * 100:>5.2f}%  {label:<26} {breakeven_accuracy(cost, E) * 100:6.2f}%")
    print("  ⚠️ 이 공식은 '방향 적중과 수익 크기가 독립' 이라는 가정 위에 있습니다.")
    print("     큰 변동일이 예측하기 어렵다면 실제 손익분기는 더 높습니다.\n")

    print("3분류 밴드별 분포  (세 클래스가 전부 15~45% 여야 통과)")
    for band in args.bands:
        dist = classify_3(수익, band=band)
        n = sum(dist.values())
        ok = "✅" if class_balance_ok(dist) else "❌"
        print(f"  ±{band * 100:>4.1f}%  상승 {dist['상승'] / n * 100:5.2f}%  "
              f"중립 {dist['중립'] / n * 100:5.2f}%  하락 {dist['하락'] / n * 100:5.2f}%  {ok}")

    전체, 소멸, 연도별 = survivorship(con)
    print("\n생존 편향 방어  (전 구간 기준)")
    print(f"  전체 종목        {전체:,}")
    print(f"  중도 소멸        {소멸:,}  ← 이 값이 0 이면 생존 편향이 들어와 있습니다")
    print("  연도별 소멸      " + " ".join(f"{y}:{n}" for y, n in sorted(연도별.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
