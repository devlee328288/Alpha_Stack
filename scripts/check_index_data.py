"""수집한 KOSPI200 지수의 **품질**을 잰다 — 정렬 · 결측 · OHLC 정합 · 거래일 공백.

    python scripts/check_index_data.py
    python scripts/check_index_data.py --index "코스닥 150"

## 기준선은 여기서 재지 않는다

예전에는 이 스크립트가 기준선도 함께 인쇄했다. 그런데 **KRX 가 소수 2자리로 반올림해
주는 등락률(`change_rate`)로 세는 바람에 52.72% 가 나왔고**, 원값(`change`)으로 세는
지평 측정과 **0.17%p 갈라졌다.**

반올림으로 `0.00` 이 되어 보합으로 빠진 날이 전구간 15일이고 **그중 7일이 실제 상승일**
이다. "항상 상승" 은 그 7일에 실제로 돈을 번다. 즉 반올림 기준은 기준선을 **낮추고**,
낮은 기준선은 **우리가 이기기 쉬워지는 방향**이다. 오차가 작아도 **부호가 한쪽으로
쏠리면 그건 잡음이 아니라 편향**이다.

같은 것을 두 곳에서 재면 언젠가 갈라지고, **갈라져도 에러는 안 난다.** 그래서 계산을
`evaluation/horizon.py` 한 벌로 모았다.

    python scripts/measure_horizon.py     # 기준선 · 손익분기 · 클래스 균형

⚠️ 여기서 재는 것은 **레이블 정의 이전의 자료 품질**이다. 학습에 쓸 레이블의 기준선은
   `evaluation/baseline.py` 가 폴드 안에서 따로 계산한다. 둘을 섞지 않는다.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingest.clients import krx_data as api  # noqa: E402
from ingest.store import krx_index as store  # noqa: E402


def _수집누락_찾기(rows) -> list:
    """데이터가 없는 평일 중 **아직 받아 보지 않은 날**만 골라낸다.

    `index_fetch_log` 에 `rows = 0` 으로 남은 날은 "받아 봤더니 없었다"(휴장)이므로
    누락이 아니다. 그 표에 아예 없는 날짜만 진짜 누락이다.
    """
    from datetime import date as _date
    from datetime import timedelta

    with store.connect() as conn:
        받아본날 = {r[0] for r in conn.execute("SELECT bas_dd FROM index_fetch_log")}

    있는날 = {r["date"].replace("-", "") for r in rows}
    누락 = []
    시작 = _date.fromisoformat(rows[0]["date"])
    끝 = _date.fromisoformat(rows[-1]["date"])
    day = 시작
    while day <= 끝:
        key = day.strftime("%Y%m%d")
        if day.weekday() < 5 and key not in 있는날 and key not in 받아본날:
            누락.append(key)
        day += timedelta(days=1)
    return 누락


def main() -> int:
    parser = argparse.ArgumentParser(description="지수 데이터 품질과 기준선을 잰다")
    parser.add_argument("--index", default=api.TARGET_INDEX, help="지수명 (기본: 코스피 200)")
    args = parser.parse_args()

    rows = store.series(args.index)
    if not rows:
        print(f"'{args.index}' 데이터가 없습니다. scripts/fetch_index.py 로 먼저 채우세요.")
        return 1

    print(f"═══ {args.index} ═══")
    print(f"구간   : {rows[0]['date']} ~ {rows[-1]['date']}  ({len(rows):,}거래일)")
    print(f"종가   : {rows[0]['close']:,.2f} → {rows[-1]['close']:,.2f}")
    print()

    # ── 1. 품질 게이트 ────────────────────────────────────────
    print("── 품질 ──")
    문제 = 0

    # (a) 날짜가 오름차순 유일값인가. 아니면 이동평균·차분이 전부 틀린다.
    dates = [r["date"] for r in rows]
    if dates != sorted(dates):
        print("  ❌ 날짜가 오름차순이 아니다")
        문제 += 1
    if len(set(dates)) != len(dates):
        print(f"  ❌ 중복 날짜 {len(dates) - len(set(dates))}건")
        문제 += 1

    # (b) 값이 비었거나 0 인 행. 지수는 0 이 될 수 없다.
    빈행 = [r for r in rows if not r["close"]]
    if 빈행:
        print(f"  ❌ 종가가 비었거나 0 인 행 {len(빈행)}건: {[r['date'] for r in 빈행[:5]]}")
        문제 += 1

    # (c) OHLC 정합성 — low ≤ open,close ≤ high 여야 한다.
    #     어긋나도 에러가 안 나고 변동성 피처만 조용히 이상해진다.
    깨진OHLC = [
        r["date"] for r in rows
        if None not in (r["open"], r["high"], r["low"], r["close"])
        and not (r["low"] <= min(r["open"], r["close"])
                 and max(r["open"], r["close"]) <= r["high"])
    ]
    if 깨진OHLC:
        print(f"  ⚠️ OHLC 정합성이 깨진 행 {len(깨진OHLC)}건: {깨진OHLC[:5]}")
        문제 += 1

    # (d) 등락률이 종가 변화와 맞는가 — FLUC_RT 를 믿어도 되는지 확인한다.
    #     이 프로젝트는 수익률을 종가 비율이 아니라 FLUC_RT 로 만든다(D-3).
    #     그 결정의 근거가 여기서 재현되어야 한다.
    어긋남 = []
    for 앞, 뒤 in zip(rows, rows[1:], strict=False):
        if not 앞["close"] or 뒤["change_rate"] is None:
            continue
        계산 = (뒤["close"] - 앞["close"]) / 앞["close"] * 100
        if abs(계산 - 뒤["change_rate"]) > 0.02:      # 반올림 여유
            어긋남.append((뒤["date"], round(계산, 3), 뒤["change_rate"]))
    if 어긋남:
        print(f"  ℹ️ 종가 역산과 FLUC_RT 가 다른 날 {len(어긋남)}건 (상위 3): {어긋남[:3]}")
        print("     → 지수 재산정·구성종목 변경 구간일 수 있다. FLUC_RT 를 정본으로 쓴다 (D-3)")

    # (e) 거래일 공백이 **확인된 휴장인가 수집 누락인가**.
    #
    # ⚠️ 날짜 간격만 보고 경고하면 정상 연휴가 전부 걸린다 — 2017-09-29→10-10 은
    #    추석 10일 연휴(10-02 임시공휴일)라 11일이 비는 것이 정상이다.
    #    그래서 `index_fetch_log` 와 대조한다. 그 표는 "받아 봤더니 0건" 을 기록하므로
    #    **확인된 휴장**과 **아직 안 받아 본 날**을 구분할 수 있다.
    #    이 구분이 없으면 진짜 누락이 연휴 경고에 묻힌다.
    공백 = _수집누락_찾기(rows)
    if 공백:
        print(f"  ❌ 수집 누락 의심 {len(공백)}건: {공백[:5]}")
        print("     → scripts/fetch_index.py 를 다시 돌리세요")
        문제 += 1
    else:
        print("  ✅ 거래일 공백이 전부 '받아서 확인한 휴장' 이다 (수집 누락 없음)")

    if not 문제:
        print("  ✅ 정렬·결측·OHLC 정합성 통과")
    print()

    print("기준선·손익분기·클래스 균형은 여기서 재지 않습니다 — "
          "python scripts/measure_horizon.py")

    return 0 if not 문제 else 1


if __name__ == "__main__":
    raise SystemExit(main())
