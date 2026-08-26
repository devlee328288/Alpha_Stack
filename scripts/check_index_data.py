"""수집한 KOSPI200 지수의 품질을 재고, 기준선을 KRX 원자료로 측정한다.

**왜 이 스크립트가 필요한가.** [시장조사](../docs/시장조사.md)의 기준선 53.30% 는
Yahoo Finance `^KS200` 로 잰 값이라 KRX 원자료가 아니다. 문서가 스스로
"KRX 원자료로 재측정 필요" 라고 적어 두었다. 그 재측정을 여기서 한다.

또 하나 — 이 프로젝트는 "숫자는 실측만" 이 규약이다(AGENTS.md 5.4).
발표에 쓸 기준선을 우리 저장소에서 직접 뽑을 수 있어야 한다.

    python scripts/check_index_data.py
    python scripts/check_index_data.py --index "코스닥 150"

⚠️ 여기서 재는 것은 **레이블 정의 이전의 순수 기준선**이다. 실제 학습에 쓸 레이블은
   시가→시가 5거래일 ±1.0% 3분류이고(→ docs/문제정의.md §2), 그 기준선은
   `evaluation/baseline.py` 가 폴드 안에서 따로 계산한다. 둘을 섞지 않는다.
"""

import argparse
import statistics
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

    # ── 2. 기준선 (KRX 원자료 실측) ──────────────────────────
    print("── 기준선 (1거래일 방향) ──")
    변화 = [r["change_rate"] for r in rows if r["change_rate"] is not None]
    상승 = sum(1 for c in 변화 if c > 0)
    하락 = sum(1 for c in 변화 if c < 0)
    보합 = len(변화) - 상승 - 하락

    print(f"  표본        : {len(변화):,}일  (상승 {상승:,} · 하락 {하락:,} · 보합 {보합:,})")
    print(f"  항상 '상승' : {상승 / len(변화) * 100:.2f}%   ← 아무 모델 없이 나오는 값")

    # 전일 방향 지속 — 오늘 방향이 어제와 같은 비율
    지속 = sum(1 for a, b in zip(변화, 변화[1:], strict=False)
               if (a > 0) == (b > 0))
    print(f"  전일 방향 지속: {지속 / (len(변화) - 1) * 100:.2f}%")
    print("  무작위      : 50.00%")
    print()

    # ── 3. 손익분기 방향정확도 ───────────────────────────────
    print("── 손익분기 방향정확도 (매 지평마다 왕복 매매 가정) ──")
    평균절대 = statistics.fmean(abs(c) for c in 변화) / 100
    print(f"  E|일수익|   : {평균절대 * 100:.3f}%  (실측)")
    print()
    print(f"  {'왕복비용':>10} {'1일 지평':>10} {'5일 지평':>10}")
    print("  " + "-" * 32)
    # 5일 지평의 기대 절대수익은 √5 배로 커진다고 본다 (독립 증분 가정 — 근사다)
    평균절대_5일 = 평균절대 * (5 ** 0.5)
    for 비용 in (0.0005, 0.0023, 0.0030, 0.0050):
        b1 = 0.5 + 비용 / (2 * 평균절대)
        b5 = 0.5 + 비용 / (2 * 평균절대_5일)
        print(f"  {비용 * 100:>9.2f}% {b1 * 100:>9.2f}% {b5 * 100:>9.2f}%")
    print()
    print("  ⚠️ 5일 지평 계산은 '증분이 독립' 이라는 가정 위에 있다 (√5 근사).")
    print("     실제 5일 절대수익 분포로 다시 재야 한다 — 아직 미측정.")
    print()

    # ── 4. 표본 크기와 검출력 ────────────────────────────────
    print("── 이 표본으로 무엇을 말할 수 있나 ──")
    기준 = 상승 / len(변화)
    for n, 이름 in ((len(변화), "전구간"), (1230, "봉인 홀드아웃(예정)")):
        se = (0.25 / n) ** 0.5
        print(f"  {이름:<20} N={n:,}  1SE={se * 100:.2f}%p  "
              f"· 기준선 {기준 * 100:.2f}% 를 유의하게 이기려면 "
              f"{(기준 + 1.645 * se) * 100:.2f}% 필요")
    print()
    print(f"  📌 KRX Open API 원자료 실측 "
          f"({rows[0]['date']} ~ {rows[-1]['date']}).")
    print("     docs/시장조사.md 의 Yahoo Finance 기준값을 이 값으로 교체할 것.")

    return 0 if not 문제 else 1


if __name__ == "__main__":
    raise SystemExit(main())
