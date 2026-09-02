"""한국은행 ECOS 거시지표 9종을 받아 `macro_series` 에 담는다.

    python scripts/fetch_macro.py                 # 9종 전부 (9콜)
    python scripts/fetch_macro.py --only cpi ppi  # 골라서
    python scripts/fetch_macro.py --years 5       # 짧게
    python scripts/fetch_macro.py --status        # 받지 않고 현황만

시세의 `fetch_krx.py`, 재무의 `fetch_dart.py` 와 같은 자리다.

## 한 번에 다 받는다

지표 하나가 1콜이고 그 1콜에 17년치가 통째로 온다. 전체가 9콜이라 —— 900만 행을
4,348콜로 받은 시세와 견주면 사실상 공짜다 —— 이어받기를 만들 이유가 없다.
**이미 받은 것도 다시 받는다.** 거시는 과거 값이 나중에 개정되기 때문이다.

## 🔴 담기는 것은 값과 **언제부터 알 수 있었나** 둘이다

ECOS 는 월별 값을 기준월 1일로 준다(7월 물가 → `2026-07-01`). 그 날짜를 그대로
쓰면 7월 물가를 7월 1일에 아는 셈인데 실제 발표는 8월 4일이었다. 그래서 지표마다
실제 공표일정에 여유를 더한 `known_at` 을 계산해 함께 담는다
(규칙과 근거: `ingest/clients/ecos_data.py` 의 `RELEASE_RULES`).

거시를 쓸 때는 `period` 가 아니라 **`known_at` 으로 거른다.**
`macro_store.as_of(지표, 날짜)` 가 그 일을 한다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest.clients import ecos_data  # noqa: E402
from ingest.store import macro_store  # noqa: E402


def 현황을_보여준다() -> int:
    """무엇이 얼마나 들어 있는지. 받지는 않는다."""
    상태 = macro_store.status()
    if not 상태:
        print("macro_series 가 비어 있다.")
        print("  할 일: python scripts/fetch_macro.py")
        return 1

    print(f"── 거시 지표 {len(상태)}종 ──")
    print(f"  {'지표':12s} {'주기':4s} {'행':>7s}  {'기간':21s} {'알게된 날':21s} 결측")
    print("  " + "-" * 74)
    합계 = 0
    for s in 상태:
        합계 += s["rows"]
        기간 = f"{s['period_min']}~{s['period_max']}"
        앎 = f"{s['known_at_min']}~{s['known_at_max']}"
        print(f"  {s['indicator_id']:12s} {s['cycle']:4s} {s['rows']:>7,}  "
              f"{기간:21s} {앎:21s} {s['null_values']:>4,}")
    print(f"\n  합계 {합계:,}행 · 지표 {len(상태)}종")

    빠진것 = set(macro_store.all_indicators()) - {s["indicator_id"] for s in 상태}
    if 빠진것:
        print(f"\n  🔴 아직 안 받은 지표: {' · '.join(sorted(빠진것))}")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ECOS 거시지표를 받아 macro_series 에 담는다")
    parser.add_argument("--only", nargs="+", metavar="지표",
                        help=f"고를 지표. 쓸 수 있는 것: "
                             f"{' · '.join(macro_store.all_indicators())}")
    parser.add_argument("--years", type=int, default=macro_store.DEFAULT_YEARS,
                        help=f"거슬러 올라갈 햇수 (기본 {macro_store.DEFAULT_YEARS})")
    parser.add_argument("--status", action="store_true",
                        help="받지 않고 현황만 보여준다")
    args = parser.parse_args()

    if args.status:
        return 현황을_보여준다()

    # 인증키가 없으면 무엇을 해야 하는지까지 알려 준다 — 막다른 길로 만들지 않는다.
    상태 = ecos_data.get_status()
    if not 상태["key_loaded"]:
        print("🔴 ECOS 인증키를 못 찾았다.")
        print("  할 일: 프로젝트 루트 `.env` 에 ECOS_API_KEY=발급받은키 한 줄을 넣는다.")
        print("  무료 발급: https://ecos.bok.or.kr/api/#/AuthKeyApply")
        return 1

    지표들 = args.only
    if 지표들:
        모르는것 = [i for i in 지표들 if i not in set(macro_store.all_indicators())]
        if 모르는것:
            print(f"🔴 모르는 지표: {' · '.join(모르는것)}")
            print(f"  쓸 수 있는 것: {' · '.join(macro_store.all_indicators())}")
            return 1

    셀_수 = len(지표들) if 지표들 else len(ecos_data.INDICATORS)
    print(f"ECOS 에서 {셀_수}종을 받는다 (지표당 1콜 · {args.years}년치)")

    def 진행(한줄):
        표시 = {"ok": "✅", "empty": "⬜", "error": "🔴"}.get(한줄["status"], "  ")
        꼬리 = f"  {한줄['note']}" if 한줄["status"] == "error" else ""
        print(f"  {표시} {한줄['id']:12s} {한줄['rows']:>6,}행{꼬리}")

    결과 = macro_store.sync(indicators=지표들, years=args.years, progress=진행)

    print()
    print(f"받음: {결과['ok']} ok · {결과['empty']} empty · {결과['error']} error "
          f"· {결과['rows']:,}행")
    if 결과["error"]:
        print("\n실패한 지표:")
        for 한줄 in 결과["details"]:
            if 한줄["status"] == "error":
                print(f"  🔴 {한줄['id']}: {한줄['note']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
