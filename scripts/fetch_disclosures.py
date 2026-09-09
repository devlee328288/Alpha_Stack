"""DART 공시 목록을 받아 `dart_disclosure` 에 담는다.

    python scripts/fetch_disclosures.py --status         # 받지 않고 현황만
    python scripts/fetch_disclosures.py --plan           # 몇 콜 쓸지만 센다
    python scripts/fetch_disclosures.py --limit 3        # 세 달만 (시험 삼아)
    python scripts/fetch_disclosures.py                  # 전 구간 이어받기
    python scripts/fetch_disclosures.py --start 202401   # 이 달부터

시세의 `fetch_krx.py`, 기본정보의 `fetch_base_info.py` 와 같은 자리다.

## 🔴 왜 지금 이것이 먼저인가

`dart_disclosure` 가 **0행**이라 텍스트 신호를 만들 재료가 없다. 신장환님이 그 신호의
as-of 조인을 맡기로 했는데 붙일 것이 없어 착수를 못 하고 있다. 모델 표보다 **원료
표가 먼저**다.

## 어떻게 받나 — 종목이 아니라 날짜로

`list.json` 은 `corp_code` 를 안 줘도 되고, 그러면 그 기간 공시가 전부 온다. 종목별로
받으면 보통주 3,482종 × 창 여러 개인데, 날짜로 받으면 **공시 건수만큼**이면 된다.

호출 수는 **페이지 수**로 정해진다. 한 페이지가 100건이고 그게 상한이다(200·500 을
줘도 100만 온다 — 실측). 2024-08 유가가 4,550건이라 그 달만 46페이지다.

## 창은 달로 끊는다

한 해를 통째로 물으면 `total_count` 가 `None` 으로 온다. 오류도 아니고 빈 결과도 아닌
**모양이 다른 응답**이라 그대로 두면 페이지를 못 넘긴다. 한 달은 정상이다.

## 이어받기

달을 **통째로 마친 뒤에만** `collect_log` 에 `ok` 를 적는다. 페이지 도중에 멈춘 달을
`ok` 로 적으면 그 달의 나머지가 영영 안 온다. 예산이 떨어지면 그 달은 안 적고 멈춘다 —
다음 실행이 처음부터 다시 받고, 접수번호가 기본키라 중복은 안 쌓인다.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import budget  # noqa: E402
from ingest.clients.dart_data import load_dart_key  # noqa: E402
from ingest.store import collect_log, disclosure_store  # noqa: E402

#: 우리 시세가 시작하는 달. 그 앞은 붙일 시세가 없다.
DEFAULT_START = "201001"


def 현황() -> None:
    st = disclosure_store.status()
    print("── dart_disclosure ──")
    if not st["rows"]:
        print("  (비어 있음) — 텍스트 신호를 만들 재료가 아직 없다")
        return
    print(f"  {st['rows']:,}행 · 법인 {st['corps']:,}곳 · 종목코드 있는 것 "
          f"{st['stocks']:,}종 ({st['first']} ~ {st['last']})")
    이름 = {"Y": "유가", "K": "코스닥", "N": "코넥스", "E": "기타"}
    for cls, n in sorted(st["by_class"].items(), key=lambda x: -x[1]):
        print(f"    {이름.get(cls, cls):<6} {n:>9,}")


def main() -> int:
    ap = argparse.ArgumentParser(description="DART 공시 목록 수집")
    ap.add_argument("--status", action="store_true", help="받지 않고 현황만")
    ap.add_argument("--plan", action="store_true", help="몇 달 남았는지만 센다")
    ap.add_argument("--limit", type=int, default=0, help="이 개수의 달만")
    ap.add_argument("--start", default=DEFAULT_START, help="시작 달 YYYYMM")
    ap.add_argument("--end", default="", help="끝 달 YYYYMM (기본: 오늘)")
    args = ap.parse_args()

    if args.status:
        현황()
        return 0

    end = args.end or time.strftime("%Y%m")
    남은달 = disclosure_store.pending_months(args.start, end)
    if args.limit:
        남은달 = 남은달[:args.limit]

    쓴만큼 = budget.usage(disclosure_store.SOURCE).get(disclosure_store.SOURCE, {})
    print(f"── 계획 ── {args.start} ~ {end}")
    print(f"  안 받은 (시장,달) {len(남은달):,}쌍")
    print(f"  오늘 {disclosure_store.SOURCE} 예산: "
          f"{쓴만큼.get('spent', 0):,} / {쓴만큼.get('limit', 0):,}")
    print("  ⚠️ 호출 수는 달 수가 아니라 **페이지 수**로 정해진다 — 한 달이 수십 콜일 수 있다")
    if args.plan or not 남은달:
        if not 남은달:
            print("\n  받을 것이 없다.")
            현황()
        return 0

    key, 출처 = load_dart_key()
    print(f"  DART 키 출처: {출처}\n")

    시작 = time.time()
    담은행, ok, empty, err = 0, 0, 0, 0
    for i, (market, ym) in enumerate(남은달, 1):
        대상 = disclosure_store.target_of(market, ym)
        try:
            rows = disclosure_store.fetch_month(market, ym, key=key)
        except disclosure_store.BudgetExhausted as exc:
            # 고장이 아니라 하루의 끝이다. 이 달은 **적지 않는다** — 페이지 도중에
            # 끊겼을 수 있고, ok 로 적으면 나머지가 영영 안 온다.
            print(f"\n{exc}")
            print(f"  ({i - 1:,}/{len(남은달):,} 처리) 내일 같은 명령으로 이어 받는다.")
            break
        except disclosure_store.DisclosureError as exc:
            err += 1
            collect_log.mark_error(disclosure_store.SOURCE, 대상, note=str(exc)[:200])
            print(f"  ⚠️ {market} {ym}: {exc}")
            continue

        새로 = disclosure_store.save(rows)
        담은행 += 새로
        if rows:
            ok += 1
            collect_log.mark_ok(disclosure_store.SOURCE, 대상, rows=len(rows))
        else:
            empty += 1
            collect_log.mark_empty(disclosure_store.SOURCE, 대상,
                                   note="그 달에 공시가 없다")
        if i % 10 == 0 or i == len(남은달):
            걸린 = time.time() - 시작
            남은 = (걸린 / i) * (len(남은달) - i)
            print(f"  [{i:>4,}/{len(남은달):,}] {market} {ym} · "
                  f"받은 {len(rows):>5,} · 새로 담은 누계 {담은행:>8,} · "
                  f"{걸린:>5.0f}초 경과 · 남은 시간 어림 {남은 / 60:.0f}분")

    print(f"\n── 끝 ── {time.time() - 시작:.0f}초 · 새로 담은 행 {담은행:,} "
          f"(ok {ok:,} · empty {empty:,} · error {err:,})")
    현황()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
