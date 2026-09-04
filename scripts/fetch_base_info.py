"""KRX 종목기본정보를 받아 `stock_base_info` 에 담는다.

    python scripts/fetch_base_info.py --status          # 받지 않고 현황만
    python scripts/fetch_base_info.py --plan            # 몇 콜 쓸지만 센다
    python scripts/fetch_base_info.py --limit 10        # 10쌍만 (시험 삼아)
    python scripts/fetch_base_info.py                   # 전 구간 이어받기
    python scripts/fetch_base_info.py --start 20250901  # 이 날짜부터

시세의 `fetch_krx.py`, 지수의 `fetch_index.py`, 신원의 `fetch_data_go_kr.py` 와 같은 자리다.

## 무엇을 받나 — **보통주인지 우선주인지의 정본**

지금 우리는 종목명이 '우' 로 끝나는지로 우선주를 추측하고 있고, 그게 보통주 7종을
우선주로 잘못 뺀다(미래에셋대우 · 연우 · 동우 · 신우 · 성우 · 에코글로우 · 이오플로우).
006800 은 20200102 코스피 시총 **48위**라 모델 파트의 "보통주 시총 상위 50" 후보에서
조용히 빠진다. 자세한 실측은 `ingest/store/base_info_store.py` 문서에 있다.

## 🔴 얼마나 걸리나 — 먼저 센다

KRX 개발계정은 하루 10,000회다. 실측 기준:

    거래일 4,102일 × 2시장(KOSPI·KOSDAQ) = 8,204콜
    호출 1건 평균 0.44초  →  약 60분

KONEX 는 받지 않는다 — `daily_price` 에 한 행도 없어 붙을 데가 없다.
한도 안이지만 여유가 크지 않으므로 `--plan` 으로 **먼저 세고** 시작한다.

## 이어받기

`collect_log` 에 대상(`base_info:<시장>:<기준일>`)마다 한 줄을 남긴다. 이미 `ok` 든
`empty` 든 남은 날은 건너뛴다 — `empty` 도 '받아 봤다' 이다. 한도에 닿으면
**멈추고 정상 종료한다**(고장이 아니라 하루의 끝이다). 다음 날 그대로 다시 부르면
남은 것부터 이어 받는다.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import budget  # noqa: E402
from ingest.clients import krx_data  # noqa: E402
from ingest.store import base_info_store  # noqa: E402


def 받을_쌍들(*, limit: int = 0, start: str = "", end: str = "",
            markets=base_info_store.MARKETS) -> list:
    """받을 (기준일, 시장) 목록. 달력에서 뽑고 이미 받은 것은 뺀다."""
    쌍들 = base_info_store.pending_days(markets)
    if start:
        쌍들 = [p for p in 쌍들 if p[0] >= start]
    if end:
        쌍들 = [p for p in 쌍들 if p[0] <= end]
    return 쌍들[:limit] if limit else 쌍들


def 현황() -> int:
    base_info_store.ensure_schema()
    s = base_info_store.status()
    print("── stock_base_info ──")
    if not s["rows"]:
        print("  비어 있다. `python scripts/fetch_base_info.py` 로 받는다.")
        return 0
    print(f"  {s['rows']:,}행 · {s['days']:,}일 · {s['codes']:,}종 "
          f"({s['first']} ~ {s['last']})")
    print("  주권종류별 고유 종목 수:")
    for k, v in sorted(s["kinds"].items(), key=lambda kv: -kv[1]):
        표시 = k or "(빈 값)"
        보통 = " ← 유니버스" if krx_data.is_common_stock(k) else ""
        print(f"    {표시:<10} {v:>6,}{보통}")
    남음 = len(받을_쌍들())
    print(f"  아직 안 받은 (날짜,시장) 쌍: {남음:,}")
    return 0


def 계획(쌍들: list) -> int:
    print("── 계획 ──")
    print(f"  받을 (날짜,시장) 쌍 : {len(쌍들):,}")
    print(f"  = 호출 수            : {len(쌍들):,}")
    print(f"  예상 소요            : 약 {len(쌍들) * 0.44 / 60:.0f}분 "
          f"(1건 0.44초 · 실측 2026-09-03)")
    쓴만큼 = budget.usage(base_info_store.SOURCE).get(base_info_store.SOURCE, {})
    남은예산 = (쓴만큼.get("limit") or 0) - (쓴만큼.get("used") or 0)
    print(f"  오늘 남은 KRX 예산   : {남은예산:,} "
          f"(한도 {쓴만큼.get('limit', 0):,} · 쓴 것 {쓴만큼.get('used', 0):,})")
    if len(쌍들) > 남은예산:
        print(f"  ⚠️ 예산이 {len(쌍들) - 남은예산:,}콜 모자란다 — 오늘 받을 수 있는 데까지"
              f" 받고 멈춘다. 내일 같은 명령을 다시 부르면 이어 받는다.")
    if 쌍들:
        print(f"  처음: {쌍들[0][0]} {쌍들[0][1]}   마지막: {쌍들[-1][0]} {쌍들[-1][1]}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="KRX 종목기본정보를 받아 stock_base_info 에 담는다")
    parser.add_argument("--status", action="store_true", help="받지 않고 현황만")
    parser.add_argument("--plan", action="store_true", help="받지 않고 콜 수만 센다")
    parser.add_argument("--limit", type=int, default=0, help="이만큼만 (0 이면 전부)")
    parser.add_argument("--start", default="", help="이 날짜부터 (YYYYMMDD)")
    parser.add_argument("--end", default="", help="이 날짜까지 (YYYYMMDD)")
    parser.add_argument("--every", type=int, default=200,
                        help="진행 상황을 몇 건마다 찍을지")
    args = parser.parse_args()

    if args.status:
        return 현황()

    base_info_store.ensure_schema()
    쌍들 = 받을_쌍들(limit=args.limit, start=args.start, end=args.end)

    if args.plan:
        return 계획(쌍들)

    if not 쌍들:
        print("받을 것이 없다 — 전부 받았다.")
        return 0

    계획(쌍들)
    print()
    시작 = time.time()
    담은행, ok, empty, err, 미룸 = 0, 0, 0, 0, []

    for i, (bas_dd, market) in enumerate(쌍들, 1):
        try:
            r = base_info_store.sync_day(bas_dd, market)
        except krx_data.KrxQuotaExhausted:
            # 🔴 고장이 아니라 하루의 끝이다. 여기까지가 오늘 몫이고 정상 종료한다.
            print(f"\n오늘 예산을 다 썼다 ({i - 1:,}/{len(쌍들):,} 처리). "
                  f"내일 같은 명령으로 이어 받는다.")
            break
        except base_info_store.BaseInfoStoreError:
            # 🔴 달력 경계다 — 마지막 거래일은 **다음 거래일을 몰라** known_at 을 못 낸다.
            #    지어내지 않는 것이 옳지만, 그렇다고 여기서 죽으면 안 된다.
            #    2026-09-03 에 실제로 죽었다. 하필 마지막 쌍이라 8,201건이 살아남았을
            #    뿐, 경계가 중간에 있었다면 3시간짜리 수집이 통째로 날아갔다.
            #
            #    대장에 남기지 **않는다.** `error` 로 적으면 사람이 고칠 것을 찾게 되고,
            #    `empty` 로 적으면 영영 다시 안 받는다. 그냥 미뤄 두면 시세가 하루
            #    늘어 달력이 넓어졌을 때 다음 실행이 저절로 집어 간다.
            미룸.append((bas_dd, market))
            print(f"  ⏸️ {bas_dd} {market}: 달력 경계라 미뤄 둔다 "
                  f"(다음 거래일을 알아야 known_at 을 낸다)")
            continue
        except krx_data.KrxError as exc:
            err += 1
            print(f"  ⚠️ {bas_dd} {market}: {exc}")
            # 한 날짜가 실패해도 나머지는 계속 받는다. 대장에 error 로 남아 있어
            # 나중에 그 대상만 다시 부를 수 있다.
            continue

        담은행 += r["rows"]
        ok += r["status"] == "ok"
        empty += r["status"] == "empty"

        if i % args.every == 0 or i == len(쌍들):
            흐른 = time.time() - 시작
            남은 = (len(쌍들) - i) * (흐른 / i)
            print(f"  {i:>6,}/{len(쌍들):,}  담은행 {담은행:>9,}  "
                  f"ok {ok:,} · empty {empty:,} · error {err:,}  "
                  f"남은 시간 약 {남은/60:.0f}분")

    print(f"\n── 끝 ── {time.time() - 시작:.0f}초 · 담은행 {담은행:,} "
          f"(ok {ok:,} · empty {empty:,} · error {err:,})")
    if 미룸:
        # 침묵하지 않는다 — 안 받은 것을 안 세면 "다 받았다" 로 읽힌다.
        print(f"⏸️ 달력 경계로 미룬 것 {len(미룸)}쌍: "
              f"{', '.join(f'{d} {m}' for d, m in 미룸)}")
        print("   시세를 하루 더 받아 달력이 넓어지면 다음 실행이 저절로 집어 간다.")
    현황()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
