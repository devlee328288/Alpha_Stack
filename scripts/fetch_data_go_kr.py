"""공공데이터포털 금융위 자료를 받아 `stock_identity`·`corp_profile` 에 담는다.

    python scripts/fetch_data_go_kr.py --status              # 받지 않고 현황만
    python scripts/fetch_data_go_kr.py --plan                # 몇 콜 쓸지만 센다
    python scripts/fetch_data_go_kr.py --listed --limit 5    # 종목 목록 5일치
    python scripts/fetch_data_go_kr.py --listed              # 종목 목록 전 구간
    python scripts/fetch_data_go_kr.py --profile --limit 50  # 법인 개요 50곳

시세의 `fetch_krx.py`, 재무의 `fetch_dart.py`, 거시의 `fetch_macro.py` 와 같은 자리다.

## 시세를 받는 게 아니다 — **다리**를 받는다

    KRX 종목코드  ↔  crno(법인등록번호)  ↔  ISIN

포털의 주식시세는 20칸뿐이고 우리 KRX 수집은 이미 9,223,644행이다. 여기서 얻는 것은
다른 자료에 붙일 **연결 고리**와, 시세로는 알 수 없는 **공식 상장폐지일**이다.

## 🔴 한도를 넘기 전에 먼저 센다

개발계정은 하루 10,000회다. 받다가 중간에 막히면 어디까지 받았는지 맞추는 일이 생기므로
`--plan` 으로 **먼저 세고** 시작한다. 실측 기준:

    종목 목록 : 2020-01-02 부터 · 약 1,180 거래일 × 3콜 ≈ 3,540콜
    법인 개요 : 법인 하나에 1콜 (전 이력이 한 번에 온다) · 약 2,700콜

둘을 같은 날 다 받아도 한도 안이지만, 여유가 없으므로 하루씩 나누는 편이 안전하다.

## 이어받기

`collect_log` 에 대상마다 한 줄을 남긴다. 이미 `ok` 인 날짜는 건너뛴다 —
`--force` 를 주면 다시 받는다 (자료가 정정됐을 때).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import budget  # noqa: E402
from common.trading_calendar import load_session_days  # noqa: E402
from ingest.clients import data_go_kr  # noqa: E402
from ingest.store import collect_log, identity_store  # noqa: E402


def 받을_날짜들(*, limit: int = 0, force: bool = False,
              start: str = "", end: str = "") -> list:
    """받을 기준일 목록. 달력에서 뽑고, 이미 받은 날은 뺀다.

    🔴 **거래일 달력에서 뽑는다.** 날짜를 계산으로 만들면 휴장일에 헛호출을 하고,
       개발구간에서만 162일(5.3%)이 어긋난다.
    """
    달력 = sorted(load_session_days())
    시작 = max(start or data_go_kr.EARLIEST_BAS_DD, data_go_kr.EARLIEST_BAS_DD)
    날짜들 = [d for d in 달력 if d >= 시작 and (not end or d <= end)]

    if not force:
        남길것 = []
        for d in 날짜들:
            기록 = collect_log.entry(identity_store.SOURCE, f"listed:{d}")
            # `empty`(휴장) 도 받은 것으로 친다 — 안 그러면 휴장일마다 영원히 다시 부른다.
            if 기록 and 기록.get("status") in ("ok", "empty", "out_of_range"):
                continue
            남길것.append(d)
        날짜들 = 남길것

    return 날짜들[:limit] if limit else 날짜들


def 현황을_보여준다() -> int:
    """무엇이 얼마나 들어 있는지. 받지는 않는다."""
    identity_store.ensure_schema()
    상태 = identity_store.status()
    신원, 개요 = 상태["stock_identity"], 상태["corp_profile"]
    print("── 종목 신원 · 법인 개요 ──\n")
    print(f"  stock_identity : {신원['rows']:>9,}행 · {신원['days']:,}일 "
          f"· {신원['codes']:,}종목")
    if 신원["first"]:
        print(f"                   {신원['first']} ~ {신원['last']}")
    print(f"  corp_profile   : {개요['rows']:>9,}행 · 법인 {개요['crno']:,}곳")
    if 개요["first"]:
        print(f"                   유효시작 {개요['first']} ~ {개요['last']}")

    남은법인 = identity_store.crno_targets()
    print(f"\n  개요를 아직 안 받은 법인 : {len(남은법인):,}곳")
    남은날짜 = 받을_날짜들()
    print(f"  목록을 아직 안 받은 날짜 : {len(남은날짜):,}일")
    if 남은날짜:
        print(f"    {남은날짜[0]} ~ {남은날짜[-1]}")
    return 0


def 계획을_보여준다(날짜들: list) -> int:
    """받기 전에 몇 콜을 쓸지 센다. **절반쯤 받다 멈추는 것을 막는다.**"""
    목록콜 = data_go_kr.estimate_calls(날짜들)
    법인콜 = len(identity_store.crno_targets())
    # `usage()` 는 `{출처: {used, limit, ...}}` 로 **감싸** 돌려준다. 오늘 한 번도 안
    # 썼으면 그 출처 키 자체가 없다 — 없는 것을 0 으로 읽는다.
    장부 = budget.usage(data_go_kr.BUDGET_SOURCE).get(data_go_kr.BUDGET_SOURCE, {})
    쓴콜 = int(장부.get("used", 0))

    print("── 받기 전 계산 ──\n")
    print(f"  종목 목록 : {len(날짜들):,}일 × 3페이지 ≈ {목록콜:,}콜")
    print(f"  법인 개요 : {법인콜:,}곳 × 1콜   = {법인콜:,}콜")
    print(f"  합계                        ≈ {목록콜 + 법인콜:,}콜")
    print(f"\n  하루 한도 : {data_go_kr.DAILY_LIMIT:,}회")
    print(f"  오늘 쓴 것: {쓴콜:,}회")
    남음 = data_go_kr.DAILY_LIMIT - 쓴콜
    총 = 목록콜 + 법인콜
    if 총 > 남음:
        print(f"\n  ⚠️ {총 - 남음:,}콜이 모자란다.")
        print("     할 일: --limit 으로 나눠 받거나, 목록과 개요를 다른 날 받는다.")
    else:
        print(f"\n  ✅ 오늘 한도 안에 들어온다 (여유 {남음 - 총:,}콜)")
    return 0


def 목록을_받는다(날짜들: list, *, dry_run: bool) -> int:
    if dry_run:
        print(f"[모의] 종목 목록 {len(날짜들):,}일 — 받지 않는다")
        return 0
    if not data_go_kr.available():
        print("🔴 DATA_GO_KR_API_KEY 가 없다.")
        print("   할 일: .env 에 넣는다. 발급 절차는")
        print("         docs/데이터파트/version3.2/API키_발급_가이드.md")
        return 1

    identity_store.ensure_schema()
    받은행, 빈날, 실패 = 0, 0, 0
    for i, 날짜 in enumerate(날짜들, 1):
        try:
            결과 = identity_store.sync_listed_day(날짜)
        except data_go_kr.DataGoKrError as exc:
            실패 += 1
            print(f"  🔴 {날짜} 실패 — 여기서 멈춘다\n{exc}")
            break
        받은행 += 결과["rows"]
        if 결과["status"] == "empty":
            빈날 += 1
        if i % 20 == 0 or i == len(날짜들):
            print(f"  {i:>5,}/{len(날짜들):,}  {날짜}  누적 {받은행:,}행")
    print(f"\n  종목 목록: {받은행:,}행 · 0건인 날 {빈날:,} · 실패 {실패}")
    return 1 if 실패 else 0


def 개요를_받는다(*, limit: int, dry_run: bool) -> int:
    대상 = identity_store.crno_targets()
    대상 = 대상[:limit] if limit else 대상
    if dry_run:
        print(f"[모의] 법인 개요 {len(대상):,}곳 — 받지 않는다")
        return 0
    if not 대상:
        print("  법인 개요: 받을 것이 없다 (stock_identity 를 먼저 받는다)")
        return 0

    identity_store.ensure_schema()
    받은행, 빈곳, 실패 = 0, 0, 0
    for i, crno in enumerate(대상, 1):
        try:
            결과 = identity_store.sync_profile(crno)
        except data_go_kr.DataGoKrError as exc:
            실패 += 1
            print(f"  🔴 {crno} 실패 — 여기서 멈춘다\n{exc}")
            break
        받은행 += 결과["rows"]
        if 결과["status"] == "empty":
            빈곳 += 1
        if i % 50 == 0 or i == len(대상):
            print(f"  {i:>5,}/{len(대상):,}  {crno}  누적 {받은행:,}행")
    print(f"\n  법인 개요: {받은행:,}행 · 0건인 곳 {빈곳:,} · 실패 {실패}")
    return 1 if 실패 else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="공공데이터포털 금융위 — 종목 신원 · 법인 개요를 받는다")
    parser.add_argument("--listed", action="store_true", help="종목 목록을 받는다")
    parser.add_argument("--profile", action="store_true", help="법인 개요를 받는다")
    parser.add_argument("--status", action="store_true", help="받지 않고 현황만")
    parser.add_argument("--plan", action="store_true", help="받지 않고 콜 수만 센다")
    parser.add_argument("--dry-run", action="store_true", help="무엇을 할지만 보여 준다")
    parser.add_argument("--limit", type=int, default=0, help="이만큼만 (0 이면 전부)")
    parser.add_argument("--force", action="store_true",
                        help="이미 받은 날짜도 다시 받는다 (자료가 정정됐을 때)")
    parser.add_argument("--start", default="", help="이 날짜부터 (YYYYMMDD)")
    parser.add_argument("--end", default="", help="이 날짜까지 (YYYYMMDD)")
    args = parser.parse_args()

    if args.status:
        return 현황을_보여준다()

    날짜들 = 받을_날짜들(limit=args.limit, force=args.force,
                    start=args.start, end=args.end)
    if args.plan:
        return 계획을_보여준다(날짜들)

    if not (args.listed or args.profile):
        # 🔴 아무것도 안 고르면 **아무것도 하지 않는다.** 기본값을 "전부 받기" 로 두면
        #    실수로 한 번 돌렸을 때 한도를 다 태운다.
        parser.print_help()
        print("\n  받으려면 --listed 또는 --profile 을 고른다.")
        print("  먼저 --plan 으로 몇 콜을 쓸지 세어 보는 것을 권한다.")
        return 0

    코드 = 0
    if args.listed:
        코드 |= 목록을_받는다(날짜들, dry_run=args.dry_run)
    if args.profile:
        코드 |= 개요를_받는다(limit=args.limit, dry_run=args.dry_run)
    return 코드


if __name__ == "__main__":
    raise SystemExit(main())
