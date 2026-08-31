"""KRX 지수 일별시세 수집 스크립트

1차 프로젝트의 **예측 대상**(코스피 200)을 채운다. 종목 시세를 채우는
`fetch_krx.py` 와 짝이고, 같은 DB 파일의 다른 표에 담는다.

사용법
------
    python scripts/fetch_index.py                    # 최근 250거래일 (없는 날짜만)
    python scripts/fetch_index.py --days 4343        # 2010-01-04 까지 전구간
    python scripts/fetch_index.py --markets KOSPI,KOSDAQ
    python scripts/fetch_index.py --status           # 받지 않고 현황만
    python scripts/fetch_index.py --list             # 어떤 지수가 쌓였는지 본다

이미 받은 날짜는 건너뛴다. 휴장일(0건)도 기록해 두므로 다시 요청하지 않는다.

⚠️ 하루 한도 — KRX 는 **인증키당 1일 10,000회**를 허용한다 (이용약관 제8조 ④).
   16년 전구간은 시장당 약 4,343콜이다. 같은 날 `fetch_krx.py`(종목, 8,686콜)를
   함께 돌리면 한도를 넘는다. **지수를 먼저 받고 종목은 다음 날 이어 받는다.**
   중단해도 안전하다 — 받은 날짜는 `index_fetch_log` 에 남아 다시 요청하지 않는다.
"""

import argparse
import sys
import time
from pathlib import Path

# 이 스크립트는 scripts/ 안에 있어서 파이썬이 프로젝트 루트를 모른다.
# (parents[0]=scripts, parents[1]=프로젝트 루트)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingest.clients import krx_data as api  # noqa: E402  (경로 설정 후에 import)
from ingest.store import krx_index as store  # noqa: E402


def human(n: int) -> str:
    """숫자에 천 단위 콤마를 붙인다."""
    return f"{n:,}"


def print_status() -> None:
    """현재 수집 현황을 표로 보여준다."""
    s = store.stats()
    print("── 지수 시세 현황 ──")
    if not s["rows"]:
        print("  비어 있음 — python scripts/fetch_index.py 로 먼저 채우세요.")
        return
    print(f"  기간      : {s['first_date']} ~ {s['last_date']}  ({human(s['days'])}거래일)")
    print(f"  지수 종류 : {human(s['indices'])}종")
    print(f"  총 행 수  : {human(s['rows'])}")
    print(f"  수집 대장 : {human(s['logged_days'])}일 (그중 휴장 확인 {human(s['closed_days'])}일)")
    print(f"  DB        : {s['db_path']}")

    # 예측 대상이 실제로 들어왔는지 확인한다 — 이게 없으면 나머지가 다 있어도 소용없다
    target = store.series(api.TARGET_INDEX)
    if target:
        print(f"  ★ {api.TARGET_INDEX} : {len(target):,}거래일 "
              f"({target[0]['date']} ~ {target[-1]['date']}) · 최근 종가 {target[-1]['close']}")
    else:
        print(f"  ⚠️ {api.TARGET_INDEX} 가 아직 없습니다 — 예측 대상이 비어 있습니다.")


def print_collect_log() -> None:
    """수집 대장 현황. **왜 안 받았는지**가 여기서 갈린다."""
    from ingest.store import collect_log

    요약 = collect_log.summary()
    if not 요약:
        return

    print()
    print("── 수집 대장 ──")
    for source in sorted(요약):
        칸 = 요약[source]
        print(f"  {source:<10} 받음 {human(칸['ok'])} · 0건 {human(칸['empty'])} · "
              f"범위밖 {human(칸['out_of_range'])} · 실패 {human(칸['error'])} · "
              f"한도소진 {human(칸['quota_exhausted'])}")
        if 칸["last_success_at"]:
            print(f"{'':<12} 마지막 성공 {칸['last_success_at']}")

    # 재시도를 다 쓰고도 실패한 것만 따로 보여 준다 — 사람이 봐야 할 유일한 목록이다.
    막힌것 = collect_log.stuck()
    if 막힌것:
        print(f"  ⚠️ 재시도를 다 쓴 대상 {len(막힌것)}건 — 사람이 봐야 합니다:")
        for row in 막힌것[:10]:
            print(f"     - {row['source']} {row['target']}: {(row['note'] or '')[:80]}")


def print_raw() -> None:
    """보존된 응답 원문 현황. 이게 있어야 다시 받지 않고 다시 정규화할 수 있다."""
    from common import raw_store, settings

    현황 = raw_store.stats()
    if not 현황:
        if settings.keep_raw_enabled():
            print()
            print("── 응답 원문 ── 아직 없음 (원문 보존은 2026-08-26 에 붙었습니다)")
        else:
            print()
            print("── 응답 원문 ── ⚠️ KEEP_RAW=off — 재정규화와 known_at 근거를 포기한 상태입니다")
        return

    print()
    print("── 응답 원문 ──")
    for source in sorted(현황):
        칸 = 현황[source]
        print(f"  {source:<10} 응답 {human(칸['responses'])}건 · 대상 {human(칸['targets'])}개 · "
              f"{human(칸['stored_bytes'])}B 저장 (원문의 {칸['ratio']:.1%})")
    print("     python scripts/renormalize.py --dry-run 으로 다시 정규화할 수 있습니다")


def print_list() -> None:
    """쌓인 지수 목록. 무엇을 피처로 쓸 수 있는지 고를 때 본다."""
    rows = store.available_indices()
    if not rows:
        print("아직 아무것도 없습니다.")
        return
    print(f"{'지수명':<32} {'시장':<8} {'거래일':>8}  {'구간'}")
    print("-" * 78)
    for r in rows:
        mark = "★" if r["index_name"] in (api.TARGET_INDEX, api.TARGET_INDEX_KOSDAQ) else " "
        print(f"{mark}{r['index_name']:<31} {r['index_class']:<8} {r['days']:>8,}  "
              f"{r['first_date']} ~ {r['last_date']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="KRX 지수 일별시세를 수집한다")
    parser.add_argument("--days", type=int, default=250,
                        help="수집할 거래일 수 (기본 250 · 전구간은 4343)")
    # ⚠️ 기본 1 이다. 지수 엔드포인트는 동시 호출 시 401 을 뱉는다 (krx_index.DEFAULT_WORKERS 주석)
    parser.add_argument("--workers", type=int, default=store.DEFAULT_WORKERS,
                        help=f"동시 호출 수 (기본 {store.DEFAULT_WORKERS} · 올리면 401 이 난다)")
    parser.add_argument("--end", help="기준일 YYYYMMDD (기본: 오늘)")
    parser.add_argument("--markets", default=",".join(store.DEFAULT_MARKETS),
                        help=f"쉼표로 구분 (기본 {','.join(store.DEFAULT_MARKETS)})")
    parser.add_argument("--status", action="store_true", help="수집하지 않고 현황만 출력")
    parser.add_argument("--list", action="store_true", help="쌓인 지수 목록만 출력")
    parser.add_argument("--import-legacy", action="store_true",
                        help="옛 대장(fetch_log·index_fetch_log)의 이력을 수집 대장으로 옮긴다")
    args = parser.parse_args()

    if args.import_legacy:
        from ingest.store import collect_log, krx_store
        # 옮기지 않으면 이미 받은 거래일이 전부 미수집으로 보여 16년치를 다시 받는다.
        옮김 = collect_log.import_legacy()
        for source, count in sorted(옮김.items()):
            print(f"  {source}: {human(count)}건 확인")
        # 종목 시세는 옛 표에 시장 칸이 없어 따로 깐다. `daily_price` 를 실제로 세어
        # `KOSPI/20260826` 형태로 만든다 — 지수 쪽과 같은 규칙이라야 화면이 한 벌로 읽는다.
        결과 = krx_store.rebuild_collect_log()
        print(f"  krx_stock: {human(결과['built'])}건 확인 "
              f"(옛 날짜 전용 줄 {human(결과['removed'])}건 정리)")
        print_collect_log()
        return 0

    if args.status:
        print_status()
        print_collect_log()
        print_raw()
        return 0
    if args.list:
        print_list()
        return 0

    markets = tuple(m.strip().upper() for m in args.markets.split(",") if m.strip())
    unknown = [m for m in markets if m not in api.INDEX_APIS]
    if unknown:
        # 막다른 길로 만들지 않는다 — 무엇을 써야 하는지까지 알려 준다
        print(f"모르는 시장입니다: {', '.join(unknown)}")
        print(f"쓸 수 있는 값: {', '.join(api.INDEX_APIS)}")
        return 1

    est_calls = args.days * len(markets)
    print(f"지수 수집 시작 — 최근 {args.days}거래일 · 시장 {', '.join(markets)} "
          f"· 동시 {args.workers}개")
    print(f"  예상 콜 최대 {human(est_calls)}회 (1일 한도 10,000 의 {est_calls / 100:.0f}%)"
          f"{'  ⚠️ 한도를 넘습니다' if est_calls > 10000 else ''}")
    started = time.time()

    def progress(done: int, total: int, label: str, rows: int, error) -> None:
        elapsed = time.time() - started
        eta = elapsed / done * (total - done) if done else 0
        mark = "실패" if error else f"{rows:>3}건"
        # \r 로 같은 줄을 갱신한다 (로그가 수천 줄 쌓이지 않도록)
        sys.stdout.write(f"\r  [{done:>4}/{total}] {label} {mark}  "
                         f"경과 {elapsed:>5.0f}초 · 남은시간 {eta:>5.0f}초   ")
        sys.stdout.flush()

    result = store.sync(days=args.days, workers=args.workers, end=args.end,
                        markets=markets, progress=progress)
    print()

    print(f"완료 — 요청 {result['requested']}건 중 "
          f"기존 {result['already']}건 · 신규 {result['fetched']}건 · "
          f"{human(result['rows'])}행 저장 ({time.time() - started:.0f}초)")

    if result["failed"]:
        print(f"실패 {len(result['failed'])}건:")
        for item in result["failed"][:10]:
            print(f"  - {item['date']} {item['market']}: {item['error'][:100]}")

    if result.get("quota_exhausted"):
        print(f"한도 소진으로 미룬 것 {result['quota_exhausted']}건 — "
              "실패가 아닙니다. 내일 다시 실행하면 이어서 받습니다.")

    print()
    print_status()
    print_collect_log()
    print_raw()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
