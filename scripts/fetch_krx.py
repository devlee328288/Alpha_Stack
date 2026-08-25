"""KRX 일별 시세 수집 스크립트

KRX 를 직접 두드려 `data/krx_cache.db` 를 채운다. 서버(`uvicorn`)와 별개로 돌릴 수 있으므로,
미리 한 번 채워 두면 화면은 처음부터 빠르게 뜬다.

사용법
------
    python3 scripts/fetch_krx.py                 # 최근 250거래일 (없는 날짜만)
    python3 scripts/fetch_krx.py --days 60       # 최근 60거래일만
    python3 scripts/fetch_krx.py --days 1        # 오늘 장 마감 후 하루치만 추가
    python3 scripts/fetch_krx.py --workers 4     # 동시 호출 수 조절 (기본 6)
    python3 scripts/fetch_krx.py --status        # 받지 않고 현재 캐시 상태만 확인

이미 받은 날짜는 건너뛴다. 휴장일(0건)도 기록해 두므로 다시 요청하지 않는다.
"""

import argparse
import sys
import time
from pathlib import Path

# 이 스크립트는 scripts/ 안에 있어서 파이썬이 프로젝트 루트를 모른다.
# `app` 패키지를 import 하려면 루트를 검색 경로에 직접 넣어 줘야 한다.
# (parents[0]=scripts, parents[1]=프로젝트 루트)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingest.store import krx_store as store  # noqa: E402  (경로 설정 후에 import 해야 한다)


def human(n: int) -> str:
    """숫자에 천 단위 콤마를 붙인다."""
    return f"{n:,}"


def print_status() -> None:
    """현재 캐시 상태를 표로 보여준다."""
    s = store.stats()
    print("── krx_cache.db 현황 ──")
    if not s["rows"]:
        print("  비어 있음 — python3 fetch_krx.py 로 먼저 채우세요.")
        return
    print(f"  기간      : {s['first_date']} ~ {s['last_date']}  ({s['days']}거래일)")
    print(f"  종목 수   : {human(s['codes'])}")
    print(f"  총 행 수  : {human(s['rows'])}")
    print(f"  파일 크기 : {s['db_size_mb']} MB ({s['db_path']})")


def main() -> int:
    parser = argparse.ArgumentParser(description="KRX 일별 시세를 SQLite 캐시에 채운다")
    parser.add_argument("--days", type=int, default=250, help="수집할 거래일 수 (기본 250)")
    parser.add_argument("--workers", type=int, default=6, help="동시 호출 수 (기본 6)")
    parser.add_argument("--end", help="기준일 YYYYMMDD (기본: 오늘)")
    parser.add_argument("--status", action="store_true", help="수집하지 않고 현황만 출력")
    args = parser.parse_args()

    if args.status:
        print_status()
        return 0

    print(f"KRX 수집 시작 — 최근 {args.days}거래일 · 시장 {', '.join(store.MARKETS)} "
          f"· 동시 {args.workers}개")
    started = time.time()

    def progress(done: int, total: int, bas_dd: str, rows: int, error) -> None:
        """한 날짜가 끝날 때마다 한 줄로 진행 상황을 덮어쓴다."""
        elapsed = time.time() - started
        eta = elapsed / done * (total - done) if done else 0
        mark = "실패" if error else f"{rows:>5}건"
        # \r 로 줄 앞으로 돌아가 같은 줄을 갱신한다 (로그가 수백 줄 쌓이지 않도록)
        sys.stdout.write(f"\r  [{done:>3}/{total}] {bas_dd} {mark}  "
                         f"경과 {elapsed:>5.0f}초 · 남은시간 {eta:>5.0f}초   ")
        sys.stdout.flush()

    result = store.sync(days=args.days, workers=args.workers, end=args.end, progress=progress)
    print()

    print(f"완료 — 요청 {result['requested']}일 중 "
          f"기존 {result['already']}일 · 신규 {result['fetched']}일 · "
          f"{human(result['rows'])}행 저장 ({time.time() - started:.0f}초)")

    if result["failed"]:
        print(f"실패 {len(result['failed'])}일:")
        for item in result["failed"][:10]:
            print(f"  - {item['date']}: {item['error'][:100]}")

    print()
    print_status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
