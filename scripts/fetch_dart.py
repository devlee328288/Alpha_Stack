"""DART 재무제표 수집 스크립트

유니버스 350종의 사업보고서를 받아 `data/krx_cache.db` 의 `dart_financial` 에 채운다.
시세를 받는 `fetch_krx.py` 와 짝이고, 같은 DB 파일의 다른 표에 담는다.

사용법
------
    python scripts/fetch_dart.py                      # 350종 × FY2021~2025
    python scripts/fetch_dart.py --years 2023,2024    # 연도를 골라서
    python scripts/fetch_dart.py --codes 005930,000660
    python scripts/fetch_dart.py --limit 20           # 앞에서 20건만 (시험용)
    python scripts/fetch_dart.py --status             # 받지 않고 현황만

이미 받은 회사·연도는 건너뛴다. 받아 봤더니 없었던 것(상장 전·미제출)도 기록해 두므로
다시 요청하지 않는다.

## 콜이 얼마나 드나

회사·연도마다 **2콜**이다 — 재무 본문(`fnlttSinglAcntAll`) 1콜과 접수일 조회
(`list.json`) 1콜. 실측 2026-09-02 로 확인했다(삼성전자 2023 을 받으니 예산 장부의
`used` 가 정확히 2 올랐다).

    350종 × 5개년 × 2콜 = 3,500콜   (DART 하루 한도 20,000 의 17.5%)

연결(CFS)이 비어 별도(OFS)로 재시도하는 회사는 3콜이 된다. 한도에 닿으면 예외를
던지지 않고 **얌전히 멈춘다** — 다음 날 다시 실행하면 받은 곳부터 이어 받는다.

## 🔴 접수일이 없으면 저장하지 않는다

`bsns_year` 는 결산기이지 세상이 알게 된 날이 아니다. 2020년 4분기 실적은 2021년
3월에 나오므로, 결산기에 값을 붙이면 석 달치 미래를 학습에 넣고도 예외는 나지 않고
성능만 좋아진다. 그래서 `rcept_dt` 가 빈 응답은 오류로 기록하고 넘어간다.
"""

import argparse
import sys
import time
from pathlib import Path

# 이 스크립트는 scripts/ 안에 있어서 파이썬이 프로젝트 루트를 모른다.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import budget  # noqa: E402
from ingest.store import dart_store  # noqa: E402


def human(n: int) -> str:
    return f"{n:,}"


def print_status() -> None:
    """받지 않고 현황만 보여준다."""
    import sqlite3

    from common.paths import krx_db_path

    db = krx_db_path()
    if not db.exists():
        print(f"🔴 DB 가 없습니다: {db}")
        return

    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT COUNT(*) FROM dart_financial").fetchone()[0]
        corps = con.execute(
            "SELECT COUNT(DISTINCT corp_code) FROM dart_financial").fetchone()[0]
        years = con.execute(
            "SELECT MIN(bsns_year), MAX(bsns_year) FROM dart_financial").fetchone()
        print("── dart_financial 현황 ──")
        print(f"  행 수    : {human(rows)}")
        print(f"  회사 수  : {human(corps)}")
        print(f"  사업연도 : {years[0]} ~ {years[1]}")

        print("\n── 수집 대장 ──")
        for status, n, rows_sum in con.execute(
            "SELECT status, COUNT(*), SUM(rows) FROM collect_log "
            "WHERE source = ? GROUP BY status ORDER BY status",
            (dart_store.SOURCE,),
        ):
            print(f"  {status:<10} {n:>6}건  {human(rows_sum or 0):>10}행")
    finally:
        con.close()

    print("\n── 오늘 호출 예산 ──")
    usage = budget.usage("dart")
    if usage:
        u = usage["dart"]
        print(f"  dart  {u['used']:,} / {u['limit']:,}  ({u['ratio'] * 100:.1f}%)")
    else:
        print("  dart  아직 오늘 부른 적이 없습니다")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="DART 재무제표를 받아 SQLite 에 채운다")
    parser.add_argument("--years", default="2021,2022,2023,2024,2025",
                        help="사업연도 (쉼표로 구분, 기본 2021~2025)")
    parser.add_argument("--codes", default=None,
                        help="종목코드를 골라서 (쉼표로 구분, 기본은 유니버스 전체)")
    parser.add_argument("--reprt", default="11011",
                        help="보고서 코드 (11011 사업보고서 · 11012 반기 · "
                             "11013 1분기 · 11014 3분기)")
    parser.add_argument("--limit", type=int, default=None,
                        help="앞에서 N 건만 받는다 (시험용)")
    parser.add_argument("--status", action="store_true", help="받지 않고 현황만")
    args = parser.parse_args()

    if args.status:
        print_status()
        return 0

    years = [int(y) for y in args.years.split(",") if y.strip()]
    codes = [c.strip() for c in args.codes.split(",")] if args.codes else None

    universe = dart_store.load_universe()
    대상수 = len(codes) if codes else len(universe)
    print(f"DART 재무 수집 — 종목 {대상수} × 연도 {len(years)} "
          f"· 보고서 {args.reprt}")
    print(f"  예상 최대 {대상수 * len(years) * 2:,}콜 (회사·연도마다 2콜)")
    usage = budget.usage("dart")
    if usage:
        u = usage["dart"]
        print(f"  오늘 이미 쓴 콜: {u['used']:,} / {u['limit']:,}")
    print()

    started = time.time()

    def progress(done, total, code, name, year, status, rows, note):
        elapsed = time.time() - started
        eta = elapsed / done * (total - done) if done else 0
        mark = {"ok": "✅", "empty": "⚪", "error": "❌"}.get(status, "  ")
        꼬리 = f"  {note[:40]}" if note else ""
        print(f"  [{done:>4}/{total}] {mark} {code} {name[:10]:<10} {year} "
              f"{rows:>4}행  경과 {elapsed:>5.0f}초 · 남은시간 {eta:>5.0f}초{꼬리}",
              flush=True)

    result = dart_store.sync(codes=codes, years=years, reprt_code=args.reprt,
                             progress=progress, limit=args.limit)

    걸림 = time.time() - started
    print()
    print(f"완료 — 요청 {result['requested']}건 중 기존 {result['already']}건 · "
          f"성공 {result['ok']}건 · 없음 {result['empty']}건 · 실패 {result['error']}건")
    print(f"  저장 {human(result['rows'])}행 ({걸림:.0f}초)")

    if result["stopped"]:
        print(f"\n⚠️ {result['stopped']}")
        print("   할 일: 내일 같은 명령을 다시 실행하면 받은 곳부터 이어 받습니다.")

    print()
    print_status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
