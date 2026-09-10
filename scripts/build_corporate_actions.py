"""`corporate_action` 표를 다시 깐다 (스키마 v16 · 이슈 #229).

    python scripts/build_corporate_actions.py           # 전 종목
    python scripts/build_corporate_actions.py --report  # 깔지 않고 세기만

## 왜 매번 통째로 다시 까나

이 표는 `daily_price` 와 `stock_base_info` 에서 **전부 계산한** 파생물이다. 원본이
늘거나 다시 받아 값이 바뀌면 사건도 바뀌는데, 덧붙이기로만 두면 **사라진 사건이 남는다.**
4만 행이라 통째로 다시 까는 비용이 없다 — `trading_calendar` 와 같은 판단이다.

## 사건 판정은 `common.corporate_actions.classify_event` 가 한다

여기서는 달력·상장일·정리매매 판정을 **한 번** 만들어 넘기는 일만 한다. 종목마다
다시 만들면 923만 행을 3,677번 훑어 한 시간을 넘긴다.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.corporate_actions import (  # noqa: E402
    codes_present_on,
    listing_days_by_code,
    market_calendar,
    market_calendar_index,
)
from common.paths import krx_db_path  # noqa: E402
from ingest.store.action_store import rebuild  # noqa: E402
from ingest.store.sqlite_db import write_lock  # noqa: E402

#: 보고 순서. 건수가 아니라 **뜻**으로 세운다 — 가격을 끊는 사건이 위, 담기만 하는
#: 사건이 아래다. 건수로 세우면 유상증자 39,515건이 늘 맨 위에 와서 눈이 거기 묶인다.
보고순서 = ("split_merge", "rights_off", "resume_revalue", "par_reduction",
         "series_restart", "market_transfer", "share_change")

설명 = {
    "split_merge": "액면분할·병합    (액면가 비율이 곧 가격 배율)",
    "rights_off": "권리락·주식배당   (기준가만 바뀐다)",
    "resume_revalue": "재개일 재평가     (KRX 가 단일가로 새로 매겼다)",
    "par_reduction": "액면가 감액       (결손금을 텄다 — 가격은 안 움직인다)",
    "series_restart": "시계열 재시작     (코드를 재사용했다 — 다른 회사다)",
    "market_transfer": "이전상장          (시장만 옮겼다 — 가격은 이어진다)",
    "share_change": "주식수 변동       (유상증자 등 — 가격은 연속이다)",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="기업행위 이벤트 표 (v16)")
    parser.add_argument("--db", default=None, help="DB 경로 (기본: data/krx_cache.db)")
    parser.add_argument("--report", action="store_true",
                        help="깔린 표를 세기만 한다 (다시 만들지 않는다)")
    args = parser.parse_args()

    db_path = Path(args.db).resolve() if args.db else krx_db_path()
    if not db_path.exists():
        print(f"DB 가 없다: {db_path}")
        return 1

    # `krx_store.connect()` 는 정본 경로만 연다. 여기서 직접 여는 까닭은 `--db` 로
    # 사본을 가리켜 시험할 수 있어야 하기 때문이다 (`build_adj_prices.py` 와 같다).
    conn = sqlite3.connect(db_path, timeout=60, isolation_level=None)
    conn.execute("PRAGMA busy_timeout=60000")
    conn.execute("PRAGMA cache_size = -1000000")
    try:
        if args.report:
            return _보고(conn)

        시작 = time.perf_counter()
        달력 = market_calendar(conn)
        if not 달력:
            print("거래일 달력이 비었다 — 먼저 "
                  "python scripts/build_adj_prices.py --calendar-only")
            return 1
        calendar_index, last_index = market_calendar_index(conn)
        listing = listing_days_by_code(conn)
        살아있음 = codes_present_on(conn, 달력[-1])
        재사용 = sum(1 for v in listing.values() if len(v) > 1)
        print(f"거래일 달력 {len(calendar_index):,}일 · 상장일이 둘 이상인 코드 {재사용}종 "
              f"· 마지막 거래일 {달력[-1]} 에 남은 종목 {len(살아있음):,}")
        print("전 종목을 훑는다 — 실측 약 4분\n")

        센것 = rebuild(conn, calendar_index=calendar_index,
                     listing_days_by_code=listing,
                     market_last_index=last_index, still_listed=살아있음,
                     collect_start=달력[0], write_lock=write_lock)
        print(f"넣은 행 {센것['행']:,} · {time.perf_counter() - 시작:.0f}초\n")
        return _보고(conn)
    finally:
        conn.close()


def _보고(conn) -> int:
    """깔린 표를 종류별로 센다. **판정은 여기서 하지 않는다** — 세어서 보여 줄 뿐이다."""
    전체 = conn.execute("SELECT COUNT(*) FROM corporate_action").fetchone()[0]
    if not 전체:
        print("표가 비었다 — --report 없이 한 번 돌려라")
        return 1

    print(f"── corporate_action · {전체:,}행 ──")
    print(f"  {'종류':<16}{'행':>8}{'배율':>7}{'맞음':>7}{'어긋남':>8}{'못 잼':>8}   설명")
    본것 = 0
    for t in 보고순서:
        r = conn.execute(
            "SELECT COUNT(*), SUM(ratio_num IS NOT NULL), "
            "       SUM(agrees = 1), SUM(agrees = 0), SUM(agrees IS NULL) "
            "FROM corporate_action WHERE event_type = ?", (t,)).fetchone()
        if not r[0]:
            continue
        본것 += r[0]
        print(f"  {t:<16}{r[0]:>8,}{(r[1] or 0):>7,}{(r[2] or 0):>7,}"
              f"{(r[3] or 0):>8,}{(r[4] or 0):>8,}   {설명[t]}")
    print("  '못 잼' 은 증거가 chain 과 같은 값이라 대조가 순환이 되는 자리다 — "
          "재면 늘 초록이다.")
    if 본것 != 전체:
        print(f"  ⚠️ 보고순서에 없는 종류가 {전체 - 본것:,}행 있다 — 목록을 고쳐라")

    어긋남 = conn.execute(
        "SELECT code, ex_date, event_type, ratio_num, ratio_den, "
        "       chain_num, chain_den, is_liquidation "
        "FROM corporate_action WHERE agrees = 0 ORDER BY ex_date").fetchall()
    print()
    print(f"── 증거와 chain 계수가 어긋난 {len(어긋남)}건 ──")
    if 어긋남:
        정리매매 = sum(1 for r in 어긋남 if r[7])
        print(f"  {'코드':<8}{'날짜':<10}{'종류':<16}{'증거 배율':>12}"
              f"{'chain':>12}   정리매매")
        for code, dd, t, rn, rd, cn, cd, liq in 어긋남:
            증거 = f"{rn}/{rd}" if rn is not None else "없음"
            체인 = f"{cn}/{cd}" if cn is not None else "없음"
            print(f"  {code:<8}{dd:<10}{t:<16}{증거:>12}{체인:>12}"
                  f"   {'예' if liq else '아니오'}")
        print()
        print(f"  그중 {정리매매}건이 정리매매다.")
    else:
        print("  없음")

    모순 = conn.execute(
        "SELECT code, ex_date, event_type, shares_before, shares_after, "
        "       basis_ratio, is_liquidation "
        "FROM corporate_action WHERE source_conflict = 1 ORDER BY ex_date").fetchall()
    print()
    print(f"── 원본이 자기모순인 {len(모순)}건 (주식수는 크게 바뀌었는데 등락률이 폭 밖) ──")
    if 모순:
        정리 = sum(1 for r in 모순 if r[6])
        print(f"  {'코드':<8}{'날짜':<10}{'종류':<16}{'주식수비':>14}{'기준가비':>10}"
              f"  정리매매")
        for code, dd, t, a, b, br, liq in 모순:
            비 = f"{b / a:.6g}" if a and b else "?"
            기 = f"{br:.6g}" if br is not None else "?"
            print(f"  {code:<8}{dd:<10}{t:<16}{비:>14}{기:>10}"
                  f"  {'예' if liq else '아니오'}")
        print()
        print(f"  {정리}/{len(모순)}건이 정리매매다 — 그 구간은 가격제한폭이 "
              "적용되지 않으므로 원본 오류가 아니고, `is_liquidation` 이 이미 덮어 "
              "`training_frame` 이 덜어낸다.")
        if 정리 != len(모순):
            print("  🔴 정리매매로 설명되지 않는 자리가 있다. 그건 새 결함이다 — 살펴봐라.")
    else:
        print("  없음")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
