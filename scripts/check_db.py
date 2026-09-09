"""수집 DB 에 **무엇이 얼마나** 들어 있는지 한 화면에 보여 준다.

    python scripts/check_db.py
    python scripts/check_db.py --log      # 수집 대장까지 함께

품질을 판정하지 않는다 — 그건 `check_data.py`(시세)와 `check_dart.py`(재무)가 한다.
여기는 **"지금 뭐가 있나"** 에만 답한다.

## 왜 따로 필요한가

표가 늘면서 현황을 보려면 `fetch_krx.py --status` · `fetch_index.py --status` ·
`fetch_dart.py --status` 를 따로 쳐야 했다. 새 표가 생길 때마다 볼 곳이 하나씩 는다.

이 스크립트는 **`sqlite_master` 를 읽어 표를 스스로 찾는다.** 그래서 마이그레이션으로
표가 추가돼도 고칠 것이 없다 — 날짜로 볼 만한 칸이 있으면 기간까지 함께 보여 준다.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import krx_db_path  # noqa: E402

#: 날짜로 볼 만한 칸 이름. 표마다 이름이 달라 하나로 못 묶는다.
#: 앞에 있는 것이 우선이다 — `bas_dd`(기준일)가 `collected_at`(받은 시각)보다 먼저다.
DATE_COLUMNS = ("bas_dd", "base_dt", "trd_dd", "rcept_dt", "bsns_year",
                "kst_date", "period", "dt", "date")


def main() -> int:
    parser = argparse.ArgumentParser(description="수집 DB 현황을 한 화면에 보여 준다")
    parser.add_argument("--log", action="store_true", help="수집 대장까지 함께")
    args = parser.parse_args()

    db = krx_db_path()
    if not db.exists():
        print(f"🔴 DB 가 없습니다: {db}")
        print("   할 일: python scripts/fetch_krx.py 로 먼저 채우세요.")
        return 1

    # ⚠️ 읽기 전용 — 수집이 도는 중에 열면 잠금을 다툰다
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        version = con.execute("PRAGMA user_version").fetchone()[0]
        크기 = db.stat().st_size / 1024 / 1024
        print(f"── {db.name} · {크기:,.0f} MB · 스키마 v{version} ──\n")

        표들 = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name")]

        print(f"  {'표':<20} {'행':>12}   기간")
        print(f"  {'-' * 20} {'-' * 12}   {'-' * 24}")
        for t in 표들:
            n = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            칸 = [r[1] for r in con.execute(f'PRAGMA table_info("{t}")')]
            날짜칸 = next((c for c in DATE_COLUMNS if c in 칸), None)
            기간 = ""
            if 날짜칸 and n:
                lo, hi = con.execute(
                    f'SELECT MIN("{날짜칸}"), MAX("{날짜칸}") FROM "{t}"').fetchone()
                기간 = f"{날짜칸} {lo} ~ {hi}"
            표시 = "" if n else "   (비어 있음)"
            print(f"  {t:<20} {n:>12,}   {기간}{표시}")

        if args.log:
            print("\n── 수집 대장 ──")
            for src, status, cnt, rows in con.execute(
                "SELECT source, status, COUNT(*), SUM(rows) FROM collect_log "
                "GROUP BY source, status ORDER BY source, status"
            ):
                print(f"  {src:<16} {status:<10} {cnt:>6}건  {rows or 0:>12,}행")

            print("\n── 오늘 호출 예산 ──")
            for src, kst_date, used, limit in con.execute(
                "SELECT source, kst_date, used, daily_limit FROM call_budget "
                "ORDER BY kst_date DESC, source LIMIT 8"
            ):
                print(f"  {kst_date}  {src:<14} {used:>7,} / {limit:,}")
    finally:
        con.close()

    print("\n품질 판정은 여기서 하지 않습니다 —")
    print("  python scripts/check_data.py    시세·지수")
    print("  python scripts/check_dart.py    재무")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
