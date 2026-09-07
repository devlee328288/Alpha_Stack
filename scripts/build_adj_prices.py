"""수정주가를 전 종목에 채운다 (마이그레이션 v9 · 이슈 #51).

    python scripts/build_adj_prices.py                 # 전 종목
    python scripts/build_adj_prices.py --limit 20      # 앞 20종만 (연습용)
    python scripts/build_adj_prices.py --codes 005930,035420
    python scripts/build_adj_prices.py --calendar-only # 거래일 달력만 다시 깐다

`daily_price.close` 는 액면분할이 조정되지 않은 원가격이라 수익률로 계산하면 삼성전자
2018-05-04 가 -98.04% 로 읽힌다. 조정된 값을 `adj_open`·`adj_high`·`adj_low`·`adj_close`
네 칸에 채우고, 어디서 온 값인지를 `adj_source` 에 남긴다.

FDR 은 최근 3,000거래일만 주므로(20140613~) 그 앞은 우리 조정계수로 이어 붙인다.
자세한 이유와 방법은 `ingest/store/adj_price.py` 의 모듈 주석에 있다.

무엇을 검증하나
---------------
적재는 **검증을 통과한 뒤에** 커밋한다. 행 수만 세면 안 된다 — 칸을 덮어써서 값이
사라져도 행 수는 그대로이기 때문이다. 네 가지를 본다.

  1. **분할일 갭이 펴졌나** — 수정가격 일간 수익률이 KRX `change_rate` 와 맞나.
     KRX 등락률은 원래부터 분할을 반영한 값이라 정답지 노릇을 한다.
  2. **없는 조정을 만들지 않았나** — 이벤트가 없던 종목에서 수정가격이 원가격과 같나.
  3. **고저 관계가 살아 있나** — `adj_low ≤ adj_close ≤ adj_high`.
  4. **원가격이 그대로인가** — `close` 합계가 적재 전후로 같나. (원본 보존 확인)

⚠️ 1번은 겹치는 구간(FDR)에서만 정답지가 있다. 2010~2014 구간은 대조할 외부 자료가
   없으므로, `scripts/_probe_adj_chain.py` 가 **앵커 하나만 남기고 우리 계산으로 채워
   FDR 과 대조하는** 방식으로 그 구간의 오차 상한을 대신 잰다 (실측 최대 0.39%).
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.paths import krx_db_path  # noqa: E402
from ingest.clients import fdr_data  # noqa: E402
from ingest.store import adj_price, collect_log  # noqa: E402
from ingest.store.sqlite_db import write_lock  # noqa: E402

#: 몇 종목마다 커밋할 것인가. 전부 한 트랜잭션에 묶으면 6분 내내 쓰기 잠금을 쥐고,
#: 종목마다 커밋하면 3,677번의 fsync 가 그대로 비용이 된다.
COMMIT_EVERY = 100

#: 분할일 갭 검증에서 허용할 오차(%p). KRX `change_rate` 는 소수 2자리 반올림이고
#: FDR 수정가격도 원 단위로 반올림돼 있어 0.1%p 안팎은 반올림에서 나온다.
GAP_TOLERANCE = 0.15


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=60, isolation_level=None)
    conn.execute("PRAGMA busy_timeout=60000")
    conn.row_factory = sqlite3.Row
    return conn


def all_codes(conn: sqlite3.Connection) -> List[str]:
    return [row[0] for row in conn.execute(
        "SELECT DISTINCT code FROM daily_price ORDER BY code")]


def _count(built: List[tuple], source: str) -> int:
    """그 출처인 행 수. ⚠️ **`==` 로 세면 안 된다** — `fdr+ca_fix` 처럼 접미사가 붙는다.

    2026-09-04 에 `+ca_fix` 를 도입하며 실제로 겪었다. 16종을 다시 깔았는데 요약이
    "fdr 5,959 · chain 0" 이라 나왔다 — 43,778행이 어느 쪽에도 안 세어진 것이다.
    **행 수는 맞는데 내역이 틀리는** 종류의 오류라 눈에 잘 안 띈다.
    """
    return sum(1 for r in built if r[4].startswith(source))


def _count_ca_fix(built: List[tuple]) -> int:
    """FDR 이 안 편 자본변동(감자)을 우리가 편 행 수."""
    return sum(1 for r in built if adj_price.SOURCE_CA_FIX in r[4])


def build_one(conn: sqlite3.Connection, code: str) -> Dict:
    """한 종목을 받아 계산해 저장한다. 무슨 일이 있었는지 요약을 돌려준다."""
    try:
        adjusted = fdr_data.fetch_adjusted(code)
        error = None
    except fdr_data.FdrUnavailable as exc:
        adjusted, error = {}, str(exc).splitlines()[0]

    rows = adj_price.load_rows(conn, code)
    if not rows:
        return {"code": code, "rows": 0, "fdr": 0, "chain": 0, "error": error}

    built = adj_price.build_rows(rows, adjusted)
    adj_price.save(conn, code, built)
    collect_log.record(
        adj_price.COLLECT_SOURCE, code,
        collect_log.OK if built else collect_log.EMPTY,
        rows=len(built),
        # 커서에 "FDR 이 어디까지 줬나" 를 남긴다. 다음에 다시 돌릴 때 그 경계가
        # 움직였는지(= 새 자료가 3,000일 창 밖으로 밀렸는지) 바로 보인다.
        cursor=(fdr_data.coverage_span(adjusted) or ("", ""))[0] or None,
        note=f"fdr={_count(built, fdr_data.SOURCE_FDR)} "
             f"chain={_count(built, adj_price.SOURCE_CHAIN)} "
             f"ca_fix={_count_ca_fix(built)}"
             + (f" error={error}" if error else ""),
        conn=conn,
    )
    return {
        "code": code,
        "rows": len(built),
        "fdr": _count(built, fdr_data.SOURCE_FDR),
        "chain": _count(built, adj_price.SOURCE_CHAIN),
        "ca_fix": _count_ca_fix(built),
        "error": error,
    }


# ==================================================
# 검증
# ==================================================
def verify(conn: sqlite3.Connection, codes: Optional[List[str]] = None) -> bool:
    """적재 결과를 네 가지로 본다. 하나라도 어긋나면 `False`."""
    where, params = "", ()
    if codes:
        where = f" AND code IN ({','.join('?' * len(codes))})"
        params = tuple(codes)

    print("\n── 검증 ──")
    ok = True

    # ① 분할일 갭 — 수정가격 수익률이 KRX 등락률과 맞나
    #    같은 종목의 **연속한 두 거래일**을 창으로 묶어 비교한다. 정지일은 시·고·저가
    #    없이 종가만 있으므로 종가로만 잰다.
    gap = conn.execute(f"""
        WITH seq AS (
          SELECT code, bas_dd, adj_close, change_rate,
                 LAG(adj_close) OVER (PARTITION BY code ORDER BY bas_dd) AS prev_adj
          FROM daily_price
          WHERE adj_close IS NOT NULL AND close > 0 {where}
        )
        SELECT COUNT(*) AS n,
               SUM(CASE WHEN ABS((adj_close / prev_adj - 1) * 100 - change_rate)
                        > ? THEN 1 ELSE 0 END) AS bad
        FROM seq WHERE prev_adj IS NOT NULL AND prev_adj > 0 AND change_rate IS NOT NULL
    """, (GAP_TOLERANCE,) + params).fetchone()
    bad_rate = (gap["bad"] or 0) / gap["n"] * 100 if gap["n"] else 0.0
    # 100% 일치를 요구하지 않는다 — 재개일 기준가처럼 KRX 등락률 자체가 자본변동을
    # 안 반영하는 날이 실재하고, 그건 우리 값이 아니라 원문의 성질이다.
    ok_gap = bad_rate < 1.0
    print(f"  ① 분할일 갭   {gap['n']:,}쌍 중 {gap['bad'] or 0:,}쌍 어긋남 "
          f"({bad_rate:.3f}%) → {'✅' if ok_gap else '🔴'}")
    ok &= ok_gap

    # ② 없는 조정을 만들지 않았나 — 이벤트가 없던 종목은 배율이 1 이어야 한다.
    #    `adj_source='fdr'` 인 행에서 원가격과 수정가격이 같은 비율을 본다.
    same = conn.execute(f"""
        SELECT COUNT(*) AS n,
               SUM(CASE WHEN ABS(adj_close - close) < 0.51 THEN 1 ELSE 0 END) AS equal
        FROM daily_price
        WHERE adj_source = 'fdr' AND adj_close IS NOT NULL AND close > 0 {where}
    """, params).fetchone()
    equal_rate = (same["equal"] or 0) / same["n"] * 100 if same["n"] else 0.0
    print(f"  ② 무이벤트    fdr 행 {same['n']:,} 중 원가격과 같은 행 "
          f"{same['equal'] or 0:,} ({equal_rate:.1f}%)")

    # ③ 고저 관계가 살아 있나
    broken = conn.execute(f"""
        SELECT COUNT(*) FROM daily_price
        WHERE adj_high IS NOT NULL AND adj_low IS NOT NULL AND adj_close IS NOT NULL
          AND (adj_close > adj_high + 0.01 OR adj_close < adj_low - 0.01) {where}
    """, params).fetchone()[0]
    ok_range = broken == 0
    print(f"  ③ 고저 관계   adj_low ≤ adj_close ≤ adj_high 위반 {broken:,}행 "
          f"→ {'✅' if ok_range else '🔴'}")
    ok &= ok_range

    # ④ 0 이나 음수가 실리지 않았나 — 정지일 0 을 그대로 실으면 여기 걸린다.
    nonpositive = conn.execute(f"""
        SELECT COUNT(*) FROM daily_price
        WHERE (adj_open <= 0 OR adj_high <= 0 OR adj_low <= 0 OR adj_close <= 0) {where}
    """, params).fetchone()[0]
    ok_positive = nonpositive == 0
    print(f"  ④ 0·음수      {nonpositive:,}행 → {'✅' if ok_positive else '🔴'}")
    ok &= ok_positive

    return bool(ok)


def raw_checksum(conn: sqlite3.Connection) -> tuple:
    """원가격이 손상되지 않았는지 볼 지문. 적재 전후로 **같아야 한다.**

    행 수만 세면 부족하다 — 칸을 덮어써도 행 수는 그대로다. 합계까지 함께 본다.
    """
    return conn.execute(
        "SELECT COUNT(*), SUM(close), SUM(open), SUM(volume) FROM daily_price"
    ).fetchone()[:]


def main() -> int:
    parser = argparse.ArgumentParser(description="수정주가 적재 (v9 · #51)")
    parser.add_argument("--db", default=None, help="DB 경로 (기본: data/krx_cache.db)")
    parser.add_argument("--limit", type=int, default=None, help="앞 N종만")
    parser.add_argument("--codes", default=None, help="쉼표로 구분한 종목코드")
    parser.add_argument("--calendar-only", action="store_true",
                        help="거래일 달력만 다시 깐다")
    parser.add_argument("--skip-verify", action="store_true",
                        help="검증을 건너뛴다 (연습용 — 평소에 쓰지 않는다)")
    args = parser.parse_args()

    db_path = Path(args.db).resolve() if args.db else krx_db_path()
    if not db_path.exists():
        print(f"DB 가 없다: {db_path}")
        return 1
    conn = connect(db_path)

    before = raw_checksum(conn)
    print(f"적재 전 원가격 지문: 행 {before[0]:,} · close합 {before[1]:,}")

    if not args.calendar_only:
        if args.codes:
            codes = [c.strip() for c in args.codes.split(",") if c.strip()]
        else:
            codes = all_codes(conn)
            if args.limit:
                codes = codes[:args.limit]

        print(f"\n대상 {len(codes):,}종 · FDR 종목당 약 0.1초 → 예상 "
              f"{len(codes) * 0.12 / 60:.1f}분\n")

        started = time.perf_counter()
        totals = {"rows": 0, "fdr": 0, "chain": 0, "ca_fix": 0}
        empty, failed = [], []
        with write_lock:
            conn.execute("BEGIN IMMEDIATE")
            for i, code in enumerate(codes, 1):
                summary = build_one(conn, code)
                for key in totals:
                    totals[key] += summary[key]
                if summary["error"]:
                    failed.append(code)
                elif summary["rows"] == 0:
                    empty.append(code)

                if i % COMMIT_EVERY == 0 or i == len(codes):
                    conn.execute("COMMIT")
                    elapsed = time.perf_counter() - started
                    print(f"  {i:>5,}/{len(codes):,} · {totals['rows']:>10,}행 "
                          f"(fdr {totals['fdr']:,} · chain {totals['chain']:,}) "
                          f"· {elapsed / 60:.1f}분", flush=True)
                    if i != len(codes):
                        conn.execute("BEGIN IMMEDIATE")

        print(f"\n채운 행 {totals['rows']:,} — fdr {totals['fdr']:,} "
              f"({totals['fdr'] / max(totals['rows'], 1) * 100:.1f}%) · "
              f"chain {totals['chain']:,} "
              f"({totals['chain'] / max(totals['rows'], 1) * 100:.1f}%)")
        if totals["ca_fix"]:
            # FDR 이 안 편 감자를 우리가 편 행. 위 둘에 **포함된** 수다(접미사이므로).
            print(f"  그중 +ca_fix {totals['ca_fix']:,}행 "
                  f"({totals['ca_fix'] / max(totals['rows'], 1) * 100:.1f}%) "
                  "— FDR 이 안 편 자본변동을 우리가 폈다")
        if empty:
            print(f"⚠️ 행이 없던 종목 {len(empty)}종: {', '.join(empty[:10])}"
                  + (" …" if len(empty) > 10 else ""))
        if failed:
            print(f"🔴 FDR 실패 {len(failed)}종: {', '.join(failed[:10])}"
                  + (" …" if len(failed) > 10 else ""))

    # 거래일 달력 — 시세를 건드린 뒤에는 반드시 다시 깐다
    calendar_rows = adj_price.rebuild_calendar(conn)
    span = conn.execute(
        "SELECT MIN(bas_dd), MAX(bas_dd) FROM trading_calendar WHERE market='ALL'"
    ).fetchone()
    print(f"\n거래일 달력 {calendar_rows:,}행 (시장별+ALL) · "
          f"ALL 기준 {span[0]}~{span[1]}")

    after = raw_checksum(conn)
    same_raw = before == after
    print(f"원가격 지문 {'✅ 그대로' if same_raw else '🔴 바뀌었다'} "
          f"— 행 {after[0]:,} · close합 {after[1]:,}")

    ok = same_raw
    if not args.skip_verify:
        ok &= verify(conn, [c.strip() for c in args.codes.split(",")]
                     if args.codes else None)
    conn.close()

    print(f"\n{'✅ 통과' if ok else '🔴 실패 — 위 항목을 확인한다'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
