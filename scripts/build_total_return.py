"""총수익 축을 전 종목에 채운다 (마이그레이션 v15).

    python scripts/build_total_return.py                 # 전 종목
    python scripts/build_total_return.py --limit 20      # 앞 20종만 (연습용)
    python scripts/build_total_return.py --codes 005930,000660

`adj_close` 는 액면분할·무상증자·주식배당까지만 편 값이라 **현금배당이 빠져 있다** —
CRSP 로 치면 `RETX` 쪽이다. 배당까지 재투자한 축을 `adj_close_tr` 에 채우고, 그날
더해진 배당을 `adj_dividend` 에, 원본 배당금이 성하지 않은 자리를
`is_dividend_suspect` 에 남긴다. 계산식과 근거는 `supply/total_return.py` 에 있다.

무엇을 검증하나
---------------
적재는 **검증을 통과한 뒤에** 커밋한다. 행 수만 세면 안 된다 — 칸을 덮어써서 값이
사라져도 행 수는 그대로이기 때문이다. 다섯 가지를 본다.

  1. **배당이 없는 종목은 두 축의 비가 상수인가** — 배당이 한 번도 없었다면
     `adj_close_tr / adj_close` 는 첫날부터 끝까지 1 이어야 한다. 배당이 없는데
     총수익이 갈라지면 계산이 새는 것이다.
  2. **배당락일에는 총수익이 가격보다 더 오르는가** — 배당이 붙은 날의 총수익
     수익률에서 가격 수익률을 빼면 `배당/직전종가` 와 같아야 한다.
  3. **조정배당이 가격과 같은 배율을 탔나** — `adj_dividend / 원배당` 이 그날의
     `adj_close / close` 와 같나. 배율을 안 태우면 분할이 있던 종목에서 어긋난다.
  4. **원가격이 그대로인가** — `close`·`adj_close` 합계가 적재 전후로 같나.
  5. **누적에 음수나 무한대가 없나** — 총수익은 언제나 0보다 크다.
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.paths import krx_db_path  # noqa: E402
from ingest.store.sqlite_db import write_lock  # noqa: E402
from supply import total_return as TR  # noqa: E402

#: 몇 종목마다 커밋할 것인가. `build_adj_prices.py` 와 같은 이유로 100 이다 —
#: 전부 한 트랜잭션에 묶으면 그동안 쓰기 잠금을 쥐고, 종목마다 커밋하면 3,678번의
#: fsync 가 그대로 비용이 된다.
COMMIT_EVERY = 100

#: 검증에서 허용할 상대오차. 부동소수 누적곱이라 정확히 떨어지지 않는다.
TOLERANCE = 1e-9

#: 🔴 `WHERE code=? AND bas_dd=?` 로 쓰지 않는다. `daily_price` 의 기본키는
#: `(bas_dd, code)` 순서라 종목을 앞세운 조건은 그 인덱스를 제대로 타지 못하고,
#: 923만 행을 종목별로 갱신하면 그 차이가 시간으로 그대로 나온다(실측: 종목당 약 2초
#: → 전체 두 시간). 읽을 때 `rowid` 를 함께 들고 와 그것으로 짚는다.
UPDATE_SQL = (
    "UPDATE daily_price SET adj_dividend=?, adj_close_tr=?, is_dividend_suspect=? "
    "WHERE rowid=?"
)


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=60)
    conn.execute("PRAGMA busy_timeout=60000")
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def pick_codes(conn: sqlite3.Connection, *, limit: Optional[int],
               codes: Optional[str]) -> List[str]:
    """채울 종목을 고른다. 배당이 없는 종목도 채운다 — 팀원이 표를 읽을 때 어떤 종목은
    총수익 칸이 비어 있으면 "빠뜨린 것인가" 를 매번 되물어야 하기 때문이다."""
    if codes:
        return [c.strip() for c in codes.split(",") if c.strip()]
    rows = conn.execute(
        "SELECT DISTINCT code FROM daily_price ORDER BY code").fetchall()
    골라진 = [r[0] for r in rows]
    return 골라진[:limit] if limit else 골라진


def series_for(conn: sqlite3.Connection, code: str
               ) -> List[Tuple[int, str, Optional[float], Optional[float]]]:
    """한 종목의 `(rowid, 날짜, 종가, 수정종가)` 를 날짜 오름차순으로.

    `rowid` 를 함께 가져오는 이유는 `UPDATE_SQL` 주석에 있다 — 갱신을 그것으로 짚는다.
    """
    return conn.execute(
        "SELECT rowid, bas_dd, close, adj_close FROM daily_price "
        "WHERE code=? ORDER BY bas_dd", (code,)).fetchall()


def verify_code(code: str,
                rows: List[Tuple[str, Optional[float], Optional[float]]],
                dividends: Dict[str, Dict],
                결과: List[Tuple[str, Optional[float], Optional[float], Optional[int]]],
                ) -> List[str]:
    """한 종목의 계산을 검증한다. 어긋난 것을 사람이 읽을 문장으로 돌려준다."""
    문제: List[str] = []
    값 = {bas_dd: (div, tr, sus) for bas_dd, div, tr, sus in 결과}

    # ⑤ 총수익은 언제나 0보다 크다.
    for bas_dd, _, tr, _ in 결과:
        if tr is None:
            continue
        if not math.isfinite(tr) or tr <= 0:
            문제.append(f"{code} {bas_dd}: 총수익이 {tr} 이다")

    붙은배당 = {d for d in dividends if d in 값 and 값[d][0] is not None}

    # ① 배당이 한 번도 안 붙었으면 두 축의 비가 처음부터 끝까지 1 이다.
    if not 붙은배당:
        for bas_dd, _close, adj_close in rows:
            tr = 값.get(bas_dd, (None, None, None))[1]
            if tr is None or adj_close is None or adj_close <= 0:
                continue
            if abs(tr / adj_close - 1.0) > 1e-6:
                문제.append(
                    f"{code} {bas_dd}: 배당이 없는데 총수익이 가격과 갈라졌다 "
                    f"(tr/adj_close={tr / adj_close:.9f})")
                break
        return 문제

    # ②③ 배당이 붙은 날을 본다.
    직전 = {}
    가격 = {}
    앞 = None
    for bas_dd, close, adj_close in rows:
        직전[bas_dd] = 앞
        가격[bas_dd] = (close, adj_close)
        if adj_close is not None:
            앞 = (bas_dd, adj_close, close)

    for bas_dd in sorted(붙은배당):
        조정배당, tr, _ = 값[bas_dd]
        앞값 = 직전.get(bas_dd)
        if 앞값 is None or 조정배당 is None:
            continue
        앞날, 앞수정, _앞종가 = 앞값
        앞tr = 값.get(앞날, (None, None, None))[1]
        종가, 수정종가 = 가격.get(bas_dd, (None, None))
        if 수정종가 is None or not 종가 or 앞tr is None or tr is None:
            continue

        # ③ 조정배당이 가격과 같은 배율을 탔나.
        원배당 = dividends[bas_dd]["amount"]
        기대배율 = 수정종가 / 종가
        실제배율 = 조정배당 / 원배당
        if abs(실제배율 - 기대배율) > TOLERANCE * max(1.0, abs(기대배율)):
            문제.append(
                f"{code} {bas_dd}: 배당 배율이 가격과 다르다 "
                f"(배당 {실제배율:.9f} vs 가격 {기대배율:.9f})")

        # ② 총수익이 가격보다 배당만큼 더 올랐나.
        if 앞수정 > 0:
            가격수익 = 수정종가 / 앞수정 - 1.0
            총수익 = tr / 앞tr - 1.0
            기대차 = 조정배당 / 앞수정
            if abs((총수익 - 가격수익) - 기대차) > 1e-9 * max(1.0, abs(기대차)):
                문제.append(
                    f"{code} {bas_dd}: 배당만큼 더 오르지 않았다 "
                    f"(차이 {총수익 - 가격수익:.9e} vs 기대 {기대차:.9e})")
    return 문제


def main() -> int:
    parser = argparse.ArgumentParser(description="총수익 축 적재 (v15)")
    parser.add_argument("--db", default=None, help="DB 경로 (기본: data/krx_cache.db)")
    parser.add_argument("--limit", type=int, default=None, help="앞 N종만")
    parser.add_argument("--codes", default=None, help="쉼표로 구분한 종목코드")
    parser.add_argument("--skip-verify", action="store_true",
                        help="검증을 건너뛴다 (권하지 않는다)")
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else krx_db_path()
    conn = connect(db_path)

    시작 = time.time()
    배당표 = TR.load_dividends(conn)
    종목별배당 = TR.dividends_by_code(배당표)
    print(f"배당 {len(배당표):,}건 · {len(종목별배당):,}종목 "
          f"({time.time() - 시작:.1f}초)")

    codes = pick_codes(conn, limit=args.limit, codes=args.codes)
    print(f"대상 {len(codes):,}종목")

    # ④ 원가격 보존 — 적재 전 합계를 재 둔다.
    전_close, 전_adj = conn.execute(
        "SELECT SUM(close), SUM(adj_close) FROM daily_price").fetchone()

    문제전체: List[str] = []
    쓴행 = 배당붙은행 = 의심행 = 0
    처리 = 0

    with write_lock:
        for i, code in enumerate(codes, 1):
            읽은행 = series_for(conn, code)
            if not 읽은행:
                continue
            rowids = [r[0] for r in 읽은행]
            rows = [(r[1], r[2], r[3]) for r in 읽은행]
            배당 = 종목별배당.get(code, {})
            결과 = list(TR.total_return_series(rows, 배당))

            if not args.skip_verify:
                문제 = verify_code(code, rows, 배당, 결과)
                if 문제:
                    문제전체.extend(문제)
                    if len(문제전체) > 40:
                        conn.rollback()
                        print("\n🔴 검증에서 걸렸다 — 아무것도 커밋하지 않았다:")
                        for m in 문제전체[:40]:
                            print("   ", m)
                        conn.close()
                        return 1

            conn.executemany(UPDATE_SQL, [
                (div, tr, sus, rid)
                for rid, (_bas_dd, div, tr, sus) in zip(rowids, 결과, strict=True)])
            쓴행 += len(결과)
            배당붙은행 += sum(1 for _, div, _, _ in 결과 if div is not None)
            의심행 += sum(1 for _, _, _, sus in 결과 if sus)
            처리 += 1

            if i % COMMIT_EVERY == 0:
                conn.commit()
                print(f"  {i:,}/{len(codes):,} · {time.time() - 시작:.0f}초", flush=True)

        conn.commit()

    if 문제전체:
        conn.rollback()
        print("\n🔴 검증에서 걸렸다 — 되돌렸다:")
        for m in 문제전체[:40]:
            print("   ", m)
        conn.close()
        return 1

    # ④ 원가격이 그대로인가.
    후_close, 후_adj = conn.execute(
        "SELECT SUM(close), SUM(adj_close) FROM daily_price").fetchone()
    if 전_close != 후_close or 전_adj != 후_adj:
        print(f"\n🔴 원가격이 바뀌었다 — close {전_close} → {후_close} · "
              f"adj_close {전_adj} → {후_adj}")
        conn.close()
        return 1

    print(f"\n✅ {처리:,}종목 · {쓴행:,}행")
    print(f"   배당이 붙은 행 {배당붙은행:,} · 단위 오류로 뺀 행 {의심행:,}")
    print("   원가격 보존 확인 (close·adj_close 합계 그대로)")
    print(f"   {time.time() - 시작:.0f}초")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
