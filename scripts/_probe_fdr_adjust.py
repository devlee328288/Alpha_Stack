"""FDR(FinanceDataReader) 수정주가가 우리 원주가의 액면분할 문제를 실제로 고치는지 잰다.

    python scripts/_probe_fdr_adjust.py --db <경로>

이슈 #51: `daily_price.close` 는 분할 미조정 원가격이라 삼성전자 2018-05-04 가
-98.04% 로 읽힌다. FDR(MIT) 수정주가로 갈아타기 전에 세 가지를 실측한다.

  1. 분할일 갭 — FDR 수정주가로 계산한 등락률이 KRX `change_rate`(이미 조정됨)와
     맞는가. 원주가 갭(-98%)이 사라지는가.
  2. 조정계수 역산 — 우리 원주가 ÷ FDR 종가 비율이 분할 전후로 갈리는 배수가
     실제 분할 비율(50:1, 5:1)과 맞는가.
  3. 음성 대조군 — 분할 이력이 없는 종목에서는 비율이 1.0 으로 일정해야 한다.
     (여기서도 배수가 잡히면 검증식 자체가 틀린 것이다)

분할일은 문서가 아니라 **DB 의 주식수 점프로 재확인**한다 — 주식수 배율과
가격 배율이 서로 역수면 분할·병합이다 (#51 과 같은 판정식).
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import FinanceDataReader as fdr  # noqa: E402
import pandas as pd  # noqa: E402

#: 분할 표본 — (코드, 이름, 분할일 YYYYMMDD, 공지된 비율)
SPLITS = [
    ("005930", "삼성전자", "20180504", 50.0),
    ("035420", "NAVER", "20181012", 5.0),
    ("035720", "카카오", "20210415", 5.0),
]

#: 음성 대조군 — 분할 이력이 없는 종목. 여기서 배수가 잡히면 검증식이 틀린 것.
CONTROLS = [
    ("000660", "SK하이닉스"),
    ("005380", "현대차"),
]


def load_ours(conn: sqlite3.Connection, code: str) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT bas_dd, close, change_rate, listed_shares FROM daily_price "
        "WHERE code = ? ORDER BY bas_dd",
        conn, params=(code,),
    )
    df["date"] = pd.to_datetime(df["bas_dd"], format="%Y%m%d")
    return df.set_index("date")


def load_fdr(code: str, start: str) -> pd.DataFrame:
    df = fdr.DataReader(code, start)
    df.index = pd.to_datetime(df.index)
    return df


def check_split(conn: sqlite3.Connection, code: str, name: str,
                split_dd: str, expected: float) -> bool:
    ours = load_ours(conn, code)
    theirs = load_fdr(code, "2015-01-01")
    both = ours.join(theirs[["Close", "Change"]], how="inner")
    both = both[both["close"] > 0]

    day = pd.to_datetime(split_dd, format="%Y%m%d")
    pos = both.index.get_loc(day)
    prev = both.iloc[pos - 1]
    cur = both.iloc[pos]

    # 분할일을 DB 주식수 점프로 재확인 — 주식수 배율 × 가격 배율 ≈ 1 이어야 분할이다
    share_ratio = cur["listed_shares"] / prev["listed_shares"]
    price_ratio = cur["close"] / prev["close"]
    is_split_day = abs(share_ratio * price_ratio - 1) < 0.10 and share_ratio > 1.5
    raw_gap = (price_ratio - 1) * 100

    # 1. FDR 수정주가 갭 vs KRX change_rate (이미 조정된 값)
    fdr_gap = (cur["Close"] / prev["Close"] - 1) * 100
    krx_rate = cur["change_rate"]
    gap_close = abs(fdr_gap - krx_rate) < 0.15   # 반올림 여유

    # 2. 조정계수 역산 — 분할 전날과 분할일의 (원주가/FDR) 비율이 갈리는 배수
    factor = (prev["close"] / prev["Close"]) / (cur["close"] / cur["Close"])
    factor_ok = abs(factor / expected - 1) < 0.01

    print(f"\n── {name}({code}) 분할일 {split_dd} ──")
    print(f"  DB 재확인   주식수 {share_ratio:.3f}배 × 가격 {price_ratio:.5f}배 "
          f"= {share_ratio * price_ratio:.3f} → 분할일 맞음: {is_split_day}")
    print(f"  원주가 갭   {raw_gap:+.2f}%  (이걸 수익률로 쓰면 안 된다)")
    print(f"  FDR 갭     {fdr_gap:+.2f}%  vs KRX change_rate {krx_rate:+.2f}% "
          f"→ {'✅ 일치' if gap_close else '🔴 불일치'}")
    print(f"  조정계수    {factor:.3f}  vs 공지 비율 {expected:.0f}:1 "
          f"→ {'✅ 일치' if factor_ok else '🔴 불일치'}")
    return is_split_day and gap_close and factor_ok


def check_control(conn: sqlite3.Connection, code: str, name: str) -> bool:
    ours = load_ours(conn, code)
    theirs = load_fdr(code, "2015-01-01")
    both = ours.join(theirs[["Close"]], how="inner")
    both = both[(both["close"] > 0) & (both["Close"] > 0)]

    # 분할 판정식으로 이벤트가 없음을 먼저 확인 (있다면 대조군 자격이 없다)
    share_r = both["listed_shares"] / both["listed_shares"].shift(1)
    price_r = both["close"] / both["close"].shift(1)
    events = both[((share_r * price_r - 1).abs() < 0.10)
                  & ((share_r > 1.5) | (share_r < 1 / 1.5))]

    ratio = both["close"] / both["Close"]
    print(f"\n── {name}({code}) 음성 대조군 · {len(both):,}일 ──")
    print(f"  분할 판정식에 걸린 날: {len(events)}일 (0이어야 대조군 자격)")
    print(f"  원주가/FDR 비율  최소 {ratio.min():.4f} · 최대 {ratio.max():.4f} "
          f"(1.0 근처로 일정해야 한다)")
    ok = len(events) == 0 and ratio.min() > 0.99 and ratio.max() < 1.01
    print("  → ✅ 비율 일정 — 검증식이 분할을 없는 곳에서 만들지 않는다" if ok
          else "  → 🔴 대조군에서 배수가 잡혔다")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="FDR 수정주가 대조 실측")
    parser.add_argument("--db", default="data/krx_cache.db")
    args = parser.parse_args()

    db_path = Path(args.db).resolve()
    if not db_path.exists():
        print(f"DB 가 없다: {db_path}")
        return 1
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)

    results = []
    for code, name, split_dd, expected in SPLITS:
        results.append(check_split(conn, code, name, split_dd, expected))
    for code, name in CONTROLS:
        results.append(check_control(conn, code, name))
    conn.close()

    print(f"\n{'✅ 전부 통과' if all(results) else '🔴 실패 있음'} "
          f"— 분할 {sum(results[:len(SPLITS)])}/{len(SPLITS)} · "
          f"대조군 {sum(results[len(SPLITS):])}/{len(CONTROLS)}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
