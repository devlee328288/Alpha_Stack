"""pit 반출본이 DB 와 **값 단위로** 같은지 대조한다 — 행 수 비교가 아니다.

    python scripts/_verify_pit_export.py --db <경로> --outbox data/outbox/dart_pit_<날짜>

행 수 검증은 덮어쓰기를 못 잡는다(실측 교훈). 그래서 여기서는 파일과 DB 를
둘 다 전 컬럼으로 정렬한 뒤 `DataFrame.equals` 로 **모든 칸**을 비교한다.

검증식 자체가 틀렸을 가능성을 배제하기 위해 음성 대조군 두 벌을 함께 돌린다.
음성이 False 로 떨어지지 않으면 양성 결과도 믿을 수 없다.

  음성 1  값 1건 변조 — 첫 파일 사본의 금액 칸 하나에 +1 → equals 가 False 여야 한다
  음성 2  어긋난 쌍 — 2015 파일 vs 2016 DB 묶음 → False 여야 한다
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from scripts.export_dart_dataset import REPORT_NAMES  # noqa: E402

#: 파일명에 쓰인 보고서 이름 → 코드 (반출 스크립트의 매핑을 그대로 뒤집는다)
NAME_TO_CODE = {name: code for code, name in REPORT_NAMES.items()}


def canonical(frame: pd.DataFrame) -> pd.DataFrame:
    """비교 가능한 형태로 만든다 — 전 컬럼 정렬 후 인덱스를 버린다.

    반출 스크립트가 ORDER BY 없이 읽었으므로 행 순서는 보장이 없다.
    기본키 일부가 아니라 **전 컬럼**으로 정렬해야 어떤 칸이 달라도 자리가 갈린다.
    """
    return frame.sort_values(by=list(frame.columns), kind="mergesort").reset_index(drop=True)


def load_db_group(conn: sqlite3.Connection, year: int, code: str) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT * FROM dart_financial WHERE bsns_year = ? AND reprt_code = ?",
        conn, params=(year, code),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="pit 반출본을 DB 와 값 단위로 대조한다")
    parser.add_argument("--db", default="data/krx_cache.db")
    parser.add_argument("--outbox", required=True)
    args = parser.parse_args()

    db_path = Path(args.db).resolve()
    if not db_path.exists():
        print(f"DB 가 없다: {db_path}")
        return 1

    pit_dir = Path(args.outbox) / "pit"
    files = sorted(pit_dir.glob("*.parquet"))
    if not files:
        print(f"pit parquet 이 없다: {pit_dir}")
        return 1

    # 수집 세션이 쓰는 중일 수 있다 — 반드시 읽기 전용으로 연다
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)

    # ── 양성: 파일별 전량 값 대조 ────────────────────────────────────
    print(f"── 값 단위 대조 — {len(files)}개 파일 ──")
    failures = 0
    first_file_frame = None
    for path in files:
        year_s, name = path.stem.split("_", 1)
        code = NAME_TO_CODE.get(name)
        if code is None:
            print(f"  🔴 {path.name}: 보고서 이름을 모른다 ({name})")
            failures += 1
            continue
        left = canonical(pd.read_parquet(path))
        right = canonical(load_db_group(conn, int(year_s), code))
        same = left.equals(right)
        print(f"  {'✅' if same else '🔴'} {path.name:28s} "
              f"파일 {len(left):>7,}행 vs DB {len(right):>7,}행 → equals={same}")
        if not same:
            failures += 1
        if first_file_frame is None:
            first_file_frame = left

    # ── 음성 1: 값 1건 변조 ─────────────────────────────────────────
    mutated = first_file_frame.copy()
    numeric_cols = [c for c in mutated.columns if mutated[c].dtype.kind in "if"]
    target_col = numeric_cols[0] if numeric_cols else mutated.columns[-1]
    if mutated[target_col].dtype.kind in "if":
        mutated.loc[0, target_col] = (mutated.loc[0, target_col] or 0) + 1
    else:
        mutated.loc[0, target_col] = str(mutated.loc[0, target_col]) + "_변조"
    neg1_ok = not mutated.equals(first_file_frame)
    print("\n── 음성 대조군 ──")
    print(f"  {'✅' if neg1_ok else '🔴'} 값 1건 변조({target_col}) "
          f"→ equals=False 가 나온다: {neg1_ok}")

    # ── 음성 2: 어긋난 쌍 (첫 파일 vs 다음 연도 DB) ──────────────────
    year0, name0 = files[0].stem.split("_", 1)
    other = canonical(load_db_group(conn, int(year0) + 1, NAME_TO_CODE[name0]))
    conn.close()
    neg2_ok = not first_file_frame.equals(other)
    print(f"  {'✅' if neg2_ok else '🔴'} {year0} 파일 vs {int(year0) + 1} DB "
          f"→ equals=False 가 나온다: {neg2_ok}")

    ok = failures == 0 and neg1_ok and neg2_ok
    print("\n✅ 전부 통과" if ok else
          f"\n🔴 실패 — 파일 대조 실패 {failures}건, 음성1 {neg1_ok}, 음성2 {neg2_ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
