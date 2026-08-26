"""보존된 응답 원문으로 정규화를 다시 돌린다 — 네트워크를 한 번도 타지 않고.

왜 필요한가
----------
정규화는 틀린다. 필드 이름을 잘못 매핑하고, 숫자 파싱이 어떤 값에서만 깨지고, 그때는
안 담은 칸이 나중에 필요해진다. 이 저장소에서만 이미 여러 번 겪었다.

원문을 남겨 두지 않았다면 고치는 방법이 **다시 받는 것뿐**이다. 16년치를 다시 받는 것은
며칠과 하루 한도를 통째로 쓰는 일이고, 출처가 그사이에 과거 값을 정정했다면 **같은
자료를 다시 받을 수조차 없다.**

사용법
------
    python scripts/renormalize.py --dry-run            # 무엇이 달라지는지만 본다
    python scripts/renormalize.py                      # 실제로 다시 채운다
    python scripts/renormalize.py --prefix idx/        # 지수만
    python scripts/renormalize.py --prefix sto/        # 종목만

**`--dry-run` 을 먼저 돌린다.** 달라지는 행이 0 이면 정규화가 그대로라는 뜻이고,
0 이 아니면 무엇이 어떻게 바뀌는지 표본을 보여 준다. 그걸 보고 나서 실행한다.

⚠️ 원문이 없는 구간은 다시 정규화할 수 없다. 2026-08-26 이전에 받은 자료가 그렇다 —
   원문 보존이 그날 붙었기 때문이다. 그 구간은 이 스크립트가 **건너뛰고 몇 건인지 알려
   준다.** 조용히 넘어가면 "다 고쳤다"고 믿게 되는데 실제로는 옛 구간이 그대로다.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 이 스크립트는 scripts/ 안에 있어서 파이썬이 프로젝트 루트를 모른다.
# (parents[0]=scripts, parents[1]=프로젝트 루트)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json  # noqa: E402

from common import raw_store  # noqa: E402  (경로 설정 후에 import)
from ingest.clients import krx_data as api  # noqa: E402
from ingest.store import krx_index  # noqa: E402


def human(n: int) -> str:
    return f"{n:,}"


def _parse_target(target: str) -> Optional[Tuple[str, str]]:
    """`idx/kospi_dd_trd/20260826` → `("idx/kospi_dd_trd", "20260826")`.

    엔드포인트 경로가 **어느 정규화 함수를 다시 돌려야 하는지**를 알려 준다.
    날짜만으로는 종목인지 지수인지 알 수 없다.
    """
    head, _, bas_dd = target.rpartition("/")
    if not head or len(bas_dd) != 8 or not bas_dd.isdigit():
        return None
    return head, bas_dd


def _renormalize_one(raw: Dict) -> Optional[Tuple[str, str, List[Dict]]]:
    """원문 하나를 다시 정규화한다. 다룰 수 없는 대상이면 `None`."""
    parsed = _parse_target(raw["target"])
    if parsed is None:
        return None
    path, bas_dd = parsed

    # 보존된 인코딩으로 되돌린다. UTF-8 로 가정하지 않는다 — euc-kr 로 오는 출처가 실재하고,
    # 잘못 디코딩하면 예외 없이 글자만 깨진다.
    payload = json.loads(raw["body"].decode(raw["encoding"] or "utf-8"))
    rows = payload.get("OutBlock_1", [])

    if path.startswith("idx/"):
        market = "KOSPI" if "kospi" in path else "KOSDAQ"
        return "index", market, [api.normalize_index_row(row, bas_dd) for row in rows]
    if path.startswith("sto/"):
        market = next((name for name, (p, _) in api.MARKET_APIS.items() if p == path), "")
        return "stock", market, [api.normalize_row(row, bas_dd, market) for row in rows]
    return None


def _current_index_rows(bas_dd: str) -> Dict[str, Dict]:
    """지금 표에 들어 있는 그 날짜의 지수 행들. 비교용이다."""
    from ingest.store.krx_store import connect

    with connect() as conn:
        rows = conn.execute(
            f"SELECT {','.join(krx_index.COLUMNS)} FROM index_price WHERE bas_dd=?",
            (bas_dd,),
        ).fetchall()
    # strict=True — 컬럼 목록과 SELECT 결과의 개수가 어긋나면 조용히 잘리는 대신 터진다.
    return {row[1]: dict(zip(krx_index.COLUMNS, row, strict=True)) for row in rows}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="보존된 응답 원문으로 정규화를 다시 돌린다 (네트워크를 타지 않는다)")
    parser.add_argument("--source", default="krx", help="출처 이름 (기본: krx)")
    parser.add_argument("--prefix", default="",
                        help="대상 앞머리로 좁힌다 (예: idx/ · sto/stk_bydd_trd)")
    parser.add_argument("--dry-run", action="store_true",
                        help="바뀌는 것만 보여주고 표는 건드리지 않는다")
    parser.add_argument("--limit", type=int, default=0,
                        help="이만큼만 처리한다 (0 = 전부). 먼저 소량으로 확인할 때 쓴다")
    args = parser.parse_args()

    보존현황 = raw_store.stats().get(args.source)
    if not 보존현황:
        print(f"'{args.source}' 의 보존된 원문이 없습니다.")
        print("  왜: 원문 보존은 2026-08-26 에 붙었습니다. 그 전에 받은 자료는 원문이 없습니다.")
        print("  할 일: 앞으로 받는 것부터 쌓입니다. KEEP_RAW=off 로 꺼 두지 않았는지 확인하세요.")
        return 1

    print(f"── 보존된 원문 — {args.source} ──")
    print(f"  응답 {human(보존현황['responses'])}건 · 대상 {human(보존현황['targets'])}개")
    print(f"  원문 {human(보존현황['raw_bytes'])}B → "
          f"저장 {human(보존현황['stored_bytes'])}B (압축 후 {보존현황['ratio']:.1%})")
    print(f"  기간 {보존현황['first_at']} ~ {보존현황['last_at']}")
    print()

    처리 = 다룰수없음 = 바뀜 = 0
    바뀐표본: List[str] = []

    for raw in raw_store.iter_latest(args.source, prefix=args.prefix):
        if args.limit and 처리 >= args.limit:
            break
        결과 = _renormalize_one(raw)
        if 결과 is None:
            다룰수없음 += 1
            continue
        종류, market, items = 결과
        처리 += 1

        if 종류 != "index":
            # 종목 재정규화는 아직 안 붙였다. **조용히 성공한 척하지 않는다** —
            # "다 고쳤다"고 믿게 만드는 것이 안 고친 것보다 나쁘다.
            다룰수없음 += 1
            처리 -= 1
            continue

        _, bas_dd = _parse_target(raw["target"])
        기존 = _current_index_rows(bas_dd)
        for item in items:
            이름 = item.get("index_name")
            옛행 = 기존.get(이름)
            새행 = {col: item.get(col) for col in krx_index.COLUMNS[1:]}
            if 옛행 is None or any(옛행.get(col) != 새행.get(col) for col in 새행):
                바뀜 += 1
                if len(바뀐표본) < 10:
                    바뀐표본.append(f"{bas_dd} {이름}: {옛행} → {새행}")

        if not args.dry_run:
            krx_index._save(bas_dd, market, items)

    print(f"처리한 대상 {human(처리)}개 · 다룰 수 없어 건너뛴 것 {human(다룰수없음)}개")
    print(f"달라지는 행 {human(바뀜)}개")
    if 바뀐표본:
        print("  표본:")
        for line in 바뀐표본:
            print(f"    {line[:160]}")
    if args.dry_run:
        print()
        print("※ --dry-run 이라 표를 건드리지 않았습니다.")
        print("   위 표본을 확인한 뒤 옵션 없이 다시 실행하세요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
