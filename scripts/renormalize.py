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

## 🔴 수집 시각을 건드리지 않는다

재정규화는 **자료를 새로 받은 것이 아니라 같은 원문을 다시 읽은 것**이다. 그러므로
`fetched_at` · `last_success_at` 이 움직이면 안 된다. 그 시각은 *"우리가 이 사실을
언제부터 알 수 있었나"* 의 근거이고, 오늘로 덮이면 **미래참조 방지의 바닥이 무너진다.**

예전 판이 `krx_index._save` 를 그대로 불러서 실제로 그 시각을 덮고 있었다. 이제
가격 표에만 쓰는 `save_renormalized` 를 쓴다.

## ⚠️ 원문이 없는 구간은 다시 정규화할 수 없다

원문 보존은 2026-08-26 에 붙었다. 그 전에 받은 자료는 원문이 없다. 이 스크립트는
그 구간을 **건너뛰고 몇 건인지 숫자로 알려 주며, 남아 있으면 0 이 아닌 종료코드**를
돌려준다. 조용히 넘어가면 *"다 고쳤다"* 고 믿게 되는데 실제로는 옛 구간이 그대로다.
"""

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# 이 스크립트는 scripts/ 안에 있어서 파이썬이 프로젝트 루트를 모른다.
# (parents[0]=scripts, parents[1]=프로젝트 루트)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json  # noqa: E402

from common import raw_store  # noqa: E402  (경로 설정 후에 import)
from ingest.clients import krx_data as api  # noqa: E402
from ingest.store import krx_index, krx_store  # noqa: E402


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


# ──────────────────────────────────────────────────────────────────────────────
# 종류별 분기표 — 새 출처를 붙일 때 여기 한 줄만 더한다
#
# 예전 판은 `if 종류 != "index": 건너뛴다` 로 되어 있었다. 그 모양이면 출처를 더할
# 때마다 조건문이 늘고, 늘리는 걸 잊으면 **조용히 건너뛴다.**
# ──────────────────────────────────────────────────────────────────────────────
#: (키 필드, 비교할 컬럼들, 지금 표를 읽는 함수, 저장 함수)
종류표: Dict[str, Tuple[str, Tuple[str, ...], Callable, Callable]] = {
    "index": ("index_name", krx_index.COLUMNS[1:],
              lambda bas_dd, market: krx_index_rows(bas_dd),
              krx_index.save_renormalized),
    "stock": ("code", krx_store.COLUMNS[1:],
              lambda bas_dd, market: krx_store.rows_for(bas_dd, market),
              krx_store.save_renormalized),
}


def krx_index_rows(bas_dd: str) -> Dict[str, Dict]:
    """지금 표에 든 그 날짜의 지수 행들. 비교용이다.

    지수명은 전역에서 유일하므로 시장 조건이 없어도 된다. **종목은 다르다** —
    `krx_store.rows_for` 는 시장 조건이 필수다.
    """
    with krx_store.connect() as conn:
        rows = conn.execute(
            f"SELECT {','.join(krx_index.COLUMNS)} FROM index_price WHERE bas_dd=?",
            (bas_dd,),
        ).fetchall()
    return {row[1]: dict(zip(krx_index.COLUMNS, row, strict=True)) for row in rows}


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
        # ⚠️ KONEX 는 MARKET_APIS 에는 있지만 수집 대상 시장이 아니다. 여기서 걸러
        #    "다룰 수 없음" 으로 세지 않으면, 저장 함수가 던지는 예외로 스크립트가 죽는다.
        if market not in krx_store.MARKETS:
            return None
        return "stock", market, [api.normalize_row(row, bas_dd, market) for row in rows]
    return None


def _raw_coverage() -> Dict[str, Dict]:
    """표가 아는 구간 중 **원문이 있는 구간이 얼마나 되나.**

    예전 판은 *"원문이 있는데 못 다룬 것"* 만 셌다. 그러면 원문이 아예 없는
    4,000여 일이 **어느 숫자에도 안 나온다** — 0 건씩 처리하고 "끝났다" 로 보인다.
    """
    있는대상 = set(raw_store.targets("krx"))
    원문날짜 = {"index": set(), "stock": set()}
    for target in 있는대상:
        parsed = _parse_target(target)
        if parsed is None:
            continue
        path, bas_dd = parsed
        if path.startswith("idx/"):
            원문날짜["index"].add(bas_dd)
        elif path.startswith("sto/"):
            원문날짜["stock"].add(bas_dd)

    with krx_store.connect() as conn:
        표날짜 = {
            "index": {r[0] for r in conn.execute(
                "SELECT DISTINCT bas_dd FROM index_fetch_log")},
            "stock": {r[0] for r in conn.execute(
                "SELECT DISTINCT bas_dd FROM fetch_log")},
        }

    결과 = {}
    for 종류 in ("index", "stock"):
        빠짐 = 표날짜[종류] - 원문날짜[종류]
        결과[종류] = {
            "표": len(표날짜[종류]),
            "원문": len(원문날짜[종류] & 표날짜[종류]),
            "빠짐": len(빠짐),
            "빠짐_처음": min(빠짐) if 빠짐 else None,
            "빠짐_마지막": max(빠짐) if 빠짐 else None,
        }
    return 결과


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

    처리 = 다룰수없음 = 바뀜 = 빈원문 = 0
    종류별: Dict[str, int] = {}
    바뀐표본: List[str] = []

    for raw in raw_store.iter_latest(args.source, prefix=args.prefix):
        if args.limit and 처리 >= args.limit:
            break
        결과 = _renormalize_one(raw)
        if 결과 is None:
            다룰수없음 += 1
            continue
        종류, market, items = 결과
        if not items:
            # 원문은 있는데 행이 0개다. 실제로 그런 파일이 있다(17B). 아무것도 쓰지
            # 않고 따로 센다 — 처리했다고 세면 "다 됐다" 로 보인다.
            빈원문 += 1
            continue

        처리 += 1
        종류별[종류] = 종류별.get(종류, 0) + 1
        키필드, 비교컬럼, 읽기, 저장 = 종류표[종류]

        _, bas_dd = _parse_target(raw["target"])
        기존 = 읽기(bas_dd, market)
        for item in items:
            키 = item.get(키필드)
            옛행 = 기존.get(키)
            새행 = {col: item.get(col) for col in 비교컬럼}
            if 옛행 is None or any(옛행.get(col) != 새행.get(col) for col in 새행):
                바뀜 += 1
                if len(바뀐표본) < 10:
                    바뀐표본.append(f"{bas_dd} {market} {키}: {옛행} → {새행}")

        if not args.dry_run:
            저장(bas_dd, market, items)

    내역 = " · ".join(f"{k} {human(v)}" for k, v in sorted(종류별.items())) or "없음"
    print(f"처리한 대상 {human(처리)}개 ({내역})")
    print(f"다룰 수 없어 건너뛴 것 {human(다룰수없음)}개 · 행이 0개인 원문 {human(빈원문)}개")
    print(f"달라지는 행 {human(바뀜)}개")
    if 바뀐표본:
        print("  표본:")
        for line in 바뀐표본:
            print(f"    {line[:160]}")
    print()

    # ── 닿지 못한 구간 ──────────────────────────────────────────────────────
    print("── 원문이 없어 재정규화할 수 없는 구간 ──")
    덮개 = _raw_coverage()
    남음 = 0
    for 종류, v in 덮개.items():
        남음 += v["빠짐"]
        상태 = "✅ 전부 덮인다" if not v["빠짐"] else (
            f"⚠️ {human(v['빠짐'])}일이 원문 없음 "
            f"({v['빠짐_처음']} ~ {v['빠짐_마지막']})")
        print(f"  {종류:<6} 표 {human(v['표'])}일 · 원문 있음 {human(v['원문'])}일  {상태}")

    if args.dry_run:
        print()
        print("※ --dry-run 이라 표를 건드리지 않았습니다.")
        print("   위 표본을 확인한 뒤 옵션 없이 다시 실행하세요.")
        return 0

    if 남음:
        print()
        print(f"⚠️ 원문이 없는 {human(남음)}일은 **다시 정규화되지 않았습니다.**")
        print("   정규화를 고쳤다면 그 구간은 옛 규칙 그대로입니다. 섞여 있다는 뜻입니다.")
        return 2                      # 0 이 아니어야 배치가 이 사실을 흘리지 않는다
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
