"""DART 고유번호 매핑표를 만든다 — 종목코드(005930) → 고유번호(00126380).

**왜 필요한가.** DART 는 회사를 종목코드가 아니라 **8자리 고유번호**로만 받는다.
그 매핑은 `corpCode.xml` 한 곳에서만 주는데 약 10만 건이라 요청할 때마다 받을 수 없다.
그래서 한 번 받아 상장사만 추려 `data/corp_code.json` 에 넣어 두고, 앱은 그 파일만 읽는다.

**이 파일이 없어서 생긴 일.** `ingest/clients/dart_data.py` 는 매핑을 못 찾으면
*"`python scripts/build_corp_code.py` 로 만들 수 있습니다"* 라고 안내해 왔는데, 정작
그 스크립트가 없었다. 안내를 따라간 사람이 막다른 길에 도달한다. 이제 그 길이 뚫렸다.

    python scripts/build_corp_code.py             # 상장사 매핑 + 전체 목록을 만든다
    python scripts/build_corp_code.py --dry-run   # 받아서 세어만 보고 파일을 쓰지 않는다
    python scripts/build_corp_code.py --no-raw    # 원문 ZIP 을 보존하지 않는다

만들어지는 것
-------------
    data/corp_code.json       상장사만  (dart_data 가 읽는 파일)
    data/corp_code_all.json   전체 법인 (비상장 모회사를 볼 일이 생길 때를 위한 사본)

⚠️ DART 하루 한도는 20,000 회다. 이 스크립트는 **1회**만 쓴다. 매핑은 자주 바뀌지
   않으므로 매일 돌릴 이유가 없다 — 새 종목이 상장했는데 못 찾을 때만 다시 돌린다.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import budget  # noqa: E402  (경로를 세운 뒤에 import 한다)
from ingest.clients import dart_data  # noqa: E402

KST = timezone(timedelta(hours=9))

LISTED_FILE = ROOT / "data" / "corp_code.json"
ALL_FILE = ROOT / "data" / "corp_code_all.json"

SOURCE_URL = "https://opendart.fss.or.kr/api/corpCode.xml"


def now_kst() -> str:
    """지금 시각을 KST 로 적는다. 파일에 남는 '언제 만들었나'의 근거다."""
    return datetime.now(KST).isoformat(timespec="seconds")


def is_listed(row: Dict[str, str]) -> bool:
    """상장사인가.

    DART 는 비상장사에도 고유번호를 주고, 그 행은 `stock_code` 가 **공백**이다.
    공백 문자열이 섞여 오므로 strip 한 뒤에 판정한다.
    """
    return bool((row.get("stock_code") or "").strip())


def pick_newer(old: Dict[str, str], new: Dict[str, str]) -> Dict[str, str]:
    """같은 종목코드가 두 번 나왔을 때 어느 쪽을 남길지 고른다.

    합병·재상장으로 한 종목코드에 고유번호가 둘 붙는 경우가 있다. 어느 쪽이 지금
    쓰이는 것인지 이름만 봐서는 알 수 없으므로 **`modify_date` 가 큰 쪽**을 남긴다.
    날짜가 같거나 비어 있으면 먼저 온 쪽을 그대로 둔다 — 임의로 바꾸면 실행할 때마다
    결과가 달라져서 무엇이 정답인지 영영 알 수 없게 된다.
    """
    return new if (new.get("modify_date") or "") > (old.get("modify_date") or "") else old


def build_map(rows: List[Dict[str, str]]) -> Tuple[Dict[str, Dict[str, str]], int]:
    """상장사 행들을 `종목코드 → {고유번호, 회사명, 수정일}` 로 접는다.

    돌려주는 둘째 값은 **겹쳐서 버린 건수**다. 0 이 아니면 위의 `pick_newer` 가
    실제로 판단을 내렸다는 뜻이므로 사람이 알아야 한다.
    """
    out: Dict[str, Dict[str, str]] = {}
    collisions = 0

    for row in rows:
        if not is_listed(row):
            continue
        code = row["stock_code"].strip()
        entry = {
            "corp_code": row["corp_code"].strip(),
            "corp_name": row["corp_name"].strip(),
            "modify_date": row["modify_date"].strip(),
        }
        if code in out:
            collisions += 1
            out[code] = pick_newer(out[code], entry)
        else:
            out[code] = entry

    return out, collisions


def odd_codes(mapping: Dict[str, Dict[str, str]]) -> List[str]:
    """여섯 자리 숫자가 아닌 종목코드를 골라낸다.

    버리지 않고 **보고만** 한다. DART 가 우리가 모르는 형식을 내려주기 시작했을 때
    조용히 사라지는 것보다 눈에 띄는 편이 낫다.
    """
    return sorted(c for c in mapping if not (len(c) == 6 and c.isdigit()))


def write_json(path: Path, payload: Dict) -> None:
    """JSON 을 원자적으로 쓴다.

    같은 파일을 웹 프로세스가 읽는 중일 수 있다. 곧바로 덮어쓰면 반쯤 쓰인 파일을
    읽어 `json.JSONDecodeError` 가 나므로, 임시 파일에 다 쓰고 마지막에 이름을 바꾼다.
    `Path.replace` 는 같은 파일시스템 안에서 원자적이다.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="DART 고유번호 매핑표(data/corp_code.json)를 만든다"
    )
    parser.add_argument("--dry-run", action="store_true", dest="dry_run",
                        help="받아서 세어만 보고 파일을 쓰지 않는다")
    parser.add_argument("--no-raw", action="store_true", dest="no_raw",
                        help="응답 원문 ZIP 을 raw_store 에 보존하지 않는다")
    parser.add_argument("--no-all", action="store_true", dest="no_all",
                        help="전체 법인 사본(corp_code_all.json)을 만들지 않는다")
    args = parser.parse_args()

    print("── DART 고유번호 매핑 만들기 ──")

    # 하루 한도(20,000)를 쓰는 호출이다. 1회지만 대장에 남겨야 사용량이 맞는다.
    if not budget.try_spend("dart"):
        print("❌ DART 하루 한도를 다 썼습니다. 내일 다시 돌리세요.")
        return 1

    # --dry-run 은 "아무것도 남기지 않는다" 는 약속이다. 원문 보존도 DB 쓰기이므로
    # 함께 끈다 — 파일만 안 쓰고 DB 는 쓰면 약속을 반만 지키는 셈이다.
    keep_raw = not args.no_raw and not args.dry_run

    try:
        rows = dart_data.fetch_corp_code_rows(keep_raw=keep_raw)
    except dart_data.DartError as error:
        print(f"❌ {error}")
        return 1

    listed_map, collisions = build_map(rows)

    print(f"  전체 법인   {len(rows):,} 건")
    print(f"  상장사      {len(listed_map):,} 건")
    if collisions:
        print(f"  ⚠️ 종목코드 겹침 {collisions} 건 — modify_date 가 최신인 쪽을 남겼습니다")

    strange = odd_codes(listed_map)
    if strange:
        print(f"  ⚠️ 여섯 자리 숫자가 아닌 종목코드 {len(strange)} 건: {strange[:5]}")

    if not listed_map:
        print("❌ 상장사가 한 건도 없습니다. 응답 형식이 바뀌었을 수 있습니다.")
        return 1

    if args.dry_run:
        print("  (--dry-run 이라 파일을 쓰지 않았습니다)")
        return 0

    generated_at = now_kst()
    write_json(LISTED_FILE, {
        "generated_at": generated_at,
        "source": SOURCE_URL,
        "scope": "listed",
        "total_rows": len(rows),
        "map": listed_map,
    })
    print(f"  ✅ {LISTED_FILE.relative_to(ROOT)}  ({LISTED_FILE.stat().st_size / 1e6:.2f} MB)")

    if not args.no_all:
        write_json(ALL_FILE, {
            "generated_at": generated_at,
            "source": SOURCE_URL,
            "scope": "all",
            "total_rows": len(rows),
            # 전체는 종목코드가 없는 행이 대부분이라 매핑이 아니라 **목록**으로 둔다.
            "rows": rows,
        })
        print(f"  ✅ {ALL_FILE.relative_to(ROOT)}  ({ALL_FILE.stat().st_size / 1e6:.2f} MB)")

    # 방금 쓴 파일을 앱이 실제로 읽을 수 있는지 여기서 확인한다.
    # 형식이 어긋나면 지금 알아야지, DART 를 부르는 순간에 404 로 알면 늦다.
    loaded = dart_data.reload_corp_map()
    if loaded != len(listed_map):
        print(f"❌ 다시 읽은 매핑이 {loaded:,} 건으로 쓴 것({len(listed_map):,})과 다릅니다.")
        return 1
    print(f"  ✅ 다시 읽기 확인 {loaded:,} 건")

    sample = next(iter(sorted(listed_map)))
    print(f"  예: {sample} → {listed_map[sample]['corp_code']} ({listed_map[sample]['corp_name']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
