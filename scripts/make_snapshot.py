"""스냅샷 박제 CLI — 실험이 딛고 선 자료를 얼리고 지문을 남긴다 (요구사항 F-02).

사용법
------
    python scripts/make_snapshot.py                      # 코스피 200 전 구간 박제
    python scripts/make_snapshot.py --end 20260825       # 종료일을 못 박아서
    python scripts/make_snapshot.py --index "코스닥 150"
    python scripts/make_snapshot.py --verify             # 박제된 것들 지문 재검사
    python scripts/make_snapshot.py --list               # 무엇이 박제돼 있나

⚠️ **두 번 돌려도 안전하다.** 내용이 같으면 지문이 같고 아무 일도 일어나지 않는다.
   내용이 달라졌으면 멈추고 무엇이 어긋났는지 말한다 (`SnapshotConflictError`).
   덮어쓰려면 `--force` 를 명시해야 하고, 그건 ADR 에 적을 일이다.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingest import snapshot as snap  # noqa: E402  (경로 설정 후에 import 해야 한다)


def 목록_출력() -> int:
    """무엇이 박제돼 있는지 표로 보여준다."""
    meta = snap.load_meta()
    항목들 = meta.get("snapshots", {})
    if not 항목들:
        print("박제된 스냅샷이 없다. python scripts/make_snapshot.py 로 먼저 만든다.")
        return 0

    print(f"── 박제된 스냅샷 {len(항목들)}건 ({snap.META_PATH.name}) ──")
    for 이름 in sorted(항목들):
        it = 항목들[이름]
        print(f"  {이름}")
        print(f"    구간   : {it['start']} ~ {it['end']}  ({it['rows']:,}행)")
        print(f"    SHA-256: {it['sha256']}")
        print(f"    박제    : {it['frozen_at_kst']}  ({it['hash_method']})")
    return 0


def 검증_출력(이름: str | None) -> int:
    """parquet 을 되읽어 지문이 지금도 맞는지 확인한다. 어긋나면 exit 1."""
    결과 = snap.verify(이름)
    if not 결과:
        print("검증할 스냅샷이 없다.")
        return 0

    실패 = 0
    print("── 스냅샷 지문 검증 ──")
    for r in 결과:
        if r["ok"]:
            print(f"  ✅ {r['name']}  {r['start']} ~ {r['end']}  ({r['rows']:,}행)")
            print(f"     {r['expected']}")
        else:
            실패 += 1
            print(f"  ❌ {r['name']} — {r['reason']}")
            if r.get("expected"):
                print(f"     박제된 해시: {r['expected']}")
                print(f"     지금  해시 : {r['actual']}")
    if 실패:
        print()
        print(f"  {실패}건이 어긋났다. 리포트에 실린 해시와 파일이 갈라졌다는 뜻이다.")
        print("  → 파일이 바뀌었는지, 아니면 메타가 바뀌었는지부터 확인한다.")
    return 1 if 실패 else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="지수 시계열을 parquet 으로 박제하고 SHA-256 지문을 남긴다")
    parser.add_argument("--index", default="코스피 200", help="지수명 (기본: 코스피 200)")
    parser.add_argument("--start", help="시작일 YYYYMMDD (기본: 있는 자료의 처음)")
    parser.add_argument("--end", help="종료일 YYYYMMDD (기본: DB 의 마지막 거래일)")
    parser.add_argument("--force", action="store_true",
                        help="이미 박제된 것과 내용이 달라도 덮어쓴다. 평소에는 쓰지 않는다")
    parser.add_argument("--verify", action="store_true", help="박제된 스냅샷의 지문을 재검사")
    parser.add_argument("--name", help="--verify 대상 하나만 지정 (예: kospi200_20260825)")
    parser.add_argument("--list", action="store_true", dest="목록",
                        help="박제된 스냅샷 목록만 출력")
    args = parser.parse_args()

    if args.목록:
        return 목록_출력()
    if args.verify:
        return 검증_출력(args.name)

    try:
        항목 = snap.freeze(index_name=args.index, start=args.start,
                          end=args.end, force=args.force)
    except snap.SnapshotConflictError as e:
        # 막다른 길로 두지 않는다 — 무엇을 해야 하는지 예외 메시지가 이미 담고 있다
        print(f"❌ {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"❌ {e}", file=sys.stderr)
        return 1

    print("── 박제 완료 ──")
    print(f"  지수    : {항목['index_name']}")
    print(f"  구간    : {항목['start']} ~ {항목['end']}  ({항목['rows']:,}행)")
    print(f"  파일    : {항목['path']}")
    print(f"  SHA-256 : {항목['sha256']}   ← 리포트에 싣는 값 (내용 해시)")
    print(f"  parquet : {항목['parquet_sha256']}   (파일 해시 · 참고용)")
    print(f"  박제 시각: {항목['frozen_at_kst']} KST")
    print()
    print(f"  메타    : {snap.META_PATH.relative_to(snap.ROOT)}")
    print("  ⚠️ parquet 은 .gitignore 대상이다. 커밋되는 것은 메타(지문)뿐이다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
