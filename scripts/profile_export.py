"""반출 폴더를 읽어 `PROFILE.json` 을 만든다 — 칸마다 무엇이 들었는지.

사용법
------
    python scripts/profile_export.py                        # 가장 최근 날짜 폴더
    python scripts/profile_export.py --path data/outbox/2026-09-03
    python scripts/profile_export.py --path ... --quiet      # 요약만

반출(`export_team_dataset.py`)은 끝에서 이걸 자동으로 부른다. 이 CLI 는 **이미 만들어
둔 반출 폴더에 나중에 붙일 때** 쓴다 — 자료는 그대로 두고 설명만 다시 내는 경우다.

두 번 돌려도 안전하다. 파일을 읽기만 하고 `PROFILE.json` 만 다시 쓴다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.export_profile import write_profile  # noqa: E402

#: 날짜 모양 폴더만 고른다. `data/outbox` 에는 `dart_20260902` 처럼 성격이 다른 반출도
#: 살고, 사전순으로는 `'d' > '2'` 라 그쪽이 이긴다.
DATE_GLOB = "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]"


def latest_outbox() -> Path | None:
    후보 = sorted(p for p in Path("data/outbox").glob(DATE_GLOB) if p.is_dir())
    return 후보[-1] if 후보 else None


def main() -> int:
    parser = argparse.ArgumentParser(description="반출 폴더의 칸별 통계를 낸다")
    parser.add_argument("--path", default=None,
                        help="반출 폴더 (기본: data/outbox 의 가장 최근 날짜)")
    parser.add_argument("--quiet", action="store_true", help="파일별 요약만 출력")
    args = parser.parse_args()

    root = Path(args.path) if args.path else latest_outbox()
    if root is None:
        print("data/outbox 에 날짜(YYYY-MM-DD) 폴더가 없다.")
        print("  할 일: python scripts/export_team_dataset.py 를 먼저 돌린다.")
        print("  다른 폴더를 재려면 --path 로 직접 지정한다.")
        return 1
    if not (root / "MANIFEST.json").exists():
        print(f"{root}/MANIFEST.json 이 없다. 반출이 끝나지 않았다.")
        return 1

    print(f"반출 폴더: {root}")
    프로필 = write_profile(root)

    print()
    print(f"  {'파일':44s} {'행':>10s} {'칸':>4s} {'결측칸':>6s}  기간")
    for f in 프로필["files"]:
        기간 = f.get("기간")
        구간 = f"{기간['처음']} ~ {기간['끝']}" if 기간 else ""
        print(f"  {f['path']:44s} {f['행']:>10,} {f['칸수']:>4d} {f['결측있는칸']:>6d}  {구간}")

    if not args.quiet:
        print()
        for f in 프로필["files"]:
            print(f"── {f['path']} ──")
            for c in f["칸들"]:
                줄 = f"  {c['이름']:16s} {c['형']:13s} 결측 {c['결측률']:>7.2%}"
                if "평균" in c:
                    print(f"{줄}  {c['min']!r} ~ {c['max']!r}  평균 {c['평균']}")
                elif c.get("비어있음"):
                    print(f"{줄}  (전부 비어 있다)")
                else:
                    print(f"{줄}  고유 {c.get('고유', 0):,}  {c['min']!r} ~ {c['max']!r}")
                if "분포" in c:
                    총 = sum(c["분포"].values()) or 1
                    몫 = " · ".join(f"{k} {v / 총:.2%}" for k, v in c["분포"].items())
                    print(f"      분포: {몫}")
            print()

    print(f"✅ PROFILE.json 기록 — 파일 {len(프로필['files'])}개 · "
          f"칸 {sum(f['칸수'] for f in 프로필['files'])}개")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
