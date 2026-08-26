"""문서 내부 링크가 실제로 존재하는 파일을 가리키는지 검사한다.

**왜 스크립트인가.** `docs/` 를 도메인·버전 폴더로 나누면 상대 경로가 전부 바뀐다.
사람 눈으로 83건을 확인하는 것은 신뢰할 수 없고, 깨진 링크는 조용히 남는다.

    python scripts/check_doc_links.py          # 깨진 링크가 있으면 exit 1
    python scripts/check_doc_links.py --list   # 전부 나열

⚠️ 문서를 옮기거나 버전 폴더를 새로 팔 때마다 돌린다.
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")

# 검사 대상. 코드 안 docstring 은 보지 않는다 (md 링크 문법이 아니다)
TARGETS = ["README.md", "AGENTS.md"]


def 검사할_파일() -> list:
    out = [ROOT / t for t in TARGETS if (ROOT / t).exists()]
    out += sorted((ROOT / "docs").rglob("*.md"))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="문서 링크가 실제 파일을 가리키는지 검사")
    parser.add_argument("--list", action="store_true", dest="전부",
                        help="깨지지 않은 링크까지 전부 나열")
    args = parser.parse_args()

    깨짐 = []
    총링크 = 0
    for path in 검사할_파일():
        rel = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        for label, target in LINK.findall(text):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            총링크 += 1
            경로부 = target.split("#", 1)[0]
            if not 경로부:
                continue
            대상 = (path.parent / 경로부).resolve()
            존재 = 대상.exists()
            if args.전부:
                표 = "✅" if 존재 else "❌"
                print(f"  {표} {rel} → {target}")
            if not 존재:
                깨짐.append((rel, label, target))

    print(f"── 문서 링크 검사 — 내부 링크 {총링크}건 ──")
    if not 깨짐:
        print("  ✅ 깨진 링크 없음")
        return 0

    print(f"  ❌ 깨진 링크 {len(깨짐)}건")
    for rel, label, target in 깨짐:
        print(f"     {rel}")
        print(f"       [{label}]({target})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
