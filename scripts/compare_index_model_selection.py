"""KOSPI200 개발구간 후보를 미확정 선정 기준별로 비교한다.

이 스크립트는 봉인 홀드아웃을 읽지 않는다. 이슈 #203에서 최종 기준을 정하기 전에
`reports/model_sweep.json`에 이미 저장된 개발구간 100개 후보만 다시 정렬한다.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.final_models import compare_index_model_selection_policies  # noqa: E402

INPUT_PATH = ROOT / "reports" / "model_sweep.json"
OUTPUT_PATH = ROOT / "reports" / "index_model_selection_comparison.json"


def main() -> None:
    """개발구간 선정 정책별 1위를 JSON 보고서로 저장한다."""

    report = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    comparison = compare_index_model_selection_policies(report)
    comparison["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    comparison["source_report"] = str(INPUT_PATH.relative_to(ROOT)).replace("\\", "/")
    comparison["source"] = report.get("source", {})
    OUTPUT_PATH.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"KOSPI200 선정 기준 {len(comparison['policies'])}개 비교: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
