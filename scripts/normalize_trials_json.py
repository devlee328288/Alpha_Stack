"""과거 trials.jsonl의 class weight 미사용값을 표준 JSON ``null``로 정규화한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "reports" / "trials.jsonl"


def normalize_trials_text(text: str) -> tuple[str, dict[str, int]]:
    """허용된 두 과거 표기만 고치고 모든 행을 엄격한 JSON으로 다시 검증한다."""

    lines = text.splitlines()
    normalized_lines: list[str] = []
    trial_ids: set[str] = set()
    nan_values = 0
    nan_ids = 0
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"trials.jsonl {line_number}행이 비어 있습니다.")
        if ": NaN" in line and '"class_weight": NaN' not in line:
            raise ValueError(
                f"trials.jsonl {line_number}행의 class_weight 이외 필드에 NaN이 있습니다."
            )
        nan_values += line.count('"class_weight": NaN')
        nan_ids += line.count("-inner-nan")
        normalized = line.replace('"class_weight": NaN', '"class_weight": null')
        normalized = normalized.replace("-inner-nan", "-inner-none")
        try:
            record = json.loads(
                normalized,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"허용되지 않는 JSON 상수입니다: {value}")
                ),
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"trials.jsonl {line_number}행이 표준 JSON이 아닙니다.") from error
        trial_id = str(record.get("trial_id", ""))
        if not trial_id:
            raise ValueError(f"trials.jsonl {line_number}행에 trial_id가 없습니다.")
        if trial_id in trial_ids:
            raise ValueError(f"정규화 뒤 trial_id가 중복됩니다: {trial_id}")
        trial_ids.add(trial_id)
        normalized_lines.append(normalized)
    return "\n".join(normalized_lines) + "\n", {
        "rows": len(lines),
        "class_weight_nan_to_null": nan_values,
        "inner_nan_to_none": nan_ids,
    }


def normalize_trials_file(path: Path = DEFAULT_PATH) -> dict[str, int]:
    original = path.read_text(encoding="utf-8")
    normalized, counts = normalize_trials_text(original)
    path.write_text(normalized, encoding="utf-8")
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_PATH)
    arguments = parser.parse_args()
    counts = normalize_trials_file(arguments.path)
    print(
        f"{counts['rows']}행 검증 · class_weight NaN→null "
        f"{counts['class_weight_nan_to_null']}건 · inner-nan→inner-none "
        f"{counts['inner_nan_to_none']}건"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
