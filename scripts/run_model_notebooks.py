"""생성된 96개 모델 노트북과 24개 비교 노트북을 순서대로 실행한다."""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_model_notebooks import MODELS, notebook_targets  # noqa: E402


def main() -> int:
    targets = []
    for _combination, _variant, directory in notebook_targets():
        targets.extend(directory / filename for filename in MODELS.values())
        targets.append(directory / "05.모델비교.ipynb")

    for number, path in enumerate(targets, start=1):
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(
            notebook,
            timeout=900,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        )
        try:
            client.execute()
        except Exception as error:
            raise RuntimeError(f"{number}/{len(targets)} 실행 실패: {path}") from error
        nbformat.write(notebook, path)
        print(f"{number:03d}/{len(targets)} 완료: {path.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
