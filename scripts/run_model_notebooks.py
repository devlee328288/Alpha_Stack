"""생성된 KOSPI200 모델·비교 노트북을 조합별로 실행한다."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_model_notebooks import MODELS, notebook_targets  # noqa: E402


def main(requested: tuple[str, ...] | None = None) -> int:
    """요청한 조합만 실행하며, 생략하면 정의된 전체 조합을 실행한다."""

    targets = []
    for _combination, _variant, directory in notebook_targets(requested):
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--combinations",
        nargs="+",
        metavar="NAME",
        help="실행할 조합 이름입니다. 생략하면 전체를 실행합니다.",
    )
    arguments = parser.parse_args()
    raise SystemExit(main(tuple(arguments.combinations) if arguments.combinations else None))
