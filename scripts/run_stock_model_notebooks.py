"""개별종목 기본모델·조합 A~G·best-result 노트북을 순서대로 실행한다."""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT = ROOT / "notebooks" / "04-모델" / "개별종목" / "실험"
BASE = EXPERIMENT / "기본모델"
BEST_ROOT = EXPERIMENT / "조합별 best result"

COMBINATION_DIRECTORIES = (
    "조합A_trend_momentum_volatility_volume_returns",
    "조합B_trend_momentum_high_distance",
    "조합C_short_reversal_intraday",
    "조합D_volatility_liquidity",
    "조합E_sector_market_relative_strength",
    "조합F_cross_sectional_ranks",
    "조합G_direction_magnitude_interaction",
)


def _targets() -> list[Path]:
    """기본모델을 먼저, 실제 조합과 요약 노트북을 나중에 실행한다."""

    base = sorted(BASE.glob("*.ipynb"))
    combinations = [EXPERIMENT / directory for directory in COMBINATION_DIRECTORIES]
    combination_models = [
        notebook
        for combination in combinations
        for notebook in sorted(combination.glob("0[1-4].*.ipynb"))
    ]
    comparison = [combination / "05.모델비교.ipynb" for combination in combinations]
    best = sorted(BEST_ROOT.glob("조합*/*.ipynb"))
    return [*base, *combination_models, *comparison, *best]


def main() -> int:
    """저장소 루트를 작업경로로 고정해 모든 노트북의 출력을 채운다."""

    # Windows 기본 CP949 콘솔에서도 별표가 붙은 best-result 파일명을 출력할 수 있게 한다.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    targets = _targets()
    for number, path in enumerate(targets, start=1):
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(
            notebook,
            timeout=1_800,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        )
        try:
            client.execute()
        except Exception as error:
            raise RuntimeError(f"{number}/{len(targets)} 실행 실패: {path}") from error
        nbformat.write(notebook, path)
        print(f"{number:02d}/{len(targets)} 완료: {path.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
