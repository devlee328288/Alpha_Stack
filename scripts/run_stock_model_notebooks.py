"""개별종목 기본모델·조합 A~J·best-result 노트북을 순서대로 실행한다."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT = ROOT / "notebooks" / "04-모델" / "개별종목" / "실험"
BASE = EXPERIMENT / "기본모델"
BEST_ROOT = EXPERIMENT / "조합별 best result"

COMBINATION_DIRECTORIES = {
    "A": "조합A_trend_momentum_volatility_volume_returns",
    "B": "조합B_trend_momentum_high_distance",
    "C": "조합C_short_reversal_intraday",
    "D": "조합D_volatility_liquidity",
    "E": "조합E_sector_market_relative_strength",
    "F": "조합F_cross_sectional_ranks",
    "G": "조합G_direction_magnitude_interaction",
    "H": "조합H_volatility_regime_interaction",
    "I": "조합I_kospi200_e_same_features",
    "J": "조합J_kospi200_e_market_relative_strength",
}


def _targets(requested: tuple[str, ...] | None = None) -> list[Path]:
    """기본모델을 먼저, 실제 조합과 요약 노트북을 나중에 실행한다."""

    names = tuple(COMBINATION_DIRECTORIES)
    if requested is not None:
        normalized = tuple(str(name).strip().upper() for name in requested)
        if len(set(normalized)) != len(normalized):
            raise ValueError("실행할 조합 이름이 중복되었습니다.")
        unknown = set(normalized) - set(names)
        if unknown:
            raise ValueError(f"정의되지 않은 개별종목 조합입니다: {sorted(unknown)}")
        selected = set(normalized)
        names = tuple(name for name in names if name in selected)
    base = sorted(BASE.glob("*.ipynb")) if requested is None else []
    combinations = [EXPERIMENT / COMBINATION_DIRECTORIES[name] for name in names]
    combination_models = [
        notebook
        for combination in combinations
        for notebook in sorted(combination.glob("0[1-4].*.ipynb"))
    ]
    comparison = [combination / "05.모델비교.ipynb" for combination in combinations]
    best = []
    for name in names:
        starred = BEST_ROOT / f"⭐조합{name}"
        directory = starred if starred.is_dir() else BEST_ROOT / f"조합{name}"
        best.extend(sorted(directory.glob("*.ipynb")))
    return [*base, *combination_models, *comparison, *best]


def main(requested: tuple[str, ...] | None = None) -> int:
    """저장소 루트를 작업경로로 고정해 모든 노트북의 출력을 채운다."""

    # Windows 기본 CP949 콘솔에서도 별표가 붙은 best-result 파일명을 출력할 수 있게 한다.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    targets = _targets(requested)
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--combinations",
        nargs="+",
        metavar="NAME",
        help="지정한 조합 노트북만 실행합니다. 생략하면 전체를 실행합니다.",
    )
    arguments = parser.parse_args()
    raise SystemExit(main(tuple(arguments.combinations) if arguments.combinations else None))
