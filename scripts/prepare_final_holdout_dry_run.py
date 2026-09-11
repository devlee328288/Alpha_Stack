"""개발구간 마지막 공통 60일을 가짜 홀드아웃으로 잘라 실행기 입력을 만든다."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from features.model_dataset import build_model_dataset  # noqa: E402
from features.stock_model_dataset import (  # noqa: E402
    STOCK_COMBINATION_FEATURES,
    StockModelDataset,
    select_stock_feature_dataset,
)
from models.final_holdout import INDEX_COMBINATION, STOCK_COMBINATION  # noqa: E402

DEFAULT_INDEX_PATH = ROOT / "data" / "raw" / "hf_snapshot" / "full" / "index_price_dev.parquet"
DEFAULT_STOCK_PANEL_PATH = ROOT / "data" / "raw" / "stock_model_panel.parquet"


def split_dry_run_frames(
    index_frame: pd.DataFrame,
    stock_frame: pd.DataFrame,
    *,
    evaluation_dates: int = 60,
    gap_dates: int = 5,
) -> dict[str, pd.DataFrame]:
    """두 트랙의 마지막 공통 거래일을 평가로 떼고 직전 5일은 학습에서 버린다."""

    if evaluation_dates < 1 or gap_dates < 1:
        raise ValueError("평가일과 gap은 모두 1 이상이어야 합니다.")
    index_dates = set(index_frame["bas_dd"].astype(str))
    stock_dates = set(stock_frame["bas_dd"].astype(str))
    common_dates = sorted(index_dates & stock_dates)
    needed = evaluation_dates + gap_dates + 1
    if len(common_dates) < needed:
        raise ValueError(f"드라이런 분할에는 공통 거래일이 최소 {needed}개 필요합니다.")
    holdout_dates = set(common_dates[-evaluation_dates:])
    gap = set(common_dates[-(evaluation_dates + gap_dates) : -evaluation_dates])
    first_holdout = min(holdout_dates)
    return {
        "index_dev": index_frame.loc[
            index_frame["bas_dd"].astype(str).lt(first_holdout)
            & ~index_frame["bas_dd"].astype(str).isin(gap)
        ].copy(),
        "index_holdout": index_frame.loc[
            index_frame["bas_dd"].astype(str).isin(holdout_dates)
        ].copy(),
        "stock_dev": stock_frame.loc[
            stock_frame["bas_dd"].astype(str).lt(first_holdout)
            & ~stock_frame["bas_dd"].astype(str).isin(gap)
        ].copy(),
        "stock_holdout": stock_frame.loc[
            stock_frame["bas_dd"].astype(str).isin(holdout_dates)
        ].copy(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--stock-panel-path", type=Path, default=DEFAULT_STOCK_PANEL_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluation-dates", type=int, default=60)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"기존 드라이런 입력을 덮어쓰지 않습니다: {args.output_dir}")

    index_prices = pd.read_parquet(args.index_path)
    index_dataset = build_model_dataset(index_prices, INDEX_COMBINATION)
    stock_panel = pd.read_parquet(args.stock_panel_path)
    stock_dataset = select_stock_feature_dataset(
        stock_panel,
        STOCK_COMBINATION_FEATURES[STOCK_COMBINATION],
    )
    if not isinstance(stock_dataset, StockModelDataset):
        raise TypeError("개별종목 드라이런 입력이 StockModelDataset이 아닙니다.")
    frames = split_dry_run_frames(
        index_dataset.frame,
        stock_dataset.frame,
        evaluation_dates=args.evaluation_dates,
    )
    args.output_dir.mkdir(parents=True)
    for name, frame in frames.items():
        frame.to_parquet(args.output_dir / f"{name}.parquet", index=False)
    print(
        f"드라이런 입력 저장: {args.output_dir} · "
        f"평가 {frames['index_holdout']['bas_dd'].nunique()}거래일"
    )


if __name__ == "__main__":
    main()
