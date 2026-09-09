import numpy as np
import pandas as pd

from features.stock_model_dataset import StockModelDataset
from scripts.run_stock_preprocessing_ablation import build_preprocessed_dataset


def test_전처리_ablation은_행과라벨을고정하고_피처만바꾼다():
    rows = []
    for day in ("20240102", "20240103"):
        for index in range(5):
            rows.append(
                {
                    "bas_dd": day,
                    "code": f"{index:06d}",
                    "label_numeric": (-1, 0, 1, 0, 1)[index],
                    "first": float(index + 1),
                    "second": float((index + 1) ** 2),
                }
            )
    frame = pd.DataFrame(rows)
    dataset = StockModelDataset(frame=frame, feature_columns=("first", "second"))

    transformed = build_preprocessed_dataset(dataset)

    pd.testing.assert_frame_equal(
        transformed.frame[["bas_dd", "code", "label_numeric"]],
        dataset.frame[["bas_dd", "code", "label_numeric"]],
    )
    assert np.isfinite(transformed.x.to_numpy(dtype=float)).all()
    assert not np.allclose(transformed.x, dataset.x)
    for _, group in transformed.frame.groupby("bas_dd"):
        assert np.allclose(group[["first", "second"]].mean(), 0.0, atol=1e-12)
