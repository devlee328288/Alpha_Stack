import numpy as np
import pandas as pd
import pytest

from models.final_holdout import (
    INDEX_FEATURES,
    STOCK_FEATURES,
    FinalModelConfig,
    run_final_holdout,
)


def _config() -> FinalModelConfig:
    return FinalModelConfig(
        schema_version=1,
        index_combination="C",
        index_model="LogisticRegression",
        index_features=INDEX_FEATURES,
        index_class_weight="balanced",
        index_up_threshold=0.3375,
        stock_combination="K",
        stock_model="LogisticRegression",
        stock_features=STOCK_FEATURES,
        stock_class_weight="balanced",
    )


def _rows(features: tuple[str, ...], dates: list[str], *, codes: list[str] | None = None):
    records = []
    for date_index, date in enumerate(dates):
        selected_codes = codes or [None]
        for code_index, code in enumerate(selected_codes):
            label = (-1, 0, 1)[(date_index + code_index) % 3]
            row = {"bas_dd": date, "label_numeric": label}
            if code is not None:
                row["code"] = code
            for feature_index, feature in enumerate(features):
                row[feature] = float(label * 3 + feature_index * 0.01 + date_index * 0.001)
            records.append(row)
    return pd.DataFrame(records)


def test_확정된_C와_K만_전체개발구간으로학습해_결합후보를출력한다():
    index_dev = _rows(INDEX_FEATURES, [f"202301{day:02d}" for day in range(1, 31)])
    index_holdout = _rows(INDEX_FEATURES, ["20230201", "20230202", "20230203"])
    stock_dev = _rows(
        STOCK_FEATURES,
        [f"202301{day:02d}" for day in range(1, 31)],
        codes=["000001", "000002"],
    )
    stock_holdout = _rows(
        STOCK_FEATURES,
        ["20230201", "20230202", "20230203"],
        codes=["000001", "000002"],
    )

    result = run_final_holdout(index_dev, index_holdout, stock_dev, stock_holdout, _config())

    assert len(result.index_predictions) == 3
    assert len(result.stock_predictions) == 6
    assert len(result.combined_predictions) == 6
    assert result.combined_predictions["buy_candidate"].dtype == bool
    assert set(result.metrics) == {"index", "stock"}
    assert len(result.config_sha256) == 64


def test_확정조합이아닌설정은_홀드아웃을보기전에중단한다():
    config = _config()
    wrong = FinalModelConfig(**{**config.__dict__, "index_combination": "B"})

    with pytest.raises(ValueError, match="확정값과 다릅니다"):
        wrong.validate()


def test_개발구간과평가구간이겹치면_중단한다():
    index_dev = _rows(INDEX_FEATURES, ["20230101", "20230102", "20230103"] * 3)
    index_holdout = _rows(INDEX_FEATURES, ["20230103"])
    stock_dev = _rows(
        STOCK_FEATURES,
        ["20230101", "20230102", "20230103"] * 3,
        codes=["000001"],
    )
    stock_holdout = _rows(STOCK_FEATURES, ["20230103"], codes=["000001"])

    with pytest.raises(ValueError, match="겹치거나 역전"):
        run_final_holdout(index_dev, index_holdout, stock_dev, stock_holdout, _config())


def test_평가일에지수예측이없으면_종목결과를조용히버리지않는다():
    dates = [f"202301{day:02d}" for day in range(1, 31)]
    index_dev = _rows(INDEX_FEATURES, dates)
    stock_dev = _rows(STOCK_FEATURES, dates, codes=["000001"])
    index_holdout = _rows(INDEX_FEATURES, ["20230201"])
    stock_holdout = _rows(STOCK_FEATURES, ["20230201", "20230202"], codes=["000001"])

    with pytest.raises(ValueError, match="KOSPI200 예측이 없습니다"):
        run_final_holdout(index_dev, index_holdout, stock_dev, stock_holdout, _config())


def test_피처결측은_자동대체하지않고중단한다():
    dates = [f"202301{day:02d}" for day in range(1, 31)]
    index_dev = _rows(INDEX_FEATURES, dates)
    stock_dev = _rows(STOCK_FEATURES, dates, codes=["000001"])
    index_holdout = _rows(INDEX_FEATURES, ["20230201"])
    stock_holdout = _rows(STOCK_FEATURES, ["20230201"], codes=["000001"])
    stock_holdout.loc[0, STOCK_FEATURES[0]] = np.nan

    with pytest.raises(ValueError, match="결측 또는 무한대"):
        run_final_holdout(index_dev, index_holdout, stock_dev, stock_holdout, _config())
