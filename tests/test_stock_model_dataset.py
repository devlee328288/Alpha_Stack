import numpy as np
import pandas as pd
import pytest

from features.stock_model_dataset import (
    ALL_STOCK_FEATURE_COLUMNS,
    STOCK_COMBINATION_FEATURES,
    STOCK_FEATURE_COLUMNS,
    StockModelDataset,
    align_stock_feature_datasets,
    build_sector_stock_model_dataset,
    build_stock_training_frame,
)
from features.volatility import atr_ratio, historical_volatility
from features.volume import obv_slope_20


def _daily_rows(
    dates: list[str],
    codes: list[str],
    *,
    entry_prices: dict[str, float] | None = None,
    exit_prices: dict[str, float] | None = None,
) -> pd.DataFrame:
    entry_prices = entry_prices or {}
    exit_prices = exit_prices or {}
    rows = []
    for date_index, date in enumerate(dates):
        for code_index, code in enumerate(codes):
            adj_open = 100.0
            if date_index == 1:
                adj_open = entry_prices.get(code, adj_open)
            if date_index == 6:
                adj_open = exit_prices.get(code, adj_open)
            rows.append(
                {
                    "bas_dd": date,
                    "code": code,
                    "market": "KOSPI",
                    "market_cap": float((code_index + 1) * 1_000),
                    "adj_open": adj_open,
                    "is_common_stock": True,
                }
            )
    return pd.DataFrame(rows)


def test_시총이더큰우선주를빼고_보통주만후보로고른다():
    dates = pd.bdate_range("2024-01-02", periods=7).strftime("%Y%m%d").tolist()
    frame = _daily_rows(dates, ["000010", "000020"])
    preferred = frame["code"].eq("000020")
    frame.loc[preferred, "is_common_stock"] = False
    frame.loc[preferred, "market_cap"] = 1_000_000.0

    result = build_stock_training_frame(frame, top_n=1)

    assert result["code"].tolist() == ["000010"]
    assert result["market_cap_rank"].tolist() == [1]


def test_기본후보수는_날짜별시가총액상위50개다():
    dates = pd.bdate_range("2024-01-02", periods=7).strftime("%Y%m%d").tolist()
    codes = [f"{number:06d}" for number in range(60)]
    frame = _daily_rows(dates, codes)

    result = build_stock_training_frame(frame)

    assert len(result) == 50
    assert result["bas_dd"].nunique() == 1
    assert result["market_cap_rank"].tolist() == list(range(1, 51))
    assert set(result["code"]) == set(codes[-50:])


def test_t1_t6수정시가수익률을_2퍼센트밴드로분류한다():
    dates = pd.bdate_range("2024-01-02", periods=7).strftime("%Y%m%d").tolist()
    frame = _daily_rows(
        dates,
        ["000010", "000020", "000030", "000040"],
        exit_prices={
            "000010": 103.0,
            "000020": 97.0,
            "000030": 102.0,
            "000040": 98.0,
        },
    )

    result = build_stock_training_frame(frame, top_n=4).set_index("code")

    assert result.loc["000010", "label"] == "상승"
    assert result.loc["000020", "label"] == "하락"
    assert result.loc["000030", "label"] == "중립"
    assert result.loc["000040", "label"] == "중립"
    assert np.isclose(result.loc["000010", "fwd_return_5d"], 0.03)
    assert result.loc["000010", "entry_bas_dd"] == dates[1]
    assert result.loc["000010", "exit_bas_dd"] == dates[6]
    assert result.loc["000010", "label_numeric"] == 1
    assert result.loc["000020", "label_numeric"] == -1


def test_t일수정시가는_라벨계산에쓰지않는다():
    dates = pd.bdate_range("2024-01-02", periods=7).strftime("%Y%m%d").tolist()
    frame = _daily_rows(dates, ["000010"], entry_prices={"000010": 200.0})
    frame.loc[frame["bas_dd"] == dates[0], "adj_open"] = 1.0
    frame.loc[frame["bas_dd"] == dates[6], "adj_open"] = 202.0

    result = build_stock_training_frame(frame, top_n=1)

    assert np.isclose(result.loc[0, "fwd_return_5d"], 0.01)
    assert result.loc[0, "label"] == "중립"


def test_거래정지로정확한진입청산가격이없으면_다음행을대신쓰지않는다():
    dates = pd.bdate_range("2024-01-02", periods=8).strftime("%Y%m%d").tolist()
    # 000020은 시장 거래일을 보존하는 작은 종목이고, 000010이 시총 1위 후보이다.
    frame = _daily_rows(dates, ["000020", "000010"])
    frame = frame.loc[
        ~((frame["bas_dd"] == dates[6]) & (frame["code"] == "000010"))
    ].copy()
    frame.loc[frame["bas_dd"] == dates[7], "adj_open"] = 120.0

    result = build_stock_training_frame(frame, top_n=1)

    assert dates[0] not in set(result["bas_dd"])
    assert result["bas_dd"].tolist() == [dates[1]]
    assert result["entry_bas_dd"].tolist() == [dates[2]]
    assert result["exit_bas_dd"].tolist() == [dates[7]]


def test_홀드아웃행이원천에하나라도있으면_필터하지않고중단한다():
    dates = [
        "20240823",
        "20240826",
        "20240827",
        "20240828",
        "20240829",
        "20240830",
        "20240902",
    ]
    frame = _daily_rows(dates, ["000010"])

    with pytest.raises(RuntimeError, match="홀드아웃 행"):
        build_stock_training_frame(frame)


def test_보통주판정열없이는_코드나종목명으로추측하지않는다():
    dates = pd.bdate_range("2024-01-02", periods=7).strftime("%Y%m%d").tolist()
    frame = _daily_rows(dates, ["000010"]).drop(columns="is_common_stock")

    with pytest.raises(ValueError, match="is_common_stock"):
        build_stock_training_frame(frame)


def test_같은날짜와종목코드가중복되면중단한다():
    dates = pd.bdate_range("2024-01-02", periods=7).strftime("%Y%m%d").tolist()
    frame = _daily_rows(dates, ["000010"])
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="두 번 이상"):
        build_stock_training_frame(frame)


def _panel_prices(periods: int = 90) -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=periods).strftime("%Y%m%d")
    rows = []
    for code, offset in (("000010", 0.0), ("000020", 20.0)):
        for index, date in enumerate(dates):
            close = 100.0 + offset + index * 0.3 + np.sin(index / 3.0)
            rows.append(
                {
                    "bas_dd": date,
                    "code": code,
                    "market": "KOSPI",
                    "adj_open": close * 0.999,
                    "adj_high": close * 1.01,
                    "adj_low": close * 0.99,
                    "adj_close": close,
                    "volume": 10_000.0 + index * 10.0,
                }
            )
    return pd.DataFrame(rows)


def test_종목피처는_후보행사이가아니라_전체종목시계열에서계산한다():
    prices = _panel_prices()
    dates = sorted(prices["bas_dd"].unique())
    candidates = pd.DataFrame(
        {
            "bas_dd": [dates[65], dates[70], dates[65], dates[70]],
            "code": ["000010", "000010", "000020", "000020"],
            "candidate_rank": [1, 1, 2, 2],
        }
    )

    dataset = build_sector_stock_model_dataset(prices, candidates)
    result = dataset.frame.set_index(["bas_dd", "code"])
    source = prices.loc[prices["code"].eq("000010")].set_index("bas_dd")
    expected = source.loc[dates[70], "adj_close"] / source.loc[dates[69], "adj_close"] - 1.0

    assert tuple(dataset.feature_columns) == STOCK_FEATURE_COLUMNS
    assert np.isclose(result.loc[(dates[70], "000010"), "daily_return"], expected)
    assert np.isfinite(dataset.x.to_numpy()).all()
    assert dataset.groups.tolist() == [dates[65], dates[65], dates[70], dates[70]]


def test_종목패널라벨은_시장달력의_t1과_t6수정시가를쓴다():
    prices = _panel_prices()
    dates = sorted(prices["bas_dd"].unique())
    candidates = pd.DataFrame(
        {"bas_dd": [dates[70]], "code": ["000010"], "candidate_rank": [1]}
    )

    row = build_sector_stock_model_dataset(prices, candidates).frame.iloc[0]

    assert row["entry_bas_dd"] == dates[71]
    assert row["exit_bas_dd"] == dates[76]
    assert np.isclose(row["entry_adj_open"], prices.iloc[71]["adj_open"])
    assert np.isclose(row["exit_adj_open"], prices.iloc[76]["adj_open"])


def test_종목패널은_홀드아웃행을조용히자르지않고중단한다():
    prices = _panel_prices()
    extra = prices.iloc[[0]].copy()
    extra["bas_dd"] = "20240902"
    candidates = pd.DataFrame({"bas_dd": [prices.iloc[70]["bas_dd"]], "code": ["000010"]})

    with pytest.raises(RuntimeError, match="홀드아웃 행"):
        build_sector_stock_model_dataset(pd.concat([prices, extra]), candidates)


def test_종목피처는_인라인이아니라_원자함수와같은값을낸다():
    """같은 공식을 두 곳에 두면 한쪽만 고쳐졌을 때 값이 조용히 갈린다(#155).

    이 시험은 데이터셋이 내는 값이 원자 함수 출력과 같은지를 본다. 누군가 다시
    손으로 짜 넣으면 여기서 걸린다.
    """
    prices = _panel_prices()
    dates = sorted(prices["bas_dd"].unique())
    candidates = pd.DataFrame(
        {
            "bas_dd": [dates[65], dates[70]],
            "code": ["000010", "000010"],
            "candidate_rank": [1, 1],
        }
    )

    dataset = build_sector_stock_model_dataset(prices, candidates)
    result = dataset.frame.set_index(["bas_dd", "code"])

    source = prices.loc[prices["code"].eq("000010")].sort_values("bas_dd")
    position = {date: index for index, date in enumerate(source["bas_dd"])}
    expected_atr = atr_ratio(
        source["adj_high"].to_numpy(dtype=float),
        source["adj_low"].to_numpy(dtype=float),
        source["adj_close"].to_numpy(dtype=float),
        14,
    )
    expected_obv = obv_slope_20(
        source["adj_close"].to_numpy(dtype=float),
        source["volume"].to_numpy(dtype=float),
        20,
    )
    expected_hv = historical_volatility(source["adj_close"].to_numpy(dtype=float), 20)

    for date in (dates[65], dates[70]):
        index = position[date]
        assert np.isclose(result.loc[(date, "000010"), "atr_ratio"], expected_atr[index])
        assert np.isclose(result.loc[(date, "000010"), "obv_slope_20"], expected_obv[index])
        assert np.isclose(result.loc[(date, "000010"), "hv_20"], expected_hv[index])


def test_조합b부터g까지_수정주가와당일횡단면만으로계산한다():
    prices = _panel_prices(periods=100)
    prices["industry"] = "건설"
    prices["value"] = prices["volume"] * prices["adj_close"]
    prices["market_cap"] = np.where(prices["code"].eq("000010"), 2e12, 1e12)
    dates = sorted(prices["bas_dd"].unique())
    candidates = prices.loc[prices["bas_dd"].isin(dates[65:90])].copy()
    candidates["industry_index_name"] = "건설"
    candidates["sector_market_cap_rank"] = 1
    candidates["industry_stock_rank"] = candidates["code"].map(
        {"000010": 1, "000020": 2}
    )
    candidates["candidate_rank"] = candidates["industry_stock_rank"]
    index_rows = []
    for index, date in enumerate(dates):
        for name, scale in (("건설", 1.2), ("코스피 200", 1.0)):
            index_rows.append(
                {
                    "bas_dd": date,
                    "index_name": name,
                    "index_class": "KOSPI",
                    "close": (200.0 + index * 0.4) * scale,
                }
            )

    dataset = build_sector_stock_model_dataset(
        prices,
        candidates,
        index_prices=pd.DataFrame(index_rows),
        feature_columns=ALL_STOCK_FEATURE_COLUMNS,
    )

    assert set(STOCK_COMBINATION_FEATURES) == set("ABCDEFG")
    assert STOCK_COMBINATION_FEATURES["G"] == (
        "dist_high_60",
        "sma_gap_20_60",
        "relative_ret_5_market",
        "rsi_14",
        "hv_20",
        "turnover_20",
    )
    assert np.isfinite(dataset.x.to_numpy()).all()
    same_day = dataset.frame.groupby("bas_dd")["market_cap_percentile"]
    assert same_day.max().eq(1.0).all()
    assert same_day.min().eq(0.5).all()


def test_조합_비교는_날짜뿐_아니라_종목까지_같은_행으로_맞춘다():
    first = StockModelDataset(
        frame=pd.DataFrame(
            {
                "bas_dd": ["20200101", "20200101", "20200102"],
                "code": ["000001", "000002", "000001"],
                "label_numeric": [1, 0, -1],
                "daily_return": [0.1, 0.2, 0.3],
            }
        ),
        feature_columns=("daily_return",),
    )
    second = StockModelDataset(
        frame=pd.DataFrame(
            {
                "bas_dd": ["20200101", "20200102", "20200102"],
                "code": ["000001", "000001", "000003"],
                "label_numeric": [1, -1, 0],
                "five_day_return": [0.4, 0.5, 0.6],
            }
        ),
        feature_columns=("five_day_return",),
    )

    aligned = align_stock_feature_datasets({"A": first, "B": second})
    expected = [("20200101", "000001"), ("20200102", "000001")]

    assert list(aligned) == ["A", "B"]
    for dataset in aligned.values():
        keys = list(dataset.frame[["bas_dd", "code"]].itertuples(index=False, name=None))
        assert keys == expected
        assert dataset.frame.attrs["stock_panel"]["alignment_keys"] == ["bas_dd", "code"]


def test_공통_날짜와_종목의_라벨이_다르면_멈춘다():
    first = StockModelDataset(
        pd.DataFrame(
            {
                "bas_dd": ["20200101"],
                "code": ["000001"],
                "label_numeric": [1],
                "daily_return": [0.1],
            }
        ),
        feature_columns=("daily_return",),
    )
    second = StockModelDataset(
        pd.DataFrame(
            {
                "bas_dd": ["20200101"],
                "code": ["000001"],
                "label_numeric": [-1],
                "five_day_return": [0.2],
            }
        ),
        feature_columns=("five_day_return",),
    )

    with pytest.raises(ValueError, match="라벨이 다른 조합과 다릅니다"):
        align_stock_feature_datasets({"A": first, "B": second})
