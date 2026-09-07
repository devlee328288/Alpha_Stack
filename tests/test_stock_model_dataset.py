import numpy as np
import pandas as pd
import pytest

from features.stock_model_dataset import build_stock_training_frame


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
