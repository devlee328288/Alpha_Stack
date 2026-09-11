import pandas as pd
import pytest

from models.final_window import (
    derive_final_window,
    split_index_window,
    split_stock_window,
)


def test_마지막날짜를_T6으로삼아_거래일여섯칸전의_판단일을고른다():
    dates = [
        "20260821",
        "20260824",
        "20260825",
        "20260826",
        "20260827",
        "20260828",
        "20260831",
        "20260901",
    ]

    window = derive_final_window(dates)

    assert window.decision_date == "20260824"
    assert window.entry_date == "20260825"
    assert window.exit_date == "20260901"


def test_마지막날짜는_달력일이아니라_실제공통거래일에서고른다():
    dates = ["20260820", "20260821", "20260825", "20260826", "20260828", "20260831", "20260902"]

    window = derive_final_window(dates)

    assert window.decision_date == "20260820"
    assert window.entry_date == "20260821"
    assert window.exit_date == "20260902"


def test_지수학습라벨의_청산위치는_최종판단일을넘지않는다():
    dates = pd.bdate_range("2026-07-01", periods=20).strftime("%Y%m%d").tolist()
    frame = pd.DataFrame(
        {
            "bas_dd": dates,
            "raw_position": range(len(dates)),
            "label_numeric": [0] * len(dates),
        }
    )
    window = derive_final_window(dates)

    train, test = split_index_window(frame, window)

    decision_position = int(test["raw_position"].iloc[0])
    assert (train["raw_position"] + 6 <= decision_position).all()
    assert test["bas_dd"].tolist() == [window.decision_date]


def test_종목학습은_청산일이판단일이하인행만사용한다():
    window = derive_final_window(
        ["20260821", "20260824", "20260825", "20260826", "20260827", "20260828", "20260901"]
    )
    frame = pd.DataFrame(
        {
            "bas_dd": ["20260810", "20260811", "20260821"],
            "code": ["000001", "000001", "000001"],
            "exit_bas_dd": ["20260821", "20260824", "20260901"],
            "label_numeric": [0, 1, -1],
        }
    )

    train, test = split_stock_window(frame, window)

    assert train["bas_dd"].tolist() == ["20260810"]
    assert test["bas_dd"].tolist() == ["20260821"]


def test_최종구간거래일이일곱개보다적으면중단한다():
    with pytest.raises(ValueError, match="최소 7개"):
        derive_final_window(["20260824", "20260825", "20260826"])
