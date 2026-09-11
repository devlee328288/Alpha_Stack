import pandas as pd
import pytest

from models.final_holdout import FinalHoldoutResult
from models.final_window import (
    derive_final_window,
    split_index_window,
    split_stock_window,
)
from scripts.run_final_window import _build_shared_stock_rows, _shared_readme


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


def test_공유종목표는업종과업종내순위뒤문자포함종목코드순으로정렬한다():
    predictions = pd.DataFrame(
        {
            "bas_dd": ["20260824"] * 3,
            "code": ["A00002", "000001", "A00001"],
            "stock_actual": [1, 0, -1],
            "stock_predicted": [1, 1, -1],
            "p_down": [0.1, 0.2, 0.7],
            "p_neutral": [0.2, 0.3, 0.2],
            "p_up": [0.7, 0.5, 0.1],
            "index_predicted": [1] * 3,
            "index_p_up": [0.6] * 3,
            "buy_candidate": [True, True, False],
        }
    )
    stock_test = pd.DataFrame(
        {
            "bas_dd": ["20260824"] * 3,
            "code": ["A00002", "000001", "A00001"],
            "name": ["둘", "셋", "하나"],
            "industry_index_name": ["전기전자", "금융", "전기전자"],
            "sector_market_cap_rank": [1, 2, 1],
            "industry_stock_rank": [2, 1, 1],
        }
    )
    result = FinalHoldoutResult(
        index_predictions=pd.DataFrame(),
        stock_predictions=pd.DataFrame(),
        combined_predictions=predictions,
        metrics={},
        config_sha256="test",
    )

    rows = _build_shared_stock_rows(result, stock_test)

    assert rows["code"].tolist() == ["A00001", "A00002", "000001"]
    assert rows["stock_prediction"].tolist() == ["하락", "상승", "상승"]
    assert rows["actual"].tolist() == ["하락", "상승", "보합"]
    assert rows["hit"].tolist() == ["O", "O", "X"]


def test_공유README에는요청한표와매수후보가들어간다():
    rows = pd.DataFrame(
        {
            "sector_rank": [1],
            "industry": ["전기전자"],
            "name": ["예시종목"],
            "code": ["A00001"],
            "industry_market_cap_rank": [1],
            "stock_prediction": ["상승"],
            "p_up": [0.52],
            "p_flat": [0.30],
            "p_down": [0.18],
            "actual": ["상승"],
            "hit": ["O"],
            "buy_candidate": ["O"],
        }
    )
    report = {
        "window": {
            "decision_date": "20260824",
            "entry_date": "20260825",
            "exit_date": "20260901",
        }
    }
    payload = {"index": {"prediction": "상승", "actual": "상승", "hit": "O"}}

    readme = _shared_readme(report, payload, rows)

    assert "| 업종 순위 | 업종 | 종목명·코드 |" in readme
    assert "| 1 | 전기전자 | 예시종목·A00001 | 1 | 상승 | 0.5200 |" in readme
    assert "매수 후보: `1`건" in readme
