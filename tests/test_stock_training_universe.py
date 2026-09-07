import pandas as pd
import pytest

from supply.stock_training_universe import (
    ELIGIBLE_INDUSTRY_INDICES,
    attach_audited_common_stock,
    build_sector_candidate_frame,
    filter_extreme_adjusted_returns,
)


def test_github_전량검증의_이름예외를_보통주판정에반영한다():
    frame = pd.DataFrame(
        {
            "bas_dd": ["20200102"] * 4,
            "code": ["006800", "025620", "000327", "000335"],
            "name": ["미래에셋대우", "신우", "디피아이홀딩스2B", "삼성전자우"],
        }
    )

    result = attach_audited_common_stock(frame).set_index("code")

    assert bool(result.loc["006800", "is_common_stock"]) is True
    assert bool(result.loc["025620", "is_common_stock"]) is True
    assert bool(result.loc["000327", "is_common_stock"]) is False
    assert bool(result.loc["000335", "is_common_stock"]) is False


def test_github_감사범위뒤의날짜는_추측하지않는다():
    frame = pd.DataFrame(
        {"bas_dd": ["20260901"], "code": ["005930"], "name": ["삼성전자"]}
    )

    with pytest.raises(RuntimeError, match="전량검증 범위"):
        attach_audited_common_stock(frame)


def test_업종상위10개와_업종별보통주상위5개를고른다():
    date = "20240102"
    sectors = sorted(ELIGIBLE_INDUSTRY_INDICES)[:12]
    index_rows = [
        {
            "bas_dd": date,
            "index_name": sector,
            "index_class": "KOSPI",
            "market_cap": float(1_000 - rank),
        }
        for rank, sector in enumerate(sectors)
    ]
    index_rows.extend(
        [
            {
                "bas_dd": date,
                "index_name": "제조",
                "index_class": "KOSPI",
                "market_cap": 1_000_000.0,
            },
            {
                "bas_dd": date,
                "index_name": "금융",
                "index_class": "KOSPI",
                "market_cap": 900_000.0,
            },
        ]
    )
    stock_rows = []
    number = 0
    for sector in [*sectors, "제조", "금융"]:
        for within in range(6):
            number += 1
            stock_rows.append(
                {
                    "bas_dd": date,
                    "code": f"{number:06d}",
                    "name": f"보통종목{number}",
                    "market": "KOSPI",
                    "market_cap": float(10_000 - within),
                    "industry": sector,
                }
            )
    # 가장 큰 종목이어도 이름 규칙상 우선주는 업종별 다섯 종목에 들어오면 안 된다.
    stock_rows.append(
        {
            "bas_dd": date,
            "code": "999995",
            "name": "가상기업우",
            "market": "KOSPI",
            "market_cap": 1_000_000.0,
            "industry": sectors[0],
        }
    )

    result = build_sector_candidate_frame(
        pd.DataFrame(stock_rows), pd.DataFrame(index_rows)
    )

    assert len(result) == 50
    assert result["industry_index_name"].nunique() == 10
    assert result.groupby("industry_index_name").size().eq(5).all()
    assert set(result["industry_index_name"]).isdisjoint({"제조", "금융"})
    assert "999995" not in set(result["code"])


def test_후보입력에홀드아웃행이있으면중단한다():
    daily = pd.DataFrame(
        {
            "bas_dd": ["20240902"],
            "code": ["005930"],
            "name": ["삼성전자"],
            "market": ["KOSPI"],
            "market_cap": [1.0],
            "industry": ["전기전자"],
        }
    )
    index = pd.DataFrame(
        {
            "bas_dd": ["20240830"],
            "index_name": ["전기전자"],
            "index_class": ["KOSPI"],
            "market_cap": [1.0],
        }
    )

    with pytest.raises(RuntimeError, match="홀드아웃"):
        build_sector_candidate_frame(daily, index)


def test_극단수익률은_전체종목시계열에서계산하고_해당후보만제거한다():
    prices = pd.DataFrame(
        {
            "bas_dd": [
                "20240102",
                "20240103",
                "20240104",
                "20240102",
                "20240103",
                "20240104",
            ],
            "code": ["000010", "000010", "000010", "000020", "000020", "000020"],
            "adj_close": [100.0, 250.0, 251.0, 100.0, 101.0, 102.0],
        }
    )
    candidates = pd.DataFrame(
        {
            "bas_dd": ["20240103", "20240104", "20240103"],
            "code": ["000010", "000010", "000020"],
            "candidate_rank": [1, 1, 2],
        }
    )

    result = filter_extreme_adjusted_returns(candidates, prices)

    assert list(zip(result["bas_dd"], result["code"], strict=True)) == [
        ("20240104", "000010"),
        ("20240103", "000020"),
    ]
    assert result.attrs["extreme_return_filter"]["source_extreme_rows"] == 1
    assert result.attrs["extreme_return_filter"]["removed_candidate_rows"] == 1
