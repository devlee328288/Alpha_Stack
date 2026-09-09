import pandas as pd
import pytest

from supply.stock_training_universe import (
    ELIGIBLE_INDUSTRY_INDICES,
    attach_common_stock,
    build_sector_candidate_frame,
    filter_extreme_adjusted_returns,
)


def test_이름이_우로끝나는_보통주를_주권종류가_구해낸다():
    """옛 이름 규칙이 우선주로 잘못 뺐던 넷. 주권종류는 바로 답한다.

    `미래에셋대우` 는 20200102 코스피 시총 48위라, 잘못 빠지면 상위 50 후보가
    조용히 한 종목 줄어든다.
    """
    frame = pd.DataFrame(
        {
            "bas_dd": ["20200102"] * 4,
            "code": ["006800", "025620", "000327", "000335"],
            "name": ["미래에셋대우", "신우", "디피아이홀딩스2B", "삼성전자우"],
            "kind_stkcert_tp_nm": ["보통주", "보통주", "신형우선주", "구형우선주"],
        }
    )

    result = attach_common_stock(frame).set_index("code")

    assert bool(result.loc["006800", "is_common_stock"]) is True
    assert bool(result.loc["025620", "is_common_stock"]) is True
    assert bool(result.loc["000327", "is_common_stock"]) is False
    assert bool(result.loc["000335", "is_common_stock"]) is False


def test_모르는_주권종류는_보통주로_보지않는다():
    """빈 값을 보통주로 치면 우선주가 후보에 조용히 섞인다."""
    frame = pd.DataFrame(
        {
            "bas_dd": ["20200102"] * 2,
            "code": ["005930", "999999"],
            "name": ["삼성전자", "이름만있는것"],
            "kind_stkcert_tp_nm": ["보통주", None],
        }
    )

    result = attach_common_stock(frame).set_index("code")

    assert bool(result.loc["005930", "is_common_stock"]) is True
    assert bool(result.loc["999999", "is_common_stock"]) is False


def test_주권종류칸이_없으면_이름으로_되돌아가지않고_멈춘다():
    """2026-09-09 이전 반출본(28칸)을 그대로 넣은 경우다.

    조용히 종목명 규칙으로 되돌아가면 팀 기준선과 다른 표본으로 학습하게 된다.
    """
    frame = pd.DataFrame(
        {"bas_dd": ["20240102"], "code": ["005930"], "name": ["삼성전자"]}
    )

    with pytest.raises(ValueError, match="kind_stkcert_tp_nm"):
        attach_common_stock(frame)


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
                    "kind_stkcert_tp_nm": "보통주",
                }
            )
    # 가장 큰 종목이어도 우선주는 업종별 다섯 종목에 들어오면 안 된다.
    stock_rows.append(
        {
            "bas_dd": date,
            "code": "999995",
            "name": "가상기업우",
            "market": "KOSPI",
            "market_cap": 1_000_000.0,
            "industry": sectors[0],
            "kind_stkcert_tp_nm": "구형우선주",
        }
    )
    # 🔴 이름은 '우' 로 끝나지 않는데 우선주인 종목. 옛 이름 규칙은 이것을 못 걸렀다.
    stock_rows.append(
        {
            "bas_dd": date,
            "code": "999994",
            "name": "가상기업2우B",
            "market": "KOSPI",
            "market_cap": 999_999.0,
            "industry": sectors[1],
            "kind_stkcert_tp_nm": "신형우선주",
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
    assert "999994" not in set(result["code"])


def test_후보입력에홀드아웃행이있으면중단한다():
    daily = pd.DataFrame(
        {
            "bas_dd": ["20240902"],
            "code": ["005930"],
            "name": ["삼성전자"],
            "market": ["KOSPI"],
            "market_cap": [1.0],
            "industry": ["전기전자"],
            "kind_stkcert_tp_nm": ["보통주"],
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

    with pytest.warns(DeprecationWarning, match="폐기"):
        result = filter_extreme_adjusted_returns(candidates, prices)

    assert list(zip(result["bas_dd"], result["code"], strict=True)) == [
        ("20240104", "000010"),
        ("20240103", "000020"),
    ]
    assert result.attrs["extreme_return_filter"]["source_extreme_rows"] == 1
    assert result.attrs["extreme_return_filter"]["removed_candidate_rows"] == 1


def test_극단수익률필터는_폐기됐고_무엇을_대신쓸지_알려준다():
    """크기로 자르는 방식은 폐기했다(#132·#166). 지우지는 않았으니 경고로 알린다.

    노트북 `06/10` 이 두 방식을 나란히 재현하므로 함수 자체는 남는다. 다만 새 코드가
    모르고 쓰지 않도록, 경고가 **무엇을 대신 쓸지**까지 말해야 한다.
    """
    prices = pd.DataFrame(
        {
            "bas_dd": ["20240102", "20240103"],
            "code": ["000010", "000010"],
            "adj_close": [100.0, 250.0],
        }
    )
    candidates = pd.DataFrame({"bas_dd": ["20240103"], "code": ["000010"]})

    with pytest.warns(DeprecationWarning) as 기록:
        filter_extreme_adjusted_returns(candidates, prices)

    말 = str(기록[0].message)
    assert "폐기" in 말
    assert "attach_adjustment_quality" in 말        # 대신 쓸 것을 짚어 준다
    assert "is_adj_suspect" in 말
