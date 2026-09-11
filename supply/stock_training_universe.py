"""HF 개발본에서 종목 패널 학습 후보를 만든다.

이 모듈은 로컬 수집 DB를 읽지 않는다. 입력은 HF의 ``daily_price_dev.parquet``와
``index_price_dev.parquet``에서 읽은 표다.

보통주 판정은 **HF 일별시세의 ``kind_stkcert_tp_nm`` 칸**이 한다 (2026-09-09 · #186 ①).

그전에는 개발본에 주권종류가 없어서 **종목명이 '우'로 끝나는지**로 추측하고, 어긋나는
10건을 감사 예외 목록으로 기웠다. 그 방식에는 두 가지 문제가 있었다.

    ① 목록이 대조한 날짜(20260831)까지만 유효해서, 그 뒤 날짜는 판정을 거부했다
    ② 새로 상장하는 '우'로 끝나는 보통주가 나오면 다시 사람이 대조해야 했다

반출본에 주권종류를 실으면서 둘 다 사라졌다. 실측으로 판정이 바뀌지 않는 것을 먼저
확인했다 — 개발구간 7,888,945행 전량에서 **이름 규칙+예외 10건과 주권종류의 불일치가
0행**이다. 즉 유니버스가 한 종목도 달라지지 않는다.

후보 규칙은 이슈 #92·#132에서 합의한 MVP다.

    세부 업종지수 시가총액 상위 10개
    -> 각 업종의 KOSPI 보통주 시가총액 상위 5개
    -> 날짜당 최대 50종목

``제조``와 ``금융``은 여러 세부 업종을 합친 지수라 중복 집계를 막기 위해 후보
업종에서 제외한다.
"""

from __future__ import annotations

import warnings

import pandas as pd

from evaluation.horizon import HOLDOUT_START
from supply.sector import index_name_for

#: KRX 종목기본정보의 주권종류 값. 이것과 정확히 같아야 보통주다.
COMMON_STOCK_NAME = "보통주"

# 제조·금융은 상위 묶음이라 뺀다. 2025년 체계에서 새로 생긴 IT 서비스·부동산·
# 오락·문화도 개발구간의 19개 업종 비교와 조건을 맞추기 위해 넣지 않는다.
ELIGIBLE_INDUSTRY_INDICES = frozenset(
    {
        "건설",
        "금속",
        "기계·장비",
        "보험",
        "비금속",
        "섬유·의류",
        "운송·창고",
        "운송장비·부품",
        "유통",
        "음식료·담배",
        "의료·정밀기기",
        "일반서비스",
        "전기·가스",
        "전기전자",
        "제약",
        "종이·목재",
        "증권",
        "통신",
        "화학",
    }
)

DAILY_REQUIRED = {"bas_dd", "code", "name", "market", "market_cap", "industry",
                  "kind_stkcert_tp_nm"}
INDEX_REQUIRED = {"bas_dd", "index_name", "index_class", "market_cap"}


def _normalize_dates(values: pd.Series, *, column: str) -> pd.Series:
    dates = (
        values.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(8)
    )
    invalid = dates.isna() | ~dates.str.fullmatch(r"\d{8}")
    if invalid.any():
        examples = values.loc[invalid].head(3).tolist()
        raise ValueError(f"{column}을 YYYYMMDD로 해석할 수 없습니다: {examples}")
    return dates


def attach_common_stock(daily_prices: pd.DataFrame) -> pd.DataFrame:
    """HF 개발본에 ``is_common_stock``을 붙인다 — **KRX 주권종류가 정한다.**

    ``kind_stkcert_tp_nm``이 정확히 ``"보통주"``인 행만 참이다. 우선주(구형·신형)와
    종류주권은 거짓이다.

    🔴 **모르는 주권종류는 보통주가 아니라고 본다.** 빈 값을 보통주로 치면 우선주가
       후보에 섞이는데, 빠진 종목은 개수로 드러나지만 섞인 종목은 성능이 조금
       이상해질 뿐 아무 데도 안 걸린다.

    🔴 **칸이 없으면 추측하지 않고 멈춘다.** 2026-09-09 이전 반출본에는 이 칸이 없다.
       옛 파일을 그대로 넣으면 종목명으로 되돌아가는 대신 여기서 터진다 — 조용히
       다른 표본으로 학습하는 것보다 낫다.
    """

    required = {"bas_dd", "code", "name", "kind_stkcert_tp_nm"}
    missing = required - set(daily_prices.columns)
    if missing:
        raise ValueError(
            f"보통주 판정에 필요한 열이 없습니다: {sorted(missing)}. "
            "2026-09-09 이후 반출본(daily_price_dev.parquet 34칸)을 쓰세요 — "
            "그 전 파일에는 주권종류가 없습니다."
        )

    out = daily_prices.copy()
    out["bas_dd"] = _normalize_dates(out["bas_dd"], column="bas_dd")
    out["code"] = out["code"].astype("string").str.strip()
    종류 = out["kind_stkcert_tp_nm"].astype("string").str.strip()
    out["is_common_stock"] = 종류.eq(COMMON_STOCK_NAME).fillna(False).astype(bool)
    out.attrs["common_stock_source"] = (
        "KRX 종목기본정보 kind_stkcert_tp_nm (개발구간 커버리지 100.0000%)"
    )
    return out


def build_sector_candidate_frame(
    daily_prices: pd.DataFrame,
    index_prices: pd.DataFrame,
    *,
    holdout_start: str = HOLDOUT_START,
    allow_unsealed: bool = False,
    sector_count: int = 10,
    stocks_per_sector: int = 5,
) -> pd.DataFrame:
    """날짜별 세부 업종 상위 N개와 업종별 보통주 상위 M개를 고른다."""

    missing_daily = DAILY_REQUIRED - set(daily_prices.columns)
    missing_index = INDEX_REQUIRED - set(index_prices.columns)
    if missing_daily:
        raise ValueError(f"종목 후보 입력 열이 없습니다: {sorted(missing_daily)}")
    if missing_index:
        raise ValueError(f"업종지수 입력 열이 없습니다: {sorted(missing_index)}")
    if sector_count <= 0 or stocks_per_sector <= 0:
        raise ValueError("sector_count와 stocks_per_sector는 1 이상이어야 합니다.")

    daily_dates = _normalize_dates(daily_prices["bas_dd"], column="bas_dd")
    if not allow_unsealed and (daily_dates >= holdout_start).any():
        raise RuntimeError("종목 후보 원천에 홀드아웃 행이 들어 있습니다.")
    # 업종 후보는 KOSPI만 사용한다. 788만 행 전체에 종목명 판정을 적용하면
    # KOSDAQ·KONEX 메모리까지 불필요하게 복제하므로 시장을 먼저 줄인다.
    kospi_rows = daily_prices["market"].eq("KOSPI")
    stock_input = daily_prices.loc[kospi_rows].copy()
    stock_input["bas_dd"] = daily_dates.loc[kospi_rows]
    stocks = attach_common_stock(stock_input)
    indices = index_prices.copy()
    indices["bas_dd"] = _normalize_dates(indices["bas_dd"], column="bas_dd")
    if not allow_unsealed and (indices["bas_dd"] >= holdout_start).any():
        raise RuntimeError("업종지수 원천에 홀드아웃 행이 들어 있습니다.")

    indices["market_cap"] = pd.to_numeric(indices["market_cap"], errors="coerce")
    sector_indices = indices.loc[
        indices["index_class"].eq("KOSPI")
        & indices["index_name"].isin(ELIGIBLE_INDUSTRY_INDICES)
        & indices["market_cap"].notna()
        & indices["market_cap"].gt(0.0)
    ].copy()
    if sector_indices.empty:
        raise ValueError("사용할 수 있는 KOSPI 세부 업종지수가 없습니다.")
    if sector_indices.duplicated(["bas_dd", "index_name"]).any():
        raise ValueError("같은 날짜·업종지수가 두 번 이상 있습니다.")

    sector_indices = sector_indices.sort_values(
        ["bas_dd", "market_cap", "index_name"],
        ascending=[True, False, True],
        kind="stable",
    )
    sector_indices["sector_market_cap_rank"] = (
        sector_indices.groupby("bas_dd", sort=False).cumcount() + 1
    )
    selected_sectors = sector_indices.loc[
        sector_indices["sector_market_cap_rank"] <= sector_count,
        ["bas_dd", "index_name", "market_cap", "sector_market_cap_rank"],
    ].rename(
        columns={
            "index_name": "industry_index_name",
            "market_cap": "industry_market_cap",
        }
    )

    stocks["market_cap"] = pd.to_numeric(stocks["market_cap"], errors="coerce")
    usable = stocks.loc[
        stocks["market"].eq("KOSPI")
        & stocks["is_common_stock"]
        & stocks["market_cap"].notna()
        & stocks["market_cap"].gt(0.0)
        & stocks["industry"].notna()
    ].copy()
    if usable.duplicated(["bas_dd", "code"]).any():
        raise ValueError("KOSPI 종목에 같은 날짜·코드가 두 번 이상 있습니다.")
    usable["industry_index_name"] = usable["industry"].astype(str).map(index_name_for)
    candidates = usable.merge(
        selected_sectors,
        on=["bas_dd", "industry_index_name"],
        how="inner",
        validate="many_to_one",
    )
    candidates = candidates.sort_values(
        ["bas_dd", "sector_market_cap_rank", "market_cap", "code"],
        ascending=[True, True, False, True],
        kind="stable",
    )
    candidates["industry_stock_rank"] = (
        candidates.groupby(["bas_dd", "industry_index_name"], sort=False).cumcount() + 1
    )
    candidates = candidates.loc[
        candidates["industry_stock_rank"] <= stocks_per_sector
    ].copy()
    candidates["candidate_rank"] = candidates.groupby("bas_dd", sort=False).cumcount() + 1
    candidates = candidates.sort_values(
        ["bas_dd", "candidate_rank", "code"], kind="stable"
    ).reset_index(drop=True)

    candidates["sector_market_cap_rank"] = candidates["sector_market_cap_rank"].astype(
        "int16"
    )
    candidates["industry_stock_rank"] = candidates["industry_stock_rank"].astype("int16")
    candidates["candidate_rank"] = candidates["candidate_rank"].astype("int16")
    candidates.attrs["candidate_rule"] = {
        "sector_count": sector_count,
        "stocks_per_sector": stocks_per_sector,
        "max_candidates_per_day": sector_count * stocks_per_sector,
        "eligible_sector_count": len(ELIGIBLE_INDUSTRY_INDICES),
        "common_stock_source": stocks.attrs["common_stock_source"],
    }
    return candidates


def filter_extreme_adjusted_returns(
    candidates: pd.DataFrame,
    daily_prices: pd.DataFrame,
    *,
    absolute_limit: float = 1.0,
) -> pd.DataFrame:
    """⚠️ **폐기됨.** 수정종가 일간 수익률 절댓값이 한계를 넘는 후보 행을 제거한다.

    새 코드는 쓰지 않는다. 대신 :func:`supply.adj_quality.attach_adjustment_quality`
    가 붙이는 ``is_adj_suspect`` 로 거른다.

    왜 폐기했나 — **크기로 자르면 진짜 사건까지 지운다**

    이 함수는 "하루에 100% 넘게 움직였으면 이상하다" 는 **크기** 기준이다. 그런데
    하루 100% 는 실제로 일어난다 — 무상증자 권리락, 거래정지 해제, 상한가 연속이
    그렇다. 그것들은 지워야 할 오류가 아니라 **모델이 배워야 할 사건**이다.

    지금은 **어긋남**으로 가른다. KRX 가 발표한 등락률과 우리가 수정주가로 계산한
    일간 수익률이 1%p 넘게 다르면 그때만 의심한다(``is_adj_suspect``). 값이 크든
    작든 KRX 와 맞으면 진짜이고, 작아도 어긋나면 우리 계산이 틀린 것이다.

    실측(이슈 #132) — 후보군 176,705행에서 **이 필터가 지우는 행은 0건**이었고,
    ``is_adj_suspect`` 로 제외되는 행도 0건, 극단 사건 5건은 그대로 남았다. 즉 이
    필터는 후보군에서 아무 일도 하지 않으면서 "무언가 거르고 있다" 는 인상만 주고
    있었다.

    남겨 두는 이유 — 노트북
    ``06-검토·발견/10.이상치는-크기가-아니라-어긋남으로-가른다.ipynb`` 가 두 방식을
    나란히 놓고 재현한다. 지우면 그 노트북이 못 돈다.

    .. deprecated::
        ``supply.adj_quality.attach_adjustment_quality`` 를 쓰고
        ``is_adj_suspect`` 인 행만 제외한다.
    """

    warnings.warn(
        "filter_extreme_adjusted_returns 는 폐기됐습니다. 크기(절댓값 한계)로 자르면 "
        "권리락·거래정지 해제 같은 진짜 사건까지 지웁니다.\n"
        "  대신: supply.adj_quality.attach_adjustment_quality 로 is_adj_suspect 를 붙이고 "
        "그 행만 제외하십시오 — KRX 등락률과 1%p 넘게 어긋난 행만 거릅니다.\n"
        "  실측(#132): 후보군 176,705행에서 이 필터가 지우는 행은 0건입니다.",
        DeprecationWarning,
        stacklevel=2,
    )

    if absolute_limit <= 0.0:
        raise ValueError("absolute_limit은 0보다 커야 합니다.")
    candidate_missing = {"bas_dd", "code"} - set(candidates.columns)
    price_missing = {"bas_dd", "code", "adj_close"} - set(daily_prices.columns)
    if candidate_missing:
        raise ValueError(f"후보 키가 없습니다: {sorted(candidate_missing)}")
    if price_missing:
        raise ValueError(f"극단수익률 입력 열이 없습니다: {sorted(price_missing)}")

    prices = daily_prices.loc[:, ["bas_dd", "code", "adj_close"]].copy()
    prices["bas_dd"] = _normalize_dates(prices["bas_dd"], column="bas_dd")
    prices["code"] = prices["code"].astype("string").str.strip().str.zfill(6)
    if prices.duplicated(["bas_dd", "code"]).any():
        raise ValueError("극단수익률 원천에 같은 날짜·종목이 두 번 이상 있습니다.")
    prices["adj_close"] = pd.to_numeric(prices["adj_close"], errors="coerce")
    prices = prices.sort_values(["code", "bas_dd"], kind="stable")
    prices["adj_return_1d"] = prices.groupby("code", sort=False)["adj_close"].pct_change(
        fill_method=None
    )
    extreme_keys = prices.loc[
        prices["adj_return_1d"].abs().gt(absolute_limit), ["bas_dd", "code"]
    ].drop_duplicates()

    out = candidates.copy()
    out["bas_dd"] = _normalize_dates(out["bas_dd"], column="bas_dd")
    out["code"] = out["code"].astype("string").str.strip().str.zfill(6)
    marked = out.merge(
        extreme_keys.assign(_extreme_return=True),
        on=["bas_dd", "code"],
        how="left",
        validate="one_to_one",
    )
    remove = marked.pop("_extreme_return").fillna(False).astype(bool)
    filtered = marked.loc[~remove].reset_index(drop=True)
    filtered.attrs.update(candidates.attrs)
    filtered.attrs["extreme_return_filter"] = {
        "absolute_limit": absolute_limit,
        "source_extreme_rows": int(len(extreme_keys)),
        "removed_candidate_rows": int(remove.sum()),
    }
    return filtered


__all__ = [
    "ELIGIBLE_INDUSTRY_INDICES",
    "attach_common_stock",
    "build_sector_candidate_frame",
    "filter_extreme_adjusted_returns",
]
