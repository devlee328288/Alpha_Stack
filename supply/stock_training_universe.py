"""HF 개발본에서 종목 패널 학습 후보를 만든다.

이 모듈은 로컬 수집 DB를 읽지 않는다. 입력은 HF의 ``daily_price_dev.parquet``와
``index_price_dev.parquet``에서 읽은 표다.

HF 일별시세에는 주권종류가 없다. 대신 GitHub에 저장된 기본정보 전량검증 노트북
(``notebooks/01-데이터수집/08.액면가로판정하니감자가빠졌다.ipynb``)이
9,220,879행을 이름 규칙과 대조해 어긋난 10개 (코드, 당시 이름)를 전부 남겼다.
``attach_audited_common_stock``은 그 고정된 감사 결과를 이름 규칙에 보정해 개발본의
보통주 판정을 재현한다. 새 날짜나 실시간 추론에는 쓰지 않는다.

후보 규칙은 이슈 #92·#132에서 합의한 MVP다.

    세부 업종지수 시가총액 상위 10개
    -> 각 업종의 KOSPI 보통주 시가총액 상위 5개
    -> 날짜당 최대 50종목

``제조``와 ``금융``은 여러 세부 업종을 합친 지수라 중복 집계를 막기 위해 후보
업종에서 제외한다.
"""

from __future__ import annotations

import pandas as pd

from evaluation.horizon import HOLDOUT_START
from supply.sector import index_name_for

AUDIT_END = "20260831"

# GitHub 저장 실행 출력의 전량 대조 결과다. 앞의 아홉은 이름 규칙이 우선주로
# 오인한 보통주이고, 마지막 하나는 반대로 보통주로 오인한 우선주다.
AUDITED_COMMON_EXCEPTIONS = frozenset(
    {
        ("115960", "연우"),
        ("088910", "동우"),
        ("025620", "신우"),
        ("294090", "이오플로우"),
        ("006800", "미래에셋대우"),
        ("047050", "포스코대우"),
        ("064960", "S&T대우"),
        ("458650", "성우"),
        ("159910", "에코글로우"),
    }
)
AUDITED_NON_COMMON_EXCEPTIONS = frozenset({("000327", "디피아이홀딩스2B")})

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

DAILY_REQUIRED = {"bas_dd", "code", "name", "market", "market_cap", "industry"}
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


def _looks_preferred(name: str) -> bool:
    """GitHub 전량검증에서 비교 대상으로 쓴 옛 이름 규칙을 그대로 재현한다."""

    normalized = (name or "").strip()
    return bool(normalized) and (
        normalized.endswith("우")
        or "우B" in normalized
        or "우(전환)" in normalized
        or normalized.endswith("우C")
    )


def attach_audited_common_stock(daily_prices: pd.DataFrame) -> pd.DataFrame:
    """HF 개발본에 GitHub 전량검증 근거의 ``is_common_stock``을 붙인다.

    이 함수는 2026-08-31까지 전량 대조한 고정 자료에만 유효하다. 그 뒤의 날짜를
    받으면 새 이름·주권종류를 검증하지 못했으므로 추측하지 않고 중단한다.
    """

    required = {"bas_dd", "code", "name"}
    missing = required - set(daily_prices.columns)
    if missing:
        raise ValueError(f"보통주 감사 판정에 필요한 열이 없습니다: {sorted(missing)}")

    out = daily_prices.copy()
    out["bas_dd"] = _normalize_dates(out["bas_dd"], column="bas_dd")
    if not out.empty and out["bas_dd"].max() > AUDIT_END:
        raise RuntimeError(
            f"GitHub 전량검증 범위({AUDIT_END}) 뒤의 행은 판정할 수 없습니다: "
            f"{out['bas_dd'].max()}"
        )
    out["code"] = out["code"].astype("string").str.strip().str.zfill(6)
    names = out["name"].astype("string").fillna("").str.strip()
    if names.eq("").any():
        raise ValueError("종목명이 빈 행은 GitHub 감사 규칙으로 판정할 수 없습니다.")

    is_common = ~names.map(_looks_preferred)
    pairs = pd.Series(zip(out["code"], names, strict=True), index=out.index)
    is_common.loc[pairs.isin(AUDITED_COMMON_EXCEPTIONS)] = True
    is_common.loc[pairs.isin(AUDITED_NON_COMMON_EXCEPTIONS)] = False
    out["is_common_stock"] = is_common.astype(bool)
    out.attrs["common_stock_source"] = (
        "GitHub 기본정보 전량검증 9,220,879행의 이름 규칙 예외 10개"
    )
    out.attrs["common_stock_audit_end"] = AUDIT_END
    return out


def build_sector_candidate_frame(
    daily_prices: pd.DataFrame,
    index_prices: pd.DataFrame,
    *,
    holdout_start: str = HOLDOUT_START,
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
    if (daily_dates >= holdout_start).any():
        raise RuntimeError("종목 후보 원천에 홀드아웃 행이 들어 있습니다.")
    # 업종 후보는 KOSPI만 사용한다. 788만 행 전체에 종목명 판정을 적용하면
    # KOSDAQ·KONEX 메모리까지 불필요하게 복제하므로 시장을 먼저 줄인다.
    kospi_rows = daily_prices["market"].eq("KOSPI")
    stock_input = daily_prices.loc[kospi_rows].copy()
    stock_input["bas_dd"] = daily_dates.loc[kospi_rows]
    stocks = attach_audited_common_stock(stock_input)
    indices = index_prices.copy()
    indices["bas_dd"] = _normalize_dates(indices["bas_dd"], column="bas_dd")
    if (indices["bas_dd"] >= holdout_start).any():
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
    """수정종가 일간 수익률 절댓값이 한계를 넘는 후보 행을 제거한다.

    수익률은 후보만 잘라 계산하지 않고 종목의 전체 개발 시계열에서 먼저 계산한다.
    후보로 뽑히지 않은 전날을 버린 뒤 계산하면 업종 진입일의 수익률이 며칠짜리로
    늘어나 다른 값을 재게 된다. 제거는 극단값이 발생한 정확한 날짜·종목에만 한다.
    """

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
    "AUDITED_COMMON_EXCEPTIONS",
    "AUDITED_NON_COMMON_EXCEPTIONS",
    "AUDIT_END",
    "ELIGIBLE_INDUSTRY_INDICES",
    "attach_audited_common_stock",
    "build_sector_candidate_frame",
    "filter_extreme_adjusted_returns",
]
