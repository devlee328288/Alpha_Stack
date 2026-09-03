"""개별 종목 랭킹 모델에 넣을 시점 정합 학습 표를 만든다.

업종 매핑이 준비되기 전 MVP는 매 거래일 KOSPI 보통주 중 시가총액 상위 50개를
후보로 삼는다. 이 모듈은 종목 피처를 만들기 전 단계인 후보 선정과 5거래일 라벨만
담당한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype

from evaluation.horizon import HOLDOUT_START

DEFAULT_TOP_N = 50
STOCK_LABEL_HORIZON = 5
STOCK_NEUTRAL_BAND = 0.02
LABEL_TO_NUMBER = {"하락": -1, "중립": 0, "상승": 1}

REQUIRED_COLUMNS = {
    "bas_dd",
    "code",
    "market",
    "market_cap",
    "adj_close",
    "is_common_stock",
}


def _normalize_dates(frame: pd.DataFrame) -> pd.Series:
    dates = (
        frame["bas_dd"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(8)
    )
    invalid = dates.isna() | ~dates.str.fullmatch(r"\d{8}")
    if invalid.any():
        examples = frame.loc[invalid, "bas_dd"].head(3).tolist()
        raise ValueError(f"YYYYMMDD로 해석할 수 없는 기준일이 있습니다: {examples}")
    return dates


def _validate_arguments(
    *,
    holdout_start: str,
    top_n: int,
    horizon: int,
    neutral_band: float,
) -> None:
    if not isinstance(holdout_start, str) or not holdout_start.isdigit():
        raise ValueError(f"holdout_start는 YYYYMMDD 문자열이어야 합니다: {holdout_start!r}")
    if len(holdout_start) != 8:
        raise ValueError(f"holdout_start는 YYYYMMDD 문자열이어야 합니다: {holdout_start!r}")
    if top_n <= 0:
        raise ValueError(f"top_n은 1 이상이어야 합니다: {top_n}")
    if horizon <= 0:
        raise ValueError(f"horizon은 1 이상이어야 합니다: {horizon}")
    if neutral_band < 0.0:
        raise ValueError(f"neutral_band는 0 이상이어야 합니다: {neutral_band}")


def build_stock_training_frame(
    daily_prices: pd.DataFrame,
    *,
    holdout_start: str = HOLDOUT_START,
    top_n: int = DEFAULT_TOP_N,
    horizon: int = STOCK_LABEL_HORIZON,
    neutral_band: float = STOCK_NEUTRAL_BAND,
) -> pd.DataFrame:
    """날짜별 KOSPI 보통주 시총 상위 N개의 5거래일 라벨을 만든다.

    ``is_common_stock``은 종목명이나 코드 모양으로 추측하지 않고 데이터 계층에서
    판정해 넘겨야 한다. 현재 HF 반출본에는 정식 종목유형이 없으므로 이 열 없이
    조용히 전체 종목을 쓰지 않고 즉시 중단한다.

    라벨의 미래 날짜는 종목별 행 번호가 아니라 KOSPI 전체 거래일 달력으로 센다.
    거래정지로 정확한 미래 거래일 가격이 없으면 그 후보를 버리되, 미래 가격이 있는
    시총 차순위 종목으로 채우지 않는다. 그렇게 채우면 미래의 거래 가능 여부로 오늘의
    후보를 바꾸는 누수가 생긴다.
    """
    _validate_arguments(
        holdout_start=holdout_start,
        top_n=top_n,
        horizon=horizon,
        neutral_band=neutral_band,
    )
    missing = REQUIRED_COLUMNS - set(daily_prices.columns)
    if missing:
        raise ValueError(f"개별 종목 학습표 필수 열이 없습니다: {sorted(missing)}")
    if not is_bool_dtype(daily_prices["is_common_stock"].dtype):
        raise TypeError("is_common_stock은 추측 문자열이 아니라 bool 열이어야 합니다.")
    if daily_prices["is_common_stock"].isna().any():
        raise ValueError("is_common_stock에 판정되지 않은 행이 있습니다.")

    source = daily_prices.copy()
    source["bas_dd"] = _normalize_dates(source)
    if (source["bas_dd"] >= holdout_start).any():
        first = str(source.loc[source["bas_dd"] >= holdout_start, "bas_dd"].min())
        raise RuntimeError(f"개별 종목 원천에 홀드아웃 행이 들어 있습니다: {first}")

    kospi = source.loc[source["market"].eq("KOSPI")].copy()
    if kospi.empty:
        raise ValueError("KOSPI 개별 종목 행이 없습니다.")
    if kospi.duplicated(["bas_dd", "code"]).any():
        duplicate = kospi.loc[
            kospi.duplicated(["bas_dd", "code"], keep=False), ["bas_dd", "code"]
        ].iloc[0]
        raise ValueError(
            "KOSPI에 같은 날짜·종목코드가 두 번 이상 있습니다: "
            f"{duplicate['bas_dd']} {duplicate['code']}"
        )

    calendar = sorted(kospi["bas_dd"].unique().tolist())
    if len(calendar) <= horizon:
        raise ValueError(
            f"거래일이 라벨 지평보다 짧습니다: 거래일 {len(calendar)}개, 지평 {horizon}개"
        )

    kospi["market_cap"] = pd.to_numeric(kospi["market_cap"], errors="coerce")
    kospi["adj_close"] = pd.to_numeric(kospi["adj_close"], errors="coerce")
    valid_price = (
        kospi["market_cap"].notna()
        & kospi["adj_close"].notna()
        & (kospi["market_cap"] > 0.0)
        & (kospi["adj_close"] > 0.0)
    )
    priced = kospi.loc[valid_price].copy()
    common = priced.loc[priced["is_common_stock"]].copy()
    if common.empty:
        raise ValueError("KOSPI 보통주로 판정된 유효 행이 없습니다.")

    # 먼저 오늘 알 수 있는 시가총액으로 후보를 확정한다. 같은 시총이면 코드순으로
    # 고정해 실행할 때마다 동일한 50종목이 나오게 한다.
    ranked = common.sort_values(
        ["bas_dd", "market_cap", "code"],
        ascending=[True, False, True],
        kind="stable",
    )
    ranked["market_cap_rank"] = ranked.groupby("bas_dd", sort=False).cumcount() + 1
    candidates = ranked.loc[ranked["market_cap_rank"] <= top_n].copy()

    calendar_frame = pd.DataFrame({"bas_dd": calendar})
    calendar_frame["future_bas_dd"] = calendar_frame["bas_dd"].shift(-horizon)
    candidates = candidates.merge(
        calendar_frame,
        on="bas_dd",
        how="left",
        validate="many_to_one",
    )
    selected_rows = len(candidates)
    # 마지막 horizon 거래일은 개발구간 안에 청산일이 없다. 빈 날짜끼리 조인하면
    # pandas가 여러 NaN을 같은 키로 세므로, 가격표를 붙이기 전에 명시적으로 비운다.
    candidates = candidates.loc[candidates["future_bas_dd"].notna()].copy()

    future_prices = priced.loc[:, ["bas_dd", "code", "adj_close"]].rename(
        columns={"bas_dd": "future_bas_dd", "adj_close": "future_adj_close"}
    )
    candidates = candidates.merge(
        future_prices,
        on=["future_bas_dd", "code"],
        how="left",
        validate="one_to_one",
    )
    candidates = candidates.loc[candidates["future_adj_close"].notna()].copy()
    if candidates.empty:
        raise ValueError("정확히 미래 5거래일 가격이 있는 학습 후보가 없습니다.")

    candidates["fwd_return_5d"] = (
        candidates["future_adj_close"] / candidates["adj_close"] - 1.0
    )
    up = candidates["future_adj_close"] > candidates["adj_close"] * (1.0 + neutral_band)
    down = candidates["future_adj_close"] < candidates["adj_close"] * (1.0 - neutral_band)
    candidates["label"] = np.select([up, down], ["상승", "하락"], default="중립")
    candidates["label_numeric"] = candidates["label"].map(LABEL_TO_NUMBER).astype("int8")
    candidates["market_cap_rank"] = candidates["market_cap_rank"].astype("int16")
    candidates = candidates.sort_values(
        ["bas_dd", "market_cap_rank", "code"], kind="stable"
    ).reset_index(drop=True)

    if candidates["bas_dd"].max() >= holdout_start:
        raise RuntimeError("개별 종목 학습표에 홀드아웃 행이 들어왔습니다.")
    candidates.attrs["stock_training_filter"] = {
        "holdout_start": holdout_start,
        "top_n": top_n,
        "horizon": horizon,
        "neutral_band": neutral_band,
        "source_rows": int(len(daily_prices)),
        "kospi_rows": int(len(kospi)),
        "invalid_price_rows": int((~valid_price).sum()),
        "common_stock_rows": int(len(common)),
        "selected_rows": int(selected_rows),
        "missing_future_rows": int(selected_rows - len(candidates)),
        "training_rows": int(len(candidates)),
        "first_date": str(candidates["bas_dd"].min()),
        "last_date": str(candidates["bas_dd"].max()),
    }
    return candidates
