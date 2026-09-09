"""수정주가 품질 플래그 — 크기가 아니라 **KRX 등락률과의 어긋남**으로 가른다 (공급 계층).

## 왜 이 파일이 있나 — "이상치" 는 두 가지를 섞은 말이다

모델 파트가 개별종목 후보군에 `|adj_close 일간수익률| > 100%` 필터를 걸었더니 딱 한 행이
지워졌다 — 경남에너지(008020) 2016-05-11, +153.66%. 그런데 KRX 가 그날 발표한 등락률도
정확히 +153.66% 였다. 수정주가 오류가 아니라 **실제로 그렇게 움직인 날**이다. 반대로
신세계(004170) 2011-06-10 은 우리 수정주가 수익률이 −60.6% 인데 KRX 등락률은 +14.95%
다 — 인적분할 재상장일에 우리 조정계수가 틀린 것이고, 크기 필터에는 걸리지 않는다.

"이상치" 라는 말은 **데이터 오류**와 **진짜 사건**을 섞는다. 진짜 사건을 지우면 모멘텀
계열 라벨이 배워야 할 표본이 빠지고(이상치 제거가 극단 추세 종목 풀을 고갈시킨다는
지적이 있다 · 이슈 #132), 오류를 남기면 없는 −60% 가 학습된다. 그래서 판별자를
**크기**에서 **독립 원천과의 어긋남**으로 바꾼다. KRX `change_rate` 는 거래소가 그날
기준가 대비로 낸 공식 수익률이라, 우리가 만든 수정주가를 검증할 독립 원천이다.
"수정가에서 계산한 수익률은 보고된 수익률과 맞아야 한다" 는 시장 데이터의 기본
불변식이고, `scripts/build_adj_prices.py` 의 `verify()` ① 이 같은 식으로 집계 검증을 한다.

## 칸 넷

    adj_return_1d         (adj_close / 전일 adj_close − 1) × 100                 %
    adj_change_rate_gap   |adj_return_1d − change_rate|                          %p
    is_adj_suspect        gap > SUSPECT_GAP_TOLERANCE → 수정주가 오류 의심. 학습에서 뺀다
    is_extreme_return     |adj_return_1d| > EXTREME_RETURN_PCT 이고 의심이 아님
                          → 진짜 극단 사건. 남긴다

비교할 수 없는 행(종목의 첫 행 · **새 시계열의 첫 행** · 전일 adj 가 없거나 0 · 종가 0 ·
등락률 없음)은 수익률·갭이 NaN 이고 두 플래그는 False 다 — **모르는 것을 오류로 치지
않는다.**

## 🔴 "전일" 은 `shift(1)` 이 아니다 — 종목코드는 재사용된다 (2026-09-09 · 이슈 #195)

이 함수는 종목별 `shift(1)` 로 전일을 찾는다. 그런데 상장폐지된 코드를 몇 년 뒤 다른
회사가 받으면 그 `shift(1)` 은 **다른 회사의 마지막 종가**를 가리킨다.

    036220  인포피아 ~2016-05-04(3,500원)  →  오상헬스케어 2024-03-13(29,350원)
            `shift(1)` 로 계산한 adj_return_1d = **+738.57%** · is_adj_suspect 켜짐
            KRX 등락률은 +46.75% — 그 종목의 진짜 첫 거래일이라 기준가가 다르다

전 구간 9,231,938행에서 이런 자리는 **2건**이고(`036220` · `101970`) 개발구간 반출본에
드는 것은 **1행**(`036220` 2024-03-13)이다. 다른 하나는 2025-03-28 이라 홀드아웃이다.

가르는 규칙은 `common.corporate_actions.is_series_restart` 에 있다 — *거래일 공백이
있고 그 뒤에 새 상장일이 생겼으면* 새 시계열이다. 문턱이 없고 거래소가 주는 사실만
쓴다. 그 판정이 `is_first_listing` 칸으로 들어오므로 **이 함수는 그 칸을 요구한다**
(`REQUIRED_COLUMNS`). 없으면 값이 조용히 틀리는 대신 `ValueError` 가 난다.

## 시점 규칙 — T−1 과 T 만 쓴다

플래그는 그 종목의 전날과 당일 값만 본다. 뒤의 행을 잘라 내도 앞 행의 플래그는 같다
(시험 `test_뒤_행을_잘라도_앞_행의_플래그는_같다`). 그래서 개발구간 학습과 개봉 뒤
홀드아웃 예측에 **같은 함수를 그대로** 쓴다. T+1 이후에 극단값이 나왔다는 이유로 T 를
지우는 일은 여기서 일어날 수 없다.

수정주가 자체는 나중 자본변동이 과거 값을 다시 쓰는 후방조정이지만, **이웃한 두 날의
비율**은 그 뒤에 무슨 분할이 오든 변하지 않는다 — 두 값이 같은 배율로 움직이기 때문이다.

## 허용폭이 verify() 와 다른 이유 (실측 2026-09-07 · HF 반출본 7,885,483쌍)

`build_adj_prices.GAP_TOLERANCE` 는 0.15%p 다. 그것은 **전체 어긋남 비율이 1% 미만인가**를
보는 집계 게이트라 민감한 게 맞다. 행 플래그는 다르다 — 반올림 잡음을 오류로 표시하면
멀쩡한 저가주 행이 학습에서 빠진다.

    허용폭     전 시장 어긋남      후보군(업종 10 × 종목 5 · 176,705행) 교집합
    0.15%p    13,677 (0.173%)     2
    1.0%p      1,406 (0.018%)     2
    30%p         144               2

0.15~1.0 사이 12,271행의 **94% 가 종가 5,000원 미만**이다 — FDR 수정가격이 원 단위로
반올림되므로 저가주에서 0.2~0.5%p 는 반올림에서 나온다. 후보군에서는 허용폭이 무엇이든
같은 2행(둘 다 인적분할 재상장일)만 남는다. 그래서 행 플래그는 1%p 다.

극단 기준 30% 는 가격제한폭이다(2015-06-15 이후 ±30%). 반출본 일간 수익률의 99.99%
분위가 정확히 +30.00% 였다. 그 전(±15% 시절) 행에서는 15~30% 움직임이 극단으로 안
잡히는데, 이 칸은 필터가 아니라 **설명**이라 한 값으로 둔다.

## 쓰는 법 (모델 파트)

    from supply.adj_quality import attach_adjustment_quality
    cand = attach_adjustment_quality(candidates, daily)   # 후보 프레임에 칸 넷이 붙는다
    train = cand[~cand["is_adj_suspect"]]                 # 의심 행만 뺀다. 진짜 사건은 남는다
    print(cand.attrs["adjustment_quality"])               # 몇 행을 왜 뺐는지 기록

`daily` 는 HF 반출본 `daily_price_dev.parquet` 그대로면 된다 — `adj_close` ·
`change_rate` · `is_first_listing` 이 이미 들어 있다(34칸).

⚠️ DB 에서 다섯 칸만 골라 읽어 넘기면 `is_first_listing` 이 없어 `ValueError` 가 난다.
   그 경로에서는 `supply.training.attach_corporate_action_flags` 로 기업행위 3칸을 먼저
   붙인다. 반출·판정기가 그렇게 한다.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

#: 행 플래그의 허용폭(%p). 이보다 크게 어긋나면 수정주가 오류 의심이다.
#: `build_adj_prices.GAP_TOLERANCE`(0.15) 와 다른 값인 이유는 위 모듈 설명에 있다.
SUSPECT_GAP_TOLERANCE = 1.0

#: 극단 수익률의 경계(%). 가격제한폭(2015-06-15 이후 ±30%)과 같다.
EXTREME_RETURN_PCT = 30.0

#: 입력에 있어야 하는 열. HF 반출본 `daily_price_dev.parquet` 에 전부 있다.
#:
#: 🔴 `is_first_listing` 이 왜 필요한가 — **종목코드는 재사용된다.** 이 함수는 종목별
#:    `shift(1)` 로 "전일" 을 찾는데, 상장폐지된 코드를 몇 년 뒤 다른 회사가 받으면 그
#:    `shift(1)` 이 **8년 전 다른 회사의 종가**를 가리킨다. 실제로 `036220` 2024-03-13
#:    오상헬스케어의 전일이 2016-05-04 인포피아가 되어 `adj_return_1d` 가 +738.57% 로
#:    나오고 `is_adj_suspect` 가 켜졌다(2026-09-09 · 이슈 #195).
#:
#:    그 자리를 판정하는 것은 `common.corporate_actions.is_series_restart` 이고, 결과가
#:    `is_first_listing` 칸으로 들어온다. **선택 인자로 두지 않고 필수로 둔 이유**는
#:    빠뜨려도 예외가 안 나고 값만 조용히 틀리기 때문이다 — 파생 칸을 늘릴 때마다 따라와야
#:    하는 자리를 09-09 에 한 곳 빠뜨려 오준영 님 로더가 멈춘 일이 있었다.
REQUIRED_COLUMNS = ("bas_dd", "code", "close", "adj_close", "change_rate",
                    "is_first_listing")

#: 돌려주는 열. 순서가 곧 계약이다.
FLAG_COLUMNS = ("adj_return_1d", "adj_change_rate_gap",
                "is_adj_suspect", "is_extreme_return")


def _as_yyyymmdd(values: pd.Series) -> pd.Series:
    """날짜 열을 `YYYYMMDD` 문자열로 맞춘다. 문자열·정수·datetime 을 받는다."""
    if pd.api.types.is_datetime64_any_dtype(values):
        out = values.dt.strftime("%Y%m%d").astype("string")
    else:
        out = values.astype("string").str.strip().str.replace("-", "", regex=False)
    bad = out.isna() | ~out.str.fullmatch(r"\d{8}").fillna(False)
    if bool(bad.any()):
        raise ValueError(
            f"bas_dd 가 YYYYMMDD 가 아닙니다: {out[bad].head(3).tolist()}"
        )
    return out.astype(str)


def _as_code(values: pd.Series) -> pd.Series:
    """종목코드 열을 여섯 자리 문자열로 맞춘다. 5·6번째 자리가 영문인 코드도 그대로 둔다."""
    return values.astype("string").str.strip().str.zfill(6).astype(str)


def flag_adjustment_quality(
    daily_prices: pd.DataFrame,
    *,
    gap_tolerance: float = SUSPECT_GAP_TOLERANCE,
    extreme_pct: float = EXTREME_RETURN_PCT,
) -> pd.DataFrame:
    """행마다 칸 넷을 매긴다. **입력과 같은 인덱스·같은 순서**로 돌려준다.

    수익률은 종목의 전체 시계열에서 이웃한 두 날로 계산한다. 후보만 잘라 낸 뒤 계산하면
    후보에 안 뽑힌 전날이 빠져 며칠짜리 수익률이 되므로, 항상 원천 전체를 넘긴다.

    `attrs["adjustment_quality"]` 에 비교 가능 행 수 · 의심 행 수 · 극단 행 수를 남긴다.
    """
    if gap_tolerance <= 0.0:
        raise ValueError("gap_tolerance 는 0 보다 커야 합니다.")
    if extreme_pct <= 0.0:
        raise ValueError("extreme_pct 는 0 보다 커야 합니다.")
    missing = set(REQUIRED_COLUMNS) - set(daily_prices.columns)
    if missing:
        raise ValueError(
            f"수정주가 품질 입력 열이 없습니다: {sorted(missing)}. "
            "HF 반출본 `full/daily_price_dev.parquet`(34칸) 을 그대로 넘기면 전부 있습니다. "
            "DB 에서 직접 읽는 경로라면 `supply.training.attach_corporate_action_flags` 로 "
            "기업행위 3칸을 먼저 붙이십시오 — `is_first_listing` 이 없으면 코드를 재사용한 "
            "종목의 '전일' 이 다른 회사의 종가가 되어 값이 조용히 틀립니다."
        )

    work = pd.DataFrame(
        {
            "bas_dd": _as_yyyymmdd(daily_prices["bas_dd"]),
            "code": _as_code(daily_prices["code"]),
            "close": pd.to_numeric(daily_prices["close"], errors="coerce"),
            "adj_close": pd.to_numeric(daily_prices["adj_close"], errors="coerce"),
            "change_rate": pd.to_numeric(daily_prices["change_rate"], errors="coerce"),
            "is_first_listing": daily_prices["is_first_listing"].fillna(False).astype(bool),
        },
        index=daily_prices.index,
    )
    if bool(work.duplicated(["bas_dd", "code"]).any()):
        raise ValueError("수정주가 품질 원천에 같은 날짜·종목이 두 번 이상 있습니다.")

    # 원래 순서로 되돌리기 위한 자리표. 정렬은 종목별 전날을 찾기 위한 것뿐이다.
    work["_pos"] = np.arange(len(work))
    work = work.sort_values(["code", "bas_dd"], kind="stable")

    prev_adj = work.groupby("code", sort=False)["adj_close"].shift(1)
    comparable = (
        work["adj_close"].notna()
        & work["close"].gt(0)
        & prev_adj.notna()
        & prev_adj.gt(0)
        & work["change_rate"].notna()
        # 🔴 새 시계열의 첫 행은 **전일이 없다.** 종목의 진짜 첫 행이면 `prev_adj` 가
        #    이미 NaN 이라 이 조건이 아무것도 안 바꾸지만, 코드를 재사용한 자리에서는
        #    `shift(1)` 이 다른 회사의 종가를 가리키므로 여기서 끊어야 한다.
        & ~work["is_first_listing"]
    )
    adj_return = pd.Series(np.nan, index=work.index, dtype="float64")
    adj_return[comparable] = (
        work.loc[comparable, "adj_close"] / prev_adj[comparable] - 1.0
    ) * 100.0
    gap = (adj_return - work["change_rate"]).abs()          # 비교 불가면 NaN 그대로
    suspect = comparable & gap.gt(gap_tolerance)
    extreme = comparable & adj_return.abs().gt(extreme_pct) & ~suspect

    work["adj_return_1d"] = adj_return
    work["adj_change_rate_gap"] = gap.where(comparable)
    work["is_adj_suspect"] = suspect.astype(bool)
    work["is_extreme_return"] = extreme.astype(bool)
    work = work.sort_values("_pos", kind="stable")

    out = work.loc[:, list(FLAG_COLUMNS)].copy()
    out.index = daily_prices.index
    out.attrs["adjustment_quality"] = {
        "gap_tolerance": float(gap_tolerance),
        "extreme_pct": float(extreme_pct),
        "rows": int(len(out)),
        "comparable_rows": int(comparable.sum()),
        "suspect_rows": int(suspect.sum()),
        "extreme_rows": int(extreme.sum()),
    }
    return out


def attach_adjustment_quality(
    candidates: pd.DataFrame,
    daily_prices: pd.DataFrame,
    *,
    gap_tolerance: float = SUSPECT_GAP_TOLERANCE,
    extreme_pct: float = EXTREME_RETURN_PCT,
) -> pd.DataFrame:
    """후보 프레임에 칸 넷을 붙인다. `(bas_dd, code)` 로 맞추고 후보의 열·순서·인덱스는 그대로다.

    원천에 없는 후보 키는 수익률·갭이 NaN 이고 플래그는 False 다.
    `attrs["adjustment_quality"]` 에 후보 안의 의심·극단 행 수와 원천에 없던 행 수를 남긴다.
    """
    key_missing = {"bas_dd", "code"} - set(candidates.columns)
    if key_missing:
        raise ValueError(f"후보 키가 없습니다: {sorted(key_missing)}")
    clash = set(FLAG_COLUMNS) & set(candidates.columns)
    if clash:
        raise ValueError(f"후보에 같은 이름의 열이 이미 있습니다: {sorted(clash)}")

    flags = flag_adjustment_quality(
        daily_prices, gap_tolerance=gap_tolerance, extreme_pct=extreme_pct
    )
    source_summary: Dict[str, Any] = dict(flags.attrs["adjustment_quality"])
    flags = flags.assign(
        _bas=_as_yyyymmdd(daily_prices["bas_dd"]).to_numpy(),
        _code=_as_code(daily_prices["code"]).to_numpy(),
    )

    keyed = candidates.assign(
        _bas=_as_yyyymmdd(candidates["bas_dd"]).to_numpy(),
        _code=_as_code(candidates["code"]).to_numpy(),
    )
    if bool(keyed.duplicated(["_bas", "_code"]).any()):
        raise ValueError("후보에 같은 날짜·종목이 두 번 이상 있습니다.")

    merged = keyed.merge(
        flags[["_bas", "_code", *FLAG_COLUMNS]],
        on=["_bas", "_code"],
        how="left",
        validate="one_to_one",
        indicator="_hit",
    )
    unmatched = int(merged["_hit"].ne("both").sum())
    merged = merged.drop(columns=["_bas", "_code", "_hit"])
    for column in ("is_adj_suspect", "is_extreme_return"):
        merged[column] = merged[column].fillna(False).astype(bool)
    merged.index = candidates.index

    merged.attrs.update(candidates.attrs)
    merged.attrs["adjustment_quality"] = {
        **source_summary,
        "candidate_rows": int(len(merged)),
        "candidate_suspect_rows": int(merged["is_adj_suspect"].sum()),
        "candidate_extreme_rows": int(merged["is_extreme_return"].sum()),
        "candidate_unmatched_rows": unmatched,
    }
    return merged


__all__ = [
    "EXTREME_RETURN_PCT",
    "FLAG_COLUMNS",
    "REQUIRED_COLUMNS",
    "SUSPECT_GAP_TOLERANCE",
    "attach_adjustment_quality",
    "flag_adjustment_quality",
]
