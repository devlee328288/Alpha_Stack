"""거시 — 그 행의 날짜에 **발표돼 있던** 지표 값.

## 무엇이 들어 있나 (실측 2026-09-11 · `main` `3fb42ee`)

    macro_series   17,864행 · 9종 · 결측 0 · known_at 이 기간 순서와 역전된 곳 0
      일별 4종   ktb3y · ktb10y · usdkrw · jpykrw          기간 → known_at  1일
      월별 5종   base_rate 31일 · cpi 40일 · ppi 60일 · leading 65일 · coincident 65일
                 (기간 시작일에서 known_at 까지의 중앙값. 규칙은 `ecos_data.RELEASE_RULES`)

## 🔴 붙이는 축은 기간이 아니라 `known_at` 이다

ECOS 는 월별 지표를 **기준월 1일**로 준다. 7월 물가를 7월 1일에 붙이면 8월 초에야 나온
값을 한 달 먼저 아는 셈이고, 경기지수는 두 달이다. ECOS 는 발표일을 주지 않으므로
규칙으로 계산해 담아 둔 `known_at` 이 유일한 시점이다. 행 T 에 보이는 조건은
`known_at <= T` 하나다.

    ⚠️ known_at 은 **계산값**이다. 규칙을 바꾸면 `macro_series` 를 다시 채워야 한다.

## 🔴 경기 순환변동치(`leading`·`coincident`)는 기본에서 뺀다

순환변동치는 추세를 새로 추정할 때마다 **과거 값이 고쳐지는** 계열이다. 우리는 받을 때마다
덮어써서(`INSERT OR REPLACE`) 2026-09 판 하나만 갖고 있고 `raw_response` 에도 ECOS 원문이
없어서, 옛 판이 무엇이었는지도 개정이 얼마였는지도 **잴 수 없다.** 이 값을 2015년 행에
붙이면 2026년에 다시 추정한 순환변동치를 2015년에 아는 셈이다.

그래서 `attach_macro`·`macro_as_of` 의 기본 지표에서 뺐다. 쓰려면
`indicators=("leading",)` 처럼 **이름을 적어야** 하고, 그 이름이 부르는 코드에 남아 눈에
띈다. (2026-09-11 합의)

⚠️ CPI·PPI 는 기준연도 개편(2020=100)으로 옛 **수준값**이 당시 발표치와 다르다. 전년 대비
   비율은 거의 보존되므로 수준보다 **비율**로 쓰는 편이 안전하다.

## 모양

    macro_as_of("20240111", as_of="2024-01-12")      지표당 한 행
    attach_macro(price_frame, as_of="2024-08-31")    지표마다 두 칸 — macro_<id> · macro_<id>_period
    macro_history("cpi", as_of="2024-08-31")         그 시점에 보이던 이력 — 전년 대비·차분용

`_period` 칸을 함께 싣는 이유: 월별 지표는 한 달 내내 같은 값이 이어진다. 값만 보면 그 값이
어느 기간의 것인지, 발표가 끊겨 오래된 값이 이어지는 중인지 알 수 없다.

전년 대비는 `macro_history` 에서 **기간끼리** 계산한다. 그 결과를 언제 알 수 있었는지는 두
기간 중 **늦은 쪽의 `known_at`** 이다.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from typing import FrozenSet, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from common.paths import krx_db_path
from ingest.clients import ecos_data
from supply.clock import AsOf, as_bas_dd, latest_known_day, row_day, to_kst

#: 수집 큐레이션의 지표 전부. 순서도 수집 쪽(`ecos_data.INDICATORS`)을 따른다.
ALL_INDICATORS: Tuple[str, ...] = tuple(spec["id"] for spec in ecos_data.INDICATORS)

#: 과거 값이 개정되는 계열 — 기본 지표에서 빠진다.
REVISED_INDICATORS: FrozenSet[str] = frozenset({"leading", "coincident"})

#: 이름을 적지 않았을 때 붙는 지표.
DEFAULT_INDICATORS: Tuple[str, ...] = tuple(
    i for i in ALL_INDICATORS if i not in REVISED_INDICATORS)

#: 점 조회·이력이 내는 칸. **행이 0개여도 이 칸들은 있다.**
MACRO_COLUMNS: Tuple[str, ...] = ("indicator_id", "period", "cycle", "value", "known_at", "unit")


def macro_columns_for(indicators: Sequence[str]) -> Tuple[str, ...]:
    """`attach_macro` 가 붙이는 칸 이름 — 지표마다 값·기간 두 칸."""
    out: List[str] = []
    for i in indicators:
        out += [f"macro_{i}", f"macro_{i}_period"]
    return tuple(out)


# ==================================================
# 1. 작은 도구
# ==================================================
def _connect(db_path=None) -> sqlite3.Connection:
    """읽기만 한다. 경로를 인자로 받아 갈아 끼울 자리를 하나로 둔다."""
    conn = sqlite3.connect(db_path or krx_db_path(), timeout=60)
    conn.execute("PRAGMA busy_timeout=60000")
    return conn


def _empty(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})


def _indicator_ids(indicators: Optional[Iterable[str]], *,
                   default: Tuple[str, ...]) -> Tuple[str, ...]:
    """지표 이름을 정리한다. 모르는 이름은 세운다 — 조용히 빼면 칸이 사라진 줄 모른다."""
    if indicators is None:
        return default
    if isinstance(indicators, str):
        indicators = (indicators,)
    ids = tuple(dict.fromkeys(str(i).strip().lower() for i in indicators if str(i).strip()))
    모름 = [i for i in ids if i not in ALL_INDICATORS]
    if 모름:
        raise ValueError(
            f"없는 지표다: {', '.join(모름)}\n"
            f"  쓸 수 있는 것: {' · '.join(ALL_INDICATORS)}\n"
            f"  ⚠️ {' · '.join(sorted(REVISED_INDICATORS))} 는 과거 값이 개정되는 계열이라 "
            "기본에서 빠져 있고, 이름을 적어야 나온다."
        )
    return ids


def _read(conn: sqlite3.Connection, ids: Sequence[str], known_upto: str) -> pd.DataFrame:
    """`known_at <= known_upto` 이고 값이 있는 행. 기간 오름차순.

    값이 빈 기간(ECOS 가 '-' 를 준 곳)은 건너뛴다 — 저장 계층의 `macro_store.as_of` 와 같은
    규칙이다. 건너뛰면 앞 기간 값이 이어지는데, 그 사실은 `_period` 칸이 드러낸다.
    """
    if not ids:
        return _empty(MACRO_COLUMNS)
    sql = (
        "SELECT indicator_id, period, cycle, value, known_at, unit FROM macro_series "
        f"WHERE indicator_id IN ({','.join('?' * len(ids))}) "
        "AND known_at <= ? AND value IS NOT NULL "
        "ORDER BY indicator_id, period"
    )
    return pd.read_sql_query(sql, conn, params=[*ids, known_upto])


def _timeline(sub: pd.DataFrame) -> pd.DataFrame:
    """한 지표에서 *"그날까지 발표된 것 중 기간이 가장 늦은 값"* 이 정해지는 시점들.

    칸: `_key`(known_at 정수) · period · value. `merge_asof` 로 행에 붙일 오른쪽 표다.

    known_at 이 기간 순서와 역전된 곳은 실측 0건이라 *"가장 늦게 발표된 값"* 과 답이 같다.
    그래도 규칙은 **기간**으로 둔다 — 재무에서 정정본이 옛 해를 늦게 드러냈듯, 역전이 한 번
    생기면 가장 최근에 발표된 옛 기간이 새 기간을 밀어내는데 예외는 나지 않는다.
    """
    s = sub.sort_values(["known_at", "period"]).reset_index(drop=True)
    # 기간을 정렬 순위로 바꿔 누적 최대를 잡는다. 알게 된 순서대로 훑으며 가장 늦은 기간을 든다.
    codes, uniques = pd.factorize(s["period"], sort=True)
    s["_best"] = uniques[pd.Series(codes).cummax().to_numpy()]
    last = s.groupby("known_at", sort=True).tail(1)
    value_by_period = s.drop_duplicates("period", keep="last").set_index("period")["value"]
    return pd.DataFrame({
        "_key": last["known_at"].astype(int).to_numpy(),
        "period": last["_best"].to_numpy(),
        "value": value_by_period.reindex(last["_best"]).to_numpy(),
    })


# ==================================================
# 2. 정문 — as_of 없이는 못 지난다
# ==================================================
def macro_as_of(bas_dd: AsOf, *, as_of: AsOf, indicators: Optional[Iterable[str]] = None,
                db_path=None) -> pd.DataFrame:
    """거래일 `bas_dd` 의 행에 보이는 거시 — 지표당 한 행.

    칸: `MACRO_COLUMNS`. 그날까지 한 번도 발표되지 않은 지표는 행이 없다.
    순서는 `indicators`(주지 않으면 `DEFAULT_INDICATORS`) 순서다.
    """
    바스 = row_day(bas_dd, as_of=as_of)
    ids = _indicator_ids(indicators, default=DEFAULT_INDICATORS)
    with closing(_connect(db_path)) as conn:
        rows = _read(conn, ids, 바스)
    if rows.empty:
        return _empty(MACRO_COLUMNS)

    pick = rows.sort_values(["indicator_id", "period"]).groupby("indicator_id", sort=False).tail(1)
    순서 = {i: n for n, i in enumerate(ids)}
    pick = pick.assign(_o=pick["indicator_id"].map(순서)).sort_values("_o")
    return pick[list(MACRO_COLUMNS)].reset_index(drop=True)


def attach_macro(frame: pd.DataFrame, *, as_of: AsOf,
                 indicators: Optional[Iterable[str]] = None, db_path=None) -> pd.DataFrame:
    """시세 표(`bas_dd` 가 있는 것)에 지표마다 `macro_<id>` · `macro_<id>_period` 두 칸을 붙인다.

    거시는 시장 전체의 값이라 `code` 가 필요 없다. 행마다 **그 행의 날짜까지 발표된 것 중
    기간이 가장 늦은 값**을 붙인다. 아직 발표된 것이 없으면 빈 칸이다.

    🔴 `as_of` 보다 뒤의 거래일이 든 표를 주면 세운다.
    ⚠️ 입력 순서를 보존한다.
    """
    if "bas_dd" not in frame.columns:
        raise ValueError("거시를 붙이려면 'bas_dd' 칸이 있어야 한다 — 시세 표를 넘겨라.")

    ids = _indicator_ids(indicators, default=DEFAULT_INDICATORS)
    out = frame.copy()
    if out.empty:
        for col in macro_columns_for(ids):
            out[col] = pd.Series(dtype="object")
        return out

    days = out["bas_dd"].map(as_bas_dd)
    상한 = latest_known_day(as_of)
    if days.max() > 상한:
        raise ValueError(
            f"표에 as_of({to_kst(as_of).date()}) 시점에 아직 오지 않은 거래일이 있다 "
            f"(최대 {days.max()} > {상한}).\n"
            "  할 일: supply.price_series(as_of=...) 로 받은 표를 넘기거나 as_of 를 뒤로 옮긴다."
        )

    with closing(_connect(db_path)) as conn:
        rows = _read(conn, ids, 상한)

    left = pd.DataFrame({"_key": days.astype(int).to_numpy(),
                         "_order": range(len(out))}).sort_values("_key")
    for i in ids:
        sub = rows[rows["indicator_id"] == i]
        if sub.empty:
            out[f"macro_{i}"] = float("nan")
            out[f"macro_{i}_period"] = pd.Series([None] * len(out), index=out.index,
                                                 dtype="object")
            continue
        merged = pd.merge_asof(left, _timeline(sub), on="_key", direction="backward",
                               allow_exact_matches=True).sort_values("_order")
        out[f"macro_{i}"] = merged["value"].to_numpy()
        out[f"macro_{i}_period"] = merged["period"].to_numpy()
    return out


def macro_history(indicator: str, *, as_of: AsOf, db_path=None) -> pd.DataFrame:
    """`as_of` 시점에 **발표돼 있던** 한 지표의 이력 — 기간 오름차순.

    전년 대비·차분을 만들 때 쓴다. 지표 이름을 반드시 적어야 하므로 순환변동치도 막지 않는다
    (적는 순간 부르는 코드에 이름이 남는다). 대신 그 계열의 옛 값은 **2026-09 판**이라는
    것을 잊지 않는다 — 모듈 머리말 참고.
    """
    ids = _indicator_ids((indicator,), default=())
    with closing(_connect(db_path)) as conn:
        rows = _read(conn, ids, latest_known_day(as_of))
    if rows.empty:
        return _empty(MACRO_COLUMNS)
    return rows[list(MACRO_COLUMNS)].sort_values("period").reset_index(drop=True)


__all__ = [
    "ALL_INDICATORS",
    "DEFAULT_INDICATORS",
    "MACRO_COLUMNS",
    "REVISED_INDICATORS",
    "attach_macro",
    "macro_as_of",
    "macro_columns_for",
    "macro_history",
]
