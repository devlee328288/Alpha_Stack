"""상장 구간 — 한 종목코드에 회사가 둘 이상 들어 있을 때, 행과 기록을 **같은 회사끼리만** 잇는다.

상장폐지된 종목의 코드를 몇 년 뒤 다른 회사가 다시 받는다. 우리 자료에는 두 곳이 있다
(실측 2026-09-11 · `daily_price` 9,231,938행 · 같은 코드 안의 거래일 공백은 이 둘뿐이다).

    036220  인포피아     ~20160504  →  오상헬스케어  20240313~
    101970  우양에이치씨 ~20150316  →  우양에이치씨  20250328~

주권종류(`supply.universe.attach_security_type`)와 업종(`supply.sector.attach_industry`)은
*"그 행의 날짜까지 알게 된 가장 최근 기록"* 을 붙인다. 그 규칙은 기록이 **얼마나 오래됐는지**
묻지 않아서, 뒤 회사의 행에 앞 회사의 기록이 붙었다.

    주권종류  036220 1행 · 101970 1행      상장 첫날 — 다른 신규상장처럼 빈 칸이어야 한다
    업종      036220 74행 · 101970 187행   새 회사가 처음 실린 스냅샷이 나오기 전까지

🔴 붙은 옛 값이 새 회사의 나중 값과 **우연히 같았다**(제약·금속·보통주). 반출본과 판정기를
   값으로 대조해도 안 보였고, 판정기는 같은 함수를 불러 같은 잘못을 공유했다.

## 구간번호

행이든 기록이든 **그 날짜 이하의 구간 시작일 수**가 구간번호다. 구간 시작일 당일은 새 구간이다.
붙이는 쪽이 `merge_asof(by=["code", 구간번호])` 로 걸면 다른 구간의 기록은 후보에 오르지 않는다.

기록은 `known_at` 이 아니라 **자기 날짜**(`bas_dd`)로 구간을 센다. 상장일보다 앞선 기본정보는
전 종목 0행이라(실측 2026-09-11) 기록의 날짜가 곧 그 기록이 말하는 회사다.

## 미래를 보지 않는다

구간 시작일은 `common.corporate_actions.is_series_restart` — 거래일 공백과 그 뒤의 새 상장일 —
로 정하므로 반출본의 `is_first_listing` 과 같은 날이다. 행 T 의 구간번호는 T 이하의 시작일만
센다. 그리고 이 판정은 **빼기만** 한다 — 옛 기록을 버려 빈 칸을 만들 뿐 새 기록을 앞당겨
붙이지 않는다.
"""

from __future__ import annotations

import sqlite3
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from common.corporate_actions import series_restart_days
from ingest.store import krx_store


def restart_days_from_db(db_path=None) -> Dict[str, Tuple[str, ...]]:
    """DB 에서 `종목 → 구간 시작일들` 표를 만든다. `db_path` 가 없으면 기본 DB 다.

    ⏱ 실측 2026-09-11 약 9초 — 거의 전부 상장일 표(`listing_days_by_code`)다. 반출처럼
       `MarketContext` 를 이미 가진 쪽은 이 함수 대신 `series_restart_days` 에 그 표를 넘긴다.
    """
    if db_path:
        conn = sqlite3.connect(db_path)
        try:
            return series_restart_days(conn)
        finally:
            conn.close()
    with krx_store.connect() as conn:
        return series_restart_days(conn)


def segment_numbers(codes: pd.Series, days: pd.Series,
                    restart_days: Mapping[str, Sequence[str]]) -> np.ndarray:
    """행마다 **그 날짜 이하의 구간 시작일 수**. 코드 재사용이 없는 종목은 전부 0 이다.

    `days` 는 `YYYYMMDD` 문자열이어야 한다(줄표 없이) — 같은 자릿수 문자열의 크기 비교로 센다.
    재사용 코드는 두 종목뿐이라 그 종목의 행만 골라 센다. 788만 행 표에서도 비교 두 번이다.
    """
    out = np.zeros(len(codes), dtype=np.int64)
    if not restart_days or len(codes) == 0:
        return out
    code_arr = pd.Series(codes).astype(str).to_numpy()
    day_arr = pd.Series(days).astype(str).to_numpy()
    for code, starts in restart_days.items():
        mask = code_arr == str(code)
        if mask.any():
            시작 = np.asarray(sorted(str(d) for d in starts), dtype=str)
            out[mask] = np.searchsorted(시작, day_arr[mask].astype(str), side="right")
    return out


__all__ = ["restart_days_from_db", "segment_numbers"]
