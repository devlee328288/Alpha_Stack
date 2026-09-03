"""유니버스 — 그날 무엇을 후보로 삼을 수 있었나.

모델 파트가 쓰기로 한 후보는 **"날짜별 KOSPI 보통주 시가총액 상위 50종목"** 이다
(#92 오준영님 4번). 그 '보통주' 를 지금까지 **종목명이 '우' 로 끝나는지로 추측**하고
있었고, 그게 틀린다.

실측 2026-09-03 · KRX 종목기본정보를 세 시장 × 세 날짜로 전수 대조:

    미래에셋대우 · 연우 · 동우 · 신우 · 성우 · 에코글로우 · 이오플로우
    → 이름이 '우' 로 끝나는 **보통주 7종**을 우선주로 잘못 뺐다

006800 은 20200102 코스피 시총 **48위**다. 상위 50 후보에서 조용히 빠진다.

🔴 이 오류는 **이름이 바뀌는 구간에만** 나타난다 — 대우증권(정상) → 미래에셋대우(깨짐)
   → 미래에셋증권(정상). 오늘 유가 943종만 세면 어긋남이 0건이다. 표본으로는 못 잡는다.

## 이 문을 지나는 이유

`stock_base_info` 를 직접 읽으면 **오늘 알게 된 주권종류로 2015년을 판정**하게 된다.
여기서 `as_of` 를 받아 그때 알 수 있었던 행만 보게 막는다. `as_of` 는 키워드 전용이고
**기본값이 없어서 빠뜨릴 수가 없다.**

## 무엇과 교집합을 내는가

`daily_price` 다. 기본정보는 *"상장돼 있다"* 를 말할 뿐 **그날 거래됐는지**는 말하지
않고, 시가총액도 시세에만 있다. 그래서 시세를 축으로 두고 기본정보를 붙인다.
"""

from __future__ import annotations

from typing import List

import pandas as pd

from ingest.store import base_info_store
from supply.clock import AsOf, as_bas_dd, latest_known_day, to_kst

#: 후보 표가 내는 칸. 부르는 쪽이 이 이름에 기대므로 함부로 바꾸지 않는다.
UNIVERSE_COLUMNS = (
    "bas_dd", "code", "name", "market", "close", "adj_close",
    "market_cap", "listed_shares", "kind_stkcert_tp_nm", "list_dd",
    "isin_cd", "isu_abbrv", "info_bas_dd", "info_known_at",
)


def _known_by(as_of: AsOf) -> str:
    """`as_of` 시점에 알 수 있었던 `known_at` 의 상한(`YYYYMMDD`).

    `stock_base_info.known_at` 은 `basDd` 의 **다음 거래일**(YYYYMMDD)로 적혀 있다.
    그래서 문자열 비교가 성립하도록 같은 표기로 맞춰 넘긴다.
    """
    return latest_known_day(as_of)


def common_stocks(bas_dd: str, *, as_of: AsOf, market: str = "KOSPI") -> pd.DataFrame:
    """그 거래일의 **보통주만**. 시가총액 큰 순서.

    우선주·종류주는 뺀다. 무엇이 보통주인지는 KRX 종목기본정보의
    `KIND_STKCERT_TP_NM` 이 정한다 — 종목명으로 추측하지 않는다.

    🔴 **모르는 주권종류는 보통주가 아니라고 본다.** 빈 값을 보통주로 치면 우선주가
       후보에 섞이는데, 빠진 종목은 개수로 드러나지만 섞인 종목은 성능이 조금
       이상해질 뿐 아무 데도 안 걸린다.
    """
    바스 = as_bas_dd(bas_dd)
    if 바스 is None:
        raise ValueError(f"bas_dd 를 읽을 수 없다: {bas_dd!r}")

    # 🔴 `as_of` 보다 뒤의 거래일을 물어보면 그 자체가 미래참조다. 조용히 빈 표를
    #    주지 않고 세운다 — 빈 표는 "그날 상장 종목이 없었다" 로 오해된다.
    상한 = latest_known_day(as_of)
    if 바스 > 상한:
        raise ValueError(
            f"{바스} 는 as_of({to_kst(as_of).date()}) 시점에 아직 오지 않은 거래일이다.\n"
            f"  그때 알 수 있었던 가장 최근 거래일: {상한}\n"
            "  할 일: bas_dd 를 그 이하로 주거나, as_of 를 뒤로 옮긴다."
        )

    rows = base_info_store.universe_rows(
        바스, market=market, common_only=True, known_by=_known_by(as_of))
    return _to_frame(rows)


def top_by_market_cap(bas_dd: str, *, as_of: AsOf, top: int = 50,
                      market: str = "KOSPI") -> pd.DataFrame:
    """그 거래일의 보통주 **시가총액 상위 `top` 종목**. 모델 파트의 후보 표다.

    ⚠️ 시가총액은 `daily_price` 의 그날 값이다. 종가가 나온 뒤라야 알 수 있으므로
       `as_of` 가 그 거래일 **다음**이어야 한다 — `common_stocks` 가 막는다.
    """
    frame = common_stocks(bas_dd, as_of=as_of, market=market)
    return frame.head(top).reset_index(drop=True)


def excluded(bas_dd: str, *, as_of: AsOf, market: str = "KOSPI") -> pd.DataFrame:
    """유니버스에서 **빠진** 종목과 그 사유. 왜 빠졌는지 눈으로 보려고 낸다.

    검증기가 "몇 종 빠졌다" 만 말하면 사람은 확인하지 않는다. 무엇이 왜 빠졌는지를
    같이 내야 이상한 제외를 알아챈다.
    """
    바스 = as_bas_dd(bas_dd)
    rows = base_info_store.universe_rows(
        바스, market=market, common_only=False, known_by=_known_by(as_of))
    뺀것 = [r for r in rows
            if (r.get("kind_stkcert_tp_nm") or "").strip() != "보통주"]
    frame = _to_frame(뺀것)
    if not frame.empty:
        # ⚠️ `pandas` 는 못 이은 칸을 `None` 이 아니라 `NaN` 으로 만들고,
        #    **`NaN` 은 참이다**(`bool(float('nan')) is True`). `if k else` 로 쓰면
        #    사유가 안 붙고 NaN 이 그대로 남는다. `pd.isna` 로 물어야 한다.
        frame["제외사유"] = frame["kind_stkcert_tp_nm"].apply(
            lambda k: "기본정보를 못 이었다 (주권종류 모름)"
            if pd.isna(k) or not str(k).strip() else str(k))
    return frame


def _to_frame(rows: List[dict]) -> pd.DataFrame:
    """딕셔너리 목록을 표로. **빈 결과에도 칸을 남긴다.**

    빈 표에 칸이 없으면 부르는 쪽의 `frame["code"]` 가 KeyError 로 터진다. 빈 날
    (휴장·수집 전)에 파이프라인 전체가 멈추는 것을 여기서 막는다.
    """
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in UNIVERSE_COLUMNS})
    frame = pd.DataFrame(rows)
    앞 = [c for c in UNIVERSE_COLUMNS if c in frame.columns]
    뒤 = [c for c in frame.columns if c not in 앞]
    return frame[앞 + 뒤].reset_index(drop=True)


def coverage(bas_dd: str, *, as_of: AsOf, market: str = "KOSPI") -> dict:
    """그날 기본정보를 못 이은 종목이 얼마나 되나. 반출 전에 확인하는 값이다.

    못 이은 종목은 주권종류를 몰라 **전부 유니버스에서 빠진다.** 그게 조용히 커지면
    후보가 줄어드는데 에러는 안 난다. 그래서 숫자로 남긴다.
    """
    바스 = as_bas_dd(bas_dd)
    rows = base_info_store.universe_rows(
        바스, market=market, common_only=False, known_by=_known_by(as_of))
    못이은 = [r for r in rows if not (r.get("kind_stkcert_tp_nm") or "").strip()]
    보통주 = [r for r in rows
              if (r.get("kind_stkcert_tp_nm") or "").strip() == "보통주"]
    return {
        "bas_dd": 바스, "market": market,
        "시세종목": len(rows), "보통주": len(보통주), "못이은": len(못이은),
        "못이은비율": round(len(못이은) / len(rows), 4) if rows else None,
        "못이은코드": [r["code"] for r in 못이은[:20]],
    }


__all__ = ["UNIVERSE_COLUMNS", "common_stocks", "top_by_market_cap",
           "excluded", "coverage"]
