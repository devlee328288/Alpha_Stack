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

import sqlite3
from typing import List

import pandas as pd

from ingest.store import base_info_store, krx_store
from supply.clock import AsOf, as_bas_dd, latest_known_day, to_kst

#: 후보 표가 내는 칸. 부르는 쪽이 이 이름에 기대므로 함부로 바꾸지 않는다.
UNIVERSE_COLUMNS = (
    "bas_dd", "code", "name", "market", "close", "adj_close",
    "market_cap", "listed_shares", "kind_stkcert_tp_nm", "list_dd",
    "isin_cd", "isu_abbrv", "info_bas_dd", "info_known_at",
)

#: 반출본에 싣는 주권종류 세 칸. 이 순서로 카드 표에 나간다.
SECURITY_TYPE_COLUMNS = ("kind_stkcert_tp_nm", "secugrp_nm", "sect_tp_nm")


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


def attach_security_type(frame: pd.DataFrame, *, as_of: AsOf,
                         db_path=None) -> pd.DataFrame:
    """시세 표(`bas_dd`·`code` 가 있는 것)에 **그 날의** 주권종류 세 칸을 붙인다.

    `common_stocks` 가 하루씩 답하는 것을 표 전체에 한 번에 하는 함수다. 반출본은
    788만 행이라 날짜마다 부르면 4,105번 조회가 된다.

    ## 왜 이름 규칙을 대신할 수 있나

    지금까지 개발본에는 주권종류가 없어서, 받아 쓰는 쪽이 **종목명이 '우' 로 끝나는지**로
    보통주를 추측했다. 그 규칙은 연우·동우·신우처럼 이름이 '우' 인 보통주를 우선주로
    잘못 뺀다. 감사 예외 목록으로 기워 왔지만 목록은 대조한 날짜까지만 유효하다.

    실측 2026-09-09 · 개발구간 7,888,945행 전량:

        stock_base_info 조인 커버리지        100.0000%   (미매칭 0행)
        이름 규칙 + 예외 10건 vs 주권종류      불일치 0행

    **판정이 한 행도 바뀌지 않으므로** 이름 규칙과 예외 목록을 지워도 유니버스가 같다.

    ## 세 칸을 다 싣는 이유

    `kind_stkcert_tp_nm` 만으로는 우선주밖에 못 거른다. KOSPI200 지수 방법론은 리츠·
    선박투자회사·기업인수목적회사·관리종목도 후보에서 빼고, CRSP 는 REIT·closed-end
    fund·외국주권·ADR 을 뺀다. 그 판정이 나머지 두 칸에 들어 있다.

        secugrp_nm   주권 · 외국주권 · 부동산투자회사(리츠) · 선박투자회사 ·
                     투자회사 · 사회간접자본투융자회사 · 주식예탁증권 · 주식예탁증서
        sect_tp_nm   소속부 — 우량기업부·중견기업부·벤처기업부·기술성장기업부 ·
                     SPAC(소속부없음) · 관리종목(소속부없음) · 투자주의환기종목 · 외국기업

    ⚠️ **무엇을 뺄지는 여기서 정하지 않는다.** 값을 그대로 실어 보내고, 거르는 규칙은
       쓰는 쪽이 고른다. 유동주식비율이 없어 KOSPI200 방법론을 완전히 재현할 수 없는데
       파생 판정 한 칸으로 뭉치면 "이대로 쓰면 KOSPI200 과 같다" 는 오해를 부른다.
    """
    for col in ("bas_dd", "code"):
        if col not in frame.columns:
            raise ValueError(f"주권종류를 붙이려면 '{col}' 칸이 있어야 한다 — 시세 표를 넘겨라.")

    out = frame.copy()
    if out.empty:
        for col in SECURITY_TYPE_COLUMNS:
            out[col] = pd.Series([], dtype="object")
        return out

    # 🔴 `as_of` 시점에 알 수 있었던 행만 본다. 오늘 알게 된 주권종류로 2015년을
    #    판정하면 그게 미래참조다. `known_at` 은 basDd 의 다음 거래일로 적혀 있다.
    상한 = _known_by(as_of)
    처음, 끝 = str(out["bas_dd"].min()), str(out["bas_dd"].max())

    conn = sqlite3.connect(db_path) if db_path else None
    try:
        if conn is None:
            with krx_store.connect() as c:
                right = _read_security_type(c, 처음, 끝, 상한)
        else:
            right = _read_security_type(conn, 처음, 끝, 상한)
    finally:
        if conn is not None:
            conn.close()

    left = out[["bas_dd", "code"]].copy()
    left["bas_dd"] = left["bas_dd"].astype(str)
    left["code"] = left["code"].astype(str)
    left["_order"] = range(len(left))
    merged = left.merge(right, on=["bas_dd", "code"], how="left")
    # 🔴 기본키가 (bas_dd, code) 라 1:1 이어야 한다. 늘어났다면 저장소가 중복을 물고
    #    있다는 뜻이고, 그대로 두면 반출본 행 수가 조용히 불어난다.
    if len(merged) != len(left):
        raise ValueError(
            f"주권종류 조인이 행을 늘렸다: {len(left):,} → {len(merged):,}. "
            "stock_base_info 에 (bas_dd, code) 중복이 있다."
        )
    merged = merged.sort_values("_order")
    for col in SECURITY_TYPE_COLUMNS:
        out[col] = merged[col].to_numpy()
    return out


def _read_security_type(conn, 처음: str, 끝: str, 상한: str) -> pd.DataFrame:
    """`stock_base_info` 에서 세 칸을 날짜 범위만큼 읽는다."""
    return pd.read_sql_query(
        "SELECT bas_dd, code, kind_stkcert_tp_nm, secugrp_nm, sect_tp_nm "
        "FROM stock_base_info "
        "WHERE bas_dd BETWEEN ? AND ? AND known_at <= ?",
        conn, params=(처음, 끝, 상한),
    )


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


__all__ = ["SECURITY_TYPE_COLUMNS", "UNIVERSE_COLUMNS", "attach_security_type",
           "common_stocks", "top_by_market_cap", "excluded", "coverage"]
