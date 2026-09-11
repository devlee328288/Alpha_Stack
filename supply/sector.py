"""업종 — 그날 종목이 어느 업종이었나.

모델 파트의 2안(MVP)은 *"시가총액 상위 10개 업종 → 업종마다 시총 상위 4~5종목"* 이다.
그 연결에 필요한 **종목 → 업종** 이 우리 자료 어디에도 없었다.

    daily_price.sector    KOSPI 는 100% 빈 값, KOSDAQ 은 소속부(중견기업부·벤처기업부…)
    index_price           업종지수 24종에 시가총액은 있지만 구성종목이 없다
    corp_profile.sic_nm   표준산업분류 522종 · 73% 만 있고 시점이 없다

있는 곳은 KRX Data Marketplace 의 **업종분류 현황** 화면뿐이다. 홈페이지 이용약관
제10조 ② 가 자동화 수집을 금지하므로 **사람이 화면에서 내려받은 스냅샷**을 반입 엔진으로
들이고(`kind="sector"`), 여기서 시점 규칙을 걸어 내준다.

## 스냅샷은 드문드문 있다 — 그래서 "그날까지 알게 된 가장 최근 것"

연 1회(매년 첫 거래일) + 2024-07-01(체계 개편일) 로 18장이다. 날짜 `d` 의 업종은
**`d` 까지 알게 된(`known_at <= d`) 가장 최근 스냅샷**에서 읽는다. 업종 소속은 잘 안
바뀌어 최대 1년 지연을 감수한다. 그 대신 **뒤의 스냅샷을 앞으로 당겨 쓰는 일은 없다** —
2026년 분류로 2015년을 판정하면 미래참조이고 에러는 나지 않는다.

🔴 2026-09-11 까지는 스냅샷 **날짜**(`bas_dd <= d`)로 잘랐다. 스냅샷은 그날 마감 뒤에 나오므로
   (`known_at` = 다음 거래일) 스냅샷 당일 행이 그날 표를 하루 먼저 봤다 — 실측 40,857행 중
   새 정보 5,300행. `as_of` 하나로는 이것이 막히지 않는다. `as_of` 는 *"지금 어디에 서
   있나"* 이고, 행마다의 경계는 **그 행의 날짜**다.

## 이름이 `index_price` 와 어긋나는 셋

2010·2020·2026 세 시점 실측으로 업종지수명과 그대로 맞는 것이 21개, 어긋나는 것이
셋이다. 원문은 그대로 두고 조인용 이름만 `index_name_for()` 로 바꾼다.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from common.trading_calendar import CalendarOutOfRange, next_session
from ingest.inbox import store as inbox_store
from supply.clock import AsOf, as_bas_dd, latest_known_day
from supply.listing_segment import restart_days_from_db, segment_numbers

#: 반입 규격 이름 (`ingest/inbox/schemas/sector.json`).
SECTOR_KIND = "sector"

#: 이 모듈이 붙여 주는 칸. 부르는 쪽이 이 이름에 기대므로 함부로 바꾸지 않는다.
#:
#: `industry_ambiguous` 는 **그 종목이 그 스냅샷에서 업종 둘에 실려 있었다**는 표시다.
#: 값이 참이면 아래 `resolve_duplicates` 가 규칙으로 하나를 골랐다는 뜻이고, 거짓이면
#: 원본에 하나뿐이었다. 고른 결과만 주고 고른 사실을 감추면, 나중에 "왜 이 종목이 이
#: 업종이지" 를 되짚을 방법이 없다. 쓰는 쪽은 이 칸을 무시해도 되고 걸러내도 된다.
INDUSTRY_COLUMNS = ("industry", "industry_bas_dd", "industry_known_at",
                    "industry_ambiguous")

#: 업종분류 현황의 업종명 → `index_price.index_name`. 실측으로 어긋난 것만 적는다.
#:
#:   은행·기타금융  자기 지수가 없다. 상위 묶음 `금융` 으로 올린다
#:   전기·전자      지수명에는 가운뎃점이 없다
#:   기타제조       자기 지수가 없다. 상위 묶음 `제조` 로 올린다
#:
#: 🔴 `보험`·`증권` 은 **넣지 않는다** — 자기 지수가 따로 있다. 처음에 `금융` 으로 올렸더니
#:    verify_sector.py §5 에서 보험 지수(41.6조)에 붙는 종목이 0 이 됐다. 이름이 그대로
#:    맞는 것은 그대로 둔다. `농업, 임업 및 어업`·`광업` 은 지수 자체가 없다.
INDEX_NAME_MAP: Dict[str, str] = {
    "은행": "금융",
    "기타금융": "금융",
    "전기·전자": "전기전자",
    "기타제조": "제조",
}

#: 🔴 **상위 묶음 지수.** 업종지수처럼 생겼지만 여러 업종을 합친 것이다.
#:
#:   제조  2024-01-02 시총 1,526조 — 전기전자(887조)·화학·제약·금속… 제조업 전부의 합
#:   금융  은행·증권·보험·기타금융의 합 (보험·증권은 자기 지수가 따로 있어 그쪽으로 붙는다)
#:
#: "시가총액 상위 10개 업종" 을 고를 때 이 둘을 넣으면 **같은 종목을 두 번 센다** —
#: 삼성전자는 전기전자에도 제조에도 들어 있다. 실측(verify_sector.py §5)으로 `제조` 를
#: 넣으면 그 자리에 `기타제조` 소형주(퍼시스·조광피혁)가 뽑혀 지수 시총과 종목이 어긋났다.
#: 업종을 고르는 쪽은 이 집합을 빼고 고른다. 종목 → 지수 조인(`index_name_for`)은 그대로다.
UMBRELLA_INDICES = frozenset({"제조", "금융"})


class SectorError(RuntimeError):
    """스냅샷을 쓸 수 없을 때 세운다. 무엇을 해야 하는지까지 문구에 담는다."""


def index_name_for(sector_nm: str) -> str:
    """업종명을 `index_price` 지수명으로. 표에 없으면 그대로."""
    return INDEX_NAME_MAP.get(sector_nm, sector_nm)


# ==================================================
# 1. 스냅샷 — 반입 엔진이 통과시킨 것을 되읽는다
# ==================================================
def snapshots(*, db_path=None) -> pd.DataFrame:
    """통과한 업종 스냅샷 전부. 칸: bas_dd · code · market · sector_nm · known_at.

    `known_at` 은 **스냅샷 날짜의 다음 거래일** — `stock_base_info` 와 같은 규칙이다.
    그날 마감 뒤 확정되는 표라 그날 안에는 알 수 없었다고 본다. 날짜 계산으로 하루를
    더하지 않고 실측 달력을 쓴다(공휴일·임시휴장).
    """
    frame = inbox_store.accepted_frame(SECTOR_KIND, db_path=db_path)
    if frame.empty:
        return pd.DataFrame(columns=["bas_dd", "code", "market", "sector_nm", "known_at"])

    out = frame[["bas_dd", "code", "market", "sector_nm"]].copy()
    out["bas_dd"] = out["bas_dd"].astype(str)
    out["code"] = out["code"].astype(str)

    known: Dict[str, str] = {}
    for day in sorted(out["bas_dd"].unique()):
        try:
            known[day] = next_session(day, db_path)
        except CalendarOutOfRange as exc:
            raise SectorError(
                f"업종 스냅샷 {day} 의 다음 거래일을 몰라 known_at 을 정할 수 없다.\n"
                f"  {exc}\n"
                "  할 일: 그 구간 시세를 먼저 받아 달력을 넓힌다."
            ) from exc
    out["known_at"] = out["bas_dd"].map(known)
    return out.sort_values(["bas_dd", "code"]).reset_index(drop=True)


def _usable(snaps: pd.DataFrame, as_of: AsOf) -> pd.DataFrame:
    """`as_of` 시점에 알 수 있었던 스냅샷만. 이 한 줄이 미래참조를 막는다."""
    return snaps[snaps["known_at"] <= latest_known_day(as_of)]


def resolve_duplicates(snaps: pd.DataFrame) -> pd.DataFrame:
    """한 종목이 업종 둘에 실린 행을 **결정적으로** 하나로 줄이고 `ambiguous` 를 붙인다.

    🔴 **왜 필요한가.** KRX 업종분류 현황 화면은 한 종목을 두 업종에 싣는 때가 있다.
       실측 2026-09-07: KOSDAQ 34행(17쌍) — 글로웍스·해성산업·솔본·신라섬유가 2010~2017
       스냅샷에서 `부동산` 과 `일반서비스` 에 동시에 들어 있다. KOSPI 는 0건이라 KOSDAQ 을
       들이기 전에는 드러나지 않았다.

       이걸 그냥 두면 `industry_as_of` 는 종목당 두 행을 내주고, `attach_industry` 의
       `merge_asof` 는 둘 중 **아무거나** 집는다 — 어느 쪽이 붙는지가 입력 순서에 달려
       재현되지 않는다. 조용히 틀리는 종류다.

    ## 고르는 규칙 — 두 단계

    1. **같은 스냅샷(날짜·시장) 안에서 종목이 많은 업종을 고른다.**
    2. 종목 수가 같으면 **업종명 사전순** 첫 번째. (완전히 결정적으로 만드는 마지막 빗장)

    1번이 그 스냅샷 안만 보므로 미래를 참조하지 않는다. 그러면서 실측으로 맞는다 —
    KRX 자신이 중복을 끝낸 2018-01-02 스냅샷에서 셋을 모두 `일반서비스` 로 정리했는데,
    이 규칙이 **8개 스냅샷 8/8** 로 같은 답을 낸다(부동산 1~3종 vs 일반서비스 65~90종).
    사전순만 쓰면 `부동산` 이 이겨 **0/8** 로 반대가 된다. 그래서 사전순은 2번에만 둔다.

    ⚠️ 2018-01-02 스냅샷을 **규칙에 쓰지는 않는다.** 그건 2010년 시점에서 볼 수 없는
       미래다. 여기서는 규칙이 맞는지 확인하는 데만 썼다.

    업계 분류 표준(GICS·ICB)도 회사마다 업종 하나만 준다 — 매출이 가장 큰 주된 사업으로
    정한다. 우리에겐 매출 비중 자료가 없어 위 대리 규칙을 쓴다.
    """
    if snaps.empty:
        out = snaps.copy()
        out["ambiguous"] = pd.Series(dtype=bool)
        return out

    out = snaps.copy()
    # 🔴 표시는 **줄이기 전에** 찍는다. 하나로 줄인 뒤에는 겹쳤던 흔적이 사라진다.
    겹친행 = out.duplicated(["bas_dd", "code"], keep=False)
    out["ambiguous"] = 겹친행
    if not 겹친행.any():
        return out.reset_index(drop=True)

    # 같은 스냅샷 안에서 그 업종에 종목이 몇인가. 이 수가 클수록 주류 업종이다.
    묶음 = out.groupby(["bas_dd", "market", "sector_nm"])["code"]
    out["_같은업종종목수"] = 묶음.transform("size")
    out = out.sort_values(
        ["bas_dd", "code", "_같은업종종목수", "sector_nm"],
        ascending=[True, True, False, True],   # 종목 수는 많은 쪽부터, 동수면 이름 사전순
    )
    out = out.drop_duplicates(["bas_dd", "code"], keep="first")
    return (out.drop(columns=["_같은업종종목수"])
               .sort_values(["bas_dd", "code"])
               .reset_index(drop=True))


# ==================================================
# 2. 정문 — as_of 없이는 못 지난다
# ==================================================
def industry_as_of(bas_dd: str, *, as_of: AsOf, market: Optional[str] = None,
                   db_path=None) -> pd.DataFrame:
    """거래일 `bas_dd` 의 종목 → 업종.

    칸: code · industry · industry_bas_dd · industry_known_at · industry_ambiguous.

    **그 행의 날짜까지 알게 된**(`known_at <= bas_dd`) 스냅샷 중 가장 최근 **하나**를 통째로
    준다. 스냅샷이 아직 없는 구간이면 빈 표(칸은 있다). 빈 표는 오류가 아니라
    *"그때는 몰랐다"* 다.

    🔴 스냅샷 **날짜**로 자르지 않는다 (2026-09-11 고침). `known_at` 은 스냅샷 날짜의 다음
       거래일이라, 날짜로 자르면 스냅샷 당일 행에 그날 마감 뒤에야 나온 표가 붙는다. `as_of`
       를 넉넉히 주는 학습·반출 경로에서만 드러났다 — 실측 스냅샷 당일 40,857행 전부가
       그랬고, 그중 5,300행(처음 실림 3,630 · 업종 바뀜 1,670)이 새 정보를 하루 먼저 봤다.

    종목마다 **한 행**이다 — 원본이 한 종목을 업종 둘에 실었으면 `resolve_duplicates`
    가 규칙으로 하나를 고르고 `industry_ambiguous` 에 참을 남긴다.
    """
    바스 = as_bas_dd(bas_dd)
    snaps = _usable(snapshots(db_path=db_path), as_of)
    if market:
        snaps = snaps[snaps["market"] == market]
    snaps = snaps[snaps["known_at"] <= 바스]
    if snaps.empty:
        return pd.DataFrame(columns=["code", *INDUSTRY_COLUMNS])
    최근 = snaps["bas_dd"].max()
    pick = resolve_duplicates(snaps[snaps["bas_dd"] == 최근])
    return pd.DataFrame({
        "code": pick["code"].to_numpy(),
        "industry": pick["sector_nm"].to_numpy(),
        "industry_bas_dd": pick["bas_dd"].to_numpy(),
        "industry_known_at": pick["known_at"].to_numpy(),
        "industry_ambiguous": pick["ambiguous"].to_numpy(),
    }).reset_index(drop=True)


def attach_industry(frame: pd.DataFrame, *, as_of: AsOf, db_path=None,
                    restart_days=None) -> pd.DataFrame:
    """시세 표(`bas_dd`·`code` 가 있는 것)에 업종 네 칸을 붙인다.

    행마다 **그 행의 날짜까지 알게 된 가장 최근 스냅샷**을 종목별로 찾는다
    (`merge_asof(direction="backward")`). 붙이는 열쇠는 스냅샷 날짜가 아니라 `known_at` 이다 —
    2026-09-11 고침, 이유는 `industry_as_of` 설명에 있다. 스냅샷이 하나도 없으면 네 칸을
    비워서 돌려준다 — 반출이 업종 때문에 죽어서는 안 되고, 빈 칸은 눈에 띈다.

    🔴 **상장 구간을 건너지 않는다** (2026-09-11 고침). 코드를 다시 받은 회사는 자기가 처음
       실린 스냅샷이 나오기 전까지 빈 칸이다 — 앞 회사가 마지막으로 실린 스냅샷을 받지 않는다
       (`supply.listing_segment`). `restart_days` 는 `종목 → 구간 시작일들` 표이고, 안 주면
       DB 에서 만든다(약 9초). `{}` 는 *"코드 재사용이 없다"* 는 뜻이다.

    ⚠️ 입력 순서를 보존한다. `merge_asof` 는 정렬을 요구하므로 안에서 정렬했다가 되돌린다.
    """
    for col in ("bas_dd", "code"):
        if col not in frame.columns:
            raise SectorError(f"업종을 붙이려면 '{col}' 칸이 있어야 한다 — 시세 표를 넘겨라.")

    out = frame.copy()
    snaps = _usable(snapshots(db_path=db_path), as_of)
    if snaps.empty:
        for col in INDUSTRY_COLUMNS:
            out[col] = pd.Series([None] * len(out), index=out.index, dtype="object")
        return out
    if restart_days is None:
        restart_days = restart_days_from_db(db_path)

    # 🔴 `merge_asof` 는 오른쪽에 (종목, 열쇠) 가 겹치면 **어느 행이 붙을지 보장하지 않는다.**
    #    스냅샷마다 하나로 줄여 두어야 몇 번을 돌려도 같은 업종이 붙는다.
    #    (`resolve_duplicates` 가 날짜별로 세므로 표를 통째로 넘겨도 날짜가 섞이지 않는다.)
    snaps = resolve_duplicates(snaps)
    right = snaps.rename(columns={
        "bas_dd": "industry_bas_dd", "sector_nm": "industry", "known_at": "industry_known_at",
        "ambiguous": "industry_ambiguous",
    })[["code", "industry_bas_dd", "industry", "industry_known_at",
        "industry_ambiguous"]].copy()
    right["code"] = right["code"].astype(str)
    # `merge_asof` 는 `on` 키가 숫자·시각이어야 한다 — YYYYMMDD 문자열은 거부한다.
    # 정수로 바꿔도 순서는 같다(자릿수가 고정된 날짜라서).
    # 🔴 열쇠는 `industry_known_at` 이다. `industry_bas_dd` 로 걸면 스냅샷 당일 행에 그날
    #    마감 뒤에 나온 표가 붙는다.
    right["_key"] = right["industry_known_at"].astype(str).astype(int)
    # 🔴 **상장 구간을 건너지 않는다** (2026-09-11 고침). 스냅샷은 연 1회라, 코드를 다시 받은
    #    회사는 자기가 처음 실린 스냅샷이 나오기 전까지 **앞 회사가 마지막으로 실린 스냅샷**을
    #    받았다 — 036220 은 2024-03-13 ~ 07-01 에 2016-01-04 의 '제약'(74행), 101970 은
    #    2025-03-28 ~ 2026-01-02 에 2015-01-02 의 '금속'(187행). 새 회사의 나중 업종과 우연히
    #    같아 값 대조로는 안 보였다. (종목, 구간번호) 로 걸고, 스냅샷은 **자기 날짜**로 센다.
    right["_seg"] = segment_numbers(right["code"], right["industry_bas_dd"], restart_days)
    # 휴장일 스냅샷이 섞이면 두 스냅샷의 다음 거래일(열쇠)이 같아진다 — 늦은 스냅샷 하나만.
    right = (right.sort_values(["code", "_seg", "_key", "industry_bas_dd"])
                  .drop_duplicates(["code", "_seg", "_key"], keep="last"))

    left = out[["bas_dd", "code"]].copy()
    left["_key"] = left["bas_dd"].astype(str).str.replace("-", "", regex=False).astype(int)
    left["code"] = left["code"].astype(str)
    left["_seg"] = segment_numbers(left["code"], left["_key"].astype(str), restart_days)
    left["_order"] = range(len(left))
    left = left.sort_values(["_key", "code"])
    right = right.sort_values(["_key", "code"])

    merged = pd.merge_asof(left, right, on="_key", by=["code", "_seg"], direction="backward")
    merged = merged.sort_values("_order")
    for col in INDUSTRY_COLUMNS:
        out[col] = merged[col].to_numpy()
    return out


__all__ = [
    "INDEX_NAME_MAP",
    "INDUSTRY_COLUMNS",
    "SECTOR_KIND",
    "UMBRELLA_INDICES",
    "SectorError",
    "attach_industry",
    "index_name_for",
    "industry_as_of",
    "resolve_duplicates",
    "snapshots",
]
