"""가격이 **시장 수익률이 아닌 날**을 가려낸다 — 정리매매 · 거래정지 · 자본변동 · 신규상장.

**왜 필요한가.** 우리 시세는 수정주가가 아니다. 그래서 하루 등락률이 ±30% 를 넘는 행이
920만 행 중 2,080건 있는데, 그건 시장이 그만큼 움직였다는 뜻이 아니다. 그 행을 그대로
학습에 넣으면 모델은 "5일에 -90%" 라는 존재하지 않는 신호를 배우고, 백테스트는 그걸
피하거나 반등을 사서 거대한 가짜 수익을 만든다. 실제로는 유동성이 없어 그 가격에
체결되지 않는다.

**틀린 설명을 쓰면 검사가 아무것도 못 잡는다.** 처음에는 양(+) 극단 354행을 통째로
"액면병합·감자" 라고 불렀다. 그런데 그 정의가 *"등락률이 30.5% 를 넘는 행"* 그 자체라
순환논법이었다 — 이상치를 이상치로 설명한 셈이다. 상장주식수 변동이라는 **독립 근거**로
다시 재 보니 진짜 자본변동은 21행(1.3%)뿐이었다.

실측으로 가른 네 가지 원인 (2026-08-29 · `daily_price` 9,209,812행)
--------------------------------------------------------------------
날짜별 가격제한폭을 넘는 2,080행이 아래 넷으로 **하나도 남김없이** 설명된다.

    A 정리매매        1,923 (92.5%)   체결이 끊기기 직전 10체결일
    B 거래정지 재개      468 (22.5%)   직전 행이 zero-OHLC — 등락률이 정지 중 잔존종가 기준
    C 자본변동            22 ( 1.1%)   상장주식수가 그 날 바뀜 (액면병합·감자·증자)
    D 신규상장 첫날      155 ( 7.5%)   등락률이 전일종가가 아니라 **공모가** 기준
    ── 설명 안 됨          0 ( 0.0%)

(합이 2,080 을 넘는 것은 한 행이 둘 이상에 해당할 수 있기 때문이다.)

### A 를 "소멸 종목의 마지막 10거래일" 로 두면 구조적으로 새는 곳이 있다

상장폐지 절차를 밟다가 **거래가 재개된 종목**은 정리매매가 이력 한가운데에 있다.
그 규칙은 이력의 끝만 보므로 영원히 닿지 않는다. 실제로 셋이 그렇게 빠져 있었다.

    감마누(192410)  → THQ → 휴림네트웍스 → 오늘이엔엠
    인포피아(036220) → 오상헬스케어  (2016-05 ~ 2024-03, 8년 공백)
    우양에이치씨(101970)

그래서 **"체결이 끊기기 직전"** 으로 일반화한다. 영구 소멸은 그 특수한 경우다.

### 거래정지는 두 가지 모양으로 나타난다

    ① 행 자체가 없다            상장폐지 · 장기정지
    ② 행은 있는데 zero-OHLC      open=high=low=0, volume=0  (감마누만 656행)

②를 세지 않으면 감마누의 정리매매 5거래일이 통째로 빠진다. 그래서 거리를 잴 때는
행이 아니라 **체결(`open>0` 이고 `volume>0`)** 을 이어 붙인다.

### D 는 제도 변경일이 자료로 확인된다

신규상장 첫날 극단 155행이 **전부 2023-06-26 이후**고 그 이전은 0행이다. KRX 가 신규
상장일 가격범위를 공모가의 60~400% 로 넓힌 바로 그 날짜다. 최대가 정확히 `+300.0%`,
최소가 정확히 `-40.0%` 이고, `close - change` 로 역산한 기준가도 11,000 · 100,000 ·
6,000원처럼 공모가 자릿수다.

⚠️ **수집 시작일의 첫 행은 신규상장이 아니다.** 우리 자료는 2010-01-04 부터인데 그 날
   첫 행이 생기는 종목이 1,961개다. 그건 상장이 아니라 수집 경계다. 실제로 그 1,961개의
   첫 행에는 극단이 **0건**이라 둘이 깨끗하게 갈린다.

### D 는 **한 종목의 첫 행**만이 아니다 — 종목코드는 재사용된다

상장폐지된 종목의 코드를 몇 년 뒤 다른 회사가 다시 받는다. 그러면 한 코드 안에 서로
다른 회사의 시계열이 둘 들어 있고, 뒤 회사의 첫 거래일도 **신규상장**이다.

    036220  인포피아 ~2016-05-04  →  오상헬스케어 2024-03-13~   공백 1,931 거래일
    101970  우양에이치씨 ~2015-03-16 → 우양에이치씨 2025-03-28~  공백 2,465 거래일

**이름으로는 못 가른다** — `101970` 은 앞뒤가 같은 이름이다. **ISIN 으로도 못 가른다** —
KRX 가 표준코드를 재발급해 `KR7036220002` · `KR7101970002` 가 양쪽에서 같다. (CRSP 의
PERMNO 는 영구 식별자로 재사용되지 않지만, 우리가 받는 KRX 표준코드는 그렇지 않다.)

가르는 것은 **상장일**(`stock_base_info.list_dd`)이다 — 거래소가 직접 주는 사실이다.

    036220   list_dd 20070605 → 20240313
    101970   list_dd 20120726 → 20250328

🔴 **그런데 상장일만 보면 시장 이전이 섞인다.** `list_dd` 가 바뀐 코드는 전 구간 24종인데
   그 중 **22종이 KOSDAQ → KOSPI 이전**이다(카카오 20170710 · 동서 20160715 · 무학
   20100720 · 신세계푸드 20100429 · 포스코퓨처엠 20190529 …). 시장을 옮겨도 같은 회사이고
   하루도 안 쉬므로 **공백이 없다.**

그래서 조건 **둘을 함께** 본다 — `is_series_restart` 다.

    거래일 공백이 있다  AND  상장일이 그 공백 뒤에 새로 생겼다  →  새 시계열의 첫 행

**경계값이 없다.** 전 구간 9,231,938행에서 같은 코드 안의 공백은 **2건뿐이고 1~1000
거래일 구간은 통째로 0건**이다(2026-09-09 실측). 정상적인 거래정지는 공백을 만들지
않기 때문이다 — 정지 중에도 행이 있고 `volume=0` 이다. 그래서 60이든 250이든 500이든,
거래일이든 달력일이든 **잡히는 것이 같은 2건**이라 문턱을 고를 필요가 없다.

    공백 있음 · 상장일 신규   →  새 시계열      2종 (위 둘)
    공백 없음 · 상장일 신규   →  그대로        22종 (시장 이전 — 같은 회사)
    공백 있음 · 상장일 그대로 →  그대로         0종 (실측에 없다)
    정상 거래정지            →  그대로        행이 있고 volume=0 이라 공백이 아니다

⚠️ **`list_dd` 는 `stock_base_info` 에 있고 그 표는 2010-01-04 부터다.** 수집 시작 이전에
   일어난 재상장은 알 수 없다. 지금 자료의 공백 2건은 둘 다 2015년 이후라 문제가 없다.

🔴 이 플래그는 **미래를 본다** — 피처로 쓰면 안 된다
------------------------------------------------------
A 는 *"이 뒤로 체결이 끊긴다"* 를 보고 판정한다. 즉 그 시점에는 알 수 없는 사실이다.
학습 자료에서 **행을 덜어내는 용도**(표본 선택)로만 쓴다. 예측 시점의 피처로 넣거나
`supply/` 가 `as_of` 안에서 계산하게 두면 그건 곧 미래 참조다.

덜어내는 양은 작다 — 정리매매 20,763행(0.23%) · 신규상장 1,716행(0.02%).

사용법
------
    from common.corporate_actions import flag_series, market_calendar_index

    index, last = market_calendar_index(con)
    flags = flag_series(rows, calendar_index=index, market_last_index=last,
                        still_listed=code in present_on_last_day,
                        collect_start=calendar[0])
    깨끗한행 = [r for r, f in zip(rows, flags) if not f.liquidation]
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

# ==================================================
# 1. 상수 — 전부 실측 근거가 있다
# ==================================================

#: 정리매매로 볼 마지막 **체결일** 수. 한국은 상장폐지가 확정되면 통상 7거래일간
#: 정리매매를 하는데 그 구간만 가격제한폭이 적용되지 않는다. 넉넉히 10일을 본다.
#: (감마누 2018-09-28~10-05 는 5일, 우양에이치씨 2015-03-06~03-16 은 7일이었다.)
LIQUIDATION_DAYS = 10

#: 체결이 이만큼 끊기면 "이탈" 로 본다. 조회공시 요구로 하루이틀 멈추는 것과
#: 상장폐지 절차로 몇 달 멈추는 것을 가르는 선이다.
#: ⚠️ 10~120 어느 값을 넣어도 설명되지 않는 극단은 0으로 같았다. 문턱에 둔감하다.
SUSPENSION_GAP_DAYS = 20

#: 가격제한폭이 ±15% 에서 ±30% 로 넓어진 날.
#: 실측으로 확인된다 — |등락률|>15.5% 인 행이 2014년 144건에서 2015년 2,745건으로 뛴다.
PRICE_LIMIT_CHANGE_DAY = "20150615"

#: 가격제한폭 밖으로 판정할 경계(%). 반올림 여유 0.5%p 를 얹은 값이다.
#: ⚠️ 전 구간에 30.5 를 쓰면 2015년 이전이 사각지대가 된다 — 그 구간의 상한은 15%다.
#:    날짜별로 갈랐더니 극단이 1,576 → 2,080행으로 늘었고, 늘어난 504행도 전부
#:    아래 플래그로 설명됐다(잔여 0). 즉 엄격하게 더 나은 문턱이다.
LIMIT_BEFORE_2015 = 15.5
LIMIT_AFTER_2015 = 30.5

#: 지수에는 가격제한폭이 없다. 실측 최대 일변동은 코스피 200 이 +19.98%,
#: 섹터지수가 +26.53% 다(2026-07-31). 10% 는 "사람이 눈으로 볼 값" 의 문턱이지
#: 오류의 경계가 아니다 — 그래서 이 값을 넘어도 게이트를 세우지 않는다.
INDEX_MOVE_NOTICE_PCT = 10.0


def price_limit_pct(bas_dd: str) -> float:
    """그 날짜에 물리적으로 불가능한 일간 등락률의 경계(%)."""
    return LIMIT_AFTER_2015 if bas_dd >= PRICE_LIMIT_CHANGE_DAY else LIMIT_BEFORE_2015


# ==================================================
# 2. 한 행의 판정 결과
# ==================================================
@dataclass(frozen=True)
class RowFlags:
    """이 행의 가격 변화가 시장 수익률이 아닐 수 있는 이유들.

    넷 다 False 면 **평범한 거래일**이다. 극단 등락률인데 넷 다 False 라면 우리가
    모르는 일이 벌어진 것이므로, 품질 게이트는 바로 그 경우에만 빨간불을 켠다.
    """

    liquidation: bool = False       # A 정리매매 (체결 단절 직전)
    halt_resume: bool = False       # B 거래정지 재개 (직전이 zero-OHLC)
    capital_change: bool = False    # C 자본변동 (상장주식수 변경)
    first_listing: bool = False     # D 신규상장 첫 거래일

    @property
    def explained(self) -> bool:
        """넷 중 하나라도 해당하면 True."""
        return self.liquidation or self.halt_resume or self.capital_change \
            or self.first_listing

    def names(self) -> Tuple[str, ...]:
        """해당하는 이유의 이름들. 리포트에 그대로 싣는다."""
        pairs = (("liquidation", self.liquidation), ("halt_resume", self.halt_resume),
                 ("capital_change", self.capital_change),
                 ("first_listing", self.first_listing))
        return tuple(name for name, on in pairs if on)


def is_halted(row: Mapping) -> bool:
    """이 행이 **거래정지 표시행**인가 — `open=high=low=0`.

    KRX 는 정지 중에도 행을 주는데 시·고·저가 0 이고 종가만 직전 값을 물고 있다.
    그 종가가 다음 재개일 등락률의 기준이 되기 때문에, 정지행을 못 알아보면
    재개일의 등락률이 어디서 왔는지 설명할 수 없다.
    """
    return row["open"] == 0 and row["high"] == 0 and row["low"] == 0


def is_traded(row: Mapping) -> bool:
    """이 행에 **체결이 있었나**. 정리매매 거리를 재는 단위다.

    거래량만 보면 부족하다 — 정지행도 거래량 0 이지만, 시가 0 이 더 확실한 표시다.
    둘을 함께 요구해 "값은 있는데 체결은 없었다" 를 걸러 낸다.
    """
    return bool(row["open"]) and row["open"] > 0 and bool(row["volume"]) and row["volume"] > 0


# ==================================================
# 3. 시장 달력 — 거리를 세는 자
# ==================================================
def market_calendar(con: sqlite3.Connection) -> List[str]:
    """시세 표에 존재하는 모든 거래일. 종목의 구멍을 판정할 자가 된다.

    ⚠️ 달력을 종목별 행 번호로 대신하면 안 된다. 3개월 정지된 종목의 정지 전후
       가격 차이가 "5일 수익률" 로 둔갑한다. 에러는 나지 않고 수익만 부풀려진다.
    """
    return [r[0] for r in con.execute(
        "SELECT DISTINCT bas_dd FROM daily_price ORDER BY bas_dd")]


def market_calendar_index(con: sqlite3.Connection) -> Tuple[Dict[str, int], int]:
    """`(거래일 → 순번, 마지막 순번)`. `flag_series` 가 그대로 받는 모양이다."""
    days = market_calendar(con)
    return {d: i for i, d in enumerate(days)}, len(days) - 1


def listing_days_by_code(con: sqlite3.Connection) -> Dict[str, Tuple[str, ...]]:
    """`종목코드 → 그 코드에 붙었던 상장일들` (오름차순).

    코드를 재사용한 종목은 값이 둘이다 — `036220` 은 `('20070605', '20240313')`.
    나머지는 하나뿐이고, 시장 이전 22종도 둘이지만 공백이 없어 `is_series_restart` 가
    걸러 낸다.

    `stock_base_info` 는 날짜마다 행이 있어 920만 행이지만 `DISTINCT` 로 줄이면 수천
    행이다 — 종목마다 다시 부르지 말고 한 번 만들어 `MarketContext` 에 담아 돌려 쓴다.
    """
    out: Dict[str, list] = {}
    for code, list_dd in con.execute(
        "SELECT DISTINCT code, list_dd FROM stock_base_info "
        "WHERE list_dd IS NOT NULL ORDER BY code, list_dd"
    ):
        out.setdefault(str(code), []).append(str(list_dd))
    return {code: tuple(days) for code, days in out.items()}


def is_series_restart(prev_row: Mapping, row: Mapping, *,
                      calendar_index: Mapping[str, int],
                      listing_days: Sequence[str]) -> bool:
    """이 행이 **다른 회사의 첫 거래일**인가 — 즉 앞 행과 같은 시계열이 아닌가.

    조건 둘을 함께 본다 (근거는 모듈 머리말의 "종목코드는 재사용된다" 절).

        ① 앞 행과 사이에 거래일 공백이 있다        (길이 무관 — 문턱이 없다)
        ② 그 공백 뒤에 새 상장일이 생겼다          (`prev < list_dd <= 이 행`)

    ①만으로는 부족하다 — 상장폐지 뒤 재등장인지 우리가 모르는 자료 구멍인지 갈리지
    않는다. ②만으로는 시장 이전 22종을 새 회사로 잘못 본다(공백이 없다).

    `listing_days` 는 그 코드에 붙었던 상장일 전부다(`listing_days_by_code`). 비어
    있으면 늘 `False` 다 — **모르는 것을 단절로 치지 않는다.** 조정을 놓치면 값이
    틀리지만, 없는 단절을 만들면 멀쩡한 시계열을 둘로 쪼갠다.

    ⚠️ 달력에 없는 날짜가 오면 그 행은 판정하지 않는다(`False`). 달력은 시세 표에서
       만들므로 정상 경로에서는 일어나지 않지만, 부분 표를 넘겼을 때 조용히 틀린
       공백을 만드는 것보다 판정을 포기하는 쪽이 안전하다.
    """
    if not listing_days:
        return False
    앞날, 이날 = str(prev_row["bas_dd"]), str(row["bas_dd"])
    if 앞날 not in calendar_index or 이날 not in calendar_index:
        return False
    if calendar_index[이날] - calendar_index[앞날] - 1 <= 0:      # ① 공백이 없다
        return False
    return any(앞날 < str(day) <= 이날 for day in listing_days)   # ② 새 상장일


def codes_present_on(con: sqlite3.Connection, bas_dd: str) -> Set[str]:
    """그 날 시세 표에 **행이 있는** 종목들.

    마지막 거래일로 부르면 "아직 상장돼 있는 종목" 이 된다. 소멸 판정에 쓴다.

    ⚠️ 체결 여부가 아니라 **행의 존재**로 판정한다. 그 날 거래정지라 체결이 없어도
       상장은 돼 있는 것이고, 그 둘을 섞으면 조회공시로 이틀 멈춘 멀쩡한 종목까지
       정리매매로 잡힌다 (실측: 910종목이어야 할 것이 1,045종목이 됐다).
    """
    return {r[0] for r in con.execute(
        "SELECT code FROM daily_price WHERE bas_dd = ?", (bas_dd,))}


# ==================================================
# 4. 판정
# ==================================================
def liquidation_positions(rows: Sequence[Mapping], *,
                          calendar_index: Mapping[str, int],
                          market_last_index: int,
                          still_listed: bool,
                          liquidation_days: int = LIQUIDATION_DAYS,
                          gap_days: int = SUSPENSION_GAP_DAYS) -> Set[int]:
    """정리매매로 볼 행의 **위치(rows 안의 인덱스)** 집합.

    `rows` 는 한 종목의 **전 구간**이어야 한다. 잘라서 주면 잘린 자리가 곧 "체결
    단절" 로 보여 없는 정리매매가 생긴다.

    이탈로 보는 자리는 셋이다.
      ① 다음 체결까지 `gap_days` 거래일 이상 빈다      — 중간 정지 후 재개된 종목
      ②a 마지막 거래일에 **행이 아예 없다**             — 소멸 (진행 중 정리매매 포함)
      ②b 마지막 체결 이후 `gap_days` 이상 지났다        — 장기 정지 중

    ②a 가 없으면 수집 끝단이 새다. 실측 2026-08-24 의 시스웍(269620)은 889원에서
    1원까지 7거래일 만에 빠지고 그 다음 날 표에서 사라졌는데, ① 만으로는 20거래일이
    아직 흐르지 않아 잡히지 않았다.
    """
    traded = [i for i, r in enumerate(rows) if is_traded(r)]
    if not traded:
        return set()

    positions = [calendar_index[rows[i]["bas_dd"]] for i in traded]
    exits = [j for j in range(len(traded) - 1)
             if positions[j + 1] - positions[j] - 1 >= gap_days]          # ①
    if not still_listed or market_last_index - positions[-1] >= gap_days:  # ②a · ②b
        exits.append(len(traded) - 1)

    flagged: Set[int] = set()
    for j in exits:
        for k in range(max(0, j - liquidation_days + 1), j + 1):
            flagged.add(traded[k])
    return flagged


def flag_series(rows: Sequence[Mapping], *,
                calendar_index: Mapping[str, int],
                market_last_index: int,
                still_listed: bool,
                collect_start: str,
                listing_days: Sequence[str] = (),
                liquidation_days: int = LIQUIDATION_DAYS,
                gap_days: int = SUSPENSION_GAP_DAYS) -> List[RowFlags]:
    """한 종목의 전 구간에 대해 행마다 `RowFlags` 를 매긴다.

    `rows` 는 `bas_dd` 오름차순이어야 하고 **그 종목의 전부**여야 한다
    (`liquidation_positions` 와 `first_listing` 둘 다 경계에 의존한다).

    `collect_start` 는 수집 시작 거래일이다. 그 날의 첫 행은 상장이 아니라 수집
    경계이므로 `first_listing` 을 켜지 않는다.

    `listing_days` 는 그 코드에 붙었던 상장일 전부다(`listing_days_by_code` 의 그 코드
    항목). 주면 **코드 재사용으로 시계열이 끊긴 자리도 `first_listing`** 이 된다 —
    한 코드 안에 회사가 둘이면 뒤 회사의 첫 거래일도 신규상장이기 때문이다.
    안 주면 예전 그대로 종목의 첫 행만 본다.
    """
    liquidation = liquidation_positions(
        rows, calendar_index=calendar_index, market_last_index=market_last_index,
        still_listed=still_listed, liquidation_days=liquidation_days, gap_days=gap_days)

    out: List[RowFlags] = []
    for i, row in enumerate(rows):
        prev = rows[i - 1] if i > 0 else None
        capital = bool(
            prev is not None
            and prev["listed_shares"] is not None
            and row["listed_shares"] is not None
            and row["listed_shares"] != prev["listed_shares"]
        )
        # 코드 재사용으로 시계열이 끊긴 자리도 신규상장이다 — 한 코드 안에 회사가
        # 둘이면 뒤 회사의 첫 거래일이 그 회사의 상장일이다.
        재시작 = bool(
            prev is not None
            and is_series_restart(prev, row, calendar_index=calendar_index,
                                  listing_days=listing_days)
        )
        out.append(RowFlags(
            liquidation=i in liquidation,
            halt_resume=prev is not None and is_halted(prev),
            capital_change=capital,
            first_listing=(i == 0 and row["bas_dd"] != collect_start) or 재시작,
        ))
    return out


def is_outlier(row: Mapping) -> bool:
    """이 행의 등락률이 그 날의 가격제한폭 밖인가 — 즉 물리적으로 불가능한 값인가."""
    rate = row["change_rate"]
    return rate is not None and abs(rate) > price_limit_pct(row["bas_dd"])


def basis_price(row: Mapping):
    """이 행이 등락률을 계산할 때 쓴 **기준가**. `종가 - 전일대비` 다.

    평소에는 전일 종가와 같지만, 액면분할·병합·감자·무상증자·주식배당이 있으면
    KRX 가 기준가를 조정하기 때문에 갈라진다.
    """
    if row["close"] is None or row["change"] is None:
        return None
    return row["close"] - row["change"]


def is_basis_adjusted(prev_row: Mapping, row: Mapping) -> bool:
    """KRX 가 이 날 **기준가를 조정**했나 — 즉 가격이 연속되지 않는 날인가.

    🔴 **등락률로는 안 보인다.** KRX 가 조정된 기준가로 등락률을 계산해 주므로 그 값은
       멀쩡하다. 그래서 가격제한폭 검사에 안 걸린다. 그런데 우리 라벨은 **시가 비율**
       (`시가(t+6)/시가(t+1)`)이라 조정폭이 그대로 수익률로 섞인다.

    실측 (2026-08-29 · 5,223일)

        기준가/전일종가   최소 0.0019배 · 중앙 0.9541배 · 최대 120배
        ×1.5 초과            964 (18.5%)   액면병합 · 감자
        ×0.5 미만            509 ( 9.7%)   액면분할 · 무상증자
        ×0.9~0.99          1,551 (29.7%)   권리락
        12월에 1,045건 몰림 — 12월 결산법인의 주식배당 권리락이다

    유한양행은 매년 12월 마지막 거래일에 걸린다. 2020-12-29 는 전일종가 76,600 →
    기준가 73,300 이라, 종가 비율로 재면 **-3.52%** 지만 실제 등락률은 **+0.82%** 다.

    ⚠️ 라벨에 준 영향은 실측했다 — KOSPI 개발구간에서 창의 0.140%만 오염됐고
       `E|5일수익|` 은 4.4832% → 4.4760% 로 0.0072%p 움직였다. 지금은 무시할 크기지만
       ×120 짜리 조정이 표본 안으로 들어오면 이야기가 달라진다. 그래서 센다.

    ⚠️ 상장주식수 변동과 **다른 사건**이다. 유상증자는 주식수만 늘리고 기준가를
       안 바꾼다(실측 39,470일). 반대로 주식배당은 기준가만 바꾼다(3,726일).
       둘 다 봐야 자본변동을 놓치지 않는다.
    """
    기준가 = basis_price(row)
    앞종가 = prev_row["close"]
    return 기준가 is not None and 앞종가 is not None and 기준가 != 앞종가


# ==================================================
# 5. 끊긴 가격을 이어 붙이는 배율
# ==================================================
#
# 위 `is_basis_adjusted` 는 **끊긴 날을 찾기만** 한다. 여기서는 그 날을 **이어 붙인다.**
#
# 왜 여기에 두나
# --------------
# `supply/training.py` 는 자본변동 행을 일부러 남긴다 — 그 날의 가격도 실제 체결가이기
# 때문이다. 그 docstring 이 "이어 붙이는 쪽(라벨 계산)이 `is_basis_adjusted` 로 판단할
# 일" 이라고 못박아 두었는데, 정작 이어 붙이는 함수가 없었다. 그래서 시가 비율로 만든
# 라벨에 조정폭이 그대로 섞였다 — 삼성전자 2018-05-04 가 **-98.04%** 로 읽힌다.
#
# 왜 `change` 로 재나 — 세 방식을 재 보고 골랐다 (2026-09-01 실측, 9,220,879행)
# ---------------------------------------------------------------------------
#     기준가(close-change)   삼성전자 4,100일 중 KRX 등락률과 0.01%p 넘게 어긋난 날 **0일**
#                            최대 오차 0.0050%p — KRX 가 등락률을 소수 2자리로 반올림한
#                            한계와 정확히 같다. 즉 이 방식 자체의 오차는 없다.
#     change_rate 누적       16년을 누적하면 **0.1235%** 벌어진다. 반올림이 쌓인다.
#     listed_shares 배율     ❌ 유상증자 39,515건을 오검출하고 권리락 3,730건을 놓친다.
#
# `change` 는 정수 원 단위라 반올림 오차가 없다. 삼성전자 2018-05-04 은
# `51,900 - (-1,100) = 53,000`, `53,000 / 2,650,000 = 0.020000` 으로 **정확히 1/50** 이다.
#
# 세 사건은 서로 다르다 (같은 실측)
# --------------------------------
#     기준가와 주식수 둘 다 바뀜    1,516   액면분할 · 병합
#     기준가만 바뀜                3,730   주식배당 · 권리락 (주식수는 그대로)
#     주식수만 바뀜               39,515   유상증자 등 — **가격이 연속이라 조정 대상이 아니다**
#
# 조정이 일어난 날은 전체의 0.057%(5,246행 · 2,181종)이고, 계수는 최소 0.0019배부터
# 최대 120.0배까지 있다. 12월에 1,045건이 몰리는데 12월 결산법인의 주식배당 권리락이다.


#: 재개일에 상장주식수가 이만큼 움직였으면 **기계적 자본변동**(액면분할·병합·감자)으로
#: 인정한다. 실측에서 진짜 액면분할·병합은 전부 ×2 이상이었고, 잡음은 ×1.1 아래였다.
#: `ingest/store/adj_price.py` 의 관문(FDR 이 안 편 자본변동)도 같은 값을 쓴다.
SHARE_RATIO_MIN = Fraction(3, 2)

#: 재개일에 상장주식수가 이만큼은 움직여야 "주식수가 바뀐 사건" 으로 본다.
#: 인적분할은 주식수가 1.1~1.5배만 움직이는 경우가 있다(NAVER 2013-08-29 는 ×0.685).
#: 그 아래(유상증자 잡음)는 주식수가 그대로인 것과 같이 다룬다.
RESUME_SHARE_CHANGE_MIN = Fraction(11, 10)

#: 재개일 기준가가 **기계적 배율**을 따른 것으로 볼 폭. KRX 는 감자 종목의 시초가를
#: 평가가격(전일종가 × 감자비율)의 50~150% 호가 범위에서 정한다(유가증권시장 업무규정
#: 시행세칙 별표 — 최저호가 50% · 자본금감소 종목 최고호가 150%). 액면분할·병합은 호가
#: 없이 평가가격 그대로라 정확히 1.0 이다. 이 폭을 벗어났다면 주식수 배율로는 그날
#: 기준가를 설명할 수 없다는 뜻이다.
RESUME_BASIS_BAND = (Fraction(1, 2), Fraction(3, 2))


def adjustment_factor(prev_row: Mapping, row: Mapping) -> Fraction:
    """이 날 가격이 끊긴 배율. 안 끊겼으면 `Fraction(1)`.

    **평상일** — `기준가 / 전일종가` 다. 액면분할 50:1 이면 `1/50`, 병합 5:1 이면 `5`.
    `close`·`change` 가 둘 다 정수라 이 값은 **정확한 유리수**다.
    `listed_shares` 가 못 잡는 무상증자·주식배당 권리락까지 여기서 잡힌다.

    🔴 **재개일(직전 행이 거래정지)에는 기준가를 그냥 믿지 않는다.**
    -------------------------------------------------------
    정지 중 종가는 직전 값을 붙들고 있고, 재개일에 KRX 는 **단일가로 기준가를 새로
    잡는다.** 그 기준가를 무조건 조정계수로 쓰면 자본변동이 아닌 것을 조정으로 오해한다.

        111610  20150817  전일종가(정지) 145 → 기준가 17,400  계수 120.0
                          그런데 주식수배율은 0.9818 — **주식수는 그대로다**

    이걸 그대로 믿으면 과거 전체를 **120배** 부풀린다. 실측에서 기준가 조정 5,246건 중
    **2,057건(39.2%)이 이 재개일**이었다.

    그래서 재개일은 **상장주식수가 바뀌었는지**로 먼저 가른다. 주식수가 그대로(변동
    10% 미만)면 정지 중에 시장이 다시 매긴 값이라 수익률이고, 조정하지 않는다.

    재개일에 주식수가 바뀌었으면 — 세 갈래다 (2026-09-07 실측 · DART 공시로 전수 대조)
    -----------------------------------------------------------------------------
    처음 규칙은 "주식수 배율이 곧 가격 배율" 하나였다. 감자 5:1 이면 가격 5배 — 맞다.
    그런데 재개일에 주식수가 바뀐 1,305건에 아래 규칙을 돌리면 **223건의 계수가 달라진다**
    (FDR 구간 171 · chain 52). 실제로 우리 값을 정하는 chain 구간 52건(51종목)을 공시로
    확인하니 두 종류였다(노트북 06/10).

        ① 인적분할·지주전환 24건  신세계 2011-06-10 주식수 ×1/3.83 · 기준가 1.31배
                                  가치의 일부가 이마트로 **나갔다.** 배율 3.83 을 그대로
                                  쓰면 분할 전 가격이 3.83배 부풀어 −60.6% 가짜 폭락이 된다
        ② 회생·합병 + 감자   28건  남광토건 2013-02-15 감자 2.67:1 · 기준가 40배
                                  출자전환으로 회사가 새로 값이 매겨졌다. 배율 2.67 만
                                  쓰면 나머지 15배가 +1,172% 가짜 수익률로 남는다

    둘 다 KRX 가 평가가격을 정하고 호가 범위(분할·합병 재상장 50~200% · 감자 50~150%)
    안에서 시초가를 잡아 그것을 그날 기준가로 삼았다. 수정주가의 표준 처리도 같다 —
    인적분할은 "빠져나간 가치만큼 과거가를 축소" 하되 그 가치는 **시장이 매긴 값**을
    쓰고(CRSP · Xignite), 회생 재상장은 시리즈를 끊어 정지 구간을 가로지르는 수익률을
    만들지 않는다(CRSP new PERMNO). 그래서 —

        ㉠ 기준가가 주식수 배율의 50~150% 안이다 → **정확한 유리수 배율** (감자·병합·
           액면분할. 시초가가 평가가격에서 얼마나 움직였든 그건 수익률이다)   1,026건
        ㉡ 그 밖이다 → **기준가 / 전일종가** (①②. 배율로는 기준가를 설명할 수 없다)
        ㉢ 주식수 변동이 1.1~1.5배로 작다 → 감자는 실측에서 전부 2배 이상이므로
           분할·합병으로 보고 기준가를 쓴다 (NAVER 2013-08-29 ×0.685 · 기준가 1.567배)

    ㉡ 은 출자전환처럼 **주식수만 늘고 가격은 연속인** 사건도 바로잡는다 — 태영건설
    2024-07-22 는 주식수 ×24.9 인데 KRX 등락률 0.00% 라 기준가비가 1 이고, 계수도 1 이다.
    옛 규칙은 여기서 1/24.9 를 만들었다.

    FDR 구간은 영향이 없다 — 계수가 달라지는 171건 중 170건을 FDR 이 이미 기준가만큼 펴
    놓았고(KRX 등락률과 1%p 안 일치 · 예외는 시알홀딩스 2023-07-28 의 2.3%p 하나), 그
    구간에서는 FDR 값을 쓴다.

    ⚠️ **재개일을 통째로 버리면 안 된다.** 액면분할·병합은 주권 교체 때문에 KRX 가
       반드시 거래를 정지시키므로 **전부 재개일에 있다** — 정지일을 빼고 세면 분할이
       0건 남는다. 삼성전자 2018-05-04 도 20180430~0503 정지 뒤의 재개일이다.

    앞 행이 없거나 값이 비면 `1` 이다 — **모르는 것을 조정으로 치지 않는다.**
    조정을 놓치면 수익률이 틀리지만, 없는 조정을 만들면 멀쩡한 가격을 망친다.

    🔴 **알려진 결함 — 코드 재사용 자리에서 이 함수는 틀린 배율을 낸다** (2026-09-09)
    ----------------------------------------------------------------------
    이 함수는 `(앞 행, 이 행)` 두 행만 받으므로 **거래일 공백을 볼 수 없다.** 그래서
    상장폐지된 회사의 마지막 종가와 몇 년 뒤 신규상장한 다른 회사의 기준가로 배율을
    만든다. `is_series_restart` 가 판정하는 바로 그 자리다.

        101970  20150316 종가 830  →  20250328 기준가 18,640    배율 1864/83 = 22.4578배
                그런데 두 행은 **다른 회사**다 (우양에이치씨 → 우양에이치씨, 이름만 같다)

    실측 영향 (`adj_source` 로 전수 확인):

        101970   fdr+ca_fix  648행 (20120726~20150316)  배율 15.379776 ~ 22.457831
                 2015-03-16 의 배율이 정확히 22.457831 = 1864/83 이다 — 위 값이
                 그대로 들어가 2012~2015 **전 구간 수준이 22.46배 부풀었다.**
        036220   fdr        2,171행 전부 배율 1.000000 — FDR 이 직접 준 값이라 새지 않았다

    **수익률은 안 틀린다** — 같은 블록 안 이웃 두 날은 같은 배율로 움직이므로 비율이
    보존되고, 그래서 `adj_quality.is_adj_suspect` 도 그 648행에서 False 다. 틀린 것은
    **수준(level)** 이고, 우리 피처는 대부분 비율이라 지금 결과를 바꾸지 않는다.
    `101970` 은 KOSDAQ 이라 KOSPI 후보군에도 들어오지 않는다.

    고치려면 `factor_series` 가 달력과 상장일을 받아 재시작 자리에서 `1` 을 내야 하고,
    그러면 **수정주가 전체 재생성**이 따라온다(FDR 창이 밀려 옛 행이 함께 바뀐다).
    값이 바뀌는 변경이라 여기서 조용히 하지 않고 이슈로 낸다.
    """
    if prev_row is None:
        return Fraction(1)

    기준가, 앞종가 = basis_price(row), prev_row["close"]
    기준가비 = (Fraction(int(기준가), int(앞종가))
             if 기준가 is not None and 기준가 > 0 and 앞종가 else None)

    if is_halted(prev_row):
        앞주식수, 주식수 = prev_row.get("listed_shares"), row.get("listed_shares")
        if not 앞주식수 or not 주식수:
            return Fraction(1)
        배율 = Fraction(int(주식수), int(앞주식수))
        if 배율 >= SHARE_RATIO_MIN or 배율 <= 1 / SHARE_RATIO_MIN:
            기계적 = 1 / 배율
            if 기준가비 is None:
                return 기계적                                  # 기준가를 모르면 배율뿐이다
            하한, 상한 = RESUME_BASIS_BAND
            if 기계적 * 하한 <= 기준가비 <= 기계적 * 상한:
                return 기계적                                  # ㉠ 감자·병합·액면분할
            return 기준가비                                    # ㉡ 인적분할 · 회생 · 출자전환
        if 배율 >= RESUME_SHARE_CHANGE_MIN or 배율 <= 1 / RESUME_SHARE_CHANGE_MIN:
            return 기준가비 if 기준가비 is not None else Fraction(1)   # ㉢
        return Fraction(1)                                     # 주식수가 그대로다

    if 기준가비 is None:
        return Fraction(1)
    return 기준가비


def factor_series(rows: Sequence[Mapping], *,
                  calendar_index: Optional[Mapping[str, int]] = None,
                  listing_days: Sequence[str] = ()) -> List[Fraction]:
    """한 종목의 행마다 그 날의 조정 배율. `rows` 와 길이가 같다.

    `rows` 는 `bas_dd` 오름차순이어야 하고 **그 종목의 전부**여야 한다.
    첫 행은 앞이 없으므로 `Fraction(1)` 이다.

    🔴 **`calendar_index` 와 `listing_days` 를 주면 코드 재사용 자리를 끊는다.**
    ---------------------------------------------------------------------
    `adjustment_factor` 는 두 행만 받아 거래일 공백을 볼 수 없으므로, 상장폐지된 회사의
    마지막 종가와 몇 년 뒤 신규상장한 **다른 회사**의 기준가로 배율을 만든다. 그 자리에서
    `Fraction(1)` 을 돌려주는 것이 이 두 인자의 역할이다 — **다른 회사 사이에는 조정이
    없다.** 판정은 `is_series_restart` 가 한다.

    실측으로 무슨 일이 벌어졌나 (2026-09-09)

        101970  20150316 종가 830  →  20250328 기준가 18,640     배율 1864/83 = 22.4578
                FDR 은 2012~2015 를 모른다(3,000거래일 창 밖). 그래서
                `adj_price.scale_series` ③ 이 그 배율을 **과거로 퍼뜨려**
                648행(20120726~20150316)의 수준이 22.4578배 부풀었다.

        🔴 그런데 `is_adj_suspect` 는 False 였다 — 같은 배율로 밀린 이웃 두 날의 비율은
           보존되므로 KRX 등락률과 갭이 0.0050%p 다. **검사와 대상이 같은 잘못을 같은
           크기로 공유하면 초록이 나온다.** 크기 필터로도 영원히 안 잡힌다.

    안 주면 예전 그대로다 — **모르는 것을 단절로 치지 않는다.**

    ⚠️ `float` 이 아니라 `Fraction` 을 준다. 1/50 · 1/3 같은 계수가 4,000행에 걸쳐
       곱해지므로, 부동소수로 누적하면 반올림 잡음이 다시 들어온다.
       쓰는 쪽에서 **마지막에 한 번만** `float()` 한다.
    """
    out: List[Fraction] = []
    for i, row in enumerate(rows):
        prev = rows[i - 1] if i > 0 else None
        if (prev is not None and calendar_index is not None
                and is_series_restart(prev, row, calendar_index=calendar_index,
                                      listing_days=listing_days)):
            out.append(Fraction(1))          # 다른 회사다 — 이어 붙일 것이 없다
            continue
        out.append(adjustment_factor(prev, row))
    return out


def span_factor(factors: Sequence[float], entry: int, exit_: int) -> float:
    """`entry` **다음 날부터** `exit_` 날까지 일어난 조정의 누적 배율.

    진입가와 청산가의 **스케일 차이**다. 수익률은 이렇게 고친다.

        raw = 청산가 / 진입가 - 1                    # 틀린다 — 조정폭이 섞인다
        adj = 청산가 / (진입가 * span) - 1           # 옳다

    구간을 벗어난 조정은 보지 않으므로 **재현성이 깨지지 않는다.** 뒤에 새 분할이
    생겨도 이미 계산한 과거 구간의 값은 그대로다 (후방조정과 다른 점이다 —
    `back_adjusted_closes` 의 경고를 보라).

    진입일 **당일**의 조정은 세지 않는다. 그 날 시가는 이미 조정된 기준으로 붙기
    때문이다. 세면 한 번 더 나눠 값이 배로 틀린다.
    """
    if exit_ <= entry:
        return 1.0
    누적 = Fraction(1)
    for k in range(entry + 1, min(exit_, len(factors) - 1) + 1):
        누적 *= factors[k]
    return float(누적)          # 곱셈은 유리수로, 밖으로는 한 번만 float


def back_adjusted_closes(rows: Sequence[Mapping]) -> List[float]:
    """전 구간을 **현재 가격 기준**으로 이어 붙인 종가. 차트·장기 수익률용.

    마지막 행은 원래 종가 그대로이고, 과거로 갈수록 그 뒤에 일어난 모든 조정이
    곱해진다. 삼성전자 2010-01-04 은 원종가 809,000 이 아니라 **16,180.00** 이 된다.

    🔴 **파이프라인에 그대로 쓰지 마라 — 재현성이 깨진다.**
       새 액면분할이 하나 생기면 **과거 전체 값이 바뀐다.** 어제 돌린 결과와 오늘
       돌린 결과가 달라지고, 그래도 에러는 나지 않는다. 학습·라벨에는 구간만 보는
       `span_factor` 를 쓰고, 이 함수는 사람이 보는 그림과 장기 성과 비교에만 쓴다.

    ⚠️ 값이 원 단위를 벗어난다. 액면분할이 잦았던 종목은 소수점 아래로 내려가므로
       가격 자릿수를 가정하는 코드에 그대로 넘기면 안 된다.
    """
    factors = factor_series(rows)
    out: List[float] = [float("nan")] * len(rows)
    누적 = Fraction(1)
    for i in range(len(rows) - 1, -1, -1):
        close = rows[i]["close"]
        out[i] = float(close * 누적) if close is not None else float("nan")
        누적 *= factors[i]          # 이 날의 조정은 그 **앞** 행들에 적용된다
    return out
