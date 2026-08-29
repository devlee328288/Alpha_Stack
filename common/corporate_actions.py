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
from typing import Dict, List, Mapping, Sequence, Set, Tuple

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
                liquidation_days: int = LIQUIDATION_DAYS,
                gap_days: int = SUSPENSION_GAP_DAYS) -> List[RowFlags]:
    """한 종목의 전 구간에 대해 행마다 `RowFlags` 를 매긴다.

    `rows` 는 `bas_dd` 오름차순이어야 하고 **그 종목의 전부**여야 한다
    (`liquidation_positions` 와 `first_listing` 둘 다 경계에 의존한다).

    `collect_start` 는 수집 시작 거래일이다. 그 날의 첫 행은 상장이 아니라 수집
    경계이므로 `first_listing` 을 켜지 않는다.
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
        out.append(RowFlags(
            liquidation=i in liquidation,
            halt_resume=prev is not None and is_halted(prev),
            capital_change=capital,
            first_listing=i == 0 and row["bas_dd"] != collect_start,
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
