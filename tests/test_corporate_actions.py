"""가격이 시장 수익률이 아닌 날을 가려내는 플래그 테스트

**왜 이 테스트가 필요한가.** 여기서 틀리면 **에러가 안 난다.** 정리매매 구간이 학습
자료에 남으면 모델은 "5일에 -90%" 라는 존재하지 않는 신호를 배우고, 백테스트는 그걸
피하거나 반등을 사서 거대한 가짜 수익을 낸다. 반대로 너무 많이 잘라 내면 멀쩡한
자료가 조용히 사라진다. 둘 다 예외 없이 진행된다.

특히 못 박아 두는 것 셋 — 전부 실제 자료에서 한 번씩 틀렸던 자리다.

1. **정리매매를 "소멸 종목의 마지막 N일" 로 두면 재개된 종목을 영원히 못 잡는다.**
   감마누·인포피아·우양에이치씨가 그렇게 빠져 있었다.
2. **거래정지가 행 삭제가 아니라 zero-OHLC 행으로 오는 경우가 있다.**
   그걸 세지 않으면 감마누의 정리매매 5거래일이 통째로 빠진다.
3. **수집 시작일의 첫 행은 신규상장이 아니다.** 우리 자료에서 1,961종목이 그렇다.
"""

from __future__ import annotations

from datetime import date, timedelta
from fractions import Fraction

from common import corporate_actions as ca


def _평일들(n: int, 시작: date = date(2024, 1, 2)) -> list:
    """2024-01-02(화)부터 평일 `n`개. 손으로 적으면 주말이 섞인다."""
    out, day = [], 시작
    while len(out) < n:
        if day.weekday() < 5:
            out.append(day.strftime("%Y%m%d"))
        day += timedelta(days=1)
    return out


# ⚠️ 달력을 `SUSPENSION_GAP_DAYS`(20)보다 **길게** 잡는다. 짧으면 어떤 공백도
#    문턱에 못 닿아서 "정리매매가 안 잡힌다" 는 결과가 나오는데, 그건 코드가
#    틀려서가 아니라 시험대가 좁아서다. 실제로 처음에 15일로 잡아 그랬다.
CAL = _평일들(40)
INDEX = {d: i for i, d in enumerate(CAL)}
LAST = len(CAL) - 1


def _행(bas_dd: str, *, close: int = 1000, change: int = 0, volume: int = 1000,
        shares: int = 100, halted: bool = False) -> dict:
    """한 행. `halted=True` 면 KRX 가 정지 중에 주는 모양(시·고·저가 0)이 된다."""
    price = 0 if halted else close
    return {
        "bas_dd": bas_dd, "open": price, "high": price, "low": price,
        "close": close, "change": change,
        "change_rate": round(change / (close - change) * 100, 2) if close != change else 0.0,
        "volume": 0 if halted else volume, "listed_shares": shares,
    }


def _플래그(rows, *, still_listed: bool, collect_start: str = CAL[0]):
    return ca.flag_series(rows, calendar_index=INDEX, market_last_index=LAST,
                          still_listed=still_listed, collect_start=collect_start)


# ── 가격제한폭 ──────────────────────────────────────────────────────────────

def test_가격제한폭은_2015년_6월_15일에_바뀐다():
    """전 구간에 30.5% 를 쓰면 2015년 이전이 사각지대가 된다.

    그 구간의 상한은 ±15% 였다. 날짜별로 가르면 극단이 1,576 → 2,080행으로 늘고,
    늘어난 504행도 전부 아래 플래그로 설명된다(실측 2026-08-29).
    """
    assert ca.price_limit_pct("20150612") == ca.LIMIT_BEFORE_2015
    assert ca.price_limit_pct("20150615") == ca.LIMIT_AFTER_2015
    assert ca.price_limit_pct("20260825") == ca.LIMIT_AFTER_2015


def test_이상치_판정이_날짜에_따라_달라진다():
    """같은 +20% 라도 2014년엔 불가능한 값이고 2016년엔 평범한 값이다."""
    assert ca.is_outlier({"bas_dd": "20140102", "change_rate": 20.0})
    assert not ca.is_outlier({"bas_dd": "20160102", "change_rate": 20.0})


# ── 정지·체결 판정 ──────────────────────────────────────────────────────────

def test_정지행은_체결이_아니다():
    """정지 중 행은 시·고·저가가 0 이고 종가만 직전 값을 물고 있다."""
    정지 = _행("20240102", close=1000, halted=True)
    거래 = _행("20240102", close=1000)

    assert ca.is_halted(정지) and not ca.is_traded(정지)
    assert not ca.is_halted(거래) and ca.is_traded(거래)


# ── A. 정리매매 ─────────────────────────────────────────────────────────────

def test_상장중_종목은_정리매매로_잡지_않는다():
    """마지막 거래일까지 멀쩡히 체결되는 종목의 최근 10일을 자르면
    그건 정리매매가 아니라 **멀쩡한 자료를 버리는 것**이다."""
    rows = [_행(d) for d in CAL]

    flags = _플래그(rows, still_listed=True)

    assert not any(f.liquidation for f in flags)


def test_소멸_종목의_마지막_체결일들이_정리매매다():
    """마지막 거래일에 행이 없으면 소멸이다. 진행 중인 정리매매도 여기 걸린다.

    실측: 시스웍(269620)은 2026-08-24 에 889원→1원으로 빠지고 다음 날 표에서
    사라졌다. '20거래일 이상 빈다' 만 보면 아직 시간이 흐르지 않아 놓친다.
    """
    rows = [_행(d) for d in CAL[:5]]          # 5일만 살고 사라진 종목

    flags = _플래그(rows, still_listed=False)

    assert all(f.liquidation for f in flags)


def test_이력_중간의_정리매매도_잡는다():
    """상장폐지 절차를 밟다 거래가 재개된 종목은 정리매매가 이력 한가운데에 있다.

    감마누(192410) → THQ → 휴림네트웍스 → 오늘이엔엠, 인포피아(036220) →
    오상헬스케어(8년 공백), 우양에이치씨(101970)가 실제로 그렇다.
    '소멸 종목의 마지막 N일' 규칙은 이력의 끝만 보므로 영원히 닿지 않는다.
    """
    # 앞 3일 체결 → 35거래일 사라짐(문턱 20 초과) → 마지막 2일에 다시 체결
    rows = [_행(d) for d in CAL[:3]] + [_행(d) for d in CAL[-2:]]

    flags = _플래그(rows, still_listed=True, collect_start=CAL[0])

    assert [f.liquidation for f in flags] == [True, True, True, False, False]


def test_정지가_행으로_표시돼도_정리매매를_잡는다():
    """거래정지가 **행 삭제가 아니라 zero-OHLC 행**으로 오는 경우가 있다.

    감마누는 그런 행이 656개다. 행의 존재만 보고 거리를 재면 달력이 안 비어서
    정리매매 5거래일이 통째로 빠진다. 그래서 **체결**을 이어 붙여 거리를 잰다.
    """
    rows = ([_행(d) for d in CAL[:3]]                       # 체결
            + [_행(d, halted=True) for d in CAL[3:28]]      # 정지행 25일 (문턱 20 초과)
            + [_행(d) for d in CAL[28:]])                   # 다시 체결

    flags = _플래그(rows, still_listed=True)

    # 정지 직전 3일이 정리매매로 잡혀야 한다 — 달력에는 구멍이 없다
    assert [f.liquidation for f in flags[:3]] == [True, True, True]
    # 재개 뒤는 아니다
    assert not any(f.liquidation for f in flags[28:])


# ── B. 거래정지 재개 ────────────────────────────────────────────────────────

def test_정지_직후_행에_재개_플래그가_붙는다():
    """정지 중 종가가 잔존하므로 재개일 등락률은 그 값을 기준으로 계산된다.

    에스와이코퍼레이션(008080)은 정지 중 종가가 1원이었고, 2013-09-11 재개일에
    +6,699,900% 가 찍혔다. 액면병합이 아니라 이 구조 때문이다.
    """
    rows = [_행(CAL[0]), _행(CAL[1], halted=True), _행(CAL[2], close=2000, change=1000)]

    flags = _플래그(rows, still_listed=True)

    assert [f.halt_resume for f in flags] == [False, False, True]


# ── C. 자본변동 ─────────────────────────────────────────────────────────────

def test_상장주식수가_바뀐_날에_자본변동_플래그가_붙는다():
    """액면병합·감자·증자의 **독립 근거**다.

    이게 없으면 '등락률이 크니까 액면병합' 이라는 순환논법이 된다. 실제로 그렇게
    불렀던 354행 중 진짜 자본변동은 21행(1.3%)뿐이었다.
    """
    rows = [_행(CAL[0], shares=100), _행(CAL[1], shares=100), _행(CAL[2], shares=10)]

    flags = _플래그(rows, still_listed=True)

    assert [f.capital_change for f in flags] == [False, False, True]


# ── D. 신규상장 첫날 ────────────────────────────────────────────────────────

def test_수집_시작일의_첫_행은_신규상장이_아니다():
    """우리 자료는 2010-01-04 부터인데 그 날 첫 행이 생기는 종목이 1,961개다.
    그건 상장이 아니라 **수집 경계**다. 실제로 그 1,961개엔 극단이 0건이다."""
    rows = [_행(d) for d in CAL]

    flags = _플래그(rows, still_listed=True, collect_start=CAL[0])

    assert not any(f.first_listing for f in flags)


def test_수집_시작_이후_첫_행은_신규상장이다():
    """그 행의 등락률은 전일종가가 아니라 **공모가** 기준이다.

    KRX 는 2023-06-26 부터 신규상장일 가격범위를 공모가의 60~400% 로 넓혔다.
    실측: 첫날 극단 155행이 전부 그 날 이후고 이전은 0행. 최대가 정확히 +300.0%,
    최소가 정확히 -40.0% 다.
    """
    rows = [_행(d) for d in CAL[3:]]

    flags = _플래그(rows, still_listed=True, collect_start=CAL[0])

    assert flags[0].first_listing
    assert not any(f.first_listing for f in flags[1:])


# ── 설명 여부 ───────────────────────────────────────────────────────────────

def test_플래그가_하나도_없으면_설명되지_않은_것이다():
    """게이트는 바로 이 경우에만 빨간불을 켠다."""
    assert not ca.RowFlags().explained
    assert ca.RowFlags(liquidation=True).explained
    assert ca.RowFlags(first_listing=True).names() == ("first_listing",)
    assert ca.RowFlags(liquidation=True, halt_resume=True).names() == (
        "liquidation", "halt_resume")



# ── 끊긴 가격 이어 붙이기 ───────────────────────────────────────────────────
#
# 여기서 틀려도 **에러가 안 난다.** 계수를 놓치면 삼성전자 2018-05-04 가 -98% 짜리
# 수익률로 학습에 들어가고, 반대로 없는 조정을 만들면 멀쩡한 가격이 조용히 망가진다.
# 후자가 더 크게 틀린다 — 실측에서 재개 단일가 기준가를 그대로 믿으면 어떤 종목은
# 과거 전체가 **120배** 부풀었다(111610, 주식수는 그대로였다).
#
# 🔴 못박아 두는 것: **액면분할·병합은 전부 거래정지 뒤 재개일에 있다.**
#    KRX 가 주권 교체 때문에 반드시 정지시키기 때문이다. 실측에서 정지일을 빼고 세면
#    분할이 0건 남는다. 그래서 "재개일은 다 버린다" 는 규칙은 모든 분할을 같이 버린다.


def _분할행(bas_dd: str, close: int, change: int, shares: int) -> dict:
    """기준가를 직접 지정하는 행. `기준가 = close - change` 다."""
    기준가 = close - change
    return {
        "bas_dd": bas_dd, "open": close, "high": close, "low": close,
        "close": close, "change": change,
        "change_rate": round(change / 기준가 * 100, 2) if 기준가 else 0.0,
        "volume": 1000, "listed_shares": shares,
    }


# ── 평상일: 기준가로 잰다 ───────────────────────────────────────────────────

def test_조정이_없으면_계수가_1이다():
    """평범한 날은 기준가가 전일종가와 같다. 건드리면 안 된다."""
    앞 = _행(CAL[0], close=1000)
    뒤 = _행(CAL[1], close=1100, change=100)      # 기준가 1000 = 앞 종가

    assert ca.adjustment_factor(앞, 뒤) == 1


def test_권리락은_주식수가_안_변해도_기준가로_잡힌다():
    """🔴 listed_shares 로만 재면 이걸 통째로 놓친다. 실측 3,730건이 이 모양이다.

    유한양행 2020-12-29 가 그렇다 — 전일종가 76,600 인데 기준가가 73,300 이다.
    종가 비율로는 -3.52% 지만 실제 등락률은 +0.82% 다.
    """
    앞 = _행(CAL[0], close=76_600, shares=100)
    뒤 = _분할행(CAL[1], close=73_900, change=600, shares=100)   # 기준가 73,300

    assert ca.adjustment_factor(앞, 뒤) == Fraction(733, 766)


def test_계수는_유리수라_반올림_오차가_없다():
    """1/50 을 float 로 누적하면 잡음이 다시 들어온다. Fraction 으로 받는다."""
    앞 = _행(CAL[0], close=2_650_000)
    뒤 = _분할행(CAL[1], close=51_900, change=-1_100, shares=100)

    계수 = ca.adjustment_factor(앞, 뒤)

    assert isinstance(계수, Fraction)
    assert 계수 == Fraction(1, 50)


# ── 재개일: 기준가를 믿지 않고 주식수로 잰다 ────────────────────────────────

def test_정지_뒤_재개일의_분할은_주식수배율로_잡는다():
    """삼성전자 2018-05-04 의 실제 모양 — 정지 3일 뒤 재개일에 50:1."""
    rows = [
        _행(CAL[0], close=2_650_000, shares=128_386_494),
        _행(CAL[1], close=2_650_000, shares=128_386_494, halted=True),
        _행(CAL[2], close=2_650_000, shares=128_386_494, halted=True),
        _분할행(CAL[3], close=51_900, change=-1_100, shares=6_419_324_700),
    ]

    factors = ca.factor_series(rows)

    assert factors[3] == Fraction(1, 50)


def test_재개일에_주식수가_그대로면_조정하지_않는다():
    """🔴 이걸 놓치면 과거 전체가 120배 부풀어난다 (111610 20150817 실측).

    정지 중 종가는 직전 값을 붙들고 있고, 재개일에 KRX 는 단일가로 기준가를 새로
    잡는다. 그 기준가는 자본변동이 아니다.
    """
    rows = [
        _행(CAL[0], close=145, shares=1000),
        _행(CAL[1], close=145, shares=1000, halted=True),
        _분할행(CAL[2], close=17_400, change=0, shares=1000),   # 기준가 17,400 = ×120
    ]

    factors = ca.factor_series(rows)

    assert factors[2] == 1              # 주식수가 안 변했으니 조정이 아니다


def test_재개일_주식수_변동이_미미하면_조정하지_않는다():
    """1.5배 문턱 아래는 자본변동으로 치지 않는다. 실측에서 잡음은 1.1배 아래였다."""
    rows = [
        _행(CAL[0], close=1000, shares=1000),
        _행(CAL[1], close=1000, shares=1000, halted=True),
        _분할행(CAL[2], close=5000, change=0, shares=1200),      # 1.2배 — 문턱 아래
    ]

    assert ca.factor_series(rows)[2] == 1


def test_재개일_액면병합도_주식수배율로_잡는다():
    """주식수가 줄면 계수가 1보다 크다 (병합·감자)."""
    rows = [
        _행(CAL[0], close=1000, shares=5000),
        _행(CAL[1], close=1000, shares=5000, halted=True),
        _분할행(CAL[2], close=5100, change=100, shares=1000),    # 1/5 로 감소
    ]

    assert ca.factor_series(rows)[2] == 5


# ── 배열과 구간 ─────────────────────────────────────────────────────────────

def test_앞_행이_없으면_조정으로_치지_않는다():
    """모르는 것을 조정으로 만들면 첫 행부터 가격이 망가진다."""
    assert ca.adjustment_factor(None, _행(CAL[0], close=1000)) == 1


def test_계수배열은_행과_길이가_같고_첫_행은_1이다():
    rows = [_행(d, close=1000) for d in CAL[:5]]

    factors = ca.factor_series(rows)

    assert len(factors) == len(rows)
    assert factors[0] == 1


def test_구간에_조정이_없으면_배율이_1이다():
    rows = [_행(d, close=1000) for d in CAL[:6]]

    assert ca.span_factor(ca.factor_series(rows), 0, 5) == 1.0


def test_구간_안의_분할만_배율에_들어간다():
    """진입 다음 날부터 청산일까지만 센다."""
    rows = [
        _행(CAL[0], close=2_650_000, shares=100),
        _행(CAL[1], close=2_650_000, shares=100, halted=True),
        _분할행(CAL[2], close=51_900, change=-1_100, shares=5_000),
        _행(CAL[3], close=52_600, change=700, shares=5_000),
    ]

    factors = ca.factor_series(rows)

    assert ca.span_factor(factors, 0, 3) == 0.02      # 분할을 건너뛴 구간
    assert ca.span_factor(factors, 2, 3) == 1.0       # 분할 뒤만 — 이미 조정된 스케일


def test_진입일_당일의_조정은_세지_않는다():
    """그 날 시가는 이미 조정된 기준으로 붙는다. 세면 배로 틀린다."""
    rows = [
        _행(CAL[0], close=1_000, shares=100),
        _행(CAL[1], close=1_000, shares=100, halted=True),
        _분할행(CAL[2], close=100, change=0, shares=1000),       # 10배 → 계수 1/10
        _행(CAL[3], close=110, change=10, shares=1000),
    ]

    factors = ca.factor_series(rows)

    assert factors[2] == Fraction(1, 10)
    assert ca.span_factor(factors, 2, 3) == 1.0       # 진입일 당일 조정은 빠진다


def test_구간이_뒤집히거나_같으면_1이다():
    factors = ca.factor_series([_행(d, close=1000) for d in CAL[:4]])

    assert ca.span_factor(factors, 2, 2) == 1.0
    assert ca.span_factor(factors, 3, 1) == 1.0


# ── 후방조정 계열 ───────────────────────────────────────────────────────────

def test_수정주가의_마지막은_원종가와_같다():
    """후방조정이므로 현재가 기준점이다."""
    rows = [
        _행(CAL[0], close=2_650_000, shares=100),
        _행(CAL[1], close=2_650_000, shares=100, halted=True),
        _분할행(CAL[2], close=51_900, change=-1_100, shares=5_000),
    ]

    adj = ca.back_adjusted_closes(rows)

    assert adj[-1] == 51_900
    assert adj[0] == 2_650_000 * 0.02                 # 53,000


def test_수정주가로_계산한_수익률이_등락률과_맞는다():
    """🔴 이게 이 함수의 존재 이유다. 원종가로는 -98% 가 나온다."""
    rows = [
        _행(CAL[0], close=2_650_000, shares=100),
        _행(CAL[1], close=2_650_000, shares=100, halted=True),
        _분할행(CAL[2], close=51_900, change=-1_100, shares=5_000),
    ]

    adj = ca.back_adjusted_closes(rows)
    raw = rows[2]["close"] / rows[1]["close"] - 1
    fixed = adj[2] / adj[1] - 1

    assert round(raw * 100, 2) == -98.04              # 고치기 전
    assert round(fixed * 100, 2) == rows[2]["change_rate"] == -2.08
