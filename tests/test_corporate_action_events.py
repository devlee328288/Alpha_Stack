"""기업행위 이벤트 판정 시험 (`corporate_action` · 스키마 v16)

**왜 이 시험이 필요한가.** 여기서 틀리면 **에러가 안 난다.** 사건을 놓치면 표가 조금
작아질 뿐이고, 없는 사건을 만들면 배율이 하나 더 곱해질 뿐이다. 둘 다 조용히 지나간다.

특히 못 박아 두는 것 넷 — 전부 실제 자료에서 한 번씩 물었던 자리다.

1. **액면가만으로 가격 배율을 정하면 42건을 10배 망친다.** 액면가가 줄어도 주식수가
   그대로면 결손금을 턴 것(액면가 감액)이라 가격이 안 움직인다. 자본금을 함께 봐야
   순수 분할·병합과 갈린다.
2. **액면분할은 전부 재개일에 있다.** 주권 교체 때문에 KRX 가 반드시 거래를 정지시킨다.
   재개일이라는 이유로 기준가에 판정을 넘기면, KRX 가 기준가를 안 고친 자리에서
   사건이 통째로 사라진다 — 그러면 어긋남을 셀 수가 없다.
3. **한 거래일에 행은 하나다.** 배율이 두 줄이면 쓰는 쪽이 곱한다.
4. **`series_restart` 는 대조하지 않는다.** `factor_series` 도 같은 `is_series_restart`
   로 끊으므로 재면 늘 초록이 나온다 — 검사가 대상과 같은 판정을 공유하는 자리다.
"""

from __future__ import annotations

from datetime import date, timedelta
from fractions import Fraction

from common import corporate_actions as ca
from ingest.store.action_store import build_events


def _평일들(n: int, 시작: date = date(2024, 1, 2)) -> list:
    """2024-01-02(화)부터 평일 `n`개. 손으로 적으면 주말이 섞인다."""
    out, day = [], 시작
    while len(out) < n:
        if day.weekday() < 5:
            out.append(day.strftime("%Y%m%d"))
        day += timedelta(days=1)
    return out


CAL = _평일들(40)
INDEX = {d: i for i, d in enumerate(CAL)}
LAST = len(CAL) - 1


def _행(bas_dd: str, *, close: int = 1000, change: int = 0, shares: int = 100,
        halted: bool = False, code: str = "005930") -> dict:
    """`daily_price` 한 행. `halted=True` 면 KRX 가 정지 중에 주는 모양(OHL 이 0).

    ⚠️ `change_rate` 를 **계산해서** 넣는다. 손으로 적으면 `change` 와 갈라지고,
       그러면 `source_conflict`(등락률이 가격제한폭 밖인가)가 시험대에서만 다른
       답을 낸다 — 실제 표에서는 KRX 가 둘을 함께 준다.
    """
    price = 0 if halted else close
    앞종가 = close - change
    return {"code": code, "bas_dd": bas_dd, "open": price, "high": price,
            "low": price, "close": close, "change": change,
            "change_rate": round(change / 앞종가 * 100, 2) if 앞종가 else 0.0,
            "volume": 0 if halted else 1000, "listed_shares": shares}


def _기본(bas_dd: str, *, parval="500", list_dd="20200101",
          code: str = "005930") -> dict:
    """`stock_base_info` 한 행. 액면가는 **글자**다 — '무액면' 이 실재한다."""
    return {"code": code, "bas_dd": bas_dd, "parval": parval, "list_dd": list_dd}


def _판정(prev_row, row, prev_base, base, *, restart=False):
    return ca.classify_event(prev_row, row, prev_base, base, restart=restart)


# ==================================================
# 갈래마다 하나씩
# ==================================================
def test_자본금이_그대로면_액면분할이고_배율은_액면가비다():
    """액면가 500 → 100 이고 주식수가 5배면 순수 5:1 분할. 가격은 1/5 이 된다.

    배율을 **액면가**에서 얻는 까닭: 주식수 비율은 단수주 때문에 정수비가 아니다.
    실제 자료의 000040 은 ×5 여야 할 주식수가 ×4.99999977 이다(4주 차이).
    """
    prev = _행(CAL[0], close=5000, shares=100, halted=True)
    row = _행(CAL[1], close=1000, change=0, shares=500)
    ev = _판정(prev, row, _기본(CAL[0]), _기본(CAL[1], parval="100"))
    assert ev is not None
    assert ev.event_type == "split_merge"
    assert ev.ratio == Fraction(1, 5)
    assert ev.source == "par"
    assert (ev.par_before, ev.par_after) == ("500", "100")
    assert (ev.shares_before, ev.shares_after) == (100, 500)


def test_주식수가_그대로면_액면가_감액이라_배율이_없다():
    """액면가만 줄고 주식수가 안 바뀌면 결손금을 턴 것이다. **가격이 안 움직인다.**

    실제 자료에서 42건이고, 그 42건 전부에서 `adj_price` 도 조정하지 않았다.
    여기에 액면가 비율(1/10)을 적용하면 멀쩡한 과거를 10배 망친다.
    """
    prev = _행(CAL[0], close=1000, shares=100)
    row = _행(CAL[1], close=1000, change=0, shares=100)
    ev = _판정(prev, row, _기본(CAL[0], parval="5000"), _기본(CAL[1], parval="500"))
    assert ev is not None
    assert ev.event_type == "par_reduction"
    assert ev.ratio is None
    assert ev.source == "par"


def test_이전상장은_가격이_이어지므로_배율이_없다():
    """상장일이 바뀌었는데 거래일 공백이 없으면 시장만 옮긴 것이다(코스닥→유가 22건)."""
    prev = _행(CAL[0])
    row = _행(CAL[1])
    ev = _판정(prev, row, _기본(CAL[0], list_dd="20200101"),
             _기본(CAL[1], list_dd=CAL[1]))
    assert ev is not None
    assert ev.event_type == "market_transfer"
    assert ev.ratio is None
    assert ev.source == "listing"


def test_시계열_재시작은_조정이_아니라_단절이다():
    """공백 뒤에 새 상장일이 생기면 **다른 회사**다. 이어 붙일 것이 없다."""
    prev = _행(CAL[0], close=830)
    row = _행(CAL[10], close=18640, change=0)
    ev = _판정(prev, row, _기본(CAL[0]), _기본(CAL[10], list_dd=CAL[10]), restart=True)
    assert ev is not None
    assert ev.event_type == "series_restart"
    assert ev.ratio is None
    assert ev.source == "listing"


def test_평상일에_기준가만_끊기면_권리락이다():
    """주식수가 안 바뀌어도 기준가가 바뀌는 사건이 있다 — 주식배당 권리락(실측 3,730일)."""
    prev = _행(CAL[0], close=1000, shares=100)
    # 기준가 900 = 종가 990 - 전일대비 90. 전일종가 1,000 과 다르다.
    row = _행(CAL[1], close=990, change=90, shares=100)
    ev = _판정(prev, row, _기본(CAL[0]), _기본(CAL[1]))
    assert ev is not None
    assert ev.event_type == "rights_off"
    assert ev.ratio == Fraction(9, 10)
    assert ev.source == "basis"


def test_정지_뒤_재개일에_기준가가_끊기면_재평가다():
    """직전이 거래정지면 KRX 가 단일가로 기준가를 새로 잡는다."""
    prev = _행(CAL[0], close=1000, shares=100, halted=True)
    row = _행(CAL[1], close=560, change=60, shares=140)   # 기준가 500 · 주식수 1.4배
    ev = _판정(prev, row, _기본(CAL[0]), _기본(CAL[1]))
    assert ev is not None
    assert ev.event_type == "resume_revalue"
    assert ev.source == "basis"


def test_주식수만_늘고_가격이_연속이면_유상증자다():
    """실측 39,515건. 배율은 없지만 **희석이 그 자체로 신호**라 담는다."""
    prev = _행(CAL[0], close=1000, shares=100)
    row = _행(CAL[1], close=1000, change=0, shares=150)
    ev = _판정(prev, row, _기본(CAL[0]), _기본(CAL[1]))
    assert ev is not None
    assert ev.event_type == "share_change"
    assert ev.ratio is None
    assert ev.source == "shares"


def test_아무_일도_없으면_사건이_아니다():
    prev = _행(CAL[0], close=1000, shares=100)
    row = _행(CAL[1], close=1050, change=50, shares=100)
    assert _판정(prev, row, _기본(CAL[0]), _기본(CAL[1])) is None


# ==================================================
# 액면가는 숫자가 아닐 수 있다
# ==================================================
def test_무액면은_배율로_만들지_않는다():
    """`'무액면'` 73,713행 · 주식예탁증권의 `'0'` 7,616행이 실재한다.

    못 읽으면 액면가 갈래를 포기하고 기준가로 넘어간다 — **모르는 것을 배율로
    만들지 않는다.** 여기서는 기준가도 연속이므로 주식수 변동만 남는다.
    """
    prev = _행(CAL[0], close=1000, shares=100)
    row = _행(CAL[1], close=1000, change=0, shares=500)
    ev = _판정(prev, row, _기본(CAL[0], parval="무액면"), _기본(CAL[1], parval="0"))
    assert ev is not None
    assert ev.event_type == "share_change"
    assert ev.ratio is None


def test_기본정보가_없는_날은_액면가_갈래를_안_탄다():
    """시세에만 있고 종목기본정보에 없는 날이 2,765행 있다. 그 날은 기준가로만 본다."""
    prev = _행(CAL[0], close=1000, shares=100)
    row = _행(CAL[1], close=1000, change=0, shares=500)
    ev = _판정(prev, row, None, None)
    assert ev is not None
    assert ev.event_type == "share_change"
    assert ev.par_before is None and ev.par_after is None


# ==================================================
# 대조 — `agrees`
# ==================================================
def test_배율이_있으면_chain_과_정확히_같아야_맞음이다():
    ev = ca.CorporateEvent(code="A", ex_date=CAL[1], event_type="split_merge",
                           ratio=Fraction(1, 5), source="par", par_before="500",
                           par_after="100", shares_before=100, shares_after=500,
                           basis_ratio=0.2)
    assert ca.event_agrees(ev, Fraction(1, 5)) is True
    assert ca.event_agrees(ev, Fraction(1, 25)) is False
    assert ca.event_agrees(ev, None) is False


def test_가격이_연속인_사건은_chain_이_없어야_맞음이다():
    """액면가 감액·이전상장은 "안 움직인다" 를 예고한다. 그것도 대조할 수 있는 예고다."""
    ev = ca.CorporateEvent(code="A", ex_date=CAL[1], event_type="par_reduction",
                           ratio=None, source="par", par_before="5000",
                           par_after="500", shares_before=100, shares_after=100,
                           basis_ratio=1.0)
    assert ca.event_agrees(ev, None) is True
    assert ca.event_agrees(ev, Fraction(1)) is True
    assert ca.event_agrees(ev, Fraction(1, 10)) is False


def test_재시작과_주식수변동은_대조하지_않는다():
    """`series_restart` 를 재면 `factor_series` 와 같은 판정을 공유해 늘 초록이 나온다.

    `share_change` 는 가격 배율에 대해 아무 예고도 하지 않는다. 둘 다 `None` 이다 —
    **모르는 것을 어긋남으로도 맞음으로도 치지 않는다.**
    """
    for t in ca.UNJUDGED_TYPES:
        ev = ca.CorporateEvent(code="A", ex_date=CAL[1], event_type=t, ratio=None,
                               source="listing", par_before=None, par_after=None,
                               shares_before=100, shares_after=100, basis_ratio=None)
        assert ca.event_agrees(ev, None) is None
        assert ca.event_agrees(ev, Fraction(2)) is None


# ==================================================
# 시계열로 묶었을 때
# ==================================================
def _시리즈(rows, base_rows, *, listing=("20200101",)):
    return build_events(rows, base_rows, calendar_index=INDEX,
                        listing_days=listing,
                        liquidation=[False] * len(rows))


def test_한_거래일에_행은_하나다():
    """액면분할과 유상증자가 같은 날 나도 행은 하나다 — 배율이 두 줄이면 곱해진다."""
    rows = [_행(CAL[0], close=5000, shares=100, halted=True),
            _행(CAL[1], close=1000, change=0, shares=500),
            _행(CAL[2], close=1010, change=10, shares=500)]
    base = [_기본(CAL[0]), _기본(CAL[1], parval="100"), _기본(CAL[2], parval="100")]
    사건 = _시리즈(rows, base)
    날짜 = [e.ex_date for e, *_ in 사건]
    assert len(날짜) == len(set(날짜))
    assert 날짜 == [CAL[1]]


def test_첫_행은_사건이_아니다():
    """앞이 없으면 견줄 것이 없다. 조정을 놓치면 값이 틀리지만 없는 조정을 만들면 망친다."""
    rows = [_행(CAL[0], close=1000, shares=100)]
    assert _시리즈(rows, [_기본(CAL[0])]) == []


def test_정리매매_재개일의_액면병합이_어긋남으로_남는다():
    """실제로 걸린 15건의 꼴 — 084810 아이알디 2010-03-23 을 줄여 옮긴 것이다.

    10:1 액면병합인데(주식수 1/10 · 액면가 10배 · 자본금 불변) KRX 가 기준가를
    안 고쳐 기준가비가 정확히 1 이다. `adjustment_factor` 는 재개일 밴드
    (`RESUME_BASIS_BAND`) 밖이라 기준가비를 믿고 **조정을 놓친다.**

    표는 그것을 고치지 않고 **어긋남으로 남긴다** — 정리매매 구간은 가격제한폭이
    없어 원본 오류가 아니고, `is_liquidation` 이 이미 덮어 학습에 안 들어간다.
    """
    rows = [_행(CAL[0], close=45, shares=1000, halted=True),
            _행(CAL[1], close=15, change=-30, shares=100)]      # 기준가 45 = 전일종가
    base = [_기본(CAL[0], parval="100"), _기본(CAL[1], parval="1000")]
    사건 = build_events(rows, base, calendar_index=INDEX, listing_days=("20200101",),
                      liquidation=[True, True])
    assert len(사건) == 1
    ev, chain, agrees, 모순, 정리매매 = 사건[0]
    assert ev.event_type == "split_merge"
    assert ev.ratio == Fraction(10)          # 액면가 100 → 1,000 이니 가격은 10배
    assert chain is None                     # chain 은 조정이 없다고 봤다
    assert agrees is False
    assert 정리매매 is True


def test_조정이_없으면_chain_을_1로_적지_않는다():
    """`factor_series` 의 1 은 "조정 없음" 이다. 배율로 적으면 쓰는 쪽이 곱한다."""
    rows = [_행(CAL[0], close=1000, shares=100),
            _행(CAL[1], close=1000, change=0, shares=150)]
    사건 = _시리즈(rows, [_기본(CAL[0]), _기본(CAL[1])])
    assert len(사건) == 1
    assert 사건[0][1] is None


def test_사건_종류는_표의_CHECK_와_같아야_한다():
    """`EVENT_TYPES` 와 마이그레이션 CHECK 가 갈리면 넣을 때 터진다 — 여기서 먼저 잡는다."""
    from ingest.store.migrations import MIGRATIONS

    sql = "\n".join(s for s in MIGRATIONS[15][1] if isinstance(s, str))
    for t in ca.EVENT_TYPES:
        assert f"'{t}'" in sql, f"{t} 가 CHECK 에 없다"


# ==================================================
# 문턱 — 단수주와 진짜 어긋남
# ==================================================
def test_단수주로_생긴_끝자리_차이는_어긋남이_아니다():
    """chain 은 재개일에 **주식수 배율**을 쓰는데 그 비율은 정수가 아니다.

    417180 은 5:1 병합인데 주식수가 정확히 5배가 아니다(4주 차이) — 액면가 비율은
    정확히 `1/5` 이고 chain 은 `0.1999999658` 이다. 같은 사건을 두 축이 마지막
    자리까지 다르게 부르는 것이라, 여기서 붉은불을 켜면 662건이 통째로 붉어진다.
    """
    ev = ca.CorporateEvent(code="417180", ex_date=CAL[1], event_type="split_merge",
                           ratio=Fraction(1, 5), source="par", par_before="100",
                           par_after="500", shares_before=100000000,
                           shares_after=19999996, basis_ratio=0.2)
    단수주 = Fraction(19999996, 100000000)      # 0.1999999600
    assert ca.event_agrees(ev, 단수주) is True


def test_문턱_밖은_어긋남이다():
    """빈 구간 위쪽 — 221610 은 액면가가 1/5 을 예고했는데 chain 이 1/25 를 썼다."""
    ev = ca.CorporateEvent(code="221610", ex_date=CAL[1], event_type="split_merge",
                           ratio=Fraction(1, 5), source="par", par_before="500",
                           par_after="100", shares_before=3993673,
                           shares_after=19968365, basis_ratio=0.04)
    assert ca.event_agrees(ev, Fraction(1, 25)) is False


def test_문턱은_빈_구간_안에_있다():
    """실측 분포의 빈 구간은 8.408e-06 ~ 7.457e-01 이다. 문턱이 그 밖으로 나가면
    662건이 붉어지거나(아래) 4건을 놓친다(위). 값을 옮길 때 이 시험이 막는다."""
    assert 8.408e-06 < ca.AGREE_TOLERANCE < 7.457e-01


# ==================================================
# 순환 — 재면 안 되는 자리
# ==================================================
def test_기준가에서_얻은_배율은_chain_과_대조하지_않는다():
    """`rights_off`·`resume_revalue` 는 배율을 `adjustment_factor` 에서 얻는다.

    그것을 다시 `factor_series` 와 대조하면 **검사가 대상과 같은 값을 쓴다.**
    실제로 3,974건이 통째로 "맞음" 으로 나왔었다 — 초록이 아무것도 뜻하지 않는다.
    """
    for t in ("rights_off", "resume_revalue"):
        assert t in ca.UNJUDGED_TYPES
        ev = ca.CorporateEvent(code="A", ex_date=CAL[1], event_type=t,
                               ratio=Fraction(9, 10), source="basis",
                               par_before="500", par_after="500",
                               shares_before=100, shares_after=100, basis_ratio=0.9)
        assert ca.event_agrees(ev, Fraction(9, 10)) is None
        assert ca.event_agrees(ev, Fraction(1, 2)) is None


def test_대조하는_것은_액면가와_상장일_축뿐이다():
    """chain 은 액면가와 상장일을 배율 계산에 안 쓴다 — 그래서 독립이다."""
    잴수있는 = [t for t in ca.EVENT_TYPES if t not in ca.UNJUDGED_TYPES]
    assert sorted(잴수있는) == ["market_transfer", "par_reduction", "split_merge"]


# ==================================================
# 자기모순 — chain 을 안 쓰고 재는 축
# ==================================================
def test_주식수가_크게_바뀌었는데_등락률이_폭_밖이면_자기모순이다():
    """제일바이오 2026-02-09 을 줄여 옮긴 것 — 주식수 ×1/1500 에 등락률 +29,948%.

    같은 원본이 "주식수가 1,500분의 1 이 됐는데 가격은 그 폭만큼 움직였다" 고
    말한다. chain 이 무엇을 골랐는지와 **무관하게** 잴 수 있다.
    """
    row = _행(CAL[1], close=625000, change=622920, shares=19419)
    assert abs(row["change_rate"]) > ca.price_limit_pct(CAL[1])
    assert ca.source_conflict(row, 29129064, 19419) is True


def test_주식수가_그대로면_등락률이_폭_밖이어도_자기모순이_아니다():
    """상·하한가를 벗어난 행은 정리매매·재개일에 흔하다. 그것만으로는 모순이 아니다 —
    두 칸이 **서로** 어긋나야 한다."""
    row = _행(CAL[1], close=500, change=-500, shares=100)
    assert ca.source_conflict(row, 100, 100) is False


def test_주식수가_크게_바뀌어도_등락률이_폭_안이면_자기모순이_아니다():
    """정상적인 감자·병합이다. KRX 가 기준가를 제대로 고쳤다는 뜻이다."""
    row = _행(CAL[1], close=1050, change=50, shares=20)
    assert abs(row["change_rate"]) <= ca.price_limit_pct(CAL[1])
    assert ca.source_conflict(row, 100, 20) is False


def test_주식수를_모르면_자기모순으로_치지_않는다():
    row = _행(CAL[1], close=625000, change=622920, shares=19419)
    assert ca.source_conflict(row, None, 19419) is False
    assert ca.source_conflict(row, 29129064, None) is False


def test_표의_칸_목록과_저장_튜플_길이가_같다():
    """칸을 더하고 튜플을 안 고치면 `INSERT` 가 조용히 어긋난다 — 여기서 먼저 잡는다."""
    from ingest.store.action_store import TABLE_COLUMNS, _to_row

    ev = ca.CorporateEvent(code="A", ex_date=CAL[1], event_type="split_merge",
                           ratio=Fraction(1, 5), source="par", par_before="500",
                           par_after="100", shares_before=100, shares_after=500,
                           basis_ratio=0.2)
    assert len(_to_row(ev, Fraction(1, 5), True, False, False, "now")) == len(TABLE_COLUMNS)
