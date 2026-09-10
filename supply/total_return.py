"""수정주가 옆에 **총수익 축**을 세운다 — 배당까지 재투자한 값.

우리가 가진 `adj_close` 는 액면분할·무상증자·주식배당까지만 편 값이라 **현금배당이
빠져 있다.** 학계 표준인 CRSP 가 `RET`(배당 포함)와 `RETX`(배당 제외)를 나눠 주는데,
FinanceDataReader 로 만든 우리 수정주가는 `RETX` 쪽이다. 그 차이는 실측된다 —
배당락일의 평균 일간수익률이 평소보다 **1.4665%p 낮고 17년이 전부 음수**다. 개별종목
중립대가 ±2% 이므로 5거래일 창에 배당락일이 들어오면 라벨이 조용히 아래로 밀린다.

⚠️ 이 숫자는 한 번 틀렸다. 배당 자료가 없던 동안은 *"연말 마지막 거래일 대비 위치"* 라는
대용치로 재서 **−0.455%p** 였는데, 실제 배당락일로 다시 재니 **3.2배**였다 — 연말만 봐서
분기배당을, 시총 상위 100 만 봐서 고배당 중소형주를 놓쳤기 때문이다. 대용치는 **크기를
줄이는 쪽으로 틀린다.**

그래서 `adj_close` 를 **그대로 두고 옆에 더한다.** 지우고 덮는 것이 아니다. 둘은
답하는 질문이 다르다.

    adj_close     "분할·병합을 편 주가가 얼마였나"      (가격 축)
    adj_close_tr  "배당까지 재투자하면 얼마가 됐나"     (성과 축)

🔴 **성과 축이지 피처가 아니다.** 배당 확정은 주주총회다. 그날 이전에는 금액을 알 수
   없으므로 `adj_close_tr` 이나 `adj_dividend` 를 입력 피처로 쓰면 미래참조가 된다.
   라벨·성과 평가에만 쓴다.

## 계산

    adj_dividend_t = 배당금_t × (adj_close_t / close_t)
    r_t            = (adj_close_t + adj_dividend_t) / adj_close_{t-1} − 1
    adj_close_tr_t = adj_close_tr_{t-1} × (1 + r_t)

첫 거래일은 `adj_close_tr = adj_close` 로 두어 두 축이 같은 자리에서 출발한다.

🔴 **배당도 가격과 같은 배율로 조정해야 한다.** 배당금은 그날의 원(貨) 단위 절대금액이고
   `adj_close` 는 분할 배율이 곱해진 값이라, 원금액을 그대로 더하면 축이 어긋난다.
   조인되는 20,248행 중 `adj_close/close ≠ 1` 인 행이 6,245행(30.8%)이다.

🔴 **조인 축은 배당락일(`ex_date`)이지 기준일이 아니다.** 기준일은 12월 31일 같은
   휴장일이 대부분이다(12월 기준일 중 거래일은 0.3%). `ex_date` 가 있는 43,464행은
   전부 거래일이고 `daily_price` 에도 있다.
"""

from __future__ import annotations

import sqlite3
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

__all__ = [
    "CASH_DIVIDEND_KINDS",
    "STANDARD_PAR_PRICES",
    "PAR_TOLERANCE",
    "codes_with_dividends",
    "dividend_amount",
    "dividends_by_code",
    "implied_par_price",
    "is_par_consistent",
    "is_suspect",
    "load_dividends",
    "total_return_series",
]

#: 현금이 실제로 나가는 배당구분. 배당락을 만드는 것은 이 둘뿐이다.
#: `주식배당`은 FinanceDataReader 가 이미 수정주가로 펴므로 여기서 또 더하면 두 번 센다.
#: 실제로 쌍용씨앤이 5건이 같은 배당락일에 현금·주식 두 줄로 들어와 있는데, 현금만
#: 고르면 그 겹침이 저절로 풀린다 — 거른 뒤 `(code, ex_date)` 중복은 0 이다.
CASH_DIVIDEND_KINDS: Tuple[str, ...] = ("현금배당", "동시배당")

#: 우리 시장에서 실제로 쓰이는 액면가. 원본 배당금이 성한지 되짚는 데 쓴다.
STANDARD_PAR_PRICES: Tuple[float, ...] = (100.0, 200.0, 500.0, 1000.0, 2500.0, 5000.0)

#: 역산 액면가가 표준값에 이만큼 안쪽이면 "붙었다" 로 본다. 배당률이 소수점 몇 자리에서
#: 잘려 오기 때문에 정확히 떨어지지 않는다 (예: 미래에셋증권 0.73206%).
PAR_TOLERANCE = 0.01


def dividend_amount(genr: Optional[float], grdn: Optional[float]) -> float:
    """그 줄이 주는 주당 현금배당. 일반배당을 먼저 보고 없으면 차등배당을 본다.

    차등배당은 대주주가 덜 받는 제도라 일반주주 몫이 크다 — 둘 다 0보다 큰 613행에서
    일반이 더 큰 것이 608행(99.2%)이다. 우리는 소액주주 자리에서 보므로 일반배당이
    기본이고, 일반이 비어 있고 차등만 있는 7행에서 배당을 놓치지 않도록 뒤를 받친다.
    """
    if genr is not None and genr > 0:
        return float(genr)
    if grdn is not None and grdn > 0:
        return float(grdn)
    return 0.0


def implied_par_price(amount: float, cash_rate: Optional[float]) -> Optional[float]:
    """배당금과 배당률로 **그날의 액면가**를 되짚는다. 되짚을 수 없으면 `None`.

    `genr_cash_dvdn_rt` 는 시가배당률이 아니라 **액면가 대비 배당률(%)** 이다 —
    현대차 2,500원 = 액면 5,000 × 50%, SK하이닉스 375원 = 5,000 × 7.5%. 그래서 금액을
    그 비율로 나누면 액면가가 나온다.

    🔴 `dividend.par_price_at_load` 로는 이 일을 못 한다. 그 칸은 **적재 시점**의
       액면가라 삼성전자 1987년 배당 행에도 100원(2018년 분할 뒤 값)이 들어 있다.
       되짚은 값은 그 배당이 실제로 결정되던 때의 액면가다.
    """
    if not cash_rate or cash_rate <= 0 or amount <= 0:
        return None
    return amount / (cash_rate / 100.0)


def is_par_consistent(amount: float, cash_rate: Optional[float],
                      par_at_load: Optional[float] = None) -> Optional[bool]:
    """배당금과 배당률이 서로 맞는가. 되짚을 수 없으면 `None`.

    되짚은 액면가가 표준 액면가나 적재 액면가에 붙으면 원본이 제 안에서 앞뒤가 맞는다는
    뜻이므로 금액을 믿는다. 조인되는 20,248행 중 **99.29%(20,104행)** 가 여기 든다.

    붙지 않는 111행도 살펴보면 전부 정당했다 — 유전·인프라 펀드가 액면가 대신 좌당
    기준가로 배당률을 매기고(106행), 외국계는 액면이 0.25달러처럼 소수다(3행). 그래서
    이 판정은 "아니면 오류" 가 아니라 "맞으면 확실히 성하다" 로만 쓴다.
    """
    par = implied_par_price(amount, cash_rate)
    if par is None:
        return None
    for standard in STANDARD_PAR_PRICES:
        if abs(par - standard) <= PAR_TOLERANCE * standard:
            return True
    if par_at_load and par_at_load > 0:
        if abs(par - par_at_load) <= PAR_TOLERANCE * par_at_load:
            return True
    return False


def is_suspect(amount: float, cash_rate: Optional[float], close: Optional[float],
               par_at_load: Optional[float] = None) -> bool:
    """이 배당금이 원본의 단위 오류인가.

    🔴 **크기로 자르지 않는다.** 배당이 종가보다 큰 행 넷 중 셋이 정당한 청산분배금이었다 —
       유전펀드 청산(종가 28원에 1,670원 분배)·리츠 분배가 실제로 그렇게 크다. 극단값이
       데이터 오류일 때만 손대고 데이터 생성 과정에서 나온 것은 그대로 두는 것이
       표준이다. 크기로 자르면 그 셋이 함께 잘려 총수익이 아래로 편향된다.

    대신 **원본이 자기 정합적인가**를 먼저 묻는다. 금액과 배당률이 서로 맞으면 배당수익률이
    5,964%여도 믿는다. 되짚을 수 없을 때만 배당이 종가를 넘는지 본다 — 되짚기 불가인
    33행 안에서 윙입푸드홀딩스(132,722,717%)와 그다음(19.5%)의 간극이 680만 배라,
    임계를 어디에 두든 그 한 행만 걸린다.

    걸린 행도 **지우지 않는다.** `is_dividend_suspect` 로 표시만 하고 총수익 계산에서만
    배당을 0 으로 본다. 판정을 나중에 되짚을 수 있어야 하기 때문이다.
    """
    if amount <= 0:
        return False
    consistent = is_par_consistent(amount, cash_rate, par_at_load)
    if consistent is not None:
        return False                      # 되짚어 앞뒤가 맞거나, 맞지 않아도 정당한 갈래다
    return bool(close and close > 0 and amount > close)


#: 배당락일에 붙는 현금배당을 종목·날짜별로 뽑는다. 금액이 0 인 줄은 애초에 빼는데,
#: 이 출처는 **무배당과 "금액을 안 적었다" 를 0 하나로 뭉뚱그리기** 때문이다.
_DIVIDEND_SQL = f"""
SELECT code, ex_date, genr_dvdn_amt, grdn_dvdn_amt,
       genr_cash_dvdn_rt, par_price_at_load
FROM dividend
WHERE code IS NOT NULL
  AND ex_date IS NOT NULL
  AND dvdn_rcd_nm IN ({','.join('?' * len(CASH_DIVIDEND_KINDS))})
  AND (genr_dvdn_amt > 0 OR grdn_dvdn_amt > 0)
"""


def load_dividends(conn: sqlite3.Connection) -> Dict[Tuple[str, str], Dict]:
    """`(종목, 배당락일) -> {금액, 배당률, 적재액면}` 을 만든다.

    한 종목의 한 배당락일에 줄이 둘 이상이면 금액이 큰 쪽을 남긴다. 거르고 나면 실제로
    겹치는 것은 없지만(20,557행 = 20,557키), 규칙을 비워 두면 나중에 원본이 바뀌었을 때
    어느 줄이 이길지가 실행 순서에 달리게 된다.
    """
    표: Dict[Tuple[str, str], Dict] = {}
    for code, ex_date, genr, grdn, rate, par in conn.execute(
            _DIVIDEND_SQL, CASH_DIVIDEND_KINDS):
        금액 = dividend_amount(genr, grdn)
        if 금액 <= 0:
            continue
        키 = (code, ex_date)
        이전 = 표.get(키)
        if 이전 is None or 금액 > 이전["amount"]:
            표[키] = {"amount": 금액, "cash_rate": rate, "par_at_load": par}
    return 표


def total_return_series(
    rows: Sequence[Tuple[str, Optional[float], Optional[float]]],
    dividends: Dict[str, Dict],
) -> Iterator[Tuple[str, Optional[float], Optional[float], Optional[int]]]:
    """한 종목의 시계열에 총수익을 매긴다.

    `rows` 는 `(bas_dd, close, adj_close)` 를 **날짜 오름차순**으로. `dividends` 는 그
    종목의 `배당락일 -> {amount, cash_rate, par_at_load}`.

    돌려주는 것은 `(bas_dd, adj_dividend, adj_close_tr, is_dividend_suspect)` 다.
    배당이 없는 날의 `adj_dividend` 와 성하지 않은 날의 깃발은 `None` 으로 둔다 —
    923만 행에 0 을 채워 넣지 않으려는 것이고, "그날 배당이 없었다" 는 배당표에 줄이
    없다는 사실로 이미 말해진다.

    🔴 `adj_close` 가 없는 날은 총수익도 없다. 다만 **누적을 끊지는 않는다** — 다음
       유효일이 마지막 유효일에서 이어받는다. 지금 비어 있는 것은 수정주가를 아직 채우지
       않은 최근 3거래일(2026-09-02~09-04)뿐이라, 끊으면 그 뒤로 종목 전체가 사라진다.
    """
    직전_수정종가: Optional[float] = None
    누적: Optional[float] = None

    for bas_dd, close, adj_close in rows:
        정보 = dividends.get(bas_dd)
        금액 = float(정보["amount"]) if 정보 else 0.0
        의심 = bool(정보) and is_suspect(
            금액, 정보.get("cash_rate"), close, 정보.get("par_at_load"))

        if adj_close is None or close is None or close <= 0:
            # 값을 지어내지 않는다. 직전 상태는 그대로 두어 다음 유효일이 이어받는다.
            yield bas_dd, None, None, (1 if 의심 else None)
            continue

        # 배당금은 그날의 원 단위 절대금액이고 수정종가에는 분할 배율이 곱해져 있다.
        # 같은 배율을 태워야 두 값을 더할 수 있다.
        배율 = adj_close / close
        조정배당 = (금액 * 배율) if (금액 > 0 and not 의심) else None

        if 누적 is None or 직전_수정종가 is None or 직전_수정종가 <= 0:
            # 첫 유효일 — 두 축을 같은 자리에서 출발시킨다.
            누적 = float(adj_close)
        else:
            수익률 = (adj_close + (조정배당 or 0.0)) / 직전_수정종가 - 1.0
            누적 = 누적 * (1.0 + 수익률)

        직전_수정종가 = float(adj_close)
        yield bas_dd, 조정배당, 누적, (1 if 의심 else None)


def dividends_by_code(표: Dict[Tuple[str, str], Dict]) -> Dict[str, Dict[str, Dict]]:
    """`load_dividends()` 의 결과를 종목별로 나눈다 — 한 종목씩 훑을 때 쓴다."""
    나눔: Dict[str, Dict[str, Dict]] = {}
    for (code, ex_date), 값 in 표.items():
        나눔.setdefault(code, {})[ex_date] = 값
    return 나눔


def codes_with_dividends(표: Dict[Tuple[str, str], Dict]) -> List[str]:
    """배당이 한 번이라도 있는 종목. 재계산 범위를 좁힐 때 쓴다."""
    return sorted({code for code, _ in 표})
