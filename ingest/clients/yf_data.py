"""야후 파이낸스(yfinance) 호출 + 화면용 정규화 (외부 연동 계층)

`yfinance` 로 종목 하나의 **당일 가격 지표**와 **기간별 시세**를 받아, 화면과 스크립트가
그대로 쓸 수 있는 모양으로 바꿔 준다. 계산은 전부 여기서 하고, 화면(`static/pages/yf.html`)과
스크립트(`scripts/yf.py`)는 받은 값을 그리기만 한다.

yfinance 를 쓰면서 확인한 것 (2026-07 실제 호출)
------------------------------------------------
1. **잘못된 티커여도 예외가 나지 않는다.** `Ticker("ZZZZ999.KS").info` 는 예외 대신
   값이 전부 `None` 인 딕셔너리를 돌려준다. 그래서 "가격이 하나도 없으면 404" 로 직접 판단한다.
2. **현재가 키가 상황에 따라 다르다.** 장중에는 `currentPrice`, 장이 끝나면 없을 수 있어
   `regularMarketPrice` 를 함께 봐야 한다.
3. **`.info` 는 느리다.** 한 번에 1~3초가 걸리므로 같은 티커를 잠깐 사이에 다시 물으면
   메모리 캐시(`CACHE_TTL` 초)로 돌려준다. 화면에서 버튼을 연타해도 야후를 계속 두드리지 않는다.
4. 통화가 종목마다 다르다(`KRW`·`USD`·`JPY`…). 원본 예제처럼 "원"으로 고정하지 않고
   응답의 `currency` 를 그대로 실어 보낸다.

제공 함수
---------
| 함수 | 쓰임 |
|---|---|
| `fetch_quote` | 당일 5개 가격 지표(전일종가·시가·저가·고가·현재가) + 차트용 값 |
| `fetch_history` | 기간별 일봉 (캔들·거래량 차트용) |
| `price_bounds` | 가격 목록의 Y축 범위(±여백) — 스크립트와 화면이 같은 계산을 쓰게 한다 |
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import yfinance as yf

# 당일 가격 지표 5종. 이름과 순서를 여기서 정하고 화면·스크립트가 그대로 따라 쓴다.
# (원본 예제의 categories 와 같은 순서 — 전일종가 → 시가 → 저가 → 고가 → 현재가)
PRICE_FIELDS: Tuple[Tuple[str, str, str], ...] = (
    # (키, 한국어 라벨, 영어 라벨)
    ("prev_close", "전일종가", "Prev Close"),
    ("open", "시가", "Open"),
    ("day_low", "저가", "Low"),
    ("day_high", "고가", "High"),
    ("current", "현재가", "Current"),
)

# 기간별 시세에서 허용하는 조회 구간. yfinance 가 받는 값 중 실습에 쓸 만한 것만 남겼다.
#
# M2 에서 `1d`·`ytd`·`10y`·`max` 를 더했다. 시장 상세 화면(`/market`)이
# "단기부터 장기까지" 를 한 축에서 오가야 하는데, 5년이 상한이면 사이클을 볼 수 없어서다.
# (`1d` 는 일봉으로는 점이 하나뿐이라 분봉으로 받는다 — 아래 `INTRADAY_PERIODS` 참고)
PERIODS: Dict[str, str] = {
    "1d": "1일",
    "5d": "5일",
    "1mo": "1개월",
    "3mo": "3개월",
    "6mo": "6개월",
    "ytd": "연초 이후",
    "1y": "1년",
    "5y": "5년",
    "10y": "10년",
    "max": "전체",
}

# 일봉으로 받으면 점이 너무 적어 분봉으로 받는 구간. 값은 yfinance 의 `interval` 이다.
# 야후는 분봉 보관 기간이 짧다(1분봉 최대 7일). 그 한도 안에서만 쓴다.
INTRADAY_PERIODS: Dict[str, str] = {
    "1d": "5m",     # 하루를 5분 간격으로 → 78점 안팎
    "5d": "30m",    # 닷새를 30분 간격으로 → 65점 안팎
}

CACHE_TTL = 60                      # 같은 티커를 다시 물었을 때 캐시를 쓰는 시간(초)
Y_AXIS_MARGIN = 0.01                # Y축 위아래 여백 1% — 작은 가격 변화도 눈에 보이게 한다
KST = timezone(timedelta(hours=9))  # 표시 시각은 한국 시간 기준

# 티커별 마지막 응답을 담아 두는 메모리 캐시. {(종류, 티커, 옵션): (저장시각, 값)}
# 서버를 끄면 사라지는 임시 캐시라 DB 를 쓰지 않는다.
_cache: Dict[Tuple, Tuple[float, dict]] = {}
_cache_lock = threading.Lock()      # 여러 요청이 동시에 들어와도 캐시가 깨지지 않게 잠근다


class YahooError(Exception):
    """야후 파이낸스 조회 실패. `status` 는 라우터가 그대로 HTTP 상태 코드로 쓴다."""

    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


def _wrap_error(error: Exception, what: str) -> YahooError:
    """야후 호출 예외를 사용자에게 설명 가능한 오류로 바꾼다.

    **요청 한도 초과(429)를 따로 구분하는 이유** — yfinance 는 공식 API 가 아니라
    야후 웹 엔드포인트를 긁어 오는 라이브러리라, 같은 IP 에서 요청이 몰리면 야후가 막는다.
    특히 클라우드(서버리스) 배포는 **여러 사용자가 IP 를 공유**하므로 내가 조금만 호출해도
    이미 한도에 걸려 있을 수 있다. 이때 "조회 실패" 라고만 하면 원인을 알 수 없어서,
    한도 문제임을 분명히 알려 준다. (일시적이며 시간이 지나면 풀린다)
    """
    text = str(error)
    if "429" in text or "Too Many Requests" in text or "Rate limit" in text.lower():
        return YahooError(
            "야후 파이낸스 요청 한도(429)에 걸렸습니다. 잠시 후 다시 시도해 주세요. "
            "야후는 IP 단위로 한도를 두는데, 클라우드 배포는 IP 를 공유하기 때문에 "
            "직접 많이 호출하지 않아도 걸릴 수 있습니다.",
            status=429,
        )
    return YahooError(f"{what}: {error}", status=502)


# ==================================================
# 공통 도구
# ==================================================
def _now_kst() -> str:
    """현재 시각을 `2026-07-31 15:04:11 KST` 형태로 돌려준다."""
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")


def _number(value) -> Optional[float]:
    """야후가 준 값을 숫자로 바꾼다. 숫자가 아니면 `None` (결측으로 취급)."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    # NaN 은 자기 자신과 비교해도 다르다. JSON 으로 내보낼 수 없으므로 결측 처리한다.
    if number != number:
        return None
    return number


def _cached(key: Tuple, producer):
    """`CACHE_TTL` 초 안에 같은 요청이 오면 저장해 둔 값을 그대로 돌려준다.

    야후 호출이 1~3초씩 걸려서, 화면에서 조회 버튼을 연타하면 그만큼 기다려야 한다.
    잠깐 사이의 반복 호출은 캐시로 막는다.
    """
    now = time.monotonic()
    with _cache_lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < CACHE_TTL:
            return hit[1]

    # 호출 자체는 잠금 밖에서 한다 (느린 네트워크 호출이 다른 요청을 막지 않도록).
    value = producer()

    with _cache_lock:
        _cache[key] = (time.monotonic(), value)
    return value


def price_bounds(prices: List[Optional[float]], margin: float = Y_AXIS_MARGIN) -> Tuple[float, float]:
    """가격 목록에서 Y축 최소·최대를 계산한다 (위아래 `margin` 만큼 여백).

    0 부터 그리면 5개 값이 다 비슷해 보여 변화가 안 보인다.
    실제 값 구간만 확대해서 보여 주려고 범위를 좁힌다.
    """
    valid = [p for p in prices if p is not None and p > 0]
    if not valid:
        return (0.0, 1.0)
    low, high = min(valid), max(valid)
    if low == high:
        # 모든 값이 같으면(상한가 등) 폭이 0이 되어 축이 그려지지 않는다. 강제로 벌린다.
        return (low * (1 - margin), high * (1 + margin))
    return (low * (1 - margin), high * (1 + margin))


def normalize_ticker(ticker: str) -> str:
    """티커를 야후 표기로 다듬는다.

    - 앞뒤 공백 제거 · 대문자 (`005930.ks` → `005930.KS`)
    - 숫자 6자리만 주면 한국거래소 종목으로 보고 `.KS` 를 붙인다 (`005930` → `005930.KS`)
    """
    code = (ticker or "").strip().upper()
    if not code:
        raise YahooError("티커를 입력해 주세요. (예: 005930.KS · AAPL)", status=422)
    if code.isdigit() and len(code) == 6:
        return f"{code}.KS"
    return code


# ==================================================
# 1. 당일 가격 지표
# ==================================================
def fetch_quote(ticker: str) -> dict:
    """종목 하나의 **당일 가격 지표 5종**과 차트용 데이터를 돌려준다.

    돌려주는 `chart` 는 화면(ApexCharts)과 스크립트(matplotlib)가 똑같이 쓰는 형태다.
    막대와 꺾은선이 같은 값을 가리키므로 둘을 겹쳐 그리면 원본 예제와 같은 그림이 된다.
    """
    symbol = normalize_ticker(ticker)
    return _cached(("quote", symbol), lambda: _fetch_quote_uncached(symbol))


def _fetch_quote_uncached(symbol: str) -> dict:
    started = time.monotonic()
    try:
        info = yf.Ticker(symbol).info or {}
    except Exception as error:  # 네트워크 장애 · 요청 한도 · 야후 응답 형식 변경 등
        raise _wrap_error(error, "야후 파이낸스 조회에 실패했습니다") from error

    # 가격 5종을 뽑는다. 현재가는 장중/장마감에 따라 키가 달라서 두 곳을 함께 본다.
    prices = {
        "prev_close": _number(info.get("previousClose")),
        "open": _number(info.get("open") or info.get("regularMarketOpen")),
        "day_low": _number(info.get("dayLow") or info.get("regularMarketDayLow")),
        "day_high": _number(info.get("dayHigh") or info.get("regularMarketDayHigh")),
        "current": _number(info.get("currentPrice") or info.get("regularMarketPrice")),
    }

    # 잘못된 티커도 예외 없이 빈 값으로 오기 때문에, 여기서 직접 판단해 404 를 만든다.
    if not any(p for p in prices.values() if p and p > 0):
        raise YahooError(
            f"'{symbol}' 의 가격을 불러오지 못했습니다. "
            "티커가 잘못됐거나 야후에 해당 종목이 없을 수 있습니다. "
            "(국내 종목은 코스피 `.KS` · 코스닥 `.KQ` 를 붙입니다)",
            status=404,
        )

    values = [prices[key] for key, _, _ in PRICE_FIELDS]
    y_min, y_max = price_bounds(values)

    # 전일종가 대비 등락. 야후가 준 값이 있으면 그대로 쓰고, 없으면 직접 계산한다.
    diff = _number(info.get("regularMarketChange"))
    rate = _number(info.get("regularMarketChangePercent"))
    if diff is None and prices["current"] and prices["prev_close"]:
        diff = prices["current"] - prices["prev_close"]
    if rate is None and diff is not None and prices["prev_close"]:
        rate = diff / prices["prev_close"] * 100

    return {
        "ticker": symbol,
        "name": info.get("longName") or info.get("shortName") or symbol,
        "currency": info.get("currency") or "",
        "exchange": info.get("exchange") or "",
        "quote_type": info.get("quoteType") or "",
        "market_state": info.get("marketState") or "",     # REGULAR(장중) · CLOSED(장마감) 등
        "prices": prices,
        "change": {"diff": diff, "rate": rate},
        "volume": _number(info.get("volume") or info.get("regularMarketVolume")),
        "market_cap": _number(info.get("marketCap")),
        "week52_low": _number(info.get("fiftyTwoWeekLow")),
        "week52_high": _number(info.get("fiftyTwoWeekHigh")),
        "chart": {
            "categories": [ko for _, ko, _ in PRICE_FIELDS],
            "categories_en": [en for _, _, en in PRICE_FIELDS],
            "values": values,
            "y_min": y_min,
            "y_max": y_max,
        },
        "fetched_at": _now_kst(),
        "elapsed_ms": int((time.monotonic() - started) * 1000),
    }


# ==================================================
# 2. 기간별 시세 (일봉)
# ==================================================
# 한 번에 내려보내는 봉 상한 (ADR-DS-0004).
#
# `period="max"` 는 상장 이후 **전 구간**을 준다 — `^KS11` 은 1997년부터라 7,000행이 넘고
# `^DJI` 는 1992년부터다. 화면은 그렇게 촘촘한 봉을 그리지 못하고, Vercel 은 요청·응답
# 본문을 4.5MB 로 제한하며, 그 전에 `maxDuration: 60` 을 먼저 친다.
#
# 3,000 으로 잡은 이유 — **10년(약 2,470거래일)은 온전히 통과하고 `max` 만 잘린다.**
# 상한이 목록에 있는 기간을 잘라 버리면 기간 버튼이 거짓말을 하게 된다.
MAX_HISTORY_ROWS = 3000


def _limit_rows(payload: dict, max_rows: int) -> dict:
    """봉이 너무 많으면 **최근 것만** 남기고 그 사실을 밝힌다.

    `fred_data._limit_points` 와 같은 규칙이다. 앞을 자르는 이유는 시계열에서
    최근이 더 쓸모 있기 때문이고, 구간 수익률(`change_rate`)은 **남긴 구간 기준으로
    다시 계산한다** — 화면이 그리는 구간과 옆에 적히는 숫자가 어긋나면 그게 더 나쁘다.

    ⚠️ `payload` 는 캐시에 들어 있는 원본이다. 제자리에서 고치면 캐시가 오염돼
    다음 호출이 이미 잘린 데이터를 받는다. 반드시 복사해서 돌려준다.
    """
    rows = payload.get("rows") or []
    total = len(rows)
    if max_rows <= 0 or total <= max_rows:
        # 자르지 않았다는 사실도 명시한다. 필드가 있다 없다 하면 화면이 분기해야 한다.
        return {**payload, "truncated": False, "total_count": total}

    kept = rows[-max_rows:]
    first, last = kept[0]["close"], kept[-1]["close"]

    return {
        **payload,
        "rows": kept,
        "count": len(kept),
        "change_rate": (last - first) / first * 100 if first else None,
        "truncated": True,
        "total_count": total,
    }


def fetch_history(ticker: str, period: str = "3mo",
                  max_rows: int = MAX_HISTORY_ROWS) -> dict:
    """기간별 일봉을 돌려준다. 캔들 차트와 거래량 차트에 바로 쓸 수 있는 형태다.

    `max_rows` 를 넘으면 **최근 구간만** 남기고 `truncated` 로 알린다. 0 이하를 주면
    자르지 않는다 — 이동평균처럼 **자르기 전 전체가 있어야 왼쪽 끝이 채워지는** 계산을
    하는 쪽(`services/market_chart.series`)이 그렇게 부른다.

    캐시에는 자르기 전 전체가 들어가므로, 같은 구간을 다른 상한으로 다시 물어도
    야후를 한 번 더 부르지 않는다.
    """
    symbol = normalize_ticker(ticker)
    if period not in PERIODS:
        raise YahooError(
            f"period 는 {', '.join(PERIODS)} 중 하나여야 합니다. (받은 값: {period})",
            status=422,
        )
    full = _cached(("history", symbol, period),
                   lambda: _fetch_history_uncached(symbol, period))
    return _limit_rows(full, max_rows)


def _fetch_history_uncached(symbol: str, period: str) -> dict:
    started = time.monotonic()
    # 짧은 구간은 일봉으로 받으면 점이 한두 개뿐이라 차트가 그려지지 않는다.
    interval = INTRADAY_PERIODS.get(period, "1d")
    try:
        frame = yf.Ticker(symbol).history(period=period, interval=interval)
    except Exception as error:
        raise _wrap_error(error, "야후 파이낸스 시세 조회에 실패했습니다") from error

    if frame is None or frame.empty:
        raise YahooError(
            f"'{symbol}' 의 {PERIODS[period]} 시세가 비어 있습니다. 티커를 확인해 주세요.",
            status=404,
        )

    rows: List[dict] = []
    # iterrows 는 (인덱스, 행) 쌍을 준다. 인덱스가 날짜(Timestamp)라 그대로 라벨로 쓴다.
    for stamp, row in frame.iterrows():
        close = _number(row.get("Close"))
        if close is None:
            continue                                    # 결측 행은 차트에 구멍을 내므로 건너뛴다
        rows.append({
            # 분봉은 날짜만 찍으면 같은 라벨이 수십 개가 되므로 시각까지 적는다
            "date": stamp.strftime("%Y-%m-%d %H:%M" if interval != "1d" else "%Y-%m-%d"),
            "open": _number(row.get("Open")),
            "high": _number(row.get("High")),
            "low": _number(row.get("Low")),
            "close": close,
            "volume": _number(row.get("Volume")),
        })

    if not rows:
        raise YahooError(
            f"'{symbol}' 의 {PERIODS[period]} 시세에 쓸 수 있는 값이 없습니다.", status=404)

    first, last = rows[0]["close"], rows[-1]["close"]
    return {
        "ticker": symbol,
        "period": period,
        "period_label": PERIODS[period],
        "interval": interval,
        "rows": rows,
        "count": len(rows),
        # 구간 수익률 — 화면 상단 요약 타일에 쓴다
        "change_rate": (last - first) / first * 100 if first else None,
        "fetched_at": _now_kst(),
        "elapsed_ms": int((time.monotonic() - started) * 1000),
    }
