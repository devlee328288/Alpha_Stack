"""FRED(미국 연방준비은행 경제데이터) 호출 + 정규화 (외부 연동 계층)

세인트루이스 연준이 운영하는 **FRED** 는 미국 금리·물가·환율·고용 같은
거시경제 시계열 80만 개를 공개한다. 이 모듈은 그중 종목 차트에 겹쳐 보기 좋은
지표들을 골라 화면이 바로 쓸 수 있는 모양으로 바꿔 준다.

    https://fred.stlouisfed.org/docs/api/fred/

인증키
------
`.key` 에 `FRED_API_KEY = ...` 한 줄을 두면 된다. KRX·KOSIS 와 **같은 규칙**으로
`common/secrets.py` 가 읽는다 (환경변수 → `.env` → `.key` 순).
키는 응답 어디에도 실어 보내지 않는다.

FRED 를 쓰면서 확인한 것 (2026-07 실제 호출)
-------------------------------------------
1. **결측치가 문자열 `"."` 로 온다.** 휴장일·미발표 구간이 그렇다. 숫자로 바꿀 수 없으므로
   `None` 으로 바꿔 두고, 차트에서는 그 자리를 건너뛴다.
2. **키가 틀리면 HTTP 400** 이다 (401 이 아니다). 본문에 `error_message` 가 들어 있어
   그 문장을 그대로 사용자에게 보여 주면 진단이 빠르다.
3. **지표마다 발표 주기가 다르다.** 금리·환율은 일별(D)인데 물가·실업률은 월별(M)이라
   3개월치를 뽑으면 점이 3개뿐이다. 그래서 응답에 `frequency` 를 같이 실어 화면이
   "월별 지표" 라고 알려줄 수 있게 했다.
4. **주말·공휴일이 아예 빠진다.** 주가와 날짜 배열이 어긋나므로, 겹쳐 그릴 때는
   `align_to_dates()` 로 주가 날짜에 맞춰 직전 값을 끌어다 채운다(계단식 보간).

제공 함수
---------
| 함수 | 쓰임 |
|---|---|
| `fetch_series` | 지표 하나의 시계열 (날짜 오름차순) |
| `search_series` | 지표 검색 (FRED 전체에서 이름으로 찾기) |
| `align_to_dates` | 주가 날짜 배열에 맞춰 값을 채운다 (겹쳐 그리기용) |
| `get_status` | 인증키 상태 진단 (키 값은 노출하지 않는다) |
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from common import secrets

FRED_BASE_URL = "https://api.stlouisfed.org/fred"
REQUEST_TIMEOUT = 20                 # FRED 응답 대기 시간(초)
CACHE_TTL = 600                      # 거시지표는 하루 1회 갱신이라 10분 캐시로 충분하다
KST = timezone(timedelta(hours=9))

# 환경변수·파일에서 찾아볼 키 이름 (앞에 있는 것이 우선)
KEY_NAMES = ("FRED_API_KEY", "FRED_KEY")

# --------------------------------------------------
# 화면에 기본으로 띄울 거시지표 큐레이션
# --------------------------------------------------
# FRED 에는 80만 개가 있어서 그대로 보여 주면 고를 수가 없다.
# "주가와 같이 보면 의미가 있는" 것만 골라 두고, 나머지는 `search_series` 로 찾게 한다.
#
#   id          FRED 시리즈 ID
#   label       화면에 쓸 한국어 이름
#   unit        단위 (축 라벨용)
#   group       화면에서 묶어 보여줄 분류
#   note        왜 주가와 같이 보는지 — 학습용 설명
INDICATORS: Tuple[Dict[str, str], ...] = (
    {"id": "DGS10", "label": "미 국채 10년 금리", "unit": "%", "group": "금리",
     "note": "장기 금리. 오르면 미래 이익의 현재가치가 깎여 성장주에 불리하다."},
    {"id": "DGS2", "label": "미 국채 2년 금리", "unit": "%", "group": "금리",
     "note": "정책금리 기대를 가장 빠르게 반영하는 단기 금리."},
    {"id": "T10Y2Y", "label": "장단기 금리차 (10년-2년)", "unit": "%p", "group": "금리",
     "note": "마이너스(역전)가 되면 경기침체 신호로 읽는 대표 지표."},
    {"id": "FEDFUNDS", "label": "연방기금금리 (월별)", "unit": "%", "group": "금리",
     "note": "연준의 정책금리. 월별로만 발표된다."},
    {"id": "VIXCLS", "label": "VIX 변동성 지수", "unit": "p", "group": "위험",
     "note": "'공포지수'. 급등하면 위험자산이 함께 밀린다."},
    {"id": "DEXKOUS", "label": "원/달러 환율", "unit": "원", "group": "환율",
     "note": "국내 종목을 볼 때 특히 중요하다. 환율이 오르면 외국인 수급에 부담이 된다."},
    {"id": "DTWEXBGS", "label": "달러 지수 (광의)", "unit": "지수", "group": "환율",
     "note": "달러가 강하면 신흥국 증시에서 자금이 빠져나가는 경향이 있다."},
    {"id": "SP500", "label": "S&P 500 지수", "unit": "p", "group": "지수",
     "note": "미국 대형주 벤치마크. 개별 종목이 시장을 이겼는지 비교할 때 쓴다."},
    {"id": "NASDAQCOM", "label": "나스닥 종합지수", "unit": "p", "group": "지수",
     "note": "기술주 비중이 높아 성장주 비교 기준으로 적합하다."},
    {"id": "T10YIE", "label": "기대 인플레이션 (10년)", "unit": "%", "group": "물가",
     "note": "시장이 예상하는 향후 10년 평균 물가상승률."},
    {"id": "CPIAUCSL", "label": "미국 소비자물가지수 (월별)", "unit": "지수", "group": "물가",
     "note": "실제 물가. 월별 발표라 점이 듬성듬성하다."},
    {"id": "UNRATE", "label": "미국 실업률 (월별)", "unit": "%", "group": "고용",
     "note": "고용이 나빠지면 금리 인하 기대가 커진다."},
)

# ID 로 바로 찾을 수 있게 만든 색인
INDICATOR_BY_ID: Dict[str, Dict[str, str]] = {item["id"]: item for item in INDICATORS}

# 시계열 캐시. {(종류, 인자...): (저장시각, 값)}
_cache: Dict[Tuple, Tuple[float, dict]] = {}
_cache_lock = threading.Lock()

# 마지막 호출 결과 (`/api/fred/status` 에서 재호출 없이 보여주기 위함)
_last_attempt: Dict[str, Optional[str]] = {"result": None, "detail": None}


class FredError(Exception):
    """FRED 조회 실패. `status` 는 라우터가 그대로 HTTP 상태 코드로 쓴다."""

    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


# ==================================================
# 공통 도구
# ==================================================
def load_fred_key() -> Tuple[str, str]:
    """인증키와 그 출처를 함께 돌려준다. 못 찾으면 `("", "none")`.

    `allow_bare=False` 인 이유: `.key` 에는 KRX·KOSIS 키도 함께 들어 있어서,
    이름 없는 줄을 FRED 키로 오해하면 안 된다.
    """
    return secrets.load_key(KEY_NAMES, allow_bare=False)


def _now_kst() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")


def _number(value) -> Optional[float]:
    """FRED 값을 숫자로 바꾼다. 결측치(`"."`)와 빈 값은 `None`."""
    if value is None:
        return None
    text = str(value).strip()
    if text in ("", ".", "NA", "null"):     # FRED 는 결측을 마침표 하나로 표기한다
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return None if number != number else number      # NaN 방어


def _cached(key: Tuple, producer):
    """`CACHE_TTL` 초 안의 같은 요청은 저장해 둔 값을 돌려준다."""
    now = time.monotonic()
    with _cache_lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < CACHE_TTL:
            return hit[1]

    value = producer()          # 느린 네트워크 호출은 잠금 밖에서 한다

    with _cache_lock:
        _cache[key] = (time.monotonic(), value)
    return value


def _call(path: str, params: Dict[str, str]) -> dict:
    """FRED 엔드포인트를 호출하고 JSON 을 돌려준다. 인증키는 여기서만 붙인다."""
    key, _ = load_fred_key()
    if not key:
        detail = (
            "FRED 인증키가 없습니다. 프로젝트 루트 `.key` 에 "
            "`FRED_API_KEY = 발급받은_키` 한 줄을 추가하세요. "
            "(무료 발급: https://fredaccount.stlouisfed.org/apikeys)"
        )
        _last_attempt.update(result="no_key", detail=detail)
        raise FredError(detail, status=503)

    query = urlencode({**params, "api_key": key, "file_type": "json"})
    request = Request(f"{FRED_BASE_URL}/{path}?{query}",
                      headers={"Accept": "application/json"}, method="GET")

    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        # FRED 는 키 오류·잘못된 시리즈 ID 를 모두 400 으로 주고,
        # 본문 JSON 의 error_message 에 이유를 적어 준다. 그 문장을 그대로 전달한다.
        message = ""
        try:
            message = json.loads(error.read().decode("utf-8")).get("error_message", "")
        except Exception:      # 본문이 JSON 이 아니어도 진단은 계속해야 한다
            pass
        if error.code == 400 and "api_key" in message.lower():
            detail = f"FRED 인증키가 거부됐습니다. 키 값을 확인하세요. (FRED: {message})"
            _last_attempt.update(result="unauthorized", detail=detail)
            raise FredError(detail, status=502) from error
        # FRED 는 사용자 입력 문제도 400 으로 준다. 그대로 502(서버 탓)로 올리면
        # "FRED 가 고장났다" 로 읽히므로, 메시지를 보고 입력 문제와 장애를 갈라 준다.
        lowered = message.lower()
        if "does not exist" in lowered:
            detail = f"'{params.get('series_id', '')}' 지표를 찾을 수 없습니다. (FRED: {message})"
            status = 404
        elif "invalid value for variable" in lowered:
            detail = f"요청 값이 FRED 형식에 맞지 않습니다. (FRED: {message})"
            status = 422
        else:
            detail = message or f"FRED 요청 실패 (HTTP {error.code})"
            status = 502
        _last_attempt.update(result="api_error", detail=detail)
        raise FredError(detail, status=status) from error
    except (URLError, TimeoutError, json.JSONDecodeError) as error:
        detail = "FRED 응답을 가져오지 못했습니다. (네트워크 또는 응답 형식 오류)"
        _last_attempt.update(result="network_error", detail=detail)
        raise FredError(detail, status=502) from error

    _last_attempt.update(result="ok", detail=None)
    return payload


# ==================================================
# 1. 시계열 조회
# ==================================================
# 한 번에 내려보내는 관측치 상한 (ADR-DS-0004).
#
# 비우고 부르면 FRED 는 전 구간을 준다 — DGS10 은 1962년부터라 16,000행이 넘고
# DFF 는 1954년부터라 26,000행이 넘는다. 화면은 그렇게 촘촘한 점을 그리지 못하고,
# Vercel 은 요청·응답 본문을 4.5MB 로 제한하며, 그 전에 maxDuration 을 먼저 친다.
MAX_SERIES_POINTS = 2000


def _limit_points(payload: dict, max_points: int) -> dict:
    """관측치가 너무 많으면 **최근 것만** 남기고 그 사실을 밝힌다.

    앞을 자르는 이유는 시계열에서 최근이 더 쓸모 있기 때문이다.
    자른 뒤에는 통계(first·change·min·max)를 **자른 구간 기준으로 다시 계산한다** —
    화면이 그리는 구간과 옆에 적히는 숫자가 어긋나면 그게 더 나쁘다.

    ⚠️ `payload` 는 캐시에 들어 있는 원본이다. 제자리에서 고치면 캐시가 오염돼
    다음 호출이 이미 잘린 데이터를 받는다. 반드시 복사해서 돌려준다.
    """
    total = len(payload.get("values") or [])
    if max_points <= 0 or total <= max_points:
        # 자르지 않았다는 사실도 명시한다. 필드가 있다 없다 하면 화면이 분기해야 한다.
        return {**payload, "truncated": False, "total_count": total}

    dates = payload["dates"][-max_points:]
    values = payload["values"][-max_points:]
    first, last = values[0], values[-1]

    return {
        **payload,
        "dates": dates,
        "values": values,
        "count": len(values),
        "first": first,
        "latest": last,
        "latest_date": dates[-1],
        "change": last - first,
        "change_rate": (last - first) / first * 100 if first else None,
        "min": min(values),
        "max": max(values),
        "truncated": True,
        "total_count": total,
    }


def fetch_series(
    series_id: str,
    start: str = "",
    end: str = "",
    max_points: int = MAX_SERIES_POINTS,
) -> dict:
    """지표 하나의 시계열을 날짜 오름차순으로 돌려준다.

    `start`·`end` 는 `YYYY-MM-DD`. 비우면 FRED 가 전 구간을 준다.
    지표 이름·단위·발표주기는 `series` 엔드포인트에서 따로 받아 함께 실어 준다.

    `max_points` 를 넘으면 **최근 구간만** 남기고 `truncated` 로 알린다.
    캐시에는 자르기 전 전체가 들어가므로, 같은 지표를 다른 상한으로 다시 물어도
    FRED 를 한 번 더 부르지 않는다.
    """
    sid = (series_id or "").strip().upper()
    if not sid:
        raise FredError("지표 ID를 입력해 주세요. (예: DGS10)", status=422)
    full = _cached(("series", sid, start, end), lambda: _fetch_series_uncached(sid, start, end))
    return _limit_points(full, max_points)


def _fetch_series_uncached(sid: str, start: str, end: str) -> dict:
    started = time.monotonic()

    # (1) 메타 정보 — 이름·단위·발표주기
    meta_rows = _call("series", {"series_id": sid}).get("seriess") or []
    if not meta_rows:
        raise FredError(f"'{sid}' 지표를 찾을 수 없습니다.", status=404)
    meta = meta_rows[0]

    # (2) 관측치
    params = {"series_id": sid, "sort_order": "asc"}
    if start:
        params["observation_start"] = start
    if end:
        params["observation_end"] = end
    observations = _call("series/observations", params).get("observations") or []

    rows: List[dict] = []
    for ob in observations:
        value = _number(ob.get("value"))
        if value is None:
            continue                     # 결측치는 차트에 구멍을 내므로 빼 둔다
        rows.append({"date": ob.get("date", ""), "value": value})

    if not rows:
        raise FredError(
            f"'{sid}' 의 해당 기간 데이터가 비어 있습니다. 기간을 넓혀 보세요.", status=404)

    values = [r["value"] for r in rows]
    first, last = values[0], values[-1]
    curated = INDICATOR_BY_ID.get(sid, {})

    return {
        "series_id": sid,
        # 큐레이션 목록에 있으면 한국어 이름을, 없으면 FRED 원문 제목을 쓴다
        "label": curated.get("label") or meta.get("title") or sid,
        "title": meta.get("title") or sid,
        "unit": curated.get("unit") or meta.get("units_short") or "",
        "units": meta.get("units") or "",
        "frequency": meta.get("frequency") or "",              # Daily · Monthly ...
        "frequency_short": meta.get("frequency_short") or "",  # D · M · Q
        "note": curated.get("note", ""),
        "dates": [r["date"] for r in rows],
        "values": values,
        "count": len(rows),
        "latest": last,
        "latest_date": rows[-1]["date"],
        "first": first,
        # 구간 변화 — 지수·환율은 %, 금리는 %p 로 읽는 편이 자연스럽다
        "change": last - first,
        "change_rate": (last - first) / first * 100 if first else None,
        "min": min(values),
        "max": max(values),
        "fetched_at": _now_kst(),
        "elapsed_ms": int((time.monotonic() - started) * 1000),
    }


# ==================================================
# 2. 지표 검색
# ==================================================
def search_series(text: str, limit: int = 20) -> dict:
    """FRED 전체에서 이름으로 지표를 찾는다. (큐레이션 목록에 없는 지표를 쓰고 싶을 때)"""
    query = (text or "").strip()
    if not query:
        raise FredError("검색어를 입력해 주세요. (예: unemployment)", status=422)
    limit = max(1, min(limit, 100))
    return _cached(("search", query.lower(), limit), lambda: _search_uncached(query, limit))


def _search_uncached(query: str, limit: int) -> dict:
    payload = _call("series/search", {
        "search_text": query,
        "limit": str(limit),
        # 인기순 정렬 — 이름이 비슷한 시리즈가 수백 개라 많이 쓰는 것부터 보여 준다
        "order_by": "popularity",
        "sort_order": "desc",
    })
    rows = [{
        "series_id": s.get("id", ""),
        "title": s.get("title", ""),
        "unit": s.get("units_short", ""),
        "frequency": s.get("frequency_short", ""),
        "start": s.get("observation_start", ""),
        "end": s.get("observation_end", ""),
        "popularity": s.get("popularity", 0),
    } for s in (payload.get("seriess") or [])]

    return {"query": query, "count": len(rows), "rows": rows, "fetched_at": _now_kst()}


# ==================================================
# 3. 주가 날짜에 맞춰 값 채우기 (겹쳐 그리기용)
# ==================================================
def align_to_dates(series: dict, target_dates: Sequence[str]) -> List[Optional[float]]:
    """거시지표를 **주가 날짜 배열에 맞춰** 같은 길이로 돌려준다.

    두 데이터의 달력이 다르기 때문에 필요하다.
      - 미국 공휴일과 한국 공휴일이 다르다 (환율은 있는데 국내 주가는 없는 날 등)
      - 월별 지표(CPI·실업률)는 한 달에 점이 하나뿐이다

    그래서 **직전에 발표된 값을 다음 발표 전까지 그대로 이어 쓴다**(계단식 보간).
    실제로 그 시점에 시장이 알고 있던 값이 직전 발표치이므로, 앞의 값을 끌어오는 방향이 맞다.
    (뒤의 값을 끌어오면 아직 발표되지 않은 값을 쓰는 셈이라 미래 정보가 새어 든다.)

    첫 발표일보다 앞선 날짜는 참조할 값이 없으므로 `None` 이다.
    """
    dates, values = series.get("dates") or [], series.get("values") or []
    aligned: List[Optional[float]] = []
    cursor = 0                      # 지금까지 훑은 지표 위치 — 두 배열을 한 번씩만 지나간다
    last: Optional[float] = None

    for target in target_dates:
        # 목표 날짜 이하의 관측치를 모두 소비하고, 마지막 것을 기억한다
        while cursor < len(dates) and dates[cursor] <= target:
            last = values[cursor]
            cursor += 1
        aligned.append(last)

    return aligned


def correlation(xs: Sequence[Optional[float]], ys: Sequence[Optional[float]]) -> Optional[float]:
    """두 시계열의 **일간 변화율** 상관계수(피어슨)를 구한다.

    가격 수준끼리 비교하면 둘 다 우상향한다는 이유만으로 상관이 높게 나온다(허위 상관).
    변화율로 바꿔야 "같이 움직이는가" 를 본다는 원래 질문에 답할 수 있다.

    값이 비었거나 변화가 전혀 없으면(분모 0) `None` 을 돌려준다.
    """
    # 두 배열 모두 값이 있는 구간만 남겨 변화율을 만든다
    pairs = [(x, y) for x, y in zip(xs, ys, strict=False) if x is not None and y is not None]
    if len(pairs) < 3:
        return None

    dx, dy = [], []
    for (x0, y0), (x1, y1) in zip(pairs, pairs[1:], strict=False):
        if not x0 or not y0:          # 0 으로 나눌 수 없다
            continue
        dx.append((x1 - x0) / x0)
        dy.append((y1 - y0) / y0)

    n = len(dx)
    if n < 2:
        return None

    mean_x, mean_y = sum(dx) / n, sum(dy) / n
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(dx, dy, strict=False))
    var_x = sum((a - mean_x) ** 2 for a in dx)
    var_y = sum((b - mean_y) ** 2 for b in dy)
    if var_x <= 0 or var_y <= 0:
        return None

    return cov / (var_x * var_y) ** 0.5


# ==================================================
# 4. 상태 진단
# ==================================================
def get_status() -> dict:
    """인증키 유무와 마지막 호출 결과를 알려준다. (FRED 를 다시 부르지는 않는다)"""
    key, source = load_fred_key()
    return {
        "key_loaded": bool(key),
        "key_source": source,
        "key_length": len(key),        # 값은 절대 노출하지 않고 길이만 알려준다
        "indicator_count": len(INDICATORS),
        "last_result": _last_attempt["result"],
        "last_detail": _last_attempt["detail"],
    }
