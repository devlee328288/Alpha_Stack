"""ECOS(한국은행 경제통계시스템) 호출 + 정규화 (외부 연동 계층)

한국은행이 운영하는 **ECOS** 는 국내 금리·환율·물가·경기지표를 공개한다.
FRED 가 미국을 맡는다면 ECOS 는 그 국내 짝이다. 국내 종목을 분석할 때
"미 국채 10년" 만 보고 "한은 기준금리" 를 안 보면 앞뒤가 맞지 않는다.

    https://ecos.bok.or.kr/api/

인증키
------
`.key` 에 `ECOS_API_KEY = ...` 한 줄을 두면 된다. KRX·KOSIS·FRED·DART 와 **같은 규칙**으로
`common/secrets.py` 가 읽는다 (환경변수 → `.env` → `.key` 순).
키는 응답 어디에도 실어 보내지 않는다.

ECOS 를 쓰면서 확인한 것 (2026-08-01 실제 호출)
----------------------------------------------
1. **모든 인자가 URL 경로에 들어간다.** 쿼리스트링이 아니라
   `/{키}/json/kr/{시작행}/{끝행}/{통계표}/{주기}/{시작}/{끝}/{항목1}…` 형태다.
   그래서 인증키가 **주소 자체에 박힌다.** 서버 접근 로그에 남을 수 있으므로
   오류 메시지에 URL 을 그대로 싣지 않도록 조심해야 한다 (아래 `_call` 참고).

2. **오류도 HTTP 200 으로 온다.** 본문이 `{"RESULT":{"CODE":"INFO-100",...}}` 로 바뀔 뿐이다.
   그래서 HTTP 코드가 아니라 `RESULT.CODE` 를 봐야 한다.
   `INFO-200`(데이터 없음)은 **오류가 아니라 빈 결과**다 — 조회 범위를 넓히면 해결되는 일이라
   장애로 취급하면 안 된다.

3. **주기마다 날짜 형식이 다르다.** 일별은 `20260703`, 월별은 `202607`, 분기는 `2026Q2`,
   연간은 `2026` 이다. 형식이 어긋나면 데이터가 없다고 나오지 오류가 나지 않아서
   원인을 찾기 어렵다. 그래서 큐레이션 목록에 주기를 함께 적어 두고 이 모듈이 변환한다.

4. **`DATA_VALUE` 가 문자열이다.** `"3"`·`"1554.1"` 처럼 온다. 숫자로 바꿔야 차트에 쓸 수 있다.

5. **한 번에 받을 수 있는 행 수에 상한이 있다.** 시작·끝 행을 지정하는 방식이라
   일별 지표를 5년치 받으면 1,200행이 넘는다. `MAX_ROWS` 로 넉넉히 잡아 두고
   그보다 길면 잘렸다는 사실을 응답에 밝힌다.

제공 함수
---------
| 함수 | 쓰임 |
|---|---|
| `fetch_series` | 큐레이션 지표 하나의 시계열 (날짜 오름차순) |
| `list_indicators` | 큐레이션 목록 (화면 선택 상자용) |
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
from urllib.parse import quote
from urllib.request import Request, urlopen

from common import secrets

ECOS_BASE_URL = "https://ecos.bok.or.kr/api"
REQUEST_TIMEOUT = 20                 # ECOS 응답 대기 시간(초)
CACHE_TTL = 600                      # 거시지표는 하루 1회 갱신이라 10분 캐시로 충분하다
MAX_ROWS = 5000                      # 한 번에 받을 최대 행 수 (일별 5년치 ≈ 1,250행)
KST = timezone(timedelta(hours=9))

# 환경변수·파일에서 찾아볼 키 이름 (앞에 있는 것이 우선)
KEY_NAMES = ("ECOS_API_KEY", "ECOS_KEY", "BOK_API_KEY")

# --------------------------------------------------
# 큐레이션 — 리서치에 실제로 쓰는 국내 거시지표
# --------------------------------------------------
# ECOS 에도 통계표가 수백 개라 그대로 두면 고를 수가 없다.
# "국내 주가와 같이 보면 의미가 있는" 것만 골랐고, 전부 실호출로 코드를 확인했다.
#
#   id      우리가 붙인 짧은 이름 (API 파라미터로 쓴다)
#   stat    ECOS 통계표코드
#   cycle   주기 — D(일) · M(월) · Q(분기) · A(연)
#   items   항목코드 (통계표마다 뜻이 다르다)
#   label   화면에 쓸 한국어 이름
#   unit    단위 (축 라벨용)
#   group   화면에서 묶어 보여줄 분류
#   note    왜 주가와 같이 보는지 — 학습용 설명
INDICATORS: Tuple[Dict, ...] = (
    {"id": "base_rate", "stat": "722Y001", "cycle": "M", "items": ("0101000",),
     "label": "한국은행 기준금리", "unit": "%", "group": "금리",
     "note": "국내 통화정책의 기준. 오르면 차입비용이 늘어 성장주·부동산에 부담이 된다."},
    {"id": "ktb3y", "stat": "817Y002", "cycle": "D", "items": ("010200000",),
     "label": "국고채 3년", "unit": "%", "group": "금리",
     "note": "정책금리 기대를 빠르게 반영하는 단기 지표금리. 일별로 나온다."},
    {"id": "ktb10y", "stat": "817Y002", "cycle": "D", "items": ("010210000",),
     "label": "국고채 10년", "unit": "%", "group": "금리",
     "note": "장기 금리. 미 국채 10년(FRED DGS10)과 나란히 보면 국내외 금리차를 읽을 수 있다."},
    {"id": "usdkrw", "stat": "731Y001", "cycle": "D", "items": ("0000001",),
     "label": "원/달러 환율 (매매기준율)", "unit": "원", "group": "환율",
     "note": "한국은행 고시 기준율. 오르면 외국인 수급에 부담이고 수출주에는 유리하다."},
    {"id": "jpykrw", "stat": "731Y001", "cycle": "D", "items": ("0000002",),
     "label": "원/엔 환율 (100엔)", "unit": "원", "group": "환율",
     "note": "일본과 수출 경쟁이 겹치는 업종(자동차·철강·기계)을 볼 때 함께 본다."},
    {"id": "cpi", "stat": "901Y009", "cycle": "M", "items": ("0",),
     "label": "소비자물가지수", "unit": "지수", "group": "물가",
     "note": "2020=100 기준. 물가가 오르면 금리 인상 압력이 커진다. 월별 발표라 점이 듬성듬성하다."},
    {"id": "ppi", "stat": "404Y014", "cycle": "M", "items": ("*AA",),
     "label": "생산자물가지수", "unit": "지수", "group": "물가",
     "note": "기업이 치르는 물가. 소비자물가보다 먼저 움직여 마진 압박을 미리 알려 준다."},
    {"id": "leading", "stat": "901Y067", "cycle": "M", "items": ("I16E",),
     "label": "경기선행지수 순환변동치", "unit": "지수", "group": "경기",
     "note": "100 을 넘으면 경기 확장 국면으로 읽는다. 산업 사이클 판정(IND-R)의 기준선이다."},
    {"id": "coincident", "stat": "901Y067", "cycle": "M", "items": ("I16D",),
     "label": "경기동행지수 순환변동치", "unit": "지수", "group": "경기",
     "note": "지금 경기 상태. 선행지수와 벌어지는 방향을 보면 국면 전환을 가늠할 수 있다."},
)

# ID 로 바로 찾을 수 있게 만든 색인
INDICATOR_BY_ID: Dict[str, Dict] = {item["id"]: item for item in INDICATORS}

# 주기별 날짜 형식 길이 (위 메모 3번)
CYCLE_FORMATS: Dict[str, str] = {"D": "%Y%m%d", "M": "%Y%m", "Q": "quarter", "A": "%Y"}
CYCLE_NAMES: Dict[str, str] = {"D": "일별", "M": "월별", "Q": "분기별", "A": "연간"}

# 시계열 캐시. {(종류, 인자...): (저장시각, 값)}
_cache: Dict[Tuple, Tuple[float, dict]] = {}
_cache_lock = threading.Lock()

# 마지막 호출 결과 (`/api/dashboard/summary` 가 재호출 없이 보여주기 위함)
_last_attempt: Dict[str, Optional[str]] = {"result": None, "detail": None}


class EcosError(Exception):
    """ECOS 조회 실패. `status` 는 라우터가 그대로 HTTP 상태 코드로 쓴다."""

    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


# ==================================================
# 공통 도구
# ==================================================
def load_ecos_key() -> Tuple[str, str]:
    """인증키와 그 출처를 함께 돌려준다. 못 찾으면 `("", "none")`.

    `allow_bare=False` 인 이유: `.key` 에는 다른 기관 키도 함께 들어 있어서,
    이름 없는 줄을 ECOS 키로 오해하면 안 된다.
    """
    return secrets.load_key(KEY_NAMES, allow_bare=False)


def _now_kst() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")


def _number(value) -> Optional[float]:
    """ECOS 값을 숫자로 바꾼다. 못 바꾸면 `None` (결측으로 취급)."""
    if value is None:
        return None
    text = str(value).replace(",", "").strip()
    if text in ("", "-", "null", "NA"):
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


def _format_period(moment: datetime, cycle: str) -> str:
    """날짜를 주기에 맞는 ECOS 형식으로 바꾼다 (위 메모 3번).

    일별 `20260703` · 월별 `202607` · 분기 `2026Q3` · 연간 `2026`
    """
    if cycle == "Q":
        return f"{moment.year}Q{(moment.month - 1) // 3 + 1}"
    return moment.strftime(CYCLE_FORMATS.get(cycle, "%Y%m"))


def _display_date(period: str, cycle: str) -> str:
    """ECOS 기간 표기를 `YYYY-MM-DD` 로 편다. 차트가 날짜축을 그릴 수 있게 한다.

    월별은 그 달 1일로, 분기는 그 분기 첫 달 1일로, 연간은 1월 1일로 편다.
    (실제 발표 시점이 아니라 **기간의 시작점**이라는 뜻이다)
    """
    text = (period or "").strip()
    if cycle == "D" and len(text) == 8:
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    if cycle == "M" and len(text) == 6:
        return f"{text[:4]}-{text[4:]}-01"
    if cycle == "Q" and "Q" in text:
        year, _, quarter = text.partition("Q")
        month = (int(quarter) - 1) * 3 + 1 if quarter.isdigit() else 1
        return f"{year}-{month:02d}-01"
    if cycle == "A" and len(text) == 4:
        return f"{text}-01-01"
    return text


def _call(stat: str, cycle: str, start: str, end: str,
          items: Sequence[str]) -> List[Dict]:
    """ECOS 통계 조회. 인증키는 여기서만 붙인다.

    **오류 메시지에 URL 을 싣지 않는다.** ECOS 는 인증키를 경로에 박기 때문에
    URL 을 그대로 로그·응답에 내보내면 키가 새어 나간다 (모듈 설명 1번).
    """
    key, _ = load_ecos_key()
    if not key:
        detail = (
            "ECOS 인증키가 없습니다. 프로젝트 루트 `.key` 에 "
            "`ECOS_API_KEY = 발급받은_키` 한 줄을 추가하세요. "
            "(무료 발급: https://ecos.bok.or.kr/api/#/AuthKeyApply)"
        )
        _last_attempt.update(result="no_key", detail=detail)
        raise EcosError(detail, status=503)

    # 경로 조각은 각각 인코딩한다 (`*AA` 같은 항목코드에 특수문자가 들어간다)
    parts = [key, "json", "kr", "1", str(MAX_ROWS), stat, cycle, start, end]
    parts.extend(items)
    url = f"{ECOS_BASE_URL}/StatisticSearch/" + "/".join(quote(str(p), safe="") for p in parts)
    request = Request(url, headers={"Accept": "application/json"}, method="GET")

    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = f"ECOS 요청이 거부됐습니다. (HTTP {error.code})"
        _last_attempt.update(result="http_error", detail=detail)
        raise EcosError(detail, status=502) from error
    except (URLError, TimeoutError, json.JSONDecodeError) as error:
        detail = "ECOS 응답을 가져오지 못했습니다. (네트워크 또는 응답 형식 오류)"
        _last_attempt.update(result="network_error", detail=detail)
        raise EcosError(detail, status=502) from error

    # 오류도 HTTP 200 으로 온다 (위 메모 2번)
    result = payload.get("RESULT")
    if result:
        code = str(result.get("CODE", ""))
        message = result.get("MESSAGE", "")
        if code == "INFO-200":
            # 데이터 없음은 오류가 아니다. 빈 목록으로 돌려주고 호출한 쪽이 판단한다.
            _last_attempt.update(result="empty", detail=None)
            return []
        if code == "INFO-100":
            detail = "ECOS 인증키가 거부됐습니다. 키 값을 확인하세요."
            _last_attempt.update(result="unauthorized", detail=detail)
            raise EcosError(detail, status=502)
        detail = f"ECOS 오류 [{code}] {message}"
        _last_attempt.update(result="api_error", detail=detail)
        raise EcosError(detail, status=502)

    _last_attempt.update(result="ok", detail=None)
    return payload.get("StatisticSearch", {}).get("row") or []


# ==================================================
# 1. 시계열 조회
# ==================================================
def list_indicators() -> List[Dict]:
    """큐레이션 목록을 돌려준다. (화면 선택 상자·API 문서용)"""
    return [{
        "id": item["id"], "label": item["label"], "unit": item["unit"],
        "group": item["group"], "cycle": item["cycle"],
        "cycle_name": CYCLE_NAMES.get(item["cycle"], item["cycle"]),
        "note": item["note"],
    } for item in INDICATORS]


def fetch_series(indicator_id: str, years: int = 3) -> dict:
    """큐레이션 지표 하나의 시계열을 날짜 오름차순으로 돌려준다.

    `years` 는 오늘로부터 거슬러 올라갈 햇수다. 주기에 맞는 날짜 형식은 이 모듈이 만든다.
    """
    key = (indicator_id or "").strip().lower()
    spec = INDICATOR_BY_ID.get(key)
    if not spec:
        raise EcosError(
            f"'{indicator_id}' 는 없는 지표입니다. "
            f"쓸 수 있는 것: {' · '.join(INDICATOR_BY_ID)}", status=422)

    years = max(1, min(int(years), 30))
    return _cached(("series", key, years), lambda: _fetch_series_uncached(spec, years))


def _fetch_series_uncached(spec: Dict, years: int) -> dict:
    started = time.monotonic()
    cycle = spec["cycle"]

    today = datetime.now(KST)
    begin = today - timedelta(days=365 * years + 10)      # 윤년 여유를 조금 둔다
    rows = _call(spec["stat"], cycle,
                 _format_period(begin, cycle), _format_period(today, cycle),
                 spec["items"])

    points: List[Dict] = []
    for row in rows:
        value = _number(row.get("DATA_VALUE"))
        if value is None:
            continue                     # 결측치는 차트에 구멍을 내므로 빼 둔다
        points.append({
            "period": row.get("TIME", ""),
            "date": _display_date(row.get("TIME", ""), cycle),
            "value": value,
        })

    # ECOS 가 정렬을 보장하지 않으므로 직접 오름차순으로 맞춘다
    points.sort(key=lambda p: p["period"])

    if not points:
        raise EcosError(
            f"'{spec['label']}' 의 최근 {years}년 데이터가 비어 있습니다. "
            "기간을 넓혀 보세요.", status=404)

    values = [p["value"] for p in points]
    first, last = values[0], values[-1]
    meta = rows[0] if rows else {}

    return {
        "id": spec["id"],
        "label": spec["label"],
        "stat_name": meta.get("STAT_NAME", ""),
        "item_name": meta.get("ITEM_NAME1", ""),
        "unit": meta.get("UNIT_NAME") or spec["unit"],
        "group": spec["group"],
        "cycle": cycle,
        "cycle_name": CYCLE_NAMES.get(cycle, cycle),
        "note": spec["note"],
        "dates": [p["date"] for p in points],
        "periods": [p["period"] for p in points],
        "values": values,
        "count": len(points),
        "latest": last,
        "latest_date": points[-1]["date"],
        "first": first,
        # 구간 변화 — 금리는 %p, 지수·환율은 % 로 읽는 편이 자연스럽다
        "change": last - first,
        "change_rate": (last - first) / first * 100 if first else None,
        "min": min(values),
        "max": max(values),
        # 상한에 걸려 잘렸는지 밝힌다 (모듈 설명 5번)
        "truncated": len(rows) >= MAX_ROWS,
        "fetched_at": _now_kst(),
        "elapsed_ms": int((time.monotonic() - started) * 1000),
    }


# ==================================================
# 2. 주가 날짜에 맞춰 값 채우기 (겹쳐 그리기용)
# ==================================================
def align_to_dates(series: dict, target_dates: Sequence[str]) -> List[Optional[float]]:
    """거시지표를 **주가 날짜 배열에 맞춰** 같은 길이로 돌려준다.

    `fred_data.align_to_dates` 와 같은 규칙이다 — **직전에 발표된 값을 다음 발표 전까지
    그대로 이어 쓴다**(계단식 보간). 그 시점에 시장이 알고 있던 값이 직전 발표치이므로
    앞의 값을 끌어오는 방향이 맞다. (뒤의 값을 끌어오면 미래 정보가 새어 든다)

    ECOS 는 월별 지표가 많아 이 처리가 특히 중요하다. 기준금리는 한 달에 점이 하나뿐이다.
    첫 발표일보다 앞선 날짜는 참조할 값이 없으므로 `None` 이다.
    """
    dates, values = series.get("dates") or [], series.get("values") or []
    aligned: List[Optional[float]] = []
    cursor = 0                      # 지금까지 훑은 지표 위치 — 두 배열을 한 번씩만 지나간다
    last: Optional[float] = None

    for target in target_dates:
        while cursor < len(dates) and dates[cursor] <= target:
            last = values[cursor]
            cursor += 1
        aligned.append(last)

    return aligned


# ==================================================
# 3. 상태 진단
# ==================================================
def get_status() -> dict:
    """인증키 유무와 마지막 호출 결과를 알려준다. (ECOS 를 다시 부르지는 않는다)"""
    key, source = load_ecos_key()
    return {
        "key_loaded": bool(key),
        "key_source": source,
        "key_length": len(key),        # 값은 절대 노출하지 않고 길이만 알려준다
        "indicator_count": len(INDICATORS),
        "last_result": _last_attempt["result"],
        "last_detail": _last_attempt["detail"],
    }
