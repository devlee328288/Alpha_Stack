"""KRX OpenAPI 호출 + 응답 정규화 (외부 연동 계층)

강의 원본(`lecture/main.py`)의 KRX 프록시를 이 저장소의 레이어드 구조에 맞춰 옮긴 모듈이다.
FastAPI 에 의존하지 않는 순수 함수 모음이라 단독으로 실행·테스트할 수 있다.

역할 분담
--------
| 모듈 | 역할 |
|---|---|
| `ingest/clients/krx_data.py` (이 파일) | KRX 와 HTTP 로 통신하고 응답을 정규화한다 |
| `ingest/store/krx_store.py` | 받은 데이터를 SQLite 에 쌓고 꺼내 준다 |
| `market_data.py` (원본에만 있다)  | 쌓인 데이터로 스크리닝·포트폴리오·팩터를 계산한다 |

강의 원본과 달라진 점
--------------------
1. 인증키를 `.key`(단일 줄 / `KEY = VALUE` 둘 다) · `.env` · 환경변수 어디서든 읽는다.
2. KRX 응답의 대문자 축약 필드(`TDD_CLSPRC` ...)를 snake_case 로 **정규화**하고
   숫자 문자열("71,200")을 int/float 로 바꾼다. 원본은 `raw` 에 함께 담아 확인할 수 있다.
3. KOSPI 뿐 아니라 KOSDAQ·KONEX 도 같은 함수로 받을 수 있다.
4. 인증 실패가 확인되면 **차단기**를 걸어 실패가 확정된 요청을 반복하지 않는다.
"""

from __future__ import annotations

import json  # KRX 응답 파싱
import re  # 날짜 형식 검증
from pathlib import Path  # 파일 경로
from typing import Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError  # 네트워크 오류 종류
from urllib.parse import urlencode  # 쿼리스트링 생성
from urllib.request import Request, urlopen  # HTTP 요청 (표준 라이브러리)

from common import secrets  # 인증키 로딩 (공통)

# 이 파일은 <루트>/ingest/clients/ 안에 있으므로 parents[2] 가 프로젝트 루트다.
# (parents[0]=clients, parents[1]=ingest, parents[2]=프로젝트 루트)
# 인증키는 강의 원본과 같이 프로젝트 루트에 두므로 루트를 기준으로 찾는다.
BASE_DIR = Path(__file__).resolve().parents[2]           # 프로젝트 루트 (실행 위치와 무관)
KRX_BASE_URL = "https://data-dbg.krx.co.kr/svc/apis"     # KRX OpenAPI 기본 주소
DATE_PATTERN = re.compile(r"^\d{8}$")                    # YYYYMMDD 8자리
REQUEST_TIMEOUT = 20                                     # KRX 응답 대기 시간(초)

# 시장 구분 → (API 경로, 한국어 이름). KRX 는 시장마다 엔드포인트가 따로다.
MARKET_APIS = {
    "KOSPI": ("sto/stk_bydd_trd", "유가증권 일별매매정보"),
    "KOSDAQ": ("sto/ksq_bydd_trd", "코스닥 일별매매정보"),
    "KONEX": ("sto/knx_bydd_trd", "코넥스 일별매매정보"),
}

# 환경변수·파일에서 찾아볼 키 이름들 (앞에 있는 것이 우선)
KEY_NAMES = ("KRX_API_KEY", "KRX_AUTH_KEY")


# ==================================================
# 1. 인증키 로딩
# ==================================================
def load_krx_key() -> Tuple[str, str]:
    """인증키와 그 출처를 함께 돌려준다. 못 찾으면 `("", "none")`.

    우선순위: 환경변수 → `.env` → `.key`
    (배포 환경에서는 환경변수를, 로컬에서는 파일을 쓰는 흔한 구성이다.)

    실제 파싱은 `common/secrets.py` 가 한다. KOSIS 키도 같은 규칙으로 읽으므로
    공통 모듈로 빼 두었다. `allow_bare=True` 는 강의 원본처럼 `.key` 에 값만
    한 줄 적어 둔 경우를 계속 지원하기 위한 것이다.
    """
    return secrets.load_key(KEY_NAMES, allow_bare=True)


# ==================================================
# 2. KRX 응답 정규화
# ==================================================
# 정규화 필드명 → KRX 원본 필드명. 2026-07 기준 실제 응답으로 검증했다.
FIELD_MAP = {
    "code": "ISU_CD",            # 종목코드(단축)
    "name": "ISU_NM",            # 종목명
    "market": "MKT_NM",          # 시장구분
    "sector": "SECT_TP_NM",      # 소속부
    "close": "TDD_CLSPRC",       # 종가
    "change": "CMPPREVDD_PRC",   # 전일대비
    "change_rate": "FLUC_RT",    # 등락률(%)
    "open": "TDD_OPNPRC",        # 시가
    "high": "TDD_HGPRC",         # 고가
    "low": "TDD_LWPRC",          # 저가
    "volume": "ACC_TRDVOL",      # 누적거래량
    "value": "ACC_TRDVAL",       # 누적거래대금
    "market_cap": "MKTCAP",      # 시가총액
    "listed_shares": "LIST_SHRS",  # 상장주식수
}

# 정수로 바꿀 필드들 (change_rate 만 실수, 나머지는 문자열)
INT_FIELDS = ("close", "change", "open", "high", "low",
              "volume", "value", "market_cap", "listed_shares")


def _to_number(raw, as_int: bool):
    """KRX 는 숫자를 `"71,200"` 같은 문자열로 준다. 콤마를 떼고 숫자로 바꾼다.

    값이 없거나(`"-"`, `""`) 변환할 수 없으면 None 을 돌려준다.
    """
    if raw is None:
        return None
    text = str(raw).replace(",", "").strip()
    if text in ("", "-", "null"):
        return None
    try:
        return int(float(text)) if as_int else float(text)
    except ValueError:
        return None


def normalize_row(row: Dict, bas_dd: str, market: str = "", keep_raw: bool = False) -> Dict:
    """KRX 원본 한 줄을 snake_case + 숫자 타입으로 정규화한다."""
    item = {"date": f"{bas_dd[:4]}-{bas_dd[4:6]}-{bas_dd[6:]}"}   # YYYYMMDD → YYYY-MM-DD

    for field, krx_key in FIELD_MAP.items():
        raw = row.get(krx_key)
        if field in INT_FIELDS:
            item[field] = _to_number(raw, as_int=True)
        elif field == "change_rate":
            item[field] = _to_number(raw, as_int=False)
        else:
            # 문자열 필드는 공백만 정리한다 (KRX 는 종목명 뒤에 공백을 붙여 주는 경우가 있다)
            item[field] = (raw or "").strip() if isinstance(raw, str) else raw

    # 시장 구분이 비어 오는 경우가 있어 요청한 시장으로 채워 둔다
    if not item.get("market"):
        item["market"] = market

    if keep_raw:
        item["raw"] = row     # Swagger 에서 실제 KRX 필드를 그대로 확인할 수 있게 한다
    return item


# ==================================================
# 3. KRX 실호출
# ==================================================
class KrxError(Exception):
    """KRX 호출 실패. `unauthorized` 가 True 면 인증 문제다."""

    def __init__(self, message: str, unauthorized: bool = False):
        super().__init__(message)
        self.message = message
        self.unauthorized = unauthorized


# 가장 최근 KRX 호출 결과를 기억해 둔다 (`/api/krx/status` 에서 재호출 없이 보여주기 위함)
_last_attempt: Dict[str, Optional[str]] = {"result": None, "detail": None}

# 인증이 한 번 거부되면 그 사실을 기억해 두는 차단기(circuit breaker).
# 250거래일을 수집할 때 이게 없으면 실패가 확정된 요청을 500번 반복하게 된다.
_auth_blocked: Dict[str, Optional[str]] = {"reason": None}


def reset_auth_block() -> None:
    """차단기를 풀어 다음 호출에서 KRX 를 다시 시도하게 한다. (승인 직후 재시도용)"""
    _auth_blocked["reason"] = None


def fetch_snapshot(bas_dd: str, market: str = "KOSPI", keep_raw: bool = False) -> List[Dict]:
    """해당 거래일·시장의 전 종목 매매정보를 받아 정규화해서 돌려준다.

    휴장일이면 KRX 가 빈 배열을 주므로 결과도 빈 배열이다(오류가 아니다).
    """
    if not DATE_PATTERN.fullmatch(bas_dd):
        raise KrxError("bas_dd 는 YYYYMMDD 형식이어야 합니다.")
    if market not in MARKET_APIS:
        raise KrxError(f"지원하지 않는 시장입니다: {market}")

    # 이미 인증이 거부된 상태라면 네트워크를 타지 않고 곧바로 실패시킨다
    if _auth_blocked["reason"]:
        raise KrxError(_auth_blocked["reason"], unauthorized=True)

    key, _ = load_krx_key()
    if not key:
        detail = "KRX 인증키가 없습니다. .key 파일 · .env · KRX_API_KEY 환경변수 중 하나를 설정하세요."
        _last_attempt.update(result="no_key", detail=detail)
        _auth_blocked["reason"] = detail
        raise KrxError(detail, unauthorized=True)

    path, _name = MARKET_APIS[market]
    url = f"{KRX_BASE_URL}/{path}?{urlencode({'basDd': bas_dd})}"
    # KRX 는 인증키를 쿼리스트링이 아니라 AUTH_KEY 헤더로 받는다.
    # 헤더로 보내면 브라우저 주소창·서버 접근 로그에 키가 남지 않는다.
    request = Request(url, headers={"AUTH_KEY": key, "Accept": "application/json"}, method="GET")

    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        body = ""
        try:
            body = error.read().decode("utf-8")[:200]
        except Exception:      # 본문을 못 읽어도 진단은 계속해야 한다
            pass
        if error.code in (401, 403):
            # KRX 는 두 가지를 구분해서 알려준다.
            #   "Unauthorized Key"      → 키 자체가 무효 (오타·미발급)
            #   "Unauthorized API Call" → 키는 유효하지만 이 API 이용신청이 승인되지 않음
            if "API Call" in body:
                detail = (
                    f"키는 유효하지만 '{MARKET_APIS[market][1]}' 이용신청이 승인되지 않았습니다. "
                    "KRX OpenAPI 마이페이지에서 승인 상태와 이용기간을 확인하세요."
                )
            else:
                detail = "KRX 가 인증키를 인식하지 못했습니다. 키 값을 다시 확인하세요."
            _last_attempt.update(result="unauthorized", detail=detail)
            _auth_blocked["reason"] = detail
            raise KrxError(detail, unauthorized=True) from error

        detail = f"KRX API 요청 실패 (HTTP {error.code})"
        _last_attempt.update(result="http_error", detail=detail)
        raise KrxError(detail) from error
    except (URLError, TimeoutError, json.JSONDecodeError) as error:
        detail = "KRX API 응답을 가져오지 못했습니다. (네트워크 또는 응답 형식 오류)"
        _last_attempt.update(result="network_error", detail=detail)
        raise KrxError(detail) from error

    # HTTP 200 이어도 본문에 오류 코드가 실려 오는 경우가 있다
    if payload.get("respCode") and str(payload["respCode"]) != "200":
        detail = payload.get("respMsg", "KRX API 오류")
        unauthorized = str(payload["respCode"]) in ("401", "403")
        if unauthorized:
            _auth_blocked["reason"] = detail
        _last_attempt.update(result="api_error", detail=detail)
        raise KrxError(detail, unauthorized=unauthorized)

    _last_attempt.update(result="ok", detail=None)
    _auth_blocked["reason"] = None       # 성공하면 차단기를 자동으로 푼다

    return [normalize_row(row, bas_dd, market, keep_raw)
            for row in payload.get("OutBlock_1", [])]


# ==================================================
# 4. 집계 · 정렬 (화면이 바로 쓸 수 있는 형태로 가공)
# ==================================================
def summarize(items: List[Dict]) -> Dict:
    """스냅샷 차트 3종에 필요한 통계를 서버에서 계산한다.

    2,700종목을 전부 화면에 내려보내면 응답이 무거워지므로,
    집계는 여기서 끝내고 표에는 필요한 페이지만 잘라서 보낸다.
    """
    up = sum(1 for i in items if (i.get("change_rate") or 0) > 0)
    down = sum(1 for i in items if (i.get("change_rate") or 0) < 0)
    flat = len(items) - up - down

    # 거래대금 상위 15종목 (가로 막대차트용)
    ranked = sorted(items, key=lambda i: i.get("value") or 0, reverse=True)[:15]
    top_value = [
        {"code": i["code"], "name": i["name"], "market": i.get("market"),
         "value": i.get("value") or 0, "close": i.get("close"),
         "change_rate": i.get("change_rate")}
        for i in ranked
    ]

    # 등락률 분포 히스토그램. 보합(0%)은 따로 세어 막대가 한쪽으로 쏠리지 않게 한다.
    edges = [(-100, -10), (-10, -5), (-5, -3), (-3, -1), (-1, 0),
             (0, 0), (0, 1), (1, 3), (3, 5), (5, 10), (10, 100)]
    labels = ["-10%↓", "-10~-5", "-5~-3", "-3~-1", "-1~0", "보합",
              "0~1", "1~3", "3~5", "5~10", "10%↑"]
    counts = [0] * len(edges)
    for item in items:
        rate = item.get("change_rate")
        if rate is None:
            continue
        for idx, (lo, hi) in enumerate(edges):
            if lo == 0 and hi == 0:
                matched = rate == 0                  # 보합 전용 칸
            elif rate > 0:
                matched = lo < rate <= hi            # 상승은 (lo, hi] 구간
            else:
                matched = lo <= rate < hi            # 하락은 [lo, hi) 구간
            if matched:
                counts[idx] += 1
                break

    # 시장별 등락 집계 (도넛 차트용)
    by_market: Dict[str, Dict[str, int]] = {}
    for item in items:
        bucket = by_market.setdefault(item.get("market") or "기타",
                                      {"up": 0, "flat": 0, "down": 0, "value": 0})
        rate = item.get("change_rate") or 0
        bucket["up" if rate > 0 else "down" if rate < 0 else "flat"] += 1
        bucket["value"] += item.get("value") or 0

    return {
        "up": up, "flat": flat, "down": down,
        "total_value": sum(i.get("value") or 0 for i in items),
        "total_volume": sum(i.get("volume") or 0 for i in items),
        "top_value": top_value,
        "histogram": [
            {"label": label, "count": count}
            for label, count in zip(labels, counts, strict=False)
        ],
        "by_market": [{"market": m, **v} for m, v in sorted(by_market.items())],
    }


# 표 정렬에 쓸 수 있는 필드 (임의 필드 정렬을 막아 오타·오류를 방지한다)
SORTABLE = ("value", "volume", "change_rate", "close", "market_cap", "code", "name")


def paginate(items: List[Dict], q: str = "", sort: str = "value",
             order: str = "desc", page: int = 1, size: int = 50) -> Tuple[List[Dict], int]:
    """검색 → 정렬 → 페이지 자르기. (표에 내려보낼 목록, 검색 결과 총 건수)를 돌려준다."""
    rows = items
    if q:
        needle = q.strip().lower()
        # 종목명·종목코드 어느 쪽으로도 찾을 수 있게 한다
        rows = [i for i in rows
                if needle in (i.get("name") or "").lower() or needle in (i.get("code") or "")]

    reverse = (order or "desc").lower() != "asc"
    if sort in ("code", "name"):
        rows = sorted(rows, key=lambda i: i.get(sort) or "", reverse=reverse)
    else:
        # 숫자 필드에 None 이 섞여 있어도 정렬이 깨지지 않도록 0 으로 대체한다
        rows = sorted(rows, key=lambda i: i.get(sort) or 0, reverse=reverse)

    total = len(rows)
    start = max(0, (page - 1) * size)
    return rows[start:start + size], total


# ==================================================
# 5. 상태 진단 (화면 배지 · 문제 해결용)
# ==================================================
def get_status() -> Dict:
    """인증키 유무와 마지막 KRX 호출 결과를 알려준다. (KRX 를 다시 부르지는 않는다)"""
    key, key_source = load_krx_key()
    return {
        "key_loaded": bool(key),
        "key_source": key_source,
        "key_length": len(key),          # 값은 절대 노출하지 않고 길이만 알려준다
        "markets": [{"market": m, "api_id": p.split("/")[-1], "api_name": n}
                    for m, (p, n) in MARKET_APIS.items()],
        "auth_blocked": bool(_auth_blocked["reason"]),
        "last_result": _last_attempt["result"],
        "last_detail": _last_attempt["detail"],
    }
