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
import logging  # 원문 보존 실패 경고
import re  # 날짜 형식 검증
import time  # 재시도 백오프
from pathlib import Path  # 파일 경로
from typing import Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError  # 네트워크 오류 종류
from urllib.parse import urlencode  # 쿼리스트링 생성
from urllib.request import Request, urlopen  # HTTP 요청 (표준 라이브러리)

from common import (
    budget,  # 하루 호출 한도 계수 (공통)
    raw_store,  # 응답 원문 보존 (공통)
    secrets,  # 인증키 로딩 (공통)
    settings,  # 원문 보존 스위치 (공통)
)

log = logging.getLogger(__name__)

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

# 지수 일별시세. 위 `sto/*` 가 **종목**을 준다면 이쪽은 **지수 자체**의 종가를 준다.
# 1차 프로젝트의 예측 대상(KOSPI200)이 여기서 나온다 (ADR-AS-0003).
#
# ⭐ 한 번 부르면 그 시장의 **지수 전부**가 온다 — 실측 2026-08-21 기준 KOSPI 51종 ·
#    KOSDAQ 40종. 즉 "코스피 200" 하나를 받든 51종을 다 받든 **콜 비용이 같다.**
#    그래서 전부 저장한다. 섹터 지수(코스피 200 정보기술 등)는 나중에 피처가 된다.
INDEX_APIS = {
    "KOSPI": ("idx/kospi_dd_trd", "코스피 지수 일별시세"),
    "KOSDAQ": ("idx/kosdaq_dd_trd", "코스닥 지수 일별시세"),
}

# 예측 대상 지수의 **정확한 이름**. KRX 가 주는 `IDX_NM` 문자열 그대로다.
# ⚠️ 띄어쓰기까지 맞아야 한다 — "코스피200" 이 아니라 "코스피 200" 이다 (실측).
TARGET_INDEX = "코스피 200"
TARGET_INDEX_KOSDAQ = "코스닥 150"

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


class KrxQuotaExhausted(KrxError):
    """오늘 쓸 수 있는 호출을 다 썼다.

    **고장이 아니라 정상적인 하루의 끝이다.** 그래서 부르는 쪽은 이걸 실패로 세면
    안 된다 — 실패로 세면 멀쩡한 날짜가 재시도 한도를 까먹고 영영 버려진다.
    내일 다시 부르면 받아진다.
    """


#: 호출 한도를 세는 이름. **종목과 지수가 같은 통을 쓴다** — 한도가 걸리는 단위는
#: 엔드포인트가 아니라 인증키이기 때문이다. 여기를 나누면 합계가 한도를 넘겨도
#: 각자는 여유 있어 보인다.
BUDGET_SOURCE = "krx"


# 가장 최근 KRX 호출 결과를 기억해 둔다 (`/api/krx/status` 에서 재호출 없이 보여주기 위함)
_last_attempt: Dict[str, Optional[str]] = {"result": None, "detail": None}

# 인증이 거부되면 그 사실을 기억해 두는 차단기(circuit breaker).
# 250거래일을 수집할 때 이게 없으면 실패가 확정된 요청을 500번 반복하게 된다.
_auth_blocked: Dict[str, Optional[str]] = {"reason": None}

# ⭐ **한 번이 아니라 연속 N 번이다** (2026-08-26 개정).
#
# 원래는 401 을 한 번만 받아도 곧바로 차단했다. 웹 서버에서는 옳다 — 키가 틀렸는데
# 사용자 요청마다 KRX 를 두드릴 이유가 없다. 그런데 **배치 백필에서는 치명적이다.**
#
# 실측 2026-08-26: 지수 16년 백필(4,343콜) 도중 20210618 에서 일시적 401 이 한 번 났다.
# 그 순간 차단기가 걸려 **남은 3,000일이 45초 만에 전부 즉시 실패**했다.
# (네트워크를 안 타므로 빨랐다.) 곧바로 같은 날짜를 재시도하니 48행이 정상으로 왔다 —
# 키는 멀쩡했다. 즉 **깜빡임 하나가 백필 3/4 를 날렸다.**
#
# 그래서 연속 실패를 센다. 성공이 하나라도 끼면 0 으로 되돌린다.
#   · 진짜로 키가 틀렸으면 → 3번 만에 차단된다 (원래 목적 유지)
#   · 일시적 깜빡임이면    → 그 요청만 실패하고 백필은 계속 간다
#
# ⚠️ 워커가 여럿이라 이 카운터는 여러 스레드가 함께 만진다. GIL 아래의 int 증감이라
#    최악의 경우 몇 번 더 세거나 덜 셀 뿐이고, 임계값이 정확할 필요는 없다.
AUTH_FAIL_THRESHOLD = 3
_auth_failures: Dict[str, int] = {"consecutive": 0}

# ⭐⭐ **KRX 는 멀쩡한 키에도 간헐적으로 401 을 준다** (실측 2026-08-26).
#
# 진단: 지수 엔드포인트를 **순차로 2초 간격** 5회 불렀더니 1회가 401, 나머지 4회가 성공했다.
# 같은 키로 종목 엔드포인트는 2/2 성공, 같은 URL 을 urllib 로 직접 부르면 HTTP 200 에
# 정상 본문이 왔다. 즉 키 문제도, 이용신청 문제도, 속도 제한도 아니다 — **그냥 흔들린다.**
#
# 실패율이 20% 대이면 워커 6개짜리 백필에서 연속 3회는 금방 나온다. 실제로 4,343콜 백필이
# 두 번 다 초반에 차단기에 걸려 멈췄다. 그래서 **논리적 요청 하나당 재시도**를 둔다 —
# 차단기는 재시도까지 전부 소진된 뒤에야 한 번을 센다.
#
#   기대 실패율: 0.20 → 0.20³ = 0.8%  (남은 것은 스크립트를 다시 돌려 메운다)
#
# ⚠️ 재시도는 하루 한도(10,000회)를 함께 먹는다. 20% 실패면 약 1.25배다.
#    한도가 빠듯한 백필에서는 `--workers` 를 낮추는 것보다 이 값을 낮추는 편이 예측 가능하다.
AUTH_RETRIES = 2                       # 최초 1회 + 재시도 2회 = 최대 3회
AUTH_RETRY_BACKOFF = (0.5, 1.5)        # 초. 재시도 전에 이만큼 쉰다


def _note_auth_failure(detail: str) -> None:
    """인증 실패를 한 번 세고, 연속 임계에 닿으면 차단기를 건다."""
    _auth_failures["consecutive"] += 1
    if _auth_failures["consecutive"] >= AUTH_FAIL_THRESHOLD:
        _auth_blocked["reason"] = detail


def reset_auth_block() -> None:
    """차단기를 풀어 다음 호출에서 KRX 를 다시 시도하게 한다. (승인 직후 재시도용)"""
    _auth_blocked["reason"] = None
    _auth_failures["consecutive"] = 0


def _request_once(path: str, bas_dd: str, api_name: str) -> List[Dict]:
    """KRX 를 **한 번** 부르고 `OutBlock_1` 원본 행 목록을 돌려준다. 재시도는 하지 않는다.

    **종목(`sto/*`)과 지수(`idx/*`)가 이 함수를 함께 쓴다.** 인증 실패 판정과 차단기,
    본문 오류코드 처리가 두 경로에서 갈리면 한쪽만 고치는 사고가 난다 — 그래서 한 곳에 둔다.

    휴장일이면 KRX 가 빈 배열을 주므로 결과도 빈 배열이다(오류가 아니다).

    ⚠️ **제공 대상기간 밖(2010-01-04 이전)도 빈 배열이다.** 예외가 아니라 0행으로
       조용히 돌아오므로, 부르는 쪽이 "받았는데 없었다"와 "줄 수 없는 날짜다"를 구분하려면
       경계를 따로 알고 있어야 한다 (실측 2026-08-26: 20091230 → 0행 · 20100104 → 1,961행).
    """
    # 이미 인증이 거부된 상태라면 네트워크를 타지 않고 곧바로 실패시킨다
    if _auth_blocked["reason"]:
        raise KrxError(_auth_blocked["reason"], unauthorized=True)

    # ⚠️ **여기가 한도를 세는 자리다.** 한 단계 위(`_request_rows`)에서 세면 간헐적
    #    401 재시도분이 통째로 누락된다 — 재시도도 서버 입장에서는 똑같은 한 번의
    #    호출이고 한도를 똑같이 먹는다. 실측 실패율이 20%대라 그 차이가 1.25배다.
    #
    #    그리고 **부르기 전에** 센다. 부르고 나서 세면 응답을 못 받고 죽었을 때 이미
    #    나간 호출이 장부에 안 남아 한도를 넘겨 쓴다. 세고 나서 실패하면 손해는 1콜뿐이다.
    if not budget.try_spend(BUDGET_SOURCE):
        detail = ("오늘 쓸 수 있는 KRX 호출을 다 썼습니다. "
                  "내일 다시 실행하면 받은 곳부터 이어 받습니다.")
        _last_attempt.update(result="quota_exhausted", detail=detail)
        raise KrxQuotaExhausted(detail)

    key, _ = load_krx_key()
    if not key:
        detail = ("KRX 인증키가 없습니다. "
                  ".key 파일 · .env · KRX_API_KEY 환경변수 중 하나를 설정하세요.")
        _last_attempt.update(result="no_key", detail=detail)
        _auth_blocked["reason"] = detail
        raise KrxError(detail, unauthorized=True)

    url = f"{KRX_BASE_URL}/{path}?{urlencode({'basDd': bas_dd})}"
    # KRX 는 인증키를 쿼리스트링이 아니라 AUTH_KEY 헤더로 받는다.
    # 헤더로 보내면 브라우저 주소창·서버 접근 로그에 키가 남지 않는다.
    request = Request(url, headers={"AUTH_KEY": key, "Accept": "application/json"}, method="GET")

    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            # ⚠️ **바이트 그대로 받아 둔다.** 곧바로 디코딩해 버리면 원문을 보존할 길이
            #    없어진다 — 잘못 디코딩한 문자열은 더 이상 원문이 아니다.
            #    KRX 는 UTF-8 이지만 응답 헤더가 말하는 값을 우선한다.
            body = response.read()
            charset = response.headers.get_content_charset() or "utf-8"
        payload = json.loads(body.decode(charset))
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
                    f"키는 유효하지만 '{api_name}' 이용신청이 승인되지 않았습니다. "
                    "KRX OpenAPI 마이페이지에서 승인 상태와 이용기간을 확인하세요. "
                    "인증키 발급과 서비스별 이용신청은 별개의 2단계입니다."
                )
            else:
                detail = "KRX 가 인증키를 인식하지 못했습니다. 키 값을 다시 확인하세요."
            _last_attempt.update(result="unauthorized", detail=detail)
            # 차단기는 여기서 세지 않는다. 재시도를 전부 쓴 뒤 `_request_rows` 가 센다.
            raise KrxError(detail, unauthorized=True) from error

        if error.code == 429:
            # 서버가 한도 초과를 알려 왔다. **우리 계산보다 서버가 맞다** — 한도가
            # 실제로는 더 낮았거나, 같은 키를 다른 곳에서도 썼을 수 있다. 남은 예산을
            # 즉시 소진 처리해 두면 이후 `try_spend()` 가 전부 False 를 주므로
            # **재시도 루프가 저절로 멈춘다.** 재시도해 봐야 절대 성공하지 않는다.
            budget.mark_exhausted(BUDGET_SOURCE, note="KRX 가 HTTP 429 로 거절했다.")
            detail = "KRX 가 호출 한도 초과를 알려 왔습니다 (HTTP 429). 오늘은 여기까지입니다."
            _last_attempt.update(result="quota_exhausted", detail=detail)
            raise KrxQuotaExhausted(detail) from error

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
        _last_attempt.update(result="api_error", detail=detail)
        raise KrxError(detail, unauthorized=unauthorized)

    _last_attempt.update(result="ok", detail=None)
    # 성공하면 차단기를 풀고 연속 실패 카운터도 0 으로 되돌린다.
    # 카운터를 안 되돌리면 백필 내내 띄엄띄엄 난 실패가 누적돼 결국 차단된다.
    _auth_blocked["reason"] = None
    _auth_failures["consecutive"] = 0

    _keep_raw(path, bas_dd, body, charset)
    return payload.get("OutBlock_1", [])


def raw_target(path: str, bas_dd: str) -> str:
    """보존된 원문을 찾는 이름. 재정규화가 이 이름으로 되짚어 간다.

    엔드포인트 경로를 넣는 이유는 **종목이냐 지수냐를 이름만 보고 갈라야** 하기 때문이다.
    날짜만으로는 어느 정규화 함수를 다시 돌려야 하는지 알 수 없다.
    """
    return f"{path}/{bas_dd}"


def _keep_raw(path: str, bas_dd: str, body: bytes, charset: str) -> None:
    """응답 원문을 남긴다.

    ⚠️ **여기서 실패해도 수집은 계속된다.** 원문 보존은 나중을 위한 보험이지 수집의
       성립 조건이 아니다. 디스크가 찼다고 16년 백필이 멈추면 보험이 사고를 만든 셈이다.
    """
    if not settings.keep_raw_enabled():
        return
    try:
        raw_store.save(BUDGET_SOURCE, raw_target(path, bas_dd), body, encoding=charset)
    except Exception as error:                     # noqa: BLE001 — 수집을 막지 않는다
        log.warning("[원문] %s %s 를 남기지 못했다: %s", path, bas_dd, error)


def fetch_snapshot(bas_dd: str, market: str = "KOSPI", keep_raw: bool = False) -> List[Dict]:
    """해당 거래일·시장의 전 **종목** 매매정보를 받아 정규화해서 돌려준다.

    휴장일이면 빈 배열이다(오류가 아니다).
    """
    if not DATE_PATTERN.fullmatch(bas_dd):
        raise KrxError("bas_dd 는 YYYYMMDD 형식이어야 합니다.")
    if market not in MARKET_APIS:
        raise KrxError(f"지원하지 않는 시장입니다: {market}")

    path, api_name = MARKET_APIS[market]
    rows = _request_rows(path, bas_dd, api_name)
    return [normalize_row(row, bas_dd, market, keep_raw) for row in rows]


# ==================================================
# 3-B. 지수 일별시세 (예측 대상이 여기서 나온다)
# ==================================================
# 정규화 필드명 → KRX 원본 필드명. 2026-08-21 실제 응답으로 검증했다.
INDEX_FIELD_MAP = {
    "index_name": "IDX_NM",          # 지수명 — "코스피 200" (띄어쓰기 포함)
    "index_class": "IDX_CLSS",       # 시장 구분 — KOSPI · KOSDAQ
    "close": "CLSPRC_IDX",           # 종가 지수
    "change": "CMPPREVDD_IDX",       # 전일대비
    "change_rate": "FLUC_RT",        # 등락률(%)
    "open": "OPNPRC_IDX",            # 시가 지수
    "high": "HGPRC_IDX",             # 고가 지수
    "low": "LWPRC_IDX",              # 저가 지수
    "volume": "ACC_TRDVOL",          # 누적거래량
    "value": "ACC_TRDVAL",           # 누적거래대금
    "market_cap": "MKTCAP",          # 시가총액
}

# ⚠️ 지수는 **정수가 아니다.** 종목 시세(원 단위 정수)와 다르게 소수점을 가진다
#    (예: 코스피 200 이 "355.44"). int 로 깎으면 하루 등락이 통째로 사라진다.
INDEX_FLOAT_FIELDS = ("close", "change", "change_rate", "open", "high", "low")
INDEX_INT_FIELDS = ("volume", "value", "market_cap")


def normalize_index_row(row: Dict, bas_dd: str, keep_raw: bool = False) -> Dict:
    """KRX 지수 한 줄을 snake_case + 숫자 타입으로 정규화한다.

    ⚠️ **가격 필드가 빈 문자열로 오는 지수가 있다** — 실측 2026-08-21 기준
       "코스피 (외국주포함)" · "코스닥 (외국주포함)" 두 줄이 `CLSPRC_IDX: ""` 다.
       거래량·시가총액은 채워져 있으므로 행 자체를 버리면 안 되고, 값만 None 이 된다.
       부르는 쪽이 `close is None` 인 행을 걸러야 한다.
    """
    item = {"date": f"{bas_dd[:4]}-{bas_dd[4:6]}-{bas_dd[6:]}"}

    for field, krx_key in INDEX_FIELD_MAP.items():
        raw = row.get(krx_key)
        if field in INDEX_FLOAT_FIELDS:
            item[field] = _to_number(raw, as_int=False)
        elif field in INDEX_INT_FIELDS:
            item[field] = _to_number(raw, as_int=True)
        else:
            item[field] = (raw or "").strip() if isinstance(raw, str) else raw

    if keep_raw:
        item["raw"] = row
    return item


def fetch_index_snapshot(bas_dd: str, market: str = "KOSPI",
                         keep_raw: bool = False) -> List[Dict]:
    """해당 거래일·시장의 **지수 전부**를 받아 정규화해서 돌려준다.

    한 번 부르면 그 시장의 지수가 모두 온다 (실측 2026-08-21: KOSPI 51종 · KOSDAQ 40종).
    "코스피 200" 하나만 쓰더라도 콜 비용은 같으므로 전부 받아 저장한다.

    휴장일이면 빈 배열이다(오류가 아니다).
    """
    if not DATE_PATTERN.fullmatch(bas_dd):
        raise KrxError("bas_dd 는 YYYYMMDD 형식이어야 합니다.")
    if market not in INDEX_APIS:
        raise KrxError(
            f"지수를 지원하지 않는 시장입니다: {market} "
            f"(쓸 수 있는 값: {', '.join(INDEX_APIS)})"
        )

    path, api_name = INDEX_APIS[market]
    rows = _request_rows(path, bas_dd, api_name)
    return [normalize_index_row(row, bas_dd, keep_raw) for row in rows]


# ==================================================
# 3-2. 종목기본정보 — 보통주인지 우선주인지의 **정본**
# ==================================================
# 일별매매정보(`sto/*_bydd_trd`)가 값이라면 이쪽은 **종목의 신분증**이다.
# 우리에게 없던 것을 넷 준다.
#
#   KIND_STKCERT_TP_NM  주권종류 — 보통주 · 구형우선주 · 신형우선주 · 종류주권
#   LIST_DD             상장일
#   SECUGRP_NM          증권그룹 (주권 · 외국주권 등)
#   ISU_CD              ISIN (해외 자료와 잇는 다리)
#
# 🔴 왜 필요한가 — 지금 우리는 **종목명이 '우' 로 끝나는지로 우선주를 추측**하고 있고,
#    그게 보통주 7종을 우선주로 잘못 뺀다 (실측 2026-09-03, 세 시장 × 세 날짜):
#
#      미래에셋대우 · 연우 · 동우 · 신우 · 성우 · 에코글로우 · 이오플로우
#
#    그중 006800 은 20200102 코스피 시총 **48위**라, 모델 파트가 쓰기로 한
#    "KOSPI 보통주 시총 상위 50" 후보에서 조용히 빠진다 (#92 오준영님 요청).
#
#    ⚠️ 이 오류는 **이름이 바뀌는 구간에만** 나타난다 —
#       대우증권(정상) → 미래에셋대우(깨짐) → 미래에셋증권(정상).
#       오늘 유가 943종만 보면 어긋남이 0건이라 표본으로는 절대 안 잡힌다.
#
# ⭐ 한 번 부르면 그 시장의 상장종목이 **전부** 온다 (실측 20260901: 유가 943 ·
#    코스닥 1,822 · 코넥스 108). 지수와 같이 종목 하나를 받든 전부를 받든 콜이 같다.
BASE_INFO_APIS = {
    "KOSPI": ("sto/stk_isu_base_info", "유가증권 종목기본정보"),
    "KOSDAQ": ("sto/ksq_isu_base_info", "코스닥 종목기본정보"),
    "KONEX": ("sto/knx_isu_base_info", "코넥스 종목기본정보"),
}

# 정규화 필드명 → KRX 원본 필드명. 실제 응답 12칸을 전부 받는다 (실측 2026-09-03).
BASE_INFO_FIELD_MAP = {
    "isin_cd": "ISU_CD",             # 표준코드(ISIN) — KR7095570008
    "code": "ISU_SRT_CD",            # 단축코드 — 095570
    "isu_nm": "ISU_NM",              # 정식명 — AJ네트웍스보통주
    "isu_abbrv": "ISU_ABBRV",        # 한글약명 — AJ네트웍스
    "isu_eng_nm": "ISU_ENG_NM",      # 영문명
    "list_dd": "LIST_DD",            # 상장일 (YYYYMMDD)
    "market": "MKT_TP_NM",           # 시장구분 — KOSPI · KOSDAQ · KONEX
    "secugrp_nm": "SECUGRP_NM",      # 증권구분 — 주권 · 외국주권 …
    "sect_tp_nm": "SECT_TP_NM",      # 소속부 — 🔴 유가에서는 항상 빈 문자열이다
    "kind_stkcert_tp_nm": "KIND_STKCERT_TP_NM",  # 주권종류 ← 우선주 판별의 정본
    "parval": "PARVAL",              # 액면가 (문자열로 '무액면' 이 올 수 있다)
    "list_shrs": "LIST_SHRS",        # 상장주식수
}

# 액면가는 '무액면' 같은 문자열이 섞여 오므로 숫자로 단정하지 않는다.
BASE_INFO_INT_FIELDS = ("list_shrs",)

#: 주권종류 중 **보통주가 아닌 것**. 유니버스에서 뺄 대상이다.
#: 실측 2026-09-03 기준 나타난 값은 이 셋 + '보통주' 넷뿐이다.
NON_COMMON_STOCK_KINDS = ("구형우선주", "신형우선주", "종류주권")

#: 보통주를 가리키는 값. 여기 없는 값이 새로 나타나면 **보통주가 아닌 것으로 본다** —
#: 모르는 값을 보통주로 넣으면 우선주가 유니버스에 섞이고, 그건 조용히 틀린다.
COMMON_STOCK_KIND = "보통주"


def normalize_base_info_row(row: Dict, bas_dd: str, market: str = "",
                            keep_raw: bool = False) -> Dict:
    """종목기본정보 한 줄을 snake_case 로 정규화한다.

    `bas_dd` 를 함께 담는 이유는 이 응답이 **그날의 사실**이기 때문이다 — 같은
    엔드포인트를 다른 날짜로 부르면 다른 답이 온다 (실측 2026-09-03: 유가 20150102
    899행 · 20200102 916행 · 20260901 943행, 2015 에만 있고 지금은 없는 종목 159종,
    공통 740종 중 상장주식수가 다른 종목 534종). 오늘 스냅샷이 아니라 이력이다.
    """
    item = {"bas_dd": bas_dd}

    for field, krx_key in BASE_INFO_FIELD_MAP.items():
        raw = row.get(krx_key)
        if field in BASE_INFO_INT_FIELDS:
            item[field] = _to_number(raw, as_int=True)
        else:
            item[field] = (raw or "").strip() if isinstance(raw, str) else raw

    # 시장이 비어 오면 요청한 시장으로 채운다 (일별매매정보와 같은 처리)
    if not item.get("market"):
        item["market"] = market

    if keep_raw:
        item["raw"] = row
    return item


def is_common_stock(kind: Optional[str]) -> bool:
    """주권종류가 보통주인가. **모르는 값은 보통주가 아니라고 본다.**

    빈 문자열·None 을 보통주로 치면, 응답이 깨졌을 때 우선주가 유니버스에 섞인다.
    유니버스에 종목이 빠지는 것보다 틀린 종목이 들어오는 게 비싸다.
    """
    return (kind or "").strip() == COMMON_STOCK_KIND


def fetch_base_info(bas_dd: str, market: str = "KOSPI",
                    keep_raw: bool = False) -> List[Dict]:
    """해당 거래일·시장의 **상장종목 기본정보 전부**를 받아 정규화해서 돌려준다.

    휴장일이면 빈 배열이다(오류가 아니다).
    """
    if not DATE_PATTERN.fullmatch(bas_dd):
        raise KrxError("bas_dd 는 YYYYMMDD 형식이어야 합니다.")
    if market not in BASE_INFO_APIS:
        raise KrxError(
            f"종목기본정보를 지원하지 않는 시장입니다: {market} "
            f"(쓸 수 있는 값: {', '.join(BASE_INFO_APIS)})"
        )

    path, api_name = BASE_INFO_APIS[market]
    rows = _request_rows(path, bas_dd, api_name)
    return [normalize_base_info_row(row, bas_dd, market, keep_raw) for row in rows]


def _request_rows(path: str, bas_dd: str, api_name: str) -> List[Dict]:
    """`_request_once` 를 감싸 **간헐적 401 을 재시도**한다. 바깥은 이쪽만 부른다.

    KRX 는 멀쩡한 키에도 401 을 띄엄띄엄 준다(위 ⭐⭐ 주석의 실측). 재시도가 없으면
    4,343콜짜리 백필이 초반에 차단기에 걸려 멈춘다 — 실제로 두 번 그랬다.

    재시도 대상은 **인증 실패(`unauthorized`)뿐**이다. 네트워크 오류·HTTP 5xx 는
    그대로 올려보낸다 — 그쪽은 부르는 쪽(수집 스크립트)이 날짜 단위로 이미 처리하고,
    여기서까지 재시도하면 실패가 몇 겹으로 늘어져 진행 상황이 안 보인다.
    """
    if _auth_blocked["reason"]:
        raise KrxError(_auth_blocked["reason"], unauthorized=True)

    last: Optional[KrxError] = None
    for attempt in range(AUTH_RETRIES + 1):
        try:
            rows = _request_once(path, bas_dd, api_name)
        except KrxError as error:
            if not error.unauthorized:
                raise                      # 인증 문제가 아니면 재시도하지 않는다
            last = error
            if attempt < AUTH_RETRIES:
                # 키가 없어서 난 실패는 재시도해도 소용없다 — 차단기가 이미 걸려 있다
                if _auth_blocked["reason"]:
                    raise
                time.sleep(AUTH_RETRY_BACKOFF[min(attempt, len(AUTH_RETRY_BACKOFF) - 1)])
                continue
            # 재시도를 전부 썼다. 이제서야 차단기에 한 번을 센다.
            _note_auth_failure(error.message)
            raise
        else:
            _auth_failures["consecutive"] = 0
            return rows

    raise last if last else KrxError("KRX 호출에 실패했습니다.")
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
