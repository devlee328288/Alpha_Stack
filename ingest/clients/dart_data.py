"""DART(금융감독원 전자공시시스템) 호출 + 정규화 (외부 연동 계층)

금융감독원이 운영하는 **OpenDART** 는 상장·비상장 법인의 공시 원문과 재무제표를
기계가 읽을 수 있는 형태로 준다. 이 모듈은 그중 리서치에 쓰는 세 가지를 맡는다.

    https://opendart.fss.or.kr/guide/main.do

| 하는 일 | 엔드포인트 |
|---|---|
| 재무제표 213계정 | `fnlttSinglAcntAll.json` |
| 공시 목록 (이벤트 타임라인) | `list.json` |
| 기업 개황 (업종·설립일·결산월) | `company.json` |
| 고유번호 매핑 원본 | `corpCode.xml` (ZIP) |

인증키
------
`.key` 에 `DART_API_KEY = ...` 한 줄을 두면 된다. KRX·KOSIS·FRED 와 **같은 규칙**으로
`common/secrets.py` 가 읽는다 (환경변수 → `.env` → `.key` 순).
키는 응답 어디에도 실어 보내지 않는다.

DART 를 쓰면서 확인한 것 (2026-08-01 실제 호출)
----------------------------------------------
1. **종목코드로는 조회할 수 없다.** DART 는 6자리 종목코드가 아니라 **8자리 고유번호
   (`corp_code`)** 로만 받는다. 매핑표는 `corpCode.xml` 한 곳에서만 주고 약 10만 건이라,
   `scripts/build_corp_code.py` 로 미리 뽑아 `data/corp_code.json` 에 넣어 둔다.

2. **`SCE`(자본변동표)에 같은 계정이 여러 번 나온다.** 삼성전자 2024 연결을 보면
   `ifrs-full_Equity` 가 **7번**, `ifrs-full_ProfitLoss` 가 **7번** 반복된다.
   자본변동표는 "자본금·이익잉여금·비지배지분…" 열마다 한 줄씩 주기 때문이다.
   그래서 계정을 찾을 때 **`account_id` 만 보면 안 되고 `sj_div`(재무제표 구분)를 함께 못박아야
   한다.** 자산총계는 `BS`, 매출액·영업이익은 `IS` 에서만 찾는다. 이걸 놓치면
   자본총계로 402조가 아니라 자본금 10조가 잡힌다.

3. **회사마다 계정 이름이 다르다.** 제조업은 "매출액", 금융업·서비스업은 "영업수익"·
   "수익(매출액)" 을 쓴다. 그래서 `account_id`(IFRS 표준 ID) 를 먼저 보고,
   없으면 이름 사전으로 맞춘다. 둘 다 실패하면 **버리지 않고 `G-DATA` 를 발행한 뒤
   계속 진행한다** (GIC 불변원칙 §2-2 — 품질을 속이지 않고 부족함을 밝힌다).

4. **손익계산서가 `IS` 가 아니라 `CIS` 에만 있는 회사가 있다.** 단일 포괄손익계산서를
   쓰는 회사가 그렇다. 그래서 계정마다 찾아볼 `sj_div` 를 **순서 있는 목록**으로 둔다.

5. **오류를 HTTP 상태가 아니라 본문 `status` 로 준다.** 키가 틀려도 HTTP 200 이고
   본문이 `{"status":"010","message":"등록되지 않은 키입니다"}` 다. 그래서 HTTP 코드가 아니라
   `status` 를 봐야 한다. `013`(데이터 없음)은 **오류가 아니라 빈 결과**로 다뤄야 한다 —
   분기보고서를 안 낸 회사·상장 전 연도가 흔하기 때문이다.

6. **빠르다.** 삼성전자 2024 연결 213계정이 0.14초다. yfinance(국내 7초)와 자릿수가 다르다.

제공 함수
---------
| 함수 | 쓰임 |
|---|---|
| `get_corp_code` | 종목코드(005930) → 고유번호(00126380) |
| `fetch_financials` | 재무제표 213계정 정규화 + 주요계정 추출 |
| `fetch_disclosures` | 공시 목록 (최근 N개월 이벤트 타임라인) |
| `fetch_company` | 기업 개황 (업종코드·설립일·결산월) |
| `fetch_corp_code_rows` | `corpCode.xml` 원본 파싱 (빌더 스크립트 전용) |
| `get_status` | 인증키·매핑파일 상태 진단 (키 값은 노출하지 않는다) |
"""

from __future__ import annotations

import io
import json
import re
import threading
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from common import budget, codes, secrets

# 이 파일은 <루트>/ingest/clients/ 안에 있으므로 parents[2] 가 프로젝트 루트다.
BASE_DIR = Path(__file__).resolve().parents[2]
CORP_CODE_FILE = BASE_DIR / "data" / "corp_code.json"

DART_BASE_URL = "https://opendart.fss.or.kr/api"
# 재무제표는 응답이 커서 FRED(20초)보다 넉넉히 준다.
REQUEST_TIMEOUT = 30
# 재무제표 24시간(명세서 §8.1) — 공시는 하루에 여러 번 바뀌지 않는다.
CACHE_TTL = 86400
KST = timezone(timedelta(hours=9))

# 환경변수·파일에서 찾아볼 키 이름 (앞에 있는 것이 우선)
KEY_NAMES = ("DART_API_KEY", "DART_KEY")

# 보고서 코드. DART 는 분기를 이 네 자리로만 받는다.
REPORT_CODES: Dict[str, str] = {
    "11011": "사업보고서",      # 연간 (1~12월)
    "11012": "반기보고서",      # 1~6월
    "11013": "1분기보고서",     # 1~3월
    "11014": "3분기보고서",     # 1~9월
}

# 재무제표 구분 코드
STATEMENT_NAMES: Dict[str, str] = {
    "BS": "재무상태표",
    "IS": "손익계산서",
    "CIS": "포괄손익계산서",
    "CF": "현금흐름표",
    "SCE": "자본변동표",
}

# 연결/별도 구분
FS_DIVS: Dict[str, str] = {"CFS": "연결재무제표", "OFS": "별도재무제표"}

# 다중회사 주요계정(`fnlttMultiAcnt`)이 한 번에 받는 회사 수 상한.
# 실측: 100개는 정상, 200개는 `021`(조회 가능 회사 개수 초과)로 거부한다.
MULTI_ACCOUNT_LIMIT = 100

# DART 가 정한 하루 호출 한도. 아래 `DART_STATUS["020"]` 이 같은 사실을 말한다.
# ⚠️ **이것은 사실이지 정책이 아니다.** 우리가 그중 얼마를 쓸지(예산)는 수집기가 정한다
#    (원본의 `app/services/dart_collector.py` — 옮기지 않았다).
#    둘을 한 파일에 두면 "한도를 올렸다" 와
#    "예산을 늘렸다" 가 같은 diff 로 보인다.
DAILY_CALL_LIMIT = 20_000

# 호출 예산 장부(`call_budget`)에 적힐 출처 이름. 한도 자체는 `common/budget.py`
# 의 LIMITS 에 있고 위 상수와 같은 값이다 — 여기 상수는 **DART 가 정한 사실**이고,
# 그중 얼마를 쓸지(예산)는 저쪽이 정한다.
BUDGET_SOURCE = "dart"

# DART 응답 `status` 코드. 013 은 오류가 아니라 '빈 결과'다.
DART_STATUS: Dict[str, str] = {
    "000": "정상",
    "010": "등록되지 않은 인증키입니다.",
    "011": "사용할 수 없는 인증키입니다. (탈퇴·정지 상태)",
    "012": "접근할 수 없는 IP 입니다.",
    "013": "조회된 데이터가 없습니다.",
    "014": "파일이 존재하지 않습니다.",
    "020": "요청 제한을 초과했습니다. (하루 20,000건)",
    "021": "조회 가능한 회사 개수가 초과했습니다.",
    "100": "요청 값이 올바르지 않습니다.",
    "101": "부적절한 접근입니다.",
    "800": "시스템 점검 중입니다.",
    "900": "정의되지 않은 오류입니다.",
    "901": "사용자 계정이 개인정보보호 요청으로 처리됐습니다.",
}

# --------------------------------------------------
# 주요 계정 사전
# --------------------------------------------------
# 리포트가 반복해서 쓰는 계정만 추린 것이다. 213계정을 전부 이름으로 다루면
# 회사마다 표기가 달라 코드가 조건문 더미가 된다.
#
#   key    우리가 쓰는 이름 (재무비율 계산이 이 이름을 참조한다)
#   label  화면·리포트 표기
#   sj     찾아볼 재무제표 구분 — **순서가 곧 우선순위**다 (위 메모 2·4번)
#   ids    IFRS 표준계정 ID 후보 (이것부터 본다)
#   names  계정명 후보 (표준 ID 가 비어 있는 회사를 위한 대비책)
ACCOUNTS: Tuple[Dict, ...] = (
    # ── 재무상태표 ──────────────────────────────
    {"key": "assets", "label": "자산총계", "sj": ("BS",),
     "ids": ("ifrs-full_Assets",),
     "names": ("자산총계",)},
    {"key": "liabilities", "label": "부채총계", "sj": ("BS",),
     "ids": ("ifrs-full_Liabilities",),
     "names": ("부채총계",)},
    {"key": "equity", "label": "자본총계", "sj": ("BS",),
     "ids": ("ifrs-full_Equity",),
     "names": ("자본총계",)},
    {"key": "equity_owners", "label": "지배주주지분", "sj": ("BS",),
     "ids": ("ifrs-full_EquityAttributableToOwnersOfParent",),
     "names": ("지배기업 소유주지분", "지배기업의 소유주에게 귀속되는 자본")},
    {"key": "current_assets", "label": "유동자산", "sj": ("BS",),
     "ids": ("ifrs-full_CurrentAssets",),
     "names": ("유동자산",)},
    {"key": "current_liabilities", "label": "유동부채", "sj": ("BS",),
     "ids": ("ifrs-full_CurrentLiabilities",),
     "names": ("유동부채",)},
    {"key": "inventories", "label": "재고자산", "sj": ("BS",),
     "ids": ("ifrs-full_Inventories",),
     "names": ("재고자산",)},
    {"key": "cash", "label": "현금및현금성자산", "sj": ("BS",),
     "ids": ("ifrs-full_CashAndCashEquivalents",),
     "names": ("현금및현금성자산",)},

    # ── 손익계산서 ──────────────────────────────
    # 금융업·서비스업은 '매출액' 대신 '영업수익' 을 쓴다 (위 메모 3번)
    {"key": "revenue", "label": "매출액", "sj": ("IS", "CIS"),
     "ids": ("ifrs-full_Revenue", "ifrs-full_RevenueFromContractsWithCustomers"),
     "names": ("매출액", "수익(매출액)", "영업수익", "매출")},
    {"key": "cost_of_sales", "label": "매출원가", "sj": ("IS", "CIS"),
     "ids": ("ifrs-full_CostOfSales",),
     "names": ("매출원가",)},
    {"key": "gross_profit", "label": "매출총이익", "sj": ("IS", "CIS"),
     "ids": ("ifrs-full_GrossProfit",),
     "names": ("매출총이익",)},
    {"key": "operating_income", "label": "영업이익", "sj": ("IS", "CIS"),
     "ids": ("dart_OperatingIncomeLoss", "ifrs-full_ProfitLossFromOperatingActivities"),
     "names": ("영업이익", "영업이익(손실)")},
    {"key": "net_income", "label": "당기순이익", "sj": ("IS", "CIS"),
     "ids": ("ifrs-full_ProfitLoss",),
     "names": ("당기순이익", "당기순이익(손실)", "분기순이익", "반기순이익")},

    # ── 현금흐름표 ──────────────────────────────
    {"key": "cfo", "label": "영업활동현금흐름", "sj": ("CF",),
     "ids": ("ifrs-full_CashFlowsFromUsedInOperatingActivities",),
     "names": ("영업활동현금흐름", "영업활동으로 인한 현금흐름")},
    {"key": "cfi", "label": "투자활동현금흐름", "sj": ("CF",),
     "ids": ("ifrs-full_CashFlowsFromUsedInInvestingActivities",),
     "names": ("투자활동현금흐름", "투자활동으로 인한 현금흐름")},
    {"key": "cff", "label": "재무활동현금흐름", "sj": ("CF",),
     "ids": ("ifrs-full_CashFlowsFromUsedInFinancingActivities",),
     "names": ("재무활동현금흐름", "재무활동으로 인한 현금흐름")},
)

# 재무비율 계산이 없으면 곤란한 계정 — 못 찾으면 `G-DATA` 를 발행한다.
# (나머지는 없어도 리포트가 성립하므로 조용히 비워 둔다)
ESSENTIAL_KEYS = ("revenue", "operating_income", "net_income", "assets", "equity")

# 금융업 판별에 쓰는 계정. 은행·보험·증권은 **이 계정들이 원래 없다.**
#   - 유동/비유동을 나누지 않는다 (유동성 배열법을 쓴다)
#   - 재고자산이 없다
#   - '매출액' 대신 이자수익·보험수익·수수료수익으로 나눠 적는다
# 그래서 이 계정이 없다고 "데이터가 빠졌다" 고 말하면 틀린 진단이 된다.
# 실제로 KB금융 2024 연결(390계정)은 이 셋이 전부 없지만 재무제표는 완전하다.
FINANCIAL_SECTOR_ABSENT = ("current_assets", "inventories", "revenue")

# DART 고유번호는 진짜로 8자리 숫자다 (종목코드와 다른 체계다).
CORP_CODE_PATTERN = re.compile(r"^\d{8}$")

# 종목코드는 여섯 자리 **숫자가 아니다** — 5·6번째 자리에 영문이 온다. 정의는 `common/codes.py`.
# ⚠️ 지금 이 상수를 쓰는 곳은 없다. 그래도 `^\d{6}$` 로 두면 다음 사람이 무심코 집어 쓰는
#    순간 84종이 되돌아오므로 여기서 바로잡아 둔다. DART 3차에서 곧 손댈 자리다.
STOCK_CODE_PATTERN = codes.STOCK_CODE_PATTERN

# 응답 캐시. {(종류, 인자...): (저장시각, 값)}
_cache: Dict[Tuple, Tuple[float, dict]] = {}
_cache_lock = threading.Lock()

# 종목코드 → 고유번호 매핑. 파일을 매번 읽지 않도록 한 번만 올려 둔다.
_corp_map: Optional[Dict[str, Dict]] = None
_corp_map_lock = threading.Lock()

# 마지막 호출 결과 (`/api/dashboard/summary` 가 재호출 없이 보여주기 위함)
_last_attempt: Dict[str, Optional[str]] = {"result": None, "detail": None}


class DartError(Exception):
    """DART 조회 실패. `status` 는 라우터가 그대로 HTTP 상태 코드로 쓴다.

    ⚠️ **`dart_status` 는 HTTP 상태와 다른 축이다.** DART 본문 코드(`DART_STATUS` 의 열쇠)를
    그대로 싣는다. 배치 수집은 이 값으로 *중단할지 건너뛸지*를 가르는데, HTTP 상태만 보면
    `800`(시스템 점검 — 중단해야 한다)과 `900`(정의되지 않은 오류 — 그 종목만 건너뛰면 된다)이
    **둘 다 502** 라 한글 메시지를 파싱해야 구분된다. 그러면 문구를 고치는 날 조용히 깨진다.
    빈 문자열은 "DART 가 답하기 전에 실패했다"(네트워크·HTTP·인증키 없음)는 뜻이다.
    """

    def __init__(self, message: str, status: int = 502, dart_status: str = ""):
        super().__init__(message)
        self.status = status
        self.dart_status = dart_status


class DartQuotaExhausted(DartError):
    """오늘 쓸 수 있는 DART 호출을 다 썼다. **실패가 아니라 "오늘은 여기까지"** 다.

    배치 수집은 이걸 잡아서 얌전히 멈추고, 다음 날 받은 곳부터 이어 받는다.
    `DartError` 를 물려받으므로 기존에 `except DartError` 로 감싼 곳은 그대로 동작한다.
    """

    def __init__(self, message: str):
        super().__init__(message, status=429)


# ==================================================
# 공통 도구
# ==================================================
def load_dart_key() -> Tuple[str, str]:
    """인증키와 그 출처를 함께 돌려준다. 못 찾으면 `("", "none")`.

    `allow_bare=False` 인 이유: `.key` 에는 KRX·KOSIS·FRED 키도 함께 들어 있어서,
    이름 없는 줄을 DART 키로 오해하면 안 된다.
    """
    return secrets.load_key(KEY_NAMES, allow_bare=False)


def _now_kst() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")


def _number(value) -> Optional[float]:
    """DART 금액을 숫자로 바꾼다. 못 바꾸면 `None` (결측으로 취급).

    DART 는 금액을 `"514531948000000"` 처럼 문자열로 주고, 음수는 `"-1,234"` 처럼
    콤마가 섞여 오기도 한다. 값이 없는 칸은 `""` 또는 `"-"` 다.
    """
    if value is None:
        return None
    text = str(value).replace(",", "").strip()
    if text in ("", "-", "null", "N/A"):
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


def clear_cache() -> int:
    """저장해 둔 응답을 전부 버리고 몇 개였는지 돌려준다.

    ⚠️ **씻는 자리가 이 파일에 없었다.** `CACHE_TTL` 이 24시간인데 열쇠에 날짜가 없어서,
    장수하는 웹 프로세스에서는 방금 담은 공시를 리서치 화면이 **하루 내내 낡은 사본**으로
    본다. 오류가 안 뜨므로 사람이 캐시를 의심하지 못한다 — 수집이 끝난 자리에서 부른다.
    """
    with _cache_lock:
        count = len(_cache)
        _cache.clear()
    return count


def _require_key() -> str:
    """인증키를 돌려준다. 없으면 안내와 함께 503 으로 세운다."""
    key, _ = load_dart_key()
    if not key:
        detail = (
            "DART 인증키가 없습니다. 프로젝트 루트 `.key` 에 "
            "`DART_API_KEY = 발급받은_키` 한 줄을 추가하세요. "
            "(무료 발급: https://opendart.fss.or.kr/uss/umt/EgovMberInsertView.do)"
        )
        _last_attempt.update(result="no_key", detail=detail)
        raise DartError(detail, status=503)
    return key


def _call(path: str, params: Dict[str, str], allow_empty: bool = True) -> dict:
    """DART 엔드포인트를 호출하고 JSON 을 돌려준다. 인증키는 여기서만 붙인다.

    `allow_empty=True` 면 `status=013`(데이터 없음)을 오류로 보지 않고 그대로 돌려준다.
    분기보고서를 내지 않은 회사·상장 전 연도가 흔해서, 이걸 예외로 만들면
    호출하는 쪽이 전부 try/except 로 감싸야 한다.
    """
    # ⚠️ **부르기 전에** 센다. 부르고 나서 세면 응답을 못 받고 죽었을 때 이미 나간
    #    호출이 장부에 안 남아 한도를 넘겨 쓴다. 세고 나서 실패하면 손해는 1콜뿐이다.
    #
    # 그리고 **여기가 세는 자리다.** 한 단계 위(`fetch_financials`)에서 세면 재무 본문과
    # 접수일 조회(`list.json`)가 한 번으로 뭉뚱그려진다 — 실제로는 회사·연도마다 2콜이고,
    # 연결이 비어 별도로 재시도하면 3콜이다. 350종 × 5개년이면 그 차이가 1,750콜이다.
    if not budget.try_spend(BUDGET_SOURCE):
        raise DartQuotaExhausted(
            "오늘 쓸 수 있는 DART 호출을 다 썼습니다. "
            "내일 다시 실행하면 받은 곳부터 이어 받습니다."
        )

    key = _require_key()
    query = urlencode({**params, "crtfc_key": key})
    request = Request(f"{DART_BASE_URL}/{path}?{query}",
                      headers={"Accept": "application/json"}, method="GET")

    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = f"DART 요청이 거부됐습니다. (HTTP {error.code})"
        _last_attempt.update(result="http_error", detail=detail)
        raise DartError(detail, status=502) from error
    except (URLError, TimeoutError, json.JSONDecodeError) as error:
        detail = "DART 응답을 가져오지 못했습니다. (네트워크 또는 응답 형식 오류)"
        _last_attempt.update(result="network_error", detail=detail)
        raise DartError(detail, status=502) from error

    # DART 는 오류를 HTTP 상태가 아니라 본문 `status` 로 준다 (위 메모 5번)
    status = str(payload.get("status", "")).strip()
    if status == "000":
        _last_attempt.update(result="ok", detail=None)
        return payload

    message = payload.get("message") or DART_STATUS.get(status, "알 수 없는 오류")

    if status == "013":
        # 데이터가 없는 것은 오류가 아니다. 빈 결과로 돌려주고 호출한 쪽이 판단한다.
        _last_attempt.update(result="empty", detail=None)
        if allow_empty:
            return payload
        raise DartError(f"조회된 데이터가 없습니다. ({message})", status=404, dart_status=status)

    if status in ("010", "011", "012"):
        detail = f"DART 인증키 문제입니다 — {message}"
        _last_attempt.update(result="unauthorized", detail=detail)
        raise DartError(detail, status=502, dart_status=status)

    if status == "020":
        detail = f"DART 요청 한도를 초과했습니다 — {message} (하루 20,000건)"
        _last_attempt.update(result="rate_limit", detail=detail)
        raise DartError(detail, status=429, dart_status=status)

    if status == "100":
        detail = f"DART 요청 값이 올바르지 않습니다 — {message}"
        _last_attempt.update(result="bad_request", detail=detail)
        raise DartError(detail, status=422, dart_status=status)

    detail = f"DART 오류 [{status}] {message}"
    _last_attempt.update(result="api_error", detail=detail)
    raise DartError(detail, status=502, dart_status=status)


# ==================================================
# 1. 고유번호 매핑 (종목코드 → corp_code)
# ==================================================
def _load_corp_map() -> Dict[str, Dict]:
    """`data/corp_code.json` 을 메모리에 한 번만 올린다.

    파일이 없어도 **예외를 던지지 않는다.** 매핑이 없으면 DART 기능만 못 쓰는 것이고
    나머지 화면·API 는 그대로 떠야 하기 때문이다. (`get_status()` 가 그 사실을 알린다)
    """
    global _corp_map
    if _corp_map is not None:
        return _corp_map

    with _corp_map_lock:
        if _corp_map is not None:            # 기다리는 동안 다른 스레드가 채웠을 수 있다
            return _corp_map
        try:
            raw = json.loads(CORP_CODE_FILE.read_text(encoding="utf-8"))
            # 빌더가 넣어 둔 메타(`generated_at` 등)를 빼고 매핑만 쓴다
            _corp_map = raw.get("map") if isinstance(raw, dict) and "map" in raw else raw
        except Exception:
            _corp_map = {}
    return _corp_map


def reload_corp_map() -> int:
    """`data/corp_code.json` 을 **다시 읽는다.** 새로 만든 직후에 부른다.

    ⚠️ `_load_corp_map()` 은 프로세스 수명 동안 **한 번만** 읽는다. 그래서 파일을 새로
    만들어도 이미 떠 있는 웹 프로세스는 낡은 지도를 계속 쓰는데, 상태 조회는 파일을 매번
    읽으므로 **"최신" 이라고 말하면서 조회는 404** 가 된다 — 오류가 아니라 어긋남이라
    가장 찾기 어렵다.
    """
    global _corp_map
    with _corp_map_lock:
        _corp_map = None
    return len(_load_corp_map())


def loaded_corp_count() -> int:
    """지금 이 프로세스가 **실제로 들고 있는** 매핑 수. 파일이 아니라 메모리 쪽이다."""
    return len(_load_corp_map())


def get_corp_code(stock_code: str) -> str:
    """종목코드(`005930`) → DART 고유번호(`00126380`). 못 찾으면 빈 문자열.

    이미 8자리 고유번호를 넣었다면 그대로 돌려준다 (호출하는 쪽이 둘을 구분하지 않아도 되게).
    """
    code = (stock_code or "").strip()
    if CORP_CODE_PATTERN.fullmatch(code):
        return code                       # 이미 고유번호다
    entry = _load_corp_map().get(code)
    if not entry:
        return ""
    return entry["corp_code"] if isinstance(entry, dict) else str(entry)


def get_corp_name(stock_code: str) -> str:
    """종목코드 → 회사명. 못 찾으면 빈 문자열."""
    entry = _load_corp_map().get((stock_code or "").strip())
    if isinstance(entry, dict):
        return entry.get("corp_name", "")
    return ""


def resolve_corp(code: str) -> Tuple[str, str]:
    """종목코드 또는 고유번호 → `(고유번호, 회사명)`. 못 찾으면 안내와 함께 404 로 세운다."""
    corp_code = get_corp_code(code)
    if not corp_code:
        raise DartError(
            f"'{code}' 의 DART 고유번호를 찾지 못했습니다. "
            "상장 종목코드 6자리(예: 005930)를 넣었는지 확인하세요. "
            "매핑 파일이 없다면 `python scripts/build_corp_code.py` 로 만들 수 있습니다.",
            status=404,
        )
    return corp_code, get_corp_name(code)


def fetch_corp_code_rows(*, keep_raw: bool = False) -> List[Dict[str, str]]:
    """`corpCode.xml` 을 받아 전 법인 목록을 돌려준다. **빌더 스크립트 전용**이다.

    응답이 ZIP 이고 그 안에 XML 한 개가 들어 있다. 약 10만 건이라 무겁기 때문에
    앱이 실행 중에 부르지 않는다 — `scripts/build_corp_code.py` 가 한 번 돌려
    상장사만 추려 `data/corp_code.json` 으로 만들어 둔다.

    `keep_raw=True` 면 받은 ZIP 원문을 `common.raw_store` 에 남긴다. 기본값이 거짓인
    이유는 이 함수를 부르는 다른 자리가 생겼을 때 조용히 DB 를 쓰지 않게 하려는 것이고,
    빌더는 참으로 부른다 — 고유번호 매핑은 `modify_date` 로 **바뀌는 값**이라 지금
    받은 표가 언제 받은 것인지 증명할 근거가 없으면 나중에 되짚을 수 없기 때문이다.

    원문 보존이 실패해도 매핑 만들기는 계속한다. 보존은 나중을 위한 보험이지
    지금 이 작업의 목적이 아니고, 여기서 멈추면 빌더가 통째로 실패한다.
    """
    key = _require_key()
    url = f"{DART_BASE_URL}/corpCode.xml?{urlencode({'crtfc_key': key})}"
    request = Request(url, headers={"Accept": "application/octet-stream"}, method="GET")

    try:
        with urlopen(request, timeout=120) as response:      # 파일이 커서 넉넉히 준다
            body = response.read()
    except (HTTPError, URLError, TimeoutError) as error:
        raise DartError(f"corpCode.xml 을 내려받지 못했습니다: {error}", status=502) from error

    # 인증 오류일 때는 ZIP 이 아니라 XML 한 줄로 온다
    if not body.startswith(b"PK"):
        text = body.decode("utf-8", errors="replace")[:200]
        raise DartError(f"corpCode.xml 대신 오류 응답이 왔습니다: {text}", status=502)

    if keep_raw:
        # 지연 import — 원문을 안 남기는 평소 경로가 raw_store 를 끌고 오지 않게 한다.
        from common import raw_store

        try:
            raw_store.save("dart", "corpCode.xml", body, note="고유번호 매핑 원본 ZIP")
        except Exception as error:                       # noqa: BLE001 — 보존 실패는 치명적이지 않다
            print(f"⚠️  원문 보존 실패(계속 진행): {type(error).__name__}: {error}")

    try:
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            xml_name = archive.namelist()[0]
            xml_bytes = archive.read(xml_name)
    except (zipfile.BadZipFile, IndexError) as error:
        raise DartError(f"corpCode ZIP 을 열지 못했습니다: {error}", status=502) from error

    root = ElementTree.fromstring(xml_bytes)
    rows: List[Dict[str, str]] = []
    for node in root.iter("list"):
        rows.append({
            "corp_code": (node.findtext("corp_code") or "").strip(),
            "corp_name": (node.findtext("corp_name") or "").strip(),
            # 비상장사는 종목코드가 공백이다. 빌더가 이걸로 상장사를 걸러낸다.
            "stock_code": (node.findtext("stock_code") or "").strip(),
            "modify_date": (node.findtext("modify_date") or "").strip(),
        })
    return rows


# ==================================================
# 2. 재무제표
# ==================================================
def _normalize_account(row: Dict, *, fs_div: str = "", rcept_dt: str = "") -> Dict:
    """재무제표 한 줄을 숫자 타입으로 정규화한다. **이름은 DART 원본 그대로 둔다.**

    DART 는 당기·전기·전전기 3개 기간을 한 줄에 나란히 준다. 이름(`thstrm_nm`)이
    `"제 56 기"` 라 그대로는 연도를 알 수 없으므로 값과 함께 그대로 실어 보낸다.

    🔴 왜 이름을 안 바꾸나 (이슈 #43)
    ---------------------------------
    처음에는 `thstrm_amount` 를 `this_amount` 로 바꿔 내보냈다. 그런데 반입 규격
    `ingest/inbox/schemas/financial.json` 은 **DART 원본 이름**으로 서 있어서, 우리가
    만든 파일이 반입될 때 **금액 칸이 전부 `extras` 로 밀렸다.** `thstrm_amount` 는
    `required` 가 아니라 **예외도 안 났다** — 금액 없는 행이 조용히 합격했다.

    이름을 되돌린 근거는 실측이다. `this_amount`·`statements` 를 쓰는 곳을 확장자
    제한 없이 전수로 훑었더니 **이 파일 안 11줄이 전부**였다 (외부 소비자 0건).
    관례도 같은 쪽이다 — dbt 는 "rename and recast in **staging**, not in source",
    메달리온 bronze 는 "원본 형식 그대로" 다. 수집 계층은 이름을 바꾸지 않는다.

    🔴 그리고 이름만 고쳐서는 하나도 안 통과했다
    -------------------------------------------
    삼성전자 2023 연결 176행으로 실제 반입을 태워 보니, 이름을 고친 뒤에도
    **176행 전부가 격리**됐다. `bsns_year`·`reprt_code`·`fs_div` 가 `required` 인데
    없었기 때문이다. 원인은 이 함수가 **DART 가 매 행에 실어 주는 식별 칸 4개를
    버리고 있었다**는 것이다.

        DART 원본 17칸 중 우리가 버리던 것: bsns_year · corp_code · reprt_code · rcept_no
        (실측: 176/176 행 전부에 실려 온다)

    그래서 그 넷을 되살리고, 응답에 **없는** `fs_div`·`rcept_dt` 는 부르는 쪽이
    키워드로 넣어 준다. 이 둘까지 채우면 176행이 **176/0 으로 전량 합격**한다.

    `fs_div` 는 요청 파라미터라 응답에 없고, `rcept_dt` 는 공시목록(`list.json`)에만 있다.
    `rcept_dt` 가 특히 중요하다 — 규격이 시점 기준으로 못박은 칸이고, 이것이 없으면
    `has_time_anchor` 같은 error 규칙이 **아예 돌지 않는다.**
    """
    return {
        # 식별 — DART 가 매 행에 실어 준다. 버리면 규격의 required 를 못 채운다
        "corp_code": (row.get("corp_code") or "").strip(),
        "bsns_year": _number(row.get("bsns_year")),
        "reprt_code": (row.get("reprt_code") or "").strip(),
        "rcept_no": (row.get("rcept_no") or "").strip(),
        # 응답에 없어 부르는 쪽이 채워 준다
        "fs_div": fs_div,
        "rcept_dt": rcept_dt,
        # 계정
        "sj_div": row.get("sj_div", ""),
        "sj_nm": STATEMENT_NAMES.get(row.get("sj_div", ""), row.get("sj_nm", "")),
        "account_id": row.get("account_id", ""),
        "account_nm": (row.get("account_nm") or "").strip(),
        "account_detail": (row.get("account_detail") or "").strip(),
        "ord": _number(row.get("ord")),
        "currency": row.get("currency", "KRW"),
        # 당기 · 전기 · 전전기 — 이름은 DART 원본 그대로다
        "thstrm_nm": (row.get("thstrm_nm") or "").strip(),
        "thstrm_amount": _number(row.get("thstrm_amount")),
        "frmtrm_nm": (row.get("frmtrm_nm") or "").strip(),
        "frmtrm_amount": _number(row.get("frmtrm_amount")),
        "bfefrmtrm_nm": (row.get("bfefrmtrm_nm") or "").strip(),
        "bfefrmtrm_amount": _number(row.get("bfefrmtrm_amount")),
    }


def _receipt_date(corp_code: str, rcept_no: str) -> str:
    """접수번호로 **공시 접수일**(`rcept_dt`, YYYYMMDD)을 되찾는다. 못 찾으면 빈 문자열.

    재무제표 응답(`fnlttSinglAcntAll`)에는 접수일이 없고 접수번호만 있다. 그런데
    반입 규격이 시점 기준으로 못박은 칸이 바로 접수일이라, 이것이 없으면
    `has_time_anchor` 같은 error 규칙이 **아예 돌지 않는다.**

    🔴 **접수번호 앞 8자리를 잘라 쓰지 않는다.** 정기공시 4,800건 실측에서 3건(0.062%)이
       어긋났고 그중 둘은 접수일을 **3일 앞당겨** 읽는다. 틀리는 방향이 전부 우리에게
       유리한 쪽이면 잡음이 아니라 **편향**이고, 예외는 나지 않는다.
       앞 8자리는 **조회 구간을 잡는 데만** 쓴다 — 구간은 넉넉해도 값이 틀리지 않는다.

    접수일은 불변이라 캐시가 영구히 유효하다. 같은 보고서를 다시 물어도 호출이 없다.
    실패해도 예외를 올리지 않는다 — 접수일을 못 찾은 것이 수집 전체를 멈출 일은 아니고,
    빈 값이면 반입 쪽이 `has_time_anchor` 로 격리해 **조용히 넘어가지 않는다.**
    """
    if not corp_code or not rcept_no or len(rcept_no) < 8 or not rcept_no[:8].isdigit():
        return ""
    return _cached(("rcept_dt", corp_code, rcept_no),
                   lambda: _receipt_date_uncached(corp_code, rcept_no))


def _receipt_date_uncached(corp_code: str, rcept_no: str) -> str:
    try:
        기준 = datetime.strptime(rcept_no[:8], "%Y%m%d")
    except ValueError:
        return ""

    # ±10일. 앞 8자리가 최대 3일까지 어긋나므로 그보다 넉넉하게 잡는다.
    span = {
        "corp_code": corp_code,
        "bgn_de": (기준 - timedelta(days=10)).strftime("%Y%m%d"),
        "end_de": (기준 + timedelta(days=10)).strftime("%Y%m%d"),
        "page_count": "100",
    }
    try:
        payload = _call("list.json", span)
    except DartError:
        return ""            # 접수일을 못 찾는 것으로 수집을 멈추지 않는다

    for item in payload.get("list") or []:
        if (item.get("rcept_no") or "").strip() == rcept_no:
            return (item.get("rcept_dt") or "").strip()
    return ""


def _pick_accounts(rows: List[Dict]) -> Tuple[Dict[str, Dict], List[Dict]]:
    """정규화된 213계정에서 주요 계정을 뽑는다. → `(찾은 계정, 못 찾은 계정)`

    **`sj_div` 를 먼저 좁히는 것이 핵심이다.** 자본변동표(`SCE`)에는 같은
    `account_id` 가 7번까지 반복되므로, 재무제표 구분을 못박지 않으면 엉뚱한 값이 잡힌다
    (모듈 설명 2번 참고).

    찾는 순서
      1. 지정한 `sj_div` 안에서 `account_id` 일치 (IFRS 표준 ID — 회사가 달라도 같다)
      2. 같은 범위에서 `account_nm` 일치 (표준 ID 를 안 채운 회사 대비)
    """
    # 재무제표 구분별로 미리 나눠 둔다 (계정마다 213줄을 다시 훑지 않기 위해)
    by_sj: Dict[str, List[Dict]] = {}
    for row in rows:
        by_sj.setdefault(row["sj_div"], []).append(row)

    found: Dict[str, Dict] = {}
    missing: List[Dict] = []

    for spec in ACCOUNTS:
        hit: Optional[Dict] = None
        matched_by = ""

        for sj in spec["sj"]:                       # 순서가 곧 우선순위 (IS → CIS)
            candidates = by_sj.get(sj) or []
            if not candidates:
                continue

            # (1) 표준계정 ID 로 찾는다
            for account_id in spec["ids"]:
                hit = next((r for r in candidates if r["account_id"] == account_id), None)
                if hit:
                    matched_by = f"account_id={account_id}"
                    break
            if hit:
                break

            # (2) 계정명 사전으로 찾는다
            for name in spec["names"]:
                hit = next((r for r in candidates if r["account_nm"] == name), None)
                if hit:
                    matched_by = f"account_nm={name}"
                    break
            if hit:
                break

        if hit:
            found[spec["key"]] = {
                "key": spec["key"],
                "label": spec["label"],
                # 바깥으로 내보내는 이름(value·prev·period)은 그대로 둔다 —
                # 화면·리포트가 쓰는 계약이고, 바뀐 것은 안쪽 원본 줄의 이름뿐이다
                "value": hit["thstrm_amount"],
                "prev": hit["frmtrm_amount"],
                "prev2": hit["bfefrmtrm_amount"],
                "period": hit["thstrm_nm"],
                "prev_period": hit["frmtrm_nm"],
                "currency": hit["currency"],
                "sj_div": hit["sj_div"],
                "account_id": hit["account_id"],
                "account_nm": hit["account_nm"],
                "matched_by": matched_by,       # 근거 드릴다운(명세서 §7.4)이 이걸 보여준다
            }
        else:
            missing.append({"key": spec["key"], "label": spec["label"],
                            "searched": list(spec["sj"])})

    return found, missing


def fetch_financials(code: str, year: int, reprt_code: str = "11011",
                     fs_div: str = "CFS") -> dict:
    """재무제표 213계정을 받아 정규화해서 돌려준다.

    `code` 는 종목코드(6자리)든 고유번호(8자리)든 받는다.
    `fs_div="CFS"`(연결)로 요청했는데 비어 있으면 **`OFS`(별도)로 자동 재시도**한다.
    지주회사가 아닌 중소형주는 연결재무제표를 만들지 않는 경우가 흔하다 (명세서 §2.2).
    """
    corp_code, corp_name = resolve_corp(code)
    reprt = (reprt_code or "11011").strip()
    if reprt not in REPORT_CODES:
        raise DartError(
            f"보고서 코드가 올바르지 않습니다: {reprt}. "
            f"({' · '.join(f'{k}={v}' for k, v in REPORT_CODES.items())})", status=422)

    fs = (fs_div or "CFS").strip().upper()
    if fs not in FS_DIVS:
        raise DartError(f"재무제표 구분은 CFS(연결) 또는 OFS(별도)여야 합니다: {fs}", status=422)

    return _cached(("fin", corp_code, int(year), reprt, fs),
                   lambda: _fetch_financials_uncached(corp_code, corp_name, int(year), reprt, fs))


def _fetch_financials_uncached(corp_code: str, corp_name: str, year: int,
                               reprt: str, fs: str) -> dict:
    started = time.monotonic()
    gaps: List[Dict] = []

    def request(div: str) -> List[Dict]:
        payload = _call("fnlttSinglAcntAll.json", {
            "corp_code": corp_code,
            "bsns_year": str(year),
            "reprt_code": reprt,
            "fs_div": div,
        })
        return payload.get("list") or []

    used_fs = fs
    raw_rows = request(fs)

    # 연결이 비면 별도로 한 번 더 — 연결재무제표를 만들지 않는 회사가 많다
    if not raw_rows and fs == "CFS":
        raw_rows = request("OFS")
        if raw_rows:
            used_fs = "OFS"
            gaps.append({
                "code": "G-DATA",
                "message": f"{year}년 연결재무제표가 없어 별도재무제표로 대체했습니다.",
                "detail": "지배·종속 관계가 없거나 연결 대상이 아닌 회사입니다. "
                          "피어 비교 시 연결 기준 회사와 직접 비교하면 규모가 왜곡됩니다.",
            })

    if not raw_rows:
        # 오류가 아니라 '데이터 없음'이다. 빈 결과 + Gap 으로 돌려주고 리포트는 계속 진행한다.
        gaps.append({
            "code": "G-DATA",
            "message": f"{year}년 {REPORT_CODES[reprt]} 재무제표를 찾지 못했습니다.",
            "detail": "상장 이전 연도이거나 해당 보고서를 제출하지 않았을 수 있습니다.",
        })
        return {
            "corp_code": corp_code, "corp_name": corp_name,
            "bsns_year": year, "reprt_code": reprt, "reprt_name": REPORT_CODES[reprt],
            "fs_div": used_fs, "fs_div_name": FS_DIVS[used_fs],
            "currency": "KRW", "rcept_no": "", "rcept_dt": "",
            "accounts": {}, "missing": [], "statements": {}, "count": 0,
            "gaps": gaps, "empty": True,
            "fetched_at": _now_kst(),
            "elapsed_ms": int((time.monotonic() - started) * 1000),
        }

    # 응답에 없는 두 칸을 여기서 채운다 — fs_div 는 요청 파라미터라 응답에 없고,
    # rcept_dt 는 공시목록에만 있다. 둘 다 반입 규격이 required 로 요구하는 칸이다.
    rcept_no = (raw_rows[0].get("rcept_no") or "").strip()
    rcept_dt = _receipt_date(corp_code, rcept_no)
    if not rcept_dt:
        gaps.append({
            "code": "G-DATA",
            "message": "공시 접수일을 되찾지 못했습니다.",
            "detail": "이 숫자를 세상이 알게 된 날을 모르면 시점 기준으로 쓸 수 없습니다. "
                      "반입에 태우면 has_time_anchor 로 격리됩니다 — 조용히 넘어가지 않습니다.",
        })

    rows = [_normalize_account(r, fs_div=used_fs, rcept_dt=rcept_dt) for r in raw_rows]
    accounts, missing = _pick_accounts(rows)

    # 금융업인지 먼저 가른다. 없는 계정을 두고 "빠졌다" 고 할지 "원래 없다" 고 할지가 갈린다.
    missing_keys = {m["key"] for m in missing}
    financial_sector = all(key in missing_keys for key in FINANCIAL_SECTOR_ABSENT)

    # 반드시 있어야 하는 계정이 빠지면 Gap 을 남긴다 (버리지 않고 계속 진행)
    essential_missing = [m for m in missing if m["key"] in ESSENTIAL_KEYS]
    if essential_missing:
        names = " · ".join(m["label"] for m in essential_missing)
        if financial_sector:
            gaps.append({
                "code": "G-DATA",
                "message": f"금융업이라 해당 계정이 재무제표에 없습니다: {names}",
                "detail": "은행·보험·증권은 유동/비유동을 나누지 않고 "
                          "재고자산이 없으며, "
                          "매출액 대신 이자수익·보험수익·수수료수익으로 나눠 적습니다. "
                          "결측이 아니라 회계 관행의 차이이므로, 제조업 피어와 매출·마진을 "
                          "직접 비교하면 안 됩니다.",
            })
        else:
            gaps.append({
                "code": "G-DATA",
                "message": f"주요 계정을 찾지 못했습니다: {names}",
                "detail": "회사가 표준계정 ID 를 채우지 않았거나 사전에 없는 이름을 씁니다. "
                          "해당 계정을 쓰는 재무비율은 계산하지 않습니다.",
            })

    # 재무제표 구분별 전체 목록 — 근거 드릴다운에서 원문을 확인할 때 쓴다
    statements: Dict[str, List[Dict]] = {}
    for row in rows:
        statements.setdefault(row["sj_div"], []).append(row)

    return {
        "corp_code": corp_code,
        "corp_name": corp_name,
        "bsns_year": year,
        "reprt_code": reprt,
        "reprt_name": REPORT_CODES[reprt],
        "fs_div": used_fs,
        "fs_div_name": FS_DIVS[used_fs],
        "currency": rows[0]["currency"] if rows else "KRW",
        "rcept_no": rcept_no,                # 공시 원문 링크에 쓴다
        "rcept_dt": rcept_dt,                # 🔴 이 숫자를 세상이 알게 된 날. 시점 기준이다
        "accounts": accounts,
        "missing": missing,
        "statements": statements,
        "count": len(rows),
        # 다운스트림(재무비율·피어 비교)이 제조업 잣대를 그대로 대지 않도록 알린다
        "financial_sector": financial_sector,
        "gaps": gaps,
        "empty": False,
        "fetched_at": _now_kst(),
        "elapsed_ms": int((time.monotonic() - started) * 1000),
    }


def fetch_multi_accounts(corp_codes: Sequence[str], year: int,
                         reprt_code: str = "11011") -> List[Dict]:
    """**여러 회사의 주요계정을 한 번에** 받는다. (시장 스냅샷 빌더가 쓴다)

    `fnlttSinglAcntAll`(회사당 213~390계정)을 2,764번 부르는 대신,
    `fnlttMultiAcnt`(회사당 30계정)로 100개씩 묶으면 28번이면 끝난다.
    주는 계정은 매출액·영업이익·당기순이익·자산총계·부채총계·자본총계 등 핵심뿐이라
    PER·PBR·ROE 를 구하기에는 충분하다.

    한 번에 **100개가 상한**이다. 200개를 넣으면 `021`(조회 가능 회사 수 초과)로 거부한다.
    그래서 넘치면 여기서 나눠 보내고 결과를 합친다.

    돌려주는 것은 DART 원본 줄 그대로다. 어떤 계정을 어떻게 쓸지는 호출한 쪽이 정한다.
    """
    codes = [c.strip() for c in corp_codes if c and c.strip()]
    if not codes:
        return []
    if reprt_code not in REPORT_CODES:
        raise DartError(f"보고서 코드가 올바르지 않습니다: {reprt_code}", status=422)

    collected: List[Dict] = []
    for start in range(0, len(codes), MULTI_ACCOUNT_LIMIT):
        chunk = codes[start:start + MULTI_ACCOUNT_LIMIT]
        payload = _call("fnlttMultiAcnt.json", {
            "corp_code": ",".join(chunk),
            "bsns_year": str(int(year)),
            "reprt_code": reprt_code,
        })
        collected.extend(payload.get("list") or [])
    return collected


# ==================================================
# 3. 공시 목록
# ==================================================
# DART 공시유형. `pblntf_ty` 파라미터로 **서버에서** 걸러 준다.
PUBLIC_TYPES: Dict[str, str] = {
    "A": "정기공시",        # 사업·반기·분기보고서
    "B": "주요사항보고",    # 자기주식·증자·합병 등 회사의 중대 결정
    "C": "발행공시",
    "D": "지분공시",        # 대량보유·임원 소유상황 — 건수가 압도적으로 많다
    "E": "기타공시",
    "F": "외부감사관련",
    "G": "펀드공시",
    "H": "자산유동화",
    "I": "거래소공시",      # 자율공시·조회공시·수시공시
    "J": "공정위공시",
}

# 이벤트 타임라인의 기본 유형. **지분공시(D)를 뺀 것이 핵심이다.**
# 삼성전자 최근 12개월을 실제로 세어 보면 전체 3,444건 중 지분공시가 3,271건(95%)이다.
# 이걸 그대로 받으면 100건 한도가 대주주 지분 변동으로 다 차서, 정작 실적·투자·자기주식 같은
# 판단 재료가 밀려난다. A+B+I 로 좁히면 80건이 되고 전부 의미 있는 사건이다.
DEFAULT_PUBLIC_TYPES: Tuple[str, ...] = ("A", "B", "I")


def fetch_disclosures(code: str, months: int = 12, limit: int = 100,
                      types: Optional[Tuple[str, ...]] = DEFAULT_PUBLIC_TYPES,
                      *, bgn_de: str = "", end_de: str = "",
                      use_cache: bool = True) -> dict:
    """최근 `months` 개월 공시 목록을 돌려준다. (CORP-TP 의 이벤트 타임라인용)

    `types` 는 받아 올 공시유형이다. 기본값은 정기공시·주요사항보고·거래소공시로,
    지분공시를 뺀 것이다 (위 `DEFAULT_PUBLIC_TYPES` 설명 참고).
    `types=None` 또는 `()` 를 주면 필터 없이 전부 받는다.

    DART 는 유형을 한 번에 하나만 받으므로 유형 수만큼 호출하고 날짜순으로 합친다.
    (3종이면 0.5초 안팎이다)

    `bgn_de`·`end_de`(`YYYYMMDD`)
        구간을 **직접** 못 박는다. `bgn_de` 를 주면 `months` 를 이기고,
        `end_de` 를 비우면 오늘이다. 둘이 함께 있어야 **창을 반으로 쪼개** 다시 물을 수 있다 —
        어느 유형이 100건 상한에 걸렸을 때 그것이 유일한 회수 방법이다.
        ⚠️ 이것이 없으면 표현할 수 있는 가장 좁은 창이 **31일**이다(`months` 하한이 1).
        증분 수집은 "지난번에 본 날 다음부터" 를 물어야 하는데 그 단위가 한 달이면
        같은 공시를 매번 수십 건씩 다시 받는다. 창이 좁아진다고 호출 **수**가 줄지는
        않는다 — 줄어드는 것은 돌려받는 행 수와 100건 상한에 걸릴 확률이다.

    `use_cache=False`
        저장해 둔 값을 쓰지 않고 반드시 새로 받는다.
        ⚠️ 캐시 열쇠에 **날짜가 없고** TTL 이 24시간이라(`CACHE_TTL`), 배치·버튼이 캐시를
        타면 그 날 두 번째 실행부터는 **네트워크를 아예 안 타고 같은 답**을 준다.
        오류가 뜨지 않으므로 "왜 새 공시가 안 담기지" 로만 드러난다. 수집 경로는 끈다.
    """
    corp_code, corp_name = resolve_corp(code)
    months = max(1, min(int(months), 60))
    limit = max(1, min(int(limit), 100))

    wanted = tuple(t for t in (types or ()) if t in PUBLIC_TYPES) or ("",)   # ""=필터 없음

    today = datetime.now(KST)
    begin_text = (bgn_de or "").strip()
    if not (len(begin_text) == 8 and begin_text.isdigit()):
        begin_text = (today - timedelta(days=months * 31)).strftime("%Y%m%d")
    end_text = (end_de or "").strip()
    if not (len(end_text) == 8 and end_text.isdigit()):
        end_text = today.strftime("%Y%m%d")

    produce = lambda: _fetch_disclosures_uncached(          # noqa: E731 (아래 두 갈래가 같은 것을 쓴다)
        corp_code, corp_name, begin_text, end_text, limit, wanted)
    if not use_cache:
        return produce()
    # ⚠️ 열쇠에 `begin_text` 를 넣는다. `months` 만 넣으면 `bgn_de` 가 달라도 같은 칸을 친다.
    return _cached(("disc", corp_code, begin_text, end_text, limit, wanted), produce)


def _fetch_disclosures_uncached(corp_code: str, corp_name: str, bgn_de: str,
                                end_de: str, limit: int, types: Tuple[str, ...]) -> dict:
    started = time.monotonic()
    rows: List[Dict] = []
    total_available = 0
    # ⚠️ **유형별로 따로 세어 둔다.** 아래에서 셋을 합친 뒤에는 어느 유형이 100건 상한에
    #    걸렸는지 알 수 없다 — `total_count` 가 합계라서 "A 가 잘렸다" 와 "셋이 골고루
    #    나왔다" 가 같은 숫자로 보인다. 증분 수집은 잘린 유형만 창을 좁혀 다시 물어야 한다.
    by_type: List[Dict] = []

    for public_type in types:
        params = {
            "corp_code": corp_code,
            "bgn_de": bgn_de,
            "end_de": end_de,
            "page_no": "1",
            "page_count": str(limit),
            "sort": "date",
            "sort_mth": "desc",
        }
        if public_type:
            params["pblntf_ty"] = public_type

        payload = _call("list.json", params)
        type_total = int(payload.get("total_count") or 0)
        total_available += type_total
        items = payload.get("list") or []
        by_type.append({
            "code": public_type,
            "name": PUBLIC_TYPES.get(public_type, "전체"),
            "count": len(items),
            "total_count": type_total,
            # 이 유형만 놓고 봤을 때 DART 가 가진 것보다 적게 받았는가.
            "truncated": type_total > len(items),
        })

        for item in items:
            rcept_no = item.get("rcept_no", "")
            rows.append({
                "rcept_no": rcept_no,
                # 공시 제목 뒤에 공백이 잔뜩 붙어 온다 (`"기업지배구조보고서공시      "`)
                "report_name": (item.get("report_nm") or "").strip(),
                "filer": (item.get("flr_nm") or "").strip(),
                "date": _format_date(item.get("rcept_dt", "")),
                "remark": (item.get("rm") or "").strip(),
                "corp_name": (item.get("corp_name") or "").strip(),
                "stock_code": (item.get("stock_code") or "").strip(),
                # 원문 링크 — 근거(E-) 레코드의 `url` 이 된다
                "url": (
                    f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcept_no}"
                    if rcept_no
                    else ""
                ),
                "public_type": public_type,
                "public_type_name": PUBLIC_TYPES.get(public_type, "전체"),
                "category": _classify(item.get("report_nm") or ""),
            })

    # 유형별로 따로 받았으므로 다시 최신순으로 합친다. 접수번호는 같은 날 안에서도
    # 시간순으로 커지므로 2차 정렬 열쇠로 쓴다.
    rows.sort(key=lambda r: (r["date"], r["rcept_no"]), reverse=True)
    # ⚠️ 두 가지가 서로 다른 잘림이다 — ① 유형을 합친 뒤 `limit` 로 자른 것,
    #    ② 어느 한 유형이 DART 쪽에서 이미 잘려 온 것. ②는 창을 좁혀 다시 묻지 않으면
    #    **영영 못 받는다.** 하나로 뭉치면 그 차이가 사라진다.
    truncated = len(rows) > limit or any(t["truncated"] for t in by_type)
    rows = rows[:limit]

    return {
        "corp_code": corp_code,
        "corp_name": corp_name,
        "begin": _format_date(bgn_de),
        "end": _format_date(end_de),
        "types": [{"code": t, "name": PUBLIC_TYPES.get(t, "전체")} for t in types],
        "total_count": total_available,
        "by_type": by_type,
        "count": len(rows),
        # 한도에 걸려 잘렸는지 밝힌다. 리포트가 "이게 전부"라고 오해하면 안 된다.
        "truncated": truncated,
        "rows": rows,
        "by_category": _count_by_category(rows),
        "fetched_at": _now_kst(),
        "elapsed_ms": int((time.monotonic() - started) * 1000),
    }


# 공시 제목 → 분류. 리포트가 "무슨 일이 있었나"를 묶어 보여줄 때 쓴다.
# 앞에 있는 규칙이 먼저 맞는다 (구체적인 것부터 둔다).
DISCLOSURE_CATEGORIES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        "실적",
        (
            "분기보고서",
            "반기보고서",
            "사업보고서",
            "결산실적",
            "영업(잠정)실적",
            "매출액또는손익구조",
        ),
    ),
    ("지분", ("주식등의대량보유", "임원ㆍ주요주주", "임원·주요주주", "특정증권")),
    ("자본", ("유상증자", "무상증자", "전환사채", "신주인수권", "자기주식", "주식소각", "감자")),
    ("배당", ("현금·현물배당", "현금ㆍ현물배당", "배당")),
    (
        "사업",
        (
            "단일판매",
            "공급계약",
            "신규시설투자",
            "타법인주식",
            "영업양수",
            "영업양도",
            "합병",
            "분할",
        ),
    ),
    ("지배구조", ("기업지배구조", "주주총회", "대표이사", "최대주주")),
    ("감사", ("감사보고서", "회계처리기준", "내부회계관리")),
    ("제재", ("불성실공시", "관리종목", "상장폐지", "소송", "벌금", "과징금")),
)


def _classify(report_name: str) -> str:
    """공시 제목을 분류한다. 어디에도 안 맞으면 '기타'."""
    name = (report_name or "").replace(" ", "")
    for label, keywords in DISCLOSURE_CATEGORIES:
        if any(keyword.replace(" ", "") in name for keyword in keywords):
            return label
    return "기타"


def _count_by_category(rows: List[Dict]) -> List[Dict]:
    """분류별 건수 — 많은 순으로 돌려준다."""
    counts: Dict[str, int] = {}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    return [{"category": k, "count": v}
            for k, v in sorted(counts.items(), key=lambda kv: -kv[1])]


def _format_date(yyyymmdd: str) -> str:
    """`20250530` → `2025-05-30`. 형식이 다르면 그대로 돌려준다."""
    text = (yyyymmdd or "").strip()
    return f"{text[:4]}-{text[4:6]}-{text[6:]}" if len(text) == 8 and text.isdigit() else text


# ==================================================
# 4. 기업 개황
# ==================================================
def fetch_company(code: str) -> dict:
    """업종코드·설립일·결산월 등 회사 기본 정보. (피어 선정과 회계기간 정규화에 쓴다)"""
    corp_code, _ = resolve_corp(code)
    return _cached(("company", corp_code), lambda: _fetch_company_uncached(corp_code))


def _fetch_company_uncached(corp_code: str) -> dict:
    payload = _call("company.json", {"corp_code": corp_code})
    return {
        "corp_code": payload.get("corp_code", corp_code),
        "corp_name": (payload.get("corp_name") or "").strip(),
        "corp_name_eng": (payload.get("corp_name_eng") or "").strip(),
        "stock_name": (payload.get("stock_name") or "").strip(),
        "stock_code": (payload.get("stock_code") or "").strip(),
        "ceo": (payload.get("ceo_nm") or "").strip(),
        # Y=유가증권 · K=코스닥 · N=코넥스 · E=기타
        "market_class": payload.get("corp_cls", ""),
        "market": {"Y": "KOSPI", "K": "KOSDAQ", "N": "KONEX", "E": "기타"}.get(
            payload.get("corp_cls", ""), ""),
        "industry_code": (payload.get("induty_code") or "").strip(),   # 표준산업분류
        "established": _format_date(payload.get("est_dt", "")),
        # 결산월. 12월이 아닌 회사는 회계기간 정규화가 필요하다 (명세서 §5.4 H03)
        "fiscal_month": (payload.get("acc_mt") or "").strip(),
        "address": (payload.get("adres") or "").strip(),
        "homepage": (payload.get("hm_url") or "").strip(),
        "fetched_at": _now_kst(),
    }


# ==================================================
# 5. 상태 진단
# ==================================================
def get_status() -> dict:
    """인증키·매핑파일 상태를 알려준다. (DART 를 다시 부르지는 않는다)"""
    key, source = load_dart_key()
    corp_map = _load_corp_map()
    return {
        "key_loaded": bool(key),
        "key_source": source,
        "key_length": len(key),           # 값은 절대 노출하지 않고 길이만 알려준다
        "corp_code_loaded": bool(corp_map),
        "corp_code_count": len(corp_map),
        "corp_code_file": str(CORP_CODE_FILE.relative_to(BASE_DIR)),
        "account_specs": len(ACCOUNTS),
        "report_codes": [{"code": k, "name": v} for k, v in REPORT_CODES.items()],
        "last_result": _last_attempt["result"],
        "last_detail": _last_attempt["detail"],
    }
