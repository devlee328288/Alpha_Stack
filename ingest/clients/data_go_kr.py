"""공공데이터포털 금융위원회 API 연동 (외부 연동 계층)

    https://apis.data.go.kr/1160100/service/<서비스>/<오퍼레이션>    ← 구세대
    https://apis.data.go.kr/1160100/<서비스>/<오퍼레이션>            ← 신세대 (`service` 없음)

🔴 **경로 틀이 둘이다** (실측 2026-09-03). v3.2 에서 신규 서비스 6개를 여러 조합으로
두드리고 전부 `NO_OPENAPI_SERVICE_ERROR` 를 받아 "못 찾음" 으로 남겼는데, 조합이 틀린 게
아니라 **틀이 틀렸다** — 신규 서비스는 `/service/` 를 안 거친다. 그래서 `BASE_URL` 은
호스트까지만 두고, 어느 틀인지는 `EP_*` 상수가 **경로 전체**로 말한다.

**시세를 가져오려는 게 아니다.** 포털의 주식시세는 20칸뿐이고 우리 KRX 수집은 이미
9,223,644행이다. 여기서 얻는 것은 **다리**다 —

    KRX 종목코드  ↔  crno(법인등록번호)  ↔  ISIN

이 다리가 없으면 우리 시세는 DART(고유번호)에도, 해외 자료(ISIN)에도 못 붙는다.
그리고 `corp_profile` 이 **공식 상장폐지일**을 준다. 지금 우리는 "어느 날부터 시세가
안 나온다" 로 폐지를 추정하는데, 그건 장기 거래정지와 구별되지 않는다.

실제 호출로 확인한 것 (2026-09-03)
--------------------------------
| 무엇 | 실측 |
|---|---|
| KRX상장종목정보 칸 | 7개 — `basDt` `srtnCd` `isinCd` `crno` `corpNm` `itmsNm` `mrktCtg` |
| 기업기본정보 칸 | 37개 (`enp` 접두사가 많다) |
| **과거 구간** | **2020-01-02 부터.** 2019 이전은 `totalCount=0` 이다 |
| 하루 종목 수 | 2,334종(2020) → 2,728종(2025). `numOfRows=1000` 이면 3콜 |
| 페이징 | `totalCount` 와 실제로 받은 행 수가 정확히 맞는다 (2,334 · 2,728 확인) |
| `srtnCd` | **전부** `A` + 6자 = 7자. 2,728종 예외 없음 |
| 기업기본정보 이력 | 한 법인에 여러 행. `fstOpegDt`~`lastOpegDt` 가 유효구간이다 |
| 상장폐지일 | 사라진 법인 8곳 **전부** `enpXchgLstgAbolDt` 가 채워져 왔다 |

🔴🔴 가장 중요한 것 — **`basDt` 목록은 그 시점 목록이 아니다**
------------------------------------------------------------
`basDt=20200102` 로 받은 2,334종 안에 **그날 이후에야 상장된 종목이 33종** 있었다.
가장 늦은 것은 듀켐바이오(176750) 로 첫 시세가 **2024-12-20**, 4년 뒤다
(실측 2026-09-03 · `scripts/verify_identity.py` 가 매번 다시 센다).

즉 포털은 기준일 목록을 주는 게 아니라 **최신에 가까운 목록에 기준일 딱지만** 붙여 준다.

    🔴 `stock_identity` 로 유니버스를 그대로 만들면 아직 없던 종목이 섞인다.
       미래참조이고, **에러 없이 성능만 좋아진다.**
       반드시 `daily_price` 와 **교집합**을 내서 쓴다.

이 표의 쓸모는 *"그날 무엇이 상장돼 있었나"* 가 아니라 **`code ↔ crno ↔ ISIN` 다리**다.
다리로 쓰는 데는 이 성질이 해가 되지 않는다.

한 가지 더 — 포털 목록은 **우선주와 외국기업을 담지 않는다.** 20200102 기준으로
우리 시세에만 있는 141종의 내역이 우선주 120 · 외국기업(900·950 계열) 21 이었다.
그래서 신형우선주 84종이 `stock_identity` 에 안 보이는 것은 **우리가 거른 게 아니라
포털이 안 준 것**이다. 이 둘을 섞으면 없는 버그를 쫓게 된다.

🔴 조심할 것 넷
--------------
**① 서비스키를 다시 인코딩하지 않는다.** Encoding 키에는 `%3D` 가 들어 있어서
`urlencode({"serviceKey": key})` 를 하면 `%253D` 가 되고 인증이 실패한다. 실패 문구가
"키가 없다" 가 아니라 "등록되지 않은 키" 라 원인을 찾기 어렵다.

**② `srtnCd` 의 `A` 접두사를 뗀다.** 안 떼면 `daily_price.code` 와 조인이 **0행**이
되는데, 조인은 0행이어도 에러가 나지 않는다 — 그냥 아무것도 안 나온다.

**③ 종목코드는 숫자가 아니다.** 5·6번째 자리에 영문이 오는 종목이 84종 있다
(`0001A0`·`00088K` 등). `isdigit()` 으로 거르면 그 84종이 조용히 사라진다.
다만 앞 4자리는 항상 숫자다 (우리 DB 3,677종 전수 확인).

**④ 날짜 형식이 한 응답 안에서 세 가지다.**

    enpEstbDt          18970925      YYYYMMDD
    enpXchgLstgDt      76/03/24      YY/MM/DD   ← 두 자리 연도
    fssCorpChgDtm      2025/08/07    YYYY/MM/DD

두 자리 연도는 `56~99 → 1900대` · `00~55 → 2000대` 로 푼다 (`_YY_PIVOT`).
KRX 가 1956년 개장이라 그보다 이른 상장일이 없다는 도메인 근거가 있다.

책임 경계
--------
이 API 가 죽어도 시세·재무는 돈다. KRX 와 DART 가 정본이고 여기는 **다리와 보강**이다.
그래서 실패를 예외로 올리지 않고 `DataGoKrError` 로 **무엇을 해야 하는지까지** 담아
부르는 쪽이 판단하게 한다.
"""

from __future__ import annotations

import json
import re
from typing import Dict, Iterator, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from common import budget, secrets

#: 🔴 호스트까지만. 기관코드(`1160100`)와 `service` 유무는 **경로 틀이 둘이라** 여기 못
#:    둔다 — `EP_*` 가 경로 전체를 갖는다. 예탁결제원(`B552481`)은 규약이 달라
#:    `ksd_data.py` 로 따로 뺐다.
BASE_URL = "https://apis.data.go.kr"
REQUEST_TIMEOUT = 30

#: 예산 대장(`common.budget`)에 쓰는 출처 이름.
BUDGET_SOURCE = "data_go_kr"

#: 개발계정 하루 한도. 포털 마이페이지 기준값이고, 운영계정은 다르다.
#: 🔴 숫자를 여기 따로 적지 않는다 — 장부(`common.budget.LIMITS`)와 어긋나면 두 경로가
#:    서로 다른 통을 세게 된다. 근거는 그쪽 주석에 있다.
DAILY_LIMIT = budget.LIMITS[BUDGET_SOURCE]

KEY_NAMES = ("DATA_GO_KR_API_KEY", "DATA_GO_KR_KEY", "PUBLIC_DATA_API_KEY")

#: 엔드포인트 — **호스트 뒤 경로 전체**다. 경로가 틀리면 `NO_OPENAPI_SERVICE_ERROR` 가 온다.
#:
#: 🔴 `/service/` 유무는 서비스마다 다르고 **이름으로 유추할 수 없다.** `_V2` 가 붙은
#:    `GetCorpBasicInfoService_V2` 는 구세대(있음)이고 같은 `_V2` 인
#:    `GetStocDiviInfoService_V2` 는 신세대(없음)다. 아래 값은 전부 **실호출로 확인한**
#:    것이다(2026-09-03 · 포털 상세페이지 HTML 에 박힌 Swagger 에서 긁고, 127콜로 응답
#:    본문까지 열었다). 새 경로는 추측하지 말고 같은 방법으로 잰다:
#:
#:        curl -s "https://www.data.go.kr/data/<pk>/openapi.do" \
#:          | grep -oE "apis\.data\.go\.kr[^\"'<> ]*"
#:
#: ── 구세대 · `1160100/service/…` ──────────────────────────────────────
EP_LISTED = "1160100/service/GetKrxListedInfoService/getItemInfo"
EP_CORP_OUTLINE = "1160100/service/GetCorpBasicInfoService_V2/getCorpOutline_V2"
# 지배구조 — 주주(basDt 있음 · 285,730행) · 대표이사(37,064행) · 임원보수 · 임원
EP_GOV_SHAREHOLDER = "1160100/service/GetCorpGoveInfoService/getStockholderInfo"
EP_GOV_CEO = "1160100/service/GetCorpGoveInfoService/getReprDireInfo"
EP_GOV_EXEC_PAY = "1160100/service/GetCorpGoveInfoService/getExecRemuStat"
EP_GOV_EXECUTIVES = "1160100/service/GetCorpGoveInfoService/getExecutivesInfo"
#: ── 신세대 · `1160100/…` (`service` 없음) ────────────────────────────
# 🔴 주식발행정보만 **_V3** 다. 나머지 신규는 _V2.
# 종목기본정보 — 보통주/우선주(`scrsItmsKcdNm`)·상장일(`lstgDt`)·액면가·발행주식수를
# **basDt 시점별로** 준다(하루 15,913행 · 16쪽). KRX 종목기본정보(v11)와 교차검증 축.
EP_ITEM_BASIC = "1160100/GetStocIssuInfoService_V3/getItemBasiInfo_V3"
EP_ISSUE_STAT = "1160100/GetStocIssuInfoService_V3/getStocIssuStat_V3"
EP_ISSUE_HISTORY = "1160100/GetStocIssuInfoService_V3/getStocIssuInfo_V3"
EP_LOCKUP = "1160100/GetStocIssuInfoService_V3/getLockUpRetuInfo_V3"
# 배당 — ⚠️ `basDt` 는 적재일이라 이벤트 날짜가 아니다. 빼고 부르면 전량(71,674행 · 72콜).
EP_DIVIDEND = "1160100/GetStocDiviInfoService_V2/getDiviInfo_V2"
# 권리일정 — 사유(`stckIssuRcdNm`)·권리기준일(`rgtExertSttgDt`). 수정주가 chain 대조 후보.
EP_RIGHTS_SCHEDULE = "1160100/GetStocRighScheService_V2/getRighExerReasSche_V2"
# 대차 — 종목별 잔고(공매도 대용). ⚠️ "업종별참여" 의 `sicNm` 은 **참여자 구분**("외국인")
# 이지 산업 업종이 아니다 (#92). 이름만 보고 설계하면 틀린다.
EP_LEND_BY_ITEM = "1160100/GetStocLendBorrInfoService_V2/getStItemLendAndBorrStatu_V2"
EP_LEND_BY_PARTICIPANT = "1160100/GetStocLendBorrInfoService_V2/getStBusiTypePartStatu_V2"
# 주식분포(기준일 파라미터가 `basDt` 가 아니라 응답의 `rgtExertQualBasDt`) · 사고주권
EP_DISTRIBUTION = "1160100/GetStocTradInfoService_V2/getStocDistInfo_V2"
EP_IRREGULAR_STOCK = "1160100/GetStocTradInfoService_V2/getIrreRigforSecu_V2"
# 예탁가능 — 상장/비상장·예탁취소까지. 🔴 `lstgAbolDt` 에 자리표시자 `99991231` 이 온다.
EP_DEPOSIT_AVAILABLE = "1160100/GetStocDepoInfoService_V2/getDepoAvaiWhet_V2"

#: 이름 → 경로. 시험이 "어느 틀인가" 를 실측표와 대조할 때 쓴다. 상수를 더하면 여기도 더한다.
ENDPOINTS: Dict[str, str] = {
    "listed": EP_LISTED,
    "corp_outline": EP_CORP_OUTLINE,
    "gov_shareholder": EP_GOV_SHAREHOLDER,
    "gov_ceo": EP_GOV_CEO,
    "gov_exec_pay": EP_GOV_EXEC_PAY,
    "gov_executives": EP_GOV_EXECUTIVES,
    "item_basic": EP_ITEM_BASIC,
    "issue_stat": EP_ISSUE_STAT,
    "issue_history": EP_ISSUE_HISTORY,
    "lockup": EP_LOCKUP,
    "dividend": EP_DIVIDEND,
    "rights_schedule": EP_RIGHTS_SCHEDULE,
    "lend_by_item": EP_LEND_BY_ITEM,
    "lend_by_participant": EP_LEND_BY_PARTICIPANT,
    "distribution": EP_DISTRIBUTION,
    "irregular_stock": EP_IRREGULAR_STOCK,
    "deposit_available": EP_DEPOSIT_AVAILABLE,
}


def endpoint_generation(path: str) -> str:
    """`legacy`(`/service/` 있음) 또는 `modern`(없음). 경로가 어느 틀인지 사람이 읽게."""
    return "legacy" if path.startswith("1160100/service/") else "modern"

#: 한 번에 받을 행 수. 포털 상한이 1,000 이다.
PAGE_SIZE = 1_000

#: 🔴 **우리가 가진 가장 이른 기준일.** 그보다 이르면 호출해 봐야 0건이다.
#:    2019-01-02~07 을 물어도 `totalCount=0` 이고 2020-01-02 에서 2,334건이 나온다
#:    (실측 2026-09-03). 이걸 상수로 두는 이유는, 모르고 2010년부터 훑으면
#:    **2,500콜을 0건에 쓰고** 하루 한도의 4분의 1이 사라지기 때문이다.
EARLIEST_BAS_DD = "20200102"

#: 두 자리 연도를 어느 세기로 볼지 가르는 자리.
#:
#: `56~99` 는 1900대, `00~55` 는 2000대다. KRX(당시 대한증권거래소)가 **1956년 3월**에
#: 문을 열었으므로 그보다 이른 상장일은 존재하지 않는다 — 이 경계는 우리 자료의 성질에서
#: 나온 것이지 임의로 고른 값이 아니다. 2055년까지 안전하다.
#:
#: ⚠️ "올해를 기준으로" 같은 규칙은 쓰지 않는다. 해가 바뀌면 **같은 원문이 다른 값으로**
#:    읽혀서, 작년에 받은 행과 올해 받은 행이 조용히 어긋난다.
_YY_PIVOT = 56

#: 있을 수 없는 날짜의 바닥선. 이보다 이른 값은 날짜가 아니라 **자리표시자나 오타**다.
#:
#: 🔴 실측(2026-09-03 · 법인 3,142곳 40,573행)으로 셋이 갈렸다.
#:
#:     00010101   유가 51행 · 코스닥 42행 — 포털의 **"해당 없음"** 자리표시자.
#:                코스닥에만 상장한 회사의 `유가상장일` 자리에 온다
#:     11111111   1행 — 명백한 쓰레기값 ((주)케이씨씨 본부영업소)
#:     18970925   16행 — **진짜다.** 동화약품(1897-09-25 창업), 국내 최고령 등록법인
#:
#: 그래서 바닥선을 1800 으로 둔다. 앞의 둘을 버리고 동화약품은 살린다.
#: 1900 으로 올리면 동화약품이 사라지고, 0001 만 콕 집으면 `11111111` 이 남는다.
#:
#: 자리표시자를 그대로 두면 "1년에 상장한 회사" 가 생겨 상장 경과일 같은 계산이
#: 조용히 틀어진다. `None` 이면 그 칸이 비고, 비어 있는 것은 눈에 띈다.
_EARLIEST_PLAUSIBLE = "18000101"

#: 있을 수 없는 날짜의 천장. 이보다 늦은 값도 날짜가 아니라 **자리표시자**다.
#:
#: 🔴 바닥선만 있고 천장이 없었다. 2026-09-03 조사에서 반대쪽을 만났다 —
#:    `99991231` 이 "아직 폐지 안 됨" 의 표기다:
#:
#:     getDepoAvaiWhet_V2   lstgAbolDt              상장폐지일
#:     KSD getStkListInfoN1 xpitDt · dlistDt        만기일 · 상장폐지일
#:     KSD getIssucoBasicInfo custXtinDt            법인 소멸일
#:
#: 그대로 두면 "9999년에 상장폐지되는 회사" 가 생겨 상장 잔여기간 같은 계산이 에러 없이
#: 틀어진다. 2100 으로 둔 이유는, 진짜 미래 날짜(예정 상장일·만기일)는 그 안에 들고
#: 자리표시자는 항상 그 밖에 있기 때문이다. 두 자리 연도는 `_YY_PIVOT` 규칙상 2055 가
#: 최대라 천장에 닿을 수 없다.
_LATEST_PLAUSIBLE = "21001231"

#: `known_at` 을 어떤 규칙으로 냈는지. 규칙을 바꾸면 재수집해야 하므로 행마다 남긴다.
KNOWN_RULE_NEXT_SESSION = "basDt+1session"    # 종목 목록 — 계산값이다
KNOWN_RULE_OBSERVED = "fstOpegDt"             # 법인 개요 — 출처가 직접 말해 준다

_DATE_ONLY = re.compile(r"\D")


class DataGoKrError(RuntimeError):
    """포털 호출이 실패했다. **무엇을 해야 하는지까지** 문구에 담는다."""


#: 포털이 돌려주는 오류 코드별로 **할 일이 다르다.** 겉으로는 다 "안 된다" 지만
#: 키를 고쳐야 하는 것과 신청을 해야 하는 것과 내일 다시 와야 하는 것은 전혀 다르다.
_ERROR_HELP = {
    "SERVICE_KEY_IS_NOT_REGISTERED_ERROR": (
        "서비스키가 등록되지 않았다.\n"
        "  흔한 원인: Encoding 키(%3D 포함)를 urlencode 에 한 번 더 넣어 %253D 가 됐다.\n"
        "  할 일: .env 의 DATA_GO_KR_API_KEY 를 확인하고, python scripts/check_keys.py --live"
    ),
    "SERVICE_ACCESS_DENIED_ERROR": (
        "키는 맞는데 이 서비스에 **활용신청**이 없다.\n"
        "  할 일: data.go.kr 에서 해당 오픈API 활용신청을 한다 (승인은 보통 즉시)."
    ),
    "NO_OPENAPI_SERVICE_ERROR": (
        "그런 경로의 서비스가 없다 — **엔드포인트가 틀렸다.**\n"
        "  흔한 원인: 경로 틀이 둘이다. 구세대는 1160100/service/…, 신세대는 1160100/… 이고\n"
        "            이름으로는 구별이 안 된다. 주식발행정보는 _V3 다.\n"
        "  할 일: 이 모듈의 EP_* 상수를 확인한다. 서비스명 대소문자까지 맞아야 한다."
    ),
    "LIMITED_NUMBER_OF_SERVICE_REQUESTS_EXCEEDS_ERROR": (
        f"하루 호출 한도({DAILY_LIMIT:,}회)를 넘었다.\n"
        "  할 일: 내일 이어 받는다. common.budget 이 먼저 멈추게 하려면 예산을 등록한다."
    ),
    "HTTP_ERROR": (
        "요청 형식이 잘못됐다 (파라미터 이름·값).\n"
        "  할 일: 필수 파라미터가 빠지지 않았는지 본다."
    ),
}


# ==================================================
# 1. 키
# ==================================================
def load_key() -> Tuple[str, str]:
    """키와 그 출처를 돌려준다. 없으면 `("", "")`."""
    return secrets.load_key(KEY_NAMES)


def available() -> bool:
    return bool(load_key()[0])


# ==================================================
# 2. 값 정규화 — 원문이 고르지 않다
# ==================================================
def normalize_date(raw: Optional[str]) -> Optional[str]:
    """포털이 주는 날짜를 `YYYYMMDD` 로 맞춘다. 빈 값·해석 불가는 `None`.

    한 응답 안에서 세 모양이 섞여 온다 (실측):

        18970925      YYYYMMDD      enpEstbDt
        76/03/24      YY/MM/DD      enpXchgLstgDt   ← 두 자리 연도
        2025/08/07    YYYY/MM/DD    fssCorpChgDtm

    구분자를 떼고 자릿수로 가른다. **6자리는 두 자리 연도**이고 `_YY_PIVOT` 으로 푼다.

    🔴 못 읽는 값을 지어내지 않는다. `None` 을 주면 그 칸이 비고, 비어 있는 것은
       눈에 띈다. 아무 값이나 넣으면 그 뒤로 아무도 못 찾는다.

    🔴 **`00010101` 은 날짜가 아니라 "해당 없음" 이다.** 포털은 코스닥에만 상장한
       회사의 `유가상장일` 자리에 이 값을 넣어 준다. 여덟 자리라 모양은 멀쩡해서
       그냥 두면 *"서기 1년에 상장한 회사"* 가 생기고, 상장 경과일 같은 계산이
       에러 없이 틀어진다. `_EARLIEST_PLAUSIBLE` 로 거른다.

    🔴 **`99991231` 도 날짜가 아니라 "아직 폐지 안 됨" 이다.** 반대쪽 자리표시자라
       `_LATEST_PLAUSIBLE` 로 거른다. 바닥선만 두면 이쪽이 통과해 "9999년에 폐지되는
       회사" 가 생긴다.
    """
    if raw is None:
        return None
    숫자 = _DATE_ONLY.sub("", str(raw).strip())
    if not 숫자:
        return None
    if len(숫자) == 8:
        return 숫자 if _EARLIEST_PLAUSIBLE <= 숫자 <= _LATEST_PLAUSIBLE else None
    if len(숫자) == 6:
        yy = int(숫자[:2])
        세기 = 1900 if yy >= _YY_PIVOT else 2000
        return f"{세기 + yy:04d}{숫자[2:]}"   # 두 자리 연도는 바닥선에 걸릴 수 없다
    return None                      # 4자리·10자리 등 — 날짜로 볼 수 없다


def normalize_int(raw: Optional[str]) -> Optional[int]:
    """`'838'` → `838`. 빈 값·숫자 아님은 `None`.

    ⚠️ `'0'` 은 `0` 으로 남긴다 — "0명" 과 "모른다" 는 다른 사실이다.

    🔴 **그런데 이 출처에서는 그 둘을 구별할 수 없다** (2026-09-03 실측 · 이슈 #93).

        `corp_profile` 40,569행 중 종업원이 있는데 평균급여가 0인 행 **2,760**
        예: 롯데쇼핑(주)롯데마트사업본부 — 종업원 8,542명 · 1인평균급여 0원

    두 가지를 확인하고도 원칙을 안 바꿨다.

      1. **활용자 가이드에 규약이 없다.** `금융위원회_기업기본정보_활용자가이드.docx`
         (2.13MB · 229줄) 전문에서 `결측`·`미기재`·`공란`·`0으로` 를 찾았는데 한 줄도
         없다. 응답 명세표의 5번째 칸은 규약이 아니라 그냥 샘플데이터다.
      2. **원문이 숫자 칸을 절대 비우지 않는다.** `getCorpOutline_V2` 를 XML 로 1,000행
         받아 `<tag/>`·`<tag></tag>`·`<tag>0</tag>` 를 세니 **빈 값이 0건**이었다.
         법인 1,000곳 전부가 종업원 수를 신고했을 리 없다 — 출처에게 숫자 칸에서
         "모른다" 를 적을 방법이 `0` 밖에 없다는 뜻이다.

    즉 `0` 은 "0원/0명" 과 "미기재" 를 **겸한다.** `None` 으로 바꾸면 지주회사·
    페이퍼컴퍼니의 진짜 0을 잃으므로 여기서는 그대로 두고, **쓰는 쪽에서 막는다** —
    `empe_cnt > 0 AND pn1_avg_slry_amt > 0` 인 행만 쓰고 몇 행이 빠졌는지 함께 남긴다.
    지금은 둘 다 피처가 아니다(`export_team_dataset.FEATURE_COLUMNS` 에 없다).
    """
    if raw is None:
        return None
    글자 = str(raw).strip()
    if not 글자:
        return None
    try:
        return int(글자)
    except ValueError:
        return None


def strip_code_prefix(srtn_cd: Optional[str]) -> Optional[str]:
    """`'A000020'` → `'000020'`.

    🔴 이걸 빠뜨리면 `daily_price.code` 와 조인이 **0행**이 된다. 그리고 조인은 0행이어도
       에러가 나지 않는다 — 그냥 아무것도 안 나온다. 그래서 여기서 못 박는다.

    실측(2026-09-03) 2,728종 **전부** `A` + 6자였다. 그래도 접두사가 없는 경우를
    받아 주는 이유는, 규격이 조용히 바뀌었을 때 전량을 격리하는 것보다 낫기 때문이다.
    """
    if not srtn_cd:
        return None
    코드 = str(srtn_cd).strip()
    if len(코드) == 7 and 코드[0].isalpha():
        코드 = 코드[1:]
    return 코드 or None


# ==================================================
# 3. 호출 — 인증 · 오류 해석 · 페이징
# ==================================================
def _build_url(path: str, key: str, params: Dict) -> str:
    """🔴 **서비스키는 잇고, 나머지만 인코딩한다.**

    Encoding 키에는 `%3D` 가 들어 있다. `urlencode({"serviceKey": key, ...})` 를 쓰면
    `%` 자체가 다시 인코딩돼 `%253D` 가 되고 포털이 다른 키로 읽는다. 그때 돌아오는
    문구는 "키가 없다" 가 아니라 "등록되지 않은 키" 라, 키를 새로 발급받는 헛수고를
    하게 된다. 그래서 이 함수 하나로 조립 방식을 못 박는다.
    """
    나머지 = {k: v for k, v in params.items() if v is not None}
    나머지.setdefault("resultType", "json")
    return f"{BASE_URL}/{path}?serviceKey={key}&" + urlencode(나머지)


def _explain(result_code: str, result_msg: str, path: str) -> str:
    """포털 오류 코드를 **할 일**로 바꾼다. 막다른 길로 만들지 않는다."""
    도움 = _ERROR_HELP.get(result_code)
    if 도움 is None:
        # 코드 이름이 조금씩 달라지는 경우가 있어 부분 일치도 본다.
        for 코드, 글 in _ERROR_HELP.items():
            if 코드.split("_")[0] in result_code:
                도움 = 글
                break
    머리 = f"공공데이터포털 {path} 실패 — {result_code}: {result_msg}"
    return f"{머리}\n  {도움}" if 도움 else (
        f"{머리}\n  할 일: 포털의 오류코드 표를 확인한다."
    )


def _request(path: str, key: str, params: Dict) -> Dict:
    """한 번 호출해 JSON 을 돌려준다. 실패는 전부 `DataGoKrError` 로 세운다.

    🔴 **부르기 전에 예산을 센다.** 부르고 나서 세면, 응답을 못 받고 죽었을 때 이미
       나간 호출이 장부에 안 남아 한도를 넘겨 쓰게 된다. 포털이 한도 초과를 알려 줄
       때는 이미 늦었다 — 그날은 더 못 받는다.
    """
    if not budget.try_spend(BUDGET_SOURCE, 1, limit=DAILY_LIMIT):
        raise DataGoKrError(
            f"오늘 호출 예산({DAILY_LIMIT:,}회)을 다 썼다 — 부르기 전에 멈춘다.\n"
            "  할 일: 내일 이어 받는다. 어디까지 받았는지는 collect_log 에 있어\n"
            "        같은 명령을 다시 돌리면 받은 데를 건너뛴다."
        )
    url = _build_url(path, key, params)
    try:
        req = Request(url, headers={"Accept": "application/json"})
        with urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            본문 = resp.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        raise DataGoKrError(
            f"공공데이터포털 {path} 가 HTTP {exc.code} 로 답했다.\n"
            "  할 일: 파라미터 이름·값을 확인한다. 500 이면 잠시 뒤 다시 시도한다."
        ) from exc
    except URLError as exc:
        raise DataGoKrError(
            f"공공데이터포털 {path} 에 닿지 못했다: {exc.reason}\n"
            "  할 일: 네트워크를 확인한다. 이 수집원이 없어도 시세·재무는 돈다."
        ) from exc

    try:
        본문json = json.loads(본문)
    except json.JSONDecodeError:
        # 🔴 오류가 XML 로 오는 경우가 있다 (특히 키·경로 문제). 본문을 잘라 보여 준다 —
        #    "JSON 이 아니다" 만 말하면 무엇이 잘못됐는지 알 수 없다.
        raise DataGoKrError(
            f"공공데이터포털 {path} 가 JSON 이 아닌 것을 돌려줬다.\n"
            f"  받은 것(앞 300자): {본문[:300]}\n"
            "  할 일: 대개 키 또는 경로 문제다. XML 안의 <returnAuthMsg> 를 읽는다."
        ) from None

    머리 = 본문json.get("response", {}).get("header", {})
    코드 = str(머리.get("resultCode", ""))
    if 코드 not in ("00", "0", ""):
        raise DataGoKrError(_explain(코드, str(머리.get("resultMsg", "")), path))
    return 본문json


def _items(payload: Dict) -> List[Dict]:
    """응답에서 항목 목록을 꺼낸다.

    ⚠️ 항목이 **하나면 리스트가 아니라 객체**로 온다 (공공데이터포털 공통 함정).
       리스트로 단정하면 한 건짜리 날짜에서 칸 이름들이 한 글자씩 잘려 나온다.
    """
    바디 = payload.get("response", {}).get("body", {})
    항목 = (바디.get("items") or {}).get("item")
    if 항목 is None:
        return []
    return 항목 if isinstance(항목, list) else [항목]


def _total(payload: Dict) -> int:
    바디 = payload.get("response", {}).get("body", {})
    try:
        return int(바디.get("totalCount") or 0)
    except (TypeError, ValueError):
        return 0


def iter_pages(path: str, key: str, params: Dict,
               *, page_size: int = PAGE_SIZE,
               max_pages: int = 100) -> Iterator[List[Dict]]:
    """페이지를 돌며 항목 묶음을 하나씩 내놓는다.

    `max_pages` 는 폭주 방지다. 하루 2,700종이면 3페이지라 100 은 넉넉하지만, 응답이
    이상해져 `totalCount` 가 커지면 한도를 다 태울 수 있다.

    ⚠️ **총건수만 믿고 페이지 수를 계산하지 않는다.** 빈 페이지가 오면 거기서 멈춘다 —
       `totalCount` 가 실제 행보다 크면 무한히 돌기 때문이다. 반대로 총건수에 닿으면
       한 번 더 부르지 않는다 — 그 한 콜이 날마다 쌓이면 한도가 된다.
    """
    페이지 = 1
    받은수 = 0
    총 = None
    while 페이지 <= max_pages:
        payload = _request(path, key, {**params, "numOfRows": page_size,
                                       "pageNo": 페이지})
        항목 = _items(payload)
        if 총 is None:
            총 = _total(payload)
        if not 항목:
            break
        yield 항목
        받은수 += len(항목)
        if 총 and 받은수 >= 총:
            break
        페이지 += 1


def page_count(path: str, key: str, params: Dict) -> int:
    """받기 전에 **몇 페이지인지**만 확인한다 (1콜).

    대량 수집 전에 한도를 넘는지 미리 알고 싶을 때 쓴다. 절반쯤 받다 멈추면
    어디까지 받았는지 맞추는 일이 생긴다.
    """
    payload = _request(path, key, {**params, "numOfRows": 1, "pageNo": 1})
    총 = _total(payload)
    return -(-총 // PAGE_SIZE)          # 올림 나눗셈


# ==================================================
# 4. KRX상장종목정보 → stock_identity
# ==================================================
#: 응답 칸 → 우리 칸. 실측한 7칸이 전부다 (2026-09-03).
_LISTED_MAP = {
    "isinCd": "isin_cd",
    "crno": "crno",
    "corpNm": "corp_nm",
    "itmsNm": "item_nm",
    "mrktCtg": "market",
}


def normalize_crno(raw: Optional[str]) -> Optional[str]:
    """법인등록번호를 정리한다. 자리표시자는 `None`.

    🔴 **`0000000000000` 은 번호가 아니라 "없음" 이다.** 외국기업 20종이 이 값을
    함께 쓴다(실측 2026-09-03 · 3,409행). 그대로 두면 `corp_profile` 과 조인할 때
    **서로 다른 20개 회사가 같은 법인 하나에 붙는다** — 실제로 그 번호로 받아 둔
    법인 개요에 헝셩그룹유한회사와 자프코 아시아 테크놀러지 펀드 3 이 섞여 있었다.

    조인이 0행이 되는 실수는 결과가 비어서 눈에 띈다. 그런데 **틀린 짝이 붙는
    실수는 결과가 그럴듯해서 안 보인다.** 이쪽이 더 비싸다.

    다음으로 많이 공유되는 번호는 2종(키움증권 보통주·우선주)으로 정상이다.
    20종이 한 번호를 쓰는 것은 자리표시자 말고는 설명이 안 된다.
    """
    if raw is None:
        return None
    값 = str(raw).strip()
    if not 값 or not 값.isdigit() or set(값) == {"0"}:
        return None
    return 값


def parse_listed_row(item: Dict, *, known_at: str) -> Dict:
    """응답 한 줄을 `stock_identity` 한 행으로 바꾼다.

    `known_at` 은 부르는 쪽이 준다 — 거래일 달력을 알아야 계산할 수 있는데, 이 모듈은
    저장소를 모르는 외부 연동 계층이라 달력을 직접 읽지 않는다 (계층 분리).
    """
    코드 = strip_code_prefix(item.get("srtnCd"))
    행 = {
        "bas_dd": str(item.get("basDt") or "").strip(),
        "code": 코드,
        "known_at": known_at,
        "known_rule": KNOWN_RULE_NEXT_SESSION,
    }
    for 원, 우리 in _LISTED_MAP.items():
        값 = item.get(원)
        행[우리] = str(값).strip() if 값 is not None and str(값).strip() else None
    # 🔴 `crno` 만 한 번 더 거른다 — 자리표시자가 붙으면 조인이 **틀린 짝**을 만든다.
    행["crno"] = normalize_crno(행.get("crno"))
    return 행


def fetch_listed(bas_dd: str, *, key: Optional[str] = None,
                 known_at: str) -> List[Dict]:
    """한 기준일의 상장종목 전체. 페이징을 안에서 처리한다.

    🔴 `EARLIEST_BAS_DD` 보다 이르면 **호출하지 않고 빈 목록**을 준다. 2019 이전은
       전부 0건이라(실측), 모르고 훑으면 한도만 태운다.
    """
    키 = key or load_key()[0]
    if not 키:
        raise DataGoKrError(
            "DATA_GO_KR_API_KEY 가 없다.\n"
            "  할 일: .env 에 넣는다. 발급 절차는\n"
            "        docs/데이터파트/version3.2/API키_발급_가이드.md"
        )
    if bas_dd < EARLIEST_BAS_DD:
        return []

    행들: List[Dict] = []
    for 항목 in iter_pages(EP_LISTED, 키, {"basDt": bas_dd}):
        행들.extend(parse_listed_row(it, known_at=known_at) for it in 항목)
    return 행들


# ==================================================
# 5. 기업기본정보 → corp_profile
# ==================================================
#: 응답 칸 → 우리 칸. 🔴 `enp` 접두사가 붙은 이름이 많다 — 설계 문서에 적힌 이름과
#: 다르므로 실측한 쪽을 따른다. 이름이 하나 틀리면 그 칸이 조용히 NULL 이 된다.
_PROFILE_TEXT = {
    "corpNm": "corp_nm",
    "sicNm": "sic_nm",
    "enpStacMm": "stac_mm",
    "audtRptOpnnCtt": "audt_rpt_opnn",
    "actnAudpnNm": "actn_audpn",
    "smenpYn": "smenp_yn",
}
_PROFILE_DATE = {
    "enpEstbDt": "estb_dt",
    "enpXchgLstgDt": "xchg_lstg_dt",
    "enpXchgLstgAbolDt": "xchg_lstg_abol_dt",
    "enpKosdaqLstgDt": "kosdaq_lstg_dt",
    "enpKosdaqLstgAbolDt": "kosdaq_lstg_abol_dt",
}
_PROFILE_INT = {
    "enpEmpeCnt": "empe_cnt",
    "enpPn1AvgSlryAmt": "pn1_avg_slry_amt",
}


def parse_profile_row(item: Dict) -> Optional[Dict]:
    """응답 한 줄을 `corp_profile` 한 행으로. 키가 될 값이 없으면 `None`.

    `known_at` 을 **계산하지 않는다** — `fstOpegDt` 가 곧 "이 값이 언제부터 유효했나"다.
    거시(ECOS)에서는 발표일을 안 줘서 계산할 수밖에 없었고 그래서 규칙을 바꾸면
    재수집해야 하는 짐이 남았다. 여기서는 출처가 직접 말해 주므로 그 짐이 없다.
    """
    crno = str(item.get("crno") or "").strip()
    fst = normalize_date(item.get("fstOpegDt"))
    if not crno or not fst:
        # 🔴 키가 없는 행을 억지로 넣지 않는다. 넣으면 다음 수집이 같은 행을 또 넣거나,
        #    빈 키끼리 충돌해 서로를 덮어쓴다. 부르는 쪽이 격리하도록 None 을 준다.
        return None

    행 = {
        "crno": crno,
        "fst_opeg_dt": fst,
        "last_opeg_dt": normalize_date(item.get("lastOpegDt")),
        "known_at": fst,
        "known_rule": KNOWN_RULE_OBSERVED,
    }
    for 원, 우리 in _PROFILE_TEXT.items():
        값 = item.get(원)
        행[우리] = str(값).strip() if 값 is not None and str(값).strip() else None
    for 원, 우리 in _PROFILE_DATE.items():
        행[우리] = normalize_date(item.get(원))
    for 원, 우리 in _PROFILE_INT.items():
        행[우리] = normalize_int(item.get(원))
    return 행


def fetch_corp_profile(crno: str, *, key: Optional[str] = None) -> List[Dict]:
    """한 법인의 **전 이력**. 한 번 부르면 여러 행이 온다 (동화약품 17행 실측).

    날짜별로 반복해 부를 필요가 없다 — 법인 수만큼만 부르면 된다.
    """
    키 = key or load_key()[0]
    if not 키:
        raise DataGoKrError(
            "DATA_GO_KR_API_KEY 가 없다.\n"
            "  할 일: .env 에 넣는다. 발급 절차는\n"
            "        docs/데이터파트/version3.2/API키_발급_가이드.md"
        )
    행들: List[Dict] = []
    for 항목 in iter_pages(EP_CORP_OUTLINE, 키, {"crno": crno}):
        행들.extend(r for r in (parse_profile_row(it) for it in 항목) if r)
    return 행들


def estimate_calls(bas_dds: Sequence[str], *, per_day_rows: int = 2_800) -> int:
    """이만큼 받으면 호출을 몇 번 쓰는지 미리 센다.

    받기 **전에** 한도를 넘는지 알려면 필요하다 — 절반쯤 받다 멈추면 어디까지 받았는지
    맞추는 일이 생긴다. 한 날짜당 `ceil(행수 / PAGE_SIZE)` 콜이고, 종목 수는 2,334종
    (2020) ~ 2,728종(2025) 이라 2,800 으로 잡으면 넉넉하다.
    """
    쓸날 = [d for d in bas_dds if d >= EARLIEST_BAS_DD]
    페이지수 = -(-per_day_rows // PAGE_SIZE)      # 올림 나눗셈
    return len(쓸날) * 페이지수
