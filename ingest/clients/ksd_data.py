"""한국예탁결제원(KSD) 게이트웨이 연동 — 금융위 클라이언트와 **규약이 달라 따로** 둔다.

    https://apis.data.go.kr/B552481/CorpSvc/<오퍼레이션>       기업정보    20개
    https://apis.data.go.kr/B552481/StockSvc/<오퍼레이션>      주식정보    20개
    https://apis.data.go.kr/B552481/SecDepoStat/<오퍼레이션>   증권예탁통계 33개

같은 포털·같은 키인데 왜 `data_go_kr.py` 를 안 쓰나
------------------------------------------------
실측 2026-09-03. **셋 다 깨진 뒤에 알았다.**

1. 🔴 **XML 전용이다.** `resultType=json` 을 주면 안 된다. 금융위 쪽은 JSON 으로 받는다.
2. 🔴 **선언 안 된 파라미터를 거부한다.** 금융위 API 는 모르는 파라미터를 무시하는데 여기는
   `10 INVALID_REQUEST_PARAMETER_ERROR` 로 튕긴다. 습관적으로 붙인 `numOfRows`·`pageNo`
   때문에 **되는 조회가 전부 실패했다.** 그래서 오퍼레이션마다 받는 파라미터를 표
   (`OPERATIONS`)로 못 박고, 표 밖의 것은 **부르기 전에** 세운다 — 예산을 쓰고 나서
   알면 늦다.
3. 🔴 **`numOfRows` 상한이 200 이다** (200 ✅ · 500 ❌ · 1,000 ❌). 금융위는 1,000 이다.
   콜 수 계산이 5배 틀어진다.

다리 — 종목에서 발행회사번호로
----------------------------
거의 모든 조회의 열쇠가 `issucoCustno`(KSD 발행회사번호)다. 이름으로 찾는
`getIssucoCustnoByNm` 은 "삼성전자" 로 93건이 나오고 앞쪽이 전부 펀드라 쓸 수 없고,
`getIssucoCustnoByShortIsin` 은 종목코드를 어떤 모양으로 줘도(`005930`·`A005930`·
`KR7005930003`·`05930`·`5930`) INVALID 다.

**`StockSvc/getStkListInfoN1(isin=…)` 이 `issucoCustno` 를 준다** (삼성전자 → `593`).
ISIN 은 `stock_identity` 에 전 종목 있다.

무엇에 쓰나
----------
- `getIssucoStkQtyChgList` — 액면분할·무상증자·합병증자를 **사유코드와 발행일**로 준다.
  삼성전자 2018-05-04 액면분할이 그대로 나온다. 우리가 시세로 역산한 수정주가 chain 을
  **출처와 직접 대조**할 수 있다 (이슈 #51). 종목당 1콜.
- `getStkDistributionShareholderStatus` — 기준일(월별)마다 주주 유형별 지분. 외국인 지분율
  시계열은 우리에게 없던 자료다. 다만 **종목 × 기준일마다 1콜**이라 전 종목 전 구간이면
  약 24만 콜이다. 표본으로 값어치를 먼저 잰다.
- `getNewDepoSecnListN1` — 그 달 새로 예탁된 종목. 업종(`indtpNm`)이 있지만 **KSIC 대분류**
  이고 신규 상장분뿐이라 이슈 #92 의 답이 되지 못한다.

🔴 자리표시자 `99991231`
----------------------
"아직 폐지 안 됨" 을 9999-12-31 로 적는다 (`xpitDt`·`dlistDt`·`custXtinDt`). 그대로 두면
"9999년에 폐지되는 회사" 가 생겨 잔여기간 계산이 조용히 틀어진다. 날짜 칸은 전부
`data_go_kr.normalize_date` 를 거친다 — 거기 천장(`_LATEST_PLAUSIBLE`)이 있다.

책임 경계
--------
이 게이트웨이가 죽어도 시세·재무는 돈다. 실패는 `KsdError` 로 **할 일까지** 담아 세운다.
예산은 금융위와 **같은 통**(`data_go_kr`)을 쓴다 — 같은 키로 같은 포털을 두드린다.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from common import budget
from ingest.clients import data_go_kr
from ingest.clients.data_go_kr import normalize_date, normalize_int

#: 🔴 기관코드가 금융위(`1160100`)와 다르다. 별도 게이트웨이다.
BASE_URL = "https://apis.data.go.kr/B552481"
REQUEST_TIMEOUT = 30

#: 예산 통 — 금융위와 같다. 같은 키로 같은 포털을 두드리므로 한도가 하나다.
BUDGET_SOURCE = data_go_kr.BUDGET_SOURCE
DAILY_LIMIT = data_go_kr.DAILY_LIMIT
KEY_NAMES = data_go_kr.KEY_NAMES

#: 🔴 한 페이지 상한. 200 을 넘기면 값이 잘리는 게 아니라 **호출 자체가 거부된다**
#:    (`10 INVALID_REQUEST_PARAMETER_ERROR`). 실측 2026-09-03: 200 ✅ · 500 ❌ · 1,000 ❌.
PAGE_SIZE = 200

#: 페이징 파라미터. 🔴 Swagger 가 이 둘을 **선언한 오퍼레이션에만** 붙인다.
PAGING_PARAMS: Tuple[str, str] = ("pageNo", "numOfRows")

#: 시장 구분 코드 (`martTpcd` · `caltotMartTpcd`). Swagger 응답 설명에서 옮겼다.
MARKET_CODES: Dict[str, str] = {
    "11": "유가증권시장", "12": "코스닥시장", "13": "K-OTC시장",
    "14": "코넥스시장", "50": "기타비상장",
}


class KsdError(RuntimeError):
    """예탁결제원 호출이 실패했다. **무엇을 해야 하는지까지** 문구에 담는다."""


@dataclass(frozen=True)
class Operation:
    """오퍼레이션 하나가 **받는 파라미터의 전부.** 이 표 밖의 것은 부르기 전에 세운다.

    `paged` 가 참이면 `pageNo`·`numOfRows` 를 받는다. 거짓이면 붙이지 않는다 —
    붙이면 `10 INVALID` 로 튕긴다(2026-09-03 에 CorpSvc 전체가 그렇게 실패했다).
    """

    path: str
    required: Tuple[str, ...] = ()
    optional: Tuple[str, ...] = ()
    paged: bool = False

    @property
    def declared(self) -> frozenset:
        선언 = set(self.required) | set(self.optional)
        if self.paged:
            선언 |= set(PAGING_PARAMS)
        return frozenset(선언)


#: 실측으로 확인한 오퍼레이션 (2026-09-03 · 삼성전자 issucoCustno=593 으로 응답 본문까지 열었다).
#:
#: ⚠️ `CorpSvc` 의 Swagger 는 파라미터를 `get` 바깥에 두어 덤프에 "필수 없음" 으로 잡혔지만,
#:    실호출은 `issucoCustno` 없이는 돌지 않았다. 여기 적힌 것이 **실제로 통한 조합**이다.
#:    `CorpSvc` 는 페이징을 받지 않는다 — 발행회사 하나의 이력이라 200행을 넘지 않는다.
OPERATIONS: Dict[str, Operation] = {
    # ── StockSvc · 종목 축 · 페이징 있음 ─────────────────────────────
    # ISIN → 발행회사번호. 이 모듈의 **다리**다. 상장구분(`listTpcd`)마다 한 행이 온다.
    "stock_list": Operation("StockSvc/getStkListInfoN1", ("isin",), paged=True),
    # 시장별 단축코드 전량 (유가 943종). 구성종목이 아니라 **시장 전체**다 (#94).
    "codes_by_market": Operation("StockSvc/getShotnByMartN1", ("martTpcd",), paged=True),
    # 그 달 신규 예탁 종목. 업종 칸이 있지만 KSIC 대분류·신규분뿐이다 (#92).
    "new_deposits": Operation("StockSvc/getNewDepoSecnListN1", ("yyyymm",),
                              ("searchType", "issucoCustno"), paged=True),
    # ── CorpSvc · 발행회사 축 · 페이징 없음 ─────────────────────────
    "issuer_basic": Operation("CorpSvc/getIssucoBasicInfo", ("issucoCustno",)),
    # 🔴 주식수 변동 — 액면분할·합병증자를 사유코드·발행일로. 수정주가 chain 대조용.
    "issuer_stock_changes": Operation("CorpSvc/getIssucoStkQtyChgList", ("issucoCustno",)),
    "issuer_securities": Operation("CorpSvc/getSecnIssuInfoStock", ("issucoCustno",)),
    "issuer_rights_schedule": Operation("CorpSvc/getIssucoRgtSchedule", ("issucoCustno",)),
    # 주주분포 — 기준일 목록을 먼저 받고, 기준일마다 유형별 지분을 받는다.
    "distribution_dates": Operation("CorpSvc/getStkDistributionRgtStdDt", ("issucoCustno",)),
    "distribution_by_holder": Operation("CorpSvc/getStkDistributionShareholderStatus",
                                        ("issucoCustno", "rgtStdDt")),
    # KSIC 코드 → 이름 사전. 종목 매핑이 아니다 (#92).
    "industry_codes": Operation("CorpSvc/getKRIndstrClsfStndIndtpInfo"),
}

#: 포털 공통 오류 코드 → 할 일. 겉으로는 다 "안 된다" 지만 할 일이 다르다.
_ERROR_HELP: Dict[str, str] = {
    "10": ("선언 안 된 파라미터가 섞였거나 값 형식이 다르다 (INVALID_REQUEST_PARAMETER).\n"
           "  흔한 원인: 페이징 없는 오퍼레이션에 numOfRows·pageNo 를 붙였다 · numOfRows > 200\n"
           "  할 일: 이 모듈 OPERATIONS 표와 대조한다. 표가 틀렸으면 포털 Swagger 로 다시 잰다."),
    "11": ("필수 파라미터가 빠졌다.\n"
           "  할 일: OPERATIONS 의 required 를 확인한다."),
    "12": ("그런 경로의 오퍼레이션이 없다.\n"
           "  할 일: OPERATIONS 의 path 를 확인한다. 기관코드는 B552481 이다."),
    "20": ("키는 맞는데 이 서비스에 활용신청이 없다.\n"
           "  할 일: data.go.kr 에서 예탁결제원 GW 3종(기업정보·주식정보·증권예탁통계)을\n"
           "        활용신청한다. 2026-09-03 기준 우리 키는 셋 다 승인돼 있었다."),
    "22": (f"하루 호출 한도({DAILY_LIMIT:,}회)를 넘었다.\n"
           "  할 일: 내일 이어 받는다. common.budget 이 먼저 멈추도록 장부를 확인한다."),
    "30": ("서비스키가 등록되지 않았다.\n"
           "  흔한 원인: Encoding 키(%3D 포함)를 urlencode 에 한 번 더 넣어 %253D 가 됐다.\n"
           "  할 일: .env 의 DATA_GO_KR_API_KEY 를 확인하고, python scripts/check_keys.py --live"),
}

_DIGITS = re.compile(r"\d+")


# ==================================================
# 1. 키
# ==================================================
def load_key() -> Tuple[str, str]:
    """키와 그 출처. 금융위와 같은 키다."""
    return data_go_kr.load_key()


def available() -> bool:
    return bool(load_key()[0])


# ==================================================
# 2. 파라미터 검사 — 부르기 전에 세운다
# ==================================================
def operation(name: str) -> Operation:
    try:
        return OPERATIONS[name]
    except KeyError:
        raise KsdError(
            f"모르는 오퍼레이션이다: {name!r}\n"
            f"  아는 것: {', '.join(sorted(OPERATIONS))}\n"
            "  할 일: 새 오퍼레이션이면 OPERATIONS 표에 path·required·optional·paged 를 적는다."
        ) from None


def validate_params(op: Operation, params: Dict) -> Dict:
    """선언 밖 파라미터·빠진 필수값·200 초과 페이지를 **호출 전에** 잡는다.

    🔴 예산은 이 검사 **뒤에** 깎인다. 서버가 거절할 것이 뻔한 호출에 한도를 쓰지 않는다.
    """
    보낼것 = {k: v for k, v in params.items() if v is not None}
    선언밖 = sorted(set(보낼것) - op.declared)
    if 선언밖:
        raise KsdError(
            f"{op.path} 가 받지 않는 파라미터다: {선언밖}\n"
            f"  받는 것: {sorted(op.declared) or '없음'}\n"
            "  왜 세우나: 이 게이트웨이는 모르는 파라미터를 무시하지 않고 10 INVALID 로 거절한다.\n"
            "  할 일: 파라미터를 빼거나, 정말 받는 것이면 OPERATIONS 표를 고친다."
        )
    빠진것 = sorted(set(op.required) - set(보낼것))
    if 빠진것:
        raise KsdError(
            f"{op.path} 의 필수 파라미터가 빠졌다: {빠진것}\n"
            "  할 일: 값을 넣는다. issucoCustno 는 issuer_custno(isin) 으로 얻는다."
        )
    행수 = 보낼것.get("numOfRows")
    if 행수 is not None and int(행수) > PAGE_SIZE:
        raise KsdError(
            f"numOfRows={행수} — 상한은 {PAGE_SIZE} 이다 (실측 2026-09-03: 500·1,000 은 거절).\n"
            "  할 일: PAGE_SIZE 이하로 나눠 부른다. 콜 수는 estimate_calls() 로 미리 센다."
        )
    return 보낼것


def _build_url(op: Operation, key: str, params: Dict) -> str:
    """🔴 서비스키는 잇고 나머지만 인코딩한다 (`data_go_kr._build_url` 과 같은 이유).

    그리고 **`resultType` 을 붙이지 않는다** — XML 전용이라 `json` 을 주면 거절한다.
    """
    보낼것 = validate_params(op, params)
    질의 = urlencode(보낼것)
    return f"{BASE_URL}/{op.path}?serviceKey={key}" + (f"&{질의}" if 질의 else "")


# ==================================================
# 3. 호출 — XML · 오류 해석 · 페이징
# ==================================================
def _explain(code: str, msg: str, path: str) -> str:
    도움 = _ERROR_HELP.get(code.lstrip("0") or "0") or _ERROR_HELP.get(code)
    머리 = f"예탁결제원 {path} 실패 — {code}: {msg}"
    return f"{머리}\n  {도움}" if 도움 else f"{머리}\n  할 일: 포털의 오류코드 표를 확인한다."


def _is_nodata(code: str, msg: str) -> bool:
    """`03 NODATA_ERROR` — 오류가 아니라 **빈 결과**다. 코드가 `3`·`03` 로 섞여 온다."""
    return code.lstrip("0") == "3" or "NODATA" in (msg or "").upper()


def _parse(본문: str, path: str) -> ET.Element:
    try:
        return ET.fromstring(본문)
    except ET.ParseError:
        raise KsdError(
            f"예탁결제원 {path} 가 XML 이 아닌 것을 돌려줬다.\n"
            f"  받은 것(앞 300자): {본문[:300]}\n"
            "  할 일: 대개 키 또는 경로 문제다. HTML 이면 게이트웨이 점검 페이지일 수 있다."
        ) from None


def _request(op: Operation, key: str, params: Dict) -> Optional[ET.Element]:
    """한 번 호출해 XML 루트를 돌려준다. **빈 결과(NODATA)는 `None`** 이고 실패는 세운다.

    🔴 순서가 중요하다 — ① 파라미터 검사 ② 예산 ③ 호출. 서버가 거절할 호출에 예산을
       쓰지 않고, 응답을 못 받고 죽었을 때도 나간 호출은 장부에 남는다.
    """
    url = _build_url(op, key, params)                   # ① 여기서 먼저 세운다
    if not budget.try_spend(BUDGET_SOURCE, 1):          # ② 부르기 전에 깎는다
        raise KsdError(
            f"오늘 {BUDGET_SOURCE} 예산({DAILY_LIMIT:,}회)을 다 썼다 — 부르기 전에 멈춘다.\n"
            "  할 일: 내일 이어 받는다. 금융위 호출과 같은 통이다."
        )
    try:
        req = Request(url, headers={"Accept": "application/xml"})
        with urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            본문 = resp.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        raise KsdError(
            f"예탁결제원 {op.path} 가 HTTP {exc.code} 로 답했다.\n"
            "  할 일: 파라미터 값을 확인한다. 500 이면 잠시 뒤 다시 시도한다."
        ) from exc
    except URLError as exc:
        raise KsdError(
            f"예탁결제원 {op.path} 에 닿지 못했다: {exc.reason}\n"
            "  할 일: 네트워크를 확인한다. 이 수집원이 없어도 시세·재무는 돈다."
        ) from exc

    루트 = _parse(본문, op.path)
    코드 = (루트.findtext("header/resultCode") or "").strip()
    문구 = (루트.findtext("header/resultMsg") or "").strip()
    if 코드 in ("00", "0"):
        return 루트
    if _is_nodata(코드, 문구):
        return None
    raise KsdError(_explain(코드, 문구, op.path))


def _items(루트: Optional[ET.Element]) -> List[Dict[str, Optional[str]]]:
    """`<item>` 들을 `{칸: 값}` 목록으로. 빈 칸은 `None`. 루트가 없으면 빈 목록."""
    if 루트 is None:
        return []
    return [{c.tag: ((c.text or "").strip() or None) for c in it}
            for it in 루트.findall("body/items/item")]


def _total(루트: Optional[ET.Element]) -> int:
    if 루트 is None:
        return 0
    try:
        return int((루트.findtext("body/totalCount") or "0").strip())
    except ValueError:
        return 0


def fetch(name: str, params: Optional[Dict] = None, *, key: Optional[str] = None,
          page_size: int = PAGE_SIZE, max_pages: int = 100) -> List[Dict[str, Optional[str]]]:
    """오퍼레이션 하나를 끝까지 받는다. 페이징을 받는 것이면 페이지를 넘긴다.

    ⚠️ 총건수만 믿고 페이지 수를 계산하지 않는다 — 빈 페이지가 오면 멈추고, 총건수에
       닿으면 한 번 더 부르지 않는다 (`data_go_kr.iter_pages` 와 같은 규약).
    """
    op = operation(name)
    키 = key or load_key()[0]
    if not 키:
        raise KsdError(
            "DATA_GO_KR_API_KEY 가 없다 (예탁결제원도 같은 키다).\n"
            "  할 일: .env 에 넣는다. 발급 절차는 docs/데이터파트 최신판의 API키_발급_가이드.md"
        )
    params = dict(params or {})
    if not op.paged:
        return _items(_request(op, 키, params))

    행들: List[Dict[str, Optional[str]]] = []
    총 = None
    for 페이지 in range(1, max_pages + 1):
        루트 = _request(op, 키, {**params, "numOfRows": page_size, "pageNo": 페이지})
        묶음 = _items(루트)
        if 총 is None:
            총 = _total(루트)
        if not 묶음:
            break
        행들.extend(묶음)
        if 총 and len(행들) >= 총:
            break
    return 행들


def estimate_calls(rows: int, *, page_size: int = PAGE_SIZE) -> int:
    """행 수로 콜 수를 미리 센다. 🔴 상한이 200 이라 금융위(1,000)보다 5배 든다."""
    if rows <= 0:
        return 0
    return -(-rows // min(page_size, PAGE_SIZE))     # 올림 나눗셈


# ==================================================
# 4. 다리 · 파싱 — 값을 우리 모양으로
# ==================================================
def _digits(raw: Optional[str]) -> Optional[int]:
    """`'   6,648,649,811 주'` → `6648649811`. 숫자가 없으면 `None`.

    `getIssucoBasicInfo.totalStkCnt` 가 이렇게 **쉼표와 단위를 붙여** 온다. 같은 뜻의
    `getSecnIssuInfoStock.totalStkCnt` 는 맨 숫자다 — 한 출처 안에서도 표기가 다르다.
    """
    if raw is None:
        return None
    조각 = _DIGITS.findall(str(raw))
    return int("".join(조각)) if 조각 else None


def _ratio(raw: Optional[str]) -> Optional[float]:
    """`'.02'` · `'45.17'` → float. 앞자리 0 이 생략돼 온다."""
    if raw is None or not str(raw).strip():
        return None
    try:
        return float(str(raw).strip())
    except ValueError:
        return None


def issuer_custno(isin: str, *, key: Optional[str] = None) -> Optional[str]:
    """ISIN → `issucoCustno`. 없으면 `None`. **다리는 이 함수 하나다.**

    한 ISIN 에 상장구분(`listTpcd`)마다 한 행이 오지만 발행회사번호는 하나다.
    둘 이상이면 우리가 모르는 상황이므로 지어내지 않고 세운다.
    """
    번호 = {r.get("issucoCustno") for r in fetch("stock_list", {"isin": isin}, key=key)
            if r.get("issucoCustno")}
    if not 번호:
        return None
    if len(번호) > 1:
        raise KsdError(
            f"ISIN {isin} 에 발행회사번호가 여럿이다: {sorted(번호)}\n"
            "  할 일: 응답 원문을 열어 본다. 한 종목이 두 발행회사일 수는 없다."
        )
    return 번호.pop()


def parse_qty_change(item: Dict) -> Dict:
    """`getIssucoStkQtyChgList` 한 줄 → 우리 칸. 날짜는 천장·바닥을 거친다."""
    return {
        "issu_dt": normalize_date(item.get("issuDt")),
        "reason_code": item.get("secnIssuRacd"),
        "reason_nm": item.get("secnIssuRacdNm"),
        "issu_qty": normalize_int(item.get("issuQty")),
        "list_dt": normalize_date(item.get("listDt")),
    }


def stock_qty_changes(issuco_custno: str, *, key: Optional[str] = None) -> List[Dict]:
    """발행회사의 주식수 변동 이력 — 액면분할·무상증자·합병증자가 사유와 함께 온다.

    삼성전자(593) 실측: `20180503 201 액면분할 6,419,324,700 → 상장 20180504`.
    수정주가 chain(`common/corporate_actions.py`)을 출처와 대조하는 자료다.
    """
    return [parse_qty_change(r)
            for r in fetch("issuer_stock_changes", {"issucoCustno": issuco_custno}, key=key)]


def parse_basic_info(item: Dict) -> Dict:
    """`getIssucoBasicInfo` 한 줄 → 우리 칸. 🔴 `dlistDt`·`custXtinDt` 의 `99991231` 은 `None`."""
    return {
        "issuco_custno": item.get("issucoCustno"),
        "code": item.get("shotnIsin"),
        "name": item.get("repSecnNm"),
        "bizno": item.get("bizno"),
        "ceo": item.get("ceoNm"),
        "found_dt": normalize_date(item.get("founDt")),
        "market_code": item.get("caltotMartTpcd"),
        "market": MARKET_CODES.get(item.get("caltotMartTpcd") or "", item.get("caltotMartTpcdNm")),
        "list_dt": normalize_date(item.get("apliDt")),
        "delist_dt": normalize_date(item.get("dlistDt")),
        "extinct_dt": normalize_date(item.get("custXtinDt")),
        "par_value": normalize_int(item.get("pval")),
        "total_shares": _digits(item.get("totalStkCnt")),
        "electronic": item.get("eltscYn"),
    }


def issuer_basic_info(issuco_custno: str, *, key: Optional[str] = None) -> Optional[Dict]:
    행 = fetch("issuer_basic", {"issucoCustno": issuco_custno}, key=key)
    return parse_basic_info(행[0]) if 행 else None


def distribution_dates(issuco_custno: str, *, key: Optional[str] = None) -> List[str]:
    """주주분포 기준일 목록 (최근 우선 정렬 · 월별). 삼성전자 실측 81개(20260731 까지)."""
    return [d for d in (normalize_date(r.get("rgtStdDt"))
                        for r in fetch("distribution_dates", {"issucoCustno": issuco_custno},
                                       key=key)) if d]


def parse_holder_row(item: Dict) -> Dict:
    return {
        "holder_type": item.get("stkDistbutTpnm"),
        "holders": normalize_int(item.get("shrs")),
        "holders_ratio": _ratio(item.get("shrsRatio")),
        "shares": normalize_int(item.get("stkqty")),
        "shares_ratio": _ratio(item.get("stkqtyRatio")),
    }


def shareholder_distribution(issuco_custno: str, rgt_std_dt: str, *,
                             key: Optional[str] = None) -> List[Dict]:
    """한 기준일의 주주 유형별 분포. 삼성전자 20260731: 외국인 45.17% · 개인 22.60%.

    🔴 **종목 × 기준일마다 1콜**이다. 전 종목 × 81기준일이면 24만 콜 — 표본으로 먼저 잰다.
    """
    return [parse_holder_row(r)
            for r in fetch("distribution_by_holder",
                           {"issucoCustno": issuco_custno, "rgtStdDt": rgt_std_dt}, key=key)]
