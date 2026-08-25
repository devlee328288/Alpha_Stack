"""KOSIS OpenAPI 호출 + 차트용 정규화 (외부 연동 계층)

국가통계포털(KOSIS) OpenAPI 를 호출하고, 응답을 **화면이 그대로 그릴 수 있는 모양**으로
바꿔 준다. 화면(`static/pages/kosis.html`)은 계산하지 않고 받은 값을 그리기만 한다.

KOSIS OpenAPI 의 특징 (실제 호출로 확인, 2026-07)
------------------------------------------------
1. **에러도 HTTP 200 으로 온다.** 본문이 `{"err":"11","errMsg":"유효하지않은 인증KEY입니다."}`
   같은 객체면 실패, 정상이면 배열(`[...]`)이다. 상태 코드만 봐서는 성공을 판단할 수 없다.
2. **연속 호출하면 연결을 끊는다.** 짧은 간격으로 3번 부르면 `Connection reset by peer` 가
   난다. 그래서 재시도 간격을 두고 다시 시도한다.
3. 값(`DT`)은 문자열이고 결측이면 `-` 나 빈 문자열이 온다.
4. 기간(`PRD_DE`)은 주기에 따라 `2024` / `202406` / `20240601` 로 자릿수가 다르다.

엔드포인트 3종
--------------
| 함수 | KOSIS 경로 | 쓰임 |
|---|---|---|
| `search_tables` | `statisticsSearch.do` | 통계표 검색 (1단계 — 무엇을 볼지 고른다) |
| `fetch_meta` | `statisticsData.do?method=getMeta` | 항목(ITM)·기간(PRD) 목록 (2단계 — 파라미터 조립) |
| `fetch_data` | `Param/statisticsParameterData.do` | 실제 통계 수치 (3단계 — 그린다) |
"""

from __future__ import annotations

import json
import time
from typing import Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from common import secrets  # 인증키 로딩 (공통)

KOSIS_BASE_URL = "https://kosis.kr/openapi"
REQUEST_TIMEOUT = 25                               # KOSIS 응답 대기 시간(초)
RETRY_DELAYS = (1.5, 3.0)                          # 연결이 끊겼을 때 다시 시도할 간격(초)

# 환경변수·파일에서 찾아볼 키 이름들 (앞에 있는 것이 우선)
KEY_NAMES = ("KOSIS_API_KEY", "KOSIS_APIKEY")

# KOSIS 는 값만 한 줄 적는 형식을 지원하지 않는다.
# `.key` 에 KRX 키도 함께 있으므로 이름이 붙은 줄만 인정해야 서로 섞이지 않는다.
ALLOW_BARE_KEY = False

# 기간 주기 코드 → 한국어 이름. 화면의 선택지로도 쓴다.
PERIOD_TYPES = {
    "Y": "년", "H": "반기", "Q": "분기", "M": "월", "D": "일",
}

# 브라우저처럼 보이게 한다 (기본 파이썬 UA 는 막히는 경우가 있다).
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
}


class KosisError(Exception):
    """KOSIS 가 돌려준 오류. `code` 는 KOSIS 의 `err` 값이다.

    라우터가 이 예외를 잡아 적절한 HTTP 상태 코드로 바꾼다.
    """

    def __init__(self, message: str, code: str = "", status: int = 502):
        super().__init__(message)
        self.code = code
        self.status = status


# ==================================================
# 1. 인증키 · 호출
# ==================================================
def load_kosis_key() -> Tuple[str, str]:
    """KOSIS 인증키와 그 출처를 돌려준다. 못 찾으면 `("", "none")`."""
    return secrets.load_key(KEY_NAMES, allow_bare=ALLOW_BARE_KEY)


def build_url(path: str, params: Dict[str, str], key: str = "") -> str:
    """호출할 전체 URL 을 만든다. `key` 가 비어 있으면 `apiKey` 는 빼고 만든다.

    화면에 "이런 요청이 나갑니다" 를 보여줄 때는 키 없이 만들어 노출을 막는다.
    """
    query = {k: v for k, v in params.items() if v not in (None, "")}
    if key:
        query["apiKey"] = key
    return f"{KOSIS_BASE_URL}/{path}?{urlencode(query)}"


def call(path: str, params: Dict[str, str]) -> Tuple[object, int]:
    """KOSIS 를 호출해 (파싱된 JSON, 소요 밀리초) 를 돌려준다.

    실패는 전부 `KosisError` 로 통일한다. 화면이 오류 종류를 구분할 필요가 없기 때문이다.
    """
    key, _ = load_kosis_key()
    if not key:
        raise KosisError(
            "KOSIS 인증키가 없습니다. `.key` 파일에 KOSIS_API_KEY 를 넣거나 "
            "환경변수로 설정하세요. (https://kosis.kr/openapi 에서 발급)",
            code="no-key", status=503,
        )

    url = build_url(path, params, key)
    started = time.time()
    last_error: Optional[Exception] = None

    # KOSIS 는 연속 호출 시 연결을 끊는다. 간격을 두고 다시 시도한다.
    for attempt in range(len(RETRY_DELAYS) + 1):
        try:
            with urlopen(Request(url, headers=HEADERS), timeout=REQUEST_TIMEOUT) as response:
                raw = response.read().decode("utf-8", errors="replace")
            break
        except HTTPError as error:
            raise KosisError(f"KOSIS 가 HTTP {error.code} 를 돌려줬습니다.", status=502) from error
        except (URLError, TimeoutError, ConnectionError) as error:
            last_error = error
            if attempt < len(RETRY_DELAYS):
                time.sleep(RETRY_DELAYS[attempt])
    else:
        raise KosisError(f"KOSIS 에 연결하지 못했습니다: {last_error}", status=504)

    elapsed_ms = int((time.time() - started) * 1000)

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise KosisError(
            f"KOSIS 응답을 JSON 으로 읽지 못했습니다: {raw[:120]}", status=502
        ) from error

    # KOSIS 는 오류도 200 으로 준다. 본문이 객체면 오류다.
    if isinstance(payload, dict) and "err" in payload:
        code = str(payload.get("err", ""))
        message = payload.get("errMsg", "KOSIS 오류")
        # 인증키 문제(11)는 503, 없는 통계표(21)·빈 결과(30)는 404 로 구분해 준다.
        status = {"11": 503, "20": 503, "21": 404, "30": 404}.get(code, 502)
        raise KosisError(f"{message} (err={code})", code=code, status=status)

    if not isinstance(payload, list):
        raise KosisError(f"예상과 다른 응답 형태입니다: {type(payload).__name__}", status=502)

    return payload, elapsed_ms


def get_status() -> Dict:
    """인증키 로딩 상태. **키 값 자체는 절대 내보내지 않는다.**"""
    key, source = load_kosis_key()
    return {
        "key_loaded": bool(key),
        "key_source": source,
        "key_length": len(key),
        "key_masked": secrets.mask(key),
        "base_url": KOSIS_BASE_URL,
        "period_types": [{"code": c, "name": n} for c, n in PERIOD_TYPES.items()],
    }


# ==================================================
# 2. 통계표 검색 (1단계)
# ==================================================
def search_tables(keyword: str, start: int = 1, size: int = 20) -> List[Dict]:
    """통계표를 이름으로 검색한다. 화면 1단계에서 "무엇을 볼지" 고르는 데 쓴다."""
    rows, _ = call("statisticsSearch.do", {
        "method": "getList",
        "searchNm": keyword,
        "startCount": str(start),
        "resultCount": str(size),
        "format": "json",
        "jsonVD": "Y",
    })
    return [{
        "org_id": r.get("ORG_ID", ""),
        "org_name": r.get("ORG_NM", ""),
        "tbl_id": r.get("TBL_ID", ""),
        "tbl_name": r.get("TBL_NM", ""),
        "stat_name": r.get("STAT_NM", ""),
        "path": r.get("MT_ATITLE", ""),
        # 수록 기간 — 이 통계표에서 언제부터 언제까지 볼 수 있는지
        "start_period": r.get("STRT_PRD_DE", ""),
        "end_period": r.get("END_PRD_DE", ""),
        "contents": (r.get("CONTENTS", "") or "")[:200],
    } for r in rows]


# ==================================================
# 3. 메타 조회 (2단계 — 파라미터 조립용)
# ==================================================
def fetch_meta(org_id: str, tbl_id: str, meta_type: str) -> List[Dict]:
    """통계표의 항목(ITM)·분류(OBJ)·기간(PRD) 목록을 가져온다.

    이 값이 있어야 화면에서 `itmId`·`objL1` 을 **찍어서 고를 수** 있다.
    통계표에 따라 분류가 없어 `err=30`(데이터 없음)이 오기도 하므로,
    호출하는 쪽에서 빈 목록으로 처리한다.
    """
    rows, _ = call("statisticsData.do", {
        "method": "getMeta",
        "orgId": org_id,
        "tblId": tbl_id,
        "type": meta_type,
        "format": "json",
        "jsonVD": "Y",
    })

    if meta_type == "PRD":
        return [{
            "period_type": r.get("PRD_SE", ""),
            "start_period": r.get("STRT_PRD_DE", ""),
            "end_period": r.get("END_PRD_DE", ""),
        } for r in rows]

    if meta_type == "ITM":
        return [{
            "id": r.get("ITM_ID", ""),
            "name": r.get("ITM_NM", ""),
            "unit": r.get("UNIT_NM", ""),
        } for r in rows]

    # OBJ — 분류 코드 목록 (지역·성별 등)
    return [{
        "id": r.get("OBJ_ID", ""),
        "name": r.get("OBJ_NM", ""),
        "item_id": r.get("ITM_ID", ""),
        "item_name": r.get("ITM_NM", ""),
    } for r in rows]


# ==================================================
# 4. 통계 수치 조회 + 차트용 정규화 (3단계)
# ==================================================
def _to_number(raw) -> Optional[float]:
    """KOSIS 의 값(`DT`)을 숫자로 바꾼다. 결측(`-`·빈 문자열)은 None."""
    if raw is None:
        return None
    text = str(raw).strip().replace(",", "")
    if text in ("", "-", "X", "...", "N/A"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def period_label(period: str, period_type: str) -> str:
    """기간 코드를 사람이 읽는 라벨로 바꾼다.

    KOSIS 의 `PRD_DE` 는 주기마다 자릿수가 다르다.
        Y  2024      → 2024
        M  202406    → 2024-06
        Q  20242     → 2024 2Q
        D  20240601  → 2024-06-01
    """
    text = str(period or "")
    if period_type == "M" and len(text) == 6:
        return f"{text[:4]}-{text[4:]}"
    if period_type == "D" and len(text) == 8:
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    if period_type in ("Q", "H") and len(text) == 5:
        suffix = "Q" if period_type == "Q" else "H"
        return f"{text[:4]} {text[4]}{suffix}"
    return text


def normalize_rows(rows: List[Dict]) -> List[Dict]:
    """KOSIS 응답 한 줄을 snake_case + 숫자형으로 정규화한다.

    KOSIS 는 분류를 `C1_NM`, `C2_NM`, `C3_NM` … 으로 최대 8단계까지 준다.
    통계표마다 개수가 다르므로 있는 것만 모아 `groups` 리스트에 담는다.
    """
    result = []
    for r in rows:
        groups = []
        for level in range(1, 9):
            name = r.get(f"C{level}_NM")
            if name:
                groups.append({
                    "level": level,
                    "code": r.get(f"C{level}", ""),
                    "name": name,
                    "label": r.get(f"C{level}_OBJ_NM", ""),
                })
        period_type = r.get("PRD_SE", "")
        result.append({
            "period": r.get("PRD_DE", ""),
            "period_label": period_label(r.get("PRD_DE", ""), period_type),
            "period_type": period_type,
            "item_id": r.get("ITM_ID", ""),
            "item": r.get("ITM_NM", ""),
            "groups": groups,
            "group_label": " · ".join(g["name"] for g in groups),
            "value": _to_number(r.get("DT")),
            "unit": r.get("UNIT_NM", ""),
        })
    return result


def build_chart(rows: List[Dict], max_series: int = 12) -> Dict:
    """정규화된 행들을 ApexCharts 가 바로 쓰는 모양으로 바꾼다.

    KOSIS 응답은 (기간 × 분류 × 항목) 이 한 줄씩 평평하게 늘어선 형태다.
    차트는 "가로축 = 기간, 선 하나 = 분류/항목 조합" 이 필요하므로 여기서 뒤집는다.

    시리즈가 너무 많으면 (지역 250개 등) 차트가 읽히지 않으므로 최근 값이 큰 순으로
    `max_series` 개만 남기고, 몇 개를 잘랐는지 `truncated` 에 알려 준다.
    """
    if not rows:
        return {"categories": [], "series": [], "unit": "", "truncated": 0, "total_series": 0}

    # 가로축 — 기간을 코드 순으로 정렬한다 (문자열이지만 자릿수가 같아 사전순 = 시간순).
    periods = sorted({r["period"] for r in rows if r["period"]})
    labels = {p: period_label(p, next((r["period_type"] for r in rows if r["period"] == p), "")) for p in periods}

    # 시리즈 이름 — 분류와 항목 중 **여러 값이 있는 쪽만** 이름에 넣어 짧게 유지한다.
    many_groups = len({r["group_label"] for r in rows}) > 1
    many_items = len({r["item"] for r in rows}) > 1

    def series_name(row: Dict) -> str:
        parts = []
        if many_groups and row["group_label"]:
            parts.append(row["group_label"])
        if many_items and row["item"]:
            parts.append(row["item"])
        # 둘 다 하나뿐이면 항목 이름을 쓴다 (선이 하나뿐인 경우)
        return " · ".join(parts) or row["item"] or row["group_label"] or "값"

    buckets: Dict[str, Dict[str, Optional[float]]] = {}
    for row in rows:
        buckets.setdefault(series_name(row), {})[row["period"]] = row["value"]

    # 마지막 기간 값이 큰 순으로 정렬 — 잘라내도 중요한 계열이 남는다.
    def sort_key(name: str) -> float:
        values = [v for v in buckets[name].values() if v is not None]
        return abs(values[-1]) if values else 0.0

    names = sorted(buckets, key=sort_key, reverse=True)
    kept, truncated = names[:max_series], max(len(names) - max_series, 0)

    series = [{
        "name": name,
        # 기간마다 값이 없을 수 있다. ApexCharts 는 null 을 끊긴 선으로 그린다.
        "data": [buckets[name].get(p) for p in periods],
    } for name in kept]

    return {
        "categories": [labels[p] for p in periods],
        "series": series,
        "unit": next((r["unit"] for r in rows if r["unit"]), ""),
        "truncated": truncated,
        "total_series": len(names),
    }


def fetch_data(
    org_id: str,
    tbl_id: str,
    itm_id: str = "ALL",
    obj_l1: str = "ALL",
    prd_se: str = "Y",
    period_count: int = 10,
    start_period: str = "",
    end_period: str = "",
    max_rows: int = 5000,
    max_series: int = 12,
) -> Dict:
    """통계 수치를 받아 표·차트에 바로 쓸 수 있는 형태로 돌려준다.

    기간은 두 방식 중 하나로 지정한다.
      - `period_count` : 최근 N개 기간 (`newEstPrdCnt`) — 기본
      - `start_period` + `end_period` : 기간 직접 지정 (`prdSt`/`prdEd`)
    """
    params = {
        "method": "getList",
        "orgId": org_id,
        "tblId": tbl_id,
        "itmId": itm_id or "ALL",
        "objL1": obj_l1 or "ALL",
        "prdSe": prd_se,
        "format": "json",
        "jsonVD": "Y",
    }
    if start_period and end_period:
        params["startPrdDe"] = start_period
        params["endPrdDe"] = end_period
    else:
        params["newEstPrdCnt"] = str(period_count)

    raw_rows, elapsed_ms = call("Param/statisticsParameterData.do", params)

    # 응답이 수만 줄일 수 있다 (지역 전체 × 여러 해). 표에 그릴 양만 남긴다.
    total = len(raw_rows)
    clipped = raw_rows[:max_rows]

    rows = normalize_rows(clipped)
    chart = build_chart(rows, max_series=max_series)

    first = clipped[0] if clipped else {}
    return {
        "table": {
            "org_id": org_id,
            "tbl_id": tbl_id,
            "tbl_name": first.get("TBL_NM", ""),
            "period_type": prd_se,
            "period_type_name": PERIOD_TYPES.get(prd_se, prd_se),
            "unit": chart["unit"],
            "last_updated": first.get("LST_CHN_DE", ""),
        },
        "rows": rows,
        "chart": chart,
        "meta": {
            "elapsed_ms": elapsed_ms,
            "total_rows": total,
            "returned_rows": len(rows),
            # 응답이 잘렸는지 화면에서 알 수 있어야 한다 (조용히 자르지 않는다).
            "row_truncated": max(total - len(clipped), 0),
            "series_truncated": chart["truncated"],
            "total_series": chart["total_series"],
            # 화면에 보여줄 요청 URL — 인증키는 빼고 만든다.
            "request_url": build_url("Param/statisticsParameterData.do", params),
        },
    }
