"""금융감독원 금융상품통합비교공시(finlife) 연동 (외부 연동 계층 · U8 결정)

    https://finlife.fss.or.kr/finlifeapi/

M2 에서 키만 확인하고 **쓸 곳을 못 정해 U8 로 미뤄 뒀던** 소스다 (마스터인덱스 §6 U1 마지막 문단).
M5 에서 두 자리를 찾아 붙인다.

1. **무위험이자율의 교차검증** — `market_data.RISK_FREE_RATE` 가 `0.032` 로 하드코딩돼 있었다.
   샤프지수·효율적투자선·밸류에이션 할인율이 전부 이 상수에 걸려 있는데, 실측하면
   국고채 3년이 3.758% (ECOS · 2026-07-31) 라 **0.56%p 어긋나 있었다.**
   국고채를 무위험이자율로 쓰고, 이 API 의 예금금리를 **나란히 실어** 두 값이 벌어지면 밝힌다.

2. **금융업 판정의 교차검증** — 08강 06.md 421행이 "금융업은 일반 제조업 비율로 보지 않는다"
   고 했는데(은행은 예금이 부채라 부채비율 1,000% 가 정상), 지금 코드는 그냥 계산해
   신한지주를 `🔴 200% 이상 주의` 로 찍고 있었다. KSIC 64~66 을 1차 기준으로 삼고
   이 API 의 금융회사 목록을 **확인용**으로 쓴다.

실제 호출로 확인한 것 (2026-08-02)
--------------------------------
| 권역 | `topFinGrpNo` | 정기예금 12개월 | 실측 금리 |
|---|---|---|---|
| 은행 | `020000` | 38건 | 2.40 ~ **3.40(중앙값)** ~ 3.95% |
| 저축은행 | `030300` | 159건 | 2.00 ~ **3.95(중앙값)** ~ 4.20% |
| 여신전문·보험·금융투자 | `030200`·`050000`·`060000` | **0건** | 정기예금 상품이 없다 |

- 오류도 HTTP 200 으로 온다. `result.err_cd` 가 `000` 이어야 정상이다 (ECOS 와 같은 함정).
- 응답이 `baseList`(상품)와 `optionList`(기간별 금리)로 **나뉘어** 온다. 금리는 optionList 에 있고
  `save_trm`(개월)으로 갈라진다. 상품 하나에 6·12·24개월이 각각 한 줄씩이다.
- `intr_rate` 는 기본금리, `intr_rate2` 는 최고우대금리다. **우대금리는 조건을 채워야 받는다** —
  그래서 대푯값으로는 기본금리 쪽을 쓰고, 우대금리는 상한으로만 보여 준다.
- 저축은행 금리가 은행보다 높은 것은 정상이다 (예금자보호 한도 안에서 신용위험이 다르다).
  그래서 **무위험이자율 대용으로는 은행 권역만** 쓴다.

책임 경계
--------
이 API 가 죽어도 리서치는 돈다. 무위험이자율은 ECOS 국고채가 1차 소스이고 여기는 교차검증이다.
실패를 예외로 올리지 않고 `{"available": False, "reason": ...}` 로 돌려준다.
"""

from __future__ import annotations

import json
import threading
import time
from statistics import median
from typing import Dict, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from common import secrets

FSS_BASE_URL = "https://finlife.fss.or.kr/finlifeapi"
REQUEST_TIMEOUT = 15
CACHE_TTL = 21600                    # 예금금리는 하루에 한 번 바뀔까 말까다. 6시간.

KEY_NAMES = ("FINANCE_SUPERVISORY_API_KEY", "FSS_API_KEY", "FSS_KEY")

# 권역 코드 — 위 표 참고. 정기예금이 실제로 있는 곳만 둔다.
GROUP_BANK = "020000"
GROUP_SAVINGS = "030300"
GROUPS = {GROUP_BANK: "은행", GROUP_SAVINGS: "저축은행"}

# 무위험이자율 대용으로 볼 기간 (개월). 국고채 3년과 견주려면 1년 정기예금이 가장 가깝다.
DEFAULT_TERM_MONTHS = "12"

_cache: Dict[Tuple, Tuple[float, object]] = {}
_cache_lock = threading.Lock()
_last_attempt: Dict[str, Optional[str]] = {"result": None, "detail": None}


def load_fss_key() -> Tuple[str, str]:
    return secrets.load_key(KEY_NAMES)


def available() -> bool:
    return bool(load_fss_key()[0])


def _cached(key: Tuple, producer):
    now = time.monotonic()
    with _cache_lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < CACHE_TTL:
            return hit[1]
    value = producer()
    with _cache_lock:
        _cache[key] = (time.monotonic(), value)
    return value


def _number(value) -> Optional[float]:
    if value in (None, "", "-"):
        return None
    try:
        number = float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def _failure(reason: str) -> Dict:
    _last_attempt.update(result="error", detail=reason[:200])
    return {"available": False, "reason": reason}


def _call(path: str, params: Dict[str, str]) -> Dict:
    """finlife 엔드포인트를 부른다. 인증키는 여기서만 붙인다.

    `err_cd` 가 `000` 이 아니면 그 메시지를 그대로 올린다 — 우리가 지어내지 않는다.
    """
    key, _ = load_fss_key()
    if not key:
        raise RuntimeError(
            "금융감독원 인증키가 없다 — "
            ".key 에 FINANCE_SUPERVISORY_API_KEY 를 넣는다"
        )

    query = urlencode({"auth": key, **params})
    request = Request(f"{FSS_BASE_URL}/{path}.json?{query}",
                      headers={"User-Agent": "api-test-GIC-harness/M5"})
    with urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        payload = json.loads(response.read().decode("utf-8"))

    result = payload.get("result") or {}
    code = str(result.get("err_cd") or "")
    if code and code != "000":
        raise RuntimeError(f"finlife 오류 {code} — {result.get('err_msg', '')}")
    return result


# ─────────────────────────────────────────────────────────────
# 1. 정기예금 금리 — 무위험이자율 교차검증
# ─────────────────────────────────────────────────────────────
def deposit_rates(group: str = GROUP_BANK, term_months: str = DEFAULT_TERM_MONTHS) -> Dict:
    """권역별 정기예금 금리 분포를 돌려준다.

    기본금리(`intr_rate`)와 최고우대금리(`intr_rate2`)를 **갈라서** 낸다.
    우대금리는 급여이체·카드실적 같은 조건을 채워야 받는 값이라, 그것을 대푯값으로 쓰면
    실제로 받을 수 없는 금리로 밸류에이션을 하게 된다.
    """
    key = ("deposit", group, term_months)

    def produce() -> Dict:
        started = time.monotonic()
        try:
            result = _call("depositProductsSearch", {"topFinGrpNo": group, "pageNo": "1"})
        except HTTPError as error:
            return _failure(f"HTTP {error.code} — {error.reason}")
        except (URLError, TimeoutError) as error:
            return _failure(f"연결 실패 — {error}")
        except Exception as error:
            return _failure(f"{type(error).__name__}: {str(error)[:120]}")

        options = [o for o in (result.get("optionList") or [])
                   if str(o.get("save_trm") or "") == str(term_months)]
        if not options:
            return _failure(f"{GROUPS.get(group, group)} 권역에 {term_months}개월 정기예금이 없다")

        base = sorted(v for v in (_number(o.get("intr_rate")) for o in options) if v is not None)
        best = sorted(v for v in (_number(o.get("intr_rate2")) for o in options) if v is not None)
        if not base:
            return _failure("금리 값이 전부 비어 있다")

        # 기준일은 상품 목록의 `dcls_month`(공시월 YYYYMM)에서 가져온다.
        # 공시는 월 단위라 "며칠 자" 가 아니다 — 그 사실을 아래 note 가 밝힌다.
        months = sorted({str(b.get("dcls_month") or "") for b in (result.get("baseList") or [])
                         if b.get("dcls_month")})
        as_of_month = months[-1] if months else ""

        _last_attempt.update(result="ok", detail=f"{len(options)}건")
        return {
            "available": True,
            "group": group,
            "group_name": GROUPS.get(group, group),
            "term_months": int(term_months),
            "count": len(options),
            "products": len(result.get("baseList") or []),
            # 대푯값은 **기본금리 중앙값**이다 (우대금리를 대푯값으로 쓰지 않는 이유는 docstring)
            "base_median": round(median(base), 3),
            "base_min": round(base[0], 3),
            "base_max": round(base[-1], 3),
            "top_median": round(median(best), 3) if best else None,
            "top_max": round(best[-1], 3) if best else None,
            "as_of_month": as_of_month,
            "elapsed_sec": round(time.monotonic() - started, 2),
            "note": ("기본금리와 최고우대금리를 갈라 냈다. 우대금리는 조건을 채워야 받는 값이라 "
                     "대푯값으로 쓰지 않는다. 공시는 **월 단위**라 일별 금리가 아니다."),
        }

    return _cached(key, produce)


# ─────────────────────────────────────────────────────────────
# 2. 금융회사 목록 — 금융업 판정 교차검증
# ─────────────────────────────────────────────────────────────
def companies(group: str = GROUP_BANK) -> Dict:
    """권역에 등록된 금융회사 이름 목록.

    ⚠️ **이름 대조는 보조 수단이다.** 상장사는 지주회사인 경우가 많아
    (`신한지주` ↔ `신한은행`) 이름이 그대로 겹치지 않는다. 그래서 판정의 1차 기준은
    KSIC 업종코드(64~66)이고, 이 목록은 "맞다" 를 확인해 주는 쪽으로만 쓴다.
    """
    key = ("companies", group)

    def produce() -> Dict:
        try:
            result = _call("companySearch", {"topFinGrpNo": group, "pageNo": "1"})
        except HTTPError as error:
            return _failure(f"HTTP {error.code} — {error.reason}")
        except (URLError, TimeoutError) as error:
            return _failure(f"연결 실패 — {error}")
        except Exception as error:
            return _failure(f"{type(error).__name__}: {str(error)[:120]}")

        rows = [{"code": str(b.get("fin_co_no") or ""), "name": str(b.get("kor_co_nm") or "")}
                for b in (result.get("baseList") or []) if b.get("kor_co_nm")]
        _last_attempt.update(result="ok", detail=f"{len(rows)}개사")
        return {
            "available": bool(rows),
            "group": group,
            "group_name": GROUPS.get(group, group),
            "count": len(rows),
            "rows": rows,
            "note": "상장사는 지주회사 이름이라 그대로 겹치지 않는다 — 판정 1차 기준은 KSIC 다.",
        }

    return _cached(key, produce)


def get_status() -> Dict:
    """인증키 상태 진단 (키 값은 노출하지 않는다)."""
    key, source = load_fss_key()
    return {
        "configured": bool(key),
        "source": source,
        "masked": secrets.mask(key) if key else "",
        "endpoint": FSS_BASE_URL,
        "uses": ["무위험이자율 교차검증 (국고채 대비)", "금융업 판정 교차검증"],
        "last_result": _last_attempt.get("result"),
        "last_detail": _last_attempt.get("detail"),
    }
