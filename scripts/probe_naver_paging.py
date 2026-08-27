"""네이버 검색 API 진단 — 인증 경로와 페이징 경계를 실호출로 가른다

## 왜 이 스크립트가 있나

우리 문서가 오래 *"`start + display - 1 > 1000` 이면 400"* 이라고 적어 왔는데,
**공식 명세에서 그 부등식을 찾지 못했습니다.** 명세에 있는 것은 `display 1~100` 과
`start 1~1000` 두 **독립 범위**뿐입니다. 근거 없는 제약을 사실처럼 문서에 적어 두는 것이
이 프로젝트가 피하려는 바로 그 습관이라, **의견으로 정하지 않고 쏴 봅니다.**

## 우리 팀 키는 NAVER API HUB 키입니다 (2026-08-27 확인)

개발자센터(`developers.naver.com`)는 2026-07-30 로 신규 발급이 끊겼고, 우리 키는
**NAVER Cloud Platform 의 API HUB** 에서 재발급받은 것입니다. 그래서 **호스트·경로·
인증 헤더가 전부 다릅니다.**

| | 개발자센터 (구) | **API HUB (우리)** |
|---|---|---|
| 호스트 | `openapi.naver.com` | **`naverapihub.apigw.ntruss.com`** |
| 경로 | `/v1/search/news.json` | **`/search/v1/news`** |
| 헤더 | `X-Naver-Client-Id` ·
  `X-Naver-Client-Secret` | **`X-NCP-APIGW-API-KEY-ID` ·
  `X-NCP-APIGW-API-KEY`** |
| 한도 | 일 25,000 | **월 775,000 (월 단위 관리)** |

## 🔴 실측으로 배운 함정 — 호스트를 잘못 잡으면 200 이 온다

| 호스트 | 인증 | 결과 |
|---|---|---|
| `naverapihub.apigw.ntruss.com` | 올바름 | HTTP 200 · 정상 |
| `naverapihub.apigw.ntruss.com` | 틀림 | HTTP **401** ·
  `{"error":{"errorCode":"200",…}}` — 제대로 거절 |
| `openapi.naver.com` + HUB 경로 | 올바름 | **HTTP 200** · 본문 `error_code 052`
  *"등록된 파트너가 없습니다"* |

**세 번째 줄이 위험합니다.** 호스트만 틀렸는데 HTTP 는 200 이고, 본문에만 오류가 실립니다.
`raise_for_status()` 만 믿는 수집기는 이걸 **성공으로 읽고 빈 결과를 매일 조용히 적재**합니다.
DART 가 한도 초과를 `HTTP 200 + status="020"` 으로 주는 것과 같은 종류입니다.

→ **수집기는 HTTP 상태와 본문 오류코드를 둘 다 봅니다.** 키 이름도 다릅니다 —
개발자센터·HUB 게이트웨이는 `errorCode`, 개발자센터의 HUB 경로 응답은 `error_code`.

## ⚠️ 응답 본문을 저장하지 않습니다

네이버 약관 7.3 ③ 이 취득 정보의 **저장·가공·배포를 금지**합니다. 그래서 이 스크립트는
**메타만** 봅니다 — HTTP 상태 · 오류코드 · `total` · 반환 건수. 기사 제목조차 찍지 않습니다.

실행:
    python scripts/probe_naver_paging.py
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Tuple

from common import budget, secrets

#: 개발자센터 엔드포인트 (2026-07-30 신규 발급 중단 · 2027-06-30 종료).
LEGACY_URL = "https://openapi.naver.com/v1/search/news.json"

#: **우리가 쓰는 경로.** NAVER API HUB.
HUB_URL = "https://naverapihub.apigw.ntruss.com/search/v1/news"

#: 🔴 함정 확인용 — 호스트를 개발자센터로 두고 HUB 경로를 부르면 **200 에 오류가 실려 온다.**
TRAP_URL = "https://openapi.naver.com/search/v1/news"

#: 검색어. 결과가 충분히 많아야 경계까지 갈 수 있어 대형주를 씁니다.
QUERY = "삼성전자"

#: 예산 장부의 출처 이름. 뉴스·카페글이 **같은 쿼터를 나눠 씁니다**.
#: ⚠️ HUB 는 **월 단위** 한도인데 우리 예산 모듈은 **일 단위**로 셉니다. 보수적이라
#:    한도를 넘길 위험은 없지만, 표기가 사실과 다르다는 것을 알고 씁니다.
BUDGET_SOURCE = "naver_search"

TIMEOUT = 10


def auth_headers(client_id: str, client_secret: str, *, hub: bool = True) -> Dict[str, str]:
    """인증 헤더. 개발자센터와 API HUB 가 **헤더 이름부터 다릅니다.**"""
    if hub:
        return {"X-NCP-APIGW-API-KEY-ID": client_id,
                "X-NCP-APIGW-API-KEY": client_secret}
    return {"X-Naver-Client-Id": client_id,
            "X-Naver-Client-Secret": client_secret}


def _keys() -> Tuple[str, str]:
    """클라이언트 아이디와 시크릿. 없으면 무엇을 해야 하는지 알려주고 멈춥니다."""
    client_id, id_src = secrets.load_key(["NAVER_API_CLIENT_ID", "NAVER_CLIENT_ID"])
    client_secret, sec_src = secrets.load_key(
        ["NAVER_API_SECRET_KEY", "NAVER_CLIENT_SECRET"])
    if not client_id or not client_secret:
        raise SystemExit(
            "네이버 검색 API 키가 없습니다.\n"
            "  할 일: .env 에 NAVER_API_CLIENT_ID · NAVER_API_SECRET_KEY 를 넣으세요.\n"
            "  ⚠️ 개발자센터는 2026-07-30 로 신규 발급이 끊겼습니다.\n"
            "     NAVER Cloud Platform > API HUB 에서 발급받으세요."
        )
    print(f"  키 출처: id={id_src} · secret={sec_src} "
          f"({secrets.mask(client_id)} / {secrets.mask(client_secret)})")
    return client_id, client_secret


def body_error(body: Dict) -> Optional[str]:
    """본문에 실려 온 오류. **HTTP 200 이어도 여기가 차 있을 수 있다.**

    HUB 게이트웨이는 `{"error": {"errorCode": …}}` 로 중첩하고,
    개발자센터의 HUB 경로 응답은 평평한 `error_code` 를 쓴다. 둘 다 본다.
    """
    inner = body.get("error") if isinstance(body.get("error"), dict) else body
    code = inner.get("error_code") or inner.get("errorCode")
    if not code:
        return None
    말 = inner.get("message") or inner.get("errorMessage") or inner.get("details") or ""
    return f"{code}: {말}".strip()


def call(url: str, params: Dict, headers: Dict[str, str]) -> Dict:
    """한 번 쏘고 **메타만** 돌려준다. 400 도 결과이므로 예외로 끝내지 않는다."""
    target = f"{url}?{urllib.parse.urlencode(params)}"
    out: Dict = {}
    try:
        with urllib.request.urlopen(
                urllib.request.Request(target, headers=headers), timeout=TIMEOUT) as res:
            out["http"] = res.status
            body = json.loads(res.read().decode("utf-8"))
        body = body if isinstance(body, dict) else {}
        out["error"] = body_error(body)           # ⚠️ 200 이라고 성공이 아니다
        out["total"] = body.get("total")
        # ⚠️ items 의 **내용은 담지 않는다.** 개수만 센다 (약관 7.3 ③).
        out["returned"] = len(body.get("items", []))
    except urllib.error.HTTPError as exc:
        out["http"] = exc.code
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            out["error"] = body_error(json.loads(raw)) or raw[:120]
        except json.JSONDecodeError:
            out["error"] = raw[:120]
    except Exception as exc:                      # 네트워크·타임아웃
        out["http"] = None
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def diagnose(client_id: str, client_secret: str) -> Optional[str]:
    """키가 어느 문으로 들어가는지 가른다. 통하는 base URL 을 돌려준다."""
    print("── ① 인증 경로 진단 ──")
    통하는곳: Optional[str] = None
    후보 = (
        ("API HUB (우리)", HUB_URL, True),
        ("개발자센터", LEGACY_URL, False),
        ("🔴 호스트 오인 함정", TRAP_URL, True),
    )
    for 이름, url, hub in 후보:
        if not budget.try_spend(BUDGET_SOURCE):
            print(f"  {이름:<20} ⏭ 예산 소진")
            continue
        r = call(url, {"query": QUERY, "display": 1},
                 auth_headers(client_id, client_secret, hub=hub))
        if r.get("error"):
            표시 = f"🔴 {r['error']}"
            if r.get("http") == 200:
                표시 += "   ← HTTP 는 200 인데 오류다"
        else:
            표시 = f"✅ 통과 · total={r.get('total'):,}"
            if 통하는곳 is None and url != TRAP_URL:
                통하는곳 = url
        print(f"  {이름:<20} HTTP {str(r.get('http')):<5} {표시}")
    print()
    return 통하는곳


#: 무엇을 왜 쏘는지. **가설을 미리 적어 두고** 결과와 대조한다 —
#: 결과를 보고 나서 해석을 지어내지 않으려는 것이다.
CASES: List[Tuple[int, int, str, str]] = [
    (1, 10, "대조군", "정상 200 이어야 한다"),
    (901, 100, "합=1000 (경계)", "두 해석 모두 허용"),
    (1000, 100, "합=1099 (경계 초과)", "부등식이 진짜면 400 · 아니면 200"),
    (1000, 1, "start 상한, 합=1000", "두 해석 모두 허용. start=1000 도달 확인"),
    (1001, 1, "start 상한 초과", "공식 명세대로면 400"),
]


def main() -> int:
    print("── 네이버 검색 API 진단 ──")
    print(f"  검색어 {QUERY!r}")
    client_id, client_secret = _keys()
    print(f"  오늘 사용량: {budget.usage(BUDGET_SOURCE)}")
    print()

    base = diagnose(client_id, client_secret)
    if base is None:
        # 인증이 안 되면 경계 실측은 전부 같은 오류로 나와 **아무것도 못 가른다.**
        # 예산을 5콜 더 태우지 않고 여기서 멈춘다.
        print("🔴 인증되는 경로가 없습니다. 경계 실측은 건너뜁니다.")
        print()
        print("  할 일:")
        print("   1. NAVER Cloud Platform > API HUB 에서 Client ID·Secret 을 다시 확인")
        print("   2. 그 키에 **검색(Search) API 이용 신청**이 돼 있는지 확인")
        print("   3. 개발자센터 키를 쓰고 있다면 2027-06-30 까지만 유효합니다")
        return 1

    hub = base == HUB_URL
    print(f"── ② 페이징 경계 실측 ({'API HUB' if hub else '개발자센터'}) ──")
    results = []
    for start, display, label, *_ in CASES:
        if not budget.try_spend(BUDGET_SOURCE):
            print(f"  start={start:<5} ⏭ 예산 소진")
            continue
        r = call(base, {"query": QUERY, "start": start, "display": display},
                 auth_headers(client_id, client_secret, hub=hub))
        r.update({"start": start, "display": display, "sum": start + display - 1})
        results.append(r)
        꼬리 = (f"🔴 {r['error']}" if r.get("error")
                else f"total={r.get('total'):,} · 반환 {r.get('returned')}건")
        print(f"  start={start:<5} display={display:<4} 합={r['sum']:<5} "
              f"HTTP {str(r.get('http')):<5} {label:<20} {꼬리}")

    print()
    print("── 판정 ──")
    초과 = next((r for r in results
                if r["start"] == 1000 and r["display"] == 100), None)
    상한 = next((r for r in results if r["start"] == 1001), None)

    if 초과 and 초과.get("http") == 200 and not 초과.get("error"):
        print("  ✅ `start + display - 1 <= 1000` 부등식은 **존재하지 않습니다.**")
        print(f"     합 1099 가 통과했고 {초과.get('returned')}건이 왔습니다.")
        print("     → 문서에서 '안전측 가정' 표시를 떼고 두 독립 범위만 적습니다.")
    elif 초과 and (초과.get("http") != 200 or 초과.get("error")):
        print("  ✅ 부등식이 **실재합니다.** 합 1099 가 거부됐습니다.")
        print(f"     오류: {초과.get('error')}")
        print("     → 문서의 제약을 '실측 확인됨' 으로 승격합니다.")
    else:
        print("  ⚠️ 판정 불가 — 위 결과를 사람이 읽으세요.")

    if 상한:
        상태 = "거부됨" if (상한.get("http") != 200 or 상한.get("error")) else "통과함"
        print(f"  · start=1001 은 {상태} (HTTP {상한.get('http')}) "
              f"→ start 상한 1000 은 {'확인됨' if 상태 == '거부됨' else '더 넓다'}")

    print()
    print(f"  오늘 사용량: {budget.usage(BUDGET_SOURCE)}")
    print("  ⚠️ 응답 본문은 저장하지 않았습니다 (네이버 약관 7.3 ③).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
