"""네이버 검색 API 진단 — 인증 경로와 페이징 경계를 실호출로 가른다

## 왜 이 스크립트가 있나

우리 문서가 오래 *"`start + display - 1 > 1000` 이면 400"* 이라고 적어 왔는데,
**공식 명세에서 그 부등식을 찾지 못했습니다.** 명세에 있는 것은 `display 1~100` 과
`start 1~1000` 두 **독립 범위**뿐입니다. 근거 없는 제약을 사실처럼 문서에 적어 두는 것이
이 프로젝트가 피하려는 바로 그 습관이라, **의견으로 정하지 않고 쏴 봅니다.**

## 🔴 2026-08-27 실측으로 배운 것 — 여기가 이 파일의 값어치입니다

**네이버도 DART 처럼 HTTP 200 에 오류를 실어 보냅니다.**

| 경로 | 헤더 | 우리 키로 쏜 결과 |
|---|---|---|
| 개발자센터 `/v1/search/news.json` | `X-Naver-Client-*` | **HTTP 401** · `024` 인증 실패 |
| API HUB `/search/v1/news` | `X-NCP-APIGW-*` | **HTTP 200** · 본문 `error_code 052`
  *"등록된 파트너가 아닙니다"* |

HUB 경로는 **인증에 실패해도 200** 입니다. `response.raise_for_status()` 만 믿는 코드는
이 오류를 **성공으로 읽고 빈 결과를 조용히 적재**합니다. 수집기를 짤 때 **본문의
`error_code`/`errorCode` 를 반드시 봅니다.**

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

#: 개발자센터 뉴스 검색 엔드포인트 (2027-06-30 까지).
BASE_URL = "https://openapi.naver.com/v1/search/news.json"

#: 이관 뒤 경로. **인증 헤더 이름까지 함께 바뀝니다.**
HUB_URL = "https://openapi.naver.com/search/v1/news"

#: 검색어. 결과가 충분히 많아야 경계까지 갈 수 있어 대형주를 씁니다.
QUERY = "삼성전자"

#: 예산 장부의 출처 이름. 뉴스·카페글이 **같은 쿼터를 나눠 씁니다**.
BUDGET_SOURCE = "naver_search"

TIMEOUT = 10


def auth_headers(client_id: str, client_secret: str, *, hub: bool = False) -> Dict[str, str]:
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
            "  ⚠️ 2026-07-30 부터 신규 발급이 중단됐습니다. 이전 발급분만 씁니다."
        )
    print(f"  키 출처: id={id_src} · secret={sec_src} "
          f"({secrets.mask(client_id)} / {secrets.mask(client_secret)})")
    print(f"  길이: id={len(client_id)}자 · secret={len(client_secret)}자")
    return client_id, client_secret


def _read_body(res) -> Dict:
    """응답을 JSON 으로 읽는다. 항목 내용은 **담지 않는다** (약관 7.3 ③)."""
    body = json.loads(res.read().decode("utf-8"))
    return body if isinstance(body, dict) else {}


def body_error(body: Dict) -> Optional[str]:
    """본문에 실려 온 오류. **HTTP 200 이어도 여기가 차 있을 수 있다.**

    개발자센터는 `errorCode`, API HUB 는 `error_code` 로 키 이름이 다르다.
    """
    code = body.get("error_code") or body.get("errorCode")
    if not code:
        return None
    return f"{code}: {body.get('message') or body.get('errorMessage') or ''}".strip()


def diagnose(client_id: str, client_secret: str) -> Optional[bool]:
    """키가 어느 경로에서 통하는지 먼저 가른다.

    반환: `False`=개발자센터 통과 · `True`=HUB 통과 · `None`=둘 다 실패.
    (`probe()` 의 `hub` 인자에 그대로 넣을 수 있게 불리언으로 준다.)
    """
    print("── ① 인증 경로 진단 ──")
    통과: Optional[bool] = None
    for 이름, url, hub in (("개발자센터", BASE_URL, False), ("API HUB", HUB_URL, True)):
        if not budget.try_spend(BUDGET_SOURCE):
            print(f"  {이름:<10} ⏭ 예산 소진")
            continue
        target = f"{url}?{urllib.parse.urlencode({'query': QUERY, 'display': 1})}"
        req = urllib.request.Request(
            target, headers=auth_headers(client_id, client_secret, hub=hub))
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as res:
                body = _read_body(res)
            오류 = body_error(body)                # ⚠️ 200 이라고 성공이 아니다
            if 오류:
                print(f"  {이름:<10} HTTP {res.status}  🔴 본문 오류 {오류}")
            else:
                print(f"  {이름:<10} HTTP {res.status}  ✅ 통과 · total={body.get('total'):,}")
                통과 = hub if 통과 is None else 통과
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")[:100]
            print(f"  {이름:<10} HTTP {exc.code}  {raw}")
        except Exception as exc:
            print(f"  {이름:<10} {type(exc).__name__}: {exc}")
    print()
    return 통과


def probe(start: int, display: int, client_id: str, client_secret: str,
          *, hub: bool = False) -> Dict:
    """한 번 쏴 보고 **메타만** 돌려준다.

    성공이든 실패든 예외로 끝내지 않는다 — 이 스크립트의 목적이 *"어디서 400이 나는가"*
    라서 **400 도 결과**다.
    """
    if not budget.try_spend(BUDGET_SOURCE):
        return {"start": start, "display": display, "skipped": "예산 소진"}

    base = HUB_URL if hub else BASE_URL
    params = urllib.parse.urlencode({"query": QUERY, "start": start, "display": display})
    req = urllib.request.Request(
        f"{base}?{params}", headers=auth_headers(client_id, client_secret, hub=hub))

    out: Dict = {"start": start, "display": display, "sum": start + display - 1}
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as res:
            out["http"] = res.status
            body = _read_body(res)
        out["error"] = body_error(body)             # 200 이어도 오류일 수 있다
        out["total"] = body.get("total")
        # ⚠️ items 의 **내용은 담지 않는다.** 개수만 센다.
        out["returned"] = len(body.get("items", []))
    except urllib.error.HTTPError as exc:
        out["http"] = exc.code
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            err = json.loads(raw)
            inner = err.get("error", err)           # HUB 는 {"error":{...}} 로 중첩
            out["error"] = (f"{inner.get('errorCode') or inner.get('error_code')}: "
                            f"{inner.get('errorMessage') or inner.get('message')}")
        except json.JSONDecodeError:
            out["error"] = raw[:120]
    except Exception as exc:                        # 네트워크·타임아웃
        out["http"] = None
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


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

    hub = diagnose(client_id, client_secret)
    if hub is None:
        # 인증이 안 되면 경계 실측은 전부 같은 오류로 나와 **아무것도 못 가른다.**
        # 예산을 5콜 더 태우지 않고 여기서 멈춘다.
        print("🔴 두 경로 모두 인증에 실패했습니다. 경계 실측은 건너뜁니다.")
        print()
        print("  할 일:")
        print("   1. developers.naver.com > 내 애플리케이션 에서 Client ID·Secret 을 다시 확인")
        print("      (개발자센터 Secret 은 보통 10자입니다. 40자라면 다른 서비스 키일 수 있습니다)")
        print("   2. 그 애플리케이션에 **검색 API 가 추가**돼 있는지 확인 (없으면 403)")
        print("   3. 키가 아예 없다면 NAVER Cloud Platform API HUB 로 가야 합니다 —")
        print("      2026-07-30 부터 개발자센터 신규 발급이 중단됐습니다")
        return 1

    경로 = "API HUB" if hub else "개발자센터"
    print(f"── ② 페이징 경계 실측 ({경로} 경로) ──")
    results = []
    for start, display, label, *_ in CASES:
        r = probe(start, display, client_id, client_secret, hub=hub)
        results.append(r)
        상태 = r.get("http")
        if r.get("skipped"):
            꼬리 = f"⏭ {r['skipped']}"
        elif r.get("error"):
            꼬리 = f"🔴 {r['error']}"
        else:
            꼬리 = f"total={r.get('total'):,} · 반환 {r.get('returned')}건"
        print(f"  start={start:<5} display={display:<4} 합={r.get('sum', '-'):<5} "
              f"HTTP {str(상태):<5} {label:<20} {꼬리}")

    print()
    print("── 판정 ──")
    초과 = next((r for r in results
                if r["start"] == 1000 and r["display"] == 100), None)
    if 초과 and 초과.get("http") == 200 and not 초과.get("error"):
        print("  ✅ `start + display - 1 <= 1000` 부등식은 **존재하지 않습니다.**")
        print("     합 1099 가 통과했습니다 → 문서에서 '안전측 가정' 표시를 뗍니다.")
    elif 초과 and (초과.get("http") == 400 or 초과.get("error")):
        print("  ✅ 부등식이 **실재합니다.** 합 1099 가 거부됐습니다.")
        print(f"     오류: {초과.get('error')}")
        print("     → 문서의 제약을 '실측 확인됨' 으로 승격합니다.")
    else:
        print("  ⚠️ 판정 불가 — 위 결과를 사람이 읽으세요.")

    print()
    print(f"  오늘 사용량: {budget.usage(BUDGET_SOURCE)}")
    print("  ⚠️ 응답 본문은 저장하지 않았습니다 (네이버 약관 7.3 ③).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
