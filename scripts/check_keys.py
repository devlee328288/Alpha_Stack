"""내 API 키가 제대로 들어갔는지 확인한다 — 값은 한 글자도 보여주지 않고.

사용법
------
    python scripts/check_keys.py            # 있는지만 본다 (네트워크 안 씀)
    python scripts/check_keys.py --live     # 실제로 한 번씩 호출해 본다

왜 필요한가
----------
키가 없어서 안 되는 것과, 키는 있는데 **모양이 틀려서** 안 되는 것과, 모양은 맞는데
**서비스 이용신청이 안 돼서** 안 되는 것은 전부 다른 문제다. 그런데 셋 다 겉으로는
"안 된다" 로만 보인다. 여기서 그 셋을 갈라 준다.

🔴 키 값은 절대 출력하지 않는다. 이 저장소는 PUBLIC 이고, 터미널 로그는 캡처돼
   공유되기 쉽다. 길이와 앞 두 글자만 보여준다.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, List, NamedTuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import secrets  # noqa: E402


class 키정의(NamedTuple):
    이름: str                       # `.env` 에 적는 이름
    별칭: tuple                     # 코드가 함께 받아 주는 다른 이름들
    무엇: str                       # 이 키로 무엇을 하나
    필수: bool                      # 없으면 1차 범위가 안 도나
    발급: str                       # 어디서 받나
    없으면: str                     # 없을 때 무엇이 막히나
    형식: Optional[Callable[[str], Optional[str]]] = None   # 모양 검사 → 문제 설명
    실호출: Optional[Callable[[str], str]] = None           # --live 일 때 한 번 때려본다


# ══════════════════════════════════════════════════════════════════════════
# 모양 검사 — 흔한 실수를 잡는다
# ══════════════════════════════════════════════════════════════════════════


def _길이(최소: int, 최대: int) -> Callable[[str], Optional[str]]:
    def 검사(v: str) -> Optional[str]:
        if not (최소 <= len(v) <= 최대):
            return f"길이가 {len(v)}자다 — 보통 {최소}~{최대}자다. 붙여넣기가 잘렸는지 본다"
        return None
    return 검사


def _공공데이터포털(v: str) -> Optional[str]:
    """🔴 가장 많이 걸리는 곳. Encoding / Decoding 을 섞어 쓰면 인증이 실패한다."""
    if len(v) < 80:
        return f"길이가 {len(v)}자다 — 인증키는 보통 88~100자다. 잘렸는지 본다"
    if "%25" in v:
        return ("`%25` 가 들어 있다 — **두 번 인코딩된 키**다. 포털에서 준 "
                "*일반 인증키(Encoding)* 를 그대로 붙여넣어야 한다")
    if v.endswith("=="):
        return ("`==` 로 끝난다 — *Decoding* 키다. 우리 코드는 URL 에 그대로 이어 붙이므로 "
                "*Encoding* 키(`%3D` 로 끝남)를 쓴다")
    return None


def _hf(v: str) -> Optional[str]:
    if not v.startswith("hf_"):
        return "`hf_` 로 시작하지 않는다 — HuggingFace 토큰이 맞는지 본다"
    return None


# ══════════════════════════════════════════════════════════════════════════
# 실호출 — 정말 되는지 한 번씩 때려본다
# ══════════════════════════════════════════════════════════════════════════


def _get(url: str, timeout: int = 15) -> tuple:
    req = urllib.request.Request(url, headers={"User-Agent": "AlphaStack/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")[:600]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")[:600]
    except Exception as e:                                   # noqa: BLE001
        return 0, f"{type(e).__name__}: {e}"


def _호출_공공데이터포털(v: str) -> str:
    # 🔴 이미 인코딩된 키라 다시 인코딩하지 않는다. 나머지 파라미터만 인코딩한다.
    q = urllib.parse.urlencode({"numOfRows": "1", "resultType": "json"})
    코드, 본문 = _get("https://apis.data.go.kr/1160100/service/"
                      f"GetKrxListedInfoService/getItemInfo?serviceKey={v}&{q}")
    if "SERVICE_KEY_IS_NOT_REGISTERED" in 본문:
        return "🔴 등록되지 않은 키다"
    if "SERVICE_ACCESS_DENIED" in 본문:
        return "🔒 키는 유효하나 이 API 활용신청이 안 됐다 (포털에서 신청 버튼을 누른다)"
    if "LIMITED_NUMBER" in 본문:
        return "⚠️ 오늘 호출 한도를 다 썼다 (내일 풀린다)"
    if '"totalCount"' in 본문 or "<totalCount>" in 본문:
        return "✅ 된다"
    return f"❔ 알 수 없는 응답 (HTTP {코드})"


def _호출_dart(v: str) -> str:
    코드, 본문 = _get(f"https://opendart.fss.or.kr/api/list.json?crtfc_key={v}"
                      "&bgn_de=20240101&end_de=20240102&page_count=1")
    for 사인, 말 in (('"013"', "✅ 된다 (조회 결과가 없을 뿐이다)"),
                     ('"000"', "✅ 된다"),
                     ('"010"', "🔴 등록되지 않은 키다"),
                     ('"011"', "🔒 사용할 수 없는 키다 (오픈API 이용 등록 확인)"),
                     ('"020"', "⚠️ 오늘 호출 한도를 다 썼다")):
        if 사인 in 본문:
            return 말
    return f"❔ 알 수 없는 응답 (HTTP {코드})"


def _호출_ecos(v: str) -> str:
    코드, 본문 = _get(f"https://ecos.bok.or.kr/api/StatisticTableList/{v}/json/kr/1/1")
    if "INFO-100" in 본문:
        return "🔴 인증키가 유효하지 않다"
    if "INFO-200" in 본문:
        return "✅ 된다 (해당 자료가 없을 뿐이다)"
    if "StatisticTableList" in 본문:
        return "✅ 된다"
    return f"❔ 알 수 없는 응답 (HTTP {코드})"


def _호출_fred(v: str) -> str:
    코드, 본문 = _get(f"https://api.stlouisfed.org/fred/series?series_id=GDP&api_key={v}"
                      "&file_type=json")
    if 코드 == 400 and "api_key" in 본문:
        return "🔴 인증키가 유효하지 않다"
    if '"seriess"' in 본문:
        return "✅ 된다"
    return f"❔ 알 수 없는 응답 (HTTP {코드})"


def _호출_hf(v: str) -> str:
    req = urllib.request.Request("https://huggingface.co/api/whoami-v2",
                                 headers={"Authorization": f"Bearer {v}"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return "✅ 된다" if r.status == 200 else f"❔ HTTP {r.status}"
    except urllib.error.HTTPError as e:
        return "🔴 토큰이 유효하지 않다" if e.code == 401 else f"❔ HTTP {e.code}"
    except Exception as e:                                   # noqa: BLE001
        return f"❔ {type(e).__name__}"


def _호출_krx(v: str) -> str:
    req = urllib.request.Request(
        "https://data-dbg.krx.co.kr/svc/apis/sto/stk_bydd_trd?basDd=20240830",
        headers={"AUTH_KEY": v, "User-Agent": "AlphaStack/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            본문 = r.read().decode("utf-8", "replace")[:400]
        if "OutBlock" in 본문:
            return "✅ 된다"
        return f"❔ 알 수 없는 응답 ({본문[:60]!r})"
    except urllib.error.HTTPError as e:
        본문 = e.read().decode("utf-8", "replace")[:300]
        if "Unauthorized API Call" in 본문:
            return ("🔒 키는 유효하나 이 서비스 **이용신청이 승인되지 않았다** — "
                    "인증키 발급과 서비스별 이용신청은 별개의 2단계다")
        return f"❔ HTTP {e.code} ({본문[:60]!r})"
    except Exception as e:                                   # noqa: BLE001
        return f"❔ {type(e).__name__}"


def _호출_kosis(v: str) -> str:
    코드, 본문 = _get("https://kosis.kr/openapi/statisticsList.do?method=getList"
                      f"&apiKey={v}&vwCd=MT_ZTITLE&parentListId=A_7&format=json&jsonVD=Y")
    if "인증" in 본문 and ("실패" in 본문 or "오류" in 본문):
        return "🔴 인증키가 유효하지 않다"
    if "ERR" in 본문[:200] and "err" in 본문.lower():
        return f"❔ 오류 응답 ({본문[:70]!r})"
    if 본문.strip().startswith(("[", "{")):
        return "✅ 된다"
    return f"❔ 알 수 없는 응답 (HTTP {코드})"


def _호출_fss(v: str) -> str:
    코드, 본문 = _get("https://finlife.fss.or.kr/finlifeapi/depositProductsSearch.json"
                      f"?auth={v}&topFinGrpNo=020000&pageNo=1")
    if '"err_cd":"000"' in 본문.replace(" ", ""):
        return "✅ 된다"
    if '"err_cd"' in 본문:
        import re as _re
        m = _re.search(r'"err_msg"\s*:\s*"([^"]+)"', 본문)
        return f"🔴 {m.group(1)}" if m else f"🔴 오류 응답 (HTTP {코드})"
    return f"❔ 알 수 없는 응답 (HTTP {코드})"


# ══════════════════════════════════════════════════════════════════════════
# 우리가 쓰는 키 전부
# ══════════════════════════════════════════════════════════════════════════

키들: List[키정의] = [
    키정의("KRX_API_KEY", ("KRX_AUTH_KEY",), "KRX OpenAPI — 시세·지수 수집", True,
          "https://openapi.krx.co.kr",
          "학습 데이터 자체가 없다. 1차 범위가 통째로 막힌다",
          _길이(30, 60), _호출_krx),
    키정의("DATA_GO_KR_API_KEY", (f"DATA_GO_KR_API_KEY{i}" for i in range(1, 6)),
          "공공데이터포털 — 종목 마스터·기업기본·재무", False,
          "https://www.data.go.kr",
          "공공데이터포털 수집 경로만 막힌다 (다른 수집원은 그대로 돈다)",
          _공공데이터포털, _호출_공공데이터포털),
    키정의("DART_API_KEY", ("DART_KEY",), "OpenDART — 공시·재무제표", False,
          "https://opendart.fss.or.kr/uss/umt/EgovMberInsertView.do",
          "재무 피처와 공시 수집이 막힌다",
          _길이(40, 40), _호출_dart),
    키정의("ECOS_API_KEY", ("ECOS_KEY", "BOK_API_KEY"), "한국은행 — 국내 거시지표", False,
          "https://ecos.bok.or.kr/api/#/AuthKeyApply",
          "금리·환율·물가 같은 거시 피처가 막힌다",
          _길이(20, 40), _호출_ecos),
    키정의("FRED_API_KEY", ("FRED_KEY",), "세인트루이스 연준 — 미국 거시지표", False,
          "https://fredaccount.stlouisfed.org/apikeys",
          "미국 금리·물가 피처가 막힌다",
          _길이(32, 32), _호출_fred),
    키정의("KOSIS_API_KEY", ("KOSIS_APIKEY",), "국가통계포털 — 통계청 지표", False,
          "https://kosis.kr/openapi",
          "통계청 지표 조회가 막힌다", None, _호출_kosis),
    키정의("FINANCE_SUPERVISORY_API_KEY", ("FSS_API_KEY", "FSS_KEY"),
          "금융감독원 금융상품통합비교공시", False,
          "https://finlife.fss.or.kr/finlifeapi",
          "예·적금 금리 조회가 막힌다", None, _호출_fss),
    키정의("HUGGINGFACE_ACCESS_TOKEN", ("HF_TOKEN", "HUGGINGFACE_API_KEY"),
          "HuggingFace — 팀 데이터셋 주고받기·무료 인코더", False,
          "https://huggingface.co/settings/tokens",
          "팀 배포본을 받거나 올릴 수 없다", _hf, _호출_hf),
    키정의("NAVER_API_CLIENT_ID", ("NAVER_CLIENT_ID",), "네이버 검색 — 뉴스·카페글", False,
          "🚨 2026-07-30 부로 신규 발급 중단. NAVER API HUB 로 가야 한다",
          "뉴스·카페글 수집이 막힌다 (감성 피처)", None, None),
    키정의("NAVER_API_SECRET_KEY", ("NAVER_CLIENT_SECRET",), "위와 한 쌍", False,
          "위와 같다", "위와 같다", None, None),
]


def main() -> int:
    p = argparse.ArgumentParser(description="내 API 키가 제대로 들어갔는지 확인한다")
    p.add_argument("--live", action="store_true",
                   help="실제로 한 번씩 호출해 본다 (하루 호출 한도를 키마다 1회 쓴다)")
    args = p.parse_args()

    print("── API 키 점검 ──")
    print(f"  읽는 순서: 환경변수 → {secrets.ENV_FILE.name} → {secrets.KEY_FILE.name}")
    print("  🔴 키 값은 출력하지 않는다 (길이와 앞 두 글자만)\n")

    없음, 이상, 정상 = [], [], []
    for k in 키들:
        값, 출처 = secrets.load_key([k.이름, *k.별칭])
        표시 = f"{k.이름:30s}"
        if not 값:
            없음.append(k)
            딱지 = "필수" if k.필수 else "선택"
            print(f"  ❌ {표시} 없음 ({딱지})")
            continue

        문제 = k.형식(값) if k.형식 else None
        가림 = f"{값[:2]}…{len(값)}자"
        if 문제:
            이상.append((k, 문제))
            print(f"  ⚠️ {표시} {가림}  ({출처})")
            print(f"     └ {문제}")
        else:
            정상.append(k)
            print(f"  ✅ {표시} {가림}  ({출처})")

        if args.live and k.실호출:
            print(f"     └ 실호출: {k.실호출(값)}")

    print(f"\n  정상 {len(정상)} · 모양 이상 {len(이상)} · 없음 {len(없음)}")

    if 없음:
        print("\n── 없는 키로 무엇이 막히나 ──")
        for k in 없음:
            print(f"  {k.이름}")
            print(f"    무엇   : {k.무엇}")
            print(f"    막히는 것: {k.없으면}")
            print(f"    발급   : {k.발급}")

    if not args.live:
        print("\n  키가 실제로 통하는지까지 보려면 --live 를 준다.")
    print("  자세한 발급 절차는 docs/데이터파트/version3.1/API키_발급_가이드.md")

    # 필수 키가 없을 때만 실패로 친다. 선택 키는 없어도 정상이다.
    return 1 if any(k.필수 for k in 없음) else 0


if __name__ == "__main__":
    raise SystemExit(main())
