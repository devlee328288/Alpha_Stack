#!/usr/bin/env python3
"""core 유니버스(`data/universe_core.json`) 생성 스크립트 — KOSPI 200 + KOSDAQ 150

왜 이 파일이 필요한가
--------------------
`securities.universe_tier` 는 DDL 이 `core`/`full` 두 값을 요구하는데(01-schema.sql:46-47)
실측은 **2,875종목 전부 `full`** 이었다. 구성종목 목록이 레포에 없었기 때문이다
(ADR-DS-0014 — "추정을 사실로 굳히지 않는다"). 그래서 ADR-DS-0020 의 일괄 수집은
**시가총액 상위 N 을 대용**으로 쓰면서 "이것은 core 가 아니다" 를 로그·watermark·화면
셋 다에 실어야 했다(dart_collector.universe()). 이 파일이 그 대용을 걷어낸다.

⚠️ 왜 KRX OpenAPI(`.key` 의 `KRX_API_KEY`)로 못 하는가 — **서비스가 없다**
--------------------------------------------------------------------------
실측(2026-08-25, 승인된 키로):

    sto/stk_bydd_trd       ✅ 942행  일별매매          — 지수 소속 필드 없음
    sto/stk_isu_base_info  ✅ 942행  종목 기본정보      — 지수 소속 필드 없음
    idx/kospi_dd_trd       ✅  51행  코스피 지수 시세   — 지수 **자체**의 종가다
    idx/kosdaq_dd_trd      ✅  40행  코스닥 지수 시세   — 같음
    idx/idx_isu_base_info  ❌ 404    "API referenced by the path does not exist"

`idx/*` 는 "코스피 200 이 오늘 몇 포인트인가" 를 주지 **"거기 무슨 종목이 들었나" 를 주지
않는다.** 그 필드가 스펙에 없다. 그래서 이 스크립트만 다른 문(정보데이터시스템)을 쓴다.

⚠️ 계정이 필요하다 — `KRX_ID` · `KRX_PW`
----------------------------------------
data.krx.co.kr 의 `MDCSTAT00601`(지수구성종목)은 **로그인 세션을 요구한다.** 익명으로
워밍업해 `JSESSIONID` 를 받아도 400 이다(실측). 파라미터 문제가 아니다 — pykrx 가 보내는
것과 **정확히 같은 네 개**(`bld`·`indIdx`·`indIdx2`·`trdDd`)를 같은 URL 로 보내도 400 이다.

⚠️ 왜 pykrx 를 의존성으로 들이지 않는가
--------------------------------------
필요한 것은 HTTP 두 번(로그인 · 조회)인데 pykrx 는 pandas 를 끌고 온다. 이 레포의
`ingest/clients/*.py` 아홉은 **전부 표준 라이브러리 urllib** 이고(절대제약 5 — 이미지 용량),
여기도 같은 규칙을 따른다. 프로토콜 자체는 pykrx 1.2.8 의 구현을 읽어 확인했다.

사용법
------
    python3 scripts/build_universe.py                # 최근 거래일 기준으로 새로 만든다
    python3 scripts/build_universe.py --check        # 만들지 않고 현재 파일 상태만 본다
    python3 scripts/build_universe.py --date 20260824
    python3 scripts/build_universe.py --force        # 종목 수가 예상 밖이어도 쓴다

⚠️ **구성종목은 바뀐다.** 정기변경이 연 2회(6월·12월)이고 상장폐지·합병에 따른 수시변경도
있다. 낡으면 배포본 유니버스가 지수와 어긋나는데 **오류로 뜨지 않는다** — `--check` 가
그것만 잰다. corp_code.json 과 같은 취급이다(사람이 갱신 · 낡으면 알리기만).
"""

from __future__ import annotations

import argparse
import http.cookiejar
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

# 이 스크립트는 scripts/ 안에 있다. `app.*` 를 찾으려면 루트를 import 경로에 넣어야 한다.
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from common import secrets  # noqa: E402  (경로 설정 뒤에 import)
from common.paths import DATA_DIR  # noqa: E402

OUTPUT_PATH = DATA_DIR / "universe_core.json"
KST = timezone(timedelta(hours=9))

# 낡음 기준. corp_code 는 7일인데 이쪽은 90일이다 — 정기변경이 연 2회라 그보다 촘촘히
# 재촉하면 경고가 소음이 된다. ⚠️ 다만 **수시변경은 그보다 잦다**(상장폐지·합병).
# 90일을 "안전하다" 로 읽지 않는다. 이 값은 재촉 주기이지 정확성 보증이 아니다.
STALE_DAYS = 90

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
LOGIN_PAGE = "https://data.krx.co.kr/contents/MDC/COMS/client/MDCCOMS001.cmd"
LOGIN_JSP = "https://data.krx.co.kr/contents/MDC/COMS/client/view/login.jsp?site=mdc"
LOGIN_URL = "https://data.krx.co.kr/contents/MDC/COMS/client/MDCCOMS001D1.cmd"
DATA_URL = "https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
BLD_CONSTITUENTS = "dbms/MDC/STAT/standard/MDCSTAT00601"

# 지수 정의. pykrx 의 티커 표기(`1028`·`2203`)에서 앞 한 자가 group_id, 나머지가 지수코드다
# (wrap.py:1267 — `fetch(date, ticker[1:], ticker[0])`).
#
# `expected` 는 **지수 이름에 박힌 종목 수**다. 실제로 그 수가 나오는지 확인하는 데만 쓴다 —
# 편입·제외 사이 며칠은 어긋날 수 있으므로 틀리면 막지 않고 **경고하고 --force 를 요구**한다.
# 조용히 반쪽짜리 파일을 쓰는 것이 이 스크립트의 가장 비싼 고장이다.
INDICES = (
    {"key": "KOSPI200", "name": "코스피 200", "group_id": "1", "index_code": "028",
     "market": "KOSPI", "expected": 200},
    {"key": "KOSDAQ150", "name": "코스닥 150", "group_id": "2", "index_code": "203",
     "market": "KOSDAQ", "expected": 150},
)

KEY_NAMES_ID = ("KRX_ID",)
KEY_NAMES_PW = ("KRX_PW",)


class KrxIndexError(RuntimeError):
    """이 스크립트가 스스로 판단해 멈출 때 쓰는 예외. 메시지에 **할 일**까지 담는다."""


# ==================================================
# 1. 자격증명
# ==================================================
def load_credentials() -> tuple[str, str, str]:
    """`KRX_ID`·`KRX_PW` 와 그 출처. 환경변수 → `.env` → `.key` 순서는 secrets 가 정한다.

    ⚠️ 이것은 `KRX_API_KEY`(OpenAPI 인증키)와 **다른 자격증명**이다. 같은 KRX 이지만
    문이 둘이고, OpenAPI 쪽에는 구성종목 서비스가 아예 없다(모듈 docstring 참조).
    """
    user, src_id = secrets.load_key(KEY_NAMES_ID)
    password, _src_pw = secrets.load_key(KEY_NAMES_PW)
    return user, password, src_id


def _credentials_or_die() -> tuple[str, str]:
    user, password, source = load_credentials()
    if not user or not password:
        # 환경 가드는 막다른 길로 만들지 않는다 — 무엇을 해야 하는지까지 말한다.
        raise KrxIndexError(
            "KRX 정보데이터시스템 자격증명이 없다.\n"
            "  ① https://data.krx.co.kr 에서 회원가입한다 (무료).\n"
            "  ② 프로젝트 루트 `.key` 에 두 줄을 더한다:\n"
            "       KRX_ID=<아이디>\n"
            "       KRX_PW=<비밀번호>\n"
            "  ⚠️ `.key` 의 KRX_API_KEY(OpenAPI 인증키)로는 안 된다 — 그쪽에는\n"
            "     지수구성종목 서비스가 없다(404). 문이 둘이다."
        )
    print(f"  자격증명 출처: {source}")
    return user, password


# ==================================================
# 2. 로그인
# ==================================================
def _opener() -> urllib.request.OpenerDirector:
    jar = http.cookiejar.CookieJar()
    return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))


def login(user: str, password: str) -> urllib.request.OpenerDirector:
    """로그인하고 **쿠키를 문 opener** 를 돌려준다.

    흐름은 pykrx 1.2.8 `website/comm/auth.py` 의 것을 그대로 따른다 —
      ① GET MDCCOMS001.cmd    초기 JSESSIONID 발급
      ② GET login.jsp         iframe 세션 초기화
      ③ POST MDCCOMS001D1.cmd 실제 로그인
    응답의 `_error_code` 가 `CD001` 이면 성공이다.
    """
    opener = _opener()

    for url, referer in ((LOGIN_PAGE, None), (LOGIN_JSP, LOGIN_PAGE)):
        headers = {"User-Agent": USER_AGENT}
        if referer:
            headers["Referer"] = referer
        with opener.open(urllib.request.Request(url, headers=headers), timeout=20):
            pass

    payload = {"mbrNm": "", "telNo": "", "di": "", "certType": "", "mbrId": user, "pw": password}
    headers = {"User-Agent": USER_AGENT, "Referer": LOGIN_PAGE}

    def _post(body: dict) -> dict:
        request = urllib.request.Request(
            LOGIN_URL, data=urllib.parse.urlencode(body).encode(), headers=headers, method="POST")
        with opener.open(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))

    data = _post(payload)
    code = data.get("_error_code", "")
    message = data.get("_error_message", "")

    if code == "CD010":
        raise KrxIndexError(
            "KRX 가 비밀번호 변경을 요구한다.\n"
            f"  응답: {message}\n"
            "  https://data.krx.co.kr 에서 비밀번호를 바꾼 뒤 `.key` 의 KRX_PW 도 함께 고친다."
        )

    # CD011 = 다른 곳에서 이미 로그인돼 있다. 밀어내고 들어간다.
    if code == "CD011":
        print("  이미 로그인된 세션이 있어 밀어내고 들어간다 (CD011 → skipDup).")
        data = _post({**payload, "skipDup": "Y"})
        code = data.get("_error_code", "")
        message = data.get("_error_message", "")

    if code != "CD001":
        raise KrxIndexError(
            f"KRX 로그인 실패 (_error_code={code!r}).\n"
            f"  응답: {message}\n"
            "  `.key` 의 KRX_ID·KRX_PW 를 확인한다. 웹에서 직접 로그인해 계정 상태도 본다."
        )

    print("  KRX 로그인 성공.")
    return opener


# ==================================================
# 3. 구성종목 조회
# ==================================================
def fetch_index(opener, group_id: str, index_code: str, trd_dd: str) -> list[dict]:
    """지수 하나의 구성종목. 휴장일이면 빈 목록이다(오류가 아니다)."""
    body = urllib.parse.urlencode({
        "bld": BLD_CONSTITUENTS,
        "indIdx": group_id,
        "indIdx2": index_code,
        "trdDd": trd_dd,
    }).encode()
    request = urllib.request.Request(DATA_URL, data=body, method="POST", headers={
        "User-Agent": USER_AGENT,
        "Referer": "https://data.krx.co.kr/contents/MDC/MDI/outerLoader/index.cmd",
        "X-Requested-With": "XMLHttpRequest",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    })
    try:
        with opener.open(request, timeout=25) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise KrxIndexError(
            f"구성종목 조회가 HTTP {error.code} 로 실패했다 (지수 {group_id}{index_code}).\n"
            "  400 이면 대개 세션이 끊긴 것이다 — 다시 실행하면 새로 로그인한다."
        ) from error

    rows = payload.get("output") or []
    out = []
    for row in rows:
        code = (row.get("ISU_SRT_CD") or "").strip()
        if not code:
            continue
        out.append({"code": code, "name": (row.get("ISU_ABBRV") or "").strip()})
    return out


def _recent_dates(start: str | None, back_days: int) -> list[str]:
    """조회해 볼 날짜들. 오늘부터 거슬러 올라간다 — 휴장일에 돌려도 되게."""
    if start:
        base = datetime.strptime(start, "%Y%m%d")
    else:
        base = datetime.now(KST).replace(tzinfo=None)
    return [(base - timedelta(days=n)).strftime("%Y%m%d") for n in range(back_days + 1)]


# ==================================================
# 4. 만들기
# ==================================================
def build(start: str | None, back_days: int, force: bool) -> dict:
    user, password = _credentials_or_die()
    opener = login(user, password)

    for trd_dd in _recent_dates(start, back_days):
        collected: dict[str, dict] = {}
        per_index: dict[str, int] = {}
        empty = False

        for spec in INDICES:
            rows = fetch_index(opener, spec["group_id"], spec["index_code"], trd_dd)
            if not rows:
                empty = True
                break
            per_index[spec["key"]] = len(rows)
            for row in rows:
                previous = collected.get(row["code"])
                if previous:
                    # 한 종목이 두 지수에 동시에 들 수는 없다. 그래도 조용히 덮지 않는다.
                    print(f"  ⚠️ {row['code']} 가 {previous['index']} 와 {spec['key']} 양쪽에 있다.")
                    continue
                collected[row["code"]] = {
                    "name": row["name"],
                    "index": spec["key"],
                    "market": spec["market"],
                }

        if empty:
            print(f"  {trd_dd}: 빈 응답 (휴장일로 보인다). 하루 앞으로 간다.")
            continue

        print(f"  기준일 {trd_dd} — " + " · ".join(f"{k} {v}" for k, v in per_index.items()))
        _sanity_check(per_index, force)

        return {
            "generated_at": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST"),
            "source": "KRX 정보데이터시스템 지수구성종목 (MDCSTAT00601)",
            "trade_date": trd_dd,
            "indices": {
                spec["key"]: {
                    "name": spec["name"],
                    "ticker": f"{spec['group_id']}{spec['index_code']}",
                    "market": spec["market"],
                    "count": per_index[spec["key"]],
                }
                for spec in INDICES
            },
            "count": len(collected),
            "codes": dict(sorted(collected.items())),
        }

    raise KrxIndexError(
        f"최근 {back_days + 1}일 안에 구성종목이 있는 거래일을 찾지 못했다.\n"
        "  --date 로 거래일을 직접 지정하거나 --back-days 를 늘린다."
    )


def _sanity_check(per_index: dict[str, int], force: bool) -> None:
    """종목 수가 지수 이름과 맞는지 본다. **조용히 반쪽을 쓰지 않기 위한 것이다.**

    편입·제외가 걸친 며칠은 한둘 어긋날 수 있으므로 ±5 까지는 넘어간다.
    그보다 크게 벌어지면 응답이 잘렸거나 지수코드가 바뀐 것이다.
    """
    problems = []
    for spec in INDICES:
        got = per_index.get(spec["key"], 0)
        if abs(got - spec["expected"]) > 5:
            problems.append(f"{spec['key']} 는 {spec['expected']}종목이어야 하는데 {got}종목이다")
    if not problems:
        return
    detail = "\n".join(f"    · {p}" for p in problems)
    if force:
        print(f"  ⚠️ 종목 수가 예상 밖이지만 --force 라 그대로 쓴다:\n{detail}")
        return
    raise KrxIndexError(
        "구성종목 수가 예상과 크게 다르다. 반쪽짜리 파일을 쓰지 않고 멈춘다.\n"
        f"{detail}\n"
        "  응답이 잘렸거나 지수코드가 바뀌었을 수 있다. 확인 뒤 정말 맞다면 --force 를 붙인다."
    )


# ==================================================
# 5. 읽기 · 상태
# ==================================================
def load_core(path: Path = OUTPUT_PATH) -> dict:
    """만들어 둔 파일을 읽는다. 없거나 깨졌으면 예외 — **빈 집합으로 때우지 않는다.**

    빈 집합으로 때우면 "core 유니버스 0종목" 이 그럴듯한 결과가 되어 배포본이
    조용히 빈 화면이 된다. 없으면 없다고 크게 말한다.
    """
    if not path.exists():
        raise KrxIndexError(
            f"{path.name} 이 없다. 먼저 만든다:\n"
            "    python3 scripts/build_universe.py"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("codes"):
        raise KrxIndexError(f"{path.name} 에 codes 가 비어 있다. 다시 만든다.")
    return payload


def state(path: Path = OUTPUT_PATH) -> dict:
    """파일의 나이와 규모. **막지 않고 알리는** 용도다 (corp_code_state() 와 같은 모양)."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:  # noqa: BLE001  (없어도 이 스크립트만 못 돌고 앱은 뜬다)
        return {"exists": False, "age_days": None, "count": 0, "stale": True,
                "hint": f"{path.name} 을 읽지 못했다: {error}"}

    age = None
    try:
        made = datetime.strptime(payload.get("generated_at", "").replace(" KST", ""),
                                 "%Y-%m-%d %H:%M:%S")
        age = (datetime.now(KST).replace(tzinfo=None) - made).days
    except ValueError:
        pass

    stale = age is None or age > STALE_DAYS
    return {
        "exists": True,
        "generated_at": payload.get("generated_at"),
        "trade_date": payload.get("trade_date"),
        "age_days": age,
        "count": len(payload.get("codes") or {}),
        "indices": payload.get("indices") or {},
        "stale": stale,
        "hint": ("구성종목이 낡았을 수 있다 — python3 scripts/build_universe.py 로 다시 만든다."
                 if stale else ""),
    }


def codes(path: Path = OUTPUT_PATH) -> set[str]:
    """종목코드 집합만 필요할 때. 적재기가 이것을 쓴다."""
    return set(load_core(path)["codes"])


# ==================================================
# 6. CLI
# ==================================================
def _print_state() -> int:
    info = state()
    if not info["exists"]:
        print(f"❌ {OUTPUT_PATH.name} 없음 — {info['hint']}")
        return 1
    print(f"파일     : {OUTPUT_PATH}")
    print(f"생성     : {info['generated_at']}  (기준일 {info['trade_date']})")
    print(f"나이     : {info['age_days']}일  (낡음 기준 {STALE_DAYS}일)")
    print(f"종목 수  : {info['count']}")
    for key, meta in (info["indices"] or {}).items():
        print(f"   {key:10s} {meta.get('name','')} — {meta.get('count')}종목")
    if info["stale"]:
        print(f"⚠️ {info['hint']}")
        return 1
    print("✅ 최신이다.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="KOSPI200 + KOSDAQ150 구성종목을 받아 data/universe_core.json 을 만든다.")
    parser.add_argument("--check", action="store_true",
                        help="만들지 않고 현재 파일의 나이·규모만 본다 (네트워크를 타지 않는다)")
    parser.add_argument("--date", metavar="YYYYMMDD",
                        help="기준 거래일. 생략하면 오늘부터 거슬러 올라가며 찾는다")
    parser.add_argument("--back-days", type=int, default=7,
                        help="빈 응답일 때 거슬러 올라갈 최대 일수 (기본 7)")
    parser.add_argument("--force", action="store_true",
                        help="종목 수가 예상과 크게 달라도 파일을 쓴다")
    args = parser.parse_args()

    if args.check:
        return _print_state()

    try:
        payload = build(args.date, args.back_days, args.force)
    except KrxIndexError as error:
        print(f"\n❌ {error}", file=sys.stderr)
        return 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"\n✅ {OUTPUT_PATH} 를 썼다 — {payload['count']}종목 · {size_kb:.1f}KB")
    print("   ⚠️ 커밋은 사람이 한다. push 가 곧 배포다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
