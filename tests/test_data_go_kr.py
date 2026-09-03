"""공공데이터포털 클라이언트 — 조용히 틀리는 자리를 잠근다.

이 수집원이 낼 수 있는 사고는 전부 **에러 없이 값만 어긋나는** 종류다.

  · 서비스키를 한 번 더 인코딩하면 `%3D` → `%253D` 가 되고 "등록되지 않은 키" 로 온다
  · `A` 접두사를 안 떼면 조인이 0행이 되는데 **조인은 0행이어도 에러가 아니다**
  · 코드를 숫자로 단정하면 신형우선주 84종이 통째로 사라진다
  · 두 자리 연도를 잘못 풀면 1976년 상장이 2076년이 된다

그래서 여기서 재는 것은 "함수가 도는가" 가 아니라 **"틀린 값을 내놓지 않는가"** 다.
망을 타지 않는다 — 응답은 실측해 둔 모양 그대로 손으로 만든 것을 쓴다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ingest.clients import data_go_kr as dgk  # noqa: E402

#: 실측한 KRX상장종목정보 한 줄 (2026-09-03 · 동화약품).
목록_한줄 = {
    "basDt": "20240822",
    "srtnCd": "A000020",
    "isinCd": "KR7000020008",
    "crno": "1101110043870",
    "corpNm": "동화약품(주)",
    "itmsNm": "동화약품",
    "mrktCtg": "KOSPI",
}

#: 실측한 기업기본정보 한 줄. 날짜 세 모양이 그대로 들어 있다.
개요_한줄 = {
    "crno": "1101110043870",
    "corpNm": "동화약품(주)",
    "sicNm": "",
    "enpEstbDt": "18970925",          # YYYYMMDD
    "enpStacMm": "12",
    "enpXchgLstgDt": "76/03/24",      # YY/MM/DD ← 두 자리 연도
    "enpXchgLstgAbolDt": "",
    "enpKosdaqLstgDt": "",
    "enpKosdaqLstgAbolDt": "",
    "audtRptOpnnCtt": "적정의견",
    "actnAudpnNm": "한울회계법인",
    "enpEmpeCnt": "838",
    "enpPn1AvgSlryAmt": "72000000",
    "smenpYn": "",
    "fstOpegDt": "20260319",
    "lastOpegDt": "20260901",
    "fssCorpChgDtm": "2025/08/07",    # YYYY/MM/DD
}


# ==================================================
# 1. 서비스키 — 두 번 인코딩하지 않는다
# ==================================================
def test_서비스키가_다시_인코딩되지_않는다():
    """🔴 이게 이 파일에서 가장 중요한 시험이다.

    Encoding 키에는 `%3D` 가 들어 있다. `urlencode` 에 키까지 넣으면 `%253D` 가 되고,
    포털은 "키가 없다" 가 아니라 **"등록되지 않은 키"** 로 답한다 — 그러면 키를 새로
    발급받는 헛수고를 한다.
    """
    키 = "abc%2Bdef%3D%3D"
    url = dgk._build_url(dgk.EP_LISTED, 키, {"basDt": "20240822"})

    assert f"serviceKey={키}" in url
    assert "%253D" not in url, "서비스키가 두 번 인코딩됐다"
    assert "%252B" not in url


def test_나머지_파라미터는_인코딩된다():
    """키만 예외다 — 나머지까지 날것으로 두면 값에 `&` 가 있을 때 깨진다."""
    url = dgk._build_url(dgk.EP_LISTED, "k", {"corpNm": "동화 & 약품"})
    assert "%26" in url or "%20" in url
    assert "동화 & 약품" not in url


def test_None_인_파라미터는_보내지_않는다():
    url = dgk._build_url(dgk.EP_LISTED, "k", {"basDt": "20240822", "crno": None})
    assert "crno" not in url


# ==================================================
# 2. 종목코드 — 접두사와 영문
# ==================================================
def test_A_접두사를_뗀다():
    assert dgk.strip_code_prefix("A000020") == "000020"


def test_영문이_낀_코드가_살아남는다():
    """5·6번째에 영문이 오는 종목이 84종 있다. 숫자로 단정하면 통째로 사라진다."""
    for 코드 in ("A0001A0", "A00088K", "A0004V0"):
        뗀것 = dgk.strip_code_prefix(코드)
        assert 뗀것 == 코드[1:]
        assert len(뗀것) == 6


def test_접두사가_없으면_그대로_둔다():
    """규격이 조용히 바뀌었을 때 전량을 격리하는 것보다 낫다."""
    assert dgk.strip_code_prefix("000020") == "000020"


def test_빈_코드는_None():
    assert dgk.strip_code_prefix("") is None
    assert dgk.strip_code_prefix(None) is None
    assert dgk.strip_code_prefix("   ") is None


# ==================================================
# 3. 날짜 — 한 응답에 세 모양
# ==================================================
@pytest.mark.parametrize("원문, 기대", [
    ("18970925", "18970925"),        # YYYYMMDD 그대로
    ("2025/08/07", "20250807"),      # YYYY/MM/DD
    ("76/03/24", "19760324"),        # YY/MM/DD → 1976 (76 >= 56)
    ("23/02/21", "20230221"),        # YY/MM/DD → 2023 (23 < 56)
    ("55/12/31", "20551231"),        # 경계 바로 아래
    ("56/01/01", "19560101"),        # 경계 — KRX 개장 해
    ("2020-01-02", "20200102"),      # 하이픈도 받는다
])
def test_날짜를_YYYYMMDD_로_맞춘다(원문, 기대):
    assert dgk.normalize_date(원문) == 기대


@pytest.mark.parametrize("원문", ["", "   ", None, "2026", "abc", "202601021"])
def test_읽을_수_없는_날짜는_지어내지_않는다(원문):
    """🔴 아무 값이나 넣으면 그 뒤로 아무도 못 찾는다. 비면 눈에 띈다."""
    assert dgk.normalize_date(원문) is None


@pytest.mark.parametrize("원문", ["00010101", "0001-01-01", "11111111", "00000000"])
def test_자리표시자는_날짜가_아니다(원문):
    """🔴 포털은 "해당 없음" 을 `00010101` 로 준다 — 여덟 자리라 모양은 멀쩡하다.

    코스닥에만 상장한 회사의 `유가상장일` 자리에 이 값이 온다. 그냥 두면
    *"서기 1년에 상장한 회사"* 가 생기고, 상장 경과일 같은 계산이 **에러 없이**
    틀어진다. 실측(2026-09-03 · 법인 3,142곳)에서 유가 51행 · 코스닥 42행이었다.
    `11111111` 은 쓰레기값 1행 — 날짜 모양이라 자릿수 검사로는 안 걸린다.
    """
    assert dgk.normalize_date(원문) is None


def test_국내_최고령_법인의_설립일은_살아남는다():
    """바닥선을 1900 으로 올리면 진짜 값이 사라진다.

    동화약품은 1897-09-25 창업으로 국내 최고령 등록법인이고, 우리 자료에 16행 있다.
    자리표시자를 거르겠다고 바닥선을 올리면 이 값이 함께 죽는다 —
    **거르는 것과 잃는 것을 가르는 자리**라 상수로 못 박고 여기서 지킨다.
    """
    assert dgk.normalize_date("18970925") == "18970925"
    assert dgk._EARLIEST_PLAUSIBLE < "18970925"


def test_KRX_개장일은_통과한다():
    """실측 최솟값이 정확히 이 날이다 — 우리가 옳게 걸렀다는 증거다."""
    assert dgk.normalize_date("19560303") == "19560303"


def test_두_자리_연도_경계가_해마다_움직이지_않는다():
    """'올해 기준' 규칙이면 내년에 같은 원문이 다른 값이 된다 — 그러면 안 된다."""
    assert dgk._YY_PIVOT == 56
    assert dgk.normalize_date("76/03/24") == "19760324"


# ==================================================
# 4. 정수 — 0 과 모름을 섞지 않는다
# ==================================================
def test_영뿐인_법인등록번호는_번호가_아니다():
    """🔴 조인이 0행이 되는 실수는 눈에 띈다. **틀린 짝이 붙는 실수는 안 보인다.**

    포털은 외국기업의 `crno` 에 `0000000000000` 을 준다 — 실측(2026-09-03)에서
    외국기업 **20종이 이 번호 하나를 공유**했다(3,349행). 그대로 두면
    `corp_profile` 과 조인할 때 서로 다른 20개 회사가 같은 법인에 붙는다.
    실제로 그 번호로 받아 둔 법인 개요에는 헝셩그룹유한회사와
    자프코 아시아 테크놀러지 펀드 3 이 섞여 있었다.

    정상적으로 한 번호를 공유하는 최대는 **2종**(키움증권 보통주·우선주)이다.
    """
    assert dgk.normalize_crno("0000000000000") is None
    assert dgk.normalize_crno("000") is None
    assert dgk.normalize_crno("") is None
    assert dgk.normalize_crno(None) is None
    assert dgk.normalize_crno("1101111867948") == "1101111867948"
    assert dgk.normalize_crno(" 1101111867948 ") == "1101111867948"


def test_숫자가_아닌_법인등록번호는_받지_않는다():
    """실측에서는 전부 13자리 숫자였다. 아닌 것이 오면 번호가 아니다."""
    assert dgk.normalize_crno("abc") is None
    assert dgk.normalize_crno("1101-111-867948") is None


def test_상장종목_한_줄이_자리표시자를_통과시키지_않는다():
    """`parse_listed_row` 를 거쳐도 걸러져야 한다 — 헬퍼만 고치고 배선을 빠뜨리면
    테스트는 통과하는데 자료는 그대로 들어온다."""
    행 = dgk.parse_listed_row(
        {"basDt": "20260901", "srtnCd": "A900110", "isinCd": "HK0000057197",
         "mrktCtg": "KOSDAQ", "itmsNm": "딥커머스", "crno": "0000000000000",
         "corpNm": "딥커머스리미티드"},
        known_at="20260902")
    assert 행["crno"] is None
    assert 행["code"] == "900110", "다른 칸은 그대로여야 한다"
    assert 행["isin_cd"] == "HK0000057197"


def test_영은_영으로_남는다():
    """`0명` 과 `모른다` 는 다른 사실이다."""
    assert dgk.normalize_int("0") == 0
    assert dgk.normalize_int("") is None
    assert dgk.normalize_int(None) is None
    assert dgk.normalize_int("838") == 838
    assert dgk.normalize_int("사람없음") is None


# ==================================================
# 5. 응답 파싱
# ==================================================
def test_항목이_하나면_객체로_온다():
    """🔴 공공데이터포털 공통 함정 — 한 건일 때 리스트가 아니다.

    리스트로 단정하면 한 건짜리 날짜에서 **칸 이름이 한 글자씩** 잘려 나온다.
    """
    하나 = {"response": {"body": {"items": {"item": 목록_한줄}}}}
    여럿 = {"response": {"body": {"items": {"item": [목록_한줄, 목록_한줄]}}}}
    assert dgk._items(하나) == [목록_한줄]
    assert len(dgk._items(여럿)) == 2


def test_항목이_없으면_빈_목록():
    assert dgk._items({"response": {"body": {}}}) == []
    assert dgk._items({"response": {"body": {"items": ""}}}) == []
    assert dgk._items({}) == []


def test_목록_한_줄이_우리_칸으로_바뀐다():
    행 = dgk.parse_listed_row(목록_한줄, known_at="20240823")
    assert 행["code"] == "000020"          # ← A 가 떨어졌다
    assert 행["bas_dd"] == "20240822"
    assert 행["isin_cd"] == "KR7000020008"
    assert 행["crno"] == "1101110043870"
    assert 행["item_nm"] == "동화약품"
    assert 행["market"] == "KOSPI"
    assert 행["known_at"] == "20240823"
    assert 행["known_rule"] == dgk.KNOWN_RULE_NEXT_SESSION


def test_목록의_빈_칸은_None_이_된다():
    """빈 문자열과 NULL 을 섞으면 `WHERE crno IS NULL` 이 반쪽만 잡는다."""
    행 = dgk.parse_listed_row({**목록_한줄, "crno": "  "}, known_at="20240823")
    assert 행["crno"] is None


def test_개요_한_줄이_우리_칸으로_바뀐다():
    행 = dgk.parse_profile_row(개요_한줄)
    assert 행["crno"] == "1101110043870"
    assert 행["fst_opeg_dt"] == "20260319"
    assert 행["last_opeg_dt"] == "20260901"
    assert 행["estb_dt"] == "18970925"
    assert 행["xchg_lstg_dt"] == "19760324"     # ← 76 이 1976 으로 풀렸다
    assert 행["empe_cnt"] == 838
    assert 행["pn1_avg_slry_amt"] == 72_000_000
    assert 행["audt_rpt_opnn"] == "적정의견"
    assert 행["sic_nm"] is None                 # 빈 문자열이 아니라 None


def test_개요의_known_at_은_계산이_아니라_관측이다():
    """🔴 출처가 '이 날부터 유효' 를 직접 준다 — 규칙을 바꿔도 재수집이 필요 없다."""
    행 = dgk.parse_profile_row(개요_한줄)
    assert 행["known_at"] == 개요_한줄["fstOpegDt"]
    assert 행["known_rule"] == dgk.KNOWN_RULE_OBSERVED


def test_키가_없는_개요_행은_None_을_준다():
    """빈 키로 넣으면 서로를 덮어써서 행 수는 그럴듯한데 내용이 사라진다."""
    assert dgk.parse_profile_row({**개요_한줄, "crno": ""}) is None
    assert dgk.parse_profile_row({**개요_한줄, "fstOpegDt": ""}) is None


# ==================================================
# 6. 오류를 갈라 본다 — 할 일이 다르다
# ==================================================
@pytest.mark.parametrize("코드, 들어야_할_말", [
    ("SERVICE_KEY_IS_NOT_REGISTERED_ERROR", "%253D"),
    ("SERVICE_ACCESS_DENIED_ERROR", "활용신청"),
    ("NO_OPENAPI_SERVICE_ERROR", "엔드포인트"),
    ("LIMITED_NUMBER_OF_SERVICE_REQUESTS_EXCEEDS_ERROR", "내일"),
])
def test_오류마다_할_일이_다르게_안내된다(코드, 들어야_할_말):
    """겉으로는 다 '안 된다' 지만 키를 고칠 일과 신청할 일과 기다릴 일은 다르다."""
    글 = dgk._explain(코드, "msg", dgk.EP_LISTED)
    assert 들어야_할_말 in 글
    assert "할 일" in 글


def test_모르는_오류코드도_막다른_길로_두지_않는다():
    글 = dgk._explain("ALIEN_ERROR_CODE", "무슨 일인지 모름", dgk.EP_LISTED)
    assert "할 일" in 글


def test_오류코드가_00_이_아니면_세운다(monkeypatch):
    본문 = json.dumps({"response": {
        "header": {"resultCode": "SERVICE_ACCESS_DENIED_ERROR",
                   "resultMsg": "권한 없음"}}})
    monkeypatch.setattr(dgk.budget, "try_spend", lambda *a, **k: True)

    class 가짜응답:
        def read(self):
            return 본문.encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(dgk, "urlopen", lambda *a, **k: 가짜응답())
    with pytest.raises(dgk.DataGoKrError) as 잡힌것:
        dgk._request(dgk.EP_LISTED, "k", {})
    assert "활용신청" in str(잡힌것.value)


def test_JSON_이_아니면_받은_것을_보여준다(monkeypatch):
    """'JSON 이 아니다' 만 말하면 무엇이 잘못됐는지 알 수 없다."""
    본문 = "<OpenAPI_ServiceResponse><returnAuthMsg>SERVICE KEY IS NOT REGISTERED</returnAuthMsg>"

    class 가짜응답:
        def read(self):
            return 본문.encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(dgk.budget, "try_spend", lambda *a, **k: True)
    monkeypatch.setattr(dgk, "urlopen", lambda *a, **k: 가짜응답())
    with pytest.raises(dgk.DataGoKrError) as 잡힌것:
        dgk._request(dgk.EP_LISTED, "k", {})
    assert "returnAuthMsg" in str(잡힌것.value)


# ==================================================
# 7. 예산 — 부르기 전에 멈춘다
# ==================================================
def test_예산이_없으면_부르지_않는다(monkeypatch):
    """🔴 포털이 한도 초과를 알려 줄 때는 이미 늦었다 — 그날은 더 못 받는다."""
    불렸나 = {"값": False}

    def 부르면안된다(*a, **k):
        불렸나["값"] = True
        raise AssertionError("예산이 없는데 호출했다")

    monkeypatch.setattr(dgk.budget, "try_spend", lambda *a, **k: False)
    monkeypatch.setattr(dgk, "urlopen", 부르면안된다)

    with pytest.raises(dgk.DataGoKrError) as 잡힌것:
        dgk._request(dgk.EP_LISTED, "k", {})
    assert not 불렸나["값"]
    assert "내일" in str(잡힌것.value)


# ==================================================
# 8. 이른 날짜는 부르지 않는다
# ==================================================
def test_2019년_이전은_호출조차_하지_않는다(monkeypatch):
    """실측상 2019 이전은 전부 0건이다. 훑으면 2,500콜을 0건에 쓴다."""
    def 부르면안된다(*a, **k):
        raise AssertionError("EARLIEST 이전인데 호출했다")

    monkeypatch.setattr(dgk, "_request", 부르면안된다)
    assert dgk.fetch_listed("20190102", key="k", known_at="20190103") == []


def test_콜_수를_미리_센다():
    """받다가 중간에 막히면 어디까지 받았는지 맞추는 일이 생긴다."""
    # 2019년 두 날은 세지 않는다 — 어차피 안 부른다.
    날짜 = ["20190102", "20190103", "20200102", "20200103", "20200106"]
    assert dgk.estimate_calls(날짜) == 3 * 3        # 3일 × 3페이지
    assert dgk.estimate_calls([]) == 0
