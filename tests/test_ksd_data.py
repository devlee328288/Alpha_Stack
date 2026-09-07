"""예탁결제원 게이트웨이(B552481) 클라이언트 — 규약 셋을 잠근다.

2026-09-03 에 실측으로 깨진 뒤 알게 된 것 셋이다. 문서에는 없고, 어기면 **되는 조회도
전부 실패**한다.

    ① XML 전용 — resultType=json 을 붙이지 않는다
    ② 선언 안 된 파라미터를 거부한다 — 부르기 전에 우리가 먼저 세운다 (예산을 안 쓴다)
    ③ numOfRows 상한 200

망을 타지 않는다. 응답은 실측 원문(`probe_out/`)에서 옮긴 XML 조각이다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ingest.clients import ksd_data as ksd  # noqa: E402

#: `StockSvc/getStkListInfoN1?isin=KR7005930003` 실측 (삼성전자 · 2026-09-03). 앞 두 행.
종목정보_XML = """<response>
  <header><resultCode>00</resultCode><resultMsg>NORMAL_SERVICE</resultMsg></header>
  <body>
    <items>
      <item><isin>KR7005930003</isin><korSecnNm>삼성전자</korSecnNm>
        <issucoCustno>593</issucoCustno><listTpcd>110</listTpcd>
        <apliDt>19750611</apliDt><xpitDt>99991231</xpitDt><dlistDt>99991231</dlistDt></item>
      <item><isin>KR7005930003</isin><korSecnNm>삼성전자</korSecnNm>
        <issucoCustno>593</issucoCustno><listTpcd>211</listTpcd>
        <apliDt>19750611</apliDt><xpitDt>20190912</xpitDt><dlistDt>20190913</dlistDt></item>
    </items>
    <numOfRows>200</numOfRows><pageNo>1</pageNo><totalCount>2</totalCount>
  </body>
</response>"""

#: `CorpSvc/getIssucoStkQtyChgList?issucoCustno=593` 실측. 액면분할 두 줄.
주식수변동_XML = """<response>
  <header><resultCode>00</resultCode><resultMsg>NORMAL_SERVICE</resultMsg></header>
  <body><items>
    <item><issuDt>20180503</issuDt><secnIssuRacd>201</secnIssuRacd>
      <secnIssuRacdNm>액면분할</secnIssuRacdNm><issuQty>6419324700</issuQty>
      <listDt>20180504</listDt></item>
    <item><issuDt>20120406</issuDt><secnIssuRacd>207</secnIssuRacd>
      <secnIssuRacdNm>합병</secnIssuRacdNm><issuQty>269867</issuQty><listDt></listDt></item>
  </items><totalCount>2</totalCount></body>
</response>"""

#: 선언 안 된 파라미터를 붙였을 때 오는 것. body 가 없다.
거절_XML = """<response><header><resultCode>10</resultCode>
<resultMsg>INVALID_REQUEST_PARAMETER_ERROR</resultMsg></header></response>"""

빈결과_XML = """<response><header><resultCode>03</resultCode>
<resultMsg>NODATA_ERROR</resultMsg></header></response>"""


def _응답(monkeypatch, xml: str, 기록: list | None = None):
    """`urlopen` 을 갈아 끼워 XML 을 돌려주고, 호출된 URL 을 기록한다."""
    class 가짜응답:
        def __init__(self, url):
            self.url = url

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return xml.encode("utf-8")

    def 가짜열기(req, timeout=0):
        if 기록 is not None:
            기록.append(req.full_url)
        return 가짜응답(req.full_url)

    monkeypatch.setattr(ksd, "urlopen", 가짜열기)
    monkeypatch.setattr(ksd.budget, "try_spend", lambda *a, **k: True)


# ==================================================
# ① XML 전용 · 서비스키 인코딩
# ==================================================
def test_resultType_을_붙이지_않는다():
    url = ksd._build_url(ksd.operation("stock_list"), "k", {"isin": "KR7005930003"})
    assert "resultType" not in url, "XML 전용 게이트웨이에 json 을 요구하면 거절당한다"


def test_서비스키가_다시_인코딩되지_않는다():
    키 = "abc%2Bdef%3D%3D"
    url = ksd._build_url(ksd.operation("stock_list"), 키, {"isin": "KR7005930003"})
    assert f"serviceKey={키}" in url
    assert "%253D" not in url


def test_기관코드가_B552481_이다():
    url = ksd._build_url(ksd.operation("issuer_basic"), "k", {"issucoCustno": "593"})
    assert url.startswith("https://apis.data.go.kr/B552481/CorpSvc/getIssucoBasicInfo?")


# ==================================================
# ② 선언 안 된 파라미터 — 부르기 전에 세운다
# ==================================================
def test_선언_안_된_파라미터는_부르기_전에_세운다():
    with pytest.raises(ksd.KsdError) as 잡힌것:
        ksd._build_url(ksd.operation("issuer_basic"), "k",
                       {"issucoCustno": "593", "numOfRows": 5})
    assert "받지 않는" in str(잡힌것.value)
    assert "numOfRows" in str(잡힌것.value)


def test_선언_안_된_파라미터는_예산을_쓰지_않는다(monkeypatch):
    """🔴 서버가 거절할 것이 뻔한 호출에 한도를 쓰지 않는다."""
    깎였나 = {"값": False}

    def 깎기(*a, **k):
        깎였나["값"] = True
        return True

    monkeypatch.setattr(ksd.budget, "try_spend", 깎기)
    monkeypatch.setattr(ksd, "urlopen", lambda *a, **k: pytest.fail("호출했다"))
    with pytest.raises(ksd.KsdError):
        ksd._request(ksd.operation("issuer_basic"), "k", {"issucoCustno": "593", "pageNo": 1})
    assert not 깎였나["값"]


def test_필수_파라미터가_빠지면_세운다():
    with pytest.raises(ksd.KsdError) as 잡힌것:
        ksd._build_url(ksd.operation("distribution_by_holder"), "k", {"issucoCustno": "593"})
    assert "rgtStdDt" in str(잡힌것.value)


def test_None_인_파라미터는_빠진_것으로_본다():
    with pytest.raises(ksd.KsdError):
        ksd._build_url(ksd.operation("stock_list"), "k", {"isin": None})


def test_페이징을_받는_오퍼레이션만_pageNo_를_허용한다():
    url = ksd._build_url(ksd.operation("stock_list"), "k",
                         {"isin": "KR7005930003", "numOfRows": 200, "pageNo": 1})
    assert "numOfRows=200" in url and "pageNo=1" in url


def test_모르는_오퍼레이션은_아는_것을_알려주며_세운다():
    with pytest.raises(ksd.KsdError) as 잡힌것:
        ksd.operation("없는것")
    assert "stock_list" in str(잡힌것.value)


# ==================================================
# ③ numOfRows 상한 200
# ==================================================
@pytest.mark.parametrize("행수", [201, 500, 1000])
def test_200_을_넘는_페이지는_부르기_전에_세운다(행수):
    with pytest.raises(ksd.KsdError) as 잡힌것:
        ksd._build_url(ksd.operation("stock_list"), "k", {"isin": "x", "numOfRows": 행수})
    assert "200" in str(잡힌것.value)


def test_콜_수는_200_으로_나눈다():
    assert ksd.estimate_calls(943) == 5          # 유가 943종 → 5콜 (금융위라면 1콜)
    assert ksd.estimate_calls(200) == 1
    assert ksd.estimate_calls(0) == 0
    assert ksd.estimate_calls(1000, page_size=1000) == 5, "page_size 로 상한을 못 넘긴다"


# ==================================================
# 응답 해석
# ==================================================
def test_XML_항목을_칸_사전으로_바꾼다(monkeypatch):
    _응답(monkeypatch, 종목정보_XML)
    행 = ksd.fetch("stock_list", {"isin": "KR7005930003"}, key="k")
    assert len(행) == 2
    assert 행[0]["issucoCustno"] == "593"
    assert 행[0]["korSecnNm"] == "삼성전자"


def test_빈_칸은_None_이다(monkeypatch):
    _응답(monkeypatch, 주식수변동_XML)
    행 = ksd.fetch("issuer_stock_changes", {"issucoCustno": "593"}, key="k")
    assert 행[1]["listDt"] is None


def test_거절_응답은_할_일과_함께_세운다(monkeypatch):
    _응답(monkeypatch, 거절_XML)
    with pytest.raises(ksd.KsdError) as 잡힌것:
        ksd.fetch("issuer_basic", {"issucoCustno": "593"}, key="k")
    글 = str(잡힌것.value)
    assert "INVALID_REQUEST_PARAMETER" in 글
    assert "OPERATIONS" in 글


def test_NODATA_는_오류가_아니라_빈_목록이다(monkeypatch):
    _응답(monkeypatch, 빈결과_XML)
    assert ksd.fetch("issuer_stock_changes", {"issucoCustno": "1"}, key="k") == []


def test_XML_이_아니면_받은_것을_보여준다(monkeypatch):
    _응답(monkeypatch, "<html>점검 중</html><<")
    with pytest.raises(ksd.KsdError) as 잡힌것:
        ksd.fetch("issuer_basic", {"issucoCustno": "593"}, key="k")
    assert "점검 중" in str(잡힌것.value)


def test_총건수에_닿으면_다음_페이지를_부르지_않는다(monkeypatch):
    기록: list = []
    _응답(monkeypatch, 종목정보_XML, 기록)          # totalCount=2 · 한 페이지에 2행
    ksd.fetch("stock_list", {"isin": "KR7005930003"}, key="k", page_size=2)
    assert len(기록) == 1, "총건수에 닿았는데 한 번 더 불렀다 — 날마다 쌓이면 한도가 된다"


def test_페이징_없는_오퍼레이션은_한_번만_부른다(monkeypatch):
    기록: list = []
    _응답(monkeypatch, 주식수변동_XML, 기록)
    ksd.fetch("issuer_stock_changes", {"issucoCustno": "593"}, key="k")
    assert len(기록) == 1
    assert "pageNo" not in 기록[0]


# ==================================================
# 다리 · 파싱
# ==================================================
def test_ISIN_으로_발행회사번호를_얻는다(monkeypatch):
    """상장구분마다 한 행이 와도 번호는 하나다."""
    _응답(monkeypatch, 종목정보_XML)
    assert ksd.issuer_custno("KR7005930003", key="k") == "593"


def test_발행회사번호가_둘이면_지어내지_않고_세운다(monkeypatch):
    둘 = 종목정보_XML.replace("<issucoCustno>593</issucoCustno><listTpcd>211",
                            "<issucoCustno>594</issucoCustno><listTpcd>211")
    _응답(monkeypatch, 둘)
    with pytest.raises(ksd.KsdError):
        ksd.issuer_custno("KR7005930003", key="k")


def test_없는_ISIN_은_None(monkeypatch):
    _응답(monkeypatch, 빈결과_XML)
    assert ksd.issuer_custno("KR0000000000", key="k") is None


def test_주식수_변동이_우리_칸으로_바뀐다(monkeypatch):
    _응답(monkeypatch, 주식수변동_XML)
    행 = ksd.stock_qty_changes("593", key="k")
    assert 행[0] == {"issu_dt": "20180503", "reason_code": "201", "reason_nm": "액면분할",
                    "issu_qty": 6419324700, "list_dt": "20180504"}
    assert 행[1]["list_dt"] is None


def test_아직_폐지_안_됨_자리표시자는_None_이_된다():
    """🔴 `99991231` 을 두면 "9999년에 폐지되는 회사" 가 생긴다."""
    행 = ksd.parse_basic_info({
        "issucoCustno": "593", "shotnIsin": "005930", "repSecnNm": "삼성전자",
        "founDt": "19690113", "apliDt": "19750611", "dlistDt": "99991231",
        "custXtinDt": "99991231", "caltotMartTpcd": "11", "pval": "100",
        "totalStkCnt": "   6,648,649,811 주", "eltscYn": "Y",
    })
    assert 행["delist_dt"] is None
    assert 행["extinct_dt"] is None
    assert 행["found_dt"] == "19690113"
    assert 행["market"] == "유가증권시장"


def test_쉼표와_단위가_붙은_주식수를_숫자로_읽는다():
    assert ksd._digits("   6,648,649,811 주") == 6648649811
    assert ksd._digits("5846278608") == 5846278608
    assert ksd._digits("") is None


def test_앞자리_0_이_생략된_비율을_읽는다():
    """`.02` 처럼 온다 — `float` 은 읽지만 사람이 보면 오타 같아서 여기서 못 박는다."""
    행 = ksd.parse_holder_row({"stkDistbutTpnm": "투자신탁", "shrs": "4932",
                              "shrsRatio": ".05", "stkqty": "1830632856",
                              "stkqtyRatio": "31.31"})
    assert 행["holders_ratio"] == 0.05
    assert 행["shares_ratio"] == 31.31
    assert 행["shares"] == 1830632856


def test_예산은_금융위와_같은_통이다():
    assert ksd.BUDGET_SOURCE == "data_go_kr"
    assert ksd.DAILY_LIMIT == ksd.budget.LIMITS["data_go_kr"]
