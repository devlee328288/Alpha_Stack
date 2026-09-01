"""반입 엔진이 **들일 것을 들이고 막을 것을 막는지** 확인한다.

두 방향을 함께 잰다.

- **막는가** — 일부러 어긋난 파일을 넣고 격리·거부되는지
- 🔴 **멀쩡한 것을 막지는 않는가** — 이쪽이 더 중요하다. 검사기는 전부 격리해도 "안전하게"
  보이고, 그 상태로 배포되면 팀원 자료가 통째로 사라진 채 아무 예외도 안 난다.
  지난 세션에 규격 규칙 하나가 우리 자료 283,468행(3.08%)을 격리하고 있던 것을 실측으로
  찾아낸 적이 있다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import trading_calendar  # noqa: E402
from ingest.inbox.engine import (  # noqa: E402
    InboxError,
    available_kinds,
    check_constraints,
    fill_file_meta,
    guess_kind,
    inspect_file,
    load_spec,
    map_columns,
    read_table,
    score_kind,
)

가짜달력 = frozenset({"20201231", "20210104", "20210105", "20210106", "20210107",
                    "20210108", "20210111", "20210112"})


@pytest.fixture(autouse=True)
def 달력을_끼운다(monkeypatch):
    monkeypatch.setattr(trading_calendar, "_SESSION_CACHE", 가짜달력)
    monkeypatch.setattr(trading_calendar, "_SESSION_SPAN", (min(가짜달력), max(가짜달력)))


def 파일(tmp_path: Path, name: str, text: str, encoding: str = "utf-8") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding=encoding)
    return path


깨끗한_종목 = """날짜,종목코드,종목명,시장,시가,고가,저가,종가,거래량
2021-01-04,005930,삼성전자,KOSPI,81000,84400,80200,83000,38655276
2021-01-05,005930,삼성전자,KOSPI,81600,83900,81600,83900,35335669
"""


# ==================================================
# 1. 읽기
# ==================================================
def test_모든_칸을_문자열로_읽는다(tmp_path):
    """🔴 `"005930"` 을 정수로 읽으면 그 순간 앞자리 0 이 사라져 되살릴 수 없다."""
    path = 파일(tmp_path, "a.csv", "code,volume\n005930,1000\n")
    frame = read_table(path)
    assert frame["code"].iloc[0] == "005930"


def test_cp949_로_저장된_파일도_읽는다(tmp_path):
    path = 파일(tmp_path, "b.csv", "종목명\n삼성전자\n", encoding="cp949")
    assert read_table(path)["종목명"].iloc[0] == "삼성전자"


def test_utf8_을_cp949_로_잘못_읽지_않는다(tmp_path):
    """인코딩 후보 순서를 재는 시험 — cp949 를 먼저 대면 UTF-8 한글이 깨진 채 '성공' 한다."""
    path = 파일(tmp_path, "c.csv", "종목명\n삼성전자\n", encoding="utf-8")
    assert read_table(path)["종목명"].iloc[0] == "삼성전자"


# ==================================================
# 2. 이름 맞추기
# ==================================================
def test_한글_칸_이름을_규격_칸으로_옮긴다():
    spec = load_spec("ohlcv_stock")
    frame = pd.DataFrame(columns=["날짜", "종목코드", "종가"])
    mapping = map_columns(frame, spec)
    assert mapping.mapped == {"날짜": "bas_dd", "종목코드": "code", "종가": "close"}


def test_규격_밖_칸은_버리지_않고_extras_로_남긴다():
    spec = load_spec("ohlcv_stock")
    mapping = map_columns(pd.DataFrame(columns=["날짜", "메모"]), spec)
    assert mapping.extras == ["메모"]


def test_뉴스_본문_칸이_있으면_파일째_되돌린다():
    """별칭으로 받아 주면 '본문을 담지 않는다' 는 약속이 한 줄로 뚫린다."""
    spec = load_spec("news")
    mapping = map_columns(pd.DataFrame(columns=["발행시각", "본문"]), spec)
    assert mapping.rejected is not None
    assert "기사 본문" in mapping.rejected


def test_뜻이_갈리는_이름은_추측하지_않고_묻는다():
    """`"지수"` 는 이름일 수도 값일 수도 있다 — 값만 보고 정하면 반은 틀린다."""
    spec = load_spec("ohlcv_index")
    mapping = map_columns(pd.DataFrame(columns=["지수"]), spec)
    assert mapping.mapped == {}
    assert mapping.questions[0]["column"] == "지수"
    assert set(mapping.questions[0]["candidates"]) == {"index_name", "close"}


def test_한_규격_칸을_두_원본이_노리는데_값이_다르면_되돌린다():
    spec = load_spec("ohlcv_stock")
    frame = pd.DataFrame({"code": ["005930"], "종목코드": ["000660"]})
    assert map_columns(frame, spec).rejected is not None


def test_값이_같으면_중복으로_보고_하나만_쓴다():
    spec = load_spec("ohlcv_stock")
    frame = pd.DataFrame({"code": ["005930"], "종목코드": ["005930"]})
    mapping = map_columns(frame, spec)
    assert mapping.mapped == {"code": "code"}
    assert mapping.questions[0]["column"] == "종목코드"


# ==================================================
# 3. 어느 규격인가
# ==================================================
def test_종류_폴더가_없어도_규격을_재서_정한다(tmp_path):
    """HuggingFace 의 `inbox/<이름>/` 에는 종류 폴더가 없다."""
    frame = read_table(파일(tmp_path, "x.csv", 깨끗한_종목))
    kind, scores = guess_kind(frame)
    assert kind == "ohlcv_stock"
    assert scores[0]["score"] > scores[1]["score"]


def test_애매하면_정하지_않는다():
    """틀린 규격으로 검사하면 멀쩡한 파일이 통째로 격리되고 팀원이 오해한다."""
    kind, _ = guess_kind(pd.DataFrame(columns=["뭔가", "알수없는칸"]))
    assert kind is None


def test_뉴스_파일을_종목_규격으로_재면_점수가_바닥이다():
    frame = pd.DataFrame(columns=["발행시각", "제목", "링크", "검색어"])
    assert score_kind(frame, "ohlcv_stock")["required_hit"] == 0.0
    assert score_kind(frame, "news")["required_hit"] == 1.0


# ==================================================
# 4. 칸 제약
# ==================================================
def test_required_pattern_enum_범위를_전부_잰다():
    spec = load_spec("ohlcv_stock")
    frame = pd.DataFrame({
        "bas_dd": ["20210104", "20210104", "20210104", "20210104"],
        "code": ["005930", "5930", "005930", "005930"],       # 두 번째가 pattern 위반
        "close": [83000, 83000, None, 83000],                  # 세 번째가 required 위반
        "market": ["KOSPI", "KOSPI", "KOSPI", "NASDAQ"],       # 네 번째가 enum 위반
    })
    violations = check_constraints(frame, spec)
    assert [v.rule for v in violations[1]] == ["code.pattern"]
    assert [v.rule for v in violations[2]] == ["close.required"]
    assert [v.rule for v in violations[3]] == ["market.enum"]
    assert 0 not in violations, "멀쩡한 행은 아무 위반도 없어야 한다"


@pytest.mark.parametrize("code, 무엇", [
    ("005930", "보통주"),
    ("00781K", "신형우선주 — 끝자리가 영문이다 (코리아써키트2우B)"),
    ("08537M", "신형우선주 (루트로닉3우C)"),
    ("00341A", "신형우선주 (쌍용양회4우B)"),
    ("0004Y0", "스팩 — 다섯째 자리가 영문이다 (디비금융제14호스팩)"),
    ("0007C0", "최근 상장 (아크릴)"),
])
def test_영문이_섞인_종목코드도_통과한다(code, 무엇):
    """🔴 `^[0-9]{6}$` 이던 시절 우리 자료 56,190행(0.61%·84종)이 격리되고 있었다.

    KRX 가 여섯 자리를 다 써서 영문을 섞기 시작했다. 실측으로 찾았고, 앞으로 늘어난다.
    """
    spec = load_spec("ohlcv_stock")
    frame = pd.DataFrame({"bas_dd": ["20210104"], "code": [code], "close": [1000]})
    assert check_constraints(frame, spec) == {}, 무엇


@pytest.mark.parametrize("code", ["KR7005", "A05930", "12345", "1234567", "00-781"])
def test_종목코드가_아닌_것은_여전히_막는다(code):
    """느슨하게 고치되 다 열지는 않는다 — 실측상 1~4번째 자리는 언제나 숫자다."""
    spec = load_spec("ohlcv_stock")
    frame = pd.DataFrame({"bas_dd": ["20210104"], "code": [code], "close": [1000]})
    assert [v.rule for v in check_constraints(frame, spec)[0]] == ["code.pattern"]


def test_소문자로_적어_온_코드는_올려서_들인다(tmp_path):
    """KRX 원본은 전부 대문자라 올려도 잃는 정보가 없다."""
    text = "날짜,종목코드,종가\n2021-01-04,00781k,1000\n"
    result = inspect_file(파일(tmp_path, "lower.csv", text), kind="ohlcv_stock")
    assert len(result.accepted) == 1
    assert result.accepted.iloc[0]["payload"]["code"] == "00781K"


def test_빈_칸은_pattern_이나_enum_위반이_아니다():
    """선택 칸이 비어 있는 것은 모양이 틀린 것과 다르다."""
    spec = load_spec("ohlcv_stock")
    frame = pd.DataFrame({"bas_dd": ["20210104"], "code": ["005930"],
                          "close": [83000], "market": [None]})
    assert check_constraints(frame, spec) == {}


# ==================================================
# 5. 파일 수준 메타
# ==================================================
def test_한_값뿐인_식별_칸은_파일_전체에_편다():
    """DART 응답은 한 요청에 한 회사라 이 값들이 첫 줄에만 실려 온다."""
    spec = load_spec("financial")
    frame = pd.DataFrame({"corp_code": ["00126380", None, None],
                          "account_nm": ["매출액", "영업이익", "당기순이익"]})
    notes = fill_file_meta(frame, spec)
    assert frame["corp_code"].tolist() == ["00126380"] * 3
    assert notes[0]["action"] == "filled"


def test_값이_여러_가지면_펴지_않는다():
    """여러 회사가 섞인 파일이다 — 함부로 채우면 남의 회사 값이 붙는다."""
    spec = load_spec("financial")
    frame = pd.DataFrame({"corp_code": ["00126380", "00164779", None],
                          "account_nm": ["매출액", "매출액", "매출액"]})
    notes = fill_file_meta(frame, spec)
    assert pd.isna(frame["corp_code"].tolist()[2]), "빈 채로 남아야 한다"
    assert notes[0]["action"] == "skipped"


# ==================================================
# 6. 통째로
# ==================================================
def test_깨끗한_파일은_전량_들어온다(tmp_path):
    """🔴 이 시험이 가장 중요하다 — 검사기는 전부 막아도 '안전하게' 보인다."""
    result = inspect_file(파일(tmp_path, "clean.csv", 깨끗한_종목), kind="ohlcv_stock")
    assert result.rows_total == 2
    assert len(result.accepted) == 2
    assert len(result.quarantined) == 0


def test_지저분한_값은_고쳐서_들인다(tmp_path):
    text = """날짜,종목코드,시장,종가,거래량
2021-01-04,5930,코스피,"83,000","38,655,276"
"""
    result = inspect_file(파일(tmp_path, "dirty.csv", text), kind="ohlcv_stock")
    assert len(result.accepted) == 1
    payload = result.accepted.iloc[0]["payload"]
    assert payload["code"] == "005930"
    assert payload["market"] == "KOSPI"
    assert payload["close"] == 83000


def test_결측_표기는_오류가_아니다(tmp_path):
    """거래정지 종목은 시고저가가 `-` 로 온다. 이걸 격리하면 정상 자료가 사라진다."""
    text = """날짜,종목코드,시가,고가,저가,종가,거래량
2021-01-04,035720,-,-,-,39000,0
"""
    result = inspect_file(파일(tmp_path, "halt.csv", text), kind="ohlcv_stock")
    assert len(result.accepted) == 1, "결측 표기 때문에 격리되면 안 된다"


def test_값이_있는데_못_읽으면_격리한다(tmp_path):
    """빈 칸과 못 읽은 값은 다르다 — 뒤엣것은 그 행을 믿을 수 없다는 뜻이다."""
    text = """날짜,종목코드,종가,거래량
2021-01-04,005930,일이삼,1000
"""
    result = inspect_file(파일(tmp_path, "bad.csv", text), kind="ohlcv_stock")
    assert len(result.quarantined) == 1
    rules = [v["rule"] for v in result.quarantined.iloc[0]["violations"]]
    assert "cleaner.failed" in rules


def test_행_규칙_위반은_그_행만_격리한다(tmp_path):
    text = """날짜,종목코드,시가,고가,저가,종가,거래량
2021-01-04,005930,81000,84400,80200,83000,100
2021-01-05,005930,400000,390000,395000,398000,100
"""
    result = inspect_file(파일(tmp_path, "hl.csv", text), kind="ohlcv_stock")
    assert len(result.accepted) == 1, "멀쩡한 행은 들어와야 한다"
    assert len(result.quarantined) == 1
    rules = [v["rule"] for v in result.quarantined.iloc[0]["violations"]]
    assert "high_ge_low" in rules


def test_격리된_행은_원본을_함께_남긴다(tmp_path):
    """사람이 고쳐 다시 넣으려면 우리가 정제하기 전 값이 필요하다."""
    text = """날짜,종목코드,시장,종가,거래량
2021-01-04,5930,NASDAQ,"83,000",100
"""
    result = inspect_file(파일(tmp_path, "raw.csv", text), kind="ohlcv_stock")
    row = result.quarantined.iloc[0]
    assert row["raw"]["종목코드"] == "5930", "원본 그대로여야 한다"
    assert row["payload"]["code"] == "005930", "정제 결과도 함께 남는다"


def test_규칙이_쓰는_칸이_없으면_위반이_아니라_건너뜀이다(tmp_path):
    """선택 칸을 안 담아 온 것을 위반으로 세면 파일이 통째로 격리된다."""
    text = "날짜,종목코드,종가\n2021-01-04,005930,83000\n"
    result = inspect_file(파일(tmp_path, "few.csv", text), kind="ohlcv_stock")
    assert len(result.accepted) == 1
    skipped = [t for t in result.report["row_rules"] if t.get("skipped")]
    assert skipped, "재지 못한 규칙은 그렇게 보고돼야 한다"
    assert all(t["violations"] == 0 for t in skipped)


def test_warn_은_들이되_기록한다(tmp_path):
    """`zero_ohlc` 는 우리 자료에 283,468행이나 있다 — error 로 두면 정상 자료가 격리된다."""
    text = """날짜,종목코드,시가,고가,저가,종가,거래량
2021-01-04,005930,0,0,0,83000,0
"""
    result = inspect_file(파일(tmp_path, "zero.csv", text), kind="ohlcv_stock")
    assert len(result.accepted) == 1
    warnings = [w["rule"] for w in result.accepted.iloc[0]["warnings"]]
    assert "zero_ohlc" in warnings


def test_없는_종류를_부르면_무엇이_있는지_알려_준다(tmp_path):
    with pytest.raises(InboxError, match="ohlcv_stock"):
        inspect_file(파일(tmp_path, "x.csv", 깨끗한_종목), kind="없는종류")


@pytest.mark.parametrize("kind", available_kinds())
def test_규격_다섯_장이_모두_읽히고_불변식을_지킨다(kind):
    """별칭 하나가 두 필드에 걸치면 자료가 아니라 규격이 틀린 것이다."""
    spec = load_spec(kind)
    assert spec["fields"], f"{kind} 규격에 필드가 없다"
