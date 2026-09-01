"""종목코드 형식 — 우리 실측이 규격·코드 양쪽에 반영돼 있는가.

`daily_price` 920만 행을 실측하니 5·6번째 자리에 영문이 있는 종목이 84종 있었다.
그런데 저장소 곳곳이 `^\\d{6}$` 로 가정하고 있었고, 그중 조회 분기 하나는 예외도 로그도
없이 *"그런 종목 없음"* 으로 위장했다. 이 파일은 그 회귀를 막는다.

**두 방향을 함께 잰다.** 막는 쪽만 재면 규격이 과하게 조여 정상 자료를 격리해도 통과한다.
그래서 실제 종목코드가 전부 통과하는지도 같이 잰다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from common import codes

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "ingest" / "inbox" / "schemas"

# 실측 표본. `data/krx_cache.db` 의 daily_price 에서 뽑았다 (2026-09-01).
# 앞 넷은 5·6번째가 영문이고, 뒤 둘은 350종 핵심 유니버스 안에 있다.
REAL_ODD_CODES = ("0001A0", "0004Y0", "0007C0", "00088K", "0009K0", "0126Z0")
REAL_PLAIN_CODES = ("005930", "000660", "035720", "950260")

# 종목코드가 아닌 것들. 반입 정문은 이것을 막아야 한다.
NOT_STOCK_CODES = ("KR7005", "A05930", "12345", "1234567", "00-781", "", "삼성전자")


class Test종목코드_판정:
    @pytest.mark.parametrize("code", REAL_ODD_CODES + REAL_PLAIN_CODES)
    def test_실제_종목코드는_전부_통과한다(self, code):
        """우리 DB 에 실제로 있는 코드가 하나라도 막히면 그 종목이 통째로 격리된다."""
        assert codes.is_stock_code(code), f"{code} 가 막혔다 — 실측으로 존재하는 종목이다"

    @pytest.mark.parametrize("value", NOT_STOCK_CODES)
    def test_종목코드가_아닌_것은_막는다(self, value):
        assert not codes.is_stock_code(value)

    def test_소문자로_와도_같은_것으로_본다(self):
        """엑셀·수기 입력이 소문자를 만든다. 판정에서 갈리면 안 된다."""
        assert codes.is_stock_code("00781k")
        assert codes.is_stock_code("0009k0")

    def test_앞_네_자리는_숫자여야_한다(self):
        """실측: 3,677종 전부 1~4번째가 숫자였다. 영문이 앞에 오는 형식은 아직 없다."""
        assert not codes.is_stock_code("A0593O")
        assert not codes.is_stock_code("00A930")


class Test조회_갈래고르기:
    """`looks_like_code` 는 검증이 아니라 '코드 검색인가 이름 검색인가' 를 가른다."""

    @pytest.mark.parametrize("code", REAL_ODD_CODES + REAL_PLAIN_CODES)
    def test_실제_종목코드는_코드_검색으로_간다(self, code):
        assert codes.looks_like_code(code)

    def test_한글_종목명은_이름_검색으로_간다(self):
        assert not codes.looks_like_code("삼성전자")
        assert not codes.looks_like_code("에임드바이오")

    def test_HTS_표기도_코드로_받아준다(self):
        """`A05930` 은 종목코드는 아니지만 코드 검색으로 보내는 편이 낫다 — 못 찾으면 None 이다."""
        assert codes.looks_like_code("A05930")
        assert not codes.is_stock_code("A05930")

    def test_판정이_더_넓다(self):
        """조회 쪽이 좁으면 정상 종목을 이름 검색으로 흘려보내 조용히 놓친다."""
        for code in REAL_ODD_CODES:
            assert codes.is_stock_code(code) and codes.looks_like_code(code)


def _code_fields():
    """규격 전부를 훑어 '종목코드 칸' 을 찾는다.

    칸 이름을 여기 적어 두면 규격이 늘거나 이름이 바뀔 때 검사가 조용히 비어 버린다
    (실제로 `news` 는 `stock_code` 가 아니라 `code` 였다). 그래서 **제목으로** 찾는다.
    """
    for path in sorted(SCHEMA_DIR.glob("*.json")):
        spec = json.loads(path.read_text(encoding="utf-8"))
        for field in spec.get("fields", []):
            if field.get("title") == "종목코드":
                yield path.stem, field["name"], (field.get("constraints") or {}).get("pattern")


class Test규격과_코드가_같은_것을_본다:
    """규격 JSON 이 정본이다. 파이썬 상수는 그 사본이므로 어긋나면 여기서 깨진다."""

    def test_종목코드_칸을_실제로_찾았다(self):
        """찾은 게 0건이면 아래 검사가 통째로 비어 버린다 — 그것부터 막는다."""
        found = list(_code_fields())
        assert len(found) >= 3, f"종목코드 칸을 {len(found)}개만 찾았다: {found}"

    @pytest.mark.parametrize("schema_name,field_name,pattern", list(_code_fields()))
    def test_규격의_패턴과_파이썬_상수가_같다(self, schema_name, field_name, pattern):
        assert pattern == codes.STOCK_CODE_PATTERN.pattern, (
            f"{schema_name}.json 의 {field_name}.pattern 이 common/codes.py 와 다르다.\n"
            f"  규격: {pattern}\n"
            f"  코드: {codes.STOCK_CODE_PATTERN.pattern}\n"
            "두 벌로 두면 한쪽만 고쳐진 채 오래 간다 — 함께 고쳐야 한다."
        )

    @pytest.mark.parametrize("schema_name,field_name,pattern", list(_code_fields()))
    def test_규격_패턴이_실제_종목코드를_막지_않는다(self, schema_name, field_name, pattern):
        """막는 쪽만 재면 과하게 조인 규격도 통과한다. 우리 자료를 왕복시켜 함께 잰다."""
        import re
        compiled = re.compile(pattern)
        for code in REAL_ODD_CODES + REAL_PLAIN_CODES:
            assert compiled.fullmatch(code), (
                f"{schema_name}.json 의 {field_name} 이 {code} 를 막는다 — "
                "실측으로 존재하는 종목이다"
            )


class Test고친_자리들이_실제로_낫는다:
    """상수를 바꿨어도 부르는 쪽이 옛 방식이면 소용없다. 그 자리를 직접 부른다."""

    def test_krx_pg_의_CODE_PATTERN_이_영문코드를_받는다(self):
        from ingest.store import krx_pg
        for code in REAL_ODD_CODES:
            assert krx_pg.CODE_PATTERN.fullmatch(code), (
                f"{code} 가 코드 분기를 못 탄다 — 이름 검색으로 새어 조용히 None 이 된다"
            )

    def test_야후_티커에_접미사가_붙는다(self):
        from ingest.clients import yf_data
        assert yf_data.normalize_ticker("0009K0") == "0009K0.KS"
        assert yf_data.normalize_ticker("005930") == "005930.KS"
        # 해외 티커에는 붙이지 않는다.
        assert yf_data.normalize_ticker("AAPL") == "AAPL"

    def test_dart_의_종목코드_상수도_고쳐졌다(self):
        """지금은 쓰는 곳이 없지만, 다음 사람이 집어 쓰는 순간 84종이 되돌아온다."""
        from ingest.clients import dart_data
        assert dart_data.STOCK_CODE_PATTERN.fullmatch("0009K0")
        # 8자리 고유번호는 진짜로 숫자만이다 — 함께 고치면 안 된다.
        assert dart_data.CORP_CODE_PATTERN.fullmatch("00126380")
        assert not dart_data.CORP_CODE_PATTERN.fullmatch("0012638A")

    def test_고유번호_매핑이_정상코드를_이상하다고_하지_않는다(self):
        """매 실행마다 84건을 경고하면, 정말로 새 형식이 온 날 그 경고가 소음에 묻힌다."""
        import sys
        sys.path.insert(0, str(ROOT))
        from scripts.build_corp_code import odd_codes

        mapping = {code: {"corp_code": "00000000"} for code in REAL_ODD_CODES + REAL_PLAIN_CODES}
        mapping["12345"] = {"corp_code": "00000000"}      # 진짜로 이상한 것
        assert odd_codes(mapping) == ["12345"]
