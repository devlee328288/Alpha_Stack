"""반출본에 싣는 판정 6칸의 계약 — 주권종류 3칸 · 기업행위 3칸.

## 왜 이 파일이 필요한가

이슈 [#186](https://github.com/devlee328288/Alpha_Stack/issues/186) ① 이 짚은 것은
**판정을 만들어 두고 파일에 안 싣는 것**이었다. 공급 층은 주권종류로 보통주를 가리고
정리매매·거래정지·신규상장을 덜어내는데, 반출본에는 그 결과가 없어서 받아 쓰는 쪽이
같은 판정을 각자 다시 만들어야 했다. 그 재구성이 이름 규칙이었고, 이름이 '우' 로
끝나는 보통주를 우선주로 잘못 뺐다.

칸을 실으면 그 문제가 닫히는 대신 **새 위험이 생긴다.**

    ① 붙이는 과정에서 행이 늘거나 순서가 바뀌면 다른 종목의 판정이 붙는다
    ② `is_liquidation` 은 미래를 보고 매기는 값이라 피처로 새면 미래참조다
    ③ 큰 벌에 실은 `is_halted` 가 작은 벌이 덜어내는 기준과 다르면,
       "같은 규칙 위에 세운다" 는 목적 자체가 무너진다
    ④ 🆕 그날 기본정보는 **그날 마감 뒤에** 나온다(`known_at` = 다음 거래일). 같은 날짜로
       붙이면 모든 행이 하루 앞선 정보를 본다 — 2026-09-11 에 찾아 고쳤다. 실측으로
       소속부가 바뀐 날 6,276행(중견기업부 → 관리종목 363건 포함)과 상장 첫날 3,677행이
       새 정보를 하루 먼저 봤다.

행 수를 세는 검사로는 ①·③·④ 가 안 잡힌다. 그래서 여기서 못박는다.
"""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from common.corporate_actions import is_traded
from supply.training import CORPORATE_ACTION_COLUMNS
from supply.universe import SECURITY_TYPE_COLUMNS, attach_security_type


@pytest.fixture
def 기본정보DB(tmp_path):
    """`stock_base_info` 만 있는 작은 DB. `known_at` 은 basDd 의 **다음 거래일**이다.

    20200102(목) 의 다음 거래일은 20200103(금), 20200103 의 다음 거래일은 20200106(월)이다.
    """
    경로 = tmp_path / "base.db"
    conn = sqlite3.connect(경로)
    conn.executescript(
        """
        CREATE TABLE stock_base_info (
            bas_dd TEXT, code TEXT, kind_stkcert_tp_nm TEXT,
            secugrp_nm TEXT, sect_tp_nm TEXT, known_at TEXT
        );
        """
    )
    conn.executemany(
        "INSERT INTO stock_base_info VALUES (?,?,?,?,?,?)",
        [
            # 이름이 '우' 로 끝나는 **보통주** — 옛 이름 규칙이 잘못 뺐던 종목
            ("20200102", "006800", "보통주", "주권", "", "20200103"),
            # 이름이 '우' 로 안 끝나는 **우선주** — 옛 규칙이 못 걸렀다
            ("20200102", "000327", "신형우선주", "주권", "", "20200103"),
            # 리츠 — KOSPI200 방법론·CRSP 둘 다 유니버스에서 뺀다
            ("20200102", "088260", "보통주", "부동산투자회사", "", "20200103"),
            # SPAC — 마찬가지
            ("20200102", "999999", "보통주", "주권", "SPAC(소속부없음)", "20200103"),
            # 🔴 소속부가 바뀐다 — 20200103 에 관리종목 지정 (그 사실은 20200106 에 안다)
            ("20200102", "111110", "보통주", "주권", "중견기업부", "20200103"),
            ("20200103", "111110", "보통주", "주권", "관리종목(소속부없음)", "20200106"),
            # 🔴 상장 첫날이 20200103 — 직전 기본정보가 없다
            ("20200103", "222220", "보통주", "주권", "", "20200106"),
            # 🔴 아직 알 수 없는 행 — as_of 가 그 앞이면 안 붙어야 한다
            ("20260901", "005930", "보통주", "주권", "", "20260902"),
        ],
    )
    conn.commit()
    conn.close()
    return 경로


def _시세(*키) -> pd.DataFrame:
    return pd.DataFrame([{"bas_dd": b, "code": c} for b, c in 키])


# ══════════════════════════════════════════════════════════════════════════
# ① 주권종류 세 칸 — 값이 그대로 실린다 (20200102 기본정보는 20200103 행부터)
# ══════════════════════════════════════════════════════════════════════════
def test_주권종류_세칸이_그대로_실린다(기본정보DB):
    frame = _시세(("20200103", "006800"), ("20200103", "000327"),
                  ("20200103", "088260"), ("20200103", "999999"))

    out = attach_security_type(frame, as_of="2026-09-09", db_path=str(기본정보DB))

    assert list(out.columns[-3:]) == list(SECURITY_TYPE_COLUMNS)
    표 = out.set_index("code")
    assert 표.loc["006800", "kind_stkcert_tp_nm"] == "보통주"
    assert 표.loc["000327", "kind_stkcert_tp_nm"] == "신형우선주"
    assert 표.loc["088260", "secugrp_nm"] == "부동산투자회사"
    assert 표.loc["999999", "sect_tp_nm"] == "SPAC(소속부없음)"


def test_판정을_대신_하지_않고_원문을_싣는다(기본정보DB):
    """리츠·SPAC 도 `kind_stkcert_tp_nm` 은 '보통주' 다.

    "무엇을 뺄지" 를 한 칸으로 뭉치지 않는 이유가 이것이다. 유동주식비율이 없어
    KOSPI200 방법론을 완전히 재현할 수 없는데 판정 칸을 만들면 "이대로 쓰면
    KOSPI200 과 같다" 는 오해를 부른다.
    """
    out = attach_security_type(_시세(("20200103", "088260"), ("20200103", "999999")),
                               as_of="2026-09-09", db_path=str(기본정보DB))
    assert (out["kind_stkcert_tp_nm"] == "보통주").all()
    assert set(out["secugrp_nm"]) | set(out["sect_tp_nm"]) >= {
        "부동산투자회사", "SPAC(소속부없음)"}


# ══════════════════════════════════════════════════════════════════════════
# ② 붙이다가 행이 늘거나 순서가 바뀌면 다른 종목의 판정이 붙는다
# ══════════════════════════════════════════════════════════════════════════
def test_행_수와_순서가_보존된다(기본정보DB):
    """행 수만 맞아도 순서가 바뀌면 조용히 틀린다 — 값으로 확인한다."""
    frame = _시세(("20200103", "999999"), ("20200103", "006800"),
                  ("20200103", "088260"))

    out = attach_security_type(frame, as_of="2026-09-09", db_path=str(기본정보DB))

    assert len(out) == 3
    assert list(out["code"]) == ["999999", "006800", "088260"]
    assert list(out["secugrp_nm"]) == ["주권", "주권", "부동산투자회사"]


def test_기본정보가_없는_행은_비워_두고_터지지_않는다(기본정보DB):
    """반출이 주권종류 때문에 죽어서는 안 된다. 빈 칸은 눈에 띈다."""
    out = attach_security_type(_시세(("20200103", "000001")),
                               as_of="2026-09-09", db_path=str(기본정보DB))
    assert out["kind_stkcert_tp_nm"].isna().all()


# ══════════════════════════════════════════════════════════════════════════
# ③ as_of — 오늘 알게 된 주권종류로 과거를 판정하지 않는다
# ══════════════════════════════════════════════════════════════════════════
def test_as_of_뒤에_알게_된_행은_안_붙는다(기본정보DB):
    """`known_at` 이 as_of 보다 뒤면 그때는 몰랐던 사실이다."""
    out = attach_security_type(_시세(("20260902", "005930")),
                               as_of="2026-08-01", db_path=str(기본정보DB))
    assert out["kind_stkcert_tp_nm"].isna().all()

    늦게 = attach_security_type(_시세(("20260902", "005930")),
                                as_of="2026-09-09", db_path=str(기본정보DB))
    assert 늦게["kind_stkcert_tp_nm"].iloc[0] == "보통주"


def test_as_of_는_기본값이_없어_빠뜨릴_수_없다(기본정보DB):
    with pytest.raises(TypeError):
        attach_security_type(_시세(("20200103", "006800")))  # noqa: PGH001


def test_시세_칸이_없으면_멈춘다(기본정보DB):
    with pytest.raises(ValueError, match="bas_dd"):
        attach_security_type(pd.DataFrame({"code": ["005930"]}),
                             as_of="2026-09-09", db_path=str(기본정보DB))


# ══════════════════════════════════════════════════════════════════════════
# ④ 그날 기본정보는 그날 행에 붙지 않는다 — as_of 를 넉넉히 줘도
# ══════════════════════════════════════════════════════════════════════════
def test_그날_기본정보는_그날_행에_붙지_않는다(기본정보DB):
    """🔴 2026-09-11 에 고친 자리. `as_of` 를 넉넉히 주는 반출 경로에서 드러난다.

    대조군을 이 자리에서 만든다 — **예전 규칙(같은 날짜로 조인)** 이면 이 행에 '보통주' 가
    붙는다. 그 값이 `known_at`(20200103) 보다 이른 행(20200102)에 붙는다는 것까지 센다.
    """
    out = attach_security_type(_시세(("20200102", "006800")),
                               as_of="2026-09-09", db_path=str(기본정보DB))
    assert out["kind_stkcert_tp_nm"].isna().all(), "그날 마감 뒤에 나온 기본정보가 붙었다"

    with sqlite3.connect(기본정보DB) as conn:
        예전 = pd.read_sql_query(
            "SELECT kind_stkcert_tp_nm, known_at FROM stock_base_info "
            "WHERE bas_dd = '20200102' AND code = '006800'", conn)
    assert list(예전["kind_stkcert_tp_nm"]) == ["보통주"], "대조: 같은 날짜로 붙이면 붙는다"
    assert (예전["known_at"] > "20200102").all(), "대조: 그런데 그 값은 그날 알 수 없었다"


def test_소속부가_바뀐_날은_전날_소속부가_붙는다(기본정보DB):
    """관리종목 지정일(20200103) 행은 아직 모른다. 다음 거래일(20200106) 행부터 붙는다."""
    out = attach_security_type(_시세(("20200103", "111110"), ("20200106", "111110")),
                               as_of="2026-09-09", db_path=str(기본정보DB))
    assert list(out["sect_tp_nm"]) == ["중견기업부", "관리종목(소속부없음)"]


def test_상장_첫날은_빈_칸이고_다음_거래일부터_붙는다(기본정보DB):
    """직전 기본정보가 없으므로 첫날은 모른다. 실측 2020 이후 상장 801종 전부 이 모양이다."""
    out = attach_security_type(_시세(("20200103", "222220"), ("20200106", "222220")),
                               as_of="2026-09-09", db_path=str(기본정보DB))
    assert pd.isna(out["kind_stkcert_tp_nm"].iloc[0])
    assert out["kind_stkcert_tp_nm"].iloc[1] == "보통주"


# ══════════════════════════════════════════════════════════════════════════
# ⑤ `is_halted` 는 작은 벌이 덜어내는 기준과 **같은 함수**여야 한다
#
# `training_frame` 은 `not is_traded(row)` 로 덜어낸다. 큰 벌에 실은 칸이 그것과
# 다르면 "작은 벌과 큰 벌을 같은 규칙 위에 세운다" 는 목적이 무너진다. 그런데
# `corporate_actions` 에는 비슷한 이름의 `is_halted(row)`(OHL 이 전부 0)도 있어서
# 헷갈리기 쉽다. 실측에서 두 집합이 개발구간 231,808행으로 정확히 같았지만,
# **같은 값이 나온다고 같은 정의인 것은 아니다.** 정의를 여기서 못박는다.
# ══════════════════════════════════════════════════════════════════════════
def test_거래정지_판정은_체결_여부다():
    """`open > 0` 이고 `volume > 0` 이어야 체결이다. 하나만 빠져도 거래정지다."""
    assert is_traded({"open": 100, "volume": 10}) is True
    assert is_traded({"open": 0, "volume": 10}) is False      # 시간외 단일가만 체결
    assert is_traded({"open": 100, "volume": 0}) is False     # 값은 있는데 체결이 없다
    assert is_traded({"open": 0, "volume": 0}) is False       # KRX 정지 표기행


def test_기업행위_칸은_셋이고_이름이_고정이다():
    """카드·문서·팀원 코드가 이 이름에 기댄다."""
    assert CORPORATE_ACTION_COLUMNS == (
        "is_liquidation", "is_halted", "is_first_listing")
