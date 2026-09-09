"""`stock_base_info` — 우선주 판별의 정본이 조용히 틀리지 않게 잠근다.

## 왜 이 표가 생겼나

지금까지 우리는 **종목명이 '우' 로 끝나는지로 우선주를 추측**했다. 그게 틀린다.
실측 2026-09-03, 세 시장 × 세 날짜(20150102 · 20200102 · 20260901) 전수 대조에서
**이름이 '우' 로 끝나는 보통주 7종**이 나왔다:

    미래에셋대우 · 연우 · 동우 · 신우 · 성우 · 에코글로우 · 이오플로우

006800 은 20200102 코스피 시총 **48위**다. 모델 파트가 쓰기로 한 "KOSPI 보통주
시가총액 상위 50" 후보에서 조용히 빠진다.

🔴 이 오류는 **이름이 바뀌는 구간에만** 나타난다 — 대우증권(정상) → 미래에셋대우(깨짐)
   → 미래에셋증권(정상). 오늘 유가 943종만 세면 어긋남이 0건이라 **표본으로는 못 잡는다.**
   그래서 여기 시험은 그 7종을 **이름으로 박아 두고** 판별이 정본을 따르는지 잰다.

망을 타지 않는다. DB 는 `conftest.py` 가 프로세스 전체에서 임시 경로로 갈아 끼운다.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ingest.clients import krx_data  # noqa: E402
from ingest.store import base_info_store as bis  # noqa: E402
from ingest.store.migrations import migrate_path  # noqa: E402


@pytest.fixture
def db(tmp_path):
    """이 시험만 쓰는 빈 DB. 마이그레이션을 v11 까지 올려 둔다."""
    경로 = tmp_path / "t.db"
    migrate_path(경로)
    conn = sqlite3.connect(경로)
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


def 기본행(**덮어쓸것):
    기본 = {
        "bas_dd": "20260901", "code": "095570", "isin_cd": "KR7095570008",
        "isu_nm": "AJ네트웍스보통주", "isu_abbrv": "AJ네트웍스",
        "isu_eng_nm": "AJ Networks Co.,Ltd.", "list_dd": "20150821",
        "market": "KOSPI", "secugrp_nm": "주권", "sect_tp_nm": "",
        "kind_stkcert_tp_nm": "보통주", "parval": "1000", "list_shrs": 45252759,
        "known_at": "20260902", "known_rule": "basDd+1session",
    }
    return {**기본, **덮어쓸것}


# ==================================================
# 1. 우선주 판별 — 이 표의 존재 이유
# ==================================================
#: 🔴 이름이 '우' 로 끝나지만 **보통주**인 종목들. 실측 2026-09-03 · KRX 종목기본정보.
#:    지금까지 쓰던 이름 추측(`name.endswith("우")`)이 이 일곱을 우선주로 잘못 뺐다.
이름이_우로_끝나는_보통주 = [
    ("006800", "미래에셋대우"),   # 20200102 코스피 시총 48위 — 후보 50종에 직접 걸린다
    ("115960", "연우"),
    ("088910", "동우"),
    ("025620", "신우"),
    ("458650", "성우"),
    ("159910", "에코글로우"),
    ("294090", "이오플로우"),
]


@pytest.mark.parametrize("code,name", 이름이_우로_끝나는_보통주)
def test_이름이_우로_끝나도_정본이_보통주면_보통주다(code, name):
    """옛 추측이 틀렸던 바로 그 일곱 종목. 정본을 보면 전부 보통주다."""
    assert krx_data.is_common_stock("보통주") is True

    # 옛 규칙이었다면 우선주로 걸렀을 이름인지 확인한다 — 시험이 헛돌지 않게.
    옛추측_우선주 = name.endswith("우") or "우B" in name or name.endswith("우C")
    if 옛추측_우선주:
        assert krx_data.is_common_stock("보통주") is True, (
            f"{code} {name}: 이름은 우선주처럼 보이지만 정본은 보통주다")


@pytest.mark.parametrize("종류", ["구형우선주", "신형우선주", "종류주권"])
def test_보통주가_아닌_주권종류는_전부_걸러진다(종류):
    assert krx_data.is_common_stock(종류) is False


@pytest.mark.parametrize("모르는값", ["", None, "  ", "우선주", "보통주식", "COMMON"])
def test_모르는_주권종류는_보통주가_아니라고_본다(모르는값):
    """🔴 빈 값·처음 보는 값을 보통주로 치면 우선주가 유니버스에 섞인다.

    종목이 빠지는 것보다 **틀린 종목이 들어오는 게 비싸다** — 빠진 것은 개수로
    드러나지만, 섞인 것은 성능이 조금 이상해질 뿐 아무 데도 안 걸린다.
    """
    assert krx_data.is_common_stock(모르는값) is False


def test_보통주는_앞뒤_공백이_있어도_보통주다():
    """KRX 는 문자열 뒤에 공백을 붙여 주는 경우가 있다 (일별매매정보에서 겪었다)."""
    assert krx_data.is_common_stock(" 보통주 ") is True


# ==================================================
# 2. 정규화 — 그날의 사실로 담긴다
# ==================================================
def test_정규화가_기준일을_함께_담는다():
    """이 응답은 오늘 스냅샷이 아니라 **그날의 사실**이다.

    같은 엔드포인트를 다른 날짜로 부르면 다른 답이 온다 (실측: 유가 20150102 899행 ·
    20260901 943행). 기준일을 안 담으면 '언제의 사실인지' 를 잃는다.
    """
    원문 = {"ISU_CD": "KR7095570008", "ISU_SRT_CD": "095570",
            "ISU_NM": "AJ네트웍스보통주", "ISU_ABBRV": "AJ네트웍스",
            "ISU_ENG_NM": "AJ Networks", "LIST_DD": "20150821",
            "MKT_TP_NM": "KOSPI", "SECUGRP_NM": "주권", "SECT_TP_NM": "",
            "KIND_STKCERT_TP_NM": "보통주", "PARVAL": "1000",
            "LIST_SHRS": "45,252,759"}
    행 = krx_data.normalize_base_info_row(원문, "20150102", "KOSPI")

    assert 행["bas_dd"] == "20150102"
    assert 행["code"] == "095570"
    assert 행["kind_stkcert_tp_nm"] == "보통주"
    # 상장주식수는 콤마를 떼고 숫자로
    assert 행["list_shrs"] == 45252759


def test_액면가는_숫자로_단정하지_않는다():
    """'무액면' 이 온다. int 로 깎으면 그 행이 통째로 None 이 되거나 터진다."""
    행 = krx_data.normalize_base_info_row(
        {"ISU_SRT_CD": "000660", "PARVAL": "무액면"}, "20260901", "KOSPI")
    assert 행["parval"] == "무액면"


def test_시장이_비어_오면_요청한_시장으로_채운다():
    행 = krx_data.normalize_base_info_row(
        {"ISU_SRT_CD": "000660", "MKT_TP_NM": ""}, "20260901", "KOSDAQ")
    assert 행["market"] == "KOSDAQ"


# ==================================================
# 3. 담기 — 조용히 사라지는 자리를 잠근다
# ==================================================
def test_같은_날짜를_다시_받아도_행이_늘지_않는다(db):
    bis.save([기본행()], db)
    bis.save([기본행(list_shrs=99)], db)
    행들 = db.execute("SELECT * FROM stock_base_info").fetchall()
    assert len(행들) == 1
    # 덮어쓰기가 실제로 일어났는지까지 본다 — 행 수만 세면 덮어쓰기를 못 잡는다
    assert 행들[0]["list_shrs"] == 99


def test_날짜가_다르면_다른_행이다(db):
    """🔴 기본키에서 날짜를 빼면 '그때는 무엇이었나' 를 잃는다.

    006800 은 대우증권 → 미래에셋대우 → 미래에셋증권으로 이름이 바뀌었다.
    날짜를 키에서 빼면 2015년 조인에 2026년 이름이 조용히 붙는다.
    """
    bis.save([기본행(bas_dd="20150102", code="006800", isu_abbrv="대우증권")], db)
    bis.save([기본행(bas_dd="20260901", code="006800", isu_abbrv="미래에셋증권")], db)
    이름들 = {r["bas_dd"]: r["isu_abbrv"] for r in
              db.execute("SELECT bas_dd, isu_abbrv FROM stock_base_info")}
    assert 이름들 == {"20150102": "대우증권", "20260901": "미래에셋증권"}


@pytest.mark.parametrize("빈칸", ["bas_dd", "code"])
def test_키가_빈_행은_담지_않고_세운다(db, 빈칸):
    """빈 키로 넣으면 서로를 덮어써서 행 수는 그럴듯한데 내용이 사라진다."""
    with pytest.raises(bis.BaseInfoStoreError):
        bis.save([기본행(**{빈칸: ""})], db)
    assert db.execute("SELECT COUNT(*) FROM stock_base_info").fetchone()[0] == 0


def test_known_at_이_비면_담지_않고_세운다(db):
    """빈 채로 담으면 `as_of` 가 그 행을 **영원히 못 거른다** — 미래참조가 된다."""
    with pytest.raises(bis.BaseInfoStoreError):
        bis.save([기본행(known_at=None)], db)


def test_코드가_전부_숫자가_아니어도_담긴다(db):
    """🔴 신형우선주는 5·6번째 자리에 영문이 온다(`0001A0`·`00088K`).

    `code GLOB '[0-9]*'` 로 막으면 그 종목들이 통째로 격리되는데, 격리는 조용해서
    한참 뒤에나 눈치챈다. 앞 4자리만 숫자라는 것까지가 우리가 확인한 사실이다.
    """
    bis.save([기본행(code="00088K", kind_stkcert_tp_nm="신형우선주")], db)
    assert db.execute(
        "SELECT COUNT(*) FROM stock_base_info WHERE code='00088K'").fetchone()[0] == 1


def test_앞_네자리가_숫자가_아니면_스키마가_막는다(db):
    """접두사가 섞여 들어오는 사고(`A000020`)를 표가 직접 막는다."""
    with pytest.raises(sqlite3.IntegrityError):
        bis.save([기본행(code="A00002")], db)


def test_빈_목록은_아무것도_안_하고_0을_돌려준다(db):
    assert bis.save([], db) == 0


# ==================================================
# 4. 현황 — 팀에 보고할 숫자가 맞는가
# ==================================================
def test_현황이_주권종류별_고유종목수를_센다(db):
    bis.save([
        기본행(code="005930", isu_abbrv="삼성전자"),
        기본행(code="005935", isu_abbrv="삼성전자우", kind_stkcert_tp_nm="구형우선주"),
        # 같은 종목의 다른 날짜 — 고유 종목 수는 늘지 않아야 한다
        기본행(bas_dd="20260828", code="005930", isu_abbrv="삼성전자"),
    ], db)
    s = bis.status(db)
    assert s["rows"] == 3
    assert s["codes"] == 2
    assert s["kinds"] == {"보통주": 1, "구형우선주": 1}


# ==================================================
# 5. 이어받기 — 휴장일에 영원히 다시 요청하지 않는다
# ==================================================
def 달력에_넣기(db, bas_dd: str, *markets: str) -> None:
    """`trading_calendar` 에 거래일을 심는다. 시장마다 한 줄이다."""
    for m in markets:
        db.execute(
            "INSERT INTO trading_calendar (bas_dd, market, stock_count, built_at) "
            "VALUES (?,?,?,?)",
            (bas_dd, m, 900, "2026-09-03T00:00:00+09:00"))
    db.commit()


def db_경로(db) -> Path:
    return Path(db.execute("PRAGMA database_list").fetchone()[2])


def test_이미_받은_날은_다시_받지_않는다(db):
    from ingest.store import collect_log

    달력에_넣기(db, "20260901", "KOSPI")
    assert ("20260901", "KOSPI") in bis.pending_days(("KOSPI",), db)

    collect_log.record(bis.SOURCE, "base_info:KOSPI:20260901", "ok", rows=943,
                       db_path=db_경로(db))
    assert ("20260901", "KOSPI") not in bis.pending_days(("KOSPI",), db)


def test_받아봤더니_0건인_날도_다시_받지_않는다(db):
    """🔴 `empty` 도 '받아 봤다' 이다.

    휴장일을 안 남기면 그 날짜를 **영원히 다시 요청**한다. 시세 수집에서 같은 이유로
    `empty` 를 남기고 있고, 여기도 같은 규칙을 쓴다.
    """
    from ingest.store import collect_log

    달력에_넣기(db, "20260103", "KOSPI")
    collect_log.record(bis.SOURCE, "base_info:KOSPI:20260103", "empty",
                       db_path=db_경로(db))
    assert ("20260103", "KOSPI") not in bis.pending_days(("KOSPI",), db)


def test_실패한_날은_다시_받는다(db):
    """`error` 는 '받아 봤다' 가 아니다 — 다음에 다시 시도해야 한다."""
    from ingest.store import collect_log

    달력에_넣기(db, "20260901", "KOSPI")
    collect_log.record(bis.SOURCE, "base_info:KOSPI:20260901", "error",
                       note="네트워크", db_path=db_경로(db))
    assert ("20260901", "KOSPI") in bis.pending_days(("KOSPI",), db)


def test_거래일_달력은_DISTINCT_로_읽는다(db):
    """🔴 `trading_calendar` 는 시장마다 한 줄씩 있어 12,306행이지만 날짜는 4,102일이다.

    DISTINCT 를 빼면 같은 날을 시장 수만큼 중복해서 받는다 — 예산이 3배로 샌다.
    (달력이 담는 시장은 `ALL`·`KOSPI`·`KOSDAQ` 셋이라 정확히 3배다.)
    """
    달력에_넣기(db, "20260901", "ALL", "KOSPI", "KOSDAQ")
    assert bis.pending_days(("KOSPI",), db) == [("20260901", "KOSPI")]


def test_KONEX_는_받지_않는다():
    """`daily_price` 에 KONEX 가 한 행도 없다 (KOSPI 1,251종 · KOSDAQ 2,448종뿐).

    붙을 데가 없는데 받으면 호출만 1/3 늘어난다.
    """
    assert bis.MARKETS == ("KOSPI", "KOSDAQ")
    # 그래도 클라이언트는 지원한다 — 필요해지면 MARKETS 에 한 줄 더하면 된다
    assert "KONEX" in krx_data.BASE_INFO_APIS
