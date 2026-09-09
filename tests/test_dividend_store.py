"""배당 수집·적재 — 기준일이 아니라 **배당락일**이 값이 움직이는 날이다.

이 파일이 잠그는 것 셋.

① **배당락일 계산** — 기준일에서 거래일 두 걸음 앞이다. 날짜 계산(−1일·−3일)으로 하면
   휴장·연휴에서 어긋난다. 2023-12-31 기준 → 12-27 이 실제 배당락일이고, 그 사이에
   12-29·12-30·12-31 이 휴장이라 **달력 없이는 못 맞힌다.**
② **한 ISIN 이 두 코드에 붙는 것** — 그대로 조인하면 배당 한 줄이 두 줄로 불어난다.
③ **키가 빈 행** — 재무에서 `account_detail` 을 기본키에서 빠뜨려 6.4%가 조용히 사라진
   적이 있다. 여기서는 담기 전에 세운다.

망을 타지 않는다. DB 는 임시 경로를 쓰고, 달력은 `monkeypatch` 로 갈아 끼운다.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import trading_calendar as tc  # noqa: E402
from ingest.clients import data_go_kr  # noqa: E402
from ingest.store import dividend_store as ds  # noqa: E402
from ingest.store.migrations import migrate_path  # noqa: E402

#: 2023년 12월 말의 **실제** 거래일. 12-29 부터 연말 휴장이라 폐장일이 12-28 이다.
_2023_연말 = frozenset({
    "20231220", "20231221", "20231222", "20231226", "20231227", "20231228",
    "20240102", "20240103", "20240104",
})


@pytest.fixture
def 달력(monkeypatch):
    """거래일 달력을 2023년 연말로 갈아 끼운다.

    `_sorted_days` 는 전역 변수로 캐시하고 집합의 **동일성**으로 판정하므로, 집합을
    바꾸면 저절로 다시 정렬한다. 다만 이 시험이 끝난 뒤 다른 시험이 진짜 달력을 쓸 때
    캐시가 남아 있으면 안 되므로 뒷정리에서 비운다.
    """
    monkeypatch.setattr(tc, "_SESSION_CACHE", _2023_연말)
    monkeypatch.setattr(tc, "_SESSION_SPAN", (min(_2023_연말), max(_2023_연말)))
    monkeypatch.setattr(tc, "_SESSION_SORTED", None)
    monkeypatch.setattr(tc, "_SORTED_FOR", None)
    yield _2023_연말


@pytest.fixture
def db(tmp_path):
    """이 시험만 쓰는 빈 DB. 마이그레이션을 v14 까지 올려 둔다."""
    경로 = tmp_path / "t.db"
    migrate_path(경로)
    conn = sqlite3.connect(경로)
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


def 배당행(**덮어쓸것):
    기본 = {
        "isin_cd": "KR7005930003", "dvdn_bas_dt": "20231231", "dvdn_rcd": "02",
        "crno": "1301110006246", "corp_nm": "삼성전자", "item_nm": "삼성전자",
        "scrs_itms_kcd_nm": "보통주", "stac_md": "12", "dvdn_rcd_nm": "현금배당",
        "cash_pay_dt": "20240419", "stck_hndv_dt": None,
        "genr_dvdn_amt": 361.0, "grdn_dvdn_amt": 0.0,
        "genr_cash_dvdn_rt": 361.0, "genr_dvdn_rt": 0.0,
        "cash_grdn_dvdn_rt": 0.0, "grdn_dvdn_rt": 0.0,
        "par_price_at_load": 100.0, "src_bas_dt": "20260908",
    }
    return {**기본, **덮어쓸것}


# ══════════════════════════════════════════════════════════════════════════
# 1. 배당락일 — 이 모듈의 존재 이유
# ══════════════════════════════════════════════════════════════════════════
def test_배당락일은_기준일에서_거래일_두_걸음_앞이다(달력):
    """2023-12-31(기준) → 12-28(폐장) → **12-27 배당락**. 실제와 같다."""
    assert ds.ex_date_for("20231231") == "20231227"


def test_날짜_빼기로는_못_맞힌다(달력):
    """12-31 에서 사흘을 빼면 12-28 이다 — 폐장일이지 배당락일이 아니다.

    연말 휴장(12-29~31)이 사이에 있어서, 달력을 안 보면 반드시 어긋난다.
    """
    from datetime import datetime, timedelta
    사흘전 = (datetime.strptime("20231231", "%Y%m%d") - timedelta(days=3)).strftime("%Y%m%d")
    assert 사흘전 == "20231228"
    assert ds.ex_date_for("20231231") != 사흘전


def test_기준일이_거래일이어도_한_걸음_더_간다(달력):
    """기준일이 열린 날이면 그 날이 '마지막 거래일' 이고, 배당락은 그 전날이다."""
    assert ds.ex_date_for("20231228") == "20231227"
    assert ds.ex_date_for("20231227") == "20231226"


def test_달력_밖은_지어내지_않고_비운다(달력):
    """배당 자료는 1985년부터 온다. 옛 행마다 예외를 세우면 전량 적재가 못 끝난다."""
    assert ds.ex_date_for("19851231") is None
    assert ds.ex_date_for("20301231") is None


# ══════════════════════════════════════════════════════════════════════════
# 2. 종목코드 붙이기
# ══════════════════════════════════════════════════════════════════════════
def _기본정보(conn, rows):
    conn.executemany(
        "INSERT INTO stock_base_info (bas_dd, code, isin_cd, market, known_at, "
        "known_rule, fetched_at) VALUES (?, ?, ?, 'KOSPI', ?, 'x', 'x')",
        [(d, c, i, d) for d, c, i in rows],
    )
    conn.commit()


def test_ISIN_이_두_코드에_붙으면_가장_최근_것만_남긴다(db):
    """그대로 조인하면 배당 한 줄이 두 줄로 불어난다 — 행 수는 늘고 아무도 안 본다."""
    _기본정보(db, [("20200102", "111111", "KR7111111111"),
                ("20240102", "222222", "KR7111111111")])
    지도 = ds.isin_to_code(db)
    assert 지도 == {"KR7111111111": "222222"}


def test_모르는_ISIN_은_코드가_빈다(db, 달력):
    _기본정보(db, [("20240102", "005930", "KR7005930003")])
    ds.save([배당행(isin_cd="KR9999999999")], db)
    행 = db.execute("SELECT code FROM dividend").fetchone()
    assert 행["code"] is None


# ══════════════════════════════════════════════════════════════════════════
# 3. 담기 — 무엇이 남는가
# ══════════════════════════════════════════════════════════════════════════
def test_담을_때_코드와_배당락일을_채운다(db, 달력):
    _기본정보(db, [("20240102", "005930", "KR7005930003")])
    assert ds.save([배당행()], db) == 1
    행 = db.execute("SELECT * FROM dividend").fetchone()
    assert 행["code"] == "005930"
    assert 행["ex_date"] == "20231227"
    assert 행["ex_date_rule"] == ds.EX_DATE_RULE


def test_배당락일을_못_구하면_규칙도_비운다(db, 달력):
    """계산한 값과 못 구한 값을 구별할 수 있어야 한다 — 규칙 칸이 그 표시다."""
    ds.save([배당행(dvdn_bas_dt="19851231")], db)
    행 = db.execute("SELECT ex_date, ex_date_rule FROM dividend").fetchone()
    assert 행["ex_date"] is None and 행["ex_date_rule"] is None


def test_배당구분이_다르면_다른_행이다(db, 달력):
    """한 종목·한 기준일에 현금배당과 주식배당이 함께 오는 해가 있다."""
    ds.save([배당행(dvdn_rcd="02"), 배당행(dvdn_rcd="01", dvdn_rcd_nm="주식배당")], db)
    assert db.execute("SELECT COUNT(*) FROM dividend").fetchone()[0] == 2


def test_같은_키는_덮어쓴다(db, 달력):
    """자료가 정정되면 다시 받아 덮는다. 두 줄이 되면 어느 것이 맞는지 알 수 없다."""
    ds.save([배당행(genr_dvdn_amt=361.0)], db)
    ds.save([배당행(genr_dvdn_amt=363.0)], db)
    행들 = db.execute("SELECT genr_dvdn_amt FROM dividend").fetchall()
    assert len(행들) == 1 and 행들[0][0] == 363.0


@pytest.mark.parametrize("빈칸", ["isin_cd", "dvdn_bas_dt", "dvdn_rcd"])
def test_키가_비면_담지_않고_세운다(db, 달력, 빈칸):
    with pytest.raises(ds.DividendStoreError) as 잡힌것:
        ds.save([배당행(**{빈칸: None})], db)
    assert "할 일" in str(잡힌것.value)


def test_빈_목록을_담으면_아무_일도_없다(db):
    assert ds.save([], db) == 0


# ══════════════════════════════════════════════════════════════════════════
# 4. 현황 — 행 수만 세지 않는다
# ══════════════════════════════════════════════════════════════════════════
def test_현황은_금액과_배당락일_채움까지_센다(db, 달력):
    _기본정보(db, [("20240102", "005930", "KR7005930003")])
    ds.save([
        배당행(),                                                   # 금액 O · 락일 O · 코드 O
        배당행(dvdn_rcd="03", genr_dvdn_amt=0.0),                    # 금액 X
        배당행(isin_cd="KR9999999999", dvdn_rcd="04"),               # 코드 X
        배당행(dvdn_rcd="05", dvdn_rcd_nm="무배당"),                  # 현금이 아니다
    ], db)
    현황 = ds.coverage(db)
    assert 현황["rows"] == 4
    assert 현황["cash_rows_since"] == 3                              # 무배당은 빠진다
    assert 현황["with_code"] == 2
    assert 현황["with_amount"] == 2
    assert 현황["with_ex_date"] == 3


def test_표가_비면_행_수만_준다(db):
    assert ds.coverage(db) == {"rows": 0}


# ══════════════════════════════════════════════════════════════════════════
# 5. 파서 — 외부 연동 계층은 달력을 모른다
# ══════════════════════════════════════════════════════════════════════════
def test_파서는_배당락일을_채우지_않는다():
    """계층 분리 — 달력을 아는 것은 저장 계층뿐이다."""
    행 = data_go_kr.parse_dividend_row({
        "basDt": "20260908", "isinCd": "KR7005930003", "dvdnBasDt": "20231231",
        "stckDvdnRcd": "02", "stckDvdnRcdNm": "현금배당", "stckGenrDvdnAmt": "361",
    })
    assert 행 is not None
    assert "ex_date" not in 행
    assert 행["src_bas_dt"] == "20260908"          # 적재일은 기준일과 다른 칸에 담는다


@pytest.mark.parametrize("없는칸", ["isinCd", "dvdnBasDt", "stckDvdnRcd"])
def test_파서는_키가_없으면_None_을_준다(없는칸):
    항목 = {"isinCd": "KR7005930003", "dvdnBasDt": "20231231", "stckDvdnRcd": "02"}
    항목.pop(없는칸)
    assert data_go_kr.parse_dividend_row(항목) is None


def test_배당률은_소수로_읽는다():
    """정수로 읽으면 2.5% 가 2% 로 조용히 잘린다."""
    행 = data_go_kr.parse_dividend_row({
        "isinCd": "KR7005930003", "dvdnBasDt": "20231231", "stckDvdnRcd": "02",
        "stckGenrCashDvdnRt": "2.5",
    })
    assert 행["genr_cash_dvdn_rt"] == 2.5


def test_액면가_칸_이름이_적재_시점임을_말한다():
    """`stckParPrc` 는 그날의 액면가가 아니라 **적재 시점** 값이다.

    삼성전자 1987년 배당 행에도 100원(2018년 분할 뒤 값)이 들어 있다. 옛 행의
    배당률에 이 값을 곱하면 틀리므로, 이름으로 그 사실을 남긴다.
    """
    assert "par_price_at_load" in data_go_kr._DIVIDEND_NUM.values()
    assert "par_price" not in data_go_kr._DIVIDEND_NUM.values()


def test_현금배당_구분은_둘뿐이다():
    """주식배당은 FinanceDataReader 가 이미 수정주가로 편다 — 또 빼면 두 번 뺀다."""
    assert data_go_kr.CASH_DIVIDEND_KINDS == frozenset({"현금배당", "동시배당"})
