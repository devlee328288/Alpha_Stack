"""`ingest/store/dart_store.py` — DART 재무를 받아 담는 계층의 시험대.

가장 중요한 것 둘을 본다.

  ① **행이 조용히 사라지지 않는가** — 자본변동표(SCE)는 `account_detail` 로만 갈린다.
     이 칸이 기본키에 없으면 삼성전자 2023 이 176줄 → 135줄로 줄어드는데
     예외도 경고도 나지 않는다 (실측 2026-09-02).
  ② **빈 응답과 오류를 가르는가** — 순서를 뒤집으면 "보고서가 아예 없는 것"이
     "접수일을 못 찾은 오류"로 남고, 영원히 없을 자료를 매번 다시 부르게 된다.
"""

from __future__ import annotations

import sqlite3
from typing import Dict, List

import pytest

from ingest.store import dart_store, migrations


# ── 시험용 응답 ─────────────────────────────────────────────────────────────
# DART 가 실제로 주는 모양을 그대로 흉내 낸다. 실측에서 확인한 두 가지를 담았다.
#   · bsns_year 가 2023.0, ord 가 4.0 처럼 **실수**로 온다
#   · SCE 는 같은 (account_nm, ord) 가 account_detail 만 다른 채 여러 줄 온다
def _응답(계정: List[Dict]) -> Dict:
    묶음: Dict[str, List[Dict]] = {}
    for line in 계정:
        묶음.setdefault(line["sj_div"], []).append(line)
    return {
        "corp_code": "00126380", "corp_name": "삼성전자",
        "bsns_year": 2023.0, "reprt_code": "11011", "reprt_name": "사업보고서",
        "fs_div": "CFS", "rcept_no": "20240312000736", "rcept_dt": "20240312",
        "statements": 묶음, "count": len(계정), "empty": False,
    }


def _줄(sj_div: str, nm: str, ord_: float, detail: str, amount: float) -> Dict:
    return {
        "corp_code": "00126380", "bsns_year": 2023.0, "reprt_code": "11011",
        "rcept_no": "20240312000736", "fs_div": "CFS", "rcept_dt": "20240312",
        "sj_div": sj_div, "account_id": "dart_EquityAtBeginningOfPeriod",
        "account_nm": nm, "account_detail": detail, "ord": ord_,
        "currency": "KRW", "thstrm_nm": "제 55 기", "thstrm_amount": amount,
        "frmtrm_amount": None, "bfefrmtrm_amount": None,
    }


@pytest.fixture()
def 빈DB(tmp_path, monkeypatch):
    """v6 까지 적용한 빈 DB. **저장과 수집대장이 같은 DB 를 보게** 한다.

    ⚠️ 두 곳을 함께 갈아 끼워야 한다. `krx_store` 는 모듈 상수 `DB_PATH` 를 쓰고,
       `collect_log` 는 부를 때마다 `common.paths.krx_db_path()` 로 환경변수를 읽는다.
       한쪽만 바꾸면 저장은 임시 DB 에, 대장은 다른 DB 에 남아 시험이 현실과 갈린다.
       (`tests/conftest.py` 가 이미 실제 DB 는 막아 주지만, 그건 테스트끼리 공유하는
        DB 라 이 시험의 격리에는 부족하다.)
    """
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db, isolation_level=None)
    migrations.migrate(conn)
    conn.close()

    from ingest.store import krx_store
    monkeypatch.setattr(krx_store, "DB_PATH", db)
    monkeypatch.setenv("KRX_DB_PATH", str(db))
    return db


# ── ① 행이 사라지지 않는가 ──────────────────────────────────────────────────

def test_자본변동표의_같은_계정이_열마다_남는다(빈DB):
    """🔴 이 시험이 깨지면 재무의 6.4% 가 조용히 사라진다 (실측 22,436행).

    '기초자본' 은 자본금·주식발행초과금·이익잉여금… 열마다 한 줄씩 오고
    계정명도 ord 도 account_id 도 전부 같다. 갈라 주는 것은 account_detail 뿐이다.
    """
    열들 = ["연결재무제표 [member]", "자본금 [구성요소]", "이익잉여금 [구성요소]",
            "비지배지분 [구성요소]"]
    out = _응답([_줄("SCE", "기초자본", 4.0, d, 100.0 + i)
                 for i, d in enumerate(열들)])

    저장수 = dart_store.save(out, "005930")

    with sqlite3.connect(빈DB) as con:
        실제 = con.execute("SELECT COUNT(*) FROM dart_financial").fetchone()[0]
        금액 = {r[0] for r in con.execute(
            "SELECT thstrm_amount FROM dart_financial WHERE account_nm='기초자본'")}

    assert 저장수 == len(열들)
    assert 실제 == len(열들), "열마다 한 줄씩 남아야 한다 — 덮어쓰였다"
    assert 금액 == {100.0, 101.0, 102.0, 103.0}, "값이 서로 다른데 하나로 뭉개졌다"


def test_같은_줄을_두_번_넣으면_한_줄이다(빈DB):
    """재수집은 갱신이어야 한다 — 같은 열쇠가 쌓이면 안 된다."""
    out = _응답([_줄("BS", "자산총계", 7.0, "-", 455.0)])

    dart_store.save(out, "005930")
    dart_store.save(out, "005930")

    with sqlite3.connect(빈DB) as con:
        assert con.execute("SELECT COUNT(*) FROM dart_financial").fetchone()[0] == 1


def test_계정상세가_비면_NULL_이_아니라_빈문자열이다(빈DB):
    """SQLite 는 기본키 안의 NULL 을 서로 다른 값으로 본다 — 그러면 중복이 쌓인다."""
    out = _응답([_줄("BS", "자산총계", 7.0, "", 455.0)])

    dart_store.save(out, "005930")
    dart_store.save(out, "005930")          # 같은 줄을 다시

    with sqlite3.connect(빈DB) as con:
        n, detail = con.execute(
            "SELECT COUNT(*), account_detail FROM dart_financial").fetchone()

    assert n == 1, "빈 계정상세가 NULL 이면 같은 줄이 두 번 쌓인다"
    assert detail == ""


# ── 숫자 모양 ───────────────────────────────────────────────────────────────

def test_실수로_오는_연도와_순서를_정수로_바꾼다(빈DB):
    """DART 는 bsns_year 를 2023.0, ord 를 4.0 으로 준다 (실측).

    그대로 넣으면 기본키에 실수가 들어가 2023.0 과 2023 이 다른 행이 된다.
    """
    out = _응답([_줄("BS", "자산총계", 7.0, "-", 455.0)])

    dart_store.save(out, "005930")

    with sqlite3.connect(빈DB) as con:
        year, ord_, t1, t2 = con.execute(
            "SELECT bsns_year, ord, typeof(bsns_year), typeof(ord) "
            "FROM dart_financial").fetchone()

    assert (year, ord_) == (2023, 7)
    assert (t1, t2) == ("integer", "integer")


def test_계정명이_빈_줄은_넣지_않는다(빈DB):
    """account_nm 이 기본키의 일부라, 빈 문자열이면 서로 다른 계정이 한 줄로 뭉개진다."""
    out = _응답([_줄("BS", "", 7.0, "-", 1.0), _줄("BS", "자산총계", 8.0, "-", 2.0)])

    저장수 = dart_store.save(out, "005930")

    assert 저장수 == 1
    with sqlite3.connect(빈DB) as con:
        assert con.execute(
            "SELECT account_nm FROM dart_financial").fetchone()[0] == "자산총계"


# ── 종목코드·시점 ───────────────────────────────────────────────────────────

def test_종목코드와_접수일이_모든_줄에_붙는다(빈DB):
    """응답의 계정 줄에는 종목코드가 없다 — 부르는 쪽이 채워 넣어야 한다."""
    out = _응답([_줄("BS", "자산총계", 7.0, "-", 1.0),
                 _줄("IS", "매출액", 3.0, "-", 2.0)])

    dart_store.save(out, "005930")

    with sqlite3.connect(빈DB) as con:
        줄들 = con.execute(
            "SELECT stock_code, corp_name, rcept_dt FROM dart_financial").fetchall()

    assert all(r == ("005930", "삼성전자", "20240312") for r in 줄들)


def test_수집_대장의_열쇠는_사람이_읽을_수_있다():
    """고유번호가 아니라 종목코드로 적는다 — 대장은 사람이 읽는다."""
    assert dart_store.target_of("005930", 2023, "11011") == "005930:2023:11011"


# ── ② 빈 응답과 오류를 가르는가 ─────────────────────────────────────────────

def test_보고서가_없으면_오류가_아니라_없음이다(빈DB, monkeypatch):
    """🔴 순서를 뒤집으면 영원히 없을 자료를 실행할 때마다 다시 부른다.

    실측 2026-09-02: 350종 × 5개년에서 104건이 error 로 남았는데 전부 빈 응답이었다.
    `should_collect` 는 error 를 3회까지 재시도하므로 매번 208콜을 헛되이 쓴다.
    """
    from ingest.clients import dart_data

    monkeypatch.setattr(dart_store, "load_universe",
                        lambda: {"483650": {"name": "달바글로벌"}})
    # 상장 전 연도의 실제 응답 모양 (실측): 계정도 접수일도 없다
    monkeypatch.setattr(dart_data, "fetch_financials",
                        lambda *a, **k: {"empty": True, "statements": {},
                                         "count": 0, "rcept_no": "", "rcept_dt": ""})

    result = dart_store.sync(codes=["483650"], years=[2021])

    assert result["empty"] == 1, "빈 응답은 empty 로 세야 한다"
    assert result["error"] == 0, "빈 응답을 error 로 세면 매번 다시 부르게 된다"


def test_계정은_왔는데_접수일이_없으면_저장하지_않는다(빈DB, monkeypatch):
    """🔴 시점을 못 세우는 값을 넣으면 결산기에 값을 붙이게 된다 — 조용한 미래 참조다."""
    from ingest.clients import dart_data

    monkeypatch.setattr(dart_store, "load_universe",
                        lambda: {"005930": {"name": "삼성전자"}})
    깨진응답 = _응답([_줄("BS", "자산총계", 7.0, "-", 455.0)])
    깨진응답["rcept_dt"] = ""
    for lines in 깨진응답["statements"].values():
        for line in lines:
            line["rcept_dt"] = ""
    monkeypatch.setattr(dart_data, "fetch_financials", lambda *a, **k: 깨진응답)

    result = dart_store.sync(codes=["005930"], years=[2023])

    assert result["error"] == 1
    assert result["rows"] == 0
    with sqlite3.connect(빈DB) as con:
        assert con.execute("SELECT COUNT(*) FROM dart_financial").fetchone()[0] == 0


# ── 마이그레이션 v6 ─────────────────────────────────────────────────────────

def test_v6_이_기본키에_계정상세를_넣는다(빈DB):
    with sqlite3.connect(빈DB) as con:
        pk = [r[1] for r in con.execute("PRAGMA table_info(dart_financial)") if r[5]]

    assert "account_detail" in pk, "이 칸이 없으면 자본변동표가 뭉개진다"
    assert set(pk) == {"corp_code", "bsns_year", "reprt_code", "fs_div",
                       "sj_div", "account_nm", "ord", "account_detail"}


def test_v6_이_규격의_값만_받는다(빈DB):
    """enum 밖의 값은 CHECK 이 막는다. 새 표라 검사할 기존 행이 없어 공짜다."""
    정상 = ("00126380", "005930", "삼성전자", 2023, "11011", "CFS", "BS",
            None, "자산총계", "", 7, "KRW", "제 55 기", 1.0, None, None,
            "20240312000736", "20240312", "사업보고서", None, "2026-09-02T00:00:00+09:00")
    자리 = ",".join("?" * len(정상))

    with sqlite3.connect(빈DB) as con:
        con.execute(f"INSERT INTO dart_financial VALUES ({자리})", 정상)

        for 자리번호, 나쁜값 in ((4, "99999"), (5, "XXX"), (6, "ZZ")):
            깨진 = list(정상)
            깨진[자리번호] = 나쁜값
            with pytest.raises(sqlite3.IntegrityError):
                con.execute(f"INSERT INTO dart_financial VALUES ({자리})", 깨진)


def test_v6_은_두_번_돌려도_안전하다(빈DB):
    """이미 최신이면 아무것도 하지 않는다 — 여러 번 불러도 같은 결과여야 한다."""
    conn = sqlite3.connect(빈DB, isolation_level=None)
    try:
        assert migrations.migrate(conn) == 0
        assert migrations.user_version(conn) == migrations.LATEST_VERSION
    finally:
        conn.close()
