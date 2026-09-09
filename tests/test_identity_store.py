"""`stock_identity` · `corp_profile` 저장 — 조용히 덮어쓰는 자리를 잠근다.

재무 수집에서 기본키에 `account_detail` 을 빼는 바람에 자본변동표의 **6.4%가
덮어써져 사라진** 적이 있다. 행 수만 세는 검사로는 안 잡혔다 — 덮어쓰기라 행 수가
그대로였기 때문이다. 그래서 여기서 재는 것은 행 수가 아니라 **무엇이 남았는가** 다.

망을 타지 않는다. DB 는 `conftest.py` 가 프로세스 전체에서 임시 경로로 갈아 끼운다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import trading_calendar as tc  # noqa: E402
from ingest.store import identity_store as ids  # noqa: E402
from ingest.store.migrations import migrate_path  # noqa: E402


@pytest.fixture
def db(tmp_path, monkeypatch):
    """이 시험만 쓰는 빈 DB. 마이그레이션을 v10 까지 올려 둔다."""
    경로 = tmp_path / "t.db"
    migrate_path(경로)
    import sqlite3
    conn = sqlite3.connect(경로)
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


def 신원행(**덮어쓸것):
    기본 = {
        "bas_dd": "20240822", "code": "000020", "isin_cd": "KR7000020008",
        "crno": "1101110043870", "corp_nm": "동화약품(주)", "item_nm": "동화약품",
        "market": "KOSPI", "known_at": "20240823", "known_rule": "basDt+1session",
    }
    return {**기본, **덮어쓸것}


def 개요행(**덮어쓸것):
    기본 = {
        "crno": "1101110043870", "fst_opeg_dt": "20260319",
        "last_opeg_dt": "20260901", "corp_nm": "동화약품(주)", "sic_nm": None,
        "estb_dt": "18970925", "stac_mm": "12", "xchg_lstg_dt": "19760324",
        "xchg_lstg_abol_dt": None, "kosdaq_lstg_dt": None,
        "kosdaq_lstg_abol_dt": None, "audt_rpt_opnn": "적정의견",
        "actn_audpn": "한울회계법인", "empe_cnt": 838,
        "pn1_avg_slry_amt": 72_000_000, "smenp_yn": None,
        "known_at": "20260319", "known_rule": "fstOpegDt",
    }
    return {**기본, **덮어쓸것}


# ==================================================
# 1. 기본키 — 무엇이 행을 가르는가
# ==================================================
def test_날짜가_다르면_다른_행이다(db):
    """🔴 종목명은 바뀌고 코드는 재사용된다. 날짜를 키에서 빼면 '그때는 누구였나' 를 잃는다."""
    ids.save_identity([신원행(bas_dd="20240822", item_nm="쓰리원"),
                       신원행(bas_dd="20250822", item_nm="UCI")], db)
    행들 = db.execute(
        "SELECT bas_dd, item_nm FROM stock_identity ORDER BY bas_dd").fetchall()
    assert len(행들) == 2
    assert [r["item_nm"] for r in 행들] == ["쓰리원", "UCI"]


def test_같은_날_같은_코드는_덮어쓴다(db):
    """다시 받아도 행이 늘지 않아야 한다 — 그래야 이어받기를 마음 놓고 돌린다."""
    ids.save_identity([신원행(item_nm="옛이름")], db)
    ids.save_identity([신원행(item_nm="새이름")], db)
    행들 = db.execute("SELECT item_nm FROM stock_identity").fetchall()
    assert len(행들) == 1
    assert 행들[0]["item_nm"] == "새이름"


def test_법인_개요는_유효시작일로_갈린다(db):
    """한 법인의 이력이 겹치지 않게 쌓여야 한다 — 최신이 과거를 덮으면 미래참조다."""
    ids.save_profile([개요행(fst_opeg_dt="20210316", empe_cnt=735),
                      개요행(fst_opeg_dt="20240321", empe_cnt=826),
                      개요행(fst_opeg_dt="20260319", empe_cnt=838)], db)
    행들 = db.execute(
        "SELECT fst_opeg_dt, empe_cnt FROM corp_profile ORDER BY fst_opeg_dt"
    ).fetchall()
    assert [r["empe_cnt"] for r in 행들] == [735, 826, 838]


def test_과거_종업원수가_최신에_덮이지_않는다(db):
    """🔴 이게 이력을 쌓는 이유다. 덮이면 2021년 모델이 2026년 종업원수를 본다."""
    ids.save_profile([개요행(fst_opeg_dt="20210316", empe_cnt=735)], db)
    ids.save_profile([개요행(fst_opeg_dt="20260319", empe_cnt=838)], db)
    옛값 = db.execute(
        "SELECT empe_cnt FROM corp_profile WHERE fst_opeg_dt='20210316'"
    ).fetchone()["empe_cnt"]
    assert 옛값 == 735


# ==================================================
# 2. 스키마 제약 — 접두사와 영문 코드
# ==================================================
def test_접두사가_남은_코드는_거부된다(db):
    """🔴 안 떼면 조인이 0행이 되는데, 조인은 0행이어도 에러가 아니다. 여기서 막는다."""
    import sqlite3
    with pytest.raises(sqlite3.IntegrityError):
        ids.save_identity([신원행(code="A00002")], db)


def test_일곱_자리_코드는_거부된다(db):
    import sqlite3
    with pytest.raises(sqlite3.IntegrityError):
        ids.save_identity([신원행(code="A000020")], db)


@pytest.mark.parametrize("코드", ["0001A0", "00088K", "0004V0", "000020"])
def test_영문이_낀_코드가_들어간다(db, 코드):
    """신형우선주 84종이다. 숫자로 단정한 검사를 넣으면 통째로 격리된다."""
    ids.save_identity([신원행(code=코드)], db)
    assert db.execute("SELECT COUNT(*) FROM stock_identity").fetchone()[0] == 1


def test_키가_비면_담지_않고_세운다(db):
    """빈 키로 넣으면 서로를 덮어써서 행 수는 그럴듯한데 내용이 사라진다."""
    with pytest.raises(ids.IdentityStoreError) as 잡힌것:
        ids.save_identity([신원행(code=None)], db)
    assert "할 일" in str(잡힌것.value)
    assert db.execute("SELECT COUNT(*) FROM stock_identity").fetchone()[0] == 0


# ==================================================
# 3. known_at — 계산값이라는 것을 잊지 않는다
# ==================================================
def test_known_at_은_다음_거래일이다(monkeypatch):
    """기준일 당일에는 그 자료를 볼 수 없었다 — 당일로 붙이면 미래참조다."""
    달력 = frozenset({"20210104", "20210105", "20210108", "20210111"})
    monkeypatch.setattr(tc, "_SESSION_CACHE", 달력)
    monkeypatch.setattr(tc, "_SESSION_SPAN", (min(달력), max(달력)))

    assert ids.known_at_for("20210104") == "20210105"
    # 🔴 05 다음은 06(달력에 없다)이 아니라 08 이다 — 날짜 계산으로 하면 틀린다.
    assert ids.known_at_for("20210105") == "20210108"


def test_달력_밖이면_지어내지_않고_세운다(monkeypatch):
    달력 = frozenset({"20210104", "20210105"})
    monkeypatch.setattr(tc, "_SESSION_CACHE", 달력)
    monkeypatch.setattr(tc, "_SESSION_SPAN", (min(달력), max(달력)))

    with pytest.raises(ids.IdentityStoreError) as 잡힌것:
        ids.known_at_for("20210105")
    assert "할 일" in str(잡힌것.value)


# ==================================================
# 4. 조회
# ==================================================
def test_개요가_없는_법인만_뽑는다(db):
    ids.save_identity([신원행(code="000020", crno="1101110043870"),
                       신원행(code="000030", crno="9999999999999")], db)
    ids.save_profile([개요행(crno="1101110043870")], db)
    assert ids.crno_targets(db) == ["9999999999999"]


def test_crno_가_비면_대상이_아니다(db):
    ids.save_identity([신원행(code="000040", crno=None),
                       신원행(code="000050", crno="")], db)
    assert ids.crno_targets(db) == []


def test_현황이_두_표를_함께_센다(db):
    ids.save_identity([신원행(bas_dd="20240822", code="000020"),
                       신원행(bas_dd="20240823", code="000020"),
                       신원행(bas_dd="20240823", code="000030")], db)
    ids.save_profile([개요행()], db)
    상태 = ids.status(db)
    assert 상태["stock_identity"] == {"rows": 3, "days": 2, "codes": 2,
                                     "first": "20240822", "last": "20240823"}
    assert 상태["corp_profile"]["rows"] == 1
    assert 상태["corp_profile"]["crno"] == 1


def test_빈_목록을_담으면_아무_일도_없다(db):
    assert ids.save_identity([], db) == 0
    assert ids.save_profile([], db) == 0


# ==================================================
# 5. 🔴 이 표는 '그 시점 상장 목록' 이 아니다
# ==================================================
def test_유니버스는_시세와_교집합을_내야_한다(db):
    """포털 목록에는 **그날 이후에야 상장된 종목**이 섞여 있다.

    실측(2026-09-03): `basDt=20200102` 목록 2,334종 중 33종이 그랬고, 듀켐바이오는
    첫 시세가 4년 뒤인 2024-12-20 이었다. 포털은 최신에 가까운 목록에 기준일 딱지만
    붙여 준다.

    이 시험은 코드를 잠그는 게 아니라 **그 사실을 잠근다** — 누가 `stock_identity`
    를 그대로 유니버스로 쓰면 여기서 왜 안 되는지 읽게 된다.
    """
    # `daily_price` 는 마이그레이션이 아니라 `krx_store.SCHEMA` 가 만든다. 여기서
    # 재려는 것은 두 표의 **관계**이므로 최소한의 칸만 세운다.
    db.execute("CREATE TABLE daily_price (bas_dd TEXT, code TEXT, name TEXT)")
    db.execute("INSERT INTO daily_price VALUES (?, ?, ?)",
               ("20200102", "000020", "동화약품"))
    # 듀켐바이오는 2020년 목록에 있지만 첫 시세는 2024년이다.
    db.execute("INSERT INTO daily_price VALUES (?, ?, ?)",
               ("20241220", "176750", "듀켐바이오"))
    ids.save_identity([신원행(bas_dd="20200102", code="000020"),
                       신원행(bas_dd="20200102", code="176750",
                            item_nm="듀켐바이오", crno="9999999999999")], db)

    포털목록 = {r[0] for r in db.execute(
        "SELECT code FROM stock_identity WHERE bas_dd='20200102'")}
    assert 포털목록 == {"000020", "176750"}, "포털은 아직 없던 종목도 준다"

    # 🔴 그대로 쓰면 안 된다. 시세와 교집합을 내야 그날의 유니버스가 된다.
    그날시세 = {r[0] for r in db.execute(
        "SELECT code FROM daily_price WHERE bas_dd='20200102'")}
    유니버스 = 포털목록 & 그날시세
    assert 유니버스 == {"000020"}
    assert "176750" not in 유니버스, "아직 상장 전인 종목이 유니버스에 들어왔다"
