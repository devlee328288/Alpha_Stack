"""유니버스 정문 — 오늘 알게 된 것으로 과거를 판정하지 못하게 잠근다.

모델 파트의 후보는 "날짜별 KOSPI 보통주 시가총액 상위 50" 이다(#92). 그 '보통주' 를
KRX 종목기본정보로 정하는데, **그 표에는 미래 날짜의 행도 들어 있다.** 2026년 스냅샷을
2015년 판정에 쓰면 미래참조가 되고 **에러는 나지 않는다.** 여기서 재는 것이 그것이다.

망을 타지 않는다. DB 는 `conftest.py` 가 프로세스 전체에서 임시 경로로 갈아 끼운다.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ingest.store import base_info_store as bis  # noqa: E402
from ingest.store import krx_store  # noqa: E402
from ingest.store.migrations import migrate_path  # noqa: E402
from supply import universe  # noqa: E402


@pytest.fixture
def db(tmp_path, monkeypatch):
    """이 시험만 쓰는 빈 DB.

    `supply` 는 연결을 인자로 받지 않고 저장소가 정한 경로로 스스로 연다 — 경로를
    갈아 끼우는 자리를 하나로 두기 위해서다. 그래서 `krx_store.DB_PATH` 를 바꾼다
    (`tests/test_supply_price.py` 와 같은 방식).
    """
    경로 = tmp_path / "u.db"
    monkeypatch.setattr(krx_store, "DB_PATH", 경로)
    # ⚠️ 표를 만드는 자리가 둘이다 — 시세(`daily_price`)는 `krx_store.SCHEMA` 가,
    #    `stock_base_info`·`trading_calendar` 는 마이그레이션이 만든다. 둘 다 돌린다.
    krx_store.init_db()
    migrate_path(경로)
    conn = sqlite3.connect(경로)
    yield conn
    conn.close()


def 시세(conn, bas_dd, code, name, cap, market="KOSPI"):
    conn.execute(
        "INSERT OR REPLACE INTO daily_price (bas_dd, code, name, market, close, "
        "market_cap, listed_shares) VALUES (?,?,?,?,?,?,?)",
        (bas_dd, code, name, market, 1000, cap, 100))


def 기본정보(conn, bas_dd, code, 종류, known_at, abbrv=""):
    conn.execute(
        "INSERT OR REPLACE INTO stock_base_info (bas_dd, code, isu_abbrv, market, "
        "kind_stkcert_tp_nm, known_at, known_rule, fetched_at) VALUES (?,?,?,?,?,?,?,?)",
        (bas_dd, code, abbrv or code, "KOSPI", 종류, known_at, "basDd+1session",
         "2026-09-03T00:00:00+09:00"))


def 달력(conn, *days):
    for d in days:
        for m in ("ALL", "KOSPI", "KOSDAQ"):
            conn.execute(
                "INSERT OR REPLACE INTO trading_calendar (bas_dd, market, stock_count, "
                "built_at) VALUES (?,?,?,?)", (d, m, 900, "2026-09-03T00:00:00+09:00"))


# ==================================================
# 1. 미래참조 — 이 시험이 이 파일의 존재 이유다
# ==================================================
def test_그때_몰랐던_기본정보는_보이지_않는다(db):
    """🔴 2026년에 확인한 주권종류를 2015년 판정에 쓰지 않는다.

    `stock_base_info` 를 직접 읽으면 최신 행이 딸려 온다. 정문이 `known_at` 으로
    막지 않으면 이 시험만 조용히 통과하지 못한다.
    """
    달력(db, "20150102", "20150105", "20260901", "20260902")
    시세(db, "20150102", "006800", "대우증권", 3_300_000_000_000)
    # 그때 알 수 있었던 행 — 보통주
    기본정보(db, "20150102", "006800", "보통주", "20150105", "대우증권")
    # 🔴 미래의 행. 만약 이게 보이면 판정이 뒤집힌다.
    기본정보(db, "20260901", "006800", "종류주권", "20260902", "미래에셋증권")
    db.commit()

    frame = universe.common_stocks("20150102", as_of="2015-01-06")
    assert list(frame["code"]) == ["006800"]
    assert frame.loc[0, "kind_stkcert_tp_nm"] == "보통주"
    # 어느 날짜의 기본정보를 썼는지도 표에 남는다 — 눈으로 확인할 수 있어야 한다
    assert frame.loc[0, "info_bas_dd"] == "20150102"


def test_같은_날_as_of_는_그_자체가_미래참조라_세운다(db):
    """🔴 20150102 종가는 그날 장중에 알 수 없다.

    `as_of` 를 그 거래일과 같은 날로 주면 아직 안 나온 종가로 시가총액 순위를
    매기게 된다. 정문이 여기서 세운다 — 조용히 답을 주면 누수가 그대로 흘러간다.
    """
    달력(db, "20150102", "20150105")
    시세(db, "20150102", "005930", "삼성전자", 100_000_000_000_000)
    기본정보(db, "20150102", "005930", "보통주", "20150105")
    db.commit()

    with pytest.raises(ValueError, match="아직 오지 않은 거래일"):
        universe.common_stocks("20150102", as_of="2015-01-02")
    # 다음 거래일 뒤에는 보인다
    assert len(universe.common_stocks("20150102", as_of="2015-01-06")) == 1


def test_기본정보의_known_at_이_아직_안_왔으면_그_행은_없는_것이다(db):
    """수집이 늦어 `known_at` 이 뒤로 밀린 행은 그때 못 본 것으로 친다.

    주권종류를 모르면 보통주가 아니라고 보므로 후보에서 빠진다 — 조용히 섞이는
    것보다 빠지는 쪽이 안전하다.
    """
    달력(db, "20150102", "20150105", "20150106", "20150107")
    시세(db, "20150102", "005930", "삼성전자", 100_000_000_000_000)
    # 🔴 기본정보는 같은 거래일 것이지만 `known_at` 이 이틀 더 뒤다
    기본정보(db, "20150102", "005930", "보통주", "20150107")
    db.commit()

    assert universe.common_stocks("20150102", as_of="2015-01-06").empty
    assert len(universe.common_stocks("20150102", as_of="2015-01-08")) == 1


def test_as_of_보다_뒤의_거래일을_물으면_세운다(db):
    """빈 표를 주면 '그날 상장 종목이 없었다' 로 오해된다. 세워서 알린다."""
    달력(db, "20150102", "20260901")
    db.commit()
    with pytest.raises(ValueError, match="아직 오지 않은 거래일"):
        universe.common_stocks("20260901", as_of="2015-01-06")


def test_as_of_는_기본값이_없다(db):
    """빠뜨리면 즉시 터진다 — 기본값이 '지금' 이면 빠뜨린 코드가 조용히 미래를 본다."""
    with pytest.raises(TypeError):
        universe.common_stocks("20150102")


# ==================================================
# 2. 보통주 판별 — 이름으로 추측하지 않는다
# ==================================================
def test_이름이_우로_끝나는_보통주가_후보에_남는다(db):
    """🔴 옛 규칙(`name.endswith('우')`)이 잘못 빼던 바로 그 종목이다.

    006800 은 20200102 코스피 시총 48위였다. 상위 50 후보에서 조용히 빠졌다.
    """
    달력(db, "20200102", "20200103")
    시세(db, "20200102", "006800", "미래에셋대우", 4_900_000_000_000)
    시세(db, "20200102", "005930", "삼성전자", 300_000_000_000_000)
    기본정보(db, "20200102", "006800", "보통주", "20200103", "미래에셋대우")
    기본정보(db, "20200102", "005930", "보통주", "20200103", "삼성전자")
    db.commit()

    codes = list(universe.common_stocks("20200102", as_of="2020-01-06")["code"])
    assert "006800" in codes, "이름이 '우' 로 끝나지만 정본이 보통주다 — 빠지면 안 된다"


@pytest.mark.parametrize("종류", ["구형우선주", "신형우선주", "종류주권"])
def test_보통주가_아니면_후보에서_빠진다(db, 종류):
    달력(db, "20200102", "20200103")
    시세(db, "20200102", "005935", "삼성전자우", 50_000_000_000_000)
    기본정보(db, "20200102", "005935", 종류, "20200103", "삼성전자우")
    db.commit()
    assert universe.common_stocks("20200102", as_of="2020-01-06").empty


def test_기본정보를_못_이으면_후보에서_빠지고_숫자로_남는다(db):
    """못 이은 종목은 주권종류를 몰라 전부 빠진다. 그게 조용히 커지지 않게 센다."""
    달력(db, "20200102", "20200103")
    시세(db, "20200102", "005930", "삼성전자", 300_000_000_000_000)
    시세(db, "20200102", "999999", "정체불명", 1_000_000_000_000)
    기본정보(db, "20200102", "005930", "보통주", "20200103")
    db.commit()

    assert list(universe.common_stocks("20200102", as_of="2020-01-06")["code"]) == ["005930"]
    c = universe.coverage("20200102", as_of="2020-01-06")
    assert c["시세종목"] == 2 and c["보통주"] == 1 and c["못이은"] == 1
    assert c["못이은코드"] == ["999999"]


def test_빠진_종목과_사유를_따로_낼_수_있다(db):
    """검증기가 개수만 말하면 사람은 확인하지 않는다. 무엇이 왜 빠졌는지를 낸다."""
    달력(db, "20200102", "20200103")
    시세(db, "20200102", "005935", "삼성전자우", 50_000_000_000_000)
    시세(db, "20200102", "999999", "정체불명", 1_000_000_000_000)
    기본정보(db, "20200102", "005935", "구형우선주", "20200103")
    db.commit()

    뺀것 = universe.excluded("20200102", as_of="2020-01-06")
    사유 = dict(zip(뺀것["code"], 뺀것["제외사유"], strict=True))
    assert 사유["005935"] == "구형우선주"
    assert "못 이었다" in 사유["999999"]


# ==================================================
# 3. 시가총액 순서 · 상위 N
# ==================================================
def test_시가총액_큰_순서로_나오고_상위_N만_자른다(db):
    달력(db, "20200102", "20200103")
    for i, cap in enumerate([300, 100, 200, 50], start=1):
        시세(db, "20200102", f"00000{i}", f"종목{i}", cap * 10**12)
        기본정보(db, "20200102", f"00000{i}", "보통주", "20200103")
    db.commit()

    frame = universe.top_by_market_cap("20200102", as_of="2020-01-06", top=3)
    assert list(frame["code"]) == ["000001", "000003", "000002"]
    assert len(frame) == 3


def test_시가총액이_없거나_0인_종목은_후보가_아니다(db):
    """시총이 없으면 순위를 매길 수 없다. 0 을 맨 아래에 두는 것과는 다르다."""
    달력(db, "20200102", "20200103")
    db.execute(
        "INSERT INTO daily_price (bas_dd, code, name, market, market_cap) "
        "VALUES ('20200102','000001','시총없음','KOSPI',NULL)")
    시세(db, "20200102", "000002", "시총0", 0)
    시세(db, "20200102", "000003", "정상", 10**12)
    for c in ("000001", "000002", "000003"):
        기본정보(db, "20200102", c, "보통주", "20200103")
    db.commit()

    assert list(universe.common_stocks("20200102", as_of="2020-01-06")["code"]) == ["000003"]


# ==================================================
# 4. 빈 결과 — 파이프라인을 멈추지 않는다
# ==================================================
def test_빈_결과에도_칸이_남는다(db):
    """칸이 없으면 부르는 쪽의 `frame["code"]` 가 KeyError 로 터진다."""
    달력(db, "20200102")
    db.commit()
    frame = universe.common_stocks("20200102", as_of="2020-01-06")
    assert frame.empty
    assert list(frame.columns) == list(universe.UNIVERSE_COLUMNS)


# ==================================================
# 5. 정문을 통해서만 쓴다
# ==================================================
def test_supply_정문에_노출돼_있다():
    import supply
    for 이름 in ("common_stocks", "top_by_market_cap", "excluded", "coverage"):
        assert 이름 in supply.__all__, f"{이름} 이 정문에 없다"
        assert hasattr(supply, 이름)


def test_기본정보가_그날_없으면_이전_최근_것을_쓴다(db):
    """수집 주기를 성기게 바꿔도 답이 나와야 한다.

    주권종류는 실측상 15년간 바뀐 종목이 0건이라 이 대입이 안전하다.
    다만 **미래 것을 끌어오지는 않는다** — 위 미래참조 시험이 그것을 잠근다.
    """
    달력(db, "20200102", "20200103", "20200110", "20200113")
    시세(db, "20200110", "005930", "삼성전자", 300_000_000_000_000)
    기본정보(db, "20200102", "005930", "보통주", "20200103")   # 8일 전 스냅샷뿐
    db.commit()

    frame = universe.common_stocks("20200110", as_of="2020-01-14")
    assert len(frame) == 1
    assert frame.loc[0, "info_bas_dd"] == "20200102"
    assert bis  # 저장소를 거쳐 왔다는 표시 (import 가 죽은 코드로 지워지지 않게)
