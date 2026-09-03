"""업종 정문 — 뒤의 스냅샷을 앞으로 당겨 쓰지 못하게 잠근다.

업종분류 현황은 연 1회 손으로 받은 스냅샷이다. 날짜 `d` 의 업종은 *"`d` 이전 가장 최근
스냅샷"* 에서 읽어야 하고, **2026년 스냅샷으로 2015년을 판정하면 미래참조인데 에러는
나지 않는다.** 여기서 재는 것이 그것이다.

망을 타지 않는다. DB 는 임시 경로를 만들어 `db_path` 로 넘긴다.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import trading_calendar  # noqa: E402
from ingest.inbox import store as inbox_store  # noqa: E402
from ingest.store import krx_store  # noqa: E402
from ingest.store.migrations import migrate_path  # noqa: E402
from supply import sector  # noqa: E402


@pytest.fixture
def db(tmp_path, monkeypatch):
    """이 시험만 쓰는 빈 DB. 달력 캐시는 모듈 전역이라 비워서 시작한다."""
    경로 = tmp_path / "s.db"
    monkeypatch.setattr(krx_store, "DB_PATH", 경로)
    monkeypatch.setattr(trading_calendar, "_SESSION_CACHE", None)
    monkeypatch.setattr(trading_calendar, "_SESSION_SPAN", (None, None))
    krx_store.init_db()
    migrate_path(경로)
    conn = sqlite3.connect(경로)
    yield conn, 경로
    conn.close()


def 달력(conn, *days):
    for d in days:
        for m in ("ALL", "KOSPI", "KOSDAQ"):
            conn.execute(
                "INSERT OR REPLACE INTO trading_calendar (bas_dd, market, stock_count, "
                "built_at) VALUES (?,?,?,?)", (d, m, 900, "2026-09-03T00:00:00+09:00"))
    conn.commit()


def 스냅샷(db_path: Path, tmp_path: Path, bas_dd: str, rows: dict, market: str = "KOSPI"):
    """반입 엔진이 통과시킨 것처럼 `inbox_accepted` 에 넣는다 — 엔진을 거치지 않고
    `store.load_result` 만 쓴다. 검사기의 시험은 따로 있다."""
    src = tmp_path / f"업종분류현황_{market}_{bas_dd}.csv"
    # 🔴 파일 내용이 같으면 SHA-256 이 같고 batch_id 가 같아져 앞 스냅샷을 **덮어쓴다**
    #    (INSERT OR REPLACE). 처음엔 "x" 로 두었다가 이 시험이 그것을 잡았다.
    src.write_text(f"{market},{bas_dd},{len(rows)}", encoding="utf-8")
    accepted = pd.DataFrame([
        {"row_no": i, "kind": sector.SECTOR_KIND, "key_hash": f"{bas_dd}{code}",
         "payload": {"bas_dd": bas_dd, "code": code, "name": code, "market": market,
                     "sector_nm": nm, "close": 1000, "change": 0, "change_rate": 0.0,
                     "market_cap": 1_000_000},
         "extras": None, "changes": None, "warnings": None}
        for i, (code, nm) in enumerate(rows.items())
    ])
    result = SimpleNamespace(kind=sector.SECTOR_KIND, accepted=accepted,
                             quarantined=pd.DataFrame(), report={"schema_version": "1.0"},
                             rows_total=len(accepted), rejected=None)
    # origin 은 CHECK 제약으로 local·huggingface 뿐이다. 손으로 받은 것은 local + contributor.
    inbox_store.load_result(result, src, db_path=db_path, origin="local",
                            contributor="시험 · 화면 다운로드")


# ==================================================
# 1. 미래참조 — 이 시험이 이 파일의 존재 이유다
# ==================================================
def test_뒤의_스냅샷을_앞으로_당겨_쓰지_않는다(db, tmp_path):
    """🔴 2020년 스냅샷에서 삼성전자가 '전기·전자' 라도 2018년 판정은 2015년 것으로 한다."""
    conn, path = db
    달력(conn, "20150102", "20150105", "20200102", "20200103", "20200106")
    스냅샷(path, tmp_path, "20150102", {"005930": "전기전자(옛)", "000660": "전기전자(옛)"})
    스냅샷(path, tmp_path, "20200102", {"005930": "전기·전자", "000660": "전기·전자"})

    frame = sector.industry_as_of("20180101", as_of="2026-09-03", db_path=path)
    assert set(frame["code"]) == {"005930", "000660"}
    assert set(frame["industry"]) == {"전기전자(옛)"}
    assert set(frame["industry_bas_dd"]) == {"20150102"}
    # 어느 스냅샷을 썼는지 표에 남는다 — 눈으로 확인할 수 있어야 한다
    assert set(frame["industry_known_at"]) == {"20150105"}


def test_그때_아직_몰랐던_스냅샷은_보이지_않는다(db, tmp_path):
    """🔴 20200102 스냅샷은 그날 마감 뒤에 나온다. `as_of` 가 그날이면 2015년 것으로 답한다."""
    conn, path = db
    달력(conn, "20150102", "20150105", "20200102", "20200103", "20200106")
    스냅샷(path, tmp_path, "20150102", {"005930": "옛"})
    스냅샷(path, tmp_path, "20200102", {"005930": "새"})

    # 20200102 낮 — known_at(20200103) 이 아직 안 왔다
    옛 = sector.industry_as_of("20200102", as_of="2020-01-02", db_path=path)
    assert list(옛["industry"]) == ["옛"]
    # 다음 거래일 뒤에는 보인다
    새 = sector.industry_as_of("20200102", as_of="2020-01-04", db_path=path)
    assert list(새["industry"]) == ["새"]


def test_스냅샷이_없는_구간은_빈_표이지_오류가_아니다(db, tmp_path):
    conn, path = db
    달력(conn, "20150102", "20150105")
    스냅샷(path, tmp_path, "20150102", {"005930": "옛"})

    frame = sector.industry_as_of("20141231", as_of="2026-09-03", db_path=path)
    assert frame.empty
    # 🔴 빈 표에도 칸은 있어야 한다 — 받는 쪽의 df["industry"] 가 KeyError 로 죽지 않게
    assert list(frame.columns) == ["code", *sector.INDUSTRY_COLUMNS]


def test_as_of_없이는_못_지난다(db):
    with pytest.raises(TypeError):
        sector.industry_as_of("20150102")  # type: ignore[call-arg]


# ==================================================
# 2. 시세 표에 붙이기 — 행마다 그 행의 날짜 이전 스냅샷
# ==================================================
def test_행마다_그_날짜_이전_가장_최근_스냅샷이_붙고_순서는_그대로다(db, tmp_path):
    conn, path = db
    달력(conn, "20150102", "20150105", "20200102", "20200103", "20200106")
    스냅샷(path, tmp_path, "20150102", {"005930": "옛"})
    스냅샷(path, tmp_path, "20200102", {"005930": "새", "000660": "새"})

    # 일부러 뒤섞인 순서 — merge_asof 가 정렬을 요구해도 입력 순서가 지켜져야 한다
    prices = pd.DataFrame({
        "bas_dd": ["20200106", "20141231", "20150102", "20190701", "20200106"],
        "code":   ["005930",   "005930",   "005930",   "005930",   "000660"],
        "close":  [1, 2, 3, 4, 5],
    })
    out = sector.attach_industry(prices, as_of="2026-09-03", db_path=path)

    assert list(out["close"]) == [1, 2, 3, 4, 5]
    assert list(out["industry"].fillna("-")) == ["새", "-", "옛", "옛", "새"]
    assert list(out["industry_bas_dd"].fillna("-")) == [
        "20200102", "-", "20150102", "20150102", "20200102"]
    # 원래 칸은 건드리지 않는다 — sector(소속부) 와 industry(업종) 는 다른 칸이다
    assert "sector" not in out.columns


def test_스냅샷이_하나도_없으면_빈_칸으로_돌려준다(db):
    """반출이 업종 때문에 죽어서는 안 된다. 빈 칸은 눈에 띈다."""
    conn, path = db
    달력(conn, "20150102", "20150105")
    prices = pd.DataFrame({"bas_dd": ["20150102"], "code": ["005930"]})
    out = sector.attach_industry(prices, as_of="2026-09-03", db_path=path)
    assert list(out.columns) == ["bas_dd", "code", *sector.INDUSTRY_COLUMNS]
    assert out["industry"].isna().all()


def test_bas_dd_없는_표는_세운다(db):
    conn, path = db
    with pytest.raises(sector.SectorError, match="bas_dd"):
        sector.attach_industry(pd.DataFrame({"code": ["005930"]}), as_of="2026-09-03",
                               db_path=path)


# ==================================================
# 3. 지수명 대조표 — 실측으로 어긋난 셋만
# ==================================================
@pytest.mark.parametrize("업종, 지수", [
    ("은행", "금융"), ("증권", "금융"), ("보험", "금융"), ("기타금융", "금융"),
    ("전기·전자", "전기전자"), ("기타제조", "제조"),
    ("건설", "건설"), ("IT 서비스", "IT 서비스"), ("농업, 임업 및 어업", "농업, 임업 및 어업"),
])
def test_지수명_대조표(업종, 지수):
    assert sector.index_name_for(업종) == 지수
