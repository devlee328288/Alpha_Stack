"""업종 스냅샷 — **KOSDAQ 파일이 들어와도** 같은 길로 가는지 잰다.

가이드(v3.6 까지)는 "KOSDAQ·KONEX 는 이번엔 받지 않는다" 였다. v3.7 에서 KOSDAQ 18장을
**선택**으로 열면서, 받기 전에 코드가 KOSDAQ 을 어떻게 다루는지 실측해 둔다. 받아 놓고
격리되면 18번 클릭이 헛수고다.

세 층을 본다.

  ① 판별  `normalize_manual.infer_sector_date` — KOSDAQ 종가로 되짚어도 기준일·시장이 맞는가.
          같은 날 KOSPI 파일과 KOSDAQ 파일이 **서로 다른 시장**으로 갈리는가
  ② 반입  `sector_text` — 화면 CSV 에 시장구분 칸이 없어도 `KOSDAQ` 이 앞에 붙는가
  ③ 공급  `supply.sector.industry_as_of` — 두 시장이 같은 날짜로 들어오면 함께 나오고,
          `market=` 으로 가르면 하나만 나오는가. 🔴 **날짜가 어긋나면 최근 날짜 하나만
          나온다** — 그래서 가이드가 KOSDAQ 도 KOSPI 와 같은 18개 날짜로 받으라고 한다

망을 타지 않는다. DB 는 임시 경로.
"""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import trading_calendar  # noqa: E402
from ingest.inbox import store as inbox_store  # noqa: E402
from ingest.store import krx_store  # noqa: E402
from ingest.store.migrations import migrate_path  # noqa: E402
from supply import sector  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "normalize_manual_for_kosdaq", ROOT / "scripts" / "normalize_manual.py")
nm = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = nm
_SPEC.loader.exec_module(nm)

HEAD = ["종목코드", "종목명", "시장구분", "업종명", "종가", "대비", "등락률", "시가총액"]
KOSPI_CODES = ["005930", "000660", "005380", "035420", "000270"]
#: 에코프로비엠 · 에코프로 · HLB · 알테오젠 · 에스엠
KOSDAQ_CODES = ["247540", "086520", "028300", "196170", "041510"]
DAY = "20240102"


# ── ① 판별 ────────────────────────────────────────────────────────────────

@pytest.fixture
def price_db(tmp_path) -> Path:
    """`daily_price` 만 있는 임시 DB — 같은 날 KOSPI 5종 · KOSDAQ 5종. 종가는 겹치지 않게."""
    path = tmp_path / "p.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE daily_price (bas_dd TEXT, code TEXT, market TEXT, close INTEGER)")
    rows = []
    for i, code in enumerate(KOSPI_CODES):
        rows.append((DAY, code, "KOSPI", 70_000 + i))
        rows.append(("20240103", code, "KOSPI", 71_000 + i))
    for i, code in enumerate(KOSDAQ_CODES):
        rows.append((DAY, code, "KOSDAQ", 280_000 + i))
        rows.append(("20240103", code, "KOSDAQ", 281_000 + i))
    conn.executemany("INSERT INTO daily_price VALUES (?,?,?,?)", rows)
    conn.commit()
    conn.close()
    return path


def _rows(codes: List[str], closes: List[int], market: str, with_market: bool) -> List[List[str]]:
    head = HEAD if with_market else [h for h in HEAD if h != "시장구분"]
    out = [head]
    for code, close in zip(codes, closes, strict=True):
        row = [code, "이름", market, "기타서비스", f"{close:,}", "0", "0.00", "1000000"]
        if not with_market:
            row.pop(2)
        out.append(row)
    return out


def test_KOSDAQ_종가로_되짚어도_기준일과_시장이_맞는다(price_db):
    guess = nm.infer_sector_date(
        _rows(KOSDAQ_CODES, [280_000 + i for i in range(5)], "KOSDAQ", True), db_path=price_db)
    assert guess.ok
    assert (guess.bas_dd, guess.market) == (DAY, "KOSDAQ")
    assert (guess.matched, guess.tried) == (5, 5)


def test_같은_날_KOSPI_파일과_KOSDAQ_파일은_시장으로_갈린다(price_db):
    kospi = nm.infer_sector_date(
        _rows(KOSPI_CODES, [70_000 + i for i in range(5)], "KOSPI", True), db_path=price_db)
    kosdaq = nm.infer_sector_date(
        _rows(KOSDAQ_CODES, [280_000 + i for i in range(5)], "KOSDAQ", True), db_path=price_db)
    assert kospi.bas_dd == kosdaq.bas_dd == DAY
    assert (kospi.market, kosdaq.market) == ("KOSPI", "KOSDAQ")


def test_시장구분_칸이_없는_KOSDAQ_파일에는_KOSDAQ_이_앞에_붙는다(price_db):
    """화면 CSV 에 시장구분이 없을 때 — 되짚은 시장이 변환본 둘째 칸으로 들어간다."""
    rows = _rows(KOSDAQ_CODES, [281_000 + i for i in range(5)], "KOSDAQ", False)
    guess = nm.infer_sector_date(rows, db_path=price_db)
    assert guess.ok and guess.market == "KOSDAQ" and guess.bas_dd == "20240103"
    text = "\n".join(",".join(r) for r in rows) + "\n"
    out = nm.sector_text(text, guess)
    head, first = out.splitlines()[0].split(","), out.splitlines()[1].split(",")
    assert head[:2] == [nm.SECTOR_DATE_COL, nm.SECTOR_MARKET_COL]
    assert first[:2] == ["20240103", "KOSDAQ"]


def test_변환본_이름은_시장을_품는다(price_db):
    """`업종분류현황_KOSDAQ_20240102` — KOSPI 파일과 같은 날이라도 이름이 겹치지 않는다."""
    guess = nm.infer_sector_date(
        _rows(KOSDAQ_CODES, [280_000 + i for i in range(5)], "KOSDAQ", True), db_path=price_db)
    stem = f"{nm.SECTOR_LABEL}_{guess.market}_{guess.bas_dd}"
    assert stem == "업종분류현황_KOSDAQ_20240102"


# ── ③ 공급 ────────────────────────────────────────────────────────────────

@pytest.fixture
def db(tmp_path, monkeypatch):
    """`test_supply_sector.py` 와 같은 임시 DB. 달력 캐시는 모듈 전역이라 비워서 시작한다."""
    경로 = tmp_path / "s.db"
    monkeypatch.setattr(krx_store, "DB_PATH", 경로)
    monkeypatch.setattr(trading_calendar, "_SESSION_CACHE", None)
    monkeypatch.setattr(trading_calendar, "_SESSION_SPAN", (None, None))
    krx_store.init_db()
    migrate_path(경로)
    conn = sqlite3.connect(경로)
    for d in ("20240102", "20240103", "20240104", "20240105", "20240108"):
        for m in ("ALL", "KOSPI", "KOSDAQ"):
            conn.execute(
                "INSERT OR REPLACE INTO trading_calendar (bas_dd, market, stock_count, "
                "built_at) VALUES (?,?,?,?)", (d, m, 900, "2026-09-06T00:00:00+09:00"))
    conn.commit()
    yield conn, 경로
    conn.close()


def 스냅샷(db_path: Path, tmp_path: Path, bas_dd: str, rows: dict, market: str):
    """반입 엔진이 통과시킨 것처럼 `inbox_accepted` 에 넣는다 (`test_supply_sector.py` 와 같다).

    `rows` 의 값은 업종명 하나(문자열)이거나 여럿(리스트)이다. 여럿을 받는 이유는 KRX 화면이
    실제로 **한 종목을 두 업종에 싣기 때문**이다 — 실측 2026-09-07 KOSDAQ 17쌍.
    """
    src = tmp_path / f"업종분류현황_{market}_{bas_dd}.csv"
    # 내용이 달라야 SHA-256 이 달라 batch 가 안 겹친다
    src.write_text(f"{market},{bas_dd},{len(rows)}", encoding="utf-8")
    항목 = [(code, 업종)
           for code, nm_ in rows.items()
           for 업종 in ([nm_] if isinstance(nm_, str) else nm_)]
    accepted = pd.DataFrame([
        {"row_no": i, "kind": sector.SECTOR_KIND, "key_hash": f"{bas_dd}{code}{업종}",
         "payload": {"bas_dd": bas_dd, "code": code, "name": code, "market": market,
                     "sector_nm": 업종, "close": 1000, "change": 0, "change_rate": 0.0,
                     "market_cap": 1_000_000},
         "extras": None, "changes": None, "warnings": None}
        for i, (code, 업종) in enumerate(항목)
    ])
    result = SimpleNamespace(kind=sector.SECTOR_KIND, accepted=accepted,
                             quarantined=pd.DataFrame(), report={"schema_version": "1.0"},
                             rows_total=len(accepted), rejected=None)
    inbox_store.load_result(result, src, db_path=db_path, origin="local",
                            contributor="시험 · 화면 다운로드")


def test_두_시장이_같은_날짜로_들어오면_함께_나오고_market_으로_가른다(db, tmp_path):
    _, 경로 = db
    스냅샷(경로, tmp_path, "20240102", {"005930": "전기·전자", "000660": "전기·전자"}, "KOSPI")
    스냅샷(경로, tmp_path, "20240102", {"247540": "일반전기전자", "196170": "제약"}, "KOSDAQ")

    snaps = sector.snapshots(db_path=경로)
    assert sorted(snaps["market"].unique()) == ["KOSDAQ", "KOSPI"]
    assert len(snaps) == 4

    전체 = sector.industry_as_of("20240105", as_of="2024-01-08", db_path=경로)
    assert sorted(전체["code"]) == ["000660", "005930", "196170", "247540"]

    코스닥만 = sector.industry_as_of("20240105", as_of="2024-01-08", market="KOSDAQ",
                                  db_path=경로)
    assert sorted(코스닥만["code"]) == ["196170", "247540"]
    assert set(코스닥만["industry"]) == {"일반전기전자", "제약"}


def test_날짜가_어긋나면_최근_스냅샷_하나만_나온다_그래서_같은_날짜로_받는다(db, tmp_path):
    """🔴 `industry_as_of` 는 '가장 최근 스냅샷 **하나**' 를 통째로 준다. KOSPI 를 01-02,
    KOSDAQ 을 01-03 에 받으면 01-05 기준 판정에서 KOSPI 종목이 통째로 빠진다. 결함이 아니라
    설계(스냅샷 = 그날의 표)이고, 그래서 가이드가 두 시장을 **같은 18개 날짜**로 받으라고 한다."""
    _, 경로 = db
    스냅샷(경로, tmp_path, "20240102", {"005930": "전기·전자"}, "KOSPI")
    스냅샷(경로, tmp_path, "20240103", {"247540": "일반전기전자"}, "KOSDAQ")

    전체 = sector.industry_as_of("20240105", as_of="2024-01-08", db_path=경로)
    assert list(전체["code"]) == ["247540"]                 # KOSPI 가 빠졌다
    # market 으로 가르면 각자 자기 최근 것을 준다
    코스피 = sector.industry_as_of("20240105", as_of="2024-01-08", market="KOSPI", db_path=경로)
    assert list(코스피["code"]) == ["005930"]


def test_attach_industry_는_종목별로_붙이므로_날짜가_어긋나도_둘_다_붙는다(db, tmp_path):
    """반출 경로(`attach_industry`)는 `merge_asof` 를 종목별로 하므로 위 함정이 없다."""
    _, 경로 = db
    스냅샷(경로, tmp_path, "20240102", {"005930": "전기·전자"}, "KOSPI")
    스냅샷(경로, tmp_path, "20240103", {"247540": "일반전기전자"}, "KOSDAQ")
    frame = pd.DataFrame({"bas_dd": ["20240105", "20240105"], "code": ["005930", "247540"]})
    out = sector.attach_industry(frame, as_of="2024-01-08", db_path=경로)
    assert list(out["industry"]) == ["전기·전자", "일반전기전자"]
    assert list(out["industry_bas_dd"]) == ["20240102", "20240103"]


# ── ④ 한 종목이 업종 둘에 실렸을 때 ──────────────────────────────────────────
#
# 🔴 KRX 업종분류 현황 화면은 한 종목을 두 업종에 싣는 때가 있다. 실측 2026-09-07 로
#    KOSDAQ 34행(17쌍) — 글로웍스·해성산업·솔본·신라섬유가 2010~2017 스냅샷에서 `부동산` 과
#    `일반서비스` 에 동시에 들어 있다. KOSPI 는 0건이라 KOSDAQ 을 들이기 전에는 안 보였다.
#
#    그냥 두면 `industry_as_of` 가 종목당 두 행을 내주고 `attach_industry` 의 `merge_asof` 는
#    둘 중 아무거나 집는다 — 조용히 틀리는 종류라 여기서 못 박는다.

def test_한_종목이_업종_둘에_실려도_종목마다_한_행만_나온다(db, tmp_path):
    _, 경로 = db
    스냅샷(경로, tmp_path, "20240102",
          {"034810": ["부동산", "일반서비스"], "005930": "전기·전자"}, "KOSDAQ")

    snaps = sector.snapshots(db_path=경로)
    assert len(snaps) == 3, "원본은 겹친 채로 들어와 있어야 한다 (반입은 격리하지 않는다)"

    out = sector.industry_as_of("20240105", as_of="2024-01-08", db_path=경로)
    assert len(out) == 2
    assert out["code"].is_unique


def test_겹치면_같은_스냅샷에서_종목이_많은_업종을_고른다(db, tmp_path):
    """규칙 1단계. `일반서비스` 에 종목이 셋, `부동산` 에 하나(겹친 그 종목뿐)면 `일반서비스`.

    실제 자료가 이 모양이다 — 2010~2017 KOSDAQ 에서 부동산은 1~3종, 일반서비스는 65~90종
    이었고, KRX 자신도 중복을 끝낸 2018-01-02 스냅샷에서 `일반서비스` 로 정리했다.
    """
    _, 경로 = db
    스냅샷(경로, tmp_path, "20240102", {
        "034810": ["부동산", "일반서비스"],   # 겹친 종목
        "111111": "일반서비스",
        "222222": "일반서비스",
    }, "KOSDAQ")

    out = sector.industry_as_of("20240105", as_of="2024-01-08", db_path=경로)
    골라진 = out.set_index("code")["industry"]
    assert 골라진["034810"] == "일반서비스"


def test_종목_수가_같으면_업종명_사전순으로_고른다(db, tmp_path):
    """규칙 2단계 — 마지막 빗장. 이게 없으면 동수일 때 입력 순서에 달린다."""
    _, 경로 = db
    # 두 업종 모두 그 종목 하나뿐이라 동수다
    스냅샷(경로, tmp_path, "20240102", {"034810": ["일반서비스", "부동산"]}, "KOSDAQ")

    out = sector.industry_as_of("20240105", as_of="2024-01-08", db_path=경로)
    assert out.set_index("code")["industry"]["034810"] == sorted(["일반서비스", "부동산"])[0]


def test_industry_ambiguous_는_겹쳤던_종목에만_참이다(db, tmp_path):
    """고른 결과만 주고 고른 사실을 감추면 나중에 되짚을 수 없다."""
    _, 경로 = db
    스냅샷(경로, tmp_path, "20240102",
          {"034810": ["부동산", "일반서비스"], "005930": "전기·전자"}, "KOSDAQ")

    out = sector.industry_as_of("20240105", as_of="2024-01-08", db_path=경로)
    표시 = out.set_index("code")["industry_ambiguous"]
    assert bool(표시["034810"]) is True
    assert bool(표시["005930"]) is False


def test_attach_industry_는_몇_번을_돌려도_같은_업종을_붙인다(db, tmp_path):
    """🔴 예전 결함. `merge_asof` 는 오른쪽에 (종목, 날짜) 가 겹치면 어느 행이 붙을지
    보장하지 않아, 같은 입력에도 붙는 업종이 달라질 수 있었다."""
    _, 경로 = db
    스냅샷(경로, tmp_path, "20240102",
          {"034810": ["부동산", "일반서비스"], "111111": "일반서비스"}, "KOSDAQ")
    frame = pd.DataFrame({"bas_dd": ["20240105"], "code": ["034810"]})

    결과 = [sector.attach_industry(frame, as_of="2024-01-08", db_path=경로)["industry"][0]
           for _ in range(5)]
    assert len(set(결과)) == 1, f"돌릴 때마다 달라진다: {결과}"
    assert 결과[0] == "일반서비스"


def test_attach_industry_는_겹쳐도_행이_늘지_않는다(db, tmp_path):
    """겹친 스냅샷을 그대로 조인하면 시세 표가 부푼다 — 행 수는 입력 그대로여야 한다."""
    _, 경로 = db
    # 실제 자료 모양: `일반서비스` 에 종목이 더 많아 그쪽이 뽑힌다
    스냅샷(경로, tmp_path, "20240102",
          {"034810": ["부동산", "일반서비스"], "111111": "일반서비스"}, "KOSDAQ")
    frame = pd.DataFrame({"bas_dd": ["20240103", "20240104", "20240105"],
                          "code": ["034810", "034810", "034810"]})

    out = sector.attach_industry(frame, as_of="2024-01-08", db_path=경로)
    assert len(out) == 3
    assert list(out["industry"]) == ["일반서비스"] * 3
