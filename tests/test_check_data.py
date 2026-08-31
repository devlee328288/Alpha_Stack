"""품질 게이트 테스트 — **일부러 버그를 심어 실제로 잡히는지 본다.**

검사를 짜 놓고 "돌려 보니 0 건이더라" 로 끝내면 아무것도 확인한 것이 아니다.
0 건은 자료가 깨끗해서일 수도 있고 **검사가 아무것도 안 보고 있어서**일 수도 있는데,
둘은 출력이 똑같다. 그래서 여기서는 깨끗한 DB 를 만들어 통과를 확인한 뒤,
error 검사마다 그 검사가 잡아야 할 결함을 하나씩 심어 **빨간불이 켜지는지** 본다.

음성 시험도 같이 둔다 — 거래정지 표시행(283,468건)이 OHLC 위반으로 새면 진짜 위반
0 건이 그 속에 묻힌다. 그래서 "이건 잡히면 안 된다" 도 못 박는다.
"""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List

import pytest

# `scripts/` 는 패키지가 아니라 경로로 읽어 온다.
# ⚠️ `exec_module` **전에** `sys.modules` 에 넣어야 한다. `@dataclass` 가 클래스의
#    `__module__` 로 `sys.modules` 를 되짚는데, 등록 전이면 거기서 None 이 나와
#    `AttributeError` 로 죽는다 (실측). 파이썬 문서가 권하는 순서이기도 하다.
_SPEC = importlib.util.spec_from_file_location(
    "check_data", Path(__file__).resolve().parents[1] / "scripts" / "check_data.py")
check_data = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = check_data
_SPEC.loader.exec_module(check_data)

SCHEMA = """
CREATE TABLE daily_price (
  bas_dd TEXT NOT NULL, code TEXT NOT NULL, name TEXT, market TEXT, sector TEXT,
  open INTEGER, high INTEGER, low INTEGER, close INTEGER, change INTEGER,
  change_rate REAL, volume INTEGER, value INTEGER, market_cap INTEGER,
  listed_shares INTEGER, PRIMARY KEY (bas_dd, code));
CREATE TABLE index_price (
  bas_dd TEXT NOT NULL, index_name TEXT NOT NULL, index_class TEXT,
  open REAL, high REAL, low REAL, close REAL, change REAL, change_rate REAL,
  volume INTEGER, value INTEGER, market_cap INTEGER,
  PRIMARY KEY (bas_dd, index_name));
CREATE TABLE fetch_log (bas_dd TEXT PRIMARY KEY, rows INTEGER, fetched_at TEXT);
CREATE TABLE index_fetch_log (
  bas_dd TEXT NOT NULL, market TEXT NOT NULL, rows INTEGER, fetched_at TEXT,
  PRIMARY KEY (bas_dd, market));
"""

CODES = ("000001", "000002")


def _평일들(n: int) -> List[str]:
    out, day = [], date(2024, 1, 2)
    while len(out) < n:
        if day.weekday() < 5:
            out.append(day.strftime("%Y%m%d"))
        day += timedelta(days=1)
    return out


# ⚠️ `SUSPENSION_GAP_DAYS`(20)보다 길어야 정리매매 판정이 실제로 동작한다.
DAYS = _평일들(30)


@pytest.fixture()
def 깨끗한DB(tmp_path) -> Path:
    """결함이 하나도 없는 작은 DB. 여기서 게이트가 통과해야 시험대가 성립한다."""
    db = tmp_path / "clean.db"
    con = sqlite3.connect(db)
    con.executescript(SCHEMA)
    for day in DAYS:
        con.execute("INSERT INTO fetch_log VALUES (?,?,?)", (day, 2, "2024-01-01"))
        con.execute("INSERT INTO index_fetch_log VALUES (?,?,?,?)",
                    (day, "KOSPI", 1, "2024-01-01"))
        for code in CODES:
            con.execute(
                "INSERT INTO daily_price (bas_dd, code, name, market, open, high, low, "
                "close, change, change_rate, volume, listed_shares) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (day, code, f"종목{code}", "KOSPI", 1000, 1010, 990, 1000, 0, 0.0,
                 5000, 100))
        con.execute(
            "INSERT INTO index_price (bas_dd, index_name, index_class, open, high, "
            "low, close, change, change_rate) VALUES (?,?,?,?,?,?,?,?,?)",
            (day, "코스피 200", "KOSPI", 300.0, 301.0, 299.0, 300.0, 0.0, 0.0))
    con.commit()
    con.close()
    return db


def _검사(db: Path) -> Dict[str, check_data.Check]:
    """시세 검사를 돌려 `{이름: 결과}` 로 돌려준다."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return {c.name: c for c in check_data.check_stock(con)}
    finally:
        con.close()


def _지수검사(db: Path) -> Dict[str, check_data.Check]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return {c.name: c for c in check_data.check_index(con)}
    finally:
        con.close()


def _고친다(db: Path, sql: str, args=()) -> None:
    con = sqlite3.connect(db)
    con.execute(sql, args)
    con.commit()
    con.close()


# ── 시험대가 성립하는가 ─────────────────────────────────────────────────────

def test_깨끗한_자료는_게이트를_통과한다(깨끗한DB):
    """통과가 기본값이어야 한다. 첫 실행부터 빨간불이면 팀은 게이트를 끈다."""
    checks = _검사(깨끗한DB)

    실패 = [c.name for c in checks.values() if c.failed]
    assert 실패 == []


def test_리포트에_게이트_판정이_들어간다(깨끗한DB):
    con = sqlite3.connect(f"file:{깨끗한DB}?mode=ro", uri=True)
    report = check_data.build_report({"stock": check_data.check_stock(con)})
    con.close()

    assert report["gate"]["status"] == "pass"
    assert report["gate"]["failed_checks"] == []
    assert "unexplained_outliers" in report["sections"]["stock"]


# ── error 검사마다 결함을 심는다 ────────────────────────────────────────────

def test_기준일자가_망가지면_잡는다(깨끗한DB):
    """`known_at` 은 기준일자에서 만든다. 그 값이 날짜가 아니면 시점정합이 무너진다."""
    _고친다(깨끗한DB,
            "INSERT INTO daily_price (bas_dd, code, name, market, open, high, low, "
            "close, change, change_rate, volume, listed_shares) "
            "VALUES ('2024-01-05','000003','망가진행','KOSPI',10,10,10,10,0,0.0,1,1)")

    checks = _검사(깨끗한DB)

    assert checks["malformed_rows"].count == 1
    assert checks["malformed_rows"].failed


def test_OHLC_대소가_뒤집히면_잡는다(깨끗한DB):
    """저가 > 고가 여도 에러는 안 나고 변동성 피처만 조용히 이상해진다."""
    _고친다(깨끗한DB, "UPDATE daily_price SET high = 900 WHERE bas_dd = ? AND code = ?",
            (DAYS[5], CODES[0]))

    checks = _검사(깨끗한DB)

    assert checks["ohlc_inversions"].count == 1
    assert checks["ohlc_inversions"].failed


def test_거래정지_표시행은_OHLC_위반이_아니다(깨끗한DB):
    """🔴 음성 시험. 실제 자료에 283,468건이 있어서, 이게 새면 진짜 위반이 묻힌다."""
    _고친다(깨끗한DB,
            "UPDATE daily_price SET open=0, high=0, low=0, volume=0 "
            "WHERE bas_dd = ? AND code = ?", (DAYS[5], CODES[0]))

    checks = _검사(깨끗한DB)

    assert checks["ohlc_inversions"].count == 0
    assert checks["zero_ohlc_rows"].count == 1
    assert not checks["ohlc_inversions"].failed
    # 거래량이 0 이면 정상적인 정지행이다 — 새 경고에 걸리면 안 된다
    assert checks["halted_but_traded"].count == 0


def test_정지_표시행인데_체결이_있으면_따로_센다(깨끗한DB):
    """검사의 구멍이었다. `if halted: … elif` 라서 정지행은 대소 검사를 건너뛰는데,

    이 행들은 고가가 0 이면서 종가가 양수라 검사를 받으면 전부 걸린다.
    실제 자료에 125행(2010-02-12~2025-03-21 · 97종목)이 이 상태로 숨어 있었다.
    """
    _고친다(깨끗한DB,
            "UPDATE daily_price SET open=0, high=0, low=0, volume=1015, close=1126 "
            "WHERE bas_dd = ? AND code = ?", (DAYS[5], CODES[0]))

    checks = _검사(깨끗한DB)

    assert checks["halted_but_traded"].count == 1
    # 자료 오류가 아니다 — 체결이 실재하므로 게이트를 세우지 않는다
    assert not checks["halted_but_traded"].failed
    # 정지행으로도 여전히 세지만, "정지 중이라 체결이 없다" 는 전제와는 갈라 둔다
    assert checks["zero_ohlc_rows"].count == 1
    assert checks["ohlc_inversions"].count == 0


def test_구멍_행의_표본에_체결이_남는다(깨끗한DB):
    """숫자만 있으면 왜 그런지 아무도 못 쫓아간다. 종가와 거래량을 함께 남긴다."""
    _고친다(깨끗한DB,
            "UPDATE daily_price SET open=0, high=0, low=0, volume=1015, close=1126 "
            "WHERE bas_dd = ? AND code = ?", (DAYS[5], CODES[0]))

    표본 = _검사(깨끗한DB)["halted_but_traded"].samples

    assert 표본[0]["종가"] == 1126
    assert 표본[0]["거래량"] == 1015
    assert 표본[0]["row"] == f"{DAYS[5]}/{CODES[0]}"


def test_등락률이_전일대비와_어긋나면_잡는다(깨끗한DB):
    """등락률 = 전일대비 / 기준가. 이 항등식이 깨지면 수익률 전부가 틀린다.

    실측으로는 920만 행 전부가 0.02%p 안에 들어온다.
    """
    _고친다(깨끗한DB, "UPDATE daily_price SET change_rate = 5.0 "
            "WHERE bas_dd = ? AND code = ?", (DAYS[5], CODES[0]))

    checks = _검사(깨끗한DB)

    assert checks["change_rate_mismatch"].count == 1
    assert checks["change_rate_mismatch"].failed


def test_받은_기록조차_없는_평일을_전부_열거한다(깨끗한DB):
    """휴장과 수집 누락은 다르다. 받아 봤다는 기록이 있으면 누락이 아니다.

    D-10 수용 기준이 *"누락이 0건이거나 그 날짜가 전부 열거된다"* 이므로
    표본이 아니라 **전부** 실어야 한다.
    """
    빠진날 = DAYS[7]
    _고친다(깨끗한DB, "DELETE FROM daily_price WHERE bas_dd = ?", (빠진날,))
    _고친다(깨끗한DB, "DELETE FROM fetch_log WHERE bas_dd = ?", (빠진날,))

    checks = _검사(깨끗한DB)

    assert checks["missing_trading_days"].count == 1
    assert checks["missing_trading_days"].samples == [빠진날]
    assert checks["missing_trading_days"].exhaustive


def test_받아_봤더니_0건인_날은_누락이_아니다(깨끗한DB):
    """연휴가 전부 누락으로 잡히면 진짜 누락이 그 속에 묻힌다.

    2017-09-29 → 10-10 은 추석 10일 연휴라 11일이 비는 것이 정상이다.
    """
    쉰날 = DAYS[7]
    _고친다(깨끗한DB, "DELETE FROM daily_price WHERE bas_dd = ?", (쉰날,))
    _고친다(깨끗한DB, "UPDATE fetch_log SET rows = 0 WHERE bas_dd = ?", (쉰날,))

    checks = _검사(깨끗한DB)

    assert checks["missing_trading_days"].count == 0


def test_설명되지_않는_이상치를_잡는다(깨끗한DB):
    """게이트의 핵심. 가격제한폭 밖인데 네 플래그 어디에도 안 걸리는 행이다."""
    # 기준가 1000 · 종가 1900 → 등락률 +90%. 정지도 소멸도 자본변동도 첫날도 아니다.
    _고친다(깨끗한DB,
            "UPDATE daily_price SET close=1900, change=900, change_rate=90.0, "
            "high=1900 WHERE bas_dd = ? AND code = ?", (DAYS[5], CODES[0]))

    checks = _검사(깨끗한DB)

    assert checks["unexplained_outliers"].count == 1
    assert checks["unexplained_outliers"].failed
    assert checks["explained_outliers"].count == 0


def test_정리매매로_설명되는_이상치는_게이트를_세우지_않는다(깨끗한DB):
    """🔴 음성 시험. 정리매매 구간은 가격제한폭이 적용되지 않는다 — 정상이다.

    실측 2,080행이 여기 걸린다. 이걸 실패로 처리하면 게이트가 첫 실행부터
    빨간불이고, 그러면 팀은 게이트를 고치지 않고 끈다.
    """
    # 마지막 5일을 지워 소멸 종목으로 만들고, 남은 마지막 날에 극단을 심는다.
    for day in DAYS[-5:]:
        _고친다(깨끗한DB, "DELETE FROM daily_price WHERE bas_dd = ? AND code = ?",
                (day, CODES[0]))
    _고친다(깨끗한DB,
            "UPDATE daily_price SET close=1900, change=900, change_rate=90.0, "
            "high=1900 WHERE bas_dd = ? AND code = ?", (DAYS[-6], CODES[0]))

    checks = _검사(깨끗한DB)

    assert checks["unexplained_outliers"].count == 0
    assert checks["explained_outliers"].count == 1
    assert checks["liquidation_rows"].count > 0


def test_2015년_이전에는_더_낮은_문턱을_쓴다(깨끗한DB):
    """가격제한폭이 ±15% 이던 구간에 30.5% 를 쓰면 사각지대가 된다.

    날짜별로 가르면 극단이 1,576 → 2,080행으로 늘고, 늘어난 504행도 전부
    플래그로 설명된다(실측). 즉 엄격하게 더 나은 문턱이다.
    """
    # 같은 +20% 를 2014년 날짜에 심으면 잡히고, 2024년이면 안 잡힌다.
    _고친다(깨끗한DB,
            "UPDATE daily_price SET close=1200, change=200, change_rate=20.0, "
            "high=1200 WHERE bas_dd = ? AND code = ?", (DAYS[5], CODES[0]))
    assert _검사(깨끗한DB)["unexplained_outliers"].count == 0

    _고친다(깨끗한DB, "UPDATE daily_price SET bas_dd = '20140310' "
            "WHERE bas_dd = ? AND code = ?", (DAYS[5], CODES[0]))
    _고친다(깨끗한DB, "INSERT INTO fetch_log VALUES ('20140310', 1, 'x')")

    assert _검사(깨끗한DB)["unexplained_outliers"].count == 1


def test_기준가_조정을_센다(깨끗한DB):
    """KRX 가 기준가를 조정한 날은 **등락률로는 안 보인다.**

    유한양행 2020-12-29 는 전일종가 76,600 → 기준가 73,300 인데 등락률은 +0.82% 다.
    우리 라벨은 시가 비율이라 그 3,300원이 그대로 수익률로 섞인다.
    """
    # 기준가를 900 으로 낮춘다(전일종가는 1000). 등락률은 자기 기준가와 맞춰 둔다.
    _고친다(깨끗한DB,
            "UPDATE daily_price SET close=990, change=90, change_rate=10.0, "
            "high=1000 WHERE bas_dd = ? AND code = ?", (DAYS[5], CODES[0]))
    # 다음 날의 기준가를 990 으로 이어 붙인다. 안 하면 값이 1000 으로 되돌아가면서
    # **그 날도 기준가 조정**이 되어 2건이 된다 — 조정은 가격 수준을 영구히 옮긴다.
    _고친다(깨끗한DB,
            "UPDATE daily_price SET change=10, change_rate=1.01 "
            "WHERE bas_dd = ? AND code = ?", (DAYS[6], CODES[0]))

    checks = _검사(깨끗한DB)

    assert checks["basis_price_adjusted"].count == 1
    assert checks["change_rate_mismatch"].count == 0    # 등락률 자체는 멀쩡하다
    assert not checks["basis_price_adjusted"].failed    # 사실 기록이지 실패가 아니다


# ── 지수 ────────────────────────────────────────────────────────────────────

def test_지수_등락률이_종가_역산과_어긋나면_잡는다(깨끗한DB):
    _고친다(깨끗한DB, "UPDATE index_price SET change_rate = 3.0 WHERE bas_dd = ?",
            (DAYS[5],))

    checks = _지수검사(깨끗한DB)

    assert checks["change_rate_mismatch"].count == 1
    assert checks["change_rate_mismatch"].failed


def test_종가만_들어온_지수_행은_OHLC_위반이_아니다(깨끗한DB):
    """🔴 음성 시험. 코스피 200 섹터지수 8종에 2,488건이 있다 (산출 초기 구간)."""
    _고친다(깨끗한DB, "UPDATE index_price SET open=0, high=0, low=0 WHERE bas_dd = ?",
            (DAYS[5],))

    checks = _지수검사(깨끗한DB)

    assert checks["ohlc_inversions"].count == 0
    assert checks["zero_ohlc_rows"].count == 1


def test_종가가_없는_지수_행은_실패가_아니라_기록이다(깨끗한DB):
    """'코스피 (외국주포함)' 처럼 거래량만 오는 지수가 실재한다 (4,097행).

    이걸 error 로 두면 게이트가 첫 실행부터 영구히 빨간불이다.
    """
    _고친다(깨끗한DB, "UPDATE index_price SET close = NULL, change_rate = NULL "
            "WHERE bas_dd = ?", (DAYS[5],))

    checks = _지수검사(깨끗한DB)

    assert checks["close_null_rows"].count == 1
    assert not checks["close_null_rows"].failed
    assert [c.name for c in checks.values() if c.failed] == []
