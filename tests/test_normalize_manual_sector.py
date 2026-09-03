"""업종분류 현황 — 파일에 없는 기준일을 종가로 되짚는 부분의 시험대.

KRX 화면 CSV 에는 기준일 칸이 없고 파일명의 날짜는 **내려받은 날**이다. 그래서 종가
30종을 `daily_price` 와 맞춰 날짜를 알아낸다. 여기서 보는 것은 셋이다.

  ① 전부 맞는 날이 **하나**일 때만 정한다
  ② 둘 이상이거나 하나도 없으면 **정하지 않고 까닭을 남긴다** — 틀린 날짜가 붙는 것이
     빈 것보다 비싸다
  ③ 변환본에 기준일(과 없으면 시장) 칸이 **앞에** 붙는다
"""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path
from typing import List

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "normalize_manual_for_sector",
    Path(__file__).resolve().parents[1] / "scripts" / "normalize_manual.py")
nm = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = nm
_SPEC.loader.exec_module(nm)


HEAD = ["종목코드", "종목명", "시장구분", "업종명", "종가", "대비", "등락률", "시가총액"]
CODES = ["005930", "000660", "005380", "035420", "000270"]


@pytest.fixture
def db(tmp_path) -> Path:
    """`daily_price` 만 있는 임시 DB. 되짚기는 이 표 하나만 읽는다."""
    path = tmp_path / "p.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE daily_price (bas_dd TEXT, code TEXT, market TEXT, close INTEGER)")
    # 세 거래일. 날짜마다 종가가 다르되, 005930 만 20200102 와 20200103 에 같은 값이다
    # (거래정지처럼 종가가 이어지는 종목이 섞여도 나머지가 날을 가른다는 것을 본다).
    rows = []
    for i, code in enumerate(CODES):
        rows.append(("20200102", code, "KOSPI", 1000 + i))
        rows.append(("20200103", code, "KOSPI", (1000 if code == "005930" else 2000) + i))
        rows.append(("20200106", code, "KOSPI", 3000 + i))
    conn.executemany("INSERT INTO daily_price VALUES (?,?,?,?)", rows)
    conn.commit()
    conn.close()
    return path


def _rows(closes: List[int], with_market: bool = True) -> List[List[str]]:
    head = HEAD if with_market else [h for h in HEAD if h != "시장구분"]
    out = [head]
    for code, close in zip(CODES, closes, strict=True):
        row = [code, "이름", "KOSPI", "전기·전자", f"{close:,}", "0", "0.00", "1000000"]
        if not with_market:
            row.pop(2)
        out.append(row)
    return out


# ── ① 하나일 때만 정한다 ──────────────────────────────────────────────────

def test_종가가_전부_맞는_날이_하나면_그_날이다(db):
    guess = nm.infer_sector_date(_rows([1000, 1001, 1002, 1003, 1004]), db_path=db)
    assert guess.ok
    assert guess.bas_dd == "20200102"
    assert guess.market == "KOSPI"
    assert (guess.matched, guess.tried) == (5, 5)


def test_종가가_이어지는_종목이_섞여도_나머지가_날을_가른다(db):
    """005930 은 20200102·20200103 종가가 같다. 그래도 답은 하나다."""
    guess = nm.infer_sector_date(_rows([1000, 2001, 2002, 2003, 2004]), db_path=db)
    assert guess.ok and guess.bas_dd == "20200103"


def test_쉼표_있는_종가와_앞자리_0_이_지워진_코드도_맞춰_본다(db):
    rows = _rows([3000, 3001, 3002, 3003, 3004])
    rows[1][0] = "5930"               # 엑셀이 지운 앞자리 0
    guess = nm.infer_sector_date(rows, db_path=db)
    assert guess.ok and guess.bas_dd == "20200106"


# ── ② 정하지 못하면 정하지 않는다 ────────────────────────────────────────

def test_맞는_날이_없으면_정하지_않고_까닭을_남긴다(db):
    guess = nm.infer_sector_date(_rows([9, 9, 9, 9, 9]), db_path=db)
    assert not guess.ok
    assert guess.bas_dd is None and guess.market is None
    assert "맞는 종가가 없다" in guess.note


def test_일부만_맞으면_정하지_않는다(db):
    """4/5 맞는 날이 있어도 안 된다 — 한 종목이 틀리면 그 파일은 그 날 것이 아니다."""
    guess = nm.infer_sector_date(_rows([1000, 1001, 1002, 1003, 9]), db_path=db)
    assert not guess.ok
    assert guess.matched == 4 and guess.tried == 5
    assert "전부 맞는 날이 없다" in guess.note
    assert guess.candidates[0][0] == "20200102"


def test_둘_이상의_날이_전부_맞으면_표본이_모자란_것이다(db):
    """005930 하나만 맞춰 보면 두 날이 다 맞는다. 정하지 않는다."""
    guess = nm.infer_sector_date(_rows([1000, 1001, 1002, 1003, 1004]), db_path=db, probe=1)
    assert not guess.ok
    assert "2일이 전부 맞는다" in guess.note


def test_종목코드_종가_칸이_없으면_정하지_않는다(db):
    guess = nm.infer_sector_date([["업종명", "종목명"], ["건설", "이름"]], db_path=db)
    assert not guess.ok and "칸을 못 찾았다" in guess.note


# ── ③ 변환본의 모양 ────────────────────────────────────────────────────────

def test_업종명_머리글로_알아본다():
    assert nm.is_sector_file(HEAD)
    assert nm.is_sector_file(["﻿종목코드", "업종명"])       # BOM 이 붙어도
    assert not nm.is_sector_file(["일자", "금융투자", "보험"])


def test_변환본은_기준일_칸이_앞에_붙는다(db):
    rows = _rows([1000, 1001, 1002, 1003, 1004])
    text = "\n".join(",".join(r) for r in rows) + "\n"
    guess = nm.infer_sector_date(rows, db_path=db)
    out = nm.rows_of(nm.sector_text(text, guess))
    assert out[0] == ["기준일자", *HEAD]
    assert out[1][:2] == ["20200102", "005930"]
    assert len(out) == len(rows)


def test_시장구분이_없는_파일은_되짚은_시장도_붙는다(db):
    rows = _rows([1000, 1001, 1002, 1003, 1004], with_market=False)
    text = "\n".join(",".join(r) for r in rows) + "\n"
    guess = nm.infer_sector_date(rows, db_path=db)
    out = nm.rows_of(nm.sector_text(text, guess))
    assert out[0][:2] == ["기준일자", "시장구분"]
    assert out[1][:3] == ["20200102", "KOSPI", "005930"]
