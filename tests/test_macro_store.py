"""`ingest/store/macro_store.py` — 거시를 받아 담는 계층의 시험대.

가장 중요한 것 셋을 본다.

  ① **행이 조용히 사라지지 않는가** — 기본키가 (지표, 기간) 이라 좁다. 좁은 키는
     서로 다른 값을 같은 칸에 밀어 넣고, **넣은 수와 담긴 수가 함께 줄어** 눈치채기
     어렵다. 재무에서 `account_detail` 을 빼 6.4% 를 잃은 적이 있다.
  ② **`known_at` 이 값과 함께 담기는가** — 이 칸이 비면 시점 정합이 통째로 무너지는데
     예외는 나지 않는다.
  ③ **빈 응답과 오류를 가르는가** — 순서를 뒤집으면 "ECOS 가 안 주는 지표" 가
     "받다 실패한 것" 으로 남아 영원히 다시 부른다.
"""

from __future__ import annotations

import sqlite3

import pytest

from ingest.clients import ecos_data
from ingest.store import macro_store, migrations


# ── 시험용 응답 ─────────────────────────────────────────────────────────────
# `ecos_data.fetch_series` 가 실제로 주는 모양을 그대로 흉내 낸다.
def _응답(지표: str, 기간값: list) -> dict:
    spec = ecos_data.INDICATOR_BY_ID[지표]
    return {
        "id": 지표,
        "label": spec["label"],
        "unit": spec["unit"],
        "cycle": spec["cycle"],
        "cycle_name": "월별" if spec["cycle"] == "M" else "일별",
        "periods": [p for p, _ in 기간값],
        "values": [v for _, v in 기간값],
        "dates": [],          # 화면용이라 저장 계층은 쓰지 않는다
        "truncated": False,
    }


@pytest.fixture()
def conn(tmp_path):
    """v7 까지 올린 빈 DB."""
    path = tmp_path / "t.db"
    migrations.migrate_path(path)
    c = sqlite3.connect(path)
    c.row_factory = sqlite3.Row
    yield c
    c.close()


# ==================================================
# 1. 정규화
# ==================================================
def test_기간과_값을_짝지어_담는다():
    rows = macro_store.rows_from(
        _응답("cpi", [("202605", 119.92), ("202606", 119.99), ("202607", 119.77)]),
        "2026-09-02 12:00:00 KST",
    )
    assert len(rows) == 3
    # (indicator_id, period, cycle, value, known_at, stat_code, item_code, unit, collected_at)
    assert rows[0][0] == "cpi"
    assert rows[0][1] == "202605"
    assert rows[0][2] == "M"
    assert rows[0][3] == 119.92


def test_known_at_을_값과_함께_담는다():
    """이 칸이 비면 시점 정합이 통째로 무너지는데 예외는 나지 않는다."""
    rows = macro_store.rows_from(_응답("cpi", [("202607", 119.77)]), "t")
    known_at = rows[0][4]
    assert known_at == "20260810", "cpi 는 익월 10일부터 아는 것으로 잡는다"
    # 기준월을 그대로 쓰지 않았는지 — 그게 이 프로젝트에서 가장 조용한 오류다
    assert known_at != "20260701"


def test_값이_비어도_담기지만_known_at_은_반드시_있다():
    """ECOS 가 '-' 를 주면 값은 None 이다. 그래도 시점은 있어야 한다."""
    rows = macro_store.rows_from(_응답("cpi", [("202607", None)]), "t")
    assert rows[0][3] is None
    assert rows[0][4] == "20260810"


def test_기간과_값의_개수가_다르면_세운다():
    """짝이 밀리면 값이 엉뚱한 시점에 붙는데 **행 수는 맞아서** 아무도 모른다."""
    응답 = _응답("cpi", [("202606", 119.99), ("202607", 119.77)])
    응답["values"] = [119.99]            # 하나를 잃어버린 상태
    with pytest.raises(ecos_data.EcosError) as 잡힘:
        macro_store.rows_from(응답, "t")
    assert "엉뚱한 시점" in str(잡힘.value)


def test_큐레이션에_없는_지표는_담지_않는다():
    with pytest.raises(ecos_data.EcosError):
        macro_store.rows_from({"id": "gdp", "periods": ["202607"], "values": [1.0]}, "t")


# ==================================================
# 2. 저장 — 조용히 사라지는 행이 없는가
# ==================================================
def test_담은_행이_그대로_들어간다(conn):
    담은수 = macro_store.save(
        _응답("cpi", [("202605", 119.92), ("202606", 119.99), ("202607", 119.77)]),
        conn=conn,
    )
    assert 담은수 == 3
    assert conn.execute("SELECT COUNT(*) FROM macro_series").fetchone()[0] == 3


def test_지표가_달라도_같은_기간이_서로를_덮지_않는다(conn):
    """기본키가 (지표, 기간) 이다. 지표를 빼먹으면 물가가 생산자물가를 덮는다."""
    macro_store.save(_응답("cpi", [("202607", 119.77)]), conn=conn)
    macro_store.save(_응답("ppi", [("202607", 129.39)]), conn=conn)

    담김 = {
        r["indicator_id"]: r["value"]
        for r in conn.execute("SELECT indicator_id, value FROM macro_series")
    }
    assert 담김 == {"cpi": 119.77, "ppi": 129.39}, "한쪽이 다른 쪽을 덮었다"


def test_다시_받으면_늘지_않고_갱신된다(conn):
    """거시는 과거 값이 개정된다. 다시 받아 덮되 행이 불어나면 안 된다."""
    macro_store.save(_응답("cpi", [("202607", 119.77)]), conn=conn)
    macro_store.save(_응답("cpi", [("202607", 119.80)]), conn=conn)   # 개정치

    rows = conn.execute("SELECT period, value FROM macro_series").fetchall()
    assert len(rows) == 1, "다시 받았더니 행이 불었다"
    assert rows[0]["value"] == 119.80, "개정치로 갱신되지 않았다"


def test_일별과_월별이_한_표에_섞여도_구분된다(conn):
    macro_store.save(_응답("cpi", [("202607", 119.77)]), conn=conn)
    macro_store.save(_응답("usdkrw", [("20260902", 1370.3)]), conn=conn)

    주기 = {
        r["indicator_id"]: r["cycle"]
        for r in conn.execute("SELECT indicator_id, cycle FROM macro_series")
    }
    assert 주기 == {"cpi": "M", "usdkrw": "D"}


# ==================================================
# 3. as_of — 시점으로 거르는가
# ==================================================
def test_아직_발표되지_않은_값은_주지_않는다(conn):
    """이 시험이 통과하지 않으면 거시를 쓰는 순간 미래가 학습에 샌다."""
    macro_store.save(
        _응답("cpi", [("202605", 119.92), ("202606", 119.99), ("202607", 119.77)]),
        conn=conn,
    )
    # 7월 중순 — 우리 규칙상 7월분(known_at 20260810)은 아직 모르고,
    # 6월분(known_at 20260710)까지만 알고 있다.
    받음 = macro_store.as_of("cpi", "20260715", conn=conn)
    assert 받음["period"] == "202606"

    # ECOS 가 주는 기준월 1일을 그대로 썼다면 이 날짜에 202607 이 잡혔을 것이다.
    assert 받음["period"] != "202607"


def test_그_날_발표된_값은_그_날부터_준다(conn):
    macro_store.save(_응답("cpi", [("202607", 119.77)]), conn=conn)
    assert macro_store.as_of("cpi", "20260809", conn=conn) is None
    assert macro_store.as_of("cpi", "20260810", conn=conn)["period"] == "202607"


def test_값이_없는_기간은_건너뛰고_그_앞을_준다(conn):
    """결측을 최신값으로 주면 계단식 보간이 끊긴다."""
    macro_store.save(_응답("cpi", [("202606", 119.99), ("202607", None)]), conn=conn)
    받음 = macro_store.as_of("cpi", "20260901", conn=conn)
    assert 받음["period"] == "202606"
    assert 받음["value"] == 119.99


def test_아는_것이_아무것도_없으면_None(conn):
    macro_store.save(_응답("cpi", [("202607", 119.77)]), conn=conn)
    assert macro_store.as_of("cpi", "20200101", conn=conn) is None


# ==================================================
# 4. 현황
# ==================================================
def test_현황이_지표별로_갈린다(conn):
    macro_store.save(_응답("cpi", [("202606", 1.0), ("202607", None)]), conn=conn)
    macro_store.save(_응답("usdkrw", [("20260902", 1370.3)]), conn=conn)

    현황 = {s["indicator_id"]: s for s in macro_store.status(conn=conn)}
    assert 현황["cpi"]["rows"] == 2
    assert 현황["cpi"]["null_values"] == 1
    assert 현황["usdkrw"]["cycle"] == "D"


def test_큐레이션_아홉종을_알려준다():
    이름들 = list(macro_store.all_indicators())
    assert len(이름들) == 9
    assert "cpi" in 이름들 and "usdkrw" in 이름들
