"""거시 정문 — 기간이 아니라 발표된 날로 붙는가, 개정되는 계열은 기본에서 빠지는가.

**무엇을 지키려는 시험인가.** ECOS 는 월별 지표를 기준월 1일로 준다. 그 날짜에 붙이면
물가는 한 달, 경기지수는 두 달 미래를 보는데 **예외가 나지 않는다.**

🔴 누수 검사에는 음성 대조군을 둔다 — **기간 시작일로 붙인 틀린 구현**이 실제로 붉어지는지
   본다. 검사는 정문이 낸 칸을 믿지 않고, 시험이 심은 `known_at` 을 따로 대조한다.

`known_at` 은 이 파일이 **직접 정한 값**으로 심는다. 수집 쪽 규칙(`ecos_data.known_at`)을
거치면 그 규칙이 틀려도 시험이 같이 틀린다 — 규칙 자체의 시험은 `test_macro_store.py` 에 있다.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ingest.clients import ecos_data  # noqa: E402
from ingest.store import krx_store  # noqa: E402
from ingest.store.migrations import migrate_path  # noqa: E402
from supply import macro  # noqa: E402

#: 시험이 심는 원본 — (지표, 기간, 주기, 값, known_at)
심을행 = [
    ("cpi", "202311", "M", 111.0, "20231210"),
    ("cpi", "202312", "M", 112.0, "20240110"),
    ("cpi", "202401", "M", 113.0, "20240210"),
    ("ktb3y", "20240108", "D", 3.1, "20240109"),
    ("ktb3y", "20240109", "D", 3.2, "20240110"),
    ("ktb3y", "20240110", "D", 3.3, "20240111"),
    ("ppi", "202311", "M", 120.0, "20231231"),
    ("ppi", "202312", "M", None, "20240131"),         # ECOS 가 '-' 를 준 기간
    ("leading", "202311", "M", 99.0, "20240105"),
    ("leading", "202312", "M", 100.0, "20240205"),
]
알게된날 = {(i, p): k for i, p, _, _, k in 심을행}
날들 = ["20231201", "20231211", "20240102", "20240109", "20240110", "20240111",
      "20240131", "20240201", "20240205", "20240213"]


@pytest.fixture
def db(tmp_path, monkeypatch):
    경로 = tmp_path / "m.db"
    monkeypatch.setattr(krx_store, "DB_PATH", 경로)
    krx_store.init_db()
    migrate_path(경로)
    with sqlite3.connect(경로) as conn:
        conn.executemany(
            "INSERT INTO macro_series (indicator_id, period, cycle, value, known_at, stat_code, "
            "item_code, unit, collected_at) VALUES (?,?,?,?,?,?,?,?,?)",
            [(i, p, c, v, k, "STAT", "ITEM", "u", "2026-09-11T00:00:00+09:00")
             for i, p, c, v, k in 심을행])
    return 경로


def 표(days=날들) -> pd.DataFrame:
    return pd.DataFrame({"bas_dd": list(days)})


def 샌_행(붙인: pd.DataFrame, i: str) -> list:
    """행의 날짜보다 **늦게 발표된** 기간이 붙은 행. 발표일은 시험이 심은 원본에서 꺼낸다."""
    return [(d, p) for d, p in zip(붙인["bas_dd"], 붙인[f"macro_{i}_period"], strict=True)
            if isinstance(p, str) and 알게된날[(i, p)] > d]


def 기간_시작일로_붙인다(frame: pd.DataFrame, i: str) -> pd.DataFrame:
    """🔴 틀린 구현 — ECOS 가 준 기준월 1일을 그대로 시점으로 쓴다. 정문을 부르지 않는다."""
    rows = sorted((p if len(p) == 8 else p + "01", p)
                  for ii, p, _, v, _ in 심을행 if ii == i and v is not None)
    답 = []
    for d in frame["bas_dd"]:
        보인 = [p for 시작, p in rows if 시작 <= d]
        답.append(보인[-1] if 보인 else None)
    return frame.assign(**{f"macro_{i}_period": 답})


# ==================================================
# 1. as_of 를 빠뜨릴 수 없다
# ==================================================
def test_as_of_없이_부르면_터진다():
    with pytest.raises(TypeError):
        macro.macro_as_of("20240111")                           # type: ignore[call-arg]
    with pytest.raises(TypeError):
        macro.attach_macro(표())                                # type: ignore[call-arg]
    with pytest.raises(TypeError):
        macro.macro_history("cpi")                              # type: ignore[call-arg]


def test_as_of_보다_뒤의_행을_물으면_세운다(db):
    with pytest.raises(ValueError) as 잡힘:
        macro.macro_as_of("20240111", as_of="2024-01-11", db_path=db)
    assert "as_of" in str(잡힘.value)
    with pytest.raises(ValueError):
        macro.attach_macro(표(), as_of="2024-01-01", db_path=db)


# ==================================================
# 2. 시점 — 기간이 아니라 발표된 날
# ==================================================
def test_월별은_기간이_아니라_발표된_날로_붙는다(db):
    """12월 물가(202312)는 1월 10일에 나온다. 1월 9일 행에는 11월 물가가 붙어야 한다."""
    붙인 = macro.attach_macro(표(), as_of="2024-02-14", indicators=("cpi",), db_path=db)
    기대 = [None, "202311", "202311", "202311", "202312", "202312",
          "202312", "202312", "202312", "202401"]
    assert [p if isinstance(p, str) else None for p in 붙인["macro_cpi_period"]] == 기대
    assert 붙인.loc[붙인["bas_dd"] == "20240109", "macro_cpi"].iloc[0] == 111.0
    assert pd.isna(붙인.loc[붙인["bas_dd"] == "20231201", "macro_cpi"].iloc[0]), "결측은 결측"


def test_일별은_다음날부터_보인다(db):
    붙인 = macro.attach_macro(표(["20240102", "20240109", "20240110", "20240111"]),
                            as_of="2024-01-12", indicators=("ktb3y",), db_path=db)
    assert list(붙인["macro_ktb3y"].fillna(-1)) == [-1, 3.1, 3.2, 3.3]


def test_정문은_누수가_0이고_기간_시작일_대조군은_붉어진다(db):
    """정문과 틀린 구현이 **다른 것은 시점 하나**다 — 마지막 날에는 같은 기간을 붙인다."""
    옳음 = macro.attach_macro(표(), as_of="2024-02-14", indicators=("cpi", "ppi"), db_path=db)
    for i in ("cpi", "ppi"):
        틀림 = 기간_시작일로_붙인다(표(), i)
        샘 = 샌_행(틀림, i)
        마지막_같음 = 옳음[f"macro_{i}_period"].iloc[-1] == 틀림[f"macro_{i}_period"].iloc[-1]
        print(f"\n  {i}: 정문 누수 {len(샌_행(옳음, i))} · 대조군 누수 {len(샘)} {샘} · "
              f"마지막 날 같은 기간 {마지막_같음}")
        assert len(샌_행(옳음, i)) == 0
        assert len(샘) > 0, f"{i} 를 기간 시작일로 붙였는데 검사가 못 잡았다"
        assert 마지막_같음, "대조군이 시점 말고 다른 것까지 달라졌다"

    # cpi 는 2024-01-02 에 202401(2월 10일 발표)을 붙인다 — 한 달 넘게 미래다
    assert ("20240102", "202401") in 샌_행(기간_시작일로_붙인다(표(), "cpi"), "cpi")


def test_값이_빈_기간은_건너뛰고_기간_칸이_그것을_드러낸다(db):
    붙인 = macro.attach_macro(표(["20240201"]), as_of="2024-02-02", indicators=("ppi",),
                            db_path=db)
    assert 붙인["macro_ppi"].iloc[0] == 120.0
    assert 붙인["macro_ppi_period"].iloc[0] == "202311", "12월이 비어 11월 값이 이어진다"


# ==================================================
# 3. 개정되는 계열은 기본에서 빠진다
# ==================================================
def test_순환변동치는_기본에서_빠지고_이름을_적으면_나온다(db):
    기본 = macro.attach_macro(표(["20240213"]), as_of="2024-02-14", db_path=db)
    assert "macro_leading" not in 기본.columns
    assert "macro_coincident" not in 기본.columns

    적음 = macro.attach_macro(표(["20240213"]), as_of="2024-02-14", indicators=("leading",),
                            db_path=db)
    assert 적음["macro_leading"].iloc[0] == 100.0


def test_개정_계열_목록이_수집_큐레이션과_어긋나지_않는다():
    """수집 쪽 지표가 늘거나 이름이 바뀌면 여기서 먼저 걸린다."""
    수집 = tuple(spec["id"] for spec in ecos_data.INDICATORS)
    assert macro.ALL_INDICATORS == 수집
    assert macro.REVISED_INDICATORS <= set(수집)
    assert set(macro.DEFAULT_INDICATORS) | macro.REVISED_INDICATORS == set(수집)
    assert not set(macro.DEFAULT_INDICATORS) & macro.REVISED_INDICATORS


def test_없는_지표는_세운다(db):
    with pytest.raises(ValueError) as 잡힘:
        macro.attach_macro(표(), as_of="2024-02-14", indicators=("gdp",), db_path=db)
    assert "쓸 수 있는 것" in str(잡힘.value)


# ==================================================
# 4. 모양
# ==================================================
def test_점_조회는_지표당_한_행이고_attach_와_같다(db):
    점 = macro.macro_as_of("20240111", as_of="2024-01-12", db_path=db)
    assert list(점["indicator_id"]) == [i for i in macro.DEFAULT_INDICATORS
                                        if i in {"cpi", "ktb3y", "ppi"}]
    붙인 = macro.attach_macro(표(["20240111"]), as_of="2024-01-12", db_path=db)
    for _, r in 점.iterrows():
        i = r["indicator_id"]
        assert 붙인[f"macro_{i}"].iloc[0] == r["value"]
        assert 붙인[f"macro_{i}_period"].iloc[0] == r["period"]


def test_이력은_as_of_에서_자르고_기간_순이다(db):
    assert list(macro.macro_history("cpi", as_of="2024-01-11", db_path=db)["period"]) == \
        ["202311", "202312"]
    assert list(macro.macro_history("leading", as_of="2024-02-06", db_path=db)["period"]) == \
        ["202311", "202312"]


def test_빈_결과에도_칸이_남는다(db):
    빈 = macro.macro_as_of("20231201", as_of="2023-12-02", db_path=db)
    assert len(빈) == 0 and list(빈.columns) == list(macro.MACRO_COLUMNS)

    붙인 = macro.attach_macro(표(["20231201"]), as_of="2023-12-02", db_path=db)
    for col in macro.macro_columns_for(macro.DEFAULT_INDICATORS):
        assert col in 붙인.columns
    assert 붙인[list(macro.macro_columns_for(macro.DEFAULT_INDICATORS))].isna().all(axis=None)


def test_attach_는_입력_순서를_지킨다(db):
    섞은 = ["20240213", "20231211", "20240110", "20240102"]
    붙인 = macro.attach_macro(표(섞은), as_of="2024-02-14", indicators=("cpi",), db_path=db)
    assert list(붙인["bas_dd"]) == 섞은
    assert list(붙인["macro_cpi_period"]) == ["202401", "202311", "202312", "202311"]
