"""참조 자료 반출 — 홀드아웃이 나가지 않고, 파일이 MANIFEST 와 같다는 것을 잠근다.

2026-09-02 의 DART 반출본이 자기 상수(20210901)를 들고 있다가 정본(20240901)과 갈라져
197,068행이 조용히 빠졌다. 그래서 여기 시험은 **파일을 다시 열어** 재는 쪽을 본다.

DB 는 임시 파일이다. 표 이름·칸은 시험용으로 작게 만든다 — 진짜 표를 흉내 내지 않는다.
이 스크립트가 표를 읽는 방법은 `SELECT * … WHERE 시간칸 < ?` 하나라서 그 규약만 잠그면 된다.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def 반출():
    """`scripts/` 는 패키지가 아니라 파일 경로로 읽는다."""
    spec = importlib.util.spec_from_file_location(
        "export_reference_dataset", ROOT / "scripts" / "export_reference_dataset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def db(tmp_path):
    """시간 칸이 다른 표 둘. 수집 시각 칸(`fetched_at`)도 넣어 둔다."""
    경로 = tmp_path / "t.db"
    conn = sqlite3.connect(경로)
    conn.execute("CREATE TABLE a (bas_dd TEXT, v INTEGER, fetched_at TEXT)")
    conn.execute("CREATE TABLE b (known_at TEXT, w TEXT, collected_at TEXT)")
    conn.executemany("INSERT INTO a VALUES (?,?,?)", [
        ("20240830", 1, "x"), ("20240831", 2, "x"), ("20240901", 3, "x"), ("20250101", 4, "x"),
    ])
    conn.executemany("INSERT INTO b VALUES (?,?,?)", [
        ("20200102", "p", "y"), ("20240901", "q", "y"),
    ])
    conn.commit()
    conn.close()
    return 경로


SPECS = (
    ("alpha", "a_dev.parquet", "a", "bas_dd", "표 a", "시험"),
    ("beta", "b_dev.parquet", "b", "known_at", "표 b", "시험"),
)


def test_경계_앞만_나가고_수집시각_칸은_빠진다(반출, db, tmp_path):
    out = tmp_path / "out"
    항목 = 반출.export(db, out, holdout_start="20240901", specs=SPECS)

    a = pd.read_parquet(out / "alpha" / "a_dev.parquet")
    assert list(a["bas_dd"]) == ["20240830", "20240831"], "20240901 은 홀드아웃이다"
    assert "fetched_at" not in a.columns
    b = pd.read_parquet(out / "beta" / "b_dev.parquet")
    assert list(b["known_at"]) == ["20200102"]
    assert "collected_at" not in b.columns

    assert [e["path"] for e in 항목] == ["alpha/a_dev.parquet", "beta/b_dev.parquet"]
    assert 항목[0]["rows"] == 2 and 항목[0]["range"] == ["20240830", "20240831"]


def test_MANIFEST_의_해시가_파일과_같다(반출, db, tmp_path):
    out = tmp_path / "out"
    항목 = 반출.export(db, out, holdout_start="20240901", specs=SPECS)
    경로 = 반출.write_manifest(out, 항목, db_path=db, holdout_start="20240901")

    assert 경로.name == "MANIFEST_reference.json", "MANIFEST.json 이면 시세 이력을 덮는다"
    manifest = json.loads(경로.read_text("utf-8"))
    assert manifest["holdout_start"] == "20240901"
    assert manifest["holdout_start_authority"].startswith("evaluation/horizon.py")
    for e in manifest["files"]:
        assert e["sha256"] == 반출.sha256_of(out / e["path"])


def test_재검사는_파일을_다시_열어_경계_넘은_행을_잡는다(반출, db, tmp_path):
    out = tmp_path / "out"
    항목 = 반출.export(db, out, holdout_start="20240901", specs=SPECS)
    assert 반출.recheck_holdout(out, 항목, holdout_start="20240901") == 0

    # 만드는 쪽이 틀렸다고 치자 — 파일에 홀드아웃 행을 몰래 넣는다.
    pd.DataFrame({"bas_dd": ["20240830", "20250101"], "v": [1, 9]}).to_parquet(
        out / "alpha" / "a_dev.parquet", index=False)
    assert 반출.recheck_holdout(out, 항목, holdout_start="20240901") == 1


def test_경계는_정본_상수_하나에서_온다(반출):
    from evaluation.horizon import HOLDOUT_START
    assert 반출.HOLDOUT_START == HOLDOUT_START


def test_DB_는_읽기_전용으로_연다(반출, db):
    conn = 반출._connect_ro(db)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO a VALUES ('20240101', 0, 'z')")
    finally:
        conn.close()


def test_정본_반출_목록은_종류별_폴더_넷이다(반출):
    """HF 에 이미 올라간 경로와 같아야 팀원 코드가 안 깨진다."""
    경로들 = [f"{f}/{n}" for f, n, *_ in 반출.SPECS]
    assert 경로들 == [
        "identity/stock_identity_dev.parquet",
        "identity/corp_profile_dev.parquet",
        "financial/dart_financial_dev.parquet",
        "macro/macro_series_dev.parquet",
        "calendar/trading_calendar_dev.parquet",
    ]
    assert not any(p.startswith(("full/", "small/")) for p in 경로들), \
        "full/·small/ 은 건드리지 않는다"
