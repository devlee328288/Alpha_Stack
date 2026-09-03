"""적재와 보고가 **판정을 잃지 않고 옮기는지** 확인한다.

검사가 아무리 정확해도 담는 데서 새면 소용이 없다. 특히 두 가지를 잰다.

- **한 트랜잭션인가** — 묶음만 남고 행이 안 들어가면 다음 실행이 "이미 들였다" 며
  건너뛰기 때문에 그 상태가 조용히 굳는다
- **같은 파일을 두 번 들이지 않는가** — 판단이 이름이 아니라 **내용 지문**이어야 한다
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ingest.inbox import store  # noqa: E402
from ingest.inbox.engine import inspect_file  # noqa: E402
from ingest.inbox.report import render_markdown, write_report, write_summary  # noqa: E402
from ingest.store.migrations import LATEST_VERSION, migrate_path  # noqa: E402

종목_두_줄 = """날짜,종목코드,종목명,시장,시가,고가,저가,종가,거래량,메모
2021-01-04,5930,삼성전자,코스피,81000,84400,80200,83000,38655276,첫날
2021-01-05,005930,삼성전자,NASDAQ,81600,83900,81600,83900,35335669,
"""


@pytest.fixture
def db(tmp_path):
    """빈 DB 를 v5 까지 올려 돌려준다."""
    path = tmp_path / "probe.db"
    migrate_path(path)
    return path


@pytest.fixture
def 결과(tmp_path):
    path = tmp_path / "종목.csv"
    path.write_text(종목_두_줄, encoding="utf-8")
    return inspect_file(path, kind="ohlcv_stock"), path


# ==================================================
# 1. 마이그레이션
# ==================================================
def test_v5_가_반입_표_셋을_만든다(db):
    conn = sqlite3.connect(db)
    names = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'inbox%'")}
    assert names == {"inbox_batch", "inbox_accepted", "inbox_quarantine"}
    assert conn.execute("PRAGMA user_version").fetchone()[0] == LATEST_VERSION


def test_두_번_돌려도_같다(db):
    assert migrate_path(db) == 0, "이미 최신이면 아무것도 하지 않는다"


def test_시세_표는_건드리지_않는다(db):
    """🔴 920만 행은 우리가 16년치를 받아 쌓은 것이다. 반입이 손댈 자리가 아니다."""
    conn = sqlite3.connect(db)
    statements = [row[0] for row in conn.execute(
        "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL")]
    assert not any("daily_price" in (sql or "") and "inbox" in (sql or "")
                   for sql in statements)


# ==================================================
# 2. 적재
# ==================================================
def test_합격과_격리가_제자리에_담긴다(db, 결과):
    result, path = 결과
    batch_id = store.load_result(result, path, db_path=db, contributor="오준영")

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    batch = conn.execute("SELECT * FROM inbox_batch WHERE batch_id = ?", (batch_id,)).fetchone()
    assert batch["rows_total"] == 2
    assert batch["rows_accepted"] == 1
    assert batch["rows_quarantined"] == 1
    assert batch["contributor"] == "오준영"
    assert batch["origin"] == "local"

    accepted = conn.execute("SELECT * FROM inbox_accepted").fetchall()
    assert len(accepted) == 1
    payload = json.loads(accepted[0]["payload"])
    assert payload["code"] == "005930", "정제된 값이 담긴다"


def test_규격_밖_칸은_extras_로_함께_담긴다(db, 결과):
    result, path = 결과
    store.load_result(result, path, db_path=db)
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT extras FROM inbox_accepted").fetchone()
    assert json.loads(row["extras"])["메모"] == "첫날"


def test_손댄_칸_이름이_행마다_남는다(db, 결과):
    result, path = 결과
    store.load_result(result, path, db_path=db)
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    changes = json.loads(conn.execute("SELECT changes FROM inbox_accepted").fetchone()["changes"])
    assert "code" in changes and "market" in changes


def test_격리된_행은_원본까지_담는다(db, 결과):
    """사람이 고쳐 다시 넣으려면 정제 전 값이 필요하다."""
    result, path = 결과
    store.load_result(result, path, db_path=db)
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT raw, violations FROM inbox_quarantine").fetchone()
    assert json.loads(row["raw"])["시장"] == "NASDAQ"
    assert any(v["rule"] == "market.enum" for v in json.loads(row["violations"]))


def test_파일째_거부된_것도_담는다(db, tmp_path):
    """무엇이 왜 되돌려졌는지가 팀원에게 줄 답이다 — 안 남기면 같은 파일이 또 온다."""
    path = tmp_path / "본문.csv"
    path.write_text("발행시각,제목,본문,링크,검색어\n"
                    "2021-01-04T08:00:00+09:00,기사,본문내용,https://n/1,반도체\n",
                    encoding="utf-8")
    result = inspect_file(path, kind="news")
    batch_id = store.load_result(result, path, db_path=db)

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT rejected FROM inbox_batch WHERE batch_id = ?",
                       (batch_id,)).fetchone()
    assert row["rejected"] is not None


# ==================================================
# 3. 두 번 들이지 않기
# ==================================================
def test_이름이_아니라_내용으로_같은_파일인지_본다(db, tmp_path, 결과):
    result, path = 결과
    digest = store.file_sha256(path)
    assert store.already_ingested(digest, db) is None

    store.load_result(result, path, db_path=db, sha256=digest)
    assert store.already_ingested(digest, db) is not None

    # 이름만 바꾼 같은 내용 — 다시 들이면 안 된다.
    copy = tmp_path / "다른이름.csv"
    copy.write_text(종목_두_줄, encoding="utf-8")
    assert store.file_sha256(copy) == digest


def test_내용이_바뀌면_새_파일로_본다(db, tmp_path, 결과):
    result, path = 결과
    store.load_result(result, path, db_path=db)
    path.write_text(종목_두_줄 + "2021-01-06,005930,삼성전자,KOSPI,1,2,0,1,1,\n",
                    encoding="utf-8")
    assert store.already_ingested(store.file_sha256(path), db) is None


def test_묶음_번호는_같은_파일을_다시_검사해도_겹치지_않는다():
    """규격을 고친 뒤 다시 검사하는 일은 정당하고, 그 판정도 남아야 한다."""
    first = store.make_batch_id("a" * 64, "2026-09-01T10:00:00+09:00")
    second = store.make_batch_id("a" * 64, "2026-09-01T11:00:00+09:00")
    assert first != second


# ==================================================
# 4. 되읽기
# ==================================================
def test_담은_것을_표로_되읽는다(db, 결과):
    result, path = 결과
    store.load_result(result, path, db_path=db)
    frame = store.accepted_frame("ohlcv_stock", db_path=db)
    assert len(frame) == 1
    assert frame["code"].iloc[0] == "005930"
    assert "_batch_id" in frame.columns


def test_빈_결과에도_칸이_남는다(db):
    """🔴 칸 없는 빈 표를 주면 받는 쪽 `df["close"]` 가 KeyError 로 터진다 (#45).

    `supply/` 계층은 빈 결과에도 칸이 남는 것을 계약으로 못 박아 두었는데
    (`supply/market.py` 의 `to_frame`) 반입 쪽만 지키지 않고 있었다.
    `inbox_accepted` 가 0행인 동안에는 **100% 이 분기를 탄다.**
    """
    frame = store.accepted_frame("ohlcv_stock", db_path=db)
    assert len(frame) == 0
    assert "close" in frame.columns, "빈 표에도 규격 칸이 있어야 한다"
    assert frame["close"].empty, "칸을 봐도 예외가 나지 않아야 한다"
    for name in store.ORIGIN_COLUMNS:
        assert name in frame.columns


def test_되읽은_표는_어느_파일에서_왔는지_말한다(db, 결과):
    """dbt 의 `_dbt_source_relation` · Data Vault 의 `record_source` 자리다 (#45).

    우리 시세와 맞대다 이상한 값을 만났을 때 원본까지 되짚을 수 있어야 한다.
    """
    result, path = 결과
    store.load_result(result, path, origin="huggingface", contributor="이동원", db_path=db)
    frame = store.accepted_frame("ohlcv_stock", db_path=db)
    row = frame.iloc[0]
    assert row["_origin"] == "huggingface"  # origin 은 local·huggingface 만 허용된다
    assert row["_contributor"] == "이동원"
    assert row["_loaded_at"], "언제 들어왔는지가 행에 붙어 있어야 한다"
    assert len(row["_src_sha256"]) == 64, "원본 파일 지문"


def test_파일이_안_담아_온_규격_칸도_자리를_지킨다(db, 결과):
    """어떤 파일을 들였든 받는 쪽이 보는 표의 모양이 같아야 한다.

    칸을 행에서 뽑으면 파일마다 담아 온 칸이 달라 표가 그때그때 바뀐다.
    """
    result, path = 결과
    store.load_result(result, path, db_path=db)
    frame = store.accepted_frame("ohlcv_stock", db_path=db)
    assert "market_cap" in frame.columns, "파일에 없던 규격 칸도 자리는 있다"
    assert frame["market_cap"].isna().all()


def test_격리_사유를_규칙별로_센다(db, 결과):
    result, path = 결과
    store.load_result(result, path, db_path=db)
    summary = store.quarantine_summary(db)
    assert any(item["rule"] == "market.enum" for item in summary)


def test_반입_묶음_목록을_최근_순으로_준다(db, 결과):
    result, path = 결과
    store.load_result(result, path, db_path=db)
    batches = store.list_batches(db_path=db)
    assert len(batches) == 1
    assert batches[0]["kind"] == "ohlcv_stock"


# ==================================================
# 5. 보고서
# ==================================================
def test_보고서는_사람이_읽을_수_있게_나온다(결과):
    result, _ = 결과
    text = render_markdown(result, contributor="오준영")
    assert "# 반입 판정" in text
    assert "2행 중 1행이 들어오고" in text
    assert "우리가 손댄 것" in text
    assert "`5930` → `005930`" in text, "무엇을 무엇으로 바꿨는지 보여야 한다"


def test_보고서에_자료를_통째로_싣지_않는다(결과):
    """저장소가 PUBLIC 이다. 담는 것은 판정과 표본 20건까지다."""
    result, _ = 결과
    text = render_markdown(result)
    assert "38655276" not in text, "거래량 원값이 그대로 실리면 안 된다"


def test_파일째_거부는_고칠_방법까지_적는다(tmp_path):
    path = tmp_path / "본문.csv"
    path.write_text("발행시각,제목,본문,링크,검색어\n"
                    "2021-01-04T08:00:00+09:00,기사,본문,https://n/1,반도체\n", encoding="utf-8")
    text = render_markdown(inspect_file(path, kind="news"))
    assert "파일째 되돌렸습니다" in text
    assert "다시 보내" in text


def test_보고서는_JSON_과_마크다운을_함께_남긴다(결과, tmp_path):
    result, _ = 결과
    paths = write_report(result, batch_id="테스트", root=tmp_path / "reports")
    assert Path(paths["json"]).exists()
    assert Path(paths["markdown"]).exists()
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert payload["batch_id"] == "테스트"


def test_배치가_다르면_보고서가_안_덮인다(결과, tmp_path):
    """🔴 폴더는 날짜뿐이라 이름이 종류+파일명이면 덮인다 (#45).

    팀원 둘이 같은 날 `시세.csv` 를 올리거나, `--force` 로 다시 검사하면
    DB 에는 새 `batch_id` 로 남는데 파일에서는 앞엣것이 사라진다. 격리된 행을
    다시 보려고 보고서를 찾았을 때 다른 파일의 판정이 있으면 조사가 거기서 끊긴다.
    """
    result, _ = 결과
    root = tmp_path / "reports"
    첫째 = write_report(result, batch_id="배치A", root=root)
    둘째 = write_report(result, batch_id="배치B", root=root)

    assert 첫째["json"] != 둘째["json"], "같은 파일이라도 배치가 다르면 다른 이름"
    assert Path(첫째["json"]).exists(), "앞엣것이 남아 있어야 한다"
    assert json.loads(Path(첫째["json"]).read_text(encoding="utf-8"))["batch_id"] == "배치A"
    assert json.loads(Path(둘째["json"]).read_text(encoding="utf-8"))["batch_id"] == "배치B"


def test_배치_없는_예비검사는_이름에_배치를_안_붙인다(결과, tmp_path):
    """DB 에 남는 것이 없어 덮여도 잃을 것이 없다. 이름을 어지럽히지 않는다."""
    result, _ = 결과
    paths = write_report(result, root=tmp_path / "reports")
    assert "__" not in Path(paths["json"]).stem


def test_요약은_들인_파일이_없어도_난다(tmp_path):
    path = write_summary([], root=tmp_path / "reports")
    assert "들어온 파일이 없습니다" in Path(path).read_text(encoding="utf-8")
