"""`ingest/store/run_log.py` — 수집 실행 기록의 시험대.

이 표가 답해야 하는 질문은 하나다: **"지금 돌고 있나, 아니면 죽었나?"**

`collect_log` 만 보면 그 둘이 구별되지 않는다. 둘 다 "마지막 성공이 좀 됐다" 로
똑같이 보이기 때문이다. 그래서 여기서는 *시작할 때부터 남기는가* 를 집중해서 본다 —
끝나고 한꺼번에 쓰면 죽었을 때 아무 흔적도 없어 "애초에 안 돌았다" 와 같아진다.
"""

from __future__ import annotations

import sqlite3

import pytest

from ingest.store import migrations, run_log


@pytest.fixture()
def conn(tmp_path):
    """v8 까지 올린 빈 DB."""
    path = tmp_path / "t.db"
    migrations.migrate_path(path)
    c = sqlite3.connect(path)
    c.row_factory = sqlite3.Row
    yield c
    c.close()


# ==================================================
# 1. 시작할 때부터 남는가
# ==================================================
def test_시작하면_끝나기_전에도_보인다(conn):
    """이게 안 되면 돌다 죽었을 때 아무 흔적이 없다."""
    run_log.start_run("r1", args={"only": ["macro"]}, conn=conn)

    돌고있음 = run_log.current_run(conn=conn)
    assert 돌고있음 is not None
    assert 돌고있음["run_id"] == "r1"
    assert 돌고있음["status"] == "running"
    assert 돌고있음["finished_at"] is None, "안 끝났는데 끝난 시각이 있다"


def test_실행_인자를_그대로_남긴다(conn):
    """'이 숫자가 어떤 조건에서 나왔나' 를 나중에 답하려면 명령줄이 남아야 한다."""
    run_log.start_run("r1", args={"only": ["macro"], "days": 10}, conn=conn)
    받음 = run_log.latest_runs(conn=conn)[0]
    assert 받음["args"] == {"only": ["macro"], "days": 10}


def test_단계도_시작할_때_남는다(conn):
    run_log.start_run("r1", conn=conn)
    run_log.start_stage("r1", "macro", conn=conn)

    단계 = run_log.latest_runs(conn=conn)[0]["stages"]
    assert len(단계) == 1
    assert 단계[0]["stage"] == "macro"
    assert 단계[0]["status"] == "running"
    assert 단계[0]["finished_at"] is None


def test_끝나면_상태와_행수가_갱신된다(conn):
    run_log.start_run("r1", conn=conn)
    run_log.start_stage("r1", "macro", conn=conn)
    run_log.finish_stage("r1", "macro", "ok", rows=17851, note="ok 9", conn=conn)
    run_log.finish_run("r1", "ok", conn=conn)

    받음 = run_log.latest_runs(conn=conn)[0]
    assert 받음["status"] == "ok"
    assert 받음["finished_at"] is not None
    assert 받음["stages"][0]["rows"] == 17851
    assert 받음["stages"][0]["note"] == "ok 9"
    assert run_log.current_run(conn=conn) is None, "끝났는데 아직 돌고 있다고 나온다"


# ==================================================
# 2. 건너뛴 것과 실패한 것을 가르는가
# ==================================================
def test_건너뛴_단계도_남긴다(conn):
    """안 남기면 '안 돌았다' 와 '실패했다' 가 같아 보인다."""
    run_log.start_run("r1", conn=conn)
    run_log.finish_stage("r1", "price", "skipped", note="--only 로 제외", conn=conn)
    run_log.finish_run("r1", "ok", conn=conn)

    단계 = run_log.latest_runs(conn=conn)[0]["stages"]
    assert 단계[0]["status"] == "skipped"
    assert 단계[0]["status"] != "error"


def test_열지_않고_바로_닫아도_남는다(conn):
    """건너뛴 단계는 `start_stage` 없이 곧장 닫힌다."""
    run_log.start_run("r1", conn=conn)
    run_log.finish_stage("r1", "financial", "skipped", conn=conn)
    assert len(run_log.latest_runs(conn=conn)[0]["stages"]) == 1


def test_일부만_실패하면_partial(conn):
    run_log.start_run("r1", conn=conn)
    run_log.finish_stage("r1", "price", "ok", rows=100, conn=conn)
    run_log.finish_stage("r1", "macro", "error", note="ECOS 응답 없음", conn=conn)
    run_log.finish_run("r1", "partial", note="실패 단계: macro", conn=conn)

    받음 = run_log.latest_runs(conn=conn)[0]
    assert 받음["status"] == "partial"
    실패 = [s for s in 받음["stages"] if s["status"] == "error"]
    assert 실패[0]["note"] == "ECOS 응답 없음", "무엇이 실패했는지가 비어 있다"


def test_모르는_상태는_무엇을_해야_하는지_알려준다(conn):
    run_log.start_run("r1", conn=conn)
    with pytest.raises(ValueError) as 잡힘:
        run_log.finish_run("r1", "완료", conn=conn)
    assert "쓸 수 있는 값" in str(잡힘.value)
    assert "CHECK" in str(잡힘.value)

    with pytest.raises(ValueError):
        run_log.finish_stage("r1", "macro", "성공", conn=conn)


# ==================================================
# 3. 죽은 실행을 찾는가
# ==================================================
def test_오래_running_인_실행을_죽은_것으로_본다(conn):
    run_log.start_run("r1", conn=conn)
    # 시작 시각을 하루 전으로 되돌린다 — 프로세스가 죽어 finish_run 이 안 불린 상태
    conn.execute(
        "UPDATE ingest_run SET started_at = '2026-09-01T00:00:00+09:00' "
        "WHERE run_id = 'r1'"
    )
    죽은것 = run_log.stale_runs(older_than_hours=6, conn=conn)
    assert len(죽은것) == 1
    assert 죽은것[0]["run_id"] == "r1"


def test_방금_시작한_실행은_죽은_것이_아니다(conn):
    run_log.start_run("r1", conn=conn)
    assert run_log.stale_runs(older_than_hours=6, conn=conn) == []


def test_끝난_실행은_아무리_오래돼도_죽은_것이_아니다(conn):
    run_log.start_run("r1", conn=conn)
    run_log.finish_run("r1", "ok", conn=conn)
    conn.execute(
        "UPDATE ingest_run SET started_at = '2020-01-01T00:00:00+09:00' "
        "WHERE run_id = 'r1'"
    )
    assert run_log.stale_runs(older_than_hours=6, conn=conn) == []


# ==================================================
# 4. 여러 실행
# ==================================================
def test_최신순으로_준다(conn):
    for rid in ("20260901-100000", "20260902-100000", "20260902-140000"):
        run_log.start_run(rid, conn=conn)
        run_log.finish_run(rid, "ok", conn=conn)

    이름들 = [r["run_id"] for r in run_log.latest_runs(limit=3, conn=conn)]
    assert 이름들 == ["20260902-140000", "20260902-100000", "20260901-100000"]


def test_실행_열쇠가_사람이_읽을_수_있다():
    """대시보드와 로그를 눈으로 맞춰 보기 때문에 UUID 를 쓰지 않는다."""
    rid = run_log.new_run_id()
    assert len(rid) == 15 and rid[8] == "-"
    assert rid[:8].isdigit() and rid[9:].isdigit()
