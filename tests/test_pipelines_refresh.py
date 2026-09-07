"""`pipelines/refresh.py` — 버튼 갱신의 시험대.

이 파이프라인이 답해야 하는 질문은 넷이다.

  1. **며칠치를 받아야 하나** — 며칠 안 눌러도 구멍이 안 나야 한다. 고정 창은 구멍을
     만들고, 그 구멍은 다음에 눌러도 안 메워진다.
  2. **두 번 눌러도 한 번만 도나** — 화면의 버튼은 두 번 눌린다.
  3. **멈춰야 할 때 멈추나** — 게이트가 막으면 반출이 나가면 안 된다. 그리고 게이트
     실패는 수집 실패와 **다른 색**이어야 한다. 다시 눌러 봐야 소용없는 쪽이라서다.
  4. **안 돌린 것과 실패한 것이 구별되나** — 수정주가를 껐을 때 `skipped` 로 남아야 한다.

바깥 스크립트(`check_data.py` 등)는 부르지 않는다. 여기서 보려는 것은 **잇는 방식**이지
그 스크립트들이 아니다. 그것들에는 각자의 시험이 있다.
"""

from __future__ import annotations

import sqlite3
from datetime import date

import pytest

from ingest.store import migrations, run_log
from pipelines import refresh


# ==================================================
# 1. 창 계산 — 며칠 안 눌러도 구멍이 없나
# ==================================================
def test_이틀_안_눌렀으면_그_이틀이_창에_들어온다():
    창, _ = refresh.창을_센다("20260901", 오늘=date(2026, 9, 3))
    # 9/1(화)·9/2(수)·9/3(목) = 평일 3일. 마지막 날을 포함해 다시 확인한다 —
    # 장중에 받았다면 그날 자료가 확정 전일 수 있다.
    assert 창 == 3


def test_열흘_넘게_안_눌러도_창이_그만큼_넓어진다():
    """🔴 고정 10일이면 여기서 구멍이 난다. 그리고 그 구멍은 다음에도 안 메워진다."""
    창, 설명 = refresh.창을_센다("20260801", 오늘=date(2026, 9, 5))
    assert 창 > 10
    assert "20260801" in 설명


def test_주말은_세지_않는다():
    # 2026-09-05 는 토요일. 금요일(9/4)까지만 센다.
    창_금, _ = refresh.창을_센다("20260904", 오늘=date(2026, 9, 4))
    창_토, _ = refresh.창을_센다("20260904", 오늘=date(2026, 9, 5))
    assert 창_금 == 창_토, "주말이 끼어도 셈이 늘면 안 된다"


def test_방금_받았어도_최소_창은_본다():
    """0 을 돌려주면 '받을 게 없다' 로 읽혀 그날 확정분을 영영 안 받는다."""
    창, _ = refresh.창을_센다("20260905", 오늘=date(2026, 9, 5))
    assert 창 == refresh.MIN_WINDOW


def test_DB_가_비어_있으면_창을_지어내지_않는다():
    창, 설명 = refresh.창을_센다(None, 오늘=date(2026, 9, 5))
    assert 창 == refresh.MIN_WINDOW
    assert "비어" in 설명


def test_DB_가_미래를_담고_있으면_그_사실을_남긴다():
    """음수 창을 만들지 않는다. 이상한 값은 조용히 넘기지 말고 적어 둔다."""
    창, 설명 = refresh.창을_센다("20261231", 오늘=date(2026, 9, 5))
    assert 창 == refresh.MIN_WINDOW
    assert "뒤다" in 설명


# ==================================================
# 2. 단계 구성
# ==================================================
def test_수집이_수정주가보다_먼저다():
    """수정주가는 그날 받은 시세까지 덮어야 한다. 먼저 돌리면 새 날짜가 조정 안 된 채 남는다."""
    assert refresh.STAGES.index("ingest") < refresh.STAGES.index("adj")


def test_판정이_반출보다_먼저다():
    """올릴 이유가 있는지를 먼저 묻는다. 매번 반출하면 404MB 를 매번 만든다."""
    assert refresh.STAGES.index("verify") < refresh.STAGES.index("export")
    assert refresh.STAGES.index("export") < refresh.STAGES.index("upload")


def test_모든_단계에_한국어_이름이_있다():
    """화면이 그대로 쓴다. 하나라도 비면 그 자리에 영문 열쇠가 노출된다."""
    assert set(refresh.STAGES) == set(refresh.단계이름)
    assert set(refresh.STAGES) == set(refresh.단계함수)


# ==================================================
# 3. 수정주가는 꺼도 흔적이 남나
# ==================================================
def test_수정주가를_끄면_건너뛴_것으로_남는다():
    """🔴 '안 돌렸다' 와 '돌았는데 실패했다' 는 달라야 한다."""
    결과 = refresh._단계_수정주가({"with_adj": False, "dry_run": False})
    assert 결과["skip"] is True
    assert "--with-adj" in 결과["note"]


def test_수정주가를_끈_까닭이_기록에_남는다():
    """왜 껐는지가 안 남으면 다음 사람이 '켜는 게 낫겠지' 로 간다. 그 반대다."""
    결과 = refresh._단계_수정주가({"with_adj": False, "dry_run": False})
    assert "chain" in 결과["note"]


# ==================================================
# 4. HF 토큰이 없는 PC
# ==================================================
@pytest.mark.parametrize("단계함수이름", ["_단계_판정", "_단계_반출", "_단계_업로드"])
def test_토큰이_없으면_실패가_아니라_건너뜀이다(단계함수이름):
    """팀원 PC 에는 쓰기 토큰이 없다. 붉게 칠하면 진짜 실패를 못 알아본다."""
    결과 = getattr(refresh, 단계함수이름)({"hf": False, "dry_run": False})
    assert 결과["skip"] is True


# ==================================================
# 5. 판정 결과가 반출·업로드를 가르나
# ==================================================
def test_판정이_최신이면_반출하지_않는다():
    ctx = {"hf": True, "dry_run": False, "재배포필요": False}
    assert refresh._단계_반출(ctx)["skip"] is True
    assert refresh._단계_업로드(ctx)["skip"] is True


def test_강제_반출은_판정을_빼고_불러도_먹는다(tmp_path, monkeypatch):
    """🔴 `--only export --force-export` 로 불러도 반출이 돌아야 한다.

    판정 단계 안에서만 깃발을 세우면, 판정을 뺀 호출에서는 깃발이 안 서고 반출이
    "판정이 최신이구나" 로 읽어 건너뛴다. **강제한다고 했는데 조용히 아무것도 안 하는
    것**이 가장 나쁘다. 그래서 깃발은 실행을 열 때 세운다.
    """
    monkeypatch.setattr(refresh, "LOCK_PATH", tmp_path / "refresh.lock")
    monkeypatch.setattr(refresh, "migrate_path", lambda *a, **k: None)
    monkeypatch.setattr(refresh, "db_마지막_거래일", lambda *a, **k: "20260904")
    monkeypatch.setattr(refresh, "_hf_토큰_있나", lambda: True)
    monkeypatch.setattr(run_log, "start_run", lambda *a, **k: None)
    monkeypatch.setattr(run_log, "finish_run", lambda *a, **k: None)
    monkeypatch.setattr(run_log, "start_stage", lambda *a, **k: None)
    monkeypatch.setattr(run_log, "finish_stage", lambda *a, **k: None)

    불렀나 = []
    monkeypatch.setattr(refresh, "돌린다",
                        lambda argv, **k: (불렀나.append(argv), (0, "됐다"))[1])

    인자 = _인자(only=["export"])
    인자.force_export = True
    refresh.run(인자)

    assert any("export_team_dataset.py" in " ".join(a) for a in 불렀나), \
        "강제했는데 반출이 안 돌았다"


def test_게이트_실패는_보통_실패와_다른_예외다():
    """다시 눌러도 같은 곳에서 막히는 쪽이라 화면이 갈라 보여 줘야 한다."""
    assert issubclass(refresh.GateFailed, RuntimeError)
    assert refresh.GateFailed is not RuntimeError


# ==================================================
# 6. 잠금 — 두 번 눌러도 한 번만
# ==================================================
def test_이미_잠겨_있으면_기다리지_않고_busy_를_돌려준다(tmp_path, monkeypatch):
    from filelock import FileLock

    monkeypatch.setattr(refresh, "LOCK_PATH", tmp_path / "refresh.lock")
    monkeypatch.setattr(refresh, "진행중인_실행", lambda: {"run_id": "20260905-140000",
                                                    "started_at": "2026-09-05T14:00:00+09:00"})

    먼저 = FileLock(str(tmp_path / "refresh.lock"), timeout=0)
    먼저.acquire()
    try:
        코드, 계약 = refresh.run(_인자())
    finally:
        먼저.release()

    assert 코드 == 0, "두 번째 버튼은 실패가 아니다 — 첫 번째가 돌고 있을 뿐이다"
    assert 계약["status"] == "busy"
    assert 계약["run_id"] == "20260905-140000", "화면이 폴링할 열쇠를 줘야 한다"


def test_잠금이_풀리면_다시_잡힌다(tmp_path, monkeypatch):
    """실행이 끝나면 놓아야 한다. 안 놓으면 그다음 버튼이 영영 busy 다."""
    from filelock import FileLock, Timeout

    자리 = tmp_path / "refresh.lock"
    monkeypatch.setattr(refresh, "LOCK_PATH", 자리)
    monkeypatch.setattr(refresh, "_잠금_안에서", lambda args: (0, {"status": "ok"}))

    refresh.run(_인자())

    뒤에 = FileLock(str(자리), timeout=0)
    try:
        뒤에.acquire()
    except Timeout:
        pytest.fail("실행이 끝났는데 잠금이 안 풀렸다")
    뒤에.release()


# ==================================================
# 7. 진행 중인 실행을 어떻게 찾나
# ==================================================
def test_수집_실행은_버튼_갱신으로_세지_않는다(tmp_path, monkeypatch):
    """같은 표를 나눠 쓴다. 수집이 돌고 있다고 버튼이 busy 가 되면 안 된다."""
    db = tmp_path / "t.db"
    migrations.migrate_path(db)
    monkeypatch.setattr(refresh, "krx_db_path", lambda: db)

    conn = sqlite3.connect(db)
    run_log.start_run("20260905-100000", args={"pipeline": "ingest"}, conn=conn)
    conn.commit()
    conn.close()

    assert refresh.진행중인_실행() is None


def test_버튼_갱신_실행은_찾아낸다(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    migrations.migrate_path(db)
    monkeypatch.setattr(refresh, "krx_db_path", lambda: db)

    conn = sqlite3.connect(db)
    run_log.start_run("20260905-110000",
                      args={"pipeline": refresh.PIPELINE_NAME}, conn=conn)
    conn.commit()
    conn.close()

    진행중 = refresh.진행중인_실행()
    assert 진행중 is not None
    assert 진행중["run_id"] == "20260905-110000"


def test_끝난_실행은_진행_중이_아니다(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    migrations.migrate_path(db)
    monkeypatch.setattr(refresh, "krx_db_path", lambda: db)

    conn = sqlite3.connect(db)
    run_log.start_run("20260905-120000",
                      args={"pipeline": refresh.PIPELINE_NAME}, conn=conn)
    run_log.finish_run("20260905-120000", "ok", conn=conn)
    conn.commit()
    conn.close()

    assert refresh.진행중인_실행() is None


# ==================================================
# 8. 화면에 주는 계약
# ==================================================
def test_계약에_네_칸이_다_있다(tmp_path, monkeypatch):
    """`run_id`·`status`·`stages`·`uploaded` — #89 에서 강민석 님께 약속한 모양이다.

    화면이 이 넷을 읽는다. 하나라도 빠지면 `KeyError` 가 화면 쪽에서 난다.
    """
    monkeypatch.setattr(refresh, "LOCK_PATH", tmp_path / "refresh.lock")
    monkeypatch.setattr(refresh, "진행중인_실행", lambda: None)

    from filelock import FileLock
    먼저 = FileLock(str(tmp_path / "refresh.lock"), timeout=0)
    먼저.acquire()
    try:
        _, 계약 = refresh.run(_인자())            # 잠겨 있으니 busy 로 곧장 돌아온다
    finally:
        먼저.release()

    assert set(계약) >= {"run_id", "status", "stages", "uploaded"}
    assert 계약["status"] in {"ok", "busy", "gate_failed", "error", "dry_run"}


def _인자(*, dry_run=False, only=None):
    """`argparse.Namespace` 대신 쓰는 최소 객체."""
    class 인자:
        pass
    a = 인자()
    a.dry_run = dry_run
    a.only = only
    a.with_adj = False
    a.force_export = False
    a.reuse_snapshot = False
    a.status = False
    a.limit_runs = 3
    a.json = False
    return a


# ==================================================
# 9. 자식 프로세스가 진행을 곧바로 뱉나
# ==================================================
def test_자식이_출력을_모아_두지_않게_한다():
    """🔴 이게 없으면 13분짜리 단계가 통째로 침묵한다.

    자식의 표준출력이 파이프라 파이썬이 8KB 씩 모아 쓴다. 말수가 적은 단계는 그 8KB 를
    채우는 데 몇 분이 걸리고, 사람은 멈춘 것으로 읽고 버튼을 다시 누른다.
    """
    env = refresh._자식_환경()
    assert env["PYTHONUNBUFFERED"] == "1"


def test_자식이_한글을_UTF_8_로_뱉게_한다():
    """Windows 에서 파이프로 받으면 기본이 cp949 다. 깨진 기록은 까닭을 못 알려 준다."""
    assert refresh._자식_환경()["PYTHONIOENCODING"] == "utf-8"
