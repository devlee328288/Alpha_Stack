"""홀드아웃 개봉 통제 — 순서와 횟수가 계약이다 (이슈 #240 · ADR 0004 · ADR 0009).

**무엇을 지키려는 시험인가.**

    open       이미 개봉됐으면 반출하지 않는다 · 사전등록·작업 트리·데이터 게이트가 붉으면 멈춘다
               반출 직후 한 줄을 쓰고, 그 줄의 지문이 반출 대장과 같다
    authorize  기록이 정확히 한 줄일 때만 · 입력 날짜가 섞이면 거부 · 계약 여섯 칸 · 안 덮어씀
    check      두 줄이면 "확증 불가" · 개봉 뒤 사전등록 커밋이면 붉다 · 산출물에 원자료·시크릿 없음

DB 와 git 은 건드리지 않는다. 반출·게이트·git 은 갈아 끼우고, 기록은 임시 경로에 쓴다 —
**진짜 `reports/unseal.log` 는 이 시험이 절대 만들지 않는다.**
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from hashlib import sha256
from pathlib import Path

import pandas as pd
import pytest

from common.trading_calendar import KST
from evaluation.horizon import HOLDOUT_START
from scripts import export_holdout_dataset as ehd
from scripts import unseal_holdout as uh
from supply import holdout

ROOT = Path(__file__).resolve().parents[1]
커밋 = "f5bcd82" + "0" * 33
사전등록커밋 = "2b1759d" + "0" * 33
가짜시작 = "20240603"


@pytest.fixture
def 환경(tmp_path, monkeypatch):
    """경로·git·게이트·반출을 갈아 끼운다. 반출은 대장·파일 지문이 실제로 맞는 작은 폴더다."""
    real = tmp_path / "reports" / "unseal.log"
    dry = tmp_path / "reports" / "dryrun" / "unseal.log"
    monkeypatch.setattr(uh, "log_path_for", lambda dry_run: dry if dry_run else real)
    monkeypatch.setattr(uh, "authorization_path_for",
                        lambda dry_run: tmp_path / "reports" / ("dryrun" if dry_run else "")
                        / "holdout_authorization.json")
    monkeypatch.setattr(uh, "head_commit", lambda: 커밋)
    monkeypatch.setattr(uh, "tracked_changes", lambda: [])
    monkeypatch.setattr(uh, "dry_run_start", lambda sessions=60: 가짜시작)
    monkeypatch.setattr(uh, "data_gates", lambda **kw: {"ok": True, "checks": []})
    monkeypatch.setattr(uh, "preregistration_state", lambda paths=uh.PREREGISTRATION_PATHS: {
        "missing": [], "uncommitted": [], "commits": [],
        "latest": ("docs/decisions/0004-사전등록.md", 사전등록커밋,
                   datetime(2026, 9, 1, 9, 0, tzinfo=KST))})
    불린 = []

    def 가짜_반출(root, *, holdout_start, end, dry_run, as_of, note=""):
        불린.append((holdout_start, end, dry_run))
        root = Path(root)
        (root / "full").mkdir(parents=True)
        pd.DataFrame({"bas_dd": [holdout_start], "code": ["005930"]}) \
            .to_parquet(root / holdout.HOLDOUT_DAILY, index=False)
        pd.DataFrame({"bas_dd": [holdout_start], "index_name": ["코스피 200"]}) \
            .to_parquet(root / holdout.HOLDOUT_INDEX, index=False)
        files = [{"path": p, "sha256": holdout.sha256_file(root / p)}
                 for p in (holdout.HOLDOUT_DAILY, holdout.HOLDOUT_INDEX)]
        manifest = {"dry_run": dry_run, "holdout_start": holdout_start, "end": end,
                    "files": files}
        (root / holdout.HOLDOUT_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
        return manifest

    monkeypatch.setattr(uh, "export_holdout", 가짜_반출)
    return {"real": real, "dry": dry, "tmp": tmp_path, "불린": 불린}


def 열기(tmp, **kw) -> argparse.Namespace:
    base = dict(dry_run=True, start=None, end=None, out=str(tmp / "sealed" / "snap"),
                requested_by="이동원", reason="시험")
    base.update(kw)
    return argparse.Namespace(**base)


def 입력넷(tmp: Path, *, dev_last="20240531", hold=("20240603", "20240830")) -> dict:
    폴더 = tmp / "inputs"
    폴더.mkdir(exist_ok=True)
    날짜 = {"index_dev": ["20240102", dev_last], "stock_dev": ["20240102", dev_last],
          "index_holdout": list(hold), "stock_holdout": list(hold)}
    경로 = {}
    for n, ds in 날짜.items():
        p = 폴더 / f"{n}.parquet"
        pd.DataFrame({"bas_dd": ds, "label_numeric": [0, 1]}).to_parquet(p, index=False)
        경로[n] = str(p)
    return 경로


def 승인인자(tmp: Path, snapshot: Path, 경로: dict, **kw) -> argparse.Namespace:
    cfg = tmp / "final_holdout_model.json"
    if not cfg.exists():
        # 실행기 계약의 설정 정본을 그대로 쓴다 — 칸이 모자란 가짜 설정은 실행기가 거부한다
        cfg.write_text((ROOT / "config" / "final_holdout_model.json").read_text(encoding="utf-8"),
                       encoding="utf-8")
    base = dict(dry_run=True, snapshot=str(snapshot), run_id="dry-001", config=str(cfg),
                out=None, **경로)
    base.update(kw)
    return argparse.Namespace(**base)


# ==================================================
# 1. open
# ==================================================
def test_open_은_반출_직후_기록을_한_줄_쓰고_지문이_대장과_같다(환경):
    tmp = 환경["tmp"]
    assert uh.cmd_open(열기(tmp)) == 0

    rows = holdout.read_unseal_log(환경["dry"])
    assert len(rows) == 1
    snap = tmp / "sealed" / "snap"
    assert rows[0]["snapshot_sha256"] == holdout.sha256_file(snap / holdout.HOLDOUT_MANIFEST)
    assert rows[0]["preregistration_commit"] == 사전등록커밋
    assert 환경["불린"] == [(가짜시작, uh.DEV_END, True)]
    assert not 환경["real"].exists(), "드라이런이 진짜 기록을 만들었다"


def test_open_은_이미_개봉됐으면_반출하지_않는다(환경, capsys):
    tmp = 환경["tmp"]
    assert uh.cmd_open(열기(tmp)) == 0
    assert uh.cmd_open(열기(tmp, out=str(tmp / "sealed" / "snap2"))) == 2
    assert len(환경["불린"]) == 1, "두 번째 반출이 일어났다"
    assert "확증 불가" in capsys.readouterr().out


def test_진짜_open_은_사전등록_파일이_없으면_멈춘다(환경, monkeypatch):
    monkeypatch.setattr(uh, "preregistration_state", lambda paths=uh.PREREGISTRATION_PATHS: {
        "missing": ["docs/decisions/0009-최종-홀드아웃-실행설정.md"], "uncommitted": [],
        "commits": [], "latest": ("docs/decisions/0004-사전등록.md", 사전등록커밋,
                                   datetime(2026, 9, 1, tzinfo=KST))})
    assert uh.cmd_open(열기(환경["tmp"], dry_run=False)) == 1
    assert 환경["불린"] == []
    assert not 환경["real"].exists()


def test_진짜_open_은_커밋_안_된_변경이_있으면_멈추고_드라이런은_계속한다(환경, monkeypatch):
    monkeypatch.setattr(uh, "tracked_changes", lambda: [" M supply/financial.py"])
    assert uh.cmd_open(열기(환경["tmp"], dry_run=False)) == 1
    assert 환경["불린"] == []
    assert uh.cmd_open(열기(환경["tmp"])) == 0, "드라이런은 경고만 하고 계속한다"


def test_open_은_데이터_게이트가_붉으면_반출하지_않는다(환경, monkeypatch, capsys):
    monkeypatch.setattr(uh, "data_gates", lambda **kw: {"ok": False, "checks": [
        {"name": "구간의 수정주가가 비지 않았다", "ok": False, "detail": "adj_close 빈 행 12",
         "fix": "python -m pipelines.refresh --with-adj"}]})
    assert uh.cmd_open(열기(환경["tmp"])) == 1
    assert 환경["불린"] == []
    assert "--with-adj" in capsys.readouterr().out


def test_진짜_개봉의_시작은_정본으로_고정이다(환경):
    assert uh.main(["open", "--start", "20240902", "--requested-by", "이동원",
                    "--reason", "시험"]) == 1
    assert 환경["불린"] == []


# ==================================================
# 2. authorize
# ==================================================
def test_authorize_는_실행기_계약_여섯_칸을_쓴다(환경):
    tmp = 환경["tmp"]
    assert uh.cmd_open(열기(tmp)) == 0
    snap = tmp / "sealed" / "snap"
    경로 = 입력넷(tmp)
    args = 승인인자(tmp, snap, 경로)
    assert uh.cmd_authorize(args) == 0

    승인 = json.loads(uh.authorization_path_for(True).read_text(encoding="utf-8"))
    for k in uh.CONTRACT_FIELDS:
        assert k in 승인
    assert 승인["schema_version"] == 1 and 승인["authorized"] is True
    assert 승인["holdout_start"] == HOLDOUT_START
    assert 승인["run_id"] == "dry-001"
    assert set(승인["source_sha256"]) == set(uh.SOURCE_NAMES)
    for n, p in 경로.items():
        assert 승인["source_sha256"][n] == holdout.sha256_file(Path(p))
    # 설정 지문 — 실행기 모듈이 있으면 그 계산과, 없으면 정렬된 JSON 의 SHA-256 과 대조한다
    원본 = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if 승인["config_sha256_method"] == "canonical_json_fallback":
        기대 = sha256(json.dumps(원본, ensure_ascii=False, sort_keys=True,
                               separators=(",", ":")).encode("utf-8")).hexdigest()
    else:
        from models.final_holdout import config_from_dict
        기대 = config_from_dict(원본).sha256
    assert 승인["config_sha256"] == 기대
    assert 승인["snapshot_sha256"] == holdout.read_unseal_log(환경["dry"])[0]["snapshot_sha256"]


def test_authorize_는_개봉_기록이_한_줄이_아니면_거부한다(환경):
    tmp = 환경["tmp"]
    경로 = 입력넷(tmp)
    assert uh.cmd_authorize(승인인자(tmp, tmp / "없음", 경로)) == 2      # 0줄

    assert uh.cmd_open(열기(tmp)) == 0
    한줄 = 환경["dry"].read_text(encoding="utf-8").splitlines()[1]
    with 환경["dry"].open("a", encoding="utf-8") as f:
        f.write(한줄.replace("시험", "손으로") + "\n")
    assert uh.cmd_authorize(승인인자(tmp, tmp / "sealed" / "snap", 경로)) == 2   # 2줄


def test_authorize_는_개발과_홀드아웃이_섞인_입력을_거부한다(환경, capsys):
    tmp = 환경["tmp"]
    assert uh.cmd_open(열기(tmp)) == 0
    경로 = 입력넷(tmp, dev_last="20240610")                   # 가짜 시작 20240603 뒤까지
    assert uh.cmd_authorize(승인인자(tmp, tmp / "sealed" / "snap", 경로)) == 2
    assert "홀드아웃 시작" in capsys.readouterr().out
    assert not uh.authorization_path_for(True).exists()


def test_authorize_는_승인을_덮어쓰지_않는다(환경):
    tmp = 환경["tmp"]
    assert uh.cmd_open(열기(tmp)) == 0
    args = 승인인자(tmp, tmp / "sealed" / "snap", 입력넷(tmp))
    assert uh.cmd_authorize(args) == 0
    assert uh.cmd_authorize(args) == 2


def test_authorize_는_계약에_안_맞는_설정이면_멈추고_승인을_쓰지_않는다(환경, capsys):
    """칸이 모자란 설정은 실행기가 거부한다. 트레이스백 대신 할 일을 말하고 승인 파일은 안 쓴다."""
    pytest.importorskip("models.final_holdout")
    tmp = 환경["tmp"]
    assert uh.cmd_open(열기(tmp)) == 0
    모자란 = tmp / "모자란_설정.json"
    모자란.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    args = 승인인자(tmp, tmp / "sealed" / "snap", 입력넷(tmp), config=str(모자란))
    assert uh.cmd_authorize(args) == 1
    assert "ADR 0009" in capsys.readouterr().out
    assert not uh.authorization_path_for(True).exists()


def test_실행기_계약과_실제로_맞는다(환경):
    """오준영 님 실행기가 들어오면 그 검증 함수에 이 승인 파일을 그대로 넣어 본다."""
    runner = pytest.importorskip("scripts.run_final_holdout")
    tmp = 환경["tmp"]
    assert uh.cmd_open(열기(tmp)) == 0
    assert uh.cmd_authorize(승인인자(tmp, tmp / "sealed" / "snap", 입력넷(tmp))) == 0
    승인 = json.loads(uh.authorization_path_for(True).read_text(encoding="utf-8"))
    assert runner.validate_run_mode(
        mode="official", evaluation_dates=[HOLDOUT_START],
        config_sha256=승인["config_sha256"], source_sha256=승인["source_sha256"],
        authorization=승인) == "dry-001"


# ==================================================
# 3. check
# ==================================================
def test_check_는_미개봉이면_조용히_끝난다(환경):
    assert uh.cmd_check(argparse.Namespace(dry_run=True, authorization=None, result_dir=None)) == 0


def test_check_는_두_줄이면_확증_불가로_끝난다(환경, capsys):
    tmp = 환경["tmp"]
    assert uh.cmd_open(열기(tmp)) == 0
    한줄 = 환경["dry"].read_text(encoding="utf-8").splitlines()[1]
    with 환경["dry"].open("a", encoding="utf-8") as f:
        f.write(한줄 + "\n")
    assert uh.cmd_check(argparse.Namespace(dry_run=True, authorization=None, result_dir=None)) == 2
    assert "홀드아웃 2회 개봉 — 확증 불가" in capsys.readouterr().out


def test_check_는_개봉_뒤에_사전등록이_커밋되면_붉다(환경, monkeypatch, capsys):
    tmp = 환경["tmp"]
    assert uh.cmd_open(열기(tmp)) == 0
    monkeypatch.setattr(uh, "commit_time", lambda c: datetime(2026, 9, 1, tzinfo=KST))
    monkeypatch.setattr(uh, "last_commit_of",
                        lambda p: (사전등록커밋, datetime(2099, 1, 1, tzinfo=KST)))
    assert uh.cmd_check(argparse.Namespace(dry_run=True, authorization=None, result_dir=None)) == 2
    assert "개봉 후 수정 금지" in capsys.readouterr().out


# ==================================================
# 4. 산출물 검사
# ==================================================
def test_산출물_검사는_시크릿_금지파일_원자료칸을_잡고_값은_출력하지_않는다(tmp_path):
    비밀 = "sk-TESTSECRET-1234567890"
    결과 = tmp_path / "result"
    결과.mkdir()
    (결과 / "report.json").write_text(json.dumps({"token": 비밀}), encoding="utf-8")
    (결과 / "copy.db").write_bytes(b"sqlite")
    pd.DataFrame({"bas_dd": ["20240902"], "code": ["005930"], "close": [1.0]}) \
        .to_parquet(결과 / "stock_predictions.parquet", index=False)

    문제 = uh.scan_outputs(결과, [비밀])
    assert len(문제) == 3
    assert not any(비밀 in s for s in 문제), "시크릿 값이 문구에 새어 나왔다"

    깨끗 = tmp_path / "clean"
    깨끗.mkdir()
    pd.DataFrame({"bas_dd": ["20240902"], "code": ["005930"], "p_up": [0.4]}) \
        .to_parquet(깨끗 / "stock_predictions.parquet", index=False)
    assert uh.scan_outputs(깨끗, [비밀]) == []


# ==================================================
# 5. 반출 범위 · CLI · gitignore
# ==================================================
def test_드라이런_반출은_진짜_봉인_구간에_닿지_않는다():
    with pytest.raises(ValueError, match="진짜 봉인 시작"):
        ehd.validate_range(holdout_start=HOLDOUT_START, end=ehd.DEV_END, dry_run=True)
    with pytest.raises(ValueError, match="개발구간 끝"):
        ehd.validate_range(holdout_start=가짜시작, end=HOLDOUT_START, dry_run=True)
    ehd.validate_range(holdout_start=가짜시작, end=ehd.DEV_END, dry_run=True)


def test_진짜_반출은_경계가_정본이어야_한다():
    with pytest.raises(ValueError, match="코드 정본"):
        ehd.validate_range(holdout_start="20240902", end="20250101", dry_run=False)
    ehd.validate_range(holdout_start=HOLDOUT_START, end="20250101", dry_run=False)


def test_진짜_반출은_CLI_로_만들_수_없다(capsys):
    assert ehd.main([]) == 2
    assert "unseal_holdout.py open" in capsys.readouterr().out


def test_gitignore_는_개봉_기록만_커밋하게_둔다():
    """ADR 0004 의 증거는 git 이력이다 — 개봉 기록이 무시되면 증거가 사라진다."""
    본문 = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "*.log" in 본문 and "!reports/unseal.log" in 본문
    assert 본문.index("!reports/unseal.log") > 본문.index("*.log"), "예외는 규칙 뒤에 와야 먹는다"
    assert "reports/dryrun/" in 본문
    assert "data/sealed/" in 본문
