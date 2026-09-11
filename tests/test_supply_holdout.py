"""홀드아웃 입구 — 한 번만 쓰고, 기록과 지문이 맞을 때만 연다 (ADR 0004 · ADR 0007 · 이슈 #240).

**무엇을 지키려는 시험인가.** 홀드아웃은 한 번 열면 끝이다. 두 번 연 홀드아웃은 개발구간이 되고
되돌릴 방법이 없다. 그래서 여기서 잠그는 것은 기능이 아니라 **순서와 횟수**다.

    ① 개봉 기록은 한 번만 쓴다 — 두 번째 줄은 **쓰기 전에** 막는다
    ② 손으로 두 줄이 되면 읽는 쪽이 "홀드아웃 2회 개봉 — 확증 불가" 를 알린다
    ③ 기록이 없거나 지문이 다른 반출본은 열지 않는다
    ④ 드라이런 기록과 진짜 기록은 섞이지 않는다
    ⑤ 형식은 ADR 0004 §1 글자 그대로 — `grep -vc '^#'` 이 곧 데이터 행 수다

진짜 `reports/unseal.log` 는 건드리지 않는다. 기록은 전부 임시 경로에 쓴다.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from common.trading_calendar import KST
from evaluation.horizon import HOLDOUT_START
from supply import holdout

커밋 = "f5bcd82" + "0" * 33
사전등록 = "2b1759dbfed87ecb3691802b93a9b13e64cfa987"


def 반출본(tmp_path: Path, *, dry_run: bool = False, holdout_start: str = HOLDOUT_START) -> Path:
    """봉인 반출본 모양만 갖춘 작은 폴더 — 대장·파일 지문이 실제로 맞는다."""
    root = tmp_path / ("dry" if dry_run else "real")
    (root / "full").mkdir(parents=True)
    pd.DataFrame({"bas_dd": ["20240830", "20240902"], "code": ["005930", "005930"],
                  "adj_close": [1.0, 2.0]}).to_parquet(root / holdout.HOLDOUT_DAILY, index=False)
    pd.DataFrame({"bas_dd": ["20240830", "20240902"], "index_name": ["코스피 200"] * 2,
                  "close": [1.0, 2.0]}).to_parquet(root / holdout.HOLDOUT_INDEX, index=False)
    files = [{"path": p, "sha256": holdout.sha256_file(root / p)}
             for p in (holdout.HOLDOUT_DAILY, holdout.HOLDOUT_INDEX)]
    manifest = {"schema_version": 1, "dry_run": dry_run, "holdout_start": holdout_start,
                "files": files}
    (root / holdout.HOLDOUT_MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False),
                                                 encoding="utf-8")
    return root


def 기록줄(root: Path, reason: str = "시험") -> dict:
    return holdout.build_unseal_row(
        git_commit=커밋,
        snapshot_sha256=holdout.sha256_file(root / holdout.HOLDOUT_MANIFEST),
        preregistration_commit=사전등록, requested_by="이동원", reason=reason,
        now=datetime(2026, 9, 12, 9, 0, tzinfo=KST))


# ==================================================
# 1. 형식 — ADR 0004 §1 글자 그대로
# ==================================================
def test_형식은_ADR_0004_글자_그대로다(tmp_path):
    root = 반출본(tmp_path)
    log = holdout.append_unseal_row(기록줄(root), path=tmp_path / "unseal.log")

    줄들 = log.read_text(encoding="utf-8").splitlines()
    assert 줄들[0] == holdout.UNSEAL_HEADER
    assert 줄들[1].split(" | ")[0] == "2026-09-12T09:00:00+09:00"
    assert len(줄들[1].split(" | ")) == len(holdout.UNSEAL_FIELDS) == 6
    # ADR 0004 §검증: head -2 는 주석 1줄 + 데이터 1줄, grep -vc '^#' == 1
    assert sum(1 for s in 줄들 if not s.startswith("#")) == 1


def test_칸_수가_다른_줄은_조용히_넘기지_않는다(tmp_path):
    """넘기면 '0줄이니 미개봉' 으로 읽힌다 — 가장 나쁜 방향의 오독이다."""
    log = tmp_path / "unseal.log"
    log.write_text(holdout.UNSEAL_HEADER + "\n2026-09-12 | abc | 없음 | 이동원 | 사유\n",
                   encoding="utf-8")
    with pytest.raises(holdout.UnsealError, match="칸 수"):
        holdout.read_unseal_log(log)


def test_기록_값에_구분자나_줄바꿈이_있으면_세운다(tmp_path):
    root = 반출본(tmp_path)
    with pytest.raises(holdout.UnsealError, match="구분자"):
        기록줄(root, reason="앞|뒤")
    with pytest.raises(holdout.UnsealError, match="줄바꿈"):
        기록줄(root, reason="앞\n뒤")
    with pytest.raises(holdout.UnsealError, match="64자"):
        holdout.build_unseal_row(git_commit=커밋, snapshot_sha256="abc",
                                 preregistration_commit=사전등록, requested_by="이동원",
                                 reason="시험")
    with pytest.raises(holdout.UnsealError, match="커밋 해시"):
        holdout.build_unseal_row(git_commit="main", snapshot_sha256="a" * 64,
                                 preregistration_commit=사전등록, requested_by="이동원",
                                 reason="시험")


# ==================================================
# 2. 횟수 — 한 번만 쓰고, 두 줄이면 확증 불가
# ==================================================
def test_두_번째_개봉은_쓰기_전에_막는다(tmp_path):
    root = 반출본(tmp_path)
    log = tmp_path / "unseal.log"
    holdout.append_unseal_row(기록줄(root), path=log)

    with pytest.raises(holdout.UnsealError) as 잡힘:
        holdout.append_unseal_row(기록줄(root, reason="한 번 더"), path=log)
    assert "확증 불가" in str(잡힘.value)
    assert len(holdout.read_unseal_log(log)) == 1, "막았다고 하고 실제로는 썼다"


def test_손으로_두_줄이_되면_확증_불가를_알린다(tmp_path):
    root = 반출본(tmp_path)
    log = tmp_path / "unseal.log"
    holdout.append_unseal_row(기록줄(root), path=log)
    한줄 = log.read_text(encoding="utf-8").splitlines()[1]
    with log.open("a", encoding="utf-8") as f:
        f.write(한줄.replace("시험", "손으로 추가") + "\n")

    판정 = holdout.unseal_verdict(holdout.read_unseal_log(log))
    assert 판정 == {"rows": 2, "status": "확증 불가", "confirmable": False,
                  "banner": "홀드아웃 2회 개봉 — 확증 불가"}
    with pytest.raises(holdout.UnsealError, match="2회 개봉"):
        holdout.verify_holdout_snapshot(root, log_path=log)


def test_판정은_세_갈래다():
    assert holdout.unseal_verdict([])["status"] == "미개봉"
    assert holdout.unseal_verdict([{}])["confirmable"] is True
    assert holdout.unseal_verdict([{}, {}, {}])["banner"] == "홀드아웃 3회 개봉 — 확증 불가"


# ==================================================
# 3. 여는 조건 — 기록과 지문
# ==================================================
def test_기록이_없으면_열지_않는다(tmp_path):
    root = 반출본(tmp_path)
    with pytest.raises(holdout.UnsealError, match="개봉 기록이 없다"):
        holdout.load_holdout_snapshot(root, log_path=tmp_path / "없는.log")


def test_기록과_지문이_맞으면_연다(tmp_path):
    root = 반출본(tmp_path)
    log = tmp_path / "unseal.log"
    holdout.append_unseal_row(기록줄(root), path=log)

    snap = holdout.load_holdout_snapshot(root, log_path=log, daily_columns=["bas_dd", "code"])
    assert list(snap.daily.columns) == ["bas_dd", "code"]
    assert len(snap.index) == 2
    assert snap.unseal["requested_by"] == "이동원"
    assert snap.snapshot_sha256 == holdout.sha256_file(root / holdout.HOLDOUT_MANIFEST)


def test_기록한_뒤_대장이_바뀌면_열지_않는다(tmp_path):
    """기록 뒤에 다시 반출하면 지문이 달라진다 — 그건 두 번째 개봉이다."""
    root = 반출본(tmp_path)
    log = tmp_path / "unseal.log"
    holdout.append_unseal_row(기록줄(root), path=log)
    대장 = json.loads((root / holdout.HOLDOUT_MANIFEST).read_text(encoding="utf-8"))
    대장["generated_at"] = "나중"
    (root / holdout.HOLDOUT_MANIFEST).write_text(json.dumps(대장), encoding="utf-8")

    with pytest.raises(holdout.UnsealError, match="지문"):
        holdout.load_holdout_snapshot(root, log_path=log)


def test_기록한_뒤_파일이_바뀌면_열지_않는다(tmp_path):
    root = 반출본(tmp_path)
    log = tmp_path / "unseal.log"
    holdout.append_unseal_row(기록줄(root), path=log)
    pd.DataFrame({"bas_dd": ["20240902"], "code": ["005930"], "adj_close": [9.0]}) \
        .to_parquet(root / holdout.HOLDOUT_DAILY, index=False)

    with pytest.raises(holdout.UnsealError, match="SHA-256"):
        holdout.load_holdout_snapshot(root, log_path=log)


def test_홀드아웃_시작이_정본과_다르면_세운다(tmp_path):
    root = 반출본(tmp_path, holdout_start="20210901")
    log = tmp_path / "unseal.log"
    holdout.append_unseal_row(기록줄(root), path=log)
    with pytest.raises(holdout.UnsealError, match="코드 정본"):
        holdout.load_holdout_snapshot(root, log_path=log)


# ==================================================
# 4. 드라이런과 진짜는 섞이지 않는다
# ==================================================
def test_드라이런_반출본은_진짜_기록으로_열리지_않는다(tmp_path):
    """진짜 기록 경로를 넘기면 **읽기 전에** 세운다 — 진짜 기록 파일은 열어 보지도 않는다."""
    root = 반출본(tmp_path, dry_run=True, holdout_start="20240603")
    with pytest.raises(holdout.UnsealError, match="드라이런 반출본"):
        holdout.verify_holdout_snapshot(root, log_path=holdout.UNSEAL_LOG)


def test_진짜_반출본은_드라이런_기록으로_열리지_않는다(tmp_path):
    root = 반출본(tmp_path)
    with pytest.raises(holdout.UnsealError, match="진짜 반출본"):
        holdout.verify_holdout_snapshot(root, log_path=holdout.DRY_RUN_UNSEAL_LOG)


def test_기록_경로가_ADR_과_같다():
    """ADR 0004 가 이 경로를 명령으로 적어 두었다(`grep -vc '^#' reports/unseal.log`)."""
    assert holdout.UNSEAL_LOG.parts[-2:] == ("reports", "unseal.log")
    assert holdout.DRY_RUN_UNSEAL_LOG.parts[-3:] == ("reports", "dryrun", "unseal.log")
