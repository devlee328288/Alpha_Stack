"""홀드아웃 개봉 통제 — 반출과 기록은 한 번에, 승인은 기록이 한 줄일 때만 (이슈 #240).

    # ① 개봉 — 사전 조건 확인 → 봉인 반출 → reports/unseal.log 에 한 줄
    python scripts/unseal_holdout.py open --requested-by 이동원 --reason "최종모델 C/K 1회 평가"

    # ② 승인 — 모델 파트가 만든 입력 넷의 지문과 설정 지문 (실행기 계약 여섯 칸)
    python scripts/unseal_holdout.py authorize --snapshot data/sealed/holdout_... --run-id <ID>
        --index-dev ... --index-holdout ... --stock-dev ... --stock-holdout ...

    # ③ 확인 — 개봉 1회 · 사전등록이 앞섬 · 승인·결과가 맞음 · 산출물에 원자료·시크릿 없음
    python scripts/unseal_holdout.py check --result-dir <실행기 출력 폴더>

드라이런은 셋 다 `--dry-run` 을 붙인다. 기록·승인은 `reports/dryrun/` 에 따로 남는다.

## 순서가 곧 계약이다 (ADR 0004 · ADR 0007 · ADR 0009)

1. **사전등록이 먼저 커밋돼 있어야 한다.** `open` 은 사전등록 파일들(ADR 0004 · 0007 · 0009 ·
   `config/final_holdout_model.json`)이 커밋돼 있는지 보고, 가장 늦은 커밋을 기록에 적는다.
   진짜 개봉은 추적 파일에 커밋 안 된 변경이 하나라도 있으면 멈춘다 — 무엇을 돌렸는지가
   커밋으로 남아야 하기 때문이다.
2. **데이터 게이트** — 달력·시세·지수가 끝날까지 있고, 구간의 수정주가가 비지 않았다.
   못 넘으면 `python -m pipelines.refresh --with-adj` 부터.
3. **반출이 곧 개봉이다.** 반출이 끝나면 곧바로 한 줄을 쓴다. 이미 한 줄이 있으면 반출하지 않는다.
4. **승인** 은 기록이 정확히 한 줄이고, 봉인 반출본의 지문이 그 줄과 같고, 입력 넷의 날짜가
   개발/홀드아웃으로 섞이지 않았을 때만 쓴다. 승인 파일은 덮어쓰지 않는다.
5. **확인** — 개봉 뒤에 사전등록 파일이 커밋되면 붉다(ADR 0004 §9 "개봉 후 수정 금지").
   두 줄이면 "홀드아웃 {n}회 개봉 — 확증 불가" 로 끝난다.

실행기(`scripts/run_final_holdout.py` · 오준영 님)의 공식 모드는 승인 JSON 여섯 칸
(`schema_version`·`authorized`·`run_id`·`holdout_start`·`config_sha256`·`source_sha256`)을
본다. 이 파일은 그 여섯 칸에 **되짚기 위한 칸**(개봉 행 · 스냅샷 지문 · 입력 경로 · 커밋)을 더한다.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pyarrow.parquet as pq  # noqa: E402

from common.trading_calendar import KST, session_span  # noqa: E402
from evaluation.horizon import HOLDOUT_START  # noqa: E402
from ingest.store import krx_store  # noqa: E402
from scripts.export_holdout_dataset import (  # noqa: E402
    DEFAULT_SEALED_ROOT,
    dry_run_start,
    export_holdout,
)
from scripts.export_team_dataset import DEV_END  # noqa: E402
from supply import holdout  # noqa: E402

#: 사전등록 — 개봉보다 **먼저** 커밋돼 있어야 하는 파일들. 가장 늦은 커밋이 기록에 들어간다.
PREREGISTRATION_PATHS = (
    "docs/decisions/0004-사전등록.md",
    "docs/decisions/0007-최종-모델-선정과-홀드아웃.md",
    "docs/decisions/0009-최종-홀드아웃-실행설정.md",
    "config/final_holdout_model.json",
)

#: 실행기 계약 — `scripts/run_final_holdout.py` 의 `SOURCE_NAMES` 와 필수 여섯 칸.
SOURCE_NAMES = ("index_dev", "index_holdout", "stock_dev", "stock_holdout")
CONTRACT_FIELDS = ("schema_version", "authorized", "run_id", "holdout_start",
                   "config_sha256", "source_sha256")
DEFAULT_CONFIG = ROOT / "config" / "final_holdout_model.json"

#: 최종 산출물에 있으면 안 되는 것.
FORBIDDEN_SUFFIXES = (".db", ".sqlite", ".sqlite3", ".zip", ".pdf", ".key")
FORBIDDEN_NAMES = (".env", ".key")
RAW_PRICE_COLUMNS = frozenset({
    "open", "high", "low", "close", "adj_open", "adj_high", "adj_low", "adj_close",
    "adj_close_tr", "volume", "value", "market_cap", "listed_shares",
})
SECRET_MIN_LENGTH = 12


# ==================================================
# 1. 경로 — 시험이 갈아 끼울 자리를 한 곳에 둔다
# ==================================================
def log_path_for(dry_run: bool) -> Path:
    return holdout.DRY_RUN_UNSEAL_LOG if dry_run else holdout.UNSEAL_LOG


def authorization_path_for(dry_run: bool) -> Path:
    폴더 = ROOT / "reports" / ("dryrun" if dry_run else "")
    return 폴더 / "holdout_authorization.json"


# ==================================================
# 2. git — 무엇을 돌렸나 · 사전등록이 앞섰나
# ==================================================
def _git(*args: str) -> str:
    return subprocess.run(["git", "-c", "core.quotepath=false", *args], cwd=ROOT,
                          capture_output=True, text=True, encoding="utf-8",
                          check=True).stdout.strip()


def head_commit() -> str:
    return _git("rev-parse", "HEAD")


def tracked_changes() -> List[str]:
    """추적 파일 중 커밋 안 된 변경. 새 파일(untracked)은 세지 않는다."""
    return [s for s in _git("status", "--porcelain", "--untracked-files=no").splitlines()
            if s.strip()]


def last_commit_of(path: str) -> Optional[Tuple[str, datetime]]:
    out = _git("log", "-1", "--format=%H %cI", "--", path)
    if not out:
        return None
    h, t = out.split(" ", 1)
    return h, datetime.fromisoformat(t)


def commit_time(commit: str) -> datetime:
    return datetime.fromisoformat(_git("show", "-s", "--format=%cI", commit))


def preregistration_state(paths: Sequence[str] = PREREGISTRATION_PATHS) -> Dict[str, object]:
    """사전등록 파일들의 상태. `latest` = (경로, 커밋, 커밋 시각) — 가장 늦게 커밋된 것."""
    없음, 미커밋, 커밋들 = [], [], []
    바뀐 = tracked_changes()
    for p in paths:
        if not (ROOT / p).exists():
            없음.append(p)
            continue
        c = last_commit_of(p)
        if c is None:
            미커밋.append(p)
            continue
        if any(s.endswith(p) for s in 바뀐):
            미커밋.append(f"{p} (커밋 뒤 수정됨)")
        커밋들.append((p, c[0], c[1]))
    latest = max(커밋들, key=lambda x: x[2]) if 커밋들 else None
    return {"missing": 없음, "uncommitted": 미커밋, "commits": 커밋들, "latest": latest}


def config_sha256(config_path: Path) -> Tuple[str, str]:
    """설정 지문과 계산 방법. 실행기 모듈이 있으면 **그 계산**을 쓴다(같은 값이어야 하므로)."""
    data = json.loads(Path(config_path).read_text(encoding="utf-8"))
    try:
        # 오준영 님 실행기(PR #244) — 설정 지문은 실행기와 같은 계산이어야 한다
        from models.final_holdout import config_from_dict
    except ImportError:
        canonical = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return sha256(canonical.encode("utf-8")).hexdigest(), "canonical_json_fallback"
    return config_from_dict(data).sha256, "models.final_holdout"


# ==================================================
# 3. 데이터 게이트
# ==================================================
def data_gates(*, start: str, end: str) -> Dict[str, object]:
    """달력·시세·지수·수정주가. 못 넘으면 무엇을 해야 하는지까지 돌려준다."""
    _, 달력끝 = session_span()
    with krx_store.connect() as conn:
        시세끝 = conn.execute("SELECT MAX(bas_dd) FROM daily_price").fetchone()[0] or ""
        지수끝 = conn.execute("SELECT MAX(bas_dd) FROM index_price").fetchone()[0] or ""
        빈수정 = conn.execute(
            "SELECT COUNT(*) FROM daily_price WHERE bas_dd BETWEEN ? AND ? "
            "AND adj_close IS NULL", (start, end)).fetchone()[0]
    검사 = [
        ("달력이 끝날까지 있다", 달력끝 >= end, f"달력 끝 {달력끝}", "python -m pipelines.refresh"),
        ("시세가 끝날까지 있다", 시세끝 >= end, f"daily_price 끝 {시세끝}",
         "python -m pipelines.refresh"),
        ("지수가 끝날까지 있다", 지수끝 >= end, f"index_price 끝 {지수끝}",
         "python -m pipelines.refresh"),
        ("구간의 수정주가가 비지 않았다", 빈수정 == 0, f"adj_close 빈 행 {빈수정:,}",
         "python -m pipelines.refresh --with-adj"),
    ]
    return {"ok": all(ok for _, ok, _, _ in 검사),
            "checks": [{"name": n, "ok": bool(ok), "detail": d, "fix": f}
                       for n, ok, d, f in 검사]}


# ==================================================
# 4. 산출물 검사 — 원자료 · 시크릿
# ==================================================
def secret_values() -> List[str]:
    """`.env`·`.key` 와 환경변수의 키 값. **출력하지 않는다** — 산출물 안에 있는지만 본다."""
    from common import secrets

    값들 = set()
    for 파일 in (secrets.ENV_FILE, secrets.KEY_FILE):
        if not Path(파일).exists():
            continue
        for line in Path(파일).read_text(encoding="utf-8", errors="ignore").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            v = s.partition("=")[2].strip().strip('"').strip("'")
            if len(v) >= SECRET_MIN_LENGTH:
                값들.add(v)
    for 이름, v in os.environ.items():
        if any(t in 이름.upper() for t in ("KEY", "TOKEN", "SECRET")) and \
                len(v) >= SECRET_MIN_LENGTH:
            값들.add(v)
    return sorted(값들)


def scan_outputs(root: Path, secrets_: Sequence[str]) -> List[str]:
    """최종 산출물 폴더를 전수로 본다. 문제마다 한 줄 — **시크릿 값은 문구에 넣지 않는다.**"""
    문제: List[str] = []
    for p in sorted(Path(root).rglob("*")):
        if p.is_dir():
            continue
        이름 = p.relative_to(root).as_posix()
        if p.suffix.lower() in FORBIDDEN_SUFFIXES or p.name in FORBIDDEN_NAMES:
            문제.append(f"들어 있으면 안 되는 파일: {이름}")
        내용 = p.read_bytes()
        if any(v.encode("utf-8") in 내용 for v in secrets_):
            문제.append(f"시크릿 값이 들어 있다: {이름} (값은 출력하지 않는다)")
        칸들: List[str] = []
        if p.suffix.lower() == ".parquet":
            칸들 = list(pq.ParquetFile(p).schema_arrow.names)
        elif p.suffix.lower() == ".csv":
            칸들 = 내용.split(b"\n", 1)[0].decode("utf-8", errors="ignore").strip().split(",")
        원자료 = sorted(set(칸들) & RAW_PRICE_COLUMNS)
        if 원자료:
            문제.append(f"원자료 가격 칸이 있다: {이름} {원자료}")
    return 문제


# ==================================================
# 5. 명령
# ==================================================
def cmd_open(args: argparse.Namespace) -> int:
    dry = bool(args.dry_run)
    log = log_path_for(dry)
    기존 = holdout.read_unseal_log(log)
    if 기존:
        판정 = holdout.unseal_verdict([*기존, {}])
        print(f"🔴 이미 개봉됐다 — {log} 에 {len(기존)}줄 (첫 개봉 {기존[0]['unsealed_at_kst']}).")
        print(f"   다시 열면 '{판정['banner']}' 이다. 반출도 기록도 하지 않았다.")
        return 2

    print(f"── {'드라이런' if dry else '🔴 진짜'} 개봉 · 기록 {log}")

    # 사전 조건은 **DB 를 읽기 전에** 본다 — 멈출 개봉이면 달력·시세에 손대지 않는다.
    사전 = preregistration_state()
    바뀐추적 = tracked_changes()
    문제 = []
    if 사전["missing"]:
        문제.append(("사전등록 파일이 없다", 사전["missing"]))
    if 사전["uncommitted"]:
        문제.append(("사전등록 파일이 커밋되지 않았다", 사전["uncommitted"]))
    if 바뀐추적:
        문제.append(("추적 파일에 커밋 안 된 변경이 있다", 바뀐추적[:10]))
    for 제목, 목록 in 문제:
        print(f"  {'⚠️' if dry else '🔴'} {제목}: {' · '.join(목록)}")
    if 문제 and not dry:
        print("   진짜 개봉은 무엇을 돌렸는지가 커밋으로 남아야 한다. 커밋·머지 뒤 다시 실행한다.")
        return 1
    if 문제:
        print("   (드라이런이라 계속한다 — 진짜 개봉에서는 여기서 멈춘다)")
    if 사전["latest"] is None:
        print("🔴 커밋된 사전등록 파일이 하나도 없다 — 사전등록 없이 열 수 없다.")
        return 1

    start = args.start or (dry_run_start() if dry else HOLDOUT_START)
    end = args.end or (DEV_END if dry else session_span()[1])
    print(f"   구간 {start} ~ {end}")
    게이트 = data_gates(start=start, end=end)
    for c in 게이트["checks"]:
        꼬리 = "" if c["ok"] else f" · 할 일: {c['fix']}"
        print(f"  {'✅' if c['ok'] else '🔴'} {c['name']} — {c['detail']}{꼬리}")
    if not 게이트["ok"]:
        return 1

    이름 = f"{'dryrun' if dry else 'holdout'}_{start}_{end}"
    root = Path(args.out) if args.out else DEFAULT_SEALED_ROOT / 이름
    오늘 = datetime.now(KST).strftime("%Y%m%d")
    export_holdout(root, holdout_start=start, end=end, dry_run=dry, as_of=오늘, note=args.reason)
    지문 = holdout.sha256_file(root / holdout.HOLDOUT_MANIFEST)
    try:
        줄 = holdout.build_unseal_row(
            git_commit=head_commit(), snapshot_sha256=지문,
            preregistration_commit=사전["latest"][1],
            requested_by=args.requested_by, reason=args.reason)
        holdout.append_unseal_row(줄, path=log)
    except holdout.UnsealError as exc:
        print(f"🔴 반출은 끝났는데 기록을 못 썼다 — {exc}")
        print(f"   {root} 는 기록 없이 열린 봉인 반출본이다.")
        print("   지우지 말고 사람이 사유와 함께 기록한다.")
        return 2

    개봉시각 = datetime.fromisoformat(줄["unsealed_at_kst"])
    if not 사전["latest"][2] < 개봉시각:
        print("🔴 사전등록 커밋이 개봉보다 앞서지 않는다 — ADR 0004 가 무효가 된다.")
        return 2
    print(f"✅ 기록 {log}")
    print(f"   {줄['unsealed_at_kst']} · 스냅샷 {지문[:16]}… · "
          f"사전등록 {줄['preregistration_commit'][:7]} ({사전['latest'][0]})")
    print("다음 — 모델 파트가 이 반출본으로 입력 넷을 만든 뒤:")
    print(f"   python scripts/unseal_holdout.py authorize{' --dry-run' if dry else ''} "
          f"--snapshot {root} --run-id <ID> --index-dev … --index-holdout … "
          "--stock-dev … --stock-holdout …")
    return 0


def _date_range(path: Path) -> Tuple[str, str]:
    칸 = pq.read_table(path, columns=["bas_dd"]).column("bas_dd").to_pandas().astype(str)
    return str(칸.min()), str(칸.max())


def cmd_authorize(args: argparse.Namespace) -> int:
    dry = bool(args.dry_run)
    log = log_path_for(dry)
    rows = holdout.read_unseal_log(log)
    판정 = holdout.unseal_verdict(rows)
    if 판정["rows"] != 1:
        말 = "아직 개봉하지 않았다 — open 부터" if 판정["rows"] == 0 else 판정["banner"]
        print(f"🔴 승인하지 않는다 — {log}: {말}")
        return 2

    try:
        확인 = holdout.verify_holdout_snapshot(Path(args.snapshot), log_path=log)
    except holdout.UnsealError as exc:
        print(f"🔴 봉인 반출본이 개봉 기록과 맞지 않는다 — {exc}")
        return 2
    시작 = str(확인["manifest"]["holdout_start"])
    끝 = str(확인["manifest"]["end"])

    경로들 = {n: Path(getattr(args, n)) for n in SOURCE_NAMES}
    없음 = [f"{n}={p}" for n, p in 경로들.items() if not p.exists()]
    if 없음:
        print(f"🔴 입력이 없다: {' · '.join(없음)}")
        return 1
    섞임 = []
    for n in ("index_dev", "stock_dev"):
        lo, hi = _date_range(경로들[n])
        if hi >= 시작:
            섞임.append(f"{n} 의 마지막 날({hi})이 홀드아웃 시작({시작}) 이후다")
    for n in ("index_holdout", "stock_holdout"):
        lo, hi = _date_range(경로들[n])
        if lo < 시작 or hi > 끝:
            섞임.append(f"{n} 의 구간({lo}~{hi})이 봉인 구간({시작}~{끝}) 밖이다")
    if 섞임:
        for s in 섞임:
            print(f"🔴 {s}")
        return 2

    run_id = (args.run_id or "").strip()
    if not run_id:
        print("🔴 run_id 가 비어 있다 — 결과·백테스트 원장·화면이 같은 ID 로 이어져야 한다.")
        return 1
    out = Path(args.out) if args.out else authorization_path_for(dry)
    if out.exists():
        print(f"🔴 {out} 가 이미 있다 — 승인은 덮어쓰지 않는다.")
        return 2

    try:
        지문, 방법 = config_sha256(Path(args.config))
    except (OSError, KeyError, ValueError, TypeError) as exc:
        print(f"🔴 설정 파일을 실행기 계약으로 읽지 못했다 — {args.config}")
        print(f"   {type(exc).__name__}: {exc}")
        print("   할 일: ADR 0009 정본 config/final_holdout_model.json 과 칸을 맞춘다.")
        return 1
    승인 = {
        "schema_version": 1,
        "authorized": True,
        "run_id": run_id,
        "holdout_start": HOLDOUT_START,
        "config_sha256": 지문,
        "source_sha256": {n: holdout.sha256_file(p) for n, p in 경로들.items()},
        # ── 여기부터 되짚기 위한 칸 (실행기는 보지 않는다) ──
        "dry_run": dry,
        "evaluation_range": [시작, 끝],
        "authorized_at_kst": datetime.now(KST).isoformat(timespec="seconds"),
        "git_commit": head_commit(),
        "config_path": str(args.config),
        "config_sha256_method": 방법,
        "source_path": {n: str(p) for n, p in 경로들.items()},
        "snapshot_sha256": 확인["snapshot_sha256"],
        "unseal": 확인["unseal"],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(승인, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"✅ 승인 {out} · run_id {run_id} · 설정 {지문[:16]}… ({방법})")
    if not dry:
        print("다음 — python scripts/run_final_holdout.py --mode official "
              f"--authorization {out} … --output-dir <결과 폴더>")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    dry = bool(args.dry_run)
    log = log_path_for(dry)
    rows = holdout.read_unseal_log(log)
    판정 = holdout.unseal_verdict(rows)
    print(f"── 개봉 기록 {log} — {판정['status']} ({판정['rows']}줄)")
    if 판정["rows"] == 0:
        print("   아직 열지 않았다 — 확인할 개봉이 없다.")
        return 0
    if 판정["rows"] >= 2:
        print(f"🔴 {판정['banner']}")
        return 2

    붉음: List[str] = []
    줄 = rows[0]
    개봉시각 = datetime.fromisoformat(줄["unsealed_at_kst"])
    try:
        사전시각 = commit_time(줄["preregistration_commit"])
    except subprocess.CalledProcessError:
        붉음.append(f"사전등록 커밋 {줄['preregistration_commit'][:7]} 을 git 에서 찾지 못했다")
    else:
        if not 사전시각 < 개봉시각:
            붉음.append(f"사전등록 커밋({사전시각.isoformat()})이 "
                      f"개봉({줄['unsealed_at_kst']})보다 앞서지 않는다 — ADR 0004 무효")
    for p in PREREGISTRATION_PATHS:
        c = last_commit_of(p) if (ROOT / p).exists() else None
        if c and c[1] > 개봉시각:
            붉음.append(f"개봉 뒤에 {p} 가 커밋됐다({c[1].isoformat()}) — 개봉 후 수정 금지")

    승인경로 = Path(args.authorization) if args.authorization else authorization_path_for(dry)
    승인 = None
    if 승인경로.exists():
        승인 = json.loads(승인경로.read_text(encoding="utf-8"))
        빠짐 = [k for k in CONTRACT_FIELDS if k not in 승인]
        if 빠짐:
            붉음.append(f"승인 파일에 실행기 계약 칸이 없다: {빠짐}")
        if 승인.get("snapshot_sha256") != 줄["snapshot_sha256"]:
            붉음.append("승인 파일의 스냅샷 지문이 개봉 기록과 다르다")
        for n, p in (승인.get("source_path") or {}).items():
            if Path(p).exists() and holdout.sha256_file(Path(p)) != 승인["source_sha256"].get(n):
                붉음.append(f"승인 뒤에 입력 {n} 이 바뀌었다")
    else:
        print(f"   승인 파일 {승인경로} 가 아직 없다.")

    if args.result_dir:
        결과 = Path(args.result_dir)
        보고 = 결과 / "report.json"
        if not 보고.exists():
            붉음.append(f"{보고} 가 없다 — 실행기 결과 폴더가 아니다")
        elif 승인 is not None:
            rep = json.loads(보고.read_text(encoding="utf-8"))
            if rep.get("run_id") != 승인.get("run_id"):
                붉음.append(f"결과 run_id({rep.get('run_id')})가 "
                          f"승인({승인.get('run_id')})과 다르다")
            for n in SOURCE_NAMES:
                결과지문 = ((rep.get("source") or {}).get(n) or {}).get("sha256")
                if 결과지문 != (승인.get("source_sha256") or {}).get(n):
                    붉음.append(f"결과의 입력 {n} 지문이 승인과 다르다")
        붉음.extend(scan_outputs(결과, secret_values()))

    for s in 붉음:
        print(f"🔴 {s}")
    if 붉음:
        return 2
    print("✅ 개봉 1회 · 사전등록이 앞섬 · 승인·결과·산출물 확인 — 확증 가능")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="홀드아웃 개봉 통제 — open · authorize · check")
    sub = p.add_subparsers(dest="command", required=True)

    o = sub.add_parser("open", help="사전 조건 확인 → 봉인 반출 → 개봉 기록 한 줄")
    o.add_argument("--dry-run", action="store_true")
    o.add_argument("--start", default=None,
                   help="홀드아웃 시작 (진짜는 정본 고정 · 드라이런만 바꾼다)")
    o.add_argument("--end", default=None,
                   help="끝 YYYYMMDD (기본: 진짜는 달력 끝 · 드라이런은 DEV_END)")
    o.add_argument("--out", default=None, help="봉인 반출 폴더 (기본 data/sealed/…)")
    o.add_argument("--requested-by", required=True)
    o.add_argument("--reason", required=True)

    a = sub.add_parser("authorize", help="입력 넷과 설정의 지문으로 승인 JSON 을 쓴다")
    a.add_argument("--dry-run", action="store_true")
    a.add_argument("--snapshot", required=True, help="open 이 만든 봉인 반출 폴더")
    a.add_argument("--run-id", required=True)
    a.add_argument("--config", default=str(DEFAULT_CONFIG))
    a.add_argument("--out", default=None)
    for n in SOURCE_NAMES:
        a.add_argument(f"--{n.replace('_', '-')}", dest=n, required=True)

    c = sub.add_parser("check", help="개봉 1회 · 사전등록 선행 · 승인·결과·산출물 확인")
    c.add_argument("--dry-run", action="store_true")
    c.add_argument("--authorization", default=None)
    c.add_argument("--result-dir", default=None)

    args = p.parse_args(argv)
    if args.command == "open" and not args.dry_run and args.start not in (None, HOLDOUT_START):
        print(f"🔴 진짜 개봉의 시작은 코드 정본({HOLDOUT_START})으로 고정이다.")
        return 1
    return {"open": cmd_open, "authorize": cmd_authorize, "check": cmd_check}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
