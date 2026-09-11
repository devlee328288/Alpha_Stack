"""홀드아웃 — 봉인을 **한 번만**, 기록을 남기고 연다.

## 왜 공급 계층에 있나

홀드아웃 행을 읽는 길이 여럿이면 언젠가 기록 없이 한 번 더 열린다. 그리고 두 번 연 홀드아웃은
개발구간이 된다 — 결과를 보고 무엇이든 한 번 고치는 순간 확증이 아니라 탐색이다(ADR 0004 §9).

그래서 봉인 구간을 담은 반출본은 **이 문 하나로만** 읽게 한다. 문은 `reports/unseal.log` 에
**정확히 한 줄**이 있고, 그 줄의 스냅샷 지문이 **지금 여는 반출본과 같을 때만** 열린다.
개발본을 읽는 `supply.hf_model_data` 는 홀드아웃 행이 있으면 거부하고, 이 모듈은 개봉 기록이
없으면 거부한다 — 두 문이 서로의 반대편을 막는다. (ADR 0004 §1 · ADR 0007 · 이슈 #240)

## `unseal.log` 형식 — ADR 0004 §1 이 못 박았다

    # reports/unseal.log — 한 줄 = 한 번의 개봉. 행이 2개가 되는 순간 프로젝트는 실패로 보고한다.
    unsealed_at_kst | git_commit | snapshot_sha256 | preregistration_commit | requested_by | reason

| 데이터 행 | 판정 |
|---|---|
| 0 | 미개봉 |
| 1 | 개봉 1회 — 확증 가능 |
| 2 이상 | **확증 불가** — 리포트 표지에 "홀드아웃 {n}회 개봉 — 확증 불가" |

두 번째 줄은 **쓰지 못하게 막는다**(`append_unseal_row`). 누가 손으로 적어 2줄이 되면 읽는 쪽이
확증 불가를 알린다(`unseal_verdict`) — 막는 것과 알리는 것을 둘 다 둔다.

⚠️ `*.log` 는 gitignore 대상이지만 이 파일만은 **커밋한다**(`.gitignore` 예외). ADR 0004 의
   증거가 "사전등록 커밋 시각이 이 파일 첫 행보다 앞선다" 이고, 그 대조는 git 이력으로만 된다.

## 드라이런은 기록을 따로 둔다

실제 홀드아웃을 읽지 않고 개발구간 끝을 가짜 홀드아웃으로 삼아 같은 절차를 끝까지 돈다.
기록은 `reports/dryrun/unseal.log` 에 남긴다 — 드라이런이 진짜 기록에 한 줄을 채우면
**진짜 개봉이 두 번째가 된다.** 드라이런 반출본은 진짜 기록으로 열리지 않고, 진짜 반출본은
드라이런 기록으로 열리지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from common.trading_calendar import KST
from evaluation.horizon import HOLDOUT_START

ROOT = Path(__file__).resolve().parent.parent

#: 진짜 개봉 기록. 커밋한다.
UNSEAL_LOG = ROOT / "reports" / "unseal.log"

#: 드라이런 기록. 커밋하지 않는다(gitignore).
DRY_RUN_UNSEAL_LOG = ROOT / "reports" / "dryrun" / "unseal.log"

#: ADR 0004 §1 의 머리줄과 칸 — 글자 그대로 옮긴다.
UNSEAL_HEADER = ("# reports/unseal.log — 한 줄 = 한 번의 개봉. "
                 "행이 2개가 되는 순간 프로젝트는 실패로 보고한다.")
UNSEAL_FIELDS = ("unsealed_at_kst", "git_commit", "snapshot_sha256",
                 "preregistration_commit", "requested_by", "reason")

#: 봉인 반출본의 대장과 파일. `scripts/export_holdout_dataset.py` 가 만든다.
HOLDOUT_MANIFEST = "MANIFEST_holdout.json"
HOLDOUT_DAILY = "full/daily_price_all.parquet"
HOLDOUT_INDEX = "full/index_price_all.parquet"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{7,40}$")


class UnsealError(RuntimeError):
    """개봉 기록이 없거나 어긋나서 봉인 구간을 열 수 없다. 무엇을 해야 하는지까지 담는다."""


def sha256_file(path: Path) -> str:
    """파일 지문. 큰 parquet 도 메모리에 다 올리지 않는다."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ==================================================
# 1. 개봉 기록 — 읽기 · 판정 · 한 번만 쓰기
# ==================================================
def read_unseal_log(path: Path = UNSEAL_LOG) -> List[Dict[str, str]]:
    """데이터 행만 읽는다. `#` 로 시작하는 줄과 빈 줄은 건너뛴다.

    칸 수가 ADR 형식과 다르면 세운다 — 조용히 건너뛰면 "0줄이니 미개봉" 으로 읽힌다.
    """
    path = Path(path)
    if not path.exists():
        return []
    rows: List[Dict[str, str]] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = [p.strip() for p in s.split("|")]
        if len(parts) != len(UNSEAL_FIELDS):
            raise UnsealError(
                f"{path}:{n} 의 칸 수가 {len(parts)} 다 — "
                f"ADR 0004 형식은 {len(UNSEAL_FIELDS)} 칸이다.\n"
                "  할 일: 이 줄을 지우지 말고 사람이 확인한다. "
                "기록을 고치는 것 자체가 개봉 이력이다."
            )
        rows.append(dict(zip(UNSEAL_FIELDS, parts, strict=True)))
    return rows


def unseal_verdict(rows: Sequence[Dict[str, str]]) -> Dict[str, object]:
    """기록 행 수로 판정한다. 리포트 표지·`check` 가 이 값을 그대로 쓴다."""
    n = len(rows)
    if n == 0:
        return {"rows": 0, "status": "미개봉", "confirmable": False, "banner": ""}
    if n == 1:
        return {"rows": 1, "status": "개봉 1회", "confirmable": True, "banner": ""}
    return {"rows": n, "status": "확증 불가", "confirmable": False,
            "banner": f"홀드아웃 {n}회 개봉 — 확증 불가"}


def build_unseal_row(*, git_commit: str, snapshot_sha256: str, preregistration_commit: str,
                     requested_by: str, reason: str,
                     now: Optional[datetime] = None) -> Dict[str, str]:
    """ADR 형식의 한 줄을 만든다. 값에 칸 구분자(`|`)나 줄바꿈이 있으면 세운다."""
    moment = now or datetime.now(KST)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=KST)
    row = {
        "unsealed_at_kst": moment.astimezone(KST).isoformat(timespec="seconds"),
        "git_commit": git_commit.strip(),
        "snapshot_sha256": snapshot_sha256.strip(),
        "preregistration_commit": preregistration_commit.strip(),
        "requested_by": requested_by.strip(),
        "reason": reason.strip(),
    }
    for 칸, 값 in row.items():
        if not 값:
            raise UnsealError(f"개봉 기록의 '{칸}' 이 비어 있다.")
        if "|" in 값:
            raise UnsealError(f"개봉 기록의 '{칸}' 에 칸 구분자(|)가 있다: {값!r}")
        if "\n" in 값 or "\r" in 값:
            raise UnsealError(f"개봉 기록의 '{칸}' 에 줄바꿈이 있다: {값!r}")
    if not _SHA256.match(row["snapshot_sha256"]):
        raise UnsealError("snapshot_sha256 은 소문자 16진수 64자여야 한다.")
    for 칸 in ("git_commit", "preregistration_commit"):
        if not _COMMIT.match(row[칸]):
            raise UnsealError(f"'{칸}' 은 git 커밋 해시여야 한다: {row[칸]!r}")
    return row


def append_unseal_row(row: Dict[str, str], path: Path = UNSEAL_LOG) -> Path:
    """개봉 기록을 **한 번만** 쓴다. 이미 한 줄이라도 있으면 쓰지 않고 세운다.

    🔴 두 번째 줄이 생기는 순간 주 검정은 무효다(ADR 0004 §9). 그래서 쓰는 쪽에서 막는다 —
       "쓰고 나서 알린다" 로는 되돌릴 수 없다.
    """
    path = Path(path)
    기존 = read_unseal_log(path)
    if 기존:
        판정 = unseal_verdict([*기존, row])
        raise UnsealError(
            f"🔴 이미 개봉됐다 — {path} 에 {len(기존)}줄이 있다 "
            f"(첫 개봉 {기존[0]['unsealed_at_kst']} · {기존[0]['requested_by']}).\n"
            f"  두 번째 줄을 쓰면 '{판정['banner']}' 이다. 쓰지 않았다.\n"
            "  할 일: 같은 반출본을 다시 쓰는 것이면 기록 없이 load_holdout_snapshot 으로 연다.\n"
            "        정말 다시 열어야 한다면 사람이 사유를 적어 직접 추가하고\n"
            "        확증 불가로 보고한다."
        )
    missing = set(UNSEAL_FIELDS) - set(row)
    if missing:
        raise UnsealError(f"개봉 기록에 칸이 없다: {sorted(missing)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    # ADR 0004 §검증은 `head -2` 가 "주석 1줄 + 데이터 1줄" 이라고 기대한다. 칸 이름 줄을
    # 따로 두지 않고 머리 주석 한 줄만 쓴다 — 칸 순서는 `UNSEAL_FIELDS` 와 ADR 이 정본이다.
    이미있음 = path.exists() and path.read_text(encoding="utf-8").strip()
    머리 = "" if 이미있음 else UNSEAL_HEADER + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(머리 + " | ".join(row[k] for k in UNSEAL_FIELDS) + "\n")
    return path


# ==================================================
# 2. 봉인 반출본 — 기록과 지문이 맞을 때만 연다
# ==================================================
@dataclass(frozen=True)
class HoldoutSnapshot:
    """개봉 기록과 대조를 통과한 봉인 반출본."""

    daily: pd.DataFrame
    index: pd.DataFrame
    manifest: Dict[str, object]
    snapshot_sha256: str
    unseal: Dict[str, str]


def verify_holdout_snapshot(root: Path, *, log_path: Path = UNSEAL_LOG) -> Dict[str, object]:
    """열기 전에 대조만 한다. 통과하면 대장·지문·개봉 행을 돌려주고, 아니면 세운다."""
    root = Path(root)
    대장경로 = root / HOLDOUT_MANIFEST
    if not 대장경로.exists():
        raise UnsealError(
            f"{대장경로} 가 없다 — 봉인 반출본이 아니다.\n"
            "  할 일: python scripts/unseal_holdout.py open 이 반출과 기록을 함께 만든다."
        )
    manifest = json.loads(대장경로.read_text(encoding="utf-8"))
    지문 = sha256_file(대장경로)

    드라이런 = bool(manifest.get("dry_run"))
    기록 = Path(log_path).resolve()
    if 드라이런 and 기록 == UNSEAL_LOG.resolve():
        raise UnsealError("드라이런 반출본을 진짜 개봉 기록으로 열 수 없다 — 기록이 섞인다.")
    if not 드라이런 and 기록 == DRY_RUN_UNSEAL_LOG.resolve():
        raise UnsealError("진짜 반출본을 드라이런 기록으로 열 수 없다 — 기록이 섞인다.")
    if not 드라이런 and str(manifest.get("holdout_start")) != HOLDOUT_START:
        raise UnsealError(
            f"반출본의 홀드아웃 시작({manifest.get('holdout_start')})이 코드 정본"
            f"({HOLDOUT_START})과 다르다.")

    rows = read_unseal_log(log_path)
    판정 = unseal_verdict(rows)
    if 판정["rows"] == 0:
        raise UnsealError(
            f"{log_path} 에 개봉 기록이 없다 — 기록 없이 봉인 구간을 읽지 않는다.\n"
            "  할 일: python scripts/unseal_holdout.py open (드라이런은 --dry-run)"
        )
    if not 판정["confirmable"]:
        raise UnsealError(f"🔴 {판정['banner']} — {log_path} 에 {판정['rows']}줄이 있다.")
    if rows[0]["snapshot_sha256"] != 지문:
        raise UnsealError(
            "개봉 기록의 스냅샷 지문이 이 반출본과 다르다.\n"
            f"  기록 {rows[0]['snapshot_sha256'][:16]}… · 반출본 {지문[:16]}…\n"
            "  기록한 뒤 반출본이 바뀌었거나 다른 폴더를 열었다. 다시 반출하면 두 번째 개봉이다."
        )

    for 항목 in manifest.get("files", []):
        경로 = root / str(항목["path"])
        if not 경로.exists():
            raise UnsealError(f"대장에 있는 파일이 없다: {경로}")
        if sha256_file(경로) != 항목.get("sha256"):
            raise UnsealError(f"{항목['path']} 의 SHA-256 이 대장과 다르다 — 반출본이 바뀌었다.")
    return {"manifest": manifest, "snapshot_sha256": 지문, "unseal": rows[0]}


def load_holdout_snapshot(root: Path, *, log_path: Path = UNSEAL_LOG,
                          daily_columns: Optional[Sequence[str]] = None) -> HoldoutSnapshot:
    """봉인 반출본을 연다 — 최종모델 공식 실행이 읽는 **유일한** 홀드아웃 입구.

    `supply.hf_model_data.load_hf_market_prices` 의 짝이다. 저쪽은 홀드아웃 행이 있으면 거부하고,
    이쪽은 개봉 기록이 정확히 한 줄이고 지문이 맞을 때만 연다. 칸 구성은 개발본 반출과 같다
    (`scripts/export_team_dataset.build_full_daily` 를 같이 쓴다).
    """
    확인 = verify_holdout_snapshot(root, log_path=log_path)
    root = Path(root)
    daily = pd.read_parquet(root / HOLDOUT_DAILY,
                            columns=list(daily_columns) if daily_columns else None)
    index = pd.read_parquet(root / HOLDOUT_INDEX)
    return HoldoutSnapshot(daily=daily, index=index, manifest=확인["manifest"],
                           snapshot_sha256=str(확인["snapshot_sha256"]),
                           unseal=dict(확인["unseal"]))


__all__ = [
    "DRY_RUN_UNSEAL_LOG",
    "HOLDOUT_DAILY",
    "HOLDOUT_INDEX",
    "HOLDOUT_MANIFEST",
    "UNSEAL_FIELDS",
    "UNSEAL_HEADER",
    "UNSEAL_LOG",
    "HoldoutSnapshot",
    "UnsealError",
    "append_unseal_row",
    "build_unseal_row",
    "load_holdout_snapshot",
    "read_unseal_log",
    "sha256_file",
    "unseal_verdict",
    "verify_holdout_snapshot",
]
