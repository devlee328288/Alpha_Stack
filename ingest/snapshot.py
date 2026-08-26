"""스냅샷 박제 — 실험이 딛고 선 자료를 못 박고, 해시로 증명한다 (요구사항 F-02).

왜 필요한가
----------
`data/krx_cache.db` 는 **계속 자란다.** 매일 하루치가 붙고, 백필을 돌리면 과거도
늘어난다. 그런데 결론은 *"어떤 자료로 냈는가"* 와 떼어 낼 수 없다. 9/13 발표에서
쓴 4,097거래일과 9/20 에 다시 돌린 4,120거래일은 **다른 실험**이다.

그래서 실험을 시작하기 전에 자료를 한 번 떠서 얼리고(freeze), 그 내용의 지문을
남긴다. 리포트에 그 지문이 실리고, 나중에 누가 재현하려 하면 지문부터 맞춰 본다.

⚠️ 파일 바이트가 아니라 **내용**을 해시한다
------------------------------------------
가장 먼저 떠오르는 방법은 parquet 파일의 SHA-256 을 재는 것인데, 그러면
**같은 자료인데 해시가 달라질 수 있다.** parquet 은 압축 방식·라이브러리 버전·쓰기
옵션에 따라 바이트가 달라지고, 파일 안에 `created_by` 같은 메타를 심기도 한다.

F-02 의 수용 기준은 *"다른 날 두 번 돌려도 두 산출 JSON 의 스냅샷 해시가 같다"* 이다.
파일 해시로는 그것을 **보장할 수 없다.** 그래서 자료를 정규 형태(canonical form)로
직렬화한 바이트를 해시한다 — 열 순서·행 순서·숫자 표기를 전부 고정한 문자열이다.

파일 해시(`parquet_sha256`)도 함께 적어 두되, **비교의 정본은 `sha256`(내용 해시)** 이다.

정규 형태 v1
-----------
    · 행은 (date, index_name) 오름차순
    · 열은 아래 `CANONICAL_COLUMNS` 순서 고정. 헤더도 해시에 포함한다
    · 결측은 NUL(0x00). **빈 문자열로 두면 안 된다** — `None` 과 `""` 가 같은 지문을
      갖게 되어, 서로 다른 자료를 같다고 말하게 된다
    · 실수는 `repr()` — 파이썬의 최단 왕복(shortest round-trip) 표기라 값이 같으면
      문자열도 같다. `f"{x:.6f}"` 같은 반올림은 값이 달라도 같은 문자열이 될 수 있다
    · 셀 구분 US(0x1f) · 행 구분 RS(0x1e) — 지수명에 든 공백·쉼표와 부딪히지 않는다
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ingest.store import krx_index

# 프로젝트 루트 (이 파일은 ingest/ 안에 있다)
ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = ROOT / "data" / "snapshots"
META_PATH = SNAPSHOT_DIR / "snapshot_meta.json"

# KST. 이 프로젝트의 모든 시각 표기는 한국 시간이다.
KST = timezone(timedelta(hours=9), name="KST")

# 정규 형태의 열 순서. **이 순서를 바꾸면 해시가 전부 바뀐다.**
# 바꿔야 한다면 HASH_METHOD 를 v2 로 올리고, 옛 스냅샷은 v1 로 남겨 둔다.
CANONICAL_COLUMNS: Sequence[str] = (
    "date", "index_name", "index_class",
    "open", "high", "low", "close",
    "change", "change_rate", "volume", "value", "market_cap",
)
HASH_METHOD = "canonical-v1"

CELL_SEP = "\x1f"   # US — Unit Separator
ROW_SEP = "\x1e"    # RS — Record Separator
# 결측 표식. 빈 문자열로 두면 **`None` 과 `""` 가 같은 해시를 낸다** — 서로 다른 자료가
# 같은 지문을 갖게 되므로 지문의 뜻이 사라진다. NUL 은 정상 텍스트에 나오지 않는다.
NULL_MARK = "\x00"


class SnapshotConflictError(RuntimeError):
    """이미 박제된 스냅샷과 내용이 다를 때 던진다.

    **덮어쓰지 않는다.** 박제의 요점은 "그때 그 자료"가 남는 것이다. 조용히 덮어쓰면
    리포트에 실린 해시와 파일이 갈라지고, 그 사실을 아무도 모른다.
    """


# ── 정규 직렬화 · 해시 ─────────────────────────────────────────────────────

def _cell(value: object) -> str:
    """값 하나를 정규 문자열로. **여기서 표기가 흔들리면 해시가 흔들린다.**"""
    if value is None:
        return NULL_MARK
    if isinstance(value, bool):          # bool 은 int 의 하위형이라 먼저 걸러야 한다
        return "1" if value else "0"
    if isinstance(value, float):
        # repr 은 최단 왕복 표기다. 1054.01 은 언제나 "1054.01" 이고,
        # 이 문자열을 float() 로 되읽으면 정확히 같은 값이 나온다.
        return repr(value)
    if isinstance(value, int):
        return str(value)
    return str(value)


def _sort_key(row: Dict) -> tuple:
    return (str(row.get("date") or ""), str(row.get("index_name") or ""))


def canonical_bytes(rows: Sequence[Dict]) -> bytes:
    """행 목록 → 정규 바이트열. 해시는 언제나 이것 위에서 잰다."""
    머리 = CELL_SEP.join(CANONICAL_COLUMNS)
    줄 = [
        CELL_SEP.join(_cell(row.get(col)) for col in CANONICAL_COLUMNS)
        for row in sorted(rows, key=_sort_key)
    ]
    # 헤더도 해시에 포함한다 — 열 순서가 바뀌면 해시도 바뀌어야 하기 때문이다
    return ROW_SEP.join([머리, *줄]).encode("utf-8")


def content_sha256(rows: Sequence[Dict]) -> str:
    """자료 내용의 지문. 파일 형식·라이브러리 버전과 무관하다."""
    return hashlib.sha256(canonical_bytes(rows)).hexdigest()


def file_sha256(path: Path) -> str:
    """파일 바이트의 지문. 참고용이며 비교의 정본이 아니다."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── 메타 파일 ──────────────────────────────────────────────────────────────

def load_meta() -> Dict:
    """`snapshot_meta.json` 을 읽는다. 없으면 빈 골격."""
    if not META_PATH.exists():
        return {"schema_version": 1, "hash_method": HASH_METHOD, "snapshots": {}}
    with open(META_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_meta(meta: Dict) -> None:
    """정렬·들여쓰기를 고정해 저장한다. git diff 가 읽히도록."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    with open(META_PATH, "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


# ── 박제 ───────────────────────────────────────────────────────────────────

def snapshot_name(index_name: str, end_iso: str) -> str:
    """`코스피 200` + `2026-08-25` → `kospi200_20260825`.

    파일 이름에 **오늘 날짜가 아니라 자료 종료일**을 쓴다. 오늘 날짜를 쓰면 같은 자료를
    다른 날 뜨기만 해도 파일이 갈라진다 — 박제의 반대다.
    """
    slug = index_name.replace(" ", "").lower()
    별칭 = {"코스피200": "kospi200", "코스닥150": "kosdaq150"}
    return f"{별칭.get(slug, slug)}_{end_iso.replace('-', '')}"


def freeze(index_name: str = "코스피 200",
           start: Optional[str] = None,
           end: Optional[str] = None,
           force: bool = False) -> Dict:
    """지수 시계열을 parquet 으로 박제하고 메타에 지문을 남긴다.

    Args:
        index_name: 지수명 (KRX 표기 그대로, 공백 포함)
        start:      시작일 `YYYYMMDD`. 생략하면 있는 자료의 처음부터
        end:        종료일 `YYYYMMDD`. 생략하면 DB 의 마지막 거래일
        force:      이미 있는 스냅샷과 내용이 달라도 덮어쓴다. **평소에는 쓰지 않는다**

    Returns:
        이 스냅샷의 메타 항목

    Raises:
        SnapshotConflictError: 같은 이름으로 이미 박제됐는데 내용 해시가 다를 때
    """
    import pandas as pd

    rows: List[Dict] = krx_index.series(index_name=index_name, start=start, end=end)
    if not rows:
        raise ValueError(
            f"'{index_name}' 자료가 없다 (start={start} end={end}).\n"
            f"  해결: python scripts/fetch_index.py 로 먼저 채운다."
        )

    해시 = content_sha256(rows)
    이름 = snapshot_name(index_name, str(rows[-1]["date"]))
    경로 = SNAPSHOT_DIR / f"{이름}.parquet"
    meta = load_meta()
    기존 = meta["snapshots"].get(이름)

    # 이미 박제됐는데 내용이 달라졌다면 멈춘다 — 조용히 덮어쓰면 재현이 불가능해진다
    if 기존 and 기존.get("sha256") != 해시 and not force:
        raise SnapshotConflictError(
            "\n".join([
                f"'{이름}' 은 이미 박제돼 있는데 내용이 다르다.",
                f"  박제된 해시: {기존.get('sha256')}",
                f"  지금  해시 : {해시}",
                f"  박제 시각  : {기존.get('frozen_at_kst')}  (행 {기존.get('rows')})",
                "  왜 막나: 리포트에 실린 해시와 파일이 갈라지면 재현이 불가능해진다.",
                "  해결 ①: 박제 시점 구간을 그대로 쓰려면 --end 를 그때 값으로 맞춘다",
                "  해결 ②: 새 구간을 쓰려는 것이면 --end 를 줘서 다른 이름이 되게 한다",
                "  해결 ③: 그래도 덮어써야 하면 --force. 이 선택은 ADR 에 적을 것",
            ])
        )

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)[list(CANONICAL_COLUMNS)].sort_values("date").reset_index(drop=True)
    df.to_parquet(경로, engine="pyarrow", compression="snappy", index=False)

    지금 = datetime.now(KST).isoformat(timespec="seconds")
    항목 = {
        "index_name": index_name,
        "path": str(경로.relative_to(ROOT)).replace("\\", "/"),
        "rows": len(rows),
        "columns": list(CANONICAL_COLUMNS),
        "start": str(rows[0]["date"]),
        "end": str(rows[-1]["date"]),
        "sha256": 해시,
        "hash_method": HASH_METHOD,
        "parquet_sha256": file_sha256(경로),
        "source": "KRX Open API · index_price",
        # 처음 박제한 시각은 **바꾸지 않는다.** 다시 돌렸다는 사실은 아래 칸에 남긴다
        "frozen_at_kst": (기존 or {}).get("frozen_at_kst", 지금),
        "last_verified_at_kst": 지금,
    }
    meta["snapshots"][이름] = 항목
    meta["hash_method"] = HASH_METHOD
    save_meta(meta)
    return 항목


def verify(name: Optional[str] = None) -> List[Dict]:
    """박제된 스냅샷의 지문이 지금도 맞는지 확인한다.

    parquet 을 되읽어 내용 해시를 다시 재고 메타와 대조한다.
    `name` 을 주면 그 하나만, 없으면 전부.
    """
    import pandas as pd

    meta = load_meta()
    대상 = [name] if name else sorted(meta["snapshots"])
    결과: List[Dict] = []
    for 키 in 대상:
        항목 = meta["snapshots"].get(키)
        if 항목 is None:
            결과.append({"name": 키, "ok": False, "reason": "메타에 없다"})
            continue
        경로 = ROOT / 항목["path"]
        if not 경로.exists():
            결과.append({"name": 키, "ok": False,
                        "reason": f"파일이 없다: {항목['path']}"})
            continue
        df = pd.read_parquet(경로, engine="pyarrow")
        # NaN → None. parquet 왕복에서 결측 표현이 바뀌므로 정규화한다
        rows = df.astype(object).where(df.notna(), None).to_dict(orient="records")
        지금해시 = content_sha256(rows)
        결과.append({
            "name": 키, "ok": 지금해시 == 항목["sha256"],
            "expected": 항목["sha256"], "actual": 지금해시,
            "rows": len(rows), "start": 항목["start"], "end": 항목["end"],
            "reason": None if 지금해시 == 항목["sha256"] else "내용 해시가 다르다",
        })
    return 결과
