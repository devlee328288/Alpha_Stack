"""HF 에 올라간 반출본이 지금 DB·지금 코드와 같은지 확인한다 — 재배포가 필요한가.

왜 이 스크립트가 필요한가
------------------------
반출본을 다시 만들어 SHA 를 대조하는 것이 가장 단순하지만, 그 길은 두 군데서 막힌다.

1. **재생성은 DB 에 쓴다.** `export_team_dataset.py` 는 `supply/` 를 거치고, 그 안의
   `init_db()` 가 스키마 마이그레이션을 실행한다. 확인만 하려는 작업이 공유 DB 를
   건드리는 것은 값이 너무 비싸다. 여기서는 **`mode=ro` 로만 연다.**
2. **SHA 는 한쪽으로만 강한 증거다.** `full/*.parquet` 은 `ORDER BY` 없이 뽑으므로 행
   순서가 보장되지 않고, CSV 는 float 를 적는 순간 끝자리가 흔들린다. 그래서 SHA 가
   같으면 확실히 같지만, **달라도 자료가 달라진 것은 아니다.**

그래서 이 스크립트는 **HF 파일을 받아 값으로 대조한다.** 그리고 차이가 나오면 그 크기를
ULP(그 자릿수에서 float64 가 표현할 수 있는 최소 간격)로 재서, **표기의 한계인지 자료의
변화인지 가른다.** 이 구분을 안 하면 "달라졌다" 는 답이 매번 참이 되어 쓸모가 없어진다.

무엇을 재는가
-----------
    1. HF 상태     private 인가 · 파일이 MANIFEST 와 맞는가
    2. 원시 자료   daily_price · index_price 를 DB 와 칸별로 (연도별로 끊어서)
    3. 파생 자료   표본 30종의 원시·학습용
    4. 표본 선정   지금 DB 로 다시 골라도 같은 30종인가
    5. 코드 영향   (--baseline) 같은 입력에 옛 코드와 지금 코드를 걸어 비교

5번이 이 스크립트의 핵심이다. 1~4 는 "DB 가 변했는가" 를 보고, 5 는 **"코드가 변해서
같은 DB 로도 다른 결과가 나오는가"** 를 본다. 둘 다 아니어야 재배포가 필요 없다.

쓰는 법
------
    python scripts/verify_hf_dataset.py
    python scripts/verify_hf_dataset.py --skip-download          # 이미 받아 뒀다면
    python scripts/verify_hf_dataset.py --baseline 5a52655       # 코드 영향까지

종료 코드는 **0 이면 재배포 불필요**, 2 면 필요, 1 이면 확인 자체를 못 했다는 뜻이다.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from supply.sector import attach_industry  # noqa: E402


def 오늘_as_of() -> str:
    """업종 스냅샷을 어느 시점 기준으로 볼 것인가 — 반출과 같은 값을 쓴다.

    `export_team_dataset.py` 가 `datetime.now()` 를 쓰므로 여기도 같다. 스냅샷은
    사람이 손으로 받은 것이라 오늘 이후로 늘지 않고, 두 쪽이 같은 날 돌면 같은 결과다.
    """
    return datetime.now().strftime("%Y%m%d")

DEFAULT_REPO = "qurious-quant/alphastack-krx-dev"

#: 이 값 이하의 차이는 **표기의 한계**로 본다. float 를 CSV 로 적었다 읽으면 마지막
#: 비트가 흔들리고, 표준편차·차분처럼 값을 빼는 계산에서는 그 흔들림이 몇십 배로
#: 번진다. 자료가 바뀐 것과 구분하려고 여유를 두되, 크게 두면 진짜 변화를 놓친다.
ULP_TOLERANCE = 4.0

#: 원시 자료를 한 번에 다 올리면 599만 행이라 메모리가 버겁다. 연도로 끊는다.
YEARS = range(2010, 2030)


# ══════════════════════════════════════════════════════════════════════════
# 공통
# ══════════════════════════════════════════════════════════════════════════
def db_path() -> str:
    """DB 경로. 워크트리에서는 `KRX_DB_PATH` 로 본 저장소 것을 가리킨다."""
    return os.environ.get("KRX_DB_PATH", str(ROOT / "data" / "krx_cache.db"))


def ro_connect() -> sqlite3.Connection:
    """**읽기 전용** 연결. 이 스크립트는 공유 DB 에 한 바이트도 쓰지 않는다.

    프로젝트의 `krx_store.connect()` 를 쓰지 않는 이유가 여기 있다. 그쪽은 쓰기 모드로
    열고 `journal_mode=WAL` 을 실행하며, 부르는 자리에 따라 마이그레이션까지 탄다.
    """
    return sqlite3.connect(f"file:{Path(db_path()).as_posix()}?mode=ro",
                           uri=True, timeout=60)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_export_module(tree: Path):
    """`export_team_dataset.py` 를 모듈로 읽는다 (`main()` 은 돌지 않는다).

    `tree` 를 바꾸면 **다른 시점의 코드**를 그대로 불러올 수 있다. `--baseline` 이
    이걸 이용해 옛 코드와 지금 코드를 같은 입력에 걸어 본다.
    """
    spec = importlib.util.spec_from_file_location(
        f"export_team_dataset_{abs(hash(str(tree)))}",
        tree / "scripts" / "export_team_dataset.py")
    mod = importlib.util.module_from_spec(spec)
    saved = list(sys.path)
    sys.path.insert(0, str(tree))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = saved
    return mod


# ══════════════════════════════════════════════════════════════════════════
# 값 비교 — 표기의 한계와 자료의 변화를 가른다
# ══════════════════════════════════════════════════════════════════════════
def compare_column(a: pd.Series, b: pd.Series) -> Optional[Dict]:
    """한 칸을 비교한다. 같으면 `None`, 다르면 차이를 설명하는 dict.

    수치는 **dtype 이 달라도(INTEGER↔float) 값이 같으면 같다**고 본다. 결측은 양쪽 다
    결측이면 같다 — `==` 만 쓰면 `NaN != NaN` 이라 멀쩡한 결측이 전부 차이로 잡힌다.

    문자열에서 **빈 문자열과 결측은 다른 것으로 본다.** 실제로 DB 의 빈 소속부가 CSV 를
    거치며 `NaN` 이 되는 일이 있었고, 그걸 같다고 처리하면 그 변화를 영영 못 본다.
    """
    수치 = pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b)
    if 수치:
        av = pd.to_numeric(a, errors="coerce").to_numpy(dtype="float64")
        bv = pd.to_numeric(b, errors="coerce").to_numpy(dtype="float64")
        다름 = ~((av == bv) | (np.isnan(av) & np.isnan(bv)))
        n = int(다름.sum())
        if not n:
            return None
        idx = np.where(다름)[0]
        결측엇갈림 = int((np.isnan(av[idx]) ^ np.isnan(bv[idx])).sum())
        둘다값 = idx[~(np.isnan(av[idx]) | np.isnan(bv[idx]))]
        if len(둘다값):
            절대차 = np.abs(av[둘다값] - bv[둘다값])
            # np.spacing 은 그 값 근처의 최소 간격을 준다 — 자릿수에 맞는 눈금
            눈금 = np.spacing(np.maximum(np.abs(av[둘다값]), 1e-300))
            최대ulp = float((절대차 / 눈금).max())
            최대상대 = float((절대차 / np.maximum(np.abs(av[둘다값]), 1e-300)).max())
        else:
            최대ulp = 최대상대 = float("inf")
        표기한계 = 최대ulp <= ULP_TOLERANCE and 결측엇갈림 == 0
    else:
        av = a.astype("string").fillna("\x00결측")
        bv = b.astype("string").fillna("\x00결측")
        n = int((av != bv).sum())
        if not n:
            return None
        결측엇갈림 = int((a.isna() ^ b.isna()).sum())
        최대ulp = 최대상대 = float("nan")
        # 결측 엇갈림만 있는 문자열 차이는 CSV 가 빈 칸을 결측으로 읽어서 생긴다
        표기한계 = 결측엇갈림 == n
    return {"행": n, "결측엇갈림": 결측엇갈림, "최대ulp": 최대ulp,
            "최대상대": 최대상대, "표기한계": 표기한계}


def compare_frames(left: pd.DataFrame, right: pd.DataFrame, keys: List[str],
                   name: str, *, only_common: bool = False) -> Dict:
    """두 표를 정렬해 칸별로 비교한다. 차이는 표기/실제로 나눠 돌려준다."""
    print(f"\n── {name} ──")
    공통 = sorted(set(left.columns) & set(right.columns))
    왼쪽만 = sorted(set(left.columns) - set(right.columns))
    오른쪽만 = sorted(set(right.columns) - set(left.columns))
    if not only_common and (왼쪽만 or 오른쪽만):
        print(f"  🔴 칸 구성이 다르다 — 왼쪽에만 {왼쪽만} · 오른쪽에만 {오른쪽만}")
        return {"같다": False, "이유": "칸 구성"}
    if 왼쪽만 or 오른쪽만:
        print(f"  공통 칸 {len(공통)}개만 본다 (왼쪽에만 {왼쪽만} · 오른쪽에만 {오른쪽만})")
    if len(left) != len(right):
        print(f"  🔴 행 수가 다르다 — {len(left):,} vs {len(right):,}")
        return {"같다": False, "이유": "행 수", "왼쪽": len(left), "오른쪽": len(right)}

    left = left.sort_values(keys, kind="mergesort").reset_index(drop=True)
    right = right.sort_values(keys, kind="mergesort").reset_index(drop=True)

    표기, 실제 = [], []
    for col in 공통:
        d = compare_column(left[col], right[col])
        if d:
            (표기 if d["표기한계"] else 실제).append({"칸": col, **d})

    if not 표기 and not 실제:
        print(f"  ✅ {len(left):,}행 × {len(공통)}칸 — 완전히 같다")
        return {"같다": True, "행": len(left), "칸": len(공통)}

    for it in sorted(표기, key=lambda x: -x["행"]):
        꼬리 = (f"최대 {it['최대ulp']:.1f} ULP" if np.isfinite(it["최대ulp"])
                else f"결측 표기 {it['결측엇갈림']:,}")
        print(f"  ⚪ {it['칸']:16s} {it['행']:>8,}행 · {꼬리} — 표기 한계")
    for it in sorted(실제, key=lambda x: -x["행"]):
        print(f"  🔴 {it['칸']:16s} {it['행']:>8,}행 ({it['행']/len(left):.2%}) "
              f"· 결측엇갈림 {it['결측엇갈림']:,} · 최대 {it['최대ulp']:.1f} ULP")

    return {"같다": not 실제, "행": len(left), "표기차": 표기, "실제차": 실제}


# ══════════════════════════════════════════════════════════════════════════
# 1. HF 에서 받아 온다
# ══════════════════════════════════════════════════════════════════════════
DATA_FILES = [
    "full/daily_price_dev.parquet",
    "full/index_price_dev.parquet",
    "small/index_all_dev.csv",
    "small/index_kospi200_dev.csv",
    "small/features_labels_kospi200_dev.csv",
    "small/stocks_sample30_raw_dev.csv",
    "small/stocks_sample30_train_dev.csv",
    "small/features_labels_stocks30_dev.csv",
    "small/sample_codes.json",
]


def fetch_snapshot(repo: str, dest: Path) -> Dict:
    """HF 배포본을 통째로 받고, MANIFEST 의 SHA-256 과 대조한다.

    이 대조가 먼저인 이유: 뒤의 모든 판정이 "받은 것이 HF 의 그 파일" 이라는 전제 위에
    선다. 전제가 깨진 채로 값을 비교하면 무엇을 비교한 것인지 알 수 없다.
    """
    from huggingface_hub import HfApi, hf_hub_download

    from ingest.clients import hf_data

    token, source = hf_data.load_hf_key()
    if not token:
        raise SystemExit(f"🔴 HF 토큰이 없다 ({source}) — .env 에 "
                         "HUGGINGFACE_ACCESS_TOKEN 을 넣는다")

    api = HfApi(token=token)
    info = api.repo_info(repo_id=repo, repo_type="dataset")
    print(f"── {repo} ──")
    print(f"  private     : {info.private}")
    print(f"  마지막 커밋 : {info.last_modified} (UTC)")
    if not info.private:
        raise SystemExit("🔴 이 저장소가 private 이 아니다 — KRX 이용약관 제11조 ② 위반이다")

    dest.mkdir(parents=True, exist_ok=True)
    for name in ["MANIFEST.json", *DATA_FILES]:
        cached = hf_hub_download(repo_id=repo, repo_type="dataset",
                                 filename=name, token=token)
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, target)      # 캐시는 링크라 실물로 복사한다
    print(f"  파일 {len(DATA_FILES) + 1}개 → {dest}")
    return json.loads((dest / "MANIFEST.json").read_text(encoding="utf-8"))


def verify_manifest(snap: Path, manifest: Dict) -> bool:
    기록 = {f["path"]: f for f in manifest.get("files", [])}
    틀린 = []
    for name in DATA_FILES:
        base = Path(name).name
        if base not in 기록:
            continue
        if sha256_of(snap / name) != 기록[base]["sha256"]:
            틀린.append(base)
    if 틀린:
        print(f"  🔴 MANIFEST 와 다른 파일: {틀린}")
        return False
    print(f"  ✅ MANIFEST 의 SHA-256 {len(기록)}건 전부 일치")
    return True


# ══════════════════════════════════════════════════════════════════════════
# 2. 원시 자료를 DB 와 대조
# ══════════════════════════════════════════════════════════════════════════
문자칸 = {"bas_dd": str, "code": str, "name": str, "market": str, "sector": str,
          "index_name": str, "index_class": str, "date": str}


def compare_raw(snap: Path, dev_end: str) -> bool:
    import pyarrow.parquet as pq

    모두 = True

    print("\n[2] 원시 자료 ↔ DB")
    hf = pd.read_parquet(snap / "full/index_price_dev.parquet")
    with ro_connect() as conn:
        db = pd.read_sql_query(
            "SELECT * FROM index_price WHERE bas_dd <= ?", conn, params=(dev_end,))
    모두 &= compare_frames(hf, db, ["bas_dd", "index_name"],
                           "index_price_dev.parquet")["같다"]
    del hf, db

    hf = pd.read_csv(snap / "small/index_all_dev.csv", dtype=문자칸)
    with ro_connect() as conn:
        db = pd.read_sql_query(
            "SELECT * FROM index_price WHERE bas_dd <= ?", conn, params=(dev_end,))
    모두 &= compare_frames(hf, db, ["bas_dd", "index_name"],
                           "index_all_dev.csv", only_common=True)["같다"]
    del hf, db

    # daily_price 는 599만 행이라 연도로 끊는다
    print("\n  daily_price_dev.parquet — 연도별")
    총 = 0
    for year in YEARS:
        lo, hi = f"{year}0101", min(f"{year}1231", dev_end)
        if lo > dev_end:
            break
        hf = pq.read_table(snap / "full/daily_price_dev.parquet",
                           filters=[("bas_dd", ">=", lo), ("bas_dd", "<=", hi)]
                           ).to_pandas()
        if hf.empty:
            continue
        with ro_connect() as conn:
            db = pd.read_sql_query(
                "SELECT * FROM daily_price WHERE bas_dd BETWEEN ? AND ?",
                conn, params=(lo, hi))
        # 🔴 업종 세 칸은 `daily_price` 에 없다 — 반출이 업종 스냅샷에서 붙인다.
        #    여기서 같은 규칙으로 붙이지 않으면 "칸 구성이 다르다" 로 영원히 붉다.
        #    2026-09-05 에 실제로 겪었다: 업종 칸을 올리고 나서도 재배포 필요가
        #    나왔는데, 원인은 배포본이 아니라 **판정기가 반출을 안 따라간 것**이었다.
        db = attach_industry(db, as_of=오늘_as_of())
        총 += len(hf)
        모두 &= compare_frames(hf, db, ["bas_dd", "code"], f"{year}년")["같다"]
        del hf, db
    print(f"\n  daily_price 합계 {총:,}행")
    return 모두


def compare_samples(snap: Path, dev_end: str) -> bool:
    """표본 30종의 원시·학습용을 본다."""
    모두 = True
    print("\n[3] 표본 30종 ↔ DB")
    raw = pd.read_csv(snap / "small/stocks_sample30_raw_dev.csv", dtype=문자칸)
    codes = sorted(raw["code"].unique())
    with ro_connect() as conn:
        db = pd.read_sql_query(
            "SELECT * FROM daily_price WHERE bas_dd <= ? AND code IN "
            f"({','.join('?' * len(codes))})", conn, params=(dev_end, *codes))
    모두 &= compare_frames(raw, db, ["code", "bas_dd"],
                           "stocks_sample30_raw_dev.csv", only_common=True)["같다"]

    tr = pd.read_csv(snap / "small/stocks_sample30_train_dev.csv", dtype=문자칸)
    밖 = (set(zip(tr["code"], tr["bas_dd"], strict=True))
         - set(zip(raw["code"], raw["bas_dd"], strict=True)))
    print(f"\n  학습용은 원시의 부분집합인가 — 밖에 있는 행 {len(밖):,}"
          + ("  ✅" if not 밖 else "  🔴"))
    모두 &= not 밖
    겹침 = raw.merge(tr[["code", "bas_dd"]], on=["code", "bas_dd"], how="inner")
    모두 &= compare_frames(tr, 겹침, ["code", "bas_dd"],
                           "학습용 ↔ 원시(같은 키만)")["같다"]
    return 모두


def compare_sample_selection(manifest: Dict) -> bool:
    """지금 DB 로 표본을 다시 골라도 같은 30종이 나오는가.

    `pick_sample_codes` 는 `conn` 을 받아 SELECT 만 하므로 읽기 전용 연결로 재현된다 —
    `supply/` 를 거치지 않아 마이그레이션에 막히지 않는다.
    """
    print("\n[4] 표본 선정 재현")
    ex = load_export_module(ROOT)
    with ro_connect() as conn:
        codes, meta = ex.pick_sample_codes(conn)
    기록 = manifest.get("stats", {}).get("stocks30", {})
    hf_codes = sorted(기록.get("codes", []))
    같다 = codes == hf_codes
    print(f"  지금 {len(codes)}종 · HF {len(hf_codes)}종 · "
          + ("✅ 같다" if 같다 else "🔴 다르다"))
    if not 같다:
        print(f"    지금에만 {sorted(set(codes) - set(hf_codes))}")
        print(f"    HF 에만  {sorted(set(hf_codes) - set(codes))}")
    선정 = 기록.get("선정", {})
    for 키 in ("기준일", "층화_모집단"):
        if 키 in 선정:
            일치 = str(meta.get(키)) == str(선정[키])
            print(f"  {키}: 지금 {meta.get(키)} · HF {선정[키]} "
                  + ("✅" if 일치 else "🔴"))
            같다 &= 일치
    return 같다


# ══════════════════════════════════════════════════════════════════════════
# 5. 코드가 변해서 결과가 달라지는가
# ══════════════════════════════════════════════════════════════════════════
def _recompute(tree: Path, snap: Path, dev_end: str) -> Dict[str, pd.DataFrame]:
    """HF 의 중간 산출물을 입력으로 피처를 다시 만든다.

    입력을 HF 파일에서 가져오므로 **DB 를 거치지 않는다.** 거래일 달력만 읽기 전용
    SQL 로 뽑는다. 그래서 `tree` 만 바꿔 부르면 변수가 코드 하나로 좁혀진다.
    """
    ex = load_export_module(tree)
    with ro_connect() as conn:
        cal = [r[0] for r in conn.execute(
            "SELECT DISTINCT bas_dd FROM daily_price WHERE bas_dd <= ? ORDER BY bas_dd",
            (dev_end,))]
    day_index = ex.trading_day_index(cal)

    idx_in = pd.read_csv(snap / "small/index_kospi200_dev.csv", dtype=문자칸)
    feat, _ = ex.build_feature_frame(idx_in, band=ex.BAND_INDEX, day_index=day_index)
    kospi = ex.ready_to_fit(feat)

    tr = pd.read_csv(snap / "small/stocks_sample30_train_dev.csv", dtype=문자칸)
    parts = []
    for code in dict.fromkeys(tr["code"].tolist()):
        one = tr[tr["code"] == code].reset_index(drop=True)
        f, _ = ex.build_feature_frame(one, band=ex.BAND_STOCK, day_index=day_index)
        r = ex.ready_to_fit(f)
        if not r.empty:
            parts.append(r)
    return {"kospi200": kospi,
            "stocks30": pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()}


def compare_code_effect(baseline: str, snap: Path, dev_end: str) -> bool:
    """같은 입력에 옛 코드와 지금 코드를 걸어 결과를 비교한다.

    HF 파일과 직접 비교하지 않는 이유가 있다. HF 의 값은 *계산 → CSV 쓰기 → 읽기* 를
    거쳤고 재계산은 *CSV 읽기 → 계산* 만 거쳐, 그대로 대면 **표기 오차와 코드 영향이
    섞인다.** 양쪽 다 재계산으로 맞추면 변수는 코드 하나만 남는다.
    """
    print(f"\n[5] 코드 영향 — {baseline} ↔ 지금")
    with tempfile.TemporaryDirectory(prefix="verify_hf_") as tmp:
        옛트리 = Path(tmp) / "tree"
        옛트리.mkdir()
        tar = subprocess.run(["git", "archive", baseline], cwd=ROOT,
                             capture_output=True, check=True).stdout
        # 외부 `tar` 를 부르지 않는다 — Windows 의 것은 stdin 을 읽으려면 `-f -` 가
        # 필요해 플랫폼마다 인자가 갈린다. 표준 모듈이면 그런 갈림이 없다.
        with tarfile.open(fileobj=io.BytesIO(tar)) as tf:
            tf.extractall(옛트리, filter="data")
        print(f"  {baseline} 트리를 풀었다 ({len(tar) / 1024 / 1024:.1f} MB)")

        옛 = _recompute(옛트리, snap, dev_end)
        지금 = _recompute(ROOT, snap, dev_end)

    모두 = True
    for 이름, keys in (("kospi200", ["bas_dd"]), ("stocks30", ["code", "bas_dd"])):
        r = compare_frames(옛[이름], 지금[이름], keys, f"{이름} (옛 코드 ↔ 지금 코드)")
        # 여기서는 표기 차이도 실제 차이다 — 둘 다 메모리 값이라 CSV 를 안 거친다
        모두 &= r["같다"] and not r.get("표기차")
    return 모두


# ══════════════════════════════════════════════════════════════════════════
def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="HF 반출본이 지금 DB·코드와 같은지 확인한다")
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--snapshot", default="data/outbox/hf_snapshot",
                   help="내려받아 둘 곳 (gitignore 되는 자리여야 한다)")
    p.add_argument("--skip-download", action="store_true",
                   help="이미 받아 둔 스냅샷을 그대로 쓴다")
    p.add_argument("--baseline", default=None,
                   help="이 커밋의 코드와 지금 코드를 비교한다 (예: 5a52655)")
    args = p.parse_args(argv)

    snap = Path(args.snapshot)
    print(f"DB(읽기전용): {db_path()}")

    print("\n[1] HF 상태")
    if args.skip_download:
        manifest = json.loads((snap / "MANIFEST.json").read_text(encoding="utf-8"))
        print(f"  받아 둔 스냅샷을 쓴다: {snap}")
    else:
        manifest = fetch_snapshot(args.repo, snap)
    if not verify_manifest(snap, manifest):
        return 1
    dev_end = manifest.get("dev_end", "20210831")
    print(f"  개발구간 상한 {dev_end} · 생성 {manifest.get('generated_at')}")

    결과 = {
        "원시": compare_raw(snap, dev_end),
        "표본": compare_samples(snap, dev_end),
        "선정": compare_sample_selection(manifest),
    }
    if args.baseline:
        결과["코드"] = compare_code_effect(args.baseline, snap, dev_end)

    print("\n" + "=" * 66)
    for k, v in 결과.items():
        print(f"  {k}: {'✅ 같다' if v else '🔴 다르다'}")
    if all(결과.values()):
        print("\n  ✅ 재배포가 필요 없다 — HF 배포본은 지금 DB·코드와 같다")
        print("=" * 66)
        return 0
    print("\n  🔴 재배포가 필요하다")
    print("=" * 66)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
