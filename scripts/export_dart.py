"""재무를 팀원에게 건네줄 수 있게 내보낸다 — **개발구간과 봉인구간을 나눠서**.

    python scripts/export_dart.py
    python scripts/export_dart.py --holdout-start 20250901   # 경계를 바꿔 본다
    python scripts/export_dart.py --dry-run

시세를 내보내는 `export_team_dataset.py` 와 짝이다. 저쪽이 안 다루는 재무만 맡는다.

## 🔴 왜 두 폴더로 나누는가

재무는 **결산기가 아니라 접수일**로 시점을 세운다. 그래서 "이 자료가 개발구간 것인가"
는 `bsns_year` 가 아니라 `rcept_dt` 로 갈린다 — 이걸 헷갈리면 **석 달치 미래**가
학습에 들어가고도 예외가 나지 않는다.

    dev/       접수일 <  holdout_start   학습·검증에 쓴다
    holdout/   접수일 >= holdout_start   봉인. 마지막에 딱 한 번 연다

한 덩어리로 올리면 팀원이 실수로 봉인 구간을 학습에 넣게 된다. 폴더가 갈라져 있으면
`dev/` 만 읽는 코드를 쓰는 것이 자연스러워진다.

## 경계를 인자로 받는 이유

`HOLDOUT_START` 의 정본은 `evaluation/horizon.py` 이고 그건 평가 파트 소유다.
그래서 **여기서 값을 정하지 않는다** — 기본값은 저쪽을 그대로 따르고, `--holdout-start`
로 다른 경우를 시험해 볼 수 있게만 해 둔다.

실측 2026-09-02 · 662,933행 기준으로 경계에 따라 이만큼 갈린다.

    20210901 (현재 · 5.0년 봉인)   학습 재무 300,991행 (45.4%) · 최신 FY2020
    20240901 (2.0년)               498,059행 (75.1%) · FY2023
    20250901 (1.0년)               577,311행 (87.1%) · FY2024
    20260301 (0.5년)               578,758행 (87.3%) · FY2024

1년 아래로 더 줄여도 재무는 거의 늘지 않는다 — FY2025 는 2026년 3월 접수라
그때도 안 들어온다. 근거는 이슈 #60 에 모아 두었다.

## parquet 로 내보내는 이유

실측(2026-08-31 · 시세 599만 행)으로 CSV 656.7MB → parquet+zstd 142.1MB 였다.
4.6배 작고 쓰는 것도 7배 빠르다. 정렬(`code, bas_dd`)까지 맞추면 8.1배까지 줄었다.
재무도 같은 방식으로 `(stock_code, bsns_year)` 정렬 후 zstd 로 쓴다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from common.paths import PROJECT_ROOT, krx_db_path  # noqa: E402
from common.trading_calendar import now_kst_iso, today_kst  # noqa: E402

#: 기본 봉인 시작일. **여기서 정하지 않고** 평가 파트의 값을 그대로 따른다.
try:
    from evaluation.horizon import HOLDOUT_START as DEFAULT_HOLDOUT_START
except ImportError:                                   # 평가 모듈이 없으면 보수적으로
    DEFAULT_HOLDOUT_START = "20210901"

OUTBOX = PROJECT_ROOT / "data" / "outbox"

#: 압축. zstd 가 gzip 과 크기는 같은데 29배 빠르다 (실측 2026-08-31).
COMPRESSION = "zstd"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(con: sqlite3.Connection, where: str, params: tuple) -> pd.DataFrame:
    """정렬해서 읽는다 — 같은 회사·연도가 붙어 있어야 압축이 잘 먹는다."""
    return pd.read_sql_query(
        f"SELECT * FROM dart_financial WHERE {where} "
        "ORDER BY stock_code, bsns_year, sj_div, ord",
        con, params=params,
    )


def write_parquet(frame: pd.DataFrame, path: Path, files: List[Dict], note: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression=COMPRESSION)
    files.append({
        "path": path.relative_to(OUTBOX).as_posix(),
        "rows": len(frame),
        "bytes": path.stat().st_size,
        # `upload_to_hf.build_dataset_card` 가 이 이름으로 읽는다.
        # 반출 스크립트끼리 MANIFEST 모양이 갈리면 카드 생성이 KeyError 로 죽는다.
        "size_mb": round(path.stat().st_size / 1024 / 1024, 2),
        "sha256": sha256_of(path),
        "note": note,
    })
    print(f"  {path.relative_to(OUTBOX).as_posix():<40} "
          f"{len(frame):>9,}행  {path.stat().st_size / 1024 / 1024:>7.1f} MB")


def _readme(m: Dict) -> str:
    """반출본에 함께 두는 설명. 받는 사람이 파일만 보고도 규칙을 알 수 있어야 한다."""
    return f"""# AlphaStack 재무 (DART 단일회사 전체계정)

> 생성 {m["generated_at"]} · 원천 `{m["source"]}`

## 🔴 학습에는 `dev/` 만 쓴다

    dev/financial.parquet        {m["rows_dev"]:,}행   접수일 <  {m["holdout_start"]}
    holdout/financial.parquet    {m["rows_holdout"]:,}행   접수일 >= {m["holdout_start"]}

`holdout/` 은 **봉인 구간**이다. 마지막 평가에 딱 한 번 연다. 한 번 본 구간은 다시
봉인되지 않으므로, 학습·검증·하이퍼파라미터 탐색 어디에도 쓰지 않는다.

## 🔴 시점은 결산기가 아니라 접수일이다

    rcept_dt   공시 접수일 — **이 숫자를 세상이 알게 된 날**
    bsns_year  결산기. 시계열 인덱스로 쓰면 안 된다

FY2020 사업보고서는 2021년 3월에 접수된다. `bsns_year` 로 자르면 **석 달치 미래**가
학습에 들어가고도 예외가 나지 않고 성능만 좋아진다.

as_of 날짜 D 로 볼 때는 `next_business_day(rcept_dt) > D` 인 행을 **행째** 가린다.
금액 칸만 가리면 "그 회사가 그날 보고서를 냈다" 는 사실이 남아 그 자체가 신호가 된다.

## 자료 범위

    사업연도   {m["years"][0] if m["years"] else "-"} ~ {m["years"][-1] if m["years"] else "-"}
    보고서     사업보고서(11011)
    종목       유니버스 350종 (코스피200 + 코스닥150)

⚠️ DART 는 **2015년 이전 전체계정을 주지 않는다.** 350종 × 2010~2020 을 전부 요청해
실측했고, 2010~2014 는 전부 빈 응답이었다.

## 읽는 법

```python
from huggingface_hub import hf_hub_download
import pandas as pd

path = hf_hub_download(
    repo_id="qurious-quant/alphastack-dart",
    filename="v6_with_receipt_date/dev/financial.parquet",
    repo_type="dataset",
)
df = pd.read_parquet(path)
```

## ⚠️ 이 자료는 private 다

OpenDART 이용 조건과 팀 내부 규약에 따라 조직 밖으로 내보내지 않는다.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="재무를 개발구간·봉인구간으로 나눠 내보낸다")
    parser.add_argument("--holdout-start", default=DEFAULT_HOLDOUT_START,
                        help=f"봉인 시작 접수일 YYYYMMDD (기본 {DEFAULT_HOLDOUT_START} — "
                             "evaluation/horizon.py 의 값)")
    parser.add_argument("--out", default=None, help="출력 폴더 (기본 data/outbox/dart_<오늘>)")
    parser.add_argument("--dry-run", action="store_true", help="크기만 재고 쓰지 않는다")
    args = parser.parse_args()

    경계 = args.holdout_start.strip()
    if len(경계) != 8 or not 경계.isdigit():
        print(f"🔴 봉인 시작일이 YYYYMMDD 여덟 자리가 아닙니다: {경계!r}")
        return 1

    db = krx_db_path()
    if not db.exists():
        print(f"🔴 DB 가 없습니다: {db}")
        print("   할 일: python scripts/fetch_dart.py 로 먼저 채우세요.")
        return 1

    root = Path(args.out) if args.out else OUTBOX / f"dart_{today_kst():%Y%m%d}"
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        전체 = con.execute("SELECT COUNT(*) FROM dart_financial").fetchone()[0]
        if not 전체:
            print("🔴 dart_financial 이 비어 있습니다 → python scripts/fetch_dart.py")
            return 1

        print(f"── 재무 반출 — 봉인 시작 {경계} ──")
        dev = load(con, "rcept_dt < ?", (경계,))
        holdout = load(con, "rcept_dt >= ?", (경계,))
    finally:
        con.close()

    print(f"  전량 {전체:,}행 → 개발 {len(dev):,} ({len(dev) / 전체 * 100:.1f}%) · "
          f"봉인 {len(holdout):,}")

    # 🔴 나누고 나서 **다시 센다.** 두 벌의 합이 전량과 다르면 어딘가로 샌 것이다.
    if len(dev) + len(holdout) != 전체:
        print(f"🔴 나눈 합이 전량과 다릅니다: {len(dev)} + {len(holdout)} != {전체}")
        print("   할 일: rcept_dt 가 비었거나 형식이 다른 행이 있는지 확인하세요.")
        return 1

    if args.dry_run:
        print("\n(예정만 계산했습니다 — 파일을 쓰지 않았습니다)")
        return 0

    files: List[Dict] = []
    print()
    write_parquet(dev, root / "dev" / "financial.parquet", files,
                  f"접수일 < {경계} — 학습·검증에 쓴다")
    write_parquet(holdout, root / "holdout" / "financial.parquet", files,
                  f"접수일 >= {경계} — 봉인. 마지막에 딱 한 번 연다")

    manifest = {
        "generated_at": now_kst_iso(),
        "source": "dart_financial (마이그레이션 v6)",
        "holdout_start": 경계,
        "holdout_start_default": DEFAULT_HOLDOUT_START,
        "rows_total": 전체,
        "rows_dev": len(dev),
        "rows_holdout": len(holdout),
        "years": sorted(int(y) for y in dev["bsns_year"].unique()) if len(dev) else [],
        "files": files,
        "note": (
            "🔴 학습에는 dev/ 만 쓴다. holdout/ 은 마지막 평가에 딱 한 번 연다. "
            "재무는 결산기가 아니라 접수일(rcept_dt)로 시점을 세운다 — "
            "bsns_year 로 자르면 석 달치 미래가 학습에 들어간다."
        ),
    }
    (root / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 받는 사람이 파일만 보고도 규칙을 알 수 있게 설명을 함께 둔다.
    # upload_to_hf 의 카드 생성기는 시세 반출을 전제하므로 --no-card 로 끄고
    # 이 README 를 그대로 올린다.
    (root / "README.md").write_text(_readme(manifest), encoding="utf-8")

    print(f"\n반출본: {root}")
    print(f"기록  : {(root / 'MANIFEST.json').relative_to(PROJECT_ROOT)}")
    print("\n올리려면:")
    print(f"  python scripts/upload_to_hf.py --path {root} --repo qurious-quant/<레포>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
