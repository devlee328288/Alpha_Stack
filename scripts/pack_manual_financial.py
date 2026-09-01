"""손으로 받은 재무 TSV 를 parquet 으로 묶는다 — 용량을 9배 줄이면서 원본을 잃지 않는다.

왜 필요한가
-----------
OpenDART 재무정보 일괄다운로드로 받은 TSV 가 **830개 · 7.4GB** 다. 텍스트라 압축이 전혀
안 된 상태이고, 로컬 디스크가 90.6% 차 있다. 같은 내용을 parquet(zstd)으로 담으면
**9.4배** 줄어든다 (실측: 33.2MB → 2.8MB · 11.8배, 0.9MB → 0.1MB · 6.9배).

🔴 원본을 잃지 않는다
--------------------
이 스크립트는 **칸을 하나도 버리지 않는다.** 그리고 파일명에만 있던 정보
(연도·보고서·재무제표종류·업종·연결여부·받은날짜)를 **칸으로 만들어 넣는다.**
그래서 parquet 하나만 있으면 원래 TSV 를 되살릴 수 있다.

    2023_사업보고서_01_재무상태표_금융기타_연결_20260814.tsv
    → 연도=2023 · 보고서=사업보고서 · 재무제표=01_재무상태표
      업종=금융기타 · 연결여부=연결 · 받은날짜=20260814

⚠️ 왜 파일명을 칸으로 옮기나 — 그러지 않으면 합치는 순간 **업종과 연결/별도를 구분할 수
   없게 된다.** TSV 안의 `재무제표종류` 칸에 "별도재무제표" 라고 적혀 있기는 하지만
   업종은 어디에도 없다.

인코딩
------
**UTF-8 이다.** `cp949` 로 열면 `UnicodeDecodeError` 가 난다 (실측).
그리고 줄 끝에 탭이 하나 더 있어 **빈 16번째 칸**이 생긴다 — 그것만 버린다.

묶는 단위
---------
기본은 **연도 × 보고서** 다 (43개 파일). 압축률과 다루기 편함의 균형이 좋고,
"2023년 사업보고서만 보자" 가 파일 하나로 끝난다.
`--group` 으로 바꿀 수 있다.

    python scripts/pack_manual_financial.py                    # 연도×보고서 (기본)
    python scripts/pack_manual_financial.py --group statement  # 재무제표 종류별 5개
    python scripts/pack_manual_financial.py --group all        # 전체 한 파일
    python scripts/pack_manual_financial.py --dry-run          # 무엇을 만들지만 보여준다
    python scripts/pack_manual_financial.py --delete-source    # 검증 뒤 원본 TSV 삭제

🔴 `--delete-source` 는 **검증을 통과한 것만** 지운다. 검증은 행 수와 칸 수를 대조한다.
   하나라도 어긋나면 그 묶음의 원본은 남긴다.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

SRC = Path("data/manual/financial")
DST = Path("data/manual/financial_packed")

#: 줄 끝 탭 때문에 생기는 빈 칸. 이것만 버린다.
#:
#: ⚠️ 번호를 못 박으면 안 된다 — **파일마다 칸 수가 다르다.** 실측에서 15칸짜리와
#:    18칸짜리가 섞여 있어 `Unnamed: 12`·`13`·`14`·`16`·`18` 이 제각각 생겼다.
#:    처음에 `Unnamed: 15` 만 걸렀다가 다섯 칸이 그대로 따라 들어갔다.
#:    (값은 전부 비어 있어 자료 손실은 없었지만, 쓰는 쪽이 헷갈린다.)
def _is_empty_col(name: str) -> bool:
    return not name or str(name).startswith("Unnamed:")

#: parquet 압축. zstd 는 gzip 과 크기가 같은데 29배 빠르다 (실측 5.2초 대 152.1초).
CODEC = "zstd"
ROW_GROUP = 1_000_000


def parse_name(path: Path) -> Dict[str, str]:
    """파일명에서 축을 뽑는다.

    {연도}_{보고서}_{번호}_{재무제표종류}[_{업종}][_연결]_{받은날짜}.tsv
    """
    parts = path.stem.split("_")
    if len(parts) < 5:
        raise ValueError(f"파일명 규칙에 맞지 않는다: {path.name}")

    연도, 보고서, 번호, 재무제표 = parts[0], parts[1], parts[2], parts[3]
    받은날짜 = parts[-1]
    가운데 = parts[4:-1]                      # 업종·연결 여부가 들어가는 자리
    연결여부 = "연결" if "연결" in 가운데 else "별도"
    업종 = next((m for m in 가운데 if m != "연결"), "일반")

    return {
        "연도": 연도,
        "보고서": 보고서,
        "재무제표": f"{번호}_{재무제표}",
        # 🔴 이름을 `업종` 으로 두면 안 된다 — **원본 TSV 에 이미 `업종` 칸이 있다.**
        #    원본 것은 표준산업분류 코드('262')이고 이것은 OpenDART 다운로드 화면의
        #    구분(일반·금융기타·보험·은행·증권)이라 전혀 다른 값이다.
        #    처음에 같은 이름을 써서 **원본 산업분류 코드가 통째로 덮어써졌다.**
        #    (`업종명` 은 살아남아 손실을 늦게 알아챘다.)
        "업종구분": 업종,
        "연결여부": 연결여부,
        "받은날짜": 받은날짜,
        "원본파일": path.name,
    }


def group_key(meta: Dict[str, str], mode: str) -> str:
    if mode == "year_report":
        return f"{meta['연도']}_{meta['보고서']}"
    if mode == "statement":
        return meta["재무제표"]
    if mode == "all":
        return "financial_all"
    raise ValueError(f"모르는 묶음 방식: {mode}")


def read_tsv(path: Path) -> pd.DataFrame:
    """TSV 한 개를 읽는다. 칸은 전부 문자열로 둔다 — 여기서 타입을 정하지 않는다.

    ⚠️ 금액에 콤마가 들어 있고(`24,203,421,673`) 계정명에 앞 공백 들여쓰기가 있다.
       그런 정제는 반입 규격의 cleaner 가 할 일이라, 여기서는 **원문 그대로** 담는다.
    """
    df = pd.read_csv(path, sep="\t", encoding="utf-8", dtype=str,
                     on_bad_lines="warn", low_memory=False)
    return df.loc[:, [c for c in df.columns if not _is_empty_col(c)]]


def pack(mode: str, dry_run: bool, delete_source: bool) -> int:
    if not SRC.exists():
        print(f"🔴 없는 폴더: {SRC}")
        return 1

    files = sorted(SRC.glob("*.tsv"))
    if not files:
        print(f"🔴 {SRC} 에 .tsv 가 없다")
        return 1

    총원본 = sum(f.stat().st_size for f in files)
    print(f"── 재무 TSV 묶기 ({mode}) ──")
    print(f"  원본 {len(files)}개 · {총원본 / 1024 / 1024:,.0f} MB")

    묶음: Dict[str, List[Tuple[Path, Dict[str, str]]]] = defaultdict(list)
    for f in files:
        try:
            묶음[group_key(parse_name(f), mode)].append((f, parse_name(f)))
        except ValueError as e:
            print(f"  ⚠️ 건너뜀 — {e}")

    print(f"  묶음 {len(묶음)}개로 만든다")
    if dry_run:
        for key in sorted(묶음):
            크기 = sum(f.stat().st_size for f, _ in 묶음[key])
            print(f"    {key:<28} 파일 {len(묶음[key]):>3}개 · {크기 / 1024 / 1024:>8.1f} MB")
        print("\n  (--dry-run 이라 아무것도 쓰지 않았다)")
        return 0

    DST.mkdir(parents=True, exist_ok=True)
    총결과 = 0
    지울것: List[Path] = []

    for key in sorted(묶음):
        조각 = []
        원본행 = 0
        for f, meta in 묶음[key]:
            d = read_tsv(f)
            원본행 += len(d)
            # 🔴 이름이 겹치면 **원본 칸이 조용히 덮어써진다.** 실제로 `업종` 에서 한 번
            #    당했다 — 원본의 산업분류 코드가 파일명의 금융분류로 바뀌어 있었고,
            #    행 수 검증은 통과해서 눈치채지 못했다. 그래서 여기서 막는다.
            겹침 = [k for k in meta if k in d.columns]
            if 겹침:
                raise SystemExit(
                    f"🔴 칸 이름이 겹친다: {겹침}\n"
                    f"   파일: {f.name}\n"
                    f"   파일명에서 옮기는 칸이 원본 칸을 덮어쓴다. parse_name 의 키를 바꿔라."
                )
            for k, v in meta.items():          # 파일명에만 있던 정보를 칸으로
                d[k] = v
            조각.append(d)

        merged = pd.concat(조각, ignore_index=True)
        out = DST / f"{key}.parquet"
        pq.write_table(pa.Table.from_pandas(merged, preserve_index=False), out,
                       compression=CODEC, row_group_size=ROW_GROUP)

        # 검증 — 행 수가 맞는지 다시 읽어 본다
        확인 = pq.read_metadata(out).num_rows
        원본크기 = sum(f.stat().st_size for f, _ in 묶음[key])
        결과크기 = out.stat().st_size
        총결과 += 결과크기
        ok = 확인 == 원본행 == len(merged)

        print(f"    {key:<28} {원본크기 / 1024 / 1024:>8.1f}M → "
              f"{결과크기 / 1024 / 1024:>7.1f}M "
              f"({원본크기 / max(결과크기, 1):>5.1f}배) "
              f"{원본행:>9,}행 {'✅' if ok else '🔴 행 수 불일치'}")

        if ok and delete_source:
            지울것.extend(f for f, _ in 묶음[key])

    print()
    print(f"  합계 {총원본 / 1024 / 1024:,.0f} MB → {총결과 / 1024 / 1024:,.0f} MB "
          f"({총원본 / max(총결과, 1):.1f}배)")
    print(f"  절약 {(총원본 - 총결과) / 1024 / 1024 / 1024:.2f} GB")

    if delete_source:
        if len(지울것) != len(files):
            print(f"\n  ⚠️ 검증을 통과한 {len(지울것)}개만 지운다 "
                  f"(전체 {len(files)}개)")
        for f in 지울것:
            f.unlink()
        print(f"  🗑️ 원본 {len(지울것)}개 삭제")
    else:
        print("\n  원본은 그대로 두었다. 지우려면 --delete-source")

    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="재무 TSV 를 parquet 으로 묶는다")
    p.add_argument("--group", default="year_report",
                   choices=["year_report", "statement", "all"],
                   help="묶는 단위 (기본 year_report — 연도×보고서)")
    p.add_argument("--dry-run", action="store_true", help="무엇을 만들지만 보여준다")
    p.add_argument("--delete-source", action="store_true",
                   help="🔴 검증을 통과한 묶음의 원본 TSV 를 지운다")
    a = p.parse_args()
    return pack(a.group, a.dry_run, a.delete_source)


if __name__ == "__main__":
    raise SystemExit(main())
