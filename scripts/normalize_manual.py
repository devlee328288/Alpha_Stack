"""손으로 받아 온 자료를 **UTF-8 로 맞추고, 이름만 봐도 알 수 있게** 바꾼다.

    python scripts/normalize_manual.py --dry-run   # 무엇을 할지만 보여준다
    python scripts/normalize_manual.py             # 실제로 변환한다
    python scripts/normalize_manual.py --force     # 이미 변환한 것도 다시 한다

원본은 **건드리지 않는다.** 변환본을 `data/manual/normalized/` 아래에 새로 쓴다.
손으로 받은 자료는 다시 받으려면 사람이 브라우저를 열어야 하므로, 변환이 잘못돼도
되돌릴 수 있어야 한다.

## 왜 필요한가 — 두 가지 문제

**① 인코딩.** KRX 정보데이터시스템이 주는 CSV 는 `cp949` 다. 편집기나 pandas 가
기본값(UTF-8)으로 열면 한글이 깨지거나 `UnicodeDecodeError` 로 죽는다. 실측
2026-09-02: `투자자별_거래실적/` 35개 파일 전부 cp949 였다.

**② 이름.** 같은 화면이 `data_4907_20260902.csv` · `data_4916_20260902.csv` 처럼
**KRX 일련번호**로 파일을 떨어뜨린다. 번호는 내용과 아무 관계가 없고, 30개 파일이
머리글까지 똑같아서 열어 봐도 무엇이 무엇인지 알 수 없다.

    일자,금융투자,보험,투신,사모,은행,기타금융,연기금 등,기타법인,개인,외국인,...

여섯 갈래(거래량·거래대금 × 매도·매수·순매수)가 전부 이 모양이다.

## 이름을 어떻게 되찾는가 — 값으로 판별한다

파일에 없는 정보를 지어내지 않는다. **값에서 확인할 수 있는 것만** 쓴다.

1. **기간** — 시계열 파일에는 `일자` 칸이 있다. 그대로 읽는다.
2. **순매수** — 매도·매수는 언제나 0 이상이다. **음수가 있으면 순매수다.**
3. **거래량 / 거래대금** — 같은 날 같은 투자자의 대금이 수량보다 크다. 규모로 가른다.
4. **매도 / 매수** — 둘은 생김새가 같아 구별되지 않는다. 그래서 **검산한다** —
   `순매수 = 매수 − 매도` 가 성립하는 배치가 정답이다.

실측 2026-09-02: 5개 묶음 전부 **검산 100.00%** 로 확정됐다. 이 검산은 항등식이
아니다 — 잘못된 배치에서는 일치율이 0.00% 였다.

기간 합계만 있는 **횡단면 파일**(일자 칸이 없다)은 기간을 알 수 없다. 그래서 같은
항목을 시계열에서 기간별로 더해 대조한다. 실측으로 5개 전부 1:1 로 맞았다.

## 이미 변환한 것은 건너뛴다

`normalized/MANIFEST.json` 에 **원본의 SHA-256** 을 적어 둔다. 다음에 돌릴 때
원본 해시가 그대로면 건너뛴다. 파일 이름이나 수정시각이 아니라 내용으로 판단하므로,
같은 파일을 다시 내려받아 이름이 `(1)` 로 바뀌어도 다시 변환하지 않는다.

반대로 **내용이 바뀌면 해시가 달라져 자동으로 다시 변환한다.**

## 사본으로 보이는 것은 견주어 보여 준다 — 고르지는 않는다

브라우저가 같은 자료를 여러 번 받으면 `... (1).csv` · `... (2).csv` 로 쌓인다.
그런데 **이름이 그렇다고 내용이 같은 것은 아니다.** 실측 2026-09-02 ·
`macro/수출입 총괄_20260901*.csv` 네 개가 그랬다.

    (1) 수출건수 129,148,763   수출중량 2250940321.0    수출금액 7195826806
    (3) 수출건수 129,148,763   수출중량 2250940320998   수출금액 7195826806
    (2) 수출건수 129,298,388   ...

(1)·(3)은 **중량 칸만** 다르고 정확히 1000배다 — 단위가 다른 사본이다. 반면 (2)는
수출건수부터 다르니 **아예 다른 자료**이고, 지우면 안 된다.

처음에는 *"큰 정수 자리에 소수점·지수 표기가 있으면 자릿수가 날아간 것"* 으로 자동
판정하려 했는데 **오탐이 쏟아졌다.** KRX 는 원래 큰 값을 `4.35585248E8` 처럼 주고
유효숫자는 그대로 살아 있다. 표기가 지수라는 사실과 값이 손상됐다는 사실은 별개다.

그래서 **판정하지 않고 견준다.** 어느 칸이 다른지 보여 주고, 무엇을 받았는지 아는
사람이 고른다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import PROJECT_ROOT  # noqa: E402
from common.trading_calendar import now_kst_iso  # noqa: E402

MANUAL_DIR = PROJECT_ROOT / "data" / "manual"
OUT_DIR = MANUAL_DIR / "normalized"
MANIFEST_PATH = OUT_DIR / "MANIFEST.json"

# 한국 공공기관 CSV 에서 실제로 나오는 순서.
# ⚠️ `utf-8-sig` 를 `utf-8` 보다 먼저 둔다. BOM 이 붙은 파일을 utf-8 로 읽으면
#    첫 칸 이름에 '﻿' 가 붙은 채 **성공해 버린다** — 조용히 틀리는 쪽이다.
ENCODINGS = ("utf-8-sig", "utf-8", "cp949", "euc-kr", "utf-16")

# 변환 대상. 텍스트가 아닌 것은 손대지 않는다 (parquet·xls 는 그 자체로 인코딩이 없다).
TEXT_SUFFIXES = {".csv", ".txt", ".tsv"}

# 투자자별 거래실적이 놓인 자리
INVESTOR_DIR = MANUAL_DIR / "ohlcv_stock" / "투자자별_거래실적"

# 이 자료가 무엇인지 — 파일에는 없고 사람만 아는 것이라 여기에 적어 둔다.
# (2026-09-02 확인: KRX 투자자별 거래실적 · ETF/ETN 등 포함)
INVESTOR_LABEL = "KRX투자자별거래실적"

# 🔴 **뜻이 없는 이름만 바꾼다.**
#
# KRX 정보데이터시스템이 떨어뜨리는 `data_4907_20260902.csv` 는 일련번호라 내용과
# 아무 관계가 없다. 이런 것만 값으로 판별해 이름을 붙인다.
#
# 반대로 **사람이 붙인 이름은 절대 건드리지 않는다.** 받는 사람이 화면에서 무엇을
# 골랐는지는 그 사람만 알고, 그 지식이 이름에 들어 있다. 스크립트가 값에서 알아낼 수
# 있는 것은 그보다 적다 — 예컨대 어느 시장·어느 상품군을 받았는지는 파일 안에 없다.
# 아는 것이 더 적은 쪽이 더 많은 쪽을 덮어쓰면 정보가 사라진다.
RAW_NAME = re.compile(r"^data_\d+_\d+$")

# 사본을 견줄 때, 값이 다른 칸을 몇 개까지 보여 줄지
DIFF_PREVIEW = 6


# ==================================================
# 1. 읽기 · 쓰기
# ==================================================

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def decode(path: Path) -> Tuple[str, str]:
    """(본문, 인코딩). 어떤 후보로도 못 읽으면 무엇을 해야 하는지까지 담아 올린다.

    ⚠️ **예외가 안 났다고 인코딩을 맞게 고른 것은 아니다.** `cp949` 와 `utf-16` 은
       거의 모든 바이트열을 받아들이므로 이 함수는 사실상 실패하지 않는다. 잘못
       고르면 예외 대신 **깨진 한글**이 나온다.

       방어선은 **순서**다. UTF-8 은 엄격해서 아무 바이트나 통과시키지 않으므로 먼저
       시도하고, 거기서 걸러진 것만 cp949 로 내려간다.
    """
    raw = path.read_bytes()
    for enc in ENCODINGS:
        try:
            return raw.decode(enc), enc
        except (UnicodeDecodeError, LookupError):
            continue
    raise ValueError(
        f"인코딩을 알 수 없습니다: {path.name}\n"
        f"   할 일: 시도한 것은 {', '.join(ENCODINGS)} 입니다. 파일을 편집기에서 열어\n"
        f"   인코딩을 확인하고 ENCODINGS 에 더하세요."
    )


def num(value: str) -> Optional[float]:
    text = (value or "").replace(",", "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def rows_of(text: str) -> List[List[str]]:
    return [r for r in csv.reader(text.splitlines()) if r]


# ==================================================
# 2. 투자자별 거래실적 — 값으로 정체를 알아낸다
# ==================================================

def _is_timeseries(head: Sequence[str]) -> bool:
    return bool(head) and head[0].strip() == "일자"


def _period(body: Sequence[Sequence[str]]) -> str:
    """시계열의 기간을 `20240902-20260901` 로. 일자 칸은 `2026/09/01` 꼴이다."""
    days = sorted(r[0].replace("/", "").replace("-", "").strip()
                  for r in body if r and r[0].strip())
    return f"{days[0]}-{days[-1]}" if days else "기간미상"


def _has_negative(body: Sequence[Sequence[str]]) -> bool:
    return any((num(v) or 0) < 0 for r in body for v in r[1:])


def _magnitude(body: Sequence[Sequence[str]]) -> float:
    return sum(abs(num(v) or 0) for r in body for v in r[1:])


def _residual(a: Sequence, b: Sequence, c: Sequence, tol: float = 1.0) -> float:
    """`c = b − a` 가 맞는 칸의 비율(0~1).

    허용치를 1.0 으로 둔다. 거래대금이 반올림된 단위로 오면 1 만큼 어긋난다 —
    완전일치를 요구하면 맞는 배치도 0% 로 떨어진다 (실측으로 확인했다).
    """
    맞음 = 전체 = 0
    # 줄·칸 수가 어긋난 파일이 섞일 수 있다. 짧은 쪽까지만 견준다.
    for ra, rb, rc in zip(a, b, c, strict=False):
        for va, vb, vc in zip(ra[1:], rb[1:], rc[1:], strict=False):
            fa, fb, fc = num(va), num(vb), num(vc)
            if fa is None or fb is None or fc is None:
                continue
            전체 += 1
            if abs((fb - fa) - fc) <= tol:
                맞음 += 1
    return 맞음 / 전체 if 전체 else 0.0


def classify_group(group: Dict[Path, List[List[str]]]) -> Dict[Path, str]:
    """같은 기간의 파일들에 `거래량_매도` 같은 이름을 붙인다.

    6개가 아니면 붙일 수 있는 것만 붙이고 나머지는 빈 문자열로 둔다 — 모르는 것을
    아는 척하는 것보다 이름 없이 두는 편이 낫다.
    """
    labels: Dict[Path, str] = {p: "" for p in group}

    net = [p for p, body in group.items() if _has_negative(body)]
    gross = [p for p in group if p not in net]
    if len(net) != 2 or len(gross) != 4:
        return labels

    # 규모가 작은 쪽이 수량, 큰 쪽이 금액이다
    net.sort(key=lambda p: _magnitude(group[p]))
    gross.sort(key=lambda p: _magnitude(group[p]))
    축배치 = (("거래량", gross[0], gross[1], net[0]),
              ("거래대금", gross[2], gross[3], net[1]))

    for 축, x, y, n in 축배치:
        정 = _residual(group[x], group[y], group[n])
        역 = _residual(group[y], group[x], group[n])
        if max(정, 역) < 0.99:
            # 검산이 안 서면 매도·매수를 가를 근거가 없다. 순매수만 확실하다.
            labels[n] = f"{축}_순매수"
            continue
        매도, 매수 = (x, y) if 정 >= 역 else (y, x)
        labels[매도] = f"{축}_매도"
        labels[매수] = f"{축}_매수"
        labels[n] = f"{축}_순매수"
    return labels


def plan_investor_files() -> Dict[Path, str]:
    """투자자별 거래실적 폴더의 파일마다 새 이름(확장자 제외)을 정한다."""
    plan: Dict[Path, str] = {}
    if not INVESTOR_DIR.is_dir():
        return plan

    시계열: Dict[str, Dict[Path, List[List[str]]]] = defaultdict(dict)
    횡단면: Dict[Path, List[List[str]]] = {}

    for path in sorted(INVESTOR_DIR.glob("*.csv")):
        text, _ = decode(path)
        rows = rows_of(text)
        if not rows:
            continue
        head, body = rows[0], rows[1:]
        if _is_timeseries(head):
            시계열[_period(body)][path] = body
        else:
            횡단면[path] = rows

    # ── 시계열: 기간 묶음마다 여섯 갈래를 가른다 ──
    기간별합: Dict[str, Dict[str, float]] = {}
    for 기간, group in 시계열.items():
        labels = classify_group(group)
        for path, label in labels.items():
            # 사람이 이미 이름을 붙였으면 그대로 둔다 (위 RAW_NAME 설명)
            if not RAW_NAME.match(path.stem):
                continue
            갈래 = label or "구분미상"
            plan[path] = f"{INVESTOR_LABEL}_{갈래}_{기간}"

        # 횡단면과 대조할 기준값 — '거래량_매도' 를 투자자별로 더해 둔다
        매도 = next((p for p, lb in labels.items() if lb == "거래량_매도"), None)
        if 매도 is not None:
            text, _ = decode(매도)
            rows = rows_of(text)
            head, body = rows[0], rows[1:]
            기간별합[기간] = {
                이름: sum(num(r[i]) or 0 for r in body if len(r) > i)
                for i, 이름 in enumerate(head[1:], start=1)
            }

    # ── 횡단면: 같은 항목의 기간 합과 대조해 기간을 되찾는다 ──
    for path, rows in 횡단면.items():
        if not RAW_NAME.match(path.stem):
            continue
        head, body = rows[0], rows[1:]
        기간 = _match_period(head, body, 기간별합)
        plan[path] = f"{INVESTOR_LABEL}_기간합계_{기간}"

    return plan


def _match_period(head: Sequence[str], body: Sequence[Sequence[str]],
                  기간별합: Dict[str, Dict[str, float]]) -> str:
    """횡단면 한 장이 어느 기간의 합인지 찾는다. 못 찾으면 `기간미상`."""
    try:
        열 = list(head).index("거래량_매도")
    except ValueError:
        return "기간미상"
    값 = {r[0]: (num(r[열]) or 0) for r in body if r}
    기준 = 값.get("금융투자")
    if not 기준:
        return "기간미상"
    for 기간, 합 in 기간별합.items():
        상대오차 = abs(합.get("금융투자", 0) - 기준) / abs(기준)
        if 상대오차 < 0.001:      # 지수표기 반올림만큼은 허용한다
            return 기간
    return "기간미상"


# ==================================================
# 3. 자릿수 손실 감지
# ==================================================

def sibling_diff(a: Sequence[Sequence[str]],
                 b: Sequence[Sequence[str]]) -> Optional[List[str]]:
    """머리글·행수가 같은 두 파일에서 **값이 다른 칸**의 이름을 돌려준다.

    같으면 빈 목록, 견줄 수 없으면 `None`.

    ## 왜 한 파일만 보고는 판정할 수 없는가

    처음에는 *"큰 정수 자리에 소수점·지수 표기가 있으면 자릿수가 날아간 것"* 으로
    보려 했는데 **오탐이 쏟아졌다.** KRX 는 원래 큰 값을 지수표기로 준다.

        4.35585248E8   →  435,585,248   유효숫자 9자리가 그대로 살아 있다

    표기가 지수라는 사실과 값이 손상됐다는 사실은 별개다. 그리고 손상 여부는
    **원래 몇 자리였는지를 알아야** 판정되는데 그 정보는 파일 안에 없다.

    그래서 판정하지 않고 **견준다.** 같은 폴더에 머리글이 같은 파일이 여럿 있으면
    사본일 가능성이 높으니, 어느 칸이 다른지를 사람에게 보여 준다.

    실측 2026-09-02 · `macro/수출입 총괄_20260901*.csv` 네 개:

        (1) 수출중량 2250940321.0     수출금액 7195826806
        (3) 수출중량 2250940320998    수출금액 7195826806   ← 금액은 완전히 같다

    다른 것은 **중량뿐**이고 정확히 1000배다. 손상이 아니라 **단위가 다른 사본**이었다
    (거기에 소수 1자리 반올림이 얹혔다). 한편 (1)과 (2)는 수출건수 자체가 달라
    (129,148,763 vs 129,298,388) 아예 **다른 자료**다.

    어느 쪽을 쓸지는 무엇을 받았는지 아는 사람이 정할 일이다. 스크립트는 고르지 않는다.
    """
    if len(a) < 2 or len(b) < 2 or a[0] != b[0] or len(a) != len(b):
        return None
    head = a[0]
    다른칸: List[str] = []
    for i, 이름 in enumerate(head):
        for ra, rb in zip(a[1:], b[1:], strict=False):
            if len(ra) <= i or len(rb) <= i:
                continue
            va, vb = ra[i].strip(), rb[i].strip()
            if va == vb:
                continue
            fa, fb = num(va), num(vb)
            # 문자열은 달라도 수치가 같으면 표기 차이일 뿐이다
            if fa is not None and fb is not None and fa == fb:
                continue
            다른칸.append(이름)
            break
    return 다른칸


# ==================================================
# 4. 실행
# ==================================================

def load_manifest() -> Dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {"files": {}}


def target_files() -> List[Path]:
    out: List[Path] = []
    for p in sorted(MANUAL_DIR.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if OUT_DIR in p.parents:          # 변환본을 다시 변환하지 않는다
            continue
        out.append(p)
    return out


def report_siblings(files: Sequence[Path]) -> None:
    """같은 폴더에서 **사본으로 보이는 것들**을 견주어 보여 준다. 고르지는 않는다.

    브라우저가 같은 자료를 여러 번 받으면 `... (1).csv` · `... (2).csv` 로 쌓인다.
    그런데 이름이 그렇다고 내용이 같은 것은 아니다 — 조회 조건을 바꿔 가며 받았다면
    **서로 다른 자료**이고, 지우면 안 된다.

    ⚠️ 다른 칸이 절반을 넘으면 사본으로 보지 않는다. 투자자별 거래실적의
       매도·매수 파일은 머리글이 같지만 값이 거의 전부 다른 **별개 자료**다.
    """
    묶음: Dict[Tuple[Path, Tuple[str, ...], int], List[Path]] = defaultdict(list)
    for p in files:
        try:
            rows = rows_of(decode(p)[0])
        except ValueError:
            continue
        if len(rows) < 2:
            continue
        묶음[(p.parent, tuple(rows[0]), len(rows))].append(p)

    후보 = [v for v in 묶음.values() if len(v) > 1]
    if not 후보:
        return

    표시했나 = False
    for group in 후보:
        기준 = group[0]
        기준행 = rows_of(decode(기준)[0])
        칸수 = len(기준행[0])
        for other in group[1:]:
            다름 = sibling_diff(기준행, rows_of(decode(other)[0]))
            if 다름 is None or len(다름) > 칸수 / 2:
                continue          # 아예 다른 자료다
            if not 표시했나:
                print("\n📄 사본으로 보이는 파일 — 어느 것을 쓸지는 사람이 정합니다")
                표시했나 = True
            쪽 = f"{기준.name}  ↔  {other.name}"
            if not 다름:
                print(f"   {쪽}\n      값이 완전히 같습니다 (표기 차이뿐)")
            else:
                print(f"   {쪽}\n      다른 칸: {', '.join(다름[:DIFF_PREVIEW])}"
                      f"{' …' if len(다름) > DIFF_PREVIEW else ''}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="손으로 받은 자료를 UTF-8 로 맞추고 이름을 알아볼 수 있게 바꾼다")
    parser.add_argument("--dry-run", action="store_true", help="무엇을 할지만 보여준다")
    parser.add_argument("--force", action="store_true", help="이미 변환한 것도 다시 한다")
    args = parser.parse_args()

    if not MANUAL_DIR.is_dir():
        print(f"🔴 폴더가 없습니다: {MANUAL_DIR}")
        print("   할 일: 손으로 받은 파일을 data/manual/ 아래에 두고 다시 실행하세요.")
        return 1

    files = target_files()
    if not files:
        print(f"변환할 텍스트 파일이 없습니다 ({MANUAL_DIR}).")
        return 0

    manifest = load_manifest()
    기록 = manifest.get("files", {})
    이름표 = plan_investor_files()

    변환 = 건너뜀 = 0

    print(f"── 손으로 받은 자료 정리 — 대상 {len(files)}개 ──")
    for src in files:
        rel = src.relative_to(MANUAL_DIR).as_posix()
        digest = sha256(src)
        before = 기록.get(rel)

        stem = 이름표.get(src, src.stem)
        dst = OUT_DIR / src.relative_to(MANUAL_DIR).parent / f"{stem}{src.suffix}"

        if (not args.force and before and before.get("src_sha256") == digest
                and dst.exists()):
            건너뜀 += 1
            continue

        text, enc = decode(src)

        표시 = "→" if not args.dry_run else "(예정)"
        바뀜 = "" if stem == src.stem else "  ★이름"
        print(f"  {표시} {rel}")
        print(f"      {enc} → utf-8 · {dst.relative_to(MANUAL_DIR).as_posix()}{바뀜}")

        if not args.dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            # newline="" 없이 쓰면 윈도우에서 줄바꿈이 두 번 들어간다
            dst.write_text(text, encoding="utf-8", newline="")
            기록[rel] = {
                "src_sha256": digest,
                "src_encoding": enc,
                "out": dst.relative_to(MANUAL_DIR).as_posix(),
                "renamed": stem != src.stem,
                "converted_at": now_kst_iso(),
            }
        변환 += 1

    if not args.dry_run and 변환:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        manifest["files"] = 기록
        manifest["generated_at"] = now_kst_iso()
        MANIFEST_PATH.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\n완료 — 변환 {변환}개 · 건너뜀 {건너뜀}개"
          f"{' (예정만 계산했습니다)' if args.dry_run else ''}")

    report_siblings(files)

    if not args.dry_run:
        print(f"\n변환본: {OUT_DIR}")
        print(f"기록  : {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
