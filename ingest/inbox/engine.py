"""팀원이 건네준 파일 하나를 규격에 비춰 읽고·정제하고·판정한다.

    from ingest.inbox.engine import inspect_file
    result = inspect_file("data/inbox/ohlcv_stock/삼성전자.csv", kind="ohlcv_stock")
    result.accepted      # 합격한 행
    result.quarantined   # 격리된 행 + 사유
    result.report        # 무엇을 어떻게 판정했나 (사람이 읽는 요약 포함)

단계는 여덟이고 **순서에 뜻이 있다.**

    ① 읽기 → ② 이름 맞추기 → ③ 결측 표기 정규화 → ④ 정제
    → ⑤ 파생 → ⑥ 칸 제약 → ⑦ 행 규칙 → ⑧ 가르기

③ 이 ④ 앞에 오는 이유: `"-"` 를 결측으로 보지 않은 채 `to_int` 에 넘기면 *변환 실패* 로
세어져 그 행이 격리된다 — 출처가 결측을 그렇게 적었을 뿐인데. ⑤ 가 ⑥ 앞에 오는 이유:
거시의 `known_from` 은 required 인데 우리가 채워 주는 칸이라, 채우기 전에 required 를 재면
멀쩡한 파일이 통째로 떨어진다.

무엇을 거부하고 무엇을 격리하나
-------------------------------
**파일째 거부**는 규격이 지키려는 것을 파일 구조가 통째로 깨는 경우다 — 뉴스 본문 칸이 있는
파일(`rejectedNames`), 한 규격 칸에 서로 다른 값의 원본 칸이 둘 붙는 경우. 이건 행을 골라 봐야
소용이 없다.

**행 격리**는 그 행만 못 믿는 경우다. 나머지 행은 들인다. 격리된 행도 **원본 그대로 함께
남긴다** — 사람이 고쳐서 다시 넣을 수 있어야 하고, 그러려면 우리가 정제하기 전 값이 필요하다.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from ingest.inbox import cleaners as cln
from ingest.inbox import derive as drv
from ingest.inbox.rules import (
    SPECIAL_NAMES,
    RuleSyntaxError,
    check_alias_invariant,
    compile_pattern,
    evaluate_rule,
    referenced_columns,
)

#: 규격 파일이 사는 곳.
SCHEMA_DIR = Path(__file__).parent / "schemas"

#: CSV 인코딩 후보. **순서가 중요하다** — `cp949` 는 거의 모든 바이트열을 받아 주기 때문에
#: 먼저 두면 UTF-8 파일을 깨진 한글로 읽고도 성공했다고 판단한다.
ENCODINGS = ("utf-8-sig", "utf-8", "cp949", "euc-kr")


class InboxError(ValueError):
    """파일을 들일 수 없다. 무엇을 해야 하는지까지 메시지에 담는다."""


@dataclass
class Verdict:
    """행 하나에 대한 판정 사유 하나."""

    rule: str
    severity: str
    note: str = ""


@dataclass
class InboxResult:
    """파일 하나를 검사한 결과."""

    kind: str
    source: str
    rows_total: int
    accepted: pd.DataFrame
    quarantined: pd.DataFrame
    report: dict
    rejected: Optional[str] = None          # 파일째 거부 사유 (없으면 None)
    questions: List[dict] = field(default_factory=list)   # 사람에게 물어야 하는 것

    @property
    def ok(self) -> bool:
        return self.rejected is None


# ==================================================
# 1. 규격
# ==================================================
def available_kinds() -> List[str]:
    """규격이 있는 종류 목록."""
    return sorted(p.stem for p in SCHEMA_DIR.glob("*.json"))


def load_spec(kind: str) -> dict:
    """규격을 읽고 **불변식을 먼저 검사한다.**

    한 별칭이 두 필드에 걸쳐 있으면 자료가 아니라 규격이 틀린 것이라, 파일을 읽기 전에 안다.
    """
    path = SCHEMA_DIR / f"{kind}.json"
    if not path.exists():
        raise InboxError(
            f"{kind!r} 규격이 없다.\n"
            f"  있는 것: {', '.join(available_kinds())}\n"
            f"  할 일: 종류 이름을 확인하거나 {path} 를 만든다."
        )
    spec = json.loads(path.read_text(encoding="utf-8"))
    check_alias_invariant(spec)
    return spec


def _extension(spec: dict) -> dict:
    return spec.get("x-alphastack") or {}


# ==================================================
# 2. 읽기
# ==================================================
def read_table(path: Path) -> pd.DataFrame:
    """CSV·Excel·Parquet 를 읽는다. **모든 칸을 문자열로 읽는다.**

    🔴 pandas 에게 타입 추론을 맡기지 않는 이유가 있다. `"005930"` 을 정수 5930 으로 읽어
    버리면 그 시점에 앞자리 0 이 사라져, 뒤에 `zfill6` 을 걸어도 *원래 몇 자리였는지* 알 수
    없다. 우리가 되살릴 수 있는 것은 문자열로 받았을 때뿐이다. 타입은 규격의 `cleaners` 가
    정한 순서대로 우리가 붙인다.
    """
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(path).astype("object")

    if suffix in (".xlsx", ".xls"):
        try:
            return pd.read_excel(path, dtype=str)
        except ImportError as error:
            raise InboxError(
                f"엑셀 파일을 읽을 수 없다: {path.name}\n"
                f"  까닭: {error}\n"
                "  할 일: pip install openpyxl · 또는 CSV 로 내보내 달라고 부탁한다."
            ) from error

    last_error: Optional[Exception] = None
    for encoding in ENCODINGS:
        try:
            return pd.read_csv(path, dtype=str, encoding=encoding, keep_default_na=False,
                               na_values=[""])
        except (UnicodeDecodeError, UnicodeError) as error:
            last_error = error
            continue
        except pd.errors.EmptyDataError as error:
            raise InboxError(f"{path.name} 이 비어 있다 — 칸 이름 줄조차 없다.") from error

    raise InboxError(
        f"{path.name} 의 인코딩을 알 수 없다 — {', '.join(ENCODINGS)} 를 다 대 봤다.\n"
        f"  마지막 오류: {last_error}\n"
        "  할 일: 파일을 UTF-8 로 다시 저장해 달라고 부탁한다."
    )


# ==================================================
# 3. 이름 맞추기
# ==================================================
@dataclass
class ColumnMapping:
    """원본 칸 이름을 규격 칸으로 옮긴 결과."""

    mapped: Dict[str, str] = field(default_factory=dict)        # 원본 → 규격
    extras: List[str] = field(default_factory=list)             # 규격 밖 칸
    questions: List[dict] = field(default_factory=list)         # 사람에게 물을 것
    rejected: Optional[str] = None                              # 파일째 거부 사유


def map_columns(frame: pd.DataFrame, spec: dict) -> ColumnMapping:
    """원본 칸 이름을 규격 칸 이름으로 옮긴다.

    네 갈래로 나뉜다.

    - **규격 칸 이름 그대로**거나 **별칭**이면 → 옮긴다
    - `rejectedNames` 에 있으면 → **파일째 거부.** 뉴스 본문이 그 예다. 별칭으로 받아
      주면 "본문 칸을 안 만든다" 는 방어가 한 줄로 뚫린다
    - `ambiguousNames` 에 있으면 → **추측하지 않고 사람에게 묻는다.** `"지수"` 가 이름인지
      값인지는 값만 봐서 정할 수 없고, 반은 틀린다
    - 나머지 → `extras` 로 보존한다. **버리지 않는다** — 팀원이 애써 붙여 온 칸이고,
      나중에 쓸모가 생겼을 때 원본을 다시 받는 것보다 싸다
    """
    extension = _extension(spec)
    fields = {f["name"] for f in spec["fields"]}
    aliases = extension.get("aliases") or {}
    rejected_names = {k.lower(): v for k, v in (extension.get("rejectedNames") or {}).items()
                      if not k.startswith("_")}
    ambiguous = extension.get("ambiguousNames") or {}

    lookup: Dict[str, str] = {}
    for target, names in aliases.items():
        for alias in names:
            lookup[alias.strip().lower()] = target
    for name in fields:
        lookup[name.lower()] = name

    result = ColumnMapping()
    claimed: Dict[str, str] = {}         # 규격 칸 → 이미 차지한 원본 칸

    for column in frame.columns:
        key = str(column).strip().lower()

        if key in rejected_names:
            result.rejected = (
                f"{column!r} 칸이 있다 — {rejected_names[key]} 은 들이지 않는다.\n"
                "  왜 파일째 되돌리나: 이 칸을 별칭으로 받아 주면 '본문을 담지 않는다' 는\n"
                "  약속이 우회된다. 행을 골라 봐야 소용이 없어 파일 단위로 돌려보낸다."
            )
            return result

        if key in ambiguous:
            entry = ambiguous[key]
            result.questions.append({
                "column": str(column),
                "candidates": entry.get("candidates", []),
                "note": entry.get("note", ""),
            })
            result.extras.append(str(column))
            continue

        target = lookup.get(key)
        if target is None:
            result.extras.append(str(column))
            continue

        if target in claimed:
            # 같은 규격 칸을 두 원본 칸이 노린다. 값이 같으면 단순 중복이라 하나만 쓰고,
            # 다르면 어느 쪽이 옳은지 우리가 정할 수 없다 → 파일째 되돌린다.
            first = claimed[target]
            if frame[first].astype("object").equals(frame[column].astype("object")):
                result.questions.append({
                    "column": str(column),
                    "candidates": [target],
                    "note": f"{first!r} 와 값이 같아 중복으로 보고 하나만 썼다.",
                })
                result.extras.append(str(column))
                continue
            result.rejected = (
                f"{first!r} 와 {column!r} 이 둘 다 규격의 {target!r} 을 가리키는데 값이 다르다.\n"
                "  왜 파일째 되돌리나: 어느 쪽이 옳은지 값만 보고는 정할 수 없다.\n"
                "  할 일: 한 칸만 남기고 다시 보내 달라고 부탁한다."
            )
            return result

        claimed[target] = str(column)
        result.mapped[str(column)] = target

    return result


# ==================================================
# 3.5 어느 규격인가 — 폴더가 안 알려 줄 때
# ==================================================
#: 1등이 2등보다 이만큼은 앞서야 종류를 정한다. 아니면 사람에게 묻는다.
KIND_MARGIN = 0.25


def score_kind(frame: pd.DataFrame, kind: str) -> dict:
    """이 표가 그 규격에 얼마나 들어맞나. `0.0`~`1.0`.

    HuggingFace 쪽 `inbox/<이름>/파일.csv` 에는 **종류 폴더가 없다.** 로컬
    `data/inbox/<종류>/` 처럼 경로가 알려 주지 않으므로 무엇으로 검사할지 정해야 하는데,
    파일 이름으로 찍으면 틀린다 — `2026-08-31_거래대금상위.csv` 가 종목인지 지수인지
    이름만 봐서는 모른다.

    **그래서 추측하지 않고 잰다.** 규격 5장의 별칭으로 각각 몇 칸이나 매핑되는지 세고,
    특히 `required` 필드를 덮는지를 무겁게 본다. 종목 파일을 뉴스 규격에 대면 `pub_dt`·
    `title`·`link` 가 하나도 안 잡혀 점수가 바닥이다.
    """
    spec = load_spec(kind)
    mapping = map_columns(frame, spec)
    if mapping.rejected:
        return {"kind": kind, "score": 0.0, "required_hit": 0.0, "mapped": 0}

    required = {f["name"] for f in spec["fields"]
                if (f.get("constraints") or {}).get("required")}
    targets = set(mapping.mapped.values())

    # required 를 덮는 비율이 주(主)다 — 그게 그 규격의 정체성이다.
    required_hit = len(required & targets) / len(required) if required else 0.0
    # 전체 칸 중 몇 개나 규격이 아는 이름인지가 부(副).
    coverage = len(targets) / len(frame.columns) if len(frame.columns) else 0.0
    return {
        "kind": kind,
        "score": round(required_hit * 0.75 + coverage * 0.25, 4),
        "required_hit": round(required_hit, 4),
        "mapped": len(targets),
    }


def guess_kind(frame: pd.DataFrame) -> tuple:
    """규격 5장에 다 대 보고 `(정해진 종류 또는 None, 점수표)` 를 돌려준다.

    1등이 2등보다 `KIND_MARGIN` 만큼 앞서고 required 를 절반 넘게 덮을 때만 정한다.
    **애매하면 None 을 준다** — 틀린 규격으로 검사하면 멀쩡한 파일이 통째로 격리되고,
    그 결과를 보고 팀원이 자기 자료가 잘못됐다고 오해한다.
    """
    scores = sorted((score_kind(frame, kind) for kind in available_kinds()),
                    key=lambda s: -s["score"])
    if not scores:
        return None, []
    best = scores[0]
    runner_up = scores[1]["score"] if len(scores) > 1 else 0.0
    if best["required_hit"] >= 0.5 and best["score"] - runner_up >= KIND_MARGIN:
        return best["kind"], scores
    return None, scores


# ==================================================
# 4. 파일 수준 메타 채우기
# ==================================================
def fill_file_meta(frame: pd.DataFrame, spec: dict) -> List[dict]:
    """행마다 없을 수 있는 식별 칸을 파일 전체에 편다.

    DART 응답은 한 요청에 한 회사·한 기간이라 `corp_code`·`bsns_year` 같은 값이 **응답
    최상위에 한 번만** 붙는다. 계정 줄에는 안 실려서, CSV 로 내리면 첫 줄에만 있거나
    아예 없다.

    유일한 값 하나만 있을 때만 편다. 두 가지 이상이면 그 파일은 여러 회사가 섞인 것이라
    **함부로 채우면 남의 회사 값이 붙는다.**
    """
    meta_fields = (_extension(spec).get("fileMeta") or {}).get("fields") or []
    notes: List[dict] = []
    for name in meta_fields:
        if name not in frame.columns:
            continue
        present = frame[name].dropna()
        present = present[present.astype(str).str.strip() != ""]
        missing = len(frame) - len(present)
        if missing == 0 or present.empty:
            continue
        unique = present.astype(str).unique()
        if len(unique) != 1:
            notes.append({"column": name, "action": "skipped", "distinct": len(unique),
                          "note": "값이 여러 가지라 파일 전체에 펴지 않았다"})
            continue
        frame[name] = frame[name].where(frame[name].notna() & (frame[name] != ""), unique[0])
        notes.append({"column": name, "action": "filled", "value": str(unique[0]),
                      "rows": int(missing)})
    return notes


# ==================================================
# 5. 칸 제약
# ==================================================
def check_constraints(frame: pd.DataFrame, spec: dict) -> Dict[str, List[Verdict]]:
    """규격의 `constraints` 를 판다 — required · pattern · enum · minimum · maximum.

    행 번호별 위반 목록을 돌려준다. 전부 `error` 다 — 이 셋은 "값이 규격 밖" 이라는 뜻이고,
    그런 값을 들이면 뒤의 조인·계산이 조용히 틀린다.
    """
    violations: Dict[str, List[Verdict]] = {}

    def add(mask: pd.Series, rule: str, note: str) -> None:
        for index in frame.index[mask.fillna(False)]:
            violations.setdefault(index, []).append(Verdict(rule, "error", note))

    for spec_field in spec["fields"]:
        name = spec_field["name"]
        if name not in frame.columns:
            constraints = spec_field.get("constraints") or {}
            if constraints.get("required"):
                # 칸 자체가 없다 — 모든 행이 위반이다.
                add(pd.Series(True, index=frame.index), f"{name}.required",
                    f"필수 칸이 파일에 없다: {name}")
            continue

        column = frame[name]
        constraints = spec_field.get("constraints") or {}
        blank = column.isna() | (column.astype("object").map(
            lambda v: isinstance(v, str) and not v.strip()))

        if constraints.get("required"):
            add(blank, f"{name}.required", f"반드시 있어야 하는 칸이 비었다: {name}")

        if constraints.get("pattern"):
            pattern = compile_pattern(constraints["pattern"])
            bad = ~blank & ~column.astype("object").map(
                lambda v, p=pattern: bool(p.fullmatch(str(v))))
            add(bad, f"{name}.pattern",
                f"{name} 이 {constraints['pattern']} 모양이 아니다")

        if constraints.get("enum"):
            allowed = set(constraints["enum"])
            bad = ~blank & ~column.astype("object").map(lambda v, a=allowed: str(v) in a)
            add(bad, f"{name}.enum",
                f"{name} 은 {', '.join(constraints['enum'])} 중 하나여야 한다")

        if "minimum" in constraints or "maximum" in constraints:
            numeric = pd.to_numeric(column, errors="coerce")
            if "minimum" in constraints:
                add(numeric.notna() & (numeric < constraints["minimum"]),
                    f"{name}.minimum", f"{name} 이 {constraints['minimum']} 보다 작다")
            if "maximum" in constraints:
                add(numeric.notna() & (numeric > constraints["maximum"]),
                    f"{name}.maximum", f"{name} 이 {constraints['maximum']} 보다 크다")

    return violations


# ==================================================
# 6. 행 규칙
# ==================================================
def _all_missing(index) -> pd.Series:
    """파일에 없는 칸을 대신할 "전부 결측" 열.

    `object` 형에 `NaN` 을 담는다. 형이 왜 중요한가 —
    - `float64` 로 두면 `release_date >= period_end` 처럼 **문자열 칸과 비교**하는 규칙이
      *Invalid comparison between dtype=str and ndarray* 로 죽는다.
    - `None` 을 담으면 `high >= low` 같은 **수치 비교**가
      *'>=' not supported between 'int' and 'NoneType'* 으로 죽는다.
    - `pd.NA` 를 담으면 비교 결과가 masked 가 되어 *boolean value of NA is ambiguous* 로 죽는다.

    `object` + `NaN` 은 규격 5개의 행 규칙 52개를 전부 통과했다(2026-09-03 실측).
    정제기(`cleaners`)를 거친 실제 결측 칸도 같은 모양이다.
    """
    return pd.Series([float("nan")] * len(index), index=index, dtype="object")


def check_row_rules(frame: pd.DataFrame, spec: dict) -> tuple:
    """규격의 `rowRules` 를 전부 적용한다.

    돌려주는 것은 `(행별 위반 목록, 규칙별 집계)` 다. `severity` 가 `error` 면 격리하고
    `warn` 이면 통과시키되 센다 — 그 판단은 이 함수가 아니라 `split_rows` 가 한다.
    """
    rules = _extension(spec).get("rowRules") or []
    violations: Dict[str, List[Verdict]] = {}
    tally: List[dict] = []

    for rule in rules:
        expression = rule["expr"]

        # 🔴 규칙이 쓰는 칸이 파일에 없으면 **그 칸이 통째로 결측인 것으로 보고 잰다.**
        #
        #    예전에는 여기서 규칙 전체를 건너뛰었다. 의도는 "팀원이 선택 칸을 안 담아 온
        #    것뿐인데 위반으로 세면 파일이 통째로 격리된다" 였고, `X and Y` 꼴에는 맞다.
        #    그런데 **`X is not null or Y is not null` 꼴에는 정반대로 작동한다** —
        #    "둘 중 하나만 있으면 된다" 가 "둘 다 있어야 잰다" 로 뒤집힌다.
        #
        #    그래서 재무 규격의 `has_time_anchor` 는 한 번도 돈 적이 없었다. 접수일도
        #    접수번호도 없는 행을 학습에 넣으면 결산기에 값을 붙이게 되고, 석 달치 미래를
        #    넣고도 예외는 나지 않고 성능만 좋아진다 — 이 프로젝트에서 가장 조용한 오류다.
        #    그걸 막으라고 둔 규칙이 그 자신이 조용히 꺼져 있었다.
        #
        #    없는 칸 = 그 칸의 모든 값이 결측. 이렇게 보면 `X is null` 은 참이 되어
        #    가드 꼴(`X is null or …`)은 그대로 통과하고, OR 꼴은 나머지 항으로 판정된다.
        #    실측(2026-09-03): 이렇게 바꿔도 **error 로 새로 격리되는 규칙은 0건**이고,
        #    새로 걸리는 4건은 전부 `warn` 이며 그 규칙들이 재려던 바로 그것이다
        #    (`known_from_basis_present` · `ohl_missing_together`).
        needed = referenced_columns(expression) - SPECIAL_NAMES
        absent = sorted(needed - set(frame.columns))
        measured = frame
        if absent:
            measured = frame.copy()
            for name in absent:
                measured[name] = _all_missing(frame.index)

        try:
            passed = evaluate_rule(expression, measured)
        except RuleSyntaxError as error:
            # 규격이 틀렸을 때 나는 예외다. 미리 걸러 내지 않고 여기서 잡는 이유는
            # 규격의 오타가 조용히 "건너뜀" 으로 묻히지 않게 하기 위해서다.
            raise InboxError(
                f"규격의 규칙 {rule.get('id')!r} 을 읽을 수 없다: {expression}\n  {error}\n"
                "  할 일: 규격 파일의 식을 고친다 (자료가 아니라 규격의 문제다)."
            ) from error
        except (TypeError, ValueError) as error:
            # 식은 읽히는데 이 파일의 칸 **형**이 규칙이 전제한 것과 다르다.
            # 그냥 두면 pandas 의 traceback 이 그대로 올라와 무엇이 문제인지 안 보인다.
            채운말 = f" (파일에 없어 결측으로 채운 칸: {', '.join(absent)})" if absent else ""
            raise InboxError(
                f"규격의 규칙 {rule.get('id')!r} 을 이 파일에 적용할 수 없다: {expression}\n"
                f"  {type(error).__name__}: {error}{채운말}\n"
                "  칸의 형이 규칙이 전제한 것과 다르다.\n"
                "  할 일: 규격의 `cleaners` 로 그 칸의 형을 맞추거나, 규칙의 식을 고친다."
            ) from error

        failed = ~passed.fillna(False)
        count = int(failed.sum())
        entry = {"rule": rule.get("id"), "severity": rule.get("severity"),
                 "violations": count}
        if absent:
            # 무엇을 채워서 쟀는지 반드시 남긴다. 안 남기면 보고서를 읽는 사람이
            # "이 검사는 실제 값으로 통과했다" 고 오해한다.
            entry["filled_columns"] = absent
            entry["note"] = ("파일에 없는 칸을 전부 결측으로 보고 쟀다: "
                             f"{', '.join(absent)}")
        tally.append(entry)
        if count:
            for index in frame.index[failed]:
                violations.setdefault(index, []).append(
                    Verdict(rule.get("id", "?"), rule.get("severity", "error"),
                            rule.get("note", "")))

    return violations, tally


# ==================================================
# 7. 가르기
# ==================================================
def _key_hash(row: pd.Series, primary_key: Sequence[str]) -> str:
    """열쇠 칸들로 만드는 지문. 중복 판정과 조인에 쓴다."""
    parts = [str(row.get(name, "")) for name in primary_key]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:32]


def _merge(*mappings: Dict[str, List[Verdict]]) -> Dict[str, List[Verdict]]:
    out: Dict[str, List[Verdict]] = {}
    for mapping in mappings:
        for index, items in mapping.items():
            out.setdefault(index, []).extend(items)
    return out


def split_rows(frame: pd.DataFrame, raw: pd.DataFrame, spec: dict,
               violations: Dict[str, List[Verdict]], extras: Sequence[str],
               *, kind: str, touched: Optional[Dict[str, pd.Series]] = None) -> tuple:
    """합격과 격리로 가른다.

    `error` 가 하나라도 있으면 격리, 없으면 합격이다. `warn` 은 합격시키되 그 행에 붙여
    둔다 — 규격의 `zero_ohlc` 처럼 **우리 자료에 283,468행이나 있는** 상태는 오류가 아니라
    알아 둘 사실이라, 이걸 error 로 두면 멀쩡한 자료가 통째로 격리된다.
    """
    primary_key = spec.get("primaryKey") or []
    accepted_rows: List[dict] = []
    quarantined_rows: List[dict] = []

    spec_columns = [f["name"] for f in spec["fields"] if f["name"] in frame.columns]

    for index in frame.index:
        items = violations.get(index, [])
        errors = [v for v in items if v.severity == "error"]
        warns = [v for v in items if v.severity != "error"]

        payload = {name: _jsonable(frame.at[index, name]) for name in spec_columns}
        extra_payload = {name: _jsonable(raw.at[index, name]) for name in extras
                         if name in raw.columns}

        # 이 행에서 우리가 손댄 칸 이름. 값이 아니라 이름만 남긴다 — 무엇을 무엇으로
        # 바꿨는지는 보고서의 표본 20건에서 보고, 여기서는 **어느 행이 손질됐는지**를 센다.
        record = {
            "row_no": int(frame.index.get_loc(index)) + 1,
            "kind": kind,
            "payload": payload,
            "extras": extra_payload,
            "changes": sorted(name for name, mask in (touched or {}).items()
                              if bool(mask.get(index, False))),
        }

        if errors:
            # 격리된 행은 **원본 그대로도 함께 남긴다.** 사람이 고쳐 다시 넣으려면
            # 우리가 정제하기 전 값이 필요하다.
            record["raw"] = {str(c): _jsonable(raw.at[index, c]) for c in raw.columns}
            record["violations"] = [
                {"rule": v.rule, "severity": v.severity, "note": v.note} for v in errors + warns
            ]
            quarantined_rows.append(record)
        else:
            record["key_hash"] = _key_hash(frame.loc[index], primary_key) if primary_key else ""
            record["warnings"] = [
                {"rule": v.rule, "severity": v.severity, "note": v.note} for v in warns
            ]
            accepted_rows.append(record)

    return pd.DataFrame(accepted_rows), pd.DataFrame(quarantined_rows)


def _jsonable(value):
    """JSON 으로 담을 수 있는 값으로 낮춘다. 결측은 None 이다."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    return value if isinstance(value, (str, int, float, bool)) else str(value)


# ==================================================
# 8. 한 파일을 통째로
# ==================================================
def inspect_file(path, kind: str, *, db_path=None) -> InboxResult:
    """파일 하나를 규격에 비춰 끝까지 검사한다."""
    path = Path(path)
    spec = load_spec(kind)
    extension = _extension(spec)

    raw = read_table(path)
    raw.index = range(len(raw))
    rows_total = len(raw)

    report: dict = {
        "kind": kind,
        "source": str(path),
        "schema_version": extension.get("version"),
        "rows_total": rows_total,
        "encoding_candidates": list(ENCODINGS),
    }

    mapping = map_columns(raw, spec)
    if mapping.rejected:
        report["rejected"] = mapping.rejected
        return InboxResult(kind, str(path), rows_total, pd.DataFrame(), pd.DataFrame(),
                           report, rejected=mapping.rejected, questions=mapping.questions)

    report["columns"] = {
        "mapped": mapping.mapped,
        "extras": mapping.extras,
        "unmapped_count": len(mapping.extras),
    }
    report["questions"] = mapping.questions

    frame = raw.rename(columns=mapping.mapped)[list(mapping.mapped.values())].copy()

    # ③ 결측 표기를 진짜 결측으로. 정제보다 **먼저** 돈다.
    missing_tokens = spec.get("missingValues") or []
    for column in frame.columns:
        frame[column] = cln.normalize_missing(frame[column], missing_tokens)

    report["file_meta"] = fill_file_meta(frame, spec)

    # ④ 정제 — 규격이 적은 순서 그대로.
    cleaner_map = extension.get("cleaners") or {}
    log = cln.CleanerLog()
    cast_failed = pd.Series(False, index=frame.index)
    touched: Dict[str, pd.Series] = {}          # 칸 → 그 칸에 손댄 행 마스크
    for column, chain in cleaner_map.items():
        if column not in frame.columns:
            continue
        values, column_log, failed, changed = cln.apply_chain(frame[column], chain, column=column)
        frame[column] = values
        log.entries.extend(column_log.entries)
        cast_failed = cast_failed | failed
        if changed.any():
            touched[column] = changed
    report["cleaners"] = log.to_list()
    report["cleaner_totals"] = {"changed": log.total_changed, "failed": log.total_failed}

    violations: Dict[str, List[Verdict]] = {}
    for index in frame.index[cast_failed]:
        violations.setdefault(index, []).append(Verdict(
            "cleaner.failed", "error",
            "값이 있었는데 규격이 정한 형으로 읽을 수 없다 — 결측과 구별해 격리한다"))

    # ⑤ 파생 — required 를 재기 **전에** 채운다.
    derived = drv.derive(kind, frame, spec, db_path=db_path)
    frame = derived.frame
    report["derive"] = derived.to_list()
    for index in frame.index[derived.blocked]:
        violations.setdefault(index, []).append(
            Verdict("lookahead", "error", str(derived.reasons.loc[index])))

    # ⑥⑦ 제약과 규칙.
    violations = _merge(violations, check_constraints(frame, spec))
    rule_violations, tally = check_row_rules(frame, spec)
    violations = _merge(violations, rule_violations)
    report["row_rules"] = tally

    accepted, quarantined = split_rows(frame, raw, spec, violations, mapping.extras,
                                       kind=kind, touched=touched)
    report["rows_accepted"] = len(accepted)
    report["rows_quarantined"] = len(quarantined)
    report["quarantine_reasons"] = _reason_tally(quarantined)

    return InboxResult(kind, str(path), rows_total, accepted, quarantined, report,
                       questions=mapping.questions)


def _reason_tally(quarantined: pd.DataFrame) -> List[dict]:
    """격리 사유를 규칙별로 센다 — 보고서에서 가장 먼저 읽히는 줄이다."""
    if quarantined.empty:
        return []
    counter: Dict[str, int] = {}
    for items in quarantined["violations"]:
        for item in items:
            if item["severity"] == "error":
                counter[item["rule"]] = counter.get(item["rule"], 0) + 1
    return [{"rule": rule, "rows": count}
            for rule, count in sorted(counter.items(), key=lambda kv: -kv[1])]


__all__ = [
    "SCHEMA_DIR",
    "ENCODINGS",
    "InboxError",
    "Verdict",
    "InboxResult",
    "ColumnMapping",
    "KIND_MARGIN",
    "available_kinds",
    "load_spec",
    "score_kind",
    "guess_kind",
    "read_table",
    "map_columns",
    "fill_file_meta",
    "check_constraints",
    "check_row_rules",
    "split_rows",
    "inspect_file",
]
