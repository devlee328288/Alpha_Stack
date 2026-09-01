"""반입 규격의 `rowRules.expr` 을 읽어 표 전체에 한꺼번에 적용한다.

**왜 파서를 쓰는가.** 처음에는 `and` → `&`, `or` → `|` 문자열 치환으로 옮겼다.
그런데 파이썬에서 비트 연산자는 비교보다 **먼저** 묶인다 — `a | b == 0` 이
`a | (b == 0)` 이 된다. 그 상태로 종목 규격을 우리 자료에 돌렸더니 920만 행 중
**892만 행이 위반**으로 나왔다. 규격이 아니라 옮기는 쪽이 틀린 것이었는데,
숫자만 보고는 어느 쪽이 틀렸는지 알 수 없었다.

`expr` 은 통째로 **유효한 파이썬 식**이다(`high is null` 은 `Name('null')` 과의
`is` 비교로 파싱된다). 그래서 문법을 새로 만들지 않고 `ast` 로 읽는다 —
우선순위는 파이썬 파서가 이미 옳게 알고 있으므로 그것을 빌린다.

    from ingest.inbox.rules import evaluate_rule
    ok = evaluate_rule("high is null or high >= low", frame)   # 참=통과인 불리언 시리즈

⚠️ **`eval` 을 쓰지 않는다.** 규격 파일은 사람이 손으로 고치는 것이고, 거기 적힌
문자열을 그대로 실행하면 규격이 코드가 된다. 여기서는 트리를 걸어가며 우리가 허용한
연산만 수행하고, 모르는 구문을 만나면 `RuleSyntaxError` 로 세운다.
"""

from __future__ import annotations

import ast
import re
from typing import Set

import numpy as np
import pandas as pd


class RuleSyntaxError(ValueError):
    """규격에 적힌 식을 우리 문법으로 읽을 수 없다."""


#: `is` 의 오른쪽에 올 수 있는 특별한 이름.
SPECIAL_PREDICATES = frozenset({"null", "weekday"})

#: 칸 이름이 아니라 바깥에서 주어지는 값.
SPECIAL_NAMES = frozenset({"today", "now"})

COMPARE_OPS = {
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
}

BINARY_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
}

#: 허용하는 함수. 이 열 개뿐이다 — 늘리려면 규격 README 에도 함께 적는다.
FUNCTIONS = frozenset({
    "abs", "len", "year", "month", "day", "date", "time",
    "starts_with", "contains", "matches",
})


def referenced_columns(expr: str) -> Set[str]:
    """식이 가리키는 이름을 모은다 — 규칙이 없는 칸을 쓰는지 검사할 때 쓴다.

    `null` · `weekday` 는 값이 아니라 술어이므로 세지 않는다.
    """
    tree = ast.parse(expr, mode="eval")
    out: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in SPECIAL_PREDICATES:
            if node.id not in FUNCTIONS:
                out.add(node.id)
    return out


class _Evaluator:
    """식 하나를 표 하나에 대해 계산한다."""

    def __init__(self, frame: pd.DataFrame, *, today: str, now: pd.Timestamp):
        self.frame = frame
        self.today = today
        self.now = now
        self._weekday = None

    # ── 요일은 쓰일 때 한 번만 만든다 (920만 행에서 비싸다) ──
    @property
    def weekday(self) -> pd.Series:
        if self._weekday is None:
            column = "bas_dd" if "bas_dd" in self.frame.columns else "eff_dd"
            parsed = pd.to_datetime(self.frame[column], format="%Y%m%d", errors="coerce")
            self._weekday = parsed.dt.weekday < 5
        return self._weekday

    def walk(self, node):
        if isinstance(node, ast.BoolOp):
            # 괄호 구조는 ast 가 이미 지켜 놓았다. 여기서는 순서대로 접기만 한다.
            parts = [self.as_bool(self.walk(v)) for v in node.values]
            join = (lambda a, b: a & b) if isinstance(node.op, ast.And) else (lambda a, b: a | b)
            result = parts[0]
            for part in parts[1:]:
                result = join(result, part)
            return result

        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                return ~self.as_bool(self.walk(node.operand))
            if isinstance(node.op, ast.USub):
                return -self.walk(node.operand)
            raise RuleSyntaxError(f"다룰 수 없는 단항 연산자: {type(node.op).__name__}")

        if isinstance(node, ast.Compare):
            return self.compare(node)

        if isinstance(node, ast.BinOp):
            operation = BINARY_OPS.get(type(node.op))
            if operation is None:
                raise RuleSyntaxError(f"다룰 수 없는 산술 연산자: {type(node.op).__name__}")
            return operation(self.walk(node.left), self.walk(node.right))

        if isinstance(node, ast.Call):
            return self.call(node)

        if isinstance(node, ast.Constant):
            return node.value

        if isinstance(node, ast.List):
            return [self.walk(e) for e in node.elts]

        if isinstance(node, ast.Name):
            if node.id == "today":
                return self.today
            if node.id == "now":
                return self.now
            if node.id in self.frame.columns:
                return self.frame[node.id]
            raise RuleSyntaxError(f"표에 없는 칸: {node.id}")

        raise RuleSyntaxError(f"다룰 수 없는 구문: {type(node).__name__}")

    def compare(self, node: ast.Compare):
        first_op = node.ops[0]
        first_right = node.comparators[0]

        # `X is null` · `X is not null` · `X is weekday`
        if isinstance(first_op, (ast.Is, ast.IsNot)) and isinstance(first_right, ast.Name):
            if first_right.id == "null":
                series = self.walk(node.left)
                if not isinstance(series, pd.Series):
                    return pd.Series(series is None, index=self.frame.index)
                return series.isna() if isinstance(first_op, ast.Is) else series.notna()
            if first_right.id == "weekday":
                return self.weekday if isinstance(first_op, ast.Is) else ~self.weekday
            raise RuleSyntaxError(f"`is` 오른쪽에 올 수 없는 이름: {first_right.id}")

        if isinstance(first_op, (ast.In, ast.NotIn)):
            values = self.walk(first_right)
            series = self.walk(node.left)
            inside = series.isin(values)
            return inside if isinstance(first_op, ast.In) else ~inside

        # 연쇄 비교(a < b < c)는 마디마다 계산해 and 로 잇는다
        result = None
        left = self.walk(node.left)
        for operator, comparator in zip(node.ops, node.comparators, strict=True):
            function = COMPARE_OPS.get(type(operator))
            if function is None:
                raise RuleSyntaxError(f"다룰 수 없는 비교 연산자: {type(operator).__name__}")
            right = self.walk(comparator)
            piece = function(left, right)
            result = piece if result is None else (self.as_bool(result) & self.as_bool(piece))
            left = right
        return result

    def call(self, node: ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
            name = getattr(node.func, "id", type(node.func).__name__)
            raise RuleSyntaxError(
                f"허용되지 않은 함수: {name}\n  쓸 수 있는 것: {', '.join(sorted(FUNCTIONS))}"
            )
        name = node.func.id
        args = [self.walk(a) for a in node.args]
        target = args[0]

        if name == "abs":
            return target.abs() if isinstance(target, pd.Series) else abs(target)
        if name == "len":
            return self.as_text(target).str.len()
        if name in ("year", "month", "day"):
            return self.date_part(target, name)
        if name == "date":
            return self.as_datetime(target).dt.strftime("%Y%m%d")
        if name == "time":
            return self.as_datetime(target).dt.strftime("%H%M%S")
        if name == "starts_with":
            return self.text_pair(target, args[1], "starts")
        if name == "contains":
            return self.text_pair(target, args[1], "contains")
        if name == "matches":
            # 정규식만은 고정된 값이어야 한다. 칸에서 온 패턴을 컴파일하면 자료가 코드가 된다.
            return self.as_text(target).str.match(self.plain(args[1]), na=False)
        raise RuleSyntaxError(f"구현되지 않은 함수: {name}")   # 방어선 — FUNCTIONS 와 어긋날 때

    def text_pair(self, left, right, how: str) -> pd.Series:
        """문자열 두 개를 맞댄다. 오른쪽이 **칸일 수도 있다.**

        `starts_with(rcept_no, rcept_dt)` 처럼 두 칸을 맞대는 규칙이 실제로 있다
        (접수번호 앞 여덟 자리가 접수일인지 보는 검사). 처음에는 오른쪽을 고정값으로만
        받게 막아 두어 그 규칙이 아예 돌지 않았는데, 규격에는 멀쩡히 적혀 있어서
        **검사가 도는 줄 알았지만 안 돌고 있는** 상태였다. 일부러 어긋난 행을 넣어 보는
        시험이 없었다면 못 찾았을 자리다.

        오른쪽이 고정값이면 pandas 의 벡터 연산을 쓰고, 칸이면 행 단위로 돈다.
        행 단위는 느리지만 이 경우는 반입 파일(수천~수만 행)에만 쓰이고, 920만 행짜리
        시세 규격에는 두 칸을 맞대는 규칙이 없다.
        """
        text = self.as_text(left)
        if not isinstance(right, pd.Series):
            if how == "starts":
                return text.str.startswith(right, na=False)
            return text.str.contains(right, regex=False, na=False)

        other = self.as_text(right)
        rows = [
            False if (pd.isna(a) or pd.isna(b))
            else (a.startswith(b) if how == "starts" else b in a)
            for a, b in zip(text, other, strict=True)
        ]
        return pd.Series(rows, index=self.frame.index)

    # ── 형 맞추기 ────────────────────────────────
    def date_part(self, value, part: str) -> pd.Series:
        """`YYYYMMDD` 문자열에서 연·월·일을 정수로 뽑는다.

        날짜형으로 바꾸지 않고 자릿수로 자르는 이유는, 규격이 날짜를 **문자열**로
        못박았기 때문이다. 한 칸을 어떤 규칙은 날짜로 어떤 규칙은 문자열로 전제하면
        어느 쪽으로 정해도 나머지가 깨진다.
        """
        text = self.as_text(value)
        cut = {"year": (0, 4), "month": (4, 6), "day": (6, 8)}[part]
        sliced = text.str.slice(*cut)
        return pd.to_numeric(sliced, errors="coerce")

    def as_text(self, value) -> pd.Series:
        if isinstance(value, pd.Series):
            return value.astype("string")
        return pd.Series([value] * len(self.frame), index=self.frame.index, dtype="string")

    def as_datetime(self, value) -> pd.Series:
        if not isinstance(value, pd.Series):
            value = pd.Series([value] * len(self.frame), index=self.frame.index)
        if pd.api.types.is_datetime64_any_dtype(value):
            return value
        return pd.to_datetime(value, errors="coerce", format="mixed")

    def as_bool(self, value) -> pd.Series:
        """판정을 불리언 시리즈로 맞춘다.

        ⚠️ 결측은 **참(통과)** 으로 둔다. 널 비교의 결과는 참도 거짓도 아닌데,
        거짓으로 두면 값이 없다는 이유로 위반이 되어 멀쩡한 행이 격리된다.
        널 자체를 문제 삼고 싶으면 규칙에 `X is not null` 을 직접 쓴다.
        """
        if isinstance(value, pd.Series):
            return value.fillna(True).astype(bool)
        return pd.Series(bool(value), index=self.frame.index)

    def plain(self, value):
        """함수의 두 번째 인자처럼 시리즈가 아니어야 하는 자리."""
        if isinstance(value, pd.Series):
            raise RuleSyntaxError("이 자리에는 칸이 아니라 고정된 값이 와야 한다")
        return value


def evaluate_rule(expr: str, frame: pd.DataFrame, *,
                  today: str = None, now: pd.Timestamp = None) -> pd.Series:
    """규칙 하나를 표에 적용해 **참=통과** 인 불리언 시리즈를 돌려준다.

    거짓인 자리가 그 규칙을 어긴 행이다. `severity` 가 `error` 면 격리하고
    `warn` 이면 통과시키되 보고서에 센다 — 그 판단은 부르는 쪽이 한다.
    """
    if now is None:
        now = pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None)
    if today is None:
        today = now.strftime("%Y%m%d")

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as error:
        raise RuleSyntaxError(f"식을 파싱할 수 없다: {expr}\n  {error}") from error

    evaluator = _Evaluator(frame, today=today, now=now)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = evaluator.walk(tree.body)
    return evaluator.as_bool(result)


def check_alias_invariant(spec: dict) -> None:
    """한 이름이 두 필드의 별칭이면 세운다.

    자료가 아니라 **규격이 틀린** 경우라, 검사기가 돌기 전에 알아야 한다.
    """
    owner = {}
    for field, names in spec.get("x-alphastack", {}).get("aliases", {}).items():
        for alias in names:
            key = alias.strip().lower()
            if key in owner and owner[key] != field:
                raise RuleSyntaxError(
                    f"별칭 {alias!r} 이 {owner[key]} 와 {field} 양쪽에 있다.\n"
                    "  한 이름은 한 필드에만 속해야 한다 — 정할 수 없으면 어느 쪽에도 넣지 말고\n"
                    "  ambiguousNames 에 적어 사람에게 묻게 한다."
                )
            owner[key] = field


__all__ = [
    "RuleSyntaxError",
    "FUNCTIONS",
    "evaluate_rule",
    "referenced_columns",
    "check_alias_invariant",
]


# `matches` 가 쓰는 정규식은 규격 파일에서 오므로, 잘못된 패턴이 조용히 통과하지 않게
# 미리 컴파일해 본다. 이 모듈을 import 하는 시점에 도는 것은 아니고 부르는 쪽이 쓴다.
def compile_pattern(pattern: str):
    try:
        return re.compile(pattern)
    except re.error as error:
        raise RuleSyntaxError(f"정규식이 잘못됐다: {pattern!r} — {error}") from error
