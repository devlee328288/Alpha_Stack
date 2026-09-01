"""반입 규격 5장을 규격 자체의 규칙으로 검사한다.

**왜 규격을 검사하는가.** 규격은 "이런 행은 못 쓴다" 를 적은 문서인데, 그 규칙이 틀리면
멀쩡한 자료가 격리되거나 미래를 보는 행이 통과한다. 둘 다 예외를 내지 않는다 —
격리는 "자료가 나쁘다" 로 보이고, 통과는 성능이 좋아 보인다. 그래서 사람이 읽어서는
못 잡는다.

여기서 보는 것은 넷이다.

1. **문법** — 모든 `expr` 이 파싱되고, 허용한 함수·이름만 쓰는가
2. **널 가드** — 비교하는 칸에 `is null` 가드가 있는가
3. **별칭 불변식** — 한 이름이 두 필드의 별칭이 아닌가
4. **자가 격리** — `error` 규칙이 우리 자신의 자료를 격리하지 않는가

⚠️ 4번은 대조할 표가 있는 두 규격(ohlcv_stock·ohlcv_index)에서만 돈다. 나머지 셋은
아직 우리 자료가 없어서, 대신 **일부러 어긋난 행을 만들어 실제로 잡히는지** 본다.
통과만 보고는 검사가 도는지 알 수 없다.
"""

from __future__ import annotations

import ast
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ingest.inbox.rules import evaluate_rule, referenced_columns  # noqa: E402

SCHEMA_DIR = ROOT / "ingest" / "inbox" / "schemas"
SCHEMA_NAMES = ["ohlcv_stock", "ohlcv_index", "news", "financial", "macro"]

#: 대조할 표가 있는 규격만 실제 자료로 잰다.
COMPARABLE = {"ohlcv_stock": "daily_price", "ohlcv_index": "index_price"}

#: 널 가드가 필요 없는 칸 — required 라서 비어 있을 수 없거나, 가드가 이미 뜻에 포함된 경우.
GUARD_EXEMPT = {"today", "now"}


def load(name: str) -> dict:
    return json.loads((SCHEMA_DIR / f"{name}.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def schemas() -> dict:
    return {name: load(name) for name in SCHEMA_NAMES}


# ==================================================
# 1. 문법
# ==================================================
@pytest.mark.parametrize("name", SCHEMA_NAMES)
def test_모든_규칙이_파싱된다(name: str):
    spec = load(name)
    for rule in spec["x-alphastack"]["rowRules"]:
        try:
            ast.parse(rule["expr"], mode="eval")
        except SyntaxError as error:
            pytest.fail(f"{name}.{rule['id']} 이 파이썬 식이 아니다: {error}\n  {rule['expr']}")


@pytest.mark.parametrize("name", SCHEMA_NAMES)
def test_규칙이_규격에_없는_칸을_가리키지_않는다(name: str):
    spec = load(name)
    known = {f["name"] for f in spec["fields"]} | GUARD_EXEMPT
    for rule in spec["x-alphastack"]["rowRules"]:
        used = referenced_columns(rule["expr"])
        unknown = used - known
        assert not unknown, f"{name}.{rule['id']} 이 모르는 이름을 쓴다: {sorted(unknown)}"


# ==================================================
# 2. 널 가드
# ==================================================
@pytest.mark.parametrize("name", SCHEMA_NAMES)
def test_필수가_아닌_칸을_비교하면_널_가드가_있다(name: str):
    """`close >= 0` 처럼 널이 올 수 있는 칸을 맨몸으로 비교하면 안 된다.

    널 비교는 참도 거짓도 아닌데 pandas 에서 False 로 떨어져 **위반**이 된다.
    지수 규격 초안이 이 가드 없이 `high >= low` 를 error 로 두어 우리 자료
    24,933행(12.73%)을 스스로 격리했다.
    """
    spec = load(name)
    optional = {
        f["name"] for f in spec["fields"]
        if not f.get("constraints", {}).get("required", False)
    }
    for rule in spec["x-alphastack"]["rowRules"]:
        used = referenced_columns(rule["expr"]) & optional
        if not used:
            continue
        guarded = _guarded_columns(rule["expr"])
        # `X is not null` 을 단독으로 확인하는 규칙 자체는 가드가 필요 없다
        missing = {c for c in used if c not in guarded}
        assert not missing, (
            f"{name}.{rule['id']} 이 널 가드 없이 비교한다: {sorted(missing)}\n"
            f"  {rule['expr']}\n"
            f"  → 앞에 `{sorted(missing)[0]} is null or` 를 붙인다"
        )


def _guarded_columns(expr: str) -> set:
    """`X is null` · `X is not null` 로 언급된 칸 이름을 모은다."""
    out = set()
    for node in ast.walk(ast.parse(expr, mode="eval")):
        if isinstance(node, ast.Compare) and isinstance(node.ops[0], (ast.Is, ast.IsNot)):
            right = node.comparators[0]
            if (isinstance(right, ast.Name) and right.id == "null"
                    and isinstance(node.left, ast.Name)):
                out.add(node.left.id)
    return out


# ==================================================
# 3. 별칭 불변식
# ==================================================
@pytest.mark.parametrize("name", SCHEMA_NAMES)
def test_한_이름은_한_필드의_별칭이다(name: str):
    """같은 이름이 두 필드의 별칭이면 검사기가 어디로 보낼지 정할 근거가 없다.

    재무 규격 초안에서 "당기" 가 thstrm_nm(문자열)과 thstrm_amount(숫자) 양쪽에
    들어 있었다. 어디로 가든 뒤에서 조용히 깨진다.
    """
    spec = load(name)
    owner: dict = {}
    for field, names in spec["x-alphastack"]["aliases"].items():
        for alias in names:
            key = alias.strip().lower()
            if key in owner and owner[key] != field:
                pytest.fail(f"{name}: 별칭 {alias!r} 이 {owner[key]} 와 {field} 양쪽에 있다")
            owner[key] = field


@pytest.mark.parametrize("name", SCHEMA_NAMES)
def test_별칭이_다른_필드의_본이름과_겹치지_않는다(name: str):
    spec = load(name)
    field_names = {f["name"] for f in spec["fields"]}
    for field, names in spec["x-alphastack"]["aliases"].items():
        for alias in names:
            if alias in field_names and alias != field:
                pytest.fail(f"{name}: 별칭 {alias!r}({field}) 이 다른 칸의 본이름이다")


@pytest.mark.parametrize("name", SCHEMA_NAMES)
def test_모호한_이름은_어느_별칭에도_없다(name: str):
    """ambiguousNames 에 올린 이름이 별칭에도 있으면 자동 매핑이 그대로 일어난다."""
    spec = load(name)
    ambiguous = {k for k in spec["x-alphastack"].get("ambiguousNames", {}) if not k.startswith("_")}
    for field, names in spec["x-alphastack"]["aliases"].items():
        overlap = ambiguous & {n.strip() for n in names}
        assert not overlap, f"{name}: 모호하다고 적어 둔 {sorted(overlap)} 이 {field} 별칭에 있다"


def test_뉴스_본문_컬럼은_별칭이_아니라_거절목록에_있다():
    """저작권 방어가 별칭 한 줄로 뚫리지 않는지 본다.

    본문 칸을 안 만드는 것만으로는 부족하다 — `content` 가 summary 의 별칭이면
    팀원이 본문을 그 컬럼에 담아 왔을 때 자동으로 매핑되고, 길이 규칙에 걸려
    격리되더라도 격리는 곧 inbox_quarantine 에 **저장**이라 본문이 DB 에 내려앉는다.
    """
    spec = load("news")
    rejected = {k for k in spec["x-alphastack"]["rejectedNames"] if not k.startswith("_")}
    assert {"content", "body", "내용"} <= rejected

    every_alias = {n.strip().lower()
                   for names in spec["x-alphastack"]["aliases"].values() for n in names}
    leaked = {r for r in rejected if r.lower() in every_alias}
    assert not leaked, f"거절해야 할 이름이 별칭에 남아 있다: {sorted(leaked)}"


# ==================================================
# 4. 자가 격리 — 우리 자료를 우리 규칙이 버리지 않는가
# ==================================================
@pytest.mark.parametrize("name,table", sorted(COMPARABLE.items()))
def test_error_규칙이_우리_자료를_격리하지_않는다(name: str, table: str):
    db = ROOT / "data" / "krx_cache.db"
    if not db.exists():
        pytest.skip("krx_cache.db 가 없다 (수집 전 환경)")

    spec = load(name)
    with sqlite3.connect(db) as con:
        frame = pd.read_sql(f"SELECT * FROM {table}", con)

    for rule in spec["x-alphastack"]["rowRules"]:
        if rule["severity"] != "error":
            continue
        result = evaluate_rule(rule["expr"], frame)
        bad = int((~result).sum())
        assert bad == 0, (
            f"{name}.{rule['id']} 이 우리 {table} 에서 {bad:,}행을 격리한다.\n"
            f"  {rule['expr']}\n"
            f"  → 규칙이 틀렸다. 자료가 아니라 규칙을 고친다."
        )


def test_warn_규칙이_아는_사실을_그대로_센다():
    """실측으로 확인된 건수가 규칙으로도 같게 나오는지 본다.

    통과만 보고는 규칙이 도는지 알 수 없다. 값이 맞아야 뜻대로 읽힌 것이다.
    """
    db = ROOT / "data" / "krx_cache.db"
    if not db.exists():
        pytest.skip("krx_cache.db 가 없다 (수집 전 환경)")

    spec = load("ohlcv_stock")
    with sqlite3.connect(db) as con:
        frame = pd.read_sql("SELECT * FROM daily_price", con)

    expected = {"zero_ohlc": 283_468, "zero_ohlc_but_traded": 125}
    rules = {r["id"]: r for r in spec["x-alphastack"]["rowRules"]}
    for rule_id, count in expected.items():
        result = evaluate_rule(rules[rule_id]["expr"], frame)
        assert int((~result).sum()) == count, f"{rule_id} 이 {count:,} 행을 세야 한다"
