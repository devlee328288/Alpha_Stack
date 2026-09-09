"""데이터셋 카드의 칸 설명을 못박는다 — 새 칸이 뜻 없이 나가지 못하게.

## 왜 이 테스트가 필요한가

카드에서 `full/daily_price_dev.parquet` 은 접히지 않고 **펼쳐지는 두 표 중 하나**다
(`CARD_EXPANDED`). 팀원이 자료를 받아 가장 먼저 마주치는 표다. 그런데 칸 설명은
`COLUMN_NOTES` 라는 손으로 적는 사전에서 오고, **없으면 그냥 빈칸으로 나간다.**
경고도 오류도 없다.

실제로 그렇게 나갔다 — `industry_ambiguous` 는 2026-09-05 첫 반출부터 09-07 까지
세 번의 배포에서 "무엇인가" 칸이 비어 있었다. 아무도 몰랐던 것은, 빈칸이 **정상적인
표처럼 보이기 때문**이다.

품질 플래그 넷은 그렇게 나가면 안 되는 칸이다. `is_extreme_return` 은 이름만 보면
"극단값이니 빼라" 로 읽히지만 실제로는 **"진짜 사건이니 남겨라"** 다. 표에서 뜻 없이
마주친 사람은 이름대로 거르고, 팀 기준선과 다른 표본으로 학습한다. 그 사고를
막으려고 넣은 칸이 그 사고를 일으키는 셈이 된다.

## 무엇을 고정하나

    ① 반출본에 실리는 칸은 전부 COLUMN_NOTES 에 설명이 있다 (빈 문자열도 없다)
    ② 임계값은 글자로 박지 않고 상수에서 온다 — 상수를 바꾸면 카드도 따라 바뀐다
    ③ 거르라는 것과 남기라는 것이 설명에서 갈린다

②가 따로 있는 이유는, 손으로 적은 숫자는 **자료가 바뀌어도 안 바뀌기 때문**이다.
카드 전체가 `MANIFEST.json`·`PROFILE.json` 에서 숫자를 읽어 오는 것도 같은 이유다.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from supply.adj_quality import (  # noqa: E402
    EXTREME_RETURN_PCT,
    FLAG_COLUMNS,
    SUSPECT_GAP_TOLERANCE,
)
from supply.sector import INDUSTRY_COLUMNS  # noqa: E402


def _load():
    """스크립트를 모듈로 읽는다 — `scripts/` 는 패키지가 아니다."""
    spec = importlib.util.spec_from_file_location(
        "upload_to_hf", ROOT / "scripts" / "upload_to_hf.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


U = _load()


# ══════════════════════════════════════════════════════════════════════════
# ① 설명이 비어 있지 않다
# ══════════════════════════════════════════════════════════════════════════
def test_품질_플래그_넷은_전부_설명이_있다():
    빈것 = [칸 for 칸 in FLAG_COLUMNS if not U.COLUMN_NOTES.get(칸, "").strip()]
    assert not 빈것, f"카드에 뜻 없이 나가는 칸: {빈것}"


def test_업종_칸_넷은_전부_설명이_있다():
    """`industry_ambiguous` 가 세 번의 배포에서 비어 나갔던 자리다."""
    빈것 = [칸 for 칸 in INDUSTRY_COLUMNS if not U.COLUMN_NOTES.get(칸, "").strip()]
    assert not 빈것, f"카드에 뜻 없이 나가는 칸: {빈것}"


# ══════════════════════════════════════════════════════════════════════════
# ② 임계값은 상수에서 온다
# ══════════════════════════════════════════════════════════════════════════
def test_의심_임계는_손으로_적지_않고_상수에서_온다():
    """`SUSPECT_GAP_TOLERANCE` 를 바꾸면 카드 설명도 따라 바뀌어야 한다."""
    assert f"{SUSPECT_GAP_TOLERANCE:g}%p" in U.COLUMN_NOTES["is_adj_suspect"]


def test_극단_임계는_손으로_적지_않고_상수에서_온다():
    assert f"{EXTREME_RETURN_PCT:g}%" in U.COLUMN_NOTES["is_extreme_return"]


# ══════════════════════════════════════════════════════════════════════════
# ③ 거르라는 것과 남기라는 것이 갈린다
# ══════════════════════════════════════════════════════════════════════════
def test_거르는_것은_의심_하나뿐이라고_적혀_있다():
    assert "거른다" in U.COLUMN_NOTES["is_adj_suspect"]
    assert "거르지 않는다" in U.COLUMN_NOTES["is_extreme_return"]


def test_표로_그려도_뜻_칸이_비지_않는다():
    """`_column_table` 이 실제로 뜻을 채우는지 — 사전에 있어도 못 붙으면 소용없다."""
    파일 = {
        "칸들": [
            {"이름": 칸, "형": "bool", "결측": 0, "결측률": 0.0,
             "분포": {"False": 100}}
            for 칸 in (*FLAG_COLUMNS, *INDUSTRY_COLUMNS)
        ]
    }
    for 줄 in U._column_table(파일).splitlines()[2:]:
        칸이름 = 줄.split("|")[1].strip().strip("`")
        뜻 = 줄.rsplit("|", 2)[1].strip()
        assert 뜻, f"`{칸이름}` 의 '무엇인가' 가 비어 있다"


# ══════════════════════════════════════════════════════════════════════════
# ④ 같은 한 행이 두 칸에 들어 있으면 경고도 두 칸에
#
# 008080 이 2013-09-11 에 거래정지 중 `1`원에서 재개된 날, 등락률이 6,699,900% 다.
# 그 값은 `change_rate` 에도 `adj_return_1d` 에도 똑같이 들어 있다. 경고를 한쪽에만
# 달면 다른 쪽을 그대로 피처에 넣는다 — 그리고 그 행은 `is_adj_suspect` 가 아니라
# `is_extreme_return` 이라, 우리가 **남기라고** 안내하는 행이다.
# ══════════════════════════════════════════════════════════════════════════
def _칸(이름: str, **덮어쓰기):
    c = {"이름": 이름, "형": "double", "결측": 0, "결측률": 0.0}
    c.update(덮어쓰기)
    return c


def _경고들(칸: dict) -> list[str]:
    return [문구 for 이름, 판정, 문구 in U.COLUMN_WARNINGS
            if 이름 == 칸["이름"] and 판정(칸)]


def test_거래정지_재개행은_두_칸_모두_경고가_붙는다():
    큰값 = {"min": -98.08, "max": 6_699_900}
    for 칸이름 in ("change_rate", "adj_return_1d"):
        assert _경고들(_칸(칸이름, **큰값)), f"`{칸이름}` 에 경고가 없다"


def test_그_행이_없는_파일에는_경고를_붙이지_않는다():
    """지수 파일의 등락률은 -15 ~ 23 이다. 거기에 붙이면 카드가 거짓말을 한다."""
    보통 = {"min": -15.38, "max": 23.42}
    for 칸이름 in ("change_rate", "adj_return_1d"):
        assert not _경고들(_칸(칸이름, **보통))
