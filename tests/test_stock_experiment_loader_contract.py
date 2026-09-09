"""실험 로더가 읽는 칸 ⊇ 공급 층이 요구하는 칸 — 반출을 따라가야 하는 곳이 또 하나 있었다.

2026-09-09 실제: PR #189 가 `supply/stock_training_universe.py` 의 보통주 판정을 이름 규칙에서
HF 칸 `kind_stkcert_tp_nm` 으로 바꾸면서 `DAILY_REQUIRED` 에 그 칸을 넣었는데,
`scripts/run_stock_model_experiment.py` 가 parquet 에서 읽는 `DAILY_COLUMNS` 는 그대로였다.
시험은 전부 초록이었다 — 공급 층 시험은 칸이 있는 표를 손으로 만들어 넘기고, 로더는 시험이
없었기 때문이다. 같은 날 오후 `load_stock_model_dataset` 이 `종목 후보 입력 열이 없습니다:
['kind_stkcert_tp_nm']` 로 멈췄다.

"파생 칸을 늘리면 따라와야 하는 곳" 이 넷이 됐다 — `COLUMN_NOTES` · `EXPORT_DAILY_COLUMNS` ·
`verify_hf_dataset._attach_export_derived` · 그리고 **모델 파트 로더**. 이 시험은 마지막 것을
코드로 못박는다.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.run_stock_model_experiment as exp  # noqa: E402
from supply import stock_training_universe as stu  # noqa: E402


def test_로더가_읽는_일별시세_칸이_공급_층_요구_칸을_전부_담는다():
    빠짐 = set(stu.DAILY_REQUIRED) - set(exp.DAILY_COLUMNS)
    assert not 빠짐, (
        f"`build_sector_candidate_frame` 이 요구하는데 로더가 안 읽는 칸: {sorted(빠짐)}\n"
        "  할 일: scripts/run_stock_model_experiment.py 의 DAILY_COLUMNS 에 더한다."
    )


def test_로더가_읽는_지수_칸이_공급_층_요구_칸을_전부_담는다():
    빠짐 = set(stu.INDEX_REQUIRED) - set(exp.INDEX_COLUMNS)
    assert not 빠짐, f"지수 표에서 로더가 안 읽는 칸: {sorted(빠짐)}"


def test_로더가_읽는_칸은_반출본_카드에_실제로_있다():
    """반출이 싣지 않는 칸을 읽으려 하면 pyarrow 가 먼저 멈춘다 — 카드 칸 사전과 맞댄다.

    반출본 칸의 정본은 `scripts/upload_to_hf.py` 의 `COLUMN_NOTES` 다
    (`test_hf_card_notes.py` 가 "실리는 칸은 전부 설명이 있다" 를 지킨다).
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "upload_to_hf", ROOT / "scripts" / "upload_to_hf.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    없음 = set(exp.DAILY_COLUMNS) - set(mod.COLUMN_NOTES)
    assert not 없음, f"로더가 읽으려는데 반출본 카드에 없는 칸: {sorted(없음)}"
