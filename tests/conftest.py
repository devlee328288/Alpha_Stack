"""테스트 전역 안전장치.

**무엇을 막는가.** 테스트가 실수로 진짜 수집 DB(`data/krx_cache.db`)에 쓰는 것을 막는다.

가상의 위험이 아니다. 2026-08-26 에 실제로 겪었다 — 수집 대장을 붙이면서 대장 함수에
DB 경로를 안 넘긴 자리가 하나 있었다. 테스트는 `DB_PATH` 를 임시 파일로 갈아 끼웠지만
그 함수만 기본 경로를 봤고, `pytest` 한 번에 진짜 DB 에 20행이 들어갔다. 이번에는 내용이
사실이라 무해했지만, 같은 구멍으로 **백필 이력이 지워졌다면 16년치를 다시 받아야 했다.**

개별 테스트의 `monkeypatch` 를 믿지 않는 이유가 이것이다. 경로를 갈아 끼우는 자리가
여러 곳이면 언젠가 하나를 빠뜨리고, **빠뜨려도 테스트는 통과한다.** 그래서 프로세스
전체에서 진짜 경로가 아예 나오지 않게 막는다.

⚠️ 이 파일은 **모듈이 import 되기 전에** 환경변수를 세팅해야 한다. `krx_store.DB_PATH`
   같은 모듈 전역이 import 시점에 경로를 굳히기 때문이다. pytest 는 테스트 모듈보다
   `conftest.py` 를 먼저 읽으므로 여기 맨 위에서 하면 늦지 않는다.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# 세션마다 다른 폴더를 쓴다 — 어제 돌린 테스트가 남긴 DB 를 오늘 테스트가 물려받으면
# 통과·실패가 실행 순서에 따라 달라진다.
_TEST_DB_DIR = Path(tempfile.mkdtemp(prefix="alphastack-test-"))

# `common.paths.krx_db_path()` 가 가장 먼저 보는 값이라 이 한 줄이 모든 경로를 덮는다.
# `setdefault` 가 아니라 **강제 대입**이다. 밖에서 잘못 지정된 값을 물려받으면
# 이 파일이 막으려는 사고가 그대로 일어난다.
os.environ["KRX_DB_PATH"] = str(_TEST_DB_DIR / "krx_cache.db")


def pytest_report_header() -> str:
    """어느 DB 를 쓰는지 실행할 때마다 보여 준다 — 안전장치는 보여야 신뢰가 된다."""
    return f"수집 DB (테스트 전용): {os.environ['KRX_DB_PATH']}"
