"""저장 계층이 공유하는 것들 테스트 — 자물쇠와 마이그레이션 번호

**왜 이 테스트가 필요한가.** 둘 다 **틀려도 대부분의 실행에서는 아무 일도 안 일어난다.**

- 자물쇠가 둘로 갈리면 평소에는 잘 돌고 **동시 쓰기가 겹치는 순간에만** `database is
  locked` 가 뜬다. 재현이 안 되는 버그가 된다
- 마이그레이션 번호가 겹치면 **새 DB 는 멀쩡하고 이미 그 버전을 지난 DB 만** 조용히
  항목을 건너뛴다. 내 컴퓨터에서는 절대 안 보인다
"""

from __future__ import annotations

import ast
from pathlib import Path

from ingest.store import krx_index, krx_store, migrations
from ingest.store.sqlite_db import NEXT_MIGRATION_VERSION, write_lock

루트 = Path(__file__).resolve().parents[1]


# ── 자물쇠 ──────────────────────────────────────────────────────────────────

def test_같은_DB_에_쓰는_모듈은_자물쇠를_공유한다():
    """🔴 자물쇠가 둘이면 자물쇠가 없는 것과 같다.

    `krx_store` 와 `krx_index` 는 같은 파일에 쓴다. 각자 `threading.Lock()` 을 만들면
    서로를 모른 채 동시에 INSERT 하고, SQLite 는 그때만 `database is locked` 를 낸다.
    """
    assert krx_store._write_lock is write_lock
    assert krx_index.write_lock is write_lock


#: 쓰기 자물쇠로 쓰이는 이름들. 이 이름으로 **새 `Lock()` 을 만들면** 공유가 깨진다.
쓰기자물쇠_이름 = {"write_lock", "_write_lock"}


def test_저장_모듈이_쓰기_자물쇠를_새로_만들지_않는다():
    """새 저장 모듈이 자기 `write_lock` 을 만들면 여기서 걸린다.

    ⚠️ **모든 `Lock()` 을 막지는 않는다.** 저장 계층에는 DB 쓰기와 무관한 자물쇠가
       이미 셋 있고 셋 다 정당하다 — 스키마 확인 중복 방지(`_schema_lock`), 축약본
       메모리 캐시(`krx_bundle._lock`), 라이브 조회 캐시(`_live_lock`). 그것들은
       공유할 이유가 없다.

    막는 것은 **이름이 쓰기 자물쇠인데 새로 만든 것** 하나다. 별칭 대입
    (`_write_lock = write_lock`)은 `Lock()` 호출이 아니라서 통과한다.
    """
    위반 = []
    for 파일 in (루트 / "ingest" / "store").glob("*.py"):
        if 파일.name == "sqlite_db.py":          # 여기가 원본이다
            continue
        나무 = ast.parse(파일.read_text(encoding="utf-8"), filename=str(파일))
        for 노드 in ast.walk(나무):
            if not isinstance(노드, ast.Assign):
                continue
            이름 = {t.id for t in 노드.targets if isinstance(t, ast.Name)}
            if not (이름 & 쓰기자물쇠_이름):
                continue
            값 = 노드.value
            if (isinstance(값, ast.Call) and isinstance(값.func, ast.Attribute)
                    and 값.func.attr == "Lock"):
                위반.append(f"{파일.name}:{노드.lineno}")
    assert not 위반, (
        "저장 모듈이 쓰기 자물쇠를 새로 만들었습니다: " + ", ".join(위반) + "\n"
        "  같은 DB 파일에 쓴다면 ingest.store.sqlite_db.write_lock 을 import 해서 쓰세요."
    )


# ── 마이그레이션 번호 ───────────────────────────────────────────────────────

def test_예약된_다음_번호가_실제_최신보다_앞서_있다():
    """🔴 번호는 이름이 아니라 **인덱스**다.

    두 갈래가 각자 "다음은 v5" 로 붙인 채 합쳐지면, 이미 v5 를 적용한 DB 는 나중에
    v5 자리에 들어온 항목을 **영원히 건너뛴다.** 예외도 경고도 없다.

    지금 v5·v6 을 예약해 뒀으므로 다음 빈 번호는 **적어도 v7** 이어야 한다.
    """
    assert NEXT_MIGRATION_VERSION > migrations.LATEST_VERSION, (
        f"예약 번호({NEXT_MIGRATION_VERSION})가 실제 최신({migrations.LATEST_VERSION})보다 "
        "앞서지 않습니다. 마이그레이션을 더했다면 sqlite_db.NEXT_MIGRATION_VERSION 과 "
        "MIGRATIONS 위 배정표를 함께 고치세요."
    )


def test_마이그레이션_이름이_겹치지_않는다():
    """같은 이름이 둘이면 무엇이 적용됐는지 로그로 구별할 수 없다."""
    이름들 = [name for name, _ in migrations.MIGRATIONS]

    assert len(이름들) == len(set(이름들)), f"이름이 겹칩니다: {이름들}"


def test_마이그레이션_번호와_이름이_어긋나지_않는다():
    """`v3` 라고 적힌 항목이 네 번째 자리에 있으면 사람이 반드시 헷갈린다."""
    어긋남 = [(i + 1, name) for i, (name, _) in enumerate(migrations.MIGRATIONS)
              if not name.startswith(f"v{i + 1}:")]

    assert not 어긋남, f"자리와 이름이 다릅니다: {어긋남}"
