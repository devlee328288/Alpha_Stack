"""경로 기준점을 한 곳에서 정한다.

**왜 필요한가.** 저장소 루트를 파일마다 따로 계산하면, 파일을 옮기는 순간 조용히 다른
곳을 가리킨다. 예외도 안 나고 서버도 뜬다 — 그냥 빈 결과가 나올 뿐이라 알아채기 어렵다.

이건 가상의 위험이 아니다. 이 저장소를 만들 때 실제로 겪었다. 원본에서 코드를 옮겨 오며
`ingest/clients/` 를 `ingest/clients/` 로 바꿨는데, 그 안의 `parents[2]` 는 원래 `app` 을
건너뛰어 루트를 가리키도록 세어 둔 숫자였다. 중간에 폴더가 하나 끼자 루트가 아닌 곳을
가리켰지만 **import 는 멀쩡히 통과했다.** 파일을 실제로 읽는 순간에야 드러난다.

그래서 새 코드는 깊이를 세지 말고 여기 상수를 쓴다.

    from common.paths import DATA_DIR, ARTIFACTS_DIR

⚠️ 이 파일을 다른 깊이로 옮기면 아래 `parents[1]` 도 함께 고쳐야 한다.
   현재 위치는 `<루트>/common/paths.py` 다.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# 이 파일은 <루트>/common/paths.py 이므로 parents[1] 이 저장소 루트다.
# (parents[0]=common, parents[1]=저장소 루트)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# 시세·유니버스 등 입력 자료. universe_core.json 만 git 이 추적하고 나머지는 무시된다.
DATA_DIR: Path = PROJECT_ROOT / "data"

# 수집·유니버스 구축 스크립트.
SCRIPTS_DIR: Path = PROJECT_ROOT / "scripts"

# 학습 산출물 — 적합된 모델·피처 표·백테스트 결과.
# ⚠️ git 이 추적하지 않는다(.gitignore). **재현은 파이프라인으로 하지 파일 공유로 하지
#    않는다.** 팀원끼리 pkl 을 주고받기 시작하면 "누구 모델이 맞나"를 아무도 답할 수 없다.
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"

# 최소 API 가 내보낼 정적 자산이 생기면 여기에 둔다. 아직 비어 있다.
STATIC_DIR: Path = PROJECT_ROOT / "static"

# 사람이 읽는 검사 결과 — 품질 게이트(D-10)의 `data_quality.json`,
# 호출 예산(D-04)의 `quota_usage.json` 이 여기로 나간다.
#
# ⚠️ **여기 있는 것은 보고서지 상태의 정본이 아니다.** 워터마크·예산의 실제 값은
#    SQLite 안에 있고(같은 트랜잭션에서 갱신돼야 재개가 성립한다), 이 폴더의 JSON 은
#    거기서 파생 생성된다. 이 파일을 고쳐도 수집기는 달라지지 않는다.
REPORTS_DIR: Path = PROJECT_ROOT / "reports"


def krx_db_path() -> Path:
    """수집 DB(`data/krx_cache.db`) 의 경로. **쓸 수 있는 곳**이어야 한다.

    평소에는 `data/krx_cache.db` 지만, 서버리스(Vercel 등)에 올리면 배포된 파일이
    **읽기 전용**이라 그 자리에 DB 를 만들 수 없다. SQLite 는 파일을 열 때 없으면 만들려 하고,
    `PRAGMA journal_mode=WAL` 도 쓰기라서 곧바로 예외가 난다. 그래서 쓰기가 막혀 있으면
    임시 폴더로 옮긴다 — 500 으로 죽는 것보다 빈 DB 로 안내 문구를 내는 편이 낫다.

    `KRX_DB_PATH` 환경변수로 직접 지정할 수도 있다.

    ⚠️ **이 함수가 이 경로의 유일한 정의다.** 원래는 `ingest/store/krx_store.py` 안에
    있었는데, `common/budget.py`(D-04 호출 예산)가 같은 파일을 열어야 하면서 문제가 됐다 —
    `ingest/store` 는 `ingest/clients` 를 import 하므로, 예산을 쓰는 clients 가 store 를
    import 하면 **순환**이 된다. 그래서 두 계층 아래인 여기로 내렸다.
    경로 계산을 두 곳에 두면 언젠가 서로 다른 파일을 가리킨다 — 이 파일 맨 위 주석이
    경고하는 바로 그 사고다.
    """
    override = os.getenv("KRX_DB_PATH", "").strip()
    if override:
        path = Path(override)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    default = DATA_DIR / "krx_cache.db"
    try:
        # 최초 실행 시 data/ 가 없으면 sqlite3.connect 가 실패하므로 미리 만들어 둔다.
        default.parent.mkdir(parents=True, exist_ok=True)
        # 폴더가 있어도 쓰기 권한이 없을 수 있다. 실제로 쓸 수 있는지 확인한다.
        if os.access(default.parent, os.W_OK):
            return default
    except OSError:
        pass          # 폴더를 만들 수 없는 환경 (읽기 전용 배포)

    return Path(tempfile.gettempdir()) / "krx_cache.db"
