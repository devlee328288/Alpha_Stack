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
