"""인증키 로딩 (공통 유틸)

KRX·KOSIS 두 곳의 인증키를 같은 규칙으로 읽는다.
`.key` 한 파일에 여러 키를 나란히 둘 수 있다.

    KRX_API_KEY = 발급받은_KRX_인증키
    KOSIS_API_KEY = 발급받은_KOSIS_인증키

읽는 순서는 **환경변수 → `.env` → `.key`** 다.
(배포 환경에서는 환경변수를, 로컬에서는 파일을 쓰는 흔한 구성이다.)

이 모듈은 다른 계층을 import 하지 않는 순수 함수 모음이라 순환 import 가 생기지 않는다.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Tuple

# 이 파일은 <루트>/common/ 안에 있으므로 parents[1] 이 프로젝트 루트다.
# (parents[0]=common, parents[1]=프로젝트 루트)
# ⚠️ 원본(data-service)에서는 common/ 였기에 parents[2] 였다. 평탄화하며 한 단계
#    얕아졌다 — 이 숫자를 되돌리면 자격증명 파일을 못 찾는다.
BASE_DIR = Path(__file__).resolve().parents[1]
KEY_FILE = BASE_DIR / ".key"      # 강의 원본과 같은 이름
ENV_FILE = BASE_DIR / ".env"      # dotenv 형식도 지원


def parse_key_text(text: str, names: Iterable[str], allow_bare: bool = False) -> str:
    """`.key` / `.env` 내용에서 원하는 이름의 값만 뽑아낸다.

    아래 형태를 모두 받아들인다.

        KOSIS_API_KEY = 발급받은_인증키    (등호 양쪽 공백 허용)
        KRX_AUTH_KEY="발급받은_인증키"     (따옴표 허용)
        발급받은_인증키                    (값만 한 줄 — `allow_bare=True` 일 때만)

    `allow_bare` 는 강의 원본처럼 키 값만 달랑 적어 둔 `.key` 를 지원하기 위한 것이다.
    한 파일에 키가 여러 개 있을 수 있으므로, 이름이 붙은 줄은 `names` 와 맞을 때만 쓴다.
    """
    wanted = {n.upper() for n in names}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):     # 빈 줄과 주석은 건너뛴다
            continue
        if "=" in line:
            name, _, value = line.partition("=")
            # 이름이 우리가 찾는 키가 아니면 무시한다 (한 파일에 여러 키가 섞여 있다)
            if name.strip().upper() not in wanted:
                continue
            value = value.strip().strip("\"'")   # 앞뒤 공백·따옴표 제거
        elif allow_bare:
            value = line                          # 등호가 없으면 줄 전체가 키다
        else:
            continue
        if value:
            return value
    return ""


def load_key(names: Iterable[str], allow_bare: bool = False) -> Tuple[str, str]:
    """인증키와 그 출처를 함께 돌려준다. 못 찾으면 `("", "none")`."""
    names = tuple(names)

    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value, f"환경변수 {name}"

    for path, label in ((ENV_FILE, ".env"), (KEY_FILE, ".key")):
        if path.exists():
            key = parse_key_text(path.read_text(encoding="utf-8"), names, allow_bare)
            if key:
                return key, label

    return "", "none"


def mask(key: str) -> str:
    """키를 화면·로그에 보여줄 때 쓰는 마스킹. 앞 4자만 남긴다.

    실습 화면에서 "키가 제대로 읽혔는지"만 확인하면 되므로 전체 값을 내보내지 않는다.
    """
    if not key:
        return ""
    return f"{key[:4]}{'*' * max(len(key) - 4, 0)}"
