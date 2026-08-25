"""`/tmp` best-effort 캐시 (저장 계층)

배포 환경(Vercel 서버리스)에서 **쓸 수 있는 경로는 `/tmp` 뿐**이다. 번들은 읽기 전용이라
파일을 만들 수 없다. 그래서 느린 외부 호출(야후 7초·DART 0.15초·FRED 0.3초)의 결과를
여기에 잠깐 얹어 두고 다시 쓴다.

**이 캐시는 속도만 담당한다** (명세서 §1.3·§8.1).
캐시가 없어도, 읽기에 실패해도, 쓰기가 막혀도 앱은 똑같이 동작해야 한다.
그래서 이 모듈의 모든 함수는 **예외를 밖으로 내보내지 않는다.** 실패하면 조용히
"없음"으로 처리하고 호출한 쪽이 원본을 다시 부르게 둔다.

    from app.repositories import tmp_cache

    hit = tmp_cache.read("dart", "00126380_2024_11011", ttl=86400)
    if hit is None:
        hit = 원본을_부른다()
        tmp_cache.write("dart", "00126380_2024_11011", hit)

왜 서버리스에서 캐시를 믿으면 안 되는가
--------------------------------------
인스턴스가 요청마다 새로 뜰 수 있다. 방금 쓴 파일이 다음 요청에서 없을 수 있고,
반대로 몇 시간 전 인스턴스가 살아 있어 오래된 값이 남아 있을 수도 있다.
**그래서 TTL 을 읽는 쪽이 정하고**(`read` 의 인자), 캐시 미스가 결과를 바꾸지 않도록
원본 호출과 같은 값을 넣는다.

네임스페이스 (명세서 §8.1)
--------------------------
| 이름 | 담는 것 | 권장 TTL |
|---|---|---|
| `ts` | yfinance 일봉 | 6시간 |
| `dart` | DART 재무제표 | 24시간 |
| `dashboard` | 대시보드 티커·요약 | 3~5분 |
| `market` | 차트 화면 시계열 | 10분 |
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

# `/tmp/api-test/{네임스페이스}/{이름}.json` 구조로 담는다.
# `tempfile.gettempdir()` 를 쓰는 이유 — 리눅스는 `/tmp`, 윈도우는 `%TEMP%` 라
# 로컬(윈도우·WSL)과 배포(리눅스)에서 같은 코드가 그대로 돈다.
CACHE_ROOT = Path(tempfile.gettempdir()) / "api-test"

# 파일 이름에 쓸 수 없는 글자를 막는다. 티커·종목코드가 그대로 이름이 되므로
# `../` 같은 경로 탈출이 섞여 들어오지 않도록 여기서 한 번 거른다.
SAFE_NAME = re.compile(r"[^A-Za-z0-9._가-힣-]")

# 권장 TTL — 읽는 쪽이 그대로 쓸 수 있게 상수로 둔다 (명세서 §8.1)
TTL_TIMESERIES = 6 * 3600        # yfinance 일봉 6시간
TTL_DART = 24 * 3600             # DART 재무제표 24시간
TTL_TICKER = 180                 # 대시보드 티커바 3분
TTL_SUMMARY = 300                # 대시보드 카드 5분
TTL_MARKET = 600                 # 차트 화면 시계열 10분


def _safe(text: str) -> str:
    """파일 이름으로 쓸 수 있게 다듬는다. 너무 길면 자른다(파일시스템 한도 방어)."""
    cleaned = SAFE_NAME.sub("_", (text or "").strip())
    return cleaned[:120] or "_"


def _path(namespace: str, name: str) -> Path:
    return CACHE_ROOT / _safe(namespace) / f"{_safe(name)}.json"


def read(namespace: str, name: str, ttl: int) -> Optional[dict]:
    """`ttl` 초 안에 저장한 값이 있으면 돌려준다. 없거나 읽기에 실패하면 `None`.

    실패를 예외로 올리지 않는 것이 이 모듈의 계약이다 — 캐시는 있으면 좋고 없어도 그만이다.
    """
    try:
        path = _path(namespace, name)
        if not path.exists():
            return None
        if time.time() - path.stat().st_mtime > ttl:
            return None                       # 오래된 값은 없는 것으로 친다
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write(namespace: str, name: str, payload: dict) -> bool:
    """값을 저장한다. 성공하면 `True`. 실패해도 예외를 내보내지 않는다.

    **임시 파일에 쓴 뒤 이름을 바꾼다.** 그냥 덮어쓰면 쓰는 도중에 다른 요청이 읽어
    잘린 JSON 을 만날 수 있다. 이름 바꾸기는 같은 파일시스템 안에서 원자적이다.
    """
    try:
        path = _path(namespace, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        # 임시 이름에 프로세스 번호를 섞는다 — 같은 순간 두 워커가 써도 안 부딪히게
        temp = path.with_suffix(f".{os.getpid()}.tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temp.replace(path)
        return True
    except Exception:
        return False


def clear(namespace: str = "") -> int:
    """네임스페이스를 비운다. 비운 파일 수를 돌려준다. (개발·시험용)

    `namespace` 를 비우면 캐시 전체를 지운다.
    """
    target = CACHE_ROOT / _safe(namespace) if namespace else CACHE_ROOT
    removed = 0
    try:
        if not target.exists():
            return 0
        for path in target.rglob("*.json"):
            try:
                path.unlink()
                removed += 1
            except Exception:
                pass
    except Exception:
        pass
    return removed


def stats() -> Dict:
    """캐시 현황 — 대시보드 '데이터 상태' 카드가 쓴다.

    네임스페이스별 파일 수와 가장 최근에 쓴 시각(몇 초 전인지)을 알려준다.
    쓰기가 아예 막힌 환경인지도 함께 확인한다.
    """
    namespaces: List[Dict] = []
    total = 0

    try:
        if CACHE_ROOT.exists():
            for folder in sorted(CACHE_ROOT.iterdir()):
                if not folder.is_dir():
                    continue
                files = list(folder.glob("*.json"))
                if not files:
                    continue
                newest = max(f.stat().st_mtime for f in files)
                namespaces.append({
                    "namespace": folder.name,
                    "count": len(files),
                    "age_seconds": int(time.time() - newest),
                })
                total += len(files)
    except Exception:
        pass

    return {
        "root": str(CACHE_ROOT),
        "writable": _probe_writable(),
        "namespaces": namespaces,
        "total_files": total,
    }


def _probe_writable() -> bool:
    """실제로 써 보고 지운다. 서버리스는 경로가 있어도 못 쓰는 경우가 있어서 존재만으로는 모른다."""
    try:
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        probe = CACHE_ROOT / f".probe-{os.getpid()}"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except Exception:
        return False
