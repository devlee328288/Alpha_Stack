"""배포용 KRX 시세 번들 읽기 (저장소 계층) — 명세서 §8

`krx_store` 가 보는 원본 캐시(`data/krx_cache.db`, 123MB)는 배포 번들에 못 올린다.
그래서 **읽기 전용 축약본**을 따로 싣고, 원본이 없을 때 이 모듈이 대신 답한다.

    scripts/build_krx_bundle.py  →  data/krx_bundle.db     전종목 150거래일 OHLCV (29MB)
                                 →  data/krx_derived.json  거래일 캘린더 · 시장의 폭 (13KB)
                                          ↑ 이 모듈이 읽는다

이 모듈은 **읽기만 한다.** 무엇을 고를지·어떻게 셀지는 서비스 계층의 일이다.

읽기 전용을 어떻게 여는가
----------------------
`sqlite3.connect(path)` 는 파일이 없으면 만들려 하고, `PRAGMA journal_mode=WAL` 도 쓰기다.
배포 번들은 읽기 전용이라 둘 다 그 자리에서 예외가 난다. 그래서 URI 형식으로 열어
**`mode=ro`** 를 못박고 WAL 설정을 아예 하지 않는다.

    sqlite3.connect("file:/var/task/data/krx_bundle.db?mode=ro", uri=True)

번들이 원본과 다른 점 (있는 척하지 않는다)
--------------------------------------
| | 원본 캐시 | 번들 |
|---|---|---|
| 기간 | 282거래일 | **150거래일** — 그보다 앞은 없다 |
| 종목 메타 | 행마다 반복 | `stock` 표에 한 번 (JOIN 해서 붙인다) |
| `market_cap` | 날짜별 실값 | 기준일만 실값. **과거 날짜는 `종가 × 최근 상장주식수` 근사** |
| 나머지 시세 | | 그대로 — 값을 역산하지 않는다 |

시가총액을 왜 근사로 두는가 — 날짜별로 담으면 13자리 정수가 41만 개라 파일이 훌쩍 커진다.
반면 상장주식수는 유상증자·액면분할이 없는 한 몇 달을 그대로 간다. 그래서 최근 값 하나만 담고
과거는 종가에 곱한다. 근사인 사실은 `stats()` 의 `notes` 에 적어 화면·리포트가 그대로 밝힌다.

(`change`(전일대비)는 `close`·`change_rate` 로 역산해 뺐다가 되돌렸다. 등락률이 소수 2자리라
삼성전자에서 55,500원이 55,497원으로 나왔다. 컬럼 하나가 1.2MB 인데 그 값을 어긋나게 둘 이유가 없다.)
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# 이 파일은 <루트>/ingest/store/ 안에 있으므로 parents[2] 가 프로젝트 루트다.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_PATH = PROJECT_ROOT / "data" / "krx_bundle.db"
DERIVED_PATH = PROJECT_ROOT / "data" / "krx_derived.json"

# 종목 메타를 시세에 붙일 때 쓰는 SELECT 조각.
# 원본 `krx_store` 의 `SELECT *` 와 **같은 컬럼 구성**을 만들어야 부르는 쪽이 둘을 구분하지 않는다.
# 유일하게 되살리는 값은 과거 시가총액(= 종가 × 최근 상장주식수)이다.
_JOINED_SELECT = """
  p.bas_dd, p.code, s.name, s.market, s.sector,
  p.open, p.high, p.low, p.close, p.change, p.change_rate, p.volume, p.value,
  CAST(p.close * COALESCE(s.listed_shares, 0) AS INTEGER) AS market_cap,
  s.listed_shares
FROM daily_price p LEFT JOIN stock s ON s.code = p.code
"""

# 파생 JSON 은 한 번 읽으면 메모리에 둔다. 읽기 전용 파일이라 도중에 바뀌지 않는다.
_derived: Optional[Dict] = None
_lock = threading.Lock()


# ==================================================
# 1. 번들 DB
# ==================================================
def available() -> bool:
    """번들 DB 를 쓸 수 있는지."""
    return BUNDLE_PATH.exists()


def connect() -> sqlite3.Connection:
    """번들 DB 를 **읽기 전용**으로 연다. 없으면 `FileNotFoundError`.

    호출한 쪽이 반드시 닫아야 한다. 쓰기가 없으므로 커밋도 자물쇠도 필요 없다.
    """
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError(f"번들이 없습니다: {BUNDLE_PATH}")
    conn = sqlite3.connect(f"file:{BUNDLE_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _query(sql: str, params: Sequence = ()) -> List[Dict]:
    """질의 한 번. 번들이 없거나 깨져 있으면 **빈 목록**을 돌려준다.

    예외를 밖으로 내보내지 않는 것이 이 모듈의 계약이다 — 번들은 원본이 없을 때의
    대타이고, 대타마저 없으면 호출한 쪽이 라이브 조회나 안내로 넘어가야 한다.
    """
    try:
        conn = connect()
    except Exception:
        return []
    try:
        return [dict(row) for row in conn.execute(sql, params).fetchall()]
    except Exception:
        return []
    finally:
        conn.close()


def meta() -> Dict:
    """번들이 스스로 밝히는 정보 (생성 시각 · 기간 · 행 수)."""
    return {row["key"]: row["value"] for row in _query("SELECT key, value FROM bundle_meta")}


def latest_date() -> Optional[str]:
    """번들에 있는 가장 최근 거래일 (`YYYYMMDD`). 비어 있으면 `None`."""
    rows = _query("SELECT MAX(bas_dd) AS d FROM daily_price")
    return rows[0]["d"] if rows and rows[0]["d"] else None


def available_dates(limit: int = 400) -> List[str]:
    """번들에 있는 거래일 목록 (`YYYYMMDD`, 최근순)."""
    return [r["bas_dd"] for r in _query(
        "SELECT DISTINCT bas_dd FROM daily_price ORDER BY bas_dd DESC LIMIT ?", (limit,))]


def snapshot(bas_dd: str, market: Optional[str] = None) -> List[Dict]:
    """해당 거래일의 전 종목. `market` 을 주면 그 시장만."""
    sql = f"SELECT {_JOINED_SELECT} WHERE p.bas_dd = ?"
    params: List = [bas_dd]
    if market:
        sql += " AND s.market = ?"
        params.append(market)
    return _query(sql, params)


def series(code: str, days: int = 250, end: Optional[str] = None) -> List[Dict]:
    """종목 하나의 일봉 (날짜 **내림차순** — 부르는 쪽이 뒤집어 쓴다).

    `krx_store.series` 와 같은 순서로 맞추기 위해 여기서는 뒤집지 않는다.
    """
    sql = f"SELECT {_JOINED_SELECT} WHERE p.code = ?"
    params: List = [code]
    if end:
        sql += " AND p.bas_dd <= ?"
        params.append(end)
    sql += " ORDER BY p.bas_dd DESC LIMIT ?"
    params.append(days)
    return _query(sql, params)


def window(days: int = 60,
           columns: Sequence[str] = ("code", "bas_dd", "close", "value", "volume")) -> List[Dict]:
    """최근 `days` 거래일치를 필요한 컬럼만 골라 한 번에 읽는다.

    `krx_store.window` 와 같은 계약이다. 종목 메타가 필요 없는 집계용이라 JOIN 하지 않는다.
    번들에 없는 컬럼(`market_cap` 등)을 물으면 **빈 목록**을 준다 — 조용히 다른 값을
    끼워 넣는 것보다 없다고 말하는 편이 낫다.
    """
    allowed = {"bas_dd", "code", "open", "high", "low", "close",
               "change", "change_rate", "volume", "value"}
    wanted = [c for c in columns if c in allowed]
    if len(wanted) != len(columns):
        return []

    dates = available_dates(limit=days)
    if not dates:
        return []
    floor = min(dates)
    return _query(
        f"SELECT {','.join(wanted)} FROM daily_price WHERE bas_dd >= ? ORDER BY bas_dd", (floor,))


def stats() -> Dict:
    """번들 현황 — 대시보드 '데이터 상태' 가 쓴다."""
    info = meta()
    size_mb = 0.0
    try:
        size_mb = round(BUNDLE_PATH.stat().st_size / 1e6, 1)
    except OSError:
        pass

    return {
        "available": available(),
        "path": "data/krx_bundle.db",
        "days": int(info.get("days") or 0),
        "rows": int(info.get("rows") or 0),
        "codes": int(info.get("codes") or 0),
        "first_date": info.get("first_date") or None,
        "last_date": info.get("last_date") or None,
        "generated_at": info.get("generated_at") or "",
        "size_mb": size_mb,
        "notes": [
            f"최근 {info.get('days') or 0}거래일만 담긴 축약본입니다. 그 이전 구간은 없습니다.",
            "과거 날짜의 시가총액은 `종가 × 최근 상장주식수` 근사입니다 (기준일만 실값).",
            "수동 갱신입니다 — `python3 scripts/build_krx_bundle.py` 후 다시 배포해야 최신이 됩니다.",
        ],
    }


# ==================================================
# 2. 파생 JSON — 거래일 캘린더 · 시장의 폭
# ==================================================
def derived() -> Dict:
    """파생 JSON 을 메모리에 한 번만 올린다. 없으면 **빈 값**을 돌려준다."""
    global _derived
    if _derived is not None:
        return _derived

    with _lock:
        if _derived is not None:              # 기다리는 동안 다른 스레드가 채웠을 수 있다
            return _derived
        try:
            _derived = json.loads(DERIVED_PATH.read_text(encoding="utf-8"))
        except Exception:
            _derived = {"trading_days": [], "breadth": {}, "available": False}
        else:
            _derived.setdefault("available", True)
    return _derived


def reload() -> Dict:
    """파생 JSON 을 메모리에서 버리고 다시 읽는다. (재빌드 직후에 쓴다)

    평소에는 필요 없다 — 배포본의 파생 JSON 은 읽기 전용이라 도중에 바뀌지 않는다.
    **로컬에서 `build_krx_bundle.py` 를 돌린 직후만** 다르다. 그때 이것을 부르지 않으면
    거래일 캘린더가 옛것으로 남아, 새로 받은 날짜를 `preprocess` 가 **휴장일로** 본다
    (ADR-DS-0017 — 화면 갱신 버튼이 이 함수를 부른다).

    번들 DB(`krx_bundle.db`) 는 대상이 아니다. 조회마다 새로 열고 닫으므로
    파일이 바뀌면 다음 조회부터 자동으로 새것을 본다.
    """
    global _derived
    with _lock:
        _derived = None
    return derived()


def trading_days(limit: int = 0) -> List[str]:
    """실제 개장일 목록 (`YYYY-MM-DD`, **오름차순**). `limit` 을 주면 최근 그만큼만.

    M3 전처리(`preprocess.run(..., trading_days=…)`)에 그대로 넘기는 값이다.
    이게 있어야 연휴를 결측으로 오판하지 않는다 (명세서 §3.1 · 변경 노트 N15).
    """
    days = derived().get("trading_days") or []
    return list(days[-limit:]) if limit and limit < len(days) else list(days)


def calendar_range() -> tuple:
    """캘린더가 덮는 구간 `(첫날, 마지막날)`. 비어 있으면 `("", "")`.

    **왜 범위를 밝히는가.** `preprocess.find_gaps` 는 캘린더에 없는 날을 "장이 안 선 날"로
    본다. 그래서 캘린더보다 앞선 구간에 캘린더를 들이대면 그 구간의 결측을 통째로 못 본다.
    캘린더를 넘길지 말지는 부르는 쪽이 이 범위를 보고 정해야 한다.
    """
    data = derived()
    days = data.get("trading_days") or []
    if not days:
        return ("", "")
    return (days[0], days[-1])


def breadth_series(days: int = 120) -> List[Dict]:
    """시장의 폭 — 날짜별 상승·하락·보합 종목 수와 거래대금 (최근 `days`일, 오름차순).

    종목별 원본이 없어도 되는 **사전집계**라, 번들 DB(150거래일)보다 긴 구간을 덮는다.
    """
    data = derived()
    dates = data.get("trading_days") or []
    breadth = data.get("breadth") or {}
    if not dates or not breadth:
        return []

    take = min(int(days), len(dates))
    if take <= 0:
        return []

    sliced = {key: (breadth.get(key) or [])[-take:] for key in ("up", "down", "flat", "value", "total")}
    return [
        {
            "date": date,
            "up": sliced["up"][i],
            "down": sliced["down"][i],
            "flat": sliced["flat"][i],
            "value": sliced["value"][i],
            "total": sliced["total"][i],
        }
        for i, date in enumerate(dates[-take:])
        if i < len(sliced["up"])
    ]


def derived_stats() -> Dict:
    """파생 JSON 현황 — 대시보드 '데이터 상태' 가 쓴다."""
    data = derived()
    days = data.get("trading_days") or []
    size_kb = 0.0
    try:
        size_kb = round(DERIVED_PATH.stat().st_size / 1024, 1)
    except OSError:
        pass
    return {
        "available": bool(days),
        "path": "data/krx_derived.json",
        "days": len(days),
        "first_date": data.get("first_date") or None,
        "last_date": data.get("last_date") or None,
        "generated_at": data.get("generated_at") or "",
        "size_kb": size_kb,
    }
