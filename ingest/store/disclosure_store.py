"""DART 공시 **목록**을 `dart_disclosure` 에 담는다.

재무제표를 담는 `dart_store` 와 표가 다르고 받는 방법도 다르다. 재무는 회사·연도로
묻지만, 공시 목록은 **날짜로** 물을 수 있어 회사 수와 무관하게 받을 수 있다.

## 🔴 왜 종목별이 아니라 날짜별인가

`ingest/clients/dart_data.fetch_disclosures()` 는 종목 하나를 받는다. 그 길로 전 종목을
받으면 보통주 3,482종 × 창 여러 개라 호출이 폭발한다. 그런데 `list.json` 은
**`corp_code` 를 안 줘도** 된다 — 그러면 그 기간의 공시가 법인 구분 없이 전부 온다
(실측 2026-09-04). 그래서 호출 수가 **종목 수가 아니라 공시 건수**로 정해진다.

    20240820 하루 : 전체 504건 · 유가 124 · 코스닥 149
    2024-08 한 달 : 유가 4,550건 (46페이지)

## 실측으로 확인한 제약 (2026-09-04)

1. **`page_count` 상한은 100 이다.** 200·500 을 줘도 100행만 온다. 조용히 잘린다.
2. **창이 넓으면 `total_count` 가 `None` 으로 온다.** 한 해(20240101~20241231)를 통째로
   물으면 그렇다. 한 달은 정상이다. 그래서 **달 단위로 끊어** 묻는다.
3. `corp_cls` 로 시장을 좁힐 수 있다 — `Y`(유가) · `K`(코스닥) · `N`(코넥스) · `E`(기타).
   우리는 `daily_price` 에 있는 두 시장만 받는다.
4. **비상장 법인이 절반을 넘는다.** 하루 504건 중 유가+코스닥이 273건이다. 좁히지
   않으면 쓸 수 없는 행을 배로 받는다.

## 이어받기

`collect_log` 에 `disclosure:<시장>:<YYYYMM>` 한 줄을 남긴다. 달을 통째로 마친 뒤에만
`ok` 를 적는다 — 페이지 도중에 멈춘 달을 `ok` 로 적으면 **그 달의 나머지가 영영
안 온다.** 끊긴 달은 다음 실행이 처음부터 다시 받는다. 같은 공시는 접수번호가
기본키라 두 번 담기지 않는다.
"""

from __future__ import annotations

import json
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from common import budget
from common.paths import krx_db_path
from ingest.clients.dart_data import load_dart_key
from ingest.store import collect_log

#: 예산·수집로그에서 이 수집을 부르는 이름. `dart_store` 와 예산을 **함께** 쓴다
#: (같은 키로 같은 서버를 두드리므로 한도가 하나다).
SOURCE = "dart"

#: 수집 대상 시장. `daily_price` 에 KONEX 가 한 행도 없어 `N` 은 받지 않는다.
MARKETS: Tuple[Tuple[str, str], ...] = (("KOSPI", "Y"), ("KOSDAQ", "K"))

#: 한 페이지 최대 행. 🔴 100 을 넘겨도 100만 온다 — 실측으로 확인했다.
PAGE_SIZE = 100

_BASE = "https://opendart.fss.or.kr/api/list.json"
_KST = timezone(timedelta(hours=9))


class DisclosureError(Exception):
    """공시 목록을 받다 생긴 오류. 부르는 쪽이 그 달을 건너뛸지 멈출지 정한다."""


class BudgetExhausted(DisclosureError):
    """오늘 예산을 다 썼다. **고장이 아니라 하루의 끝**이다.

    `DisclosureError` 를 물려받게 둔 것은 일부러다 — 오류를 다 잡는 쪽도 이걸
    놓치지 않는다. 다만 부르는 쪽은 이것을 **먼저** 잡아 멈추고 정상 종료해야 한다.
    """


def _connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path or krx_db_path())
    conn.row_factory = sqlite3.Row
    return conn


def _now_kst() -> str:
    return datetime.now(_KST).isoformat(timespec="seconds")


def months_between(start: str, end: str) -> List[str]:
    """`YYYYMM` 목록. 창을 달로 끊는 이유는 위 문서의 제약 2번이다."""
    y, m = int(start[:4]), int(start[4:6])
    ey, em = int(end[:4]), int(end[4:6])
    out = []
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}{m:02d}")
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def _month_span(yyyymm: str) -> Tuple[str, str]:
    """그 달의 첫날과 마지막 날 (`YYYYMMDD`)."""
    y, m = int(yyyymm[:4]), int(yyyymm[4:6])
    first = f"{y:04d}{m:02d}01"
    ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
    last_day = (datetime(ny, nm, 1) - timedelta(days=1)).day
    return first, f"{y:04d}{m:02d}{last_day:02d}"


def target_of(market: str, yyyymm: str) -> str:
    return f"disclosure:{market}:{yyyymm}"


def _call(key: str, params: Dict[str, str], *, retries: int = 3) -> dict:
    """`list.json` 한 번. 예산은 **부르기 전에** 깎는다 — 부른 뒤에 깎으면 실패한
    호출이 한도에서 빠져 실제보다 적게 센다."""
    if not budget.try_spend(SOURCE, 1):
        raise BudgetExhausted(
            f"오늘 {SOURCE} 예산을 다 썼다 — 내일 같은 명령으로 이어받는다")
    query = dict(params)
    query["crtfc_key"] = key
    url = _BASE + "?" + urllib.parse.urlencode(query)
    마지막 = None
    for 시도 in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            마지막 = exc
            time.sleep(1.5 * (시도 + 1))
    raise DisclosureError(f"{retries}번 시도해도 못 받았다: {마지막}")


def fetch_month(market: str, yyyymm: str, *, key: Optional[str] = None,
                sleep: float = 0.12) -> List[Dict]:
    """한 시장 한 달의 공시 목록 전량. 페이지를 끝까지 넘긴다.

    ⚠️ `total_page` 를 믿되 **받은 행이 0이면 멈춘다.** 서버가 페이지 수를 크게 주고
       뒤쪽을 비워 주는 경우가 있어, 그대로 믿으면 빈 호출로 예산을 태운다.
    """
    if key is None:
        key, _ = load_dart_key()
    _, cls = next(((m, c) for m, c in MARKETS if m == market), (market, ""))
    if not cls:
        raise DisclosureError(f"모르는 시장이다: {market} (아는 것: "
                              f"{', '.join(m for m, _ in MARKETS)})")
    bgn, end = _month_span(yyyymm)

    rows: List[Dict] = []
    page = 1
    total_page = 1
    while page <= total_page:
        payload = _call(key, {
            "bgn_de": bgn, "end_de": end, "corp_cls": cls,
            "page_no": str(page), "page_count": str(PAGE_SIZE),
            "sort": "date", "sort_mth": "asc",
        })
        status = str(payload.get("status") or "")
        if status == "013":            # 데이터 없음 — 오류가 아니라 빈 달이다
            return []
        if status != "000":
            raise DisclosureError(
                f"{market} {yyyymm} p{page}: status={status} "
                f"{payload.get('message', '')}")
        batch = payload.get("list") or []
        if not batch:
            break
        rows.extend(batch)
        total_page = int(payload.get("total_page") or 1)
        page += 1
        if sleep:
            time.sleep(sleep)
    return rows


def save(rows: Sequence[Dict], *, conn: Optional[sqlite3.Connection] = None) -> int:
    """받은 행을 담는다. 접수번호가 기본키라 두 번 담아도 늘지 않는다.

    ⚠️ `INSERT OR REPLACE` 를 쓰지 않는다 — 같은 접수번호를 다시 받았을 때 기존 행을
       통째로 갈아 끼우면, 뒤에 붙인 칸이 있을 경우 조용히 지워진다. 있는 것은 둔다.
    """
    if not rows:
        return 0
    소유 = conn is None
    conn = conn or _connect()
    now = _now_kst()
    tuples = [(
        (r.get("rcept_no") or "").strip(),
        (r.get("corp_code") or "").strip(),
        (r.get("corp_name") or "").strip() or None,
        (r.get("stock_code") or "").strip() or None,
        (r.get("corp_cls") or "").strip() or None,
        (r.get("report_nm") or "").strip() or None,
        (r.get("flr_nm") or "").strip() or None,
        (r.get("rcept_dt") or "").strip(),
        (r.get("rm") or "").strip() or None,
        now,
    ) for r in rows if (r.get("rcept_no") or "").strip()
        and (r.get("rcept_dt") or "").strip()]
    try:
        before = conn.execute("SELECT COUNT(*) FROM dart_disclosure").fetchone()[0]
        conn.executemany(
            "INSERT OR IGNORE INTO dart_disclosure "
            "(rcept_no, corp_code, corp_name, stock_code, corp_cls, report_nm, "
            " flr_nm, rcept_dt, rm, collected_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)", tuples)
        conn.commit()
        after = conn.execute("SELECT COUNT(*) FROM dart_disclosure").fetchone()[0]
        return after - before
    finally:
        if 소유:
            conn.close()


def pending_months(start: str, end: str,
                   markets: Iterable[str] = ("KOSPI", "KOSDAQ")) -> List[Tuple[str, str]]:
    """아직 안 받은 (시장, `YYYYMM`) 목록. 이미 `ok`·`empty` 인 달은 뺀다."""
    out = []
    for market in markets:
        for ym in months_between(start, end):
            기록 = collect_log.entry(SOURCE, target_of(market, ym))
            if 기록 and 기록.get("status") in ("ok", "empty"):
                continue
            out.append((market, ym))
    return out


def status() -> Dict:
    """지금 무엇이 담겨 있나. 사람이 눈으로 확인하려고 낸다."""
    conn = _connect()
    try:
        r = conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT corp_code), COUNT(DISTINCT stock_code), "
            "MIN(rcept_dt), MAX(rcept_dt) FROM dart_disclosure").fetchone()
        시장별 = dict(conn.execute(
            "SELECT corp_cls, COUNT(*) FROM dart_disclosure GROUP BY corp_cls").fetchall())
        return {"rows": r[0], "corps": r[1], "stocks": r[2],
                "first": r[3], "last": r[4], "by_class": 시장별}
    finally:
        conn.close()
