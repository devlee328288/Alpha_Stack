"""공공데이터포털 금융위 자료를 `stock_identity`·`corp_profile` 에 채운다. (수집 → 저장)

`scripts/fetch_data_go_kr.py` 가 이 모듈을 부른다. 시세의 `krx_store`, 재무의
`dart_store`, 거시의 `macro_store` 와 짝이고 같은 DB 파일의 다른 표에 담는다.

## 무엇을 담나 — 시세가 아니라 **다리**다

    KRX 종목코드  ↔  crno(법인등록번호)  ↔  ISIN

이 다리가 없으면 우리 시세(9,223,644행)는 DART 고유번호에도, 해외 자료의 ISIN 에도
못 붙는다. 그리고 `corp_profile` 이 **공식 상장폐지일**을 준다 — 지금 우리는 "어느
날부터 시세가 안 나온다" 로 폐지를 추정하는데 그건 장기 거래정지와 구별되지 않는다.

## 🔴🔴 `stock_identity` 를 **그 시점 상장 목록으로 쓰지 않는다**

포털의 `basDt` 목록에는 **그날 이후에야 상장된 종목이 섞여 있다** — 20200102 기준
2,334종 중 33종이 그랬고, 듀켐바이오(176750)는 첫 시세가 4년 뒤인 2024-12-20 이다
(실측 2026-09-03). 포털은 최신에 가까운 목록에 기준일 딱지만 붙여 준다.

    유니버스를 만들 때는 반드시 `daily_price` 와 **교집합**을 낸다.
    그대로 쓰면 아직 없던 종목이 섞이고, **에러 없이 성능만 좋아진다.**

이 표의 쓸모는 *"그날 무엇이 상장돼 있었나"* 가 아니라 **다리**다. 다리로 쓰는 데는
이 성질이 해가 되지 않는다. `scripts/verify_identity.py` 가 매번 다시 세어 알린다.

## 두 표의 시점 기준이 다르다

| 표 | `known_at` | 왜 |
|---|---|---|
| `stock_identity` | `bas_dd` **의 다음 거래일** (계산값) | 포털이 발표 시각을 안 준다 |
| `corp_profile` | `fstOpegDt` (관측값) | 출처가 "이 날부터 유효" 를 직접 준다 |

🔴 앞엣것은 **계산값**이라, 규칙을 바꾸면 다시 받아야 한다. 그래서 행마다
`known_rule` 을 남긴다 — 어느 행이 옛 규칙으로 계산됐는지 알 수 있어야 한다.
거시(ECOS)에서 같은 문제를 겪었고 거기서는 `known_rule` 이 없어 애를 먹었다.

## 어디까지 받았는지 기억한다

수집 대장(`collect_log`)에 대상마다 한 줄을 남긴다.

    ok      값이 들어왔다
    empty   받아봤는데 0건 — 휴장일이거나 그 법인 자료가 없다
    error   시도했는데 실패했다

`empty` 를 남기는 이유는 시세 수집과 같다 — **"받아 봤더니 없었다" 와 "아직 안 받았다"
를 구별**하지 않으면 휴장일마다 영원히 다시 요청하게 된다.
"""

from __future__ import annotations

import sqlite3
from contextlib import nullcontext
from typing import Dict, List, Optional, Sequence, Tuple

from common.paths import krx_db_path
from common.trading_calendar import CalendarOutOfRange, next_session, now_kst_iso
from ingest.clients import data_go_kr
from ingest.store import collect_log
from ingest.store.krx_store import connect

#: 수집 대장에 적히는 출처 이름.
SOURCE = "data_go_kr"

#: 스키마를 이미 확인했나. 한 프로세스에서 마이그레이션을 매번 돌릴 이유가 없다.
_schema_ready = False

IDENTITY_COLUMNS: Tuple[str, ...] = (
    "bas_dd", "code", "isin_cd", "crno", "corp_nm", "item_nm", "market",
    "known_at", "known_rule", "fetched_at",
)

PROFILE_COLUMNS: Tuple[str, ...] = (
    "crno", "fst_opeg_dt", "last_opeg_dt", "corp_nm", "sic_nm", "estb_dt",
    "stac_mm", "xchg_lstg_dt", "xchg_lstg_abol_dt", "kosdaq_lstg_dt",
    "kosdaq_lstg_abol_dt", "audt_rpt_opnn", "actn_audpn", "empe_cnt",
    "pn1_avg_slry_amt", "smenp_yn", "known_at", "known_rule", "fetched_at",
)

#: `INSERT OR REPLACE` 를 쓰는 이유 — 같은 날짜를 다시 받아도 행이 늘지 않아야 한다.
#:
#: ⚠️ 그 대신 **덮어쓴다.** 행 수를 세는 검사로는 덮어쓰기를 못 잡으므로, 값이 바뀌었는지
#:    보려면 `scripts/verify_identity.py` 처럼 값을 맞대는 검사가 따로 있어야 한다.
_INSERT_IDENTITY = (
    f"INSERT OR REPLACE INTO stock_identity ({', '.join(IDENTITY_COLUMNS)}) "
    f"VALUES ({', '.join('?' * len(IDENTITY_COLUMNS))})"
)
_INSERT_PROFILE = (
    f"INSERT OR REPLACE INTO corp_profile ({', '.join(PROFILE_COLUMNS)}) "
    f"VALUES ({', '.join('?' * len(PROFILE_COLUMNS))})"
)


class IdentityStoreError(RuntimeError):
    """담는 도중 세운다. 무엇을 해야 하는지까지 문구에 담는다."""


# ==================================================
# 1. known_at — 계산값이라는 것을 잊지 않는다
# ==================================================
def known_at_for(bas_dd: str, db_path=None) -> str:
    """`bas_dd` 다음 거래일. 포털이 발표 시각을 주지 않아 계산한다.

    포털 안내는 *"기준일자로부터 영업일 하루 뒤 오후 1시 이후 갱신"* 이다. 그러니
    기준일 당일에는 그 자료를 볼 수 없었다 — 당일로 붙이면 미래참조가 된다.

    🔴 **날짜 계산으로 다음 날을 구하지 않는다.** 공휴일·임시휴장이 있어서 개발구간
       평일 3,042일 중 실제 거래일은 2,880일이다(162일 · 5.3% 차이). 실측 달력을 쓴다.

    달력 밖이면 **세운다** — 지어내면 아직 열리지 않은 장에 자료를 붙이게 된다.
    """
    try:
        return next_session(bas_dd, db_path)
    except CalendarOutOfRange as exc:
        raise IdentityStoreError(
            f"{bas_dd} 의 다음 거래일을 몰라 known_at 을 정할 수 없다.\n"
            f"  {exc}\n"
            "  할 일: 그 구간 시세를 먼저 받아 달력을 넓히거나, 그 날짜를 건너뛴다."
        ) from exc


# ==================================================
# 2. 담기
# ==================================================
def _tuple(row: Dict, columns: Sequence[str]) -> Tuple:
    return tuple(row.get(c) for c in columns)


def save_identity(rows: Sequence[Dict],
                  conn: Optional[sqlite3.Connection] = None) -> int:
    """`stock_identity` 에 담는다. 담은 행 수를 돌려준다.

    🔴 키가 될 값(`bas_dd`·`code`)이 없는 행은 **담지 않고 세운다.** 빈 키로 넣으면
       서로를 덮어써서, 행 수는 그럴듯한데 내용이 사라진다.
    """
    if not rows:
        return 0
    fetched = now_kst_iso()
    묶음 = []
    for r in rows:
        if not r.get("bas_dd") or not r.get("code"):
            raise IdentityStoreError(
                f"키가 빈 행이 있다: bas_dd={r.get('bas_dd')!r} code={r.get('code')!r}\n"
                "  흔한 원인: srtnCd 의 A 접두사 처리가 어긋났다.\n"
                "  할 일: data_go_kr.strip_code_prefix 를 확인하고, 그 행을 격리한다."
            )
        묶음.append(_tuple({**r, "fetched_at": r.get("fetched_at") or fetched},
                          IDENTITY_COLUMNS))

    ctx = nullcontext(conn) if conn is not None else connect()
    with ctx as c:
        c.executemany(_INSERT_IDENTITY, 묶음)
    return len(묶음)


def save_profile(rows: Sequence[Dict],
                 conn: Optional[sqlite3.Connection] = None) -> int:
    """`corp_profile` 에 담는다. 담은 행 수를 돌려준다."""
    if not rows:
        return 0
    fetched = now_kst_iso()
    묶음 = []
    for r in rows:
        if not r.get("crno") or not r.get("fst_opeg_dt"):
            raise IdentityStoreError(
                f"키가 빈 행이 있다: crno={r.get('crno')!r} "
                f"fst_opeg_dt={r.get('fst_opeg_dt')!r}\n"
                "  할 일: data_go_kr.parse_profile_row 는 이런 행에 None 을 준다 — "
                "부르는 쪽에서 걸렀는지 확인한다."
            )
        묶음.append(_tuple({**r, "fetched_at": r.get("fetched_at") or fetched},
                          PROFILE_COLUMNS))

    ctx = nullcontext(conn) if conn is not None else connect()
    with ctx as c:
        c.executemany(_INSERT_PROFILE, 묶음)
    return len(묶음)


# ==================================================
# 3. 하루치·법인 하나 받아 담기
# ==================================================
def sync_listed_day(bas_dd: str, *, key: Optional[str] = None,
                    db_path=None, conn: Optional[sqlite3.Connection] = None) -> Dict:
    """한 기준일의 상장종목을 받아 담는다.

    돌려주는 것: `{"bas_dd", "rows", "status"}`.
    """
    대상 = f"listed:{bas_dd}"
    if bas_dd < data_go_kr.EARLIEST_BAS_DD:
        # 🔴 호출조차 하지 않는다. 2019 이전은 전부 0건이라(실측) 훑으면 한도만 태운다.
        collect_log.mark_out_of_range(
            SOURCE, 대상,
            note=f"포털은 {data_go_kr.EARLIEST_BAS_DD} 부터만 준다")
        return {"bas_dd": bas_dd, "rows": 0, "status": "out_of_range"}

    known = known_at_for(bas_dd, db_path)
    try:
        행들 = data_go_kr.fetch_listed(bas_dd, key=key, known_at=known)
    except data_go_kr.DataGoKrError as exc:
        collect_log.mark_error(SOURCE, 대상, note=str(exc)[:500])
        raise

    if not 행들:
        # 휴장일이다. **0건도 기록한다** — 안 그러면 휴장일마다 영원히 다시 요청한다.
        collect_log.mark_empty(SOURCE, 대상, note="0건 (휴장 또는 자료 없음)")
        return {"bas_dd": bas_dd, "rows": 0, "status": "empty"}

    담은수 = save_identity(행들, conn)
    collect_log.mark_ok(SOURCE, 대상, rows=담은수, cursor=bas_dd)
    return {"bas_dd": bas_dd, "rows": 담은수, "status": "ok"}


def sync_profile(crno: str, *, key: Optional[str] = None,
                 conn: Optional[sqlite3.Connection] = None) -> Dict:
    """한 법인의 개요 **전 이력**을 받아 담는다. 한 번 부르면 여러 행이 온다."""
    대상 = f"profile:{crno}"
    try:
        행들 = data_go_kr.fetch_corp_profile(crno, key=key)
    except data_go_kr.DataGoKrError as exc:
        collect_log.mark_error(SOURCE, 대상, note=str(exc)[:500])
        raise

    if not 행들:
        collect_log.mark_empty(SOURCE, 대상, note="0건 (그 법인 자료가 없다)")
        return {"crno": crno, "rows": 0, "status": "empty"}

    담은수 = save_profile(행들, conn)
    collect_log.mark_ok(SOURCE, 대상, rows=담은수)
    return {"crno": crno, "rows": 담은수, "status": "ok"}


# ==================================================
# 4. 조회
# ==================================================
def crno_targets(conn: Optional[sqlite3.Connection] = None) -> List[str]:
    """아직 `corp_profile` 이 없는 `crno` 들. 이어 받을 때 쓴다."""
    ctx = nullcontext(conn) if conn is not None else connect()
    with ctx as c:
        rows = c.execute(
            "SELECT DISTINCT i.crno FROM stock_identity i "
            "LEFT JOIN corp_profile p ON p.crno = i.crno "
            "WHERE i.crno IS NOT NULL AND i.crno <> '' AND p.crno IS NULL "
            "ORDER BY i.crno"
        ).fetchall()
    return [r[0] for r in rows]


def status(conn: Optional[sqlite3.Connection] = None) -> Dict:
    """지금 두 표에 무엇이 얼마나 있나."""
    ctx = nullcontext(conn) if conn is not None else connect()
    with ctx as c:
        신원 = c.execute(
            "SELECT COUNT(*), COUNT(DISTINCT bas_dd), COUNT(DISTINCT code), "
            "MIN(bas_dd), MAX(bas_dd) FROM stock_identity").fetchone()
        개요 = c.execute(
            "SELECT COUNT(*), COUNT(DISTINCT crno), MIN(fst_opeg_dt), "
            "MAX(fst_opeg_dt) FROM corp_profile").fetchone()
    return {
        "stock_identity": {"rows": 신원[0], "days": 신원[1], "codes": 신원[2],
                           "first": 신원[3], "last": 신원[4]},
        "corp_profile": {"rows": 개요[0], "crno": 개요[1],
                         "first": 개요[2], "last": 개요[3]},
    }


def ensure_schema() -> None:
    """두 표가 없으면 만든다 (마이그레이션 v10). 한 프로세스에서 한 번만.

    🔴 `krx_store.init_db()` 로는 안 된다 — 그쪽 `SCHEMA` 는 시세 표만 만든다.
       `stock_identity`·`corp_profile` 은 **마이그레이션이 만든다.** DDL 을 두 곳에
       두면 언젠가 갈라지기 때문이다 (`collect_log` 와 같은 이유·같은 방식).

    경로를 인자로 받지 않는다 — `krx_store` 가 `KRX_DB_PATH` 환경변수 하나로 경로를
    정하고, 테스트는 `conftest.py` 가 그 변수를 프로세스 전체에서 갈아 끼운다.
    여기서 또 경로를 받으면 갈아 끼울 자리가 둘이 되고, 하나를 빠뜨려도 테스트는
    통과한다 (2026-08-26 에 실제로 그렇게 진짜 DB 에 20행이 들어갔다).
    """
    global _schema_ready
    if _schema_ready:
        return
    from ingest.store.migrations import migrate_path
    migrate_path(krx_db_path())
    _schema_ready = True
