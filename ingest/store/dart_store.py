"""DART 재무제표를 받아 `dart_financial` 에 채운다. (수집 → 저장 계층)

`scripts/fetch_dart.py` 가 이 모듈을 부른다. 시세 쪽의 `krx_store` 와 짝이고,
같은 DB 파일의 다른 표에 담는다.

## 무엇을 받나

유니버스 350종 × 사업연도 × 사업보고서(`11011`). 회사·연도마다 **2콜**이 든다 —
재무 본문(`fnlttSinglAcntAll`) 1콜과 접수일 조회(`list.json`) 1콜이다.
실측 2026-09-02: 삼성전자 2023 을 받으니 `call_budget` 의 `used` 가 정확히 2 올랐다.

연결(`CFS`)이 비어 있으면 `fetch_financials` 가 **별도(`OFS`)로 자동 재시도**하므로
그 회사는 3콜이 된다. 지주회사가 아닌 중소형주는 연결재무제표를 만들지 않는 일이 흔하다.

## 🔴 시점 기준은 접수일 하나뿐이다

`bsns_year` 는 **결산기**이지 세상이 알게 된 날이 아니다. 2020년 4분기 실적은
2021년 3월에 나오므로, 결산기에 값을 붙이면 **석 달치 미래**를 학습에 넣고도 예외는
나지 않고 성능만 좋아진다. 이 프로젝트에서 가장 조용한 오류다.

그래서 `rcept_dt` 가 없는 응답은 **저장하지 않고 오류로 기록한다.** 반입 규격
(`ingest/inbox/schemas/financial.json`)의 `has_time_anchor` 가 같은 것을 error 로 건다.

## 어디까지 받았는지 기억한다

수집 대장(`collect_log`)에 회사·연도마다 한 줄을 남긴다. 중간에 죽어도 다음 실행이
받은 데를 건너뛴다. 상태가 셋으로 갈린다.

    ok               계정이 들어왔다
    empty            받아봤는데 없었다 — **미수집과 다르다**
    error            시도했는데 실패했다 (3회까지 다시 시도한다)

`empty` 를 따로 두는 이유가 이번 수집의 핵심이다. 상장 전 연도·사업보고서 미제출은
**정상적으로 없는 것**이라, 이걸 미수집과 구별하지 않으면 다음 실행이 같은 빈 칸을
영원히 다시 부른다. 350종 × 5개년 중 얼마가 그런지는 받아 봐야 안다.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from common.paths import PROJECT_ROOT
from common.trading_calendar import now_kst_iso
from ingest.clients import dart_data
from ingest.store import collect_log
from ingest.store.krx_store import connect, init_db

#: 수집 대장에 적히는 출처 이름. `dart_data.BUDGET_SOURCE`("dart")와 **다르다** —
#: 저쪽은 호출 예산의 단위(키 하나에 하루 20,000콜)이고, 이쪽은 무엇을 받았나의 단위다.
#: 나중에 공시 목록·기업개황을 받아도 예산은 같이 쓰지만 대장은 따로 남아야 한다.
SOURCE = "dart_financial"

#: 유니버스 파일. `scripts/build_universe.py` 가 만든다.
UNIVERSE_FILE = PROJECT_ROOT / "data" / "universe_core.json"

#: `dart_financial` 의 칸 순서. 마이그레이션 v6 의 정의와 **같은 순서**여야 한다.
COLUMNS: Tuple[str, ...] = (
    "corp_code", "stock_code", "corp_name", "bsns_year", "reprt_code", "fs_div",
    "sj_div", "account_id", "account_nm", "account_detail", "ord", "currency",
    "thstrm_nm", "thstrm_amount", "frmtrm_amount", "bfefrmtrm_amount",
    "rcept_no", "rcept_dt", "report_nm", "rm", "collected_at",
)

_INSERT = (
    f"INSERT OR REPLACE INTO dart_financial ({', '.join(COLUMNS)}) "
    f"VALUES ({', '.join('?' * len(COLUMNS))})"
)


# ==================================================
# 1. 유니버스
# ==================================================
def load_universe() -> Dict[str, Dict]:
    """`{종목코드: {name, index, market}}`. 파일이 없으면 무엇을 해야 하는지 알려준다."""
    if not UNIVERSE_FILE.exists():
        raise FileNotFoundError(
            f"유니버스 파일이 없습니다: {UNIVERSE_FILE}\n"
            "  할 일: python scripts/build_universe.py 로 먼저 만드세요."
        )
    payload = json.loads(UNIVERSE_FILE.read_text(encoding="utf-8"))
    return payload.get("codes") or {}


# ==================================================
# 2. 저장
# ==================================================
def _as_int(value, default: int = 0) -> int:
    """DART 는 `bsns_year` 를 2023.0 으로, `ord` 를 7.0 으로 준다 (실측).

    ⚠️ 그대로 넣으면 기본키에 실수가 들어가 `2023.0` 과 `2023` 이 **다른 행**이 된다.
    """
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _as_float(value) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rows_from(out: Dict, stock_code: str, collected_at: str) -> List[Tuple]:
    """`fetch_financials` 의 응답을 `dart_financial` 한 줄씩으로 편다.

    계정은 `out["statements"]` 에 **재무제표 구분별 dict** 로 담겨 온다
    (`{"BS": [...], "IS": [...], "CIS": [...], "CF": [...], "SCE": [...]}`).
    """
    corp_code = (out.get("corp_code") or "").strip()
    corp_name = (out.get("corp_name") or "").strip()
    rcept_no = (out.get("rcept_no") or "").strip()
    rcept_dt = (out.get("rcept_dt") or "").strip()
    reprt_code = (out.get("reprt_code") or "").strip()
    report_nm = (out.get("reprt_name") or "").strip()

    rows: List[Tuple] = []
    statements = out.get("statements") or {}
    for sj_div, lines in statements.items():
        for line in lines or []:
            rows.append((
                (line.get("corp_code") or corp_code).strip(),
                stock_code,
                corp_name,
                _as_int(line.get("bsns_year") or out.get("bsns_year")),
                (line.get("reprt_code") or reprt_code).strip(),
                (line.get("fs_div") or out.get("fs_div") or "").strip().upper(),
                (line.get("sj_div") or sj_div or "").strip().upper(),
                (line.get("account_id") or "").strip() or None,
                (line.get("account_nm") or "").strip(),
                # 🔴 None 이 아니라 빈 문자열이다. 이 칸은 기본키의 일부이고, SQLite 는
                #    PK 안의 NULL 을 서로 다른 값으로 보므로 같은 계정이 여러 줄 쌓인다.
                (line.get("account_detail") or "").strip(),
                _as_int(line.get("ord")),
                (line.get("currency") or "").strip() or None,
                (line.get("thstrm_nm") or "").strip() or None,
                _as_float(line.get("thstrm_amount")),
                _as_float(line.get("frmtrm_amount")),
                _as_float(line.get("bfefrmtrm_amount")),
                (line.get("rcept_no") or rcept_no).strip() or None,
                (line.get("rcept_dt") or rcept_dt).strip() or None,
                report_nm or None,
                (line.get("rm") or "").strip() or None,
                collected_at,
            ))
    return rows


def save(out: Dict, stock_code: str, conn: Optional[sqlite3.Connection] = None) -> int:
    """한 회사·한 연도치를 저장하고 **저장한 줄 수**를 돌려준다.

    ⚠️ 계정명이 빈 줄은 넣지 않는다. `account_nm` 이 기본키의 일부라 빈 문자열로 들어가면
       서로 다른 계정이 한 줄로 뭉개진다.
    """
    collected_at = now_kst_iso()
    rows = [r for r in _rows_from(out, stock_code, collected_at) if r[8]]
    if not rows:
        return 0

    if conn is not None:
        conn.executemany(_INSERT, rows)
        return len(rows)

    with connect() as own:
        own.executemany(_INSERT, rows)
    return len(rows)


# ==================================================
# 3. 수집
# ==================================================
def target_of(stock_code: str, year: int, reprt_code: str) -> str:
    """수집 대장의 대상 열쇠. **종목코드로 적는다.**

    고유번호가 아니라 종목코드를 쓰는 이유는 사람이 대장을 읽기 때문이다 —
    `005930:2023:11011` 은 알아볼 수 있지만 `00126380:2023:11011` 은 그렇지 않다.
    """
    return f"{stock_code}:{year}:{reprt_code}"


def sync(codes: Optional[Sequence[str]] = None,
         years: Iterable[int] = (2021, 2022, 2023, 2024, 2025),
         reprt_code: str = "11011",
         progress: Optional[Callable] = None,
         limit: Optional[int] = None) -> Dict:
    """유니버스 × 연도를 받아 채운다. 이미 받은 것은 건너뛴다.

    돌려주는 것: `{"requested", "already", "ok", "empty", "error", "rows", "stopped"}`

    `stopped` 가 채워져 있으면 **한도에 닿아 멈춘 것**이다. 실패가 아니라 "오늘은
    여기까지" 이고, 다음 날 다시 부르면 받은 곳부터 이어 받는다.
    """
    init_db()
    universe = load_universe()
    targets = list(codes) if codes else list(universe)
    years = list(years)

    todo: List[Tuple[str, int]] = []
    already = 0
    for stock_code in targets:
        for year in years:
            if collect_log.should_collect(SOURCE, target_of(stock_code, year, reprt_code)):
                todo.append((stock_code, year))
            else:
                already += 1

    if limit is not None:
        todo = todo[:limit]

    result = {"requested": len(targets) * len(years), "already": already,
              "ok": 0, "empty": 0, "error": 0, "rows": 0, "stopped": ""}

    for done, (stock_code, year) in enumerate(todo, start=1):
        target = target_of(stock_code, year, reprt_code)
        name = (universe.get(stock_code) or {}).get("name", "")
        status = "ok"
        rows = 0
        note = ""

        try:
            out = dart_data.fetch_financials(stock_code, year, reprt_code)
        except dart_data.DartQuotaExhausted as exc:
            # 실패가 아니다. 여기서 멈추고 다음 날 이어 받는다.
            collect_log.mark_quota_exhausted(SOURCE, target, note=str(exc))
            result["stopped"] = str(exc)
            break
        except dart_data.DartError as exc:
            collect_log.mark_error(SOURCE, target, note=f"{exc.dart_status or ''} {exc}".strip())
            result["error"] += 1
            status, note = "error", str(exc)
            if progress:
                progress(done, len(todo), stock_code, name, year, status, rows, note)
            continue

        # ⚠️ **빈 응답을 먼저 가른다.** 순서를 바꾸면 "보고서가 아예 없어서 접수일도
        #    없는 것"이 "접수일을 못 찾은 오류"로 기록된다. 실측 2026-09-02 로 그
        #    실수를 했다 — 350종 × 5개년에서 104건이 error 로 남았는데 전부 빈 응답이었다.
        #
        #    분류가 틀리면 대가가 있다. `should_collect` 는 error 를 3회까지 다시
        #    시도하므로, 영원히 없을 자료를 실행할 때마다 208콜씩 다시 부르게 된다.
        #
        #    빈 응답의 실제 모습 (실측): `empty=True · count=0 · rcept_no='' `
        #      달바글로벌 2021~2023  상장 전이라 사업보고서가 없다
        #      삼성화재 2021~2022    오래된 회사인데도 DART 가 전체계정을 안 준다
        #                            (왜 그런지는 재지 못했다)
        if out.get("empty") or not out.get("statements"):
            note = f"{year}년 {reprt_code} 보고서 전체계정이 비어 있다 (상장 전·미제출 등)"
            collect_log.mark_empty(SOURCE, target, note=note)
            result["empty"] += 1
            status = "empty"
        # 🔴 계정은 왔는데 접수일이 없으면 **저장하지 않는다.** 이 숫자를 언제부터 알 수
        #    있었는지 모르는 채로 넣으면 결산기에 값을 붙이게 되고, 그건 조용한 미래
        #    참조다. 이쪽은 진짜 오류라 다시 시도할 값어치가 있다.
        elif not (out.get("rcept_dt") or "").strip():
            note = "계정은 왔는데 접수일(rcept_dt)이 없다 — 시점을 세울 수 없어 저장하지 않는다"
            collect_log.mark_error(SOURCE, target, note=note)
            result["error"] += 1
            status = "error"
        else:
            rows = save(out, stock_code)
            collect_log.mark_ok(SOURCE, target, rows=rows,
                                cursor=(out.get("rcept_dt") or ""))
            result["ok"] += 1
            result["rows"] += rows

        if progress:
            progress(done, len(todo), stock_code, name, year, status, rows, note)

    return result
