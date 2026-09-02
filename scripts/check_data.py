"""수집한 자료의 **품질**을 재고, 못 쓸 상태면 `exit 1` 로 막는다.

    python scripts/check_data.py                 # 시세 + 지수
    python scripts/check_data.py --only index
    python scripts/check_data.py --only stock
    python scripts/check_data.py --index "코스닥 150"

결과는 사람이 읽는 표로 인쇄하고 `reports/data_quality.json` 에 함께 남긴다.

## 게이트는 "이상한 값" 이 아니라 "설명되지 않는 값" 에 건다

이 저장소의 시세는 수정주가가 아니다. 그래서 하루 등락률이 가격제한폭을 넘는 행이
920만 행 중 2,080건 있다. 그걸 그대로 실패로 처리하면 **게이트가 첫 실행부터 빨간불**
이고, 그러면 팀은 게이트를 고치지 않고 끈다. 반대로 전부 눈감으면 새 자료에 진짜
이상한 것이 들어와도 아무도 모른다.

그래서 극단 2,080행을 `common/corporate_actions.py` 의 네 플래그로 먼저 설명한다 —
정리매매 · 거래정지 재개 · 자본변동 · 신규상장 첫날. 실측으로 **잔여가 0** 이다.
게이트는 그 잔여에만 걸린다. 통과하는 것이 정상이고, 빨간불이 켜지면 그건 정말로
우리가 모르는 일이 자료에 들어왔다는 뜻이다.

같은 원칙을 dbt 도 쓴다 — `severity: error` 는 하드 계약, `warn` 은 소프트 기대이고
`error_if` 로 문턱을 준다. 검사를 늘려 알람만 쌓으면 팀이 전부 무시하게 된다는 것은
데이터 관측 분야에서 반복해서 보고된 실패 양상이다.

## 심각도 두 단계

    error   자료가 구조적으로 못 쓸 상태다. exit 1.
    warn    사실로 기록만 한다. 세고, 열거하고, 넘어간다.

`error` 는 다섯이고 현재 전부 0 이다.

    malformed_rows         `known_at` 을 만들 수 없는 행 (기준일자가 8자리 숫자가 아님)
    missing_trading_days   달력에 있는데 받은 기록조차 없는 날
    ohlc_inversions        저가 ≤ 시·종가 ≤ 고가 위반 (정지 표시행 제외)
    change_rate_mismatch   등락률이 `전일대비 / 전일종가` 와 어긋남
    unexplained_outliers   가격제한폭 밖인데 네 플래그 어디에도 안 걸림

## 등락률이 멀쩡해도 가격은 끊길 수 있다

`close - change` 가 전일 종가와 다른 날이 5,223건 있다. 액면분할·병합·감자·주식배당을
하면 KRX 가 **기준가를 조정**하기 때문이다. 조정된 기준가로 등락률을 계산해 주므로
그 값은 멀쩡하고, 그래서 **가격제한폭 검사에는 안 걸린다.** 그런데 우리 라벨은
시가 비율이라 조정폭이 그대로 수익률로 섞인다.

    기준가/전일종가   최소 0.0019배 · 중앙 0.9541배 · 최대 120배
    12월에 1,045건 — 12월 결산법인의 주식배당 권리락

지금 크기는 무시할 만하다 — KOSPI 개발구간에서 창의 0.140%가 오염됐고 `E|5일수익|`
이 4.4832% → 4.4760% 로 0.0072%p 움직였다. 그래도 **센다.** ×120 짜리 조정이 표본
안으로 들어오면 이야기가 달라지고, 들어와도 지금은 아무도 모르기 때문이다.

## OHLC 는 반드시 두 통으로 나눈다

`open=high=low=0` 인 행이 시세에 283,468건, 지수에 2,488건 있다. 전자는 거래정지
표시행이고 후자는 섹터지수 산출 초기에 종가만 들어온 구간이다. 둘 다 "저가 ≤ 시가"
를 어기지만 **자료 오류가 아니다.** 한 통에 담으면 28만 건이 리포트를 덮어서 진짜
대소 역전(실측 0건)이 보이지 않는다.

## 기준선은 여기서 재지 않는다

예전에는 이 스크립트가 기준선도 함께 인쇄했다. 그런데 **KRX 가 소수 2자리로 반올림해
주는 등락률로 세는 바람에 52.72% 가 나왔고**, 원값으로 세는 지평 측정과 **0.17%p
갈라졌다.** 반올림으로 `0.00` 이 되어 보합으로 빠진 날이 전구간 15일이고 **그중 7일이
실제 상승일**이다. 오차가 작아도 **부호가 한쪽으로 쏠리면 그건 잡음이 아니라 편향**
이다. 그래서 계산을 `evaluation/horizon.py` 한 벌로 모았다.

    python scripts/measure_horizon.py        # 지수 기준선 · 손익분기 · 클래스 균형
    python scripts/measure_stock_horizon.py  # 종목 쪽 같은 것

⚠️ 여기서 재는 것은 **레이블 정의 이전의 자료 품질**이다. 학습에 쓸 레이블의 기준선은
   `evaluation/baseline.py` 가 폴드 안에서 따로 계산한다. 둘을 섞지 않는다.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from itertools import groupby
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.corporate_actions import (  # noqa: E402
    INDEX_MOVE_NOTICE_PCT,
    RowFlags,
    basis_price,
    flag_series,
    is_basis_adjusted,
    is_halted,
    is_outlier,
    market_calendar,
    price_limit_pct,
)
from common.paths import PROJECT_ROOT, REPORTS_DIR, krx_db_path  # noqa: E402
from common.trading_calendar import now_kst_iso  # noqa: E402
from ingest.clients import krx_data as api  # noqa: E402

#: 리포트에 실을 표본 개수. 전부 싣는 항목은 따로 표시한다.
SAMPLE_LIMIT = 20

#: 등락률 역산이 저장값과 이만큼(%p)까지 어긋나는 것은 반올림으로 본다.
#: ⚠️ 실측으로는 920만 행 전부가 0.02%p 안에 들어온다. 여유가 아니라 확인된 사실이다.
RATE_TOLERANCE = 0.02

#: 읽기 캐시(MB). 검사는 920만 행을 통째로 훑는 **배치**라 일반 앱과 기준이 다르다.
#:
#: SQLite 기본은 2MB 다. 그 상태로는 `idx_code_date` 를 타며 12칸을 읽느라 행마다
#: 테이블 페이지에 임의 접근하는데, 1,578MB DB 의 페이지가 계속 밀려난다.
#: 실측(2026-08-31 · 전 종목 순회 3회):
#:
#:     기본 2MB      135.7 / 124.5 / 153.3초
#:     64MB           94.6 / 116.0초        ← 일반 앱에서 가장 흔한 값
#:     mmap 512MB    116.0초                 ← 이쪽은 별 도움이 안 됐다
#:     1GB            39.3 / 54.1 / 61.8초   ← 2.3~3.4배
#:
#: 일반 앱의 관행값(64MB)은 1.5GB DB 에 턱없이 모자란다. 반면 *"I/O 집약 작업에는
#: 캐시를 기본의 100~1,000배까지 일시적으로 올린다"* 가 이 PRAGMA 의 흔한 용법이고,
#: 1GB 는 기본의 500배로 그 범위 안이다.
#:
#: ⚠️ 상한이지 선할당이 아니다 — SQLite 는 필요한 만큼만 조금씩 잡는다.
#: ⚠️ 이 값은 **로컬 배치 스크립트에만** 쓴다. 화면(Streamlit Cloud, 메모리 약 1GB)은
#:    이 경로를 타지 않는다.
BATCH_CACHE_MB = 1024

ERROR = "error"
WARN = "warn"


def open_readonly(db: Path) -> sqlite3.Connection:
    """검사 전용 읽기 연결.

    ⚠️ **읽기 전용으로 연다.** 그냥 열면 검사가 쓰기 주체가 되고, 수집이 도는 중에
       열면 잠금을 다툰다. 검사는 자료를 보기만 해야 한다.
    """
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.execute("PRAGMA cache_size = -%d" % (BATCH_CACHE_MB * 1024))
    return conn


@dataclass
class Check:
    """검사 하나의 결과. 그대로 JSON 한 칸이 된다."""

    name: str
    severity: str
    count: int
    message: str
    samples: List = field(default_factory=list)
    exhaustive: bool = False      # samples 가 전부인가 (누락 거래일은 전부여야 한다)

    @property
    def failed(self) -> bool:
        return self.severity == ERROR and self.count > 0

    def to_json(self) -> Dict:
        out = {
            "severity": self.severity,
            "count": self.count,
            "message": self.message,
            "samples": self.samples,
            "exhaustive": self.exhaustive,
        }
        if not self.exhaustive and self.count > len(self.samples):
            out["truncated"] = True
        return out


def _mark(check: Check) -> str:
    if check.count == 0:
        return "✅"
    return "❌" if check.severity == ERROR else "⚠️"


def _print_checks(checks: Sequence[Check]) -> None:
    for c in checks:
        head = f"  {_mark(c)} {c.name:<22} {c.count:>9,}  {c.message}"
        print(head)
        if c.samples:
            shown = ", ".join(str(s) for s in c.samples[:5])
            tail = "" if c.exhaustive else f" … (표본 {len(c.samples)}/{c.count:,})"
            print(f"       {shown}{tail}")


# ==================================================
# 1. 시세 (daily_price)
# ==================================================
def _iter_stock_series(con: sqlite3.Connection) -> Iterator[Tuple[str, List[Dict]]]:
    """종목 하나씩 **전 구간**을 흘려보낸다.

    ⚠️ 전부 메모리에 올리지 않는다. 920만 행을 한 번에 담으면 수 GB 다. 종목 단위로
       처리하면 한 번에 수천 행만 든다. 실측 98.6초(2026-08-29 · SQLite 3.53.2).

    ⚠️ 자르지 않는다. 정리매매와 신규상장 판정이 **종목 이력의 양 끝**에 의존한다.
       잘라서 주면 잘린 자리가 곧 "체결 단절" 로 보여 없는 정리매매가 생긴다.
    """
    cur = con.execute(
        "SELECT code, bas_dd, name, market, open, high, low, close, change, "
        "change_rate, volume, listed_shares FROM daily_price ORDER BY code, bas_dd"
    )
    cols = [d[0] for d in cur.description]
    for code, rows in groupby(cur, key=lambda r: r[0]):
        yield code, [dict(zip(cols, r, strict=True)) for r in rows]


def _missing_trading_days(con: sqlite3.Connection, table: str, log_table: str) -> List[str]:
    """받은 기록조차 없는 평일. **연휴와 수집 누락을 가른다.**

    ⚠️ 날짜 간격만 보고 경고하면 정상 연휴가 전부 걸린다 — 2017-09-29 → 10-10 은
       추석 10일 연휴(10-02 임시공휴일)라 11일이 비는 것이 정상이다. 그래서 수집
       기록표와 대조한다. 그 표는 "받아 봤더니 0건" 을 남기므로 **확인된 휴장**과
       **아직 안 받아 본 날**을 구분할 수 있다. 이 구분이 없으면 진짜 누락이 연휴
       경고에 묻힌다.
    """
    있는날 = {r[0] for r in con.execute(f"SELECT DISTINCT bas_dd FROM {table}")}
    받아본날 = {r[0] for r in con.execute(f"SELECT bas_dd FROM {log_table}")}
    # 망가진 기준일자는 여기서 무시한다. 그건 `malformed_rows` 가 이미 세고 있고,
    # 여기서 date.fromisoformat 에 넘기면 검사가 **보고 대신 스택트레이스**로 죽는다.
    # (버그 주입 시험이 실제로 이걸 잡았다 — '2024-01-05' 를 심었더니
    #  '2024--0-1-05' 가 만들어져 ValueError 로 터졌다.)
    성한날 = {d for d in 있는날 if isinstance(d, str) and len(d) == 8 and d.isdigit()}
    if not 성한날:
        return []

    첫날 = date.fromisoformat(_iso(min(성한날)))
    끝날 = date.fromisoformat(_iso(max(성한날)))
    누락: List[str] = []
    day = 첫날
    while day <= 끝날:
        key = day.strftime("%Y%m%d")
        if day.weekday() < 5 and key not in 성한날 and key not in 받아본날:
            누락.append(key)
        day += timedelta(days=1)
    return 누락


def _iso(bas_dd: str) -> str:
    return f"{bas_dd[:4]}-{bas_dd[4:6]}-{bas_dd[6:]}"


def check_stock(con: sqlite3.Connection) -> List[Check]:
    """`daily_price` 를 검사한다. 한 번의 전 종목 순회로 전부 센다."""
    calendar = market_calendar(con)
    if not calendar:
        return [Check("empty_table", ERROR, 1, "daily_price 가 비어 있다")]

    index = {d: i for i, d in enumerate(calendar)}
    last_index = len(calendar) - 1
    collect_start = calendar[0]
    상장중 = {r[0] for r in con.execute(
        "SELECT code FROM daily_price WHERE bas_dd = ?", (calendar[-1],))}

    tally = dict.fromkeys(
        ("malformed", "zero_ohlc", "halted_but_traded", "inversion", "rate_mismatch",
         "liquidation", "capital_change", "basis_adjusted", "first_listing", "explained",
         "unexplained"), 0)
    samples: Dict[str, List] = {k: [] for k in tally}
    rows_total = codes_total = 0

    def keep(bucket: str, item) -> None:
        if len(samples[bucket]) < SAMPLE_LIMIT:
            samples[bucket].append(item)

    for code, rows in _iter_stock_series(con):
        codes_total += 1
        rows_total += len(rows)
        flags = flag_series(rows, calendar_index=index, market_last_index=last_index,
                            still_listed=code in 상장중, collect_start=collect_start)
        for i, (row, flag) in enumerate(zip(rows, flags, strict=True)):
            _tally_stock_row(code, row, rows[i - 1] if i else None, flag, tally, keep)

    checks = [
        Check("malformed_rows", ERROR, tally["malformed"],
              "기준일자·종목코드가 이상해 known_at 을 만들 수 없다",
              samples["malformed"]),
        Check("ohlc_inversions", ERROR, tally["inversion"],
              "저가 ≤ 시·종가 ≤ 고가 위반 (정지 표시행 제외)", samples["inversion"]),
        Check("change_rate_mismatch", ERROR, tally["rate_mismatch"],
              f"등락률이 전일대비/전일종가와 {RATE_TOLERANCE}%p 넘게 어긋난다",
              samples["rate_mismatch"]),
        Check("unexplained_outliers", ERROR, tally["unexplained"],
              "가격제한폭 밖인데 네 플래그 어디에도 안 걸린다", samples["unexplained"]),
    ]

    누락 = _missing_trading_days(con, "daily_price", "fetch_log")
    checks.append(Check("missing_trading_days", ERROR, len(누락),
                        "달력에 있는데 받은 기록조차 없는 평일 → scripts/fetch_krx.py",
                        누락, exhaustive=True))

    checks += [
        Check("zero_ohlc_rows", WARN, tally["zero_ohlc"],
              "거래정지 표시행 (open=high=low=0). 자료 오류가 아니다",
              samples["zero_ohlc"]),
        Check("halted_but_traded", WARN, tally["halted_but_traded"],
              "정지 표시행인데 체결이 있다 — 시·고·저만 0 이라 대소 검사를 못 받는다",
              samples["halted_but_traded"]),
        Check("liquidation_rows", WARN, tally["liquidation"],
              "정리매매 — 체결이 끊기기 직전 10체결일. 학습에서 뺀다",
              samples["liquidation"]),
        Check("first_listing_rows", WARN, tally["first_listing"],
              "신규상장 첫 거래일 — 등락률이 공모가 기준이다", samples["first_listing"]),
        Check("capital_change_rows", WARN, tally["capital_change"],
              "상장주식수가 바뀐 날 (증자·감자)", samples["capital_change"]),
        Check("basis_price_adjusted", WARN, tally["basis_adjusted"],
              "KRX 가 기준가를 조정한 날 — 가격이 연속되지 않는다",
              samples["basis_adjusted"]),
        Check("explained_outliers", WARN, tally["explained"],
              "가격제한폭 밖이지만 네 플래그로 설명된다", samples["explained"]),
    ]
    print(f"  종목 {codes_total:,} · 행 {rows_total:,} · 거래일 {len(calendar):,}")
    return checks


def _tally_stock_row(code: str, row: Dict, prev: Optional[Dict], flag: RowFlags,
                     tally: Dict, keep) -> None:
    """행 하나를 각 통에 넣는다. `check_stock` 의 안쪽 루프를 짧게 유지하려고 뺐다."""
    bas_dd = row["bas_dd"]
    ident = f"{bas_dd}/{code}"

    # ── error ───────────────────────────────────────────────
    if not (isinstance(bas_dd, str) and len(bas_dd) == 8 and bas_dd.isdigit()) or not code:
        tally["malformed"] += 1
        keep("malformed", ident)

    halted = is_halted(row)
    if halted:
        tally["zero_ohlc"] += 1
        keep("zero_ohlc", ident)
        # 정지 표시행인데 **체결이 있다.** 여기가 검사의 구멍이었다 — `elif` 라서
        # 정지행은 대소 검사를 통째로 건너뛰는데, 이 행들은 고가가 0 이면서 종가는
        # 양수라 검사를 받으면 전부 걸린다(실측 125행 전부 · 2026-08-31).
        #
        # 거래량과 거래대금이 실재하고 대체로 종가 × 거래량과 맞는다 — 체결이
        # 정말 있었고 시·고·저만 안 온 것이다. 그래서 **자료 오류가 아니라 경고**다.
        # 다만 정지행으로 세면 "정지 중이라 체결이 없다" 는 전제가 깨지므로 따로 센다.
        if row["volume"]:
            tally["halted_but_traded"] += 1
            keep("halted_but_traded",
                 {"row": ident, "종가": row["close"], "거래량": row["volume"]})
    elif None not in (row["open"], row["high"], row["low"], row["close"]) and not (
        row["low"] <= min(row["open"], row["close"])
        and max(row["open"], row["close"]) <= row["high"]
    ):
        # 정지 표시행을 뺀 **진짜** 대소 역전. 섞으면 28만 건에 묻힌다.
        tally["inversion"] += 1
        keep("inversion", ident)

    # 등락률은 전일종가를 따로 읽지 않아도 검증된다 — 전일종가 = 종가 - 전일대비.
    # 이 항등식이 깨지면 우리가 만드는 수익률 전부가 틀린다.
    prev_close = None
    if row["close"] is not None and row["change"] is not None:
        prev_close = row["close"] - row["change"]
    if prev_close and row["change_rate"] is not None:
        계산 = row["change"] / prev_close * 100
        if abs(계산 - row["change_rate"]) > RATE_TOLERANCE:
            tally["rate_mismatch"] += 1
            keep("rate_mismatch", {"row": ident, "저장": row["change_rate"],
                                   "역산": round(계산, 4)})

    # ── warn ────────────────────────────────────────────────
    for name, on in (("liquidation", flag.liquidation),
                     ("capital_change", flag.capital_change),
                     ("first_listing", flag.first_listing)):
        if on:
            tally[name] += 1
            keep(name, ident)

    # 기준가 조정은 등락률로는 안 보인다 — KRX 가 조정된 기준가로 계산해 주기 때문이다.
    # 그런데 우리 라벨은 시가 비율이라 조정폭이 그대로 수익률로 섞인다.
    if prev is not None and is_basis_adjusted(prev, row):
        tally["basis_adjusted"] += 1
        keep("basis_adjusted", {"row": ident, "name": row["name"],
                                "앞종가": prev["close"], "기준가": basis_price(row)})

    # ── 이상치는 설명 여부로 갈린다 ─────────────────────────
    if is_outlier(row):
        bucket = "explained" if flag.explained else "unexplained"
        tally[bucket] += 1
        keep(bucket, {"row": ident, "name": row["name"],
                      "change_rate": row["change_rate"],
                      "limit": price_limit_pct(bas_dd),
                      "flags": list(flag.names())})


# ==================================================
# 2. 지수 (index_price)
# ==================================================
def check_index(con: sqlite3.Connection, only: Optional[str] = None) -> List[Check]:
    """`index_price` 를 검사한다. 지수명별로 시계열을 만들어 훑는다.

    ⚠️ 지수에는 가격제한폭이 없다. 실측 최대 일변동이 코스피 200 은 +19.98%,
       섹터지수는 +26.53%(2026-07-31)이고 전부 실제 시장 사건이다 — 2020-03-19
       코로나 폭락, 2024-08-05 엔캐리 청산. 그래서 큰 변동은 **세기만** 한다.
    """
    sql = "SELECT index_name, bas_dd, open, high, low, close, change, change_rate FROM index_price"
    params: List = []
    if only:
        sql += " WHERE index_name = ?"
        params.append(only)
    sql += " ORDER BY index_name, bas_dd"

    cur = con.execute(sql, params)
    cols = [d[0] for d in cur.description]

    tally = dict.fromkeys(
        ("malformed", "zero_ohlc", "inversion", "rate_mismatch", "close_null",
         "large_move", "out_of_order"), 0)
    samples: Dict[str, List] = {k: [] for k in tally}
    지수수 = 행수 = 0

    def keep(bucket: str, item) -> None:
        if len(samples[bucket]) < SAMPLE_LIMIT:
            samples[bucket].append(item)

    for name, group in groupby(cur, key=lambda r: r[0]):
        지수수 += 1
        rows = [dict(zip(cols, r, strict=True)) for r in group]
        행수 += len(rows)
        _tally_index_series(name, rows, tally, keep)

    checks = [
        Check("malformed_rows", ERROR, tally["malformed"],
              "기준일자가 8자리 숫자가 아니다", samples["malformed"]),
        Check("ohlc_inversions", ERROR, tally["inversion"],
              "저가 ≤ 시·종가 ≤ 고가 위반 (종가만 들어온 행 제외)",
              samples["inversion"]),
        Check("change_rate_mismatch", ERROR, tally["rate_mismatch"],
              f"등락률이 종가 역산과 {RATE_TOLERANCE}%p 넘게 어긋난다",
              samples["rate_mismatch"]),
        Check("date_out_of_order", ERROR, tally["out_of_order"],
              "지수 안에서 날짜가 오름차순 유일값이 아니다", samples["out_of_order"]),
    ]

    누락 = _missing_trading_days(con, "index_price", "index_fetch_log")
    checks.append(Check("missing_trading_days", ERROR, len(누락),
                        "달력에 있는데 받은 기록조차 없는 평일 → scripts/fetch_index.py",
                        누락, exhaustive=True))

    checks += [
        Check("close_null_rows", WARN, tally["close_null"],
              "종가가 없는 행 — 거래량만 오는 지수가 있다", samples["close_null"]),
        Check("zero_ohlc_rows", WARN, tally["zero_ohlc"],
              "시·고·저가가 0 이고 종가만 있는 행 (섹터지수 산출 초기)",
              samples["zero_ohlc"]),
        Check("large_moves", WARN, tally["large_move"],
              f"일변동 {INDEX_MOVE_NOTICE_PCT}% 초과 — 실제 시장 사건이다",
              samples["large_move"]),
    ]
    print(f"  지수 {지수수:,} · 행 {행수:,}")
    return checks


def _tally_index_series(name: str, rows: List[Dict], tally: Dict, keep) -> None:
    """지수 하나의 시계열을 훑는다. 등락률은 **직전 종가 있는 행**과 비교한다."""
    앞선날 = None
    직전종가 = None
    for row in rows:
        bas_dd = row["bas_dd"]
        ident = f"{bas_dd}/{name}"

        if not (isinstance(bas_dd, str) and len(bas_dd) == 8 and bas_dd.isdigit()):
            tally["malformed"] += 1
            keep("malformed", ident)
        if 앞선날 is not None and bas_dd <= 앞선날:
            tally["out_of_order"] += 1
            keep("out_of_order", ident)
        앞선날 = bas_dd

        if row["close"] is None:
            tally["close_null"] += 1
            keep("close_null", ident)
            continue

        if is_halted(row):
            # 섹터지수는 산출 초기에 종가만 들어온다. 대소 역전이 아니다.
            tally["zero_ohlc"] += 1
            keep("zero_ohlc", ident)
        elif None not in (row["open"], row["high"], row["low"]) and not (
            row["low"] <= min(row["open"], row["close"])
            and max(row["open"], row["close"]) <= row["high"]
        ):
            tally["inversion"] += 1
            keep("inversion", ident)

        if 직전종가 and row["change_rate"] is not None:
            계산 = (row["close"] - 직전종가) / 직전종가 * 100
            if abs(계산 - row["change_rate"]) > RATE_TOLERANCE:
                tally["rate_mismatch"] += 1
                keep("rate_mismatch", {"row": ident, "저장": row["change_rate"],
                                       "역산": round(계산, 4)})
        직전종가 = row["close"]

        if row["change_rate"] is not None and abs(row["change_rate"]) > INDEX_MOVE_NOTICE_PCT:
            tally["large_move"] += 1
            keep("large_move", {"row": ident, "change_rate": row["change_rate"]})


# ==================================================
# 3. 실행
# ==================================================
def _relative_db_path() -> str:
    """DB 경로를 **저장소 기준 상대경로**로. 절대경로를 그대로 실으면 안 된다.

    이 리포트는 git 이 추적하고 저장소는 PUBLIC 이다. 절대경로를 담으면
    ① 사용자 이름이 그대로 올라가고 ② 팀원마다 경로가 달라 **매 실행마다 그 줄이
    바뀌어 충돌**한다. 리포트가 말해야 하는 것은 "어느 파일을 쟀나" 지
    "그 파일이 누구 컴퓨터 어디에 있나" 가 아니다.
    """
    path = krx_db_path()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name          # 임시 폴더 등 저장소 밖이면 이름만


def stock_scale(con: sqlite3.Connection) -> Dict:
    """`daily_price` 가 지금 얼마나 큰지 잰다. **검사가 아니라 기록이다.**

    검사 항목만 남기면 "이 리포트를 만들 때 자료가 어디까지 차 있었나"를 되짚을 수
    없다. 그런데 `krx_cache.db` 는 저장소에 올라가지 않으므로(KRX 이용약관 제11조 ②),
    DB 를 안 가진 팀원에게는 **이 리포트가 규모를 알 수 있는 유일한 경로**다.

    매일 자동으로 받기 시작하면 이 값이 특히 중요해진다 — 어제 리포트와 나란히 놓고
    "몇 행 늘었나"를 볼 수 있어야 조용한 실패(0건인데 성공으로 끝난 수집)를 알아챈다.

    실측 2026-09-02 · 920만 행 기준 세 쿼리 합계 1.7초. 게이트 전체가 전 종목을
    순회하는 데 그보다 훨씬 오래 걸리므로 다시 세는 비용은 무시할 만하다.
    """
    rows, codes, days, latest, earliest = con.execute(
        "SELECT COUNT(*), COUNT(DISTINCT code), COUNT(DISTINCT bas_dd), "
        "MAX(bas_dd), MIN(bas_dd) FROM daily_price"
    ).fetchone()
    return {"rows": rows, "codes": codes, "trading_days": days,
            "first_date": earliest, "last_date": latest}


def index_scale(con: sqlite3.Connection, only: Optional[str] = None) -> Dict:
    """`index_price` 의 규모. `only` 를 주면 그 지수만 잰다.

    ⚠️ 좁혀서 잰 값을 전체인 것처럼 남기면 나중에 읽는 사람이 속는다. 그래서 무엇으로
       좁혔는지를 `scope` 에 함께 적는다 — 없으면 전체다.
    """
    sql = ("SELECT COUNT(*), COUNT(DISTINCT index_name), MAX(bas_dd), MIN(bas_dd) "
           "FROM index_price")
    params: List = []
    if only:
        sql += " WHERE index_name = ?"
        params.append(only)
    rows, names, latest, earliest = con.execute(sql, params).fetchone()
    scale = {"rows": rows, "indices": names,
             "first_date": earliest, "last_date": latest}
    if only:
        scale["scope"] = only
    return scale


def build_report(sections: Dict[str, List[Check]],
                 scale: Optional[Dict[str, Dict]] = None) -> Dict:
    """JSON 으로 내보낼 모양. 게이트 판정도 여기서 한다.

    `scale` 은 선택이다 — 넘기지 않으면 예전과 같은 모양이 나온다. 검사만 돌려 보는
    쪽(테스트 등)이 규모를 재느라 920만 행을 훑을 이유는 없다.
    """
    failed = [f"{sec}.{c.name}" for sec, checks in sections.items()
              for c in checks if c.failed]
    report = {
        "generated_at": now_kst_iso(),
        "db_path": _relative_db_path(),
        "gate": {
            "status": "fail" if failed else "pass",
            "failed_checks": failed,
        },
    }
    if scale:
        report["scale"] = scale
    report["sections"] = {
        sec: {c.name: c.to_json() for c in checks}
        for sec, checks in sections.items()
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="수집 자료의 품질을 재고 게이트를 건다")
    parser.add_argument("--only", choices=["stock", "index"],
                        help="한쪽만 검사한다 (기본: 둘 다)")
    parser.add_argument("--index", default=None,
                        help=f"지수 하나만 (예: '{api.TARGET_INDEX}')")
    parser.add_argument("--json", default=None,
                        help="리포트 경로 (기본: reports/data_quality.json)")
    parser.add_argument("--no-json", action="store_true", help="파일을 쓰지 않는다")
    args = parser.parse_args()

    db = krx_db_path()
    if not db.exists():
        print(f"🔴 DB 가 없습니다: {db}")
        print("   할 일: python scripts/fetch_krx.py 로 먼저 채우세요.")
        return 1

    con = open_readonly(db)
    sections: Dict[str, List[Check]] = {}
    # 검사한 섹션만 규모를 잰다 — 보지도 않은 표의 크기를 리포트에 적으면
    # "이 리포트가 무엇을 확인한 것인가"가 흐려진다.
    scale: Dict[str, Dict] = {}

    if args.only != "index":
        print("═══ 시세 (daily_price) ═══")
        sections["stock"] = check_stock(con)
        scale["stock"] = stock_scale(con)
        _print_checks(sections["stock"])
        print()

    if args.only != "stock":
        title = f"지수 (index_price · {args.index})" if args.index else "지수 (index_price)"
        print(f"═══ {title} ═══")
        sections["index"] = check_index(con, only=args.index)
        scale["index"] = index_scale(con, only=args.index)
        _print_checks(sections["index"])
        print()

    report = build_report(sections, scale)

    if not args.no_json:
        out = Path(args.json) if args.json else REPORTS_DIR / "data_quality.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        print(f"리포트: {out}")

    if report["gate"]["status"] == "fail":
        print(f"\n❌ 게이트 실패 — {', '.join(report['gate']['failed_checks'])}")
        print("   위 항목은 자료가 구조적으로 못 쓸 상태라는 뜻입니다.")
        return 1

    print("\n✅ 게이트 통과 — error 항목이 전부 0 입니다.")
    print("   기준선·손익분기는 여기서 재지 않습니다 — scripts/measure_horizon.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
