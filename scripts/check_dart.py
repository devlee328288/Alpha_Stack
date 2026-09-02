"""받은 **재무**가 쓸 만한 상태인지 재고, 못 쓸 상태면 `exit 1` 로 막는다.

    python scripts/check_dart.py
    python scripts/check_dart.py --json reports/dart_quality.json
    python scripts/check_dart.py --no-json

시세를 검사하는 `check_data.py` 와 짝이고, 같은 철학을 따른다 — **막는 것(error)과
세는 것(warn)을 나눈다.** 검사를 늘려 알람만 쌓으면 팀이 전부 무시하게 된다.

## 재무에서 가장 위험한 것은 값이 아니라 시점이다

시세는 그날 값이 그날 것이지만, 재무는 **결산기와 세상이 알게 된 날이 석 달까지
벌어진다.** 2020년 4분기 실적은 2021년 3월에 나온다. 결산기에 값을 붙이면 석 달치
미래를 학습에 넣고도 **예외가 나지 않고 성능만 좋아진다** — 이 프로젝트에서 가장
조용한 오류다.

그래서 error 넷 중 셋이 시점 검사다.

    missing_receipt_date   접수일이 없다 — 언제부터 알 수 있었는지 세울 수 없다
    future_receipt_date    접수일이 오늘보다 뒤다 — 미래 자료이거나 파싱이 틀렸다
    receipt_before_period  접수일이 사업연도보다 앞이다 — 시간이 거꾸로 흐른다
    duplicate_keys         같은 열쇠가 두 번 들어갔다 — 기본키가 새고 있다

## 🔴 왜 중복을 세는가 — 6.4%가 조용히 사라진 적이 있다

`INSERT OR REPLACE` 는 같은 기본키를 만나면 **예외 없이 덮어쓴다.** 실측 2026-09-02 ·
기본키에 `account_detail` 이 빠져 있던 동안 삼성전자 2023 연결이 176줄 → 135줄로
줄었고, 전체로는 351,565행 중 22,436건(6.4%)이 사라질 뻔했다.

자본변동표(SCE)는 "자본금·주식발행초과금·이익잉여금·비지배지분…" 열마다 한 줄씩
주는데 그 열을 가리키는 칸이 `account_detail` 뿐이다 — 계정명도 `ord` 도
`account_id` 도 전부 같다.

그래서 이 검사는 **두 가지를 함께 센다.** 현재 기본키로 센 중복(0이어야 한다)과,
`account_detail` 을 빼고 센 중복(0이 아니어야 한다)이다. 뒤엣것이 0이면 이 검사가
아무것도 재고 있지 않다는 뜻이다 — **항등식을 통과시키지 않기 위해서다.**
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import REPORTS_DIR, krx_db_path  # noqa: E402
from common.trading_calendar import now_kst_iso, today_kst  # noqa: E402

#: 리포트에 실을 표본 개수
SAMPLE_LIMIT = 20

ERROR = "error"
WARN = "warn"


@dataclass
class Check:
    name: str
    severity: str
    count: int
    message: str
    samples: List = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return self.severity == ERROR and self.count > 0

    def to_json(self) -> Dict:
        return {"severity": self.severity, "count": self.count,
                "message": self.message, "samples": self.samples}


def _mark(c: Check) -> str:
    if c.count == 0:
        return "✅"
    return "❌" if c.severity == ERROR else "⚠️"


def open_readonly(db: Path) -> sqlite3.Connection:
    """⚠️ 읽기 전용으로 연다 — 수집이 도는 중에 열면 잠금을 다툰다."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _samples(con: sqlite3.Connection, sql: str, params: Sequence = ()) -> List:
    return [dict(r) for r in con.execute(sql, params).fetchmany(SAMPLE_LIMIT)]


def check_financial(con: sqlite3.Connection) -> List[Check]:
    """`dart_financial` 을 검사한다."""
    rows = con.execute("SELECT COUNT(*) FROM dart_financial").fetchone()[0]
    if not rows:
        return [Check("empty_table", ERROR, 1,
                      "dart_financial 이 비어 있다 → python scripts/fetch_dart.py")]

    오늘 = today_kst().strftime("%Y%m%d")

    빈접수일 = con.execute(
        "SELECT COUNT(*) FROM dart_financial "
        "WHERE rcept_dt IS NULL OR rcept_dt = ''").fetchone()[0]
    미래 = con.execute(
        "SELECT COUNT(*) FROM dart_financial WHERE rcept_dt > ?", (오늘,)).fetchone()[0]
    # 🔴 **12월 결산을 전제하지 않는다.**
    #
    # 처음에는 "접수 연도 ≤ 사업연도" 를 전부 error 로 잡았는데 318건이 걸렸다.
    # 열어 보니 **비츠로셀(082920) 한 회사**였고, 이 회사는 **6월 결산법인**이다 —
    # FY2015 는 2014년 7월~2015년 6월이라 2015년 9월 접수가 **정상**이다.
    #
    # 접수월 분포가 그 사실을 뒷받침한다 (실측 2026-09-02 · 3,209건):
    #     3월 2,368건 (12월 결산법인의 사업보고서) · 9월 9건 · 6월 69건 …
    #
    # 그래서 둘로 나눈다.
    #     error  접수 연도 < 사업연도      결산월이 어떻든 설명이 안 된다
    #     warn   접수 연도 = 사업연도      12월 결산이 아닌 회사다. 사실로 기록만 한다
    역전 = con.execute(
        "SELECT COUNT(*) FROM dart_financial "
        "WHERE CAST(SUBSTR(rcept_dt, 1, 4) AS INTEGER) < bsns_year").fetchone()[0]
    같은해 = con.execute(
        "SELECT COUNT(*) FROM dart_financial "
        "WHERE CAST(SUBSTR(rcept_dt, 1, 4) AS INTEGER) = bsns_year").fetchone()[0]

    중복SQL = """
        SELECT COUNT(*) FROM (
          SELECT corp_code, bsns_year, reprt_code, fs_div, sj_div,
                 account_nm, ord, account_detail, COUNT(*) AS n
          FROM dart_financial GROUP BY 1,2,3,4,5,6,7,8 HAVING n > 1)
    """
    중복 = con.execute(중복SQL).fetchone()[0]

    # 🔴 이 값이 0 이면 위 검사가 아무것도 재고 있지 않다는 뜻이다 (항등식 방지)
    상세없이 = con.execute("""
        SELECT COUNT(*) FROM (
          SELECT corp_code, bsns_year, reprt_code, fs_div, sj_div,
                 account_nm, ord, COUNT(*) AS n
          FROM dart_financial GROUP BY 1,2,3,4,5,6,7 HAVING n > 1)
    """).fetchone()[0]

    checks = [
        Check("missing_receipt_date", ERROR, 빈접수일,
              "접수일이 없다 — 언제부터 알 수 있었는지 세울 수 없다"),
        Check("future_receipt_date", ERROR, 미래,
              "접수일이 오늘보다 뒤다 — 미래 자료이거나 날짜 파싱이 틀렸다",
              _samples(con, "SELECT stock_code, bsns_year, rcept_dt FROM dart_financial "
                            "WHERE rcept_dt > ?", (오늘,))),
        Check("receipt_before_period", ERROR, 역전,
              "접수 연도가 사업연도보다 앞이다 — 결산월이 어떻든 설명이 안 된다",
              _samples(con, "SELECT stock_code, corp_name, bsns_year, rcept_dt "
                            "FROM dart_financial "
                            "WHERE CAST(SUBSTR(rcept_dt,1,4) AS INTEGER) < bsns_year")),
        Check("non_december_settlement", WARN, 같은해,
              "접수 연도 = 사업연도 — 12월 결산이 아닌 회사다 (오류가 아니다)",
              _samples(con, "SELECT DISTINCT stock_code, corp_name, bsns_year, rcept_dt "
                            "FROM dart_financial "
                            "WHERE CAST(SUBSTR(rcept_dt,1,4) AS INTEGER) = bsns_year")),
        Check("duplicate_keys", ERROR, 중복,
              "같은 기본키가 두 번 들어갔다 — 기본키가 새고 있다"),
        Check("guard_is_not_identity", ERROR,
              0 if 상세없이 > 0 else 1,
              f"account_detail 을 빼면 중복이 {상세없이:,}건이어야 한다 — "
              "0 이면 위 중복 검사가 항등식이다"),
    ]
    return checks


def scale(con: sqlite3.Connection) -> Dict:
    """규모를 기록한다. **검사가 아니라 기록이다.**

    DB 는 저장소에 올라가지 않으므로(KRX 이용약관 제11조 ②), DB 를 안 가진 팀원에게는
    이 리포트가 "재무가 얼마나 있나" 를 알 수 있는 유일한 경로다.
    """
    r = con.execute("""
        SELECT COUNT(*) AS rows, COUNT(DISTINCT corp_code) AS corps,
               COUNT(DISTINCT stock_code) AS stocks,
               MIN(bsns_year) AS first_year, MAX(bsns_year) AS last_year,
               MIN(rcept_dt) AS first_receipt, MAX(rcept_dt) AS last_receipt
        FROM dart_financial
    """).fetchone()
    out = dict(r)
    out["by_year"] = {
        str(y): n for y, n in con.execute(
            "SELECT bsns_year, COUNT(*) FROM dart_financial GROUP BY bsns_year "
            "ORDER BY bsns_year")}
    out["by_statement"] = {
        s: n for s, n in con.execute(
            "SELECT sj_div, COUNT(*) FROM dart_financial GROUP BY sj_div")}
    return out


def collect_summary(con: sqlite3.Connection) -> Dict:
    """수집 대장 — 무엇을 받았고 무엇이 없었나."""
    return {status: {"targets": n, "rows": rows or 0}
            for status, n, rows in con.execute(
                "SELECT status, COUNT(*), SUM(rows) FROM collect_log "
                "WHERE source = 'dart_financial' GROUP BY status")}


def main() -> int:
    parser = argparse.ArgumentParser(description="받은 재무의 품질을 재고 게이트를 건다")
    parser.add_argument("--json", default=None, help="리포트 경로")
    parser.add_argument("--no-json", action="store_true", help="파일을 쓰지 않는다")
    args = parser.parse_args()

    db = krx_db_path()
    if not db.exists():
        print(f"🔴 DB 가 없습니다: {db}")
        print("   할 일: python scripts/fetch_dart.py 로 먼저 채우세요.")
        return 1

    con = open_readonly(db)
    try:
        checks = check_financial(con)
        규모 = scale(con) if checks[0].name != "empty_table" else {}
        대장 = collect_summary(con)
    finally:
        con.close()

    print("═══ 재무 (dart_financial) ═══")
    if 규모:
        print(f"  행 {규모['rows']:,} · 회사 {규모['corps']:,} · "
              f"사업연도 {규모['first_year']}~{규모['last_year']}")
        print(f"  접수일 {규모['first_receipt']} ~ {규모['last_receipt']}")
        print(f"  연도별: {규모['by_year']}")
    print()
    for c in checks:
        print(f"  {_mark(c)} {c.name:<24} {c.count:>9,}  {c.message}")
        if c.samples:
            print(f"       {c.samples[:3]}")

    if 대장:
        print("\n── 수집 대장 ──")
        for status, v in sorted(대장.items()):
            print(f"  {status:<10} {v['targets']:>6}건  {v['rows']:>10,}행")

    report = {
        "generated_at": now_kst_iso(),
        "gate": {
            "status": "fail" if any(c.failed for c in checks) else "pass",
            "failed_checks": [c.name for c in checks if c.failed],
        },
        "scale": 규모,
        "collect_log": 대장,
        "checks": {c.name: c.to_json() for c in checks},
    }

    if not args.no_json:
        out = Path(args.json) if args.json else REPORTS_DIR / "dart_quality.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        print(f"\n리포트: {out}")

    if report["gate"]["status"] == "fail":
        print(f"\n❌ 게이트 실패 — {', '.join(report['gate']['failed_checks'])}")
        return 1

    print("\n✅ 게이트 통과 — error 항목이 전부 0 입니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
