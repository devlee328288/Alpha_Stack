"""재무 — 그 행의 날짜에 알 수 있었던 **결산이 가장 늦은** 사업보고서.

## 무엇이 들어 있나 (실측 2026-09-11 · `main` `3fb42ee`)

    dart_financial   662,933행 · 350종 · 사업보고서(11011)만 · FY2015~2025 · 보고서 3,209건
                     연결(CFS) 2,988 · 별도(OFS) 221 — 한 보고서에 둘이 함께 있는 경우 0

## 🔴 붙이는 축은 결산기가 아니라 접수일이다

`bsns_year` 는 결산기이지 세상이 그 숫자를 알게 된 날이 아니다. 결산기 끝에서 접수일까지
**중앙값 87일 · 최대 2,467일**이다. 결산기에 값을 붙이면 그만큼 미래가 학습에 들어가는데
예외는 나지 않고 성능만 좋아진다. 5일 지평이면 우연히 안 겹칠 수도 있지만 t+61 처럼
지평이 길어지면 반드시 겹친다.

## 시점 규칙 — 접수일 **다음 거래일** (공시 텍스트와 같은 규칙)

`rcept_dt` 에는 시각이 없다. 그래서 접수일 다음 거래일(`known_at`)부터 그 행에 보인다.
규칙은 `supply.clock.dart_known_at` 한 곳에만 있다.

    2024-03-12(화) 접수 → known_at 20240313 → 03-13 행부터 보임 → 03-14 시가 진입

행 T 에 보이는 조건은 `known_at <= T` 하나다.

## 🔴 정정본 — 저장된 접수일은 정정일이다

DART 재무 API 는 판(vintage)을 고를 수 없어 **마지막 정정본**을 준다. 보고서 3,209건 중
**835건(26.0%)** 이 `[기재정정]` 이고, 저장된 `rcept_dt` 는 원본이 아니라 정정일이다
(원본보다 중앙값 61일 · 최대 2,378일 뒤 — 두산 FY2017 원본 2018-03-30 → 정정 2024-10-02).

원본 접수일로 당기면 **정정된 숫자를 원본 날짜에 붙이게 된다** — 재작성 누수다. 그래서
정정일을 그대로 쓰고, 그 해는 정정일까지 보이지 않는다. 잃는 몫은 노트북에서 센다.

`fin_amended` 는 붙인 보고서가 **첫 제출본이 아니라는** 표시다. 그 보고서 자신의 제목
(`[기재정정]…`)에서 읽으므로 접수 시점에 알 수 있던 사실이다. 공시 목록의 `rm` 칸('정'·'철')은
**나중에 붙는 표시**라 쓰지 않는다 — 그걸로 거르면 그 자체가 미래참조다.

## 🔴 "가장 최근" 은 알게 된 순서가 아니라 결산 순서다

정정본은 옛 해를 늦게 드러낸다 — 현대자동차 FY2015 는 2022-02-17 정정본으로 받았다. 알게 된
날로 최신을 고르면(`merge_asof` 를 접수일에 그대로 걸면) 그날 FY2020 을 FY2015 가 밀어낸다.
그래서 **그날까지 알게 된 보고서 중 결산이 가장 늦은 것**을 고른다.

## 계정 — 이름표 접두어가 해마다 다르다

표준계정 id 의 접두어가 FY2018 까지 `ifrs_`, FY2019 부터 `ifrs-full_` 이다. 접두어를 떼고
뒷부분으로 모으면 보고서 3,209건 기준 채움률이 이렇다.

    자산 99.9% · 부채 99.9% · 자본 99.4% · 순이익 98.5% · 영업현금흐름 97.8%
    영업이익 97.4% · 매출 95.9% · 지배지분 87.6% · 지배순이익 81.0%

매출이 빈 곳은 은행·보험처럼 "영업수익" 을 쓰는 회사다. **0 으로 채우지 않는다** —
빈 칸은 결측이다.

비율(ROE·PBR 등)은 여기서 만들지 않는다. 이 문은 **그날 알 수 있었던 원값**까지만 내고,
그 값을 피처로 만드는 것은 피처 파트의 몫이다.

    from supply.financial import attach_financial, financial_as_of, financial_lines_as_of

    wide  = financial_as_of("20240313", as_of="2024-03-14")          # 종목당 한 행
    panel = attach_financial(price_frame, as_of="2024-08-31")        # 시세 표에 18칸
    lines = financial_lines_as_of("20240313", as_of="2024-03-14",    # 보고서 전 계정
                                  codes=["005930"])
"""

from __future__ import annotations

import re
import sqlite3
from contextlib import closing
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from common.paths import krx_db_path
from supply.clock import AsOf, as_bas_dd, dart_known_at, latest_known_day, to_kst

#: 보고서 코드 → 같은 사업연도 안의 결산 순서. 1분기 < 반기 < 3분기 < 사업보고서.
#: 지금 DB 에는 사업보고서(11011)만 있지만, 분기를 들여도 "가장 최근" 이 흔들리지 않게 둔다.
REPORT_ORDER: Dict[str, int] = {"11013": 1, "11012": 2, "11014": 3, "11011": 4}

#: 표준계정 id 접두어. **앞에 있을수록 우선**이다 — 한 보고서에 둘이 함께 있으면 앞의 것.
TAXONOMY_PREFIXES: Tuple[str, ...] = ("ifrs-full_", "ifrs_", "dart_")

#: 넓은 표의 계정 칸 → (재무제표 구분들, 표준계정 id 뒷부분). 구분도 **앞에 있을수록 우선**.
#:
#: 🔴 당기순이익은 IS 와 CIS 에 함께 실린 보고서가 557건이다. 값이 다른 것은 0건이지만
#:    (실측 2026-09-11) 순서를 정해 두어야 몇 번을 돌려도 같은 줄을 고른다.
FINANCIAL_ACCOUNTS: Dict[str, Tuple[Tuple[str, ...], str]] = {
    "fin_total_assets": (("BS",), "Assets"),
    "fin_total_liabilities": (("BS",), "Liabilities"),
    "fin_total_equity": (("BS",), "Equity"),
    "fin_equity_parent": (("BS",), "EquityAttributableToOwnersOfParent"),
    "fin_revenue": (("IS", "CIS"), "Revenue"),
    "fin_operating_income": (("IS", "CIS"), "OperatingIncomeLoss"),
    "fin_net_income": (("IS", "CIS"), "ProfitLoss"),
    "fin_net_income_parent": (("IS", "CIS"), "ProfitLossAttributableToOwnersOfParent"),
    "fin_operating_cash_flow": (("CF",), "CashFlowsFromUsedInOperatingActivities"),
}

#: 붙인 보고서가 **무엇이고 언제 알았나.** 값보다 이 칸들이 먼저다 — 누수를 의심할 때
#: 사람이 되짚을 수 있어야 한다.
#:
#:   fin_period_end   제목의 `(YYYY.MM)` 달의 말일. 결산월을 12월로 **가정하지 않는다** —
#:                    6월 결산 2건(082920 FY2015·2016)이 실재한다. 제목을 못 이으면 None
#:   fin_known_at     이 보고서가 행에 보이기 시작하는 거래일 (접수일 다음 거래일)
#:   fin_amended      첫 제출본이 아닌가 (`[기재정정]`·`[첨부정정]`·`[첨부추가]`). 모르면 None
#:   fin_report_nm    공시 목록의 제목. `dart_financial.report_nm` 은 전부 "사업보고서" 라
#:                    정정 여부가 안 보여서 접수번호로 공시 목록에서 가져온다
FINANCIAL_META_COLUMNS: Tuple[str, ...] = (
    "fin_bsns_year", "fin_reprt_code", "fin_fs_div", "fin_period_end",
    "fin_rcept_no", "fin_rcept_dt", "fin_known_at", "fin_amended", "fin_report_nm",
)

#: 넓은 표가 내는 칸(앞에 `code` 가 붙는다). **행이 0개여도 이 칸들은 있다.**
FINANCIAL_COLUMNS: Tuple[str, ...] = FINANCIAL_META_COLUMNS + tuple(FINANCIAL_ACCOUNTS)

#: 긴 표가 내는 칸. `account_key` 는 접두어를 뗀 id 다 — `-표준계정코드 미사용-` 이면 None.
FINANCIAL_LINE_COLUMNS: Tuple[str, ...] = (
    "code", "fin_rcept_no", "fin_bsns_year", "fin_reprt_code", "fin_fs_div", "fin_known_at",
    "sj_div", "account_id", "account_key", "account_nm", "account_detail", "ord",
    "currency", "thstrm_nm", "thstrm_amount", "frmtrm_amount", "bfefrmtrm_amount",
)

#: 제목 속 결산연월 — `사업보고서 (2017.12)` · `[기재정정]사업보고서 (2015.06)`.
_PERIOD_IN_TITLE = re.compile(r"\((\d{4})\.(\d{2})\)")

#: SQLite 한 문장의 `?` 상한보다 넉넉히 작게 자른다.
_CHUNK = 900


# ==================================================
# 1. 작은 도구
# ==================================================
def _connect(db_path=None) -> sqlite3.Connection:
    """읽기만 한다. 경로를 인자로 받아 갈아 끼울 자리를 하나로 둔다(`ingest/store/sqlite_db.py`)."""
    conn = sqlite3.connect(db_path or krx_db_path(), timeout=60)
    conn.execute("PRAGMA busy_timeout=60000")
    return conn


def _empty(columns: Sequence[str]) -> pd.DataFrame:
    """빈 표에도 칸을 남긴다 — 칸이 없으면 부르는 쪽의 `df["fin_revenue"]` 가 KeyError 로 터진다."""
    return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})


def _row_day(bas_dd: AsOf, as_of: AsOf) -> str:
    """행의 거래일을 `YYYYMMDD` 로 맞추고, `as_of` 시점에 아직 오지 않은 날이면 세운다.

    🔴 빈 표를 주지 않고 세우는 이유 — 빈 표는 *"그날 재무가 없었다"* 로 읽힌다.
    """
    바스 = as_bas_dd(bas_dd)
    if 바스 is None:
        raise ValueError(f"bas_dd 를 읽을 수 없다: {bas_dd!r}")
    상한 = latest_known_day(as_of)
    if 바스 > 상한:
        raise ValueError(
            f"{바스} 는 as_of({to_kst(as_of).date()}) 시점에 아직 오지 않은 거래일이다.\n"
            f"  그때 알 수 있었던 가장 최근 거래일: {상한}\n"
            "  할 일: bas_dd 를 그 이하로 주거나, as_of 를 뒤로 옮긴다."
        )
    return 바스


def _normalize_codes(codes: Optional[Iterable[str]]) -> Optional[List[str]]:
    """종목코드를 **문자열 그대로** 정리한다. 숫자로 바꾸지 않는다 — 5·6번째 자리에 영문이 있다."""
    if codes is None:
        return None
    if isinstance(codes, str):
        codes = [codes]
    return sorted({str(c).strip() for c in codes if str(c).strip()})


def _account_key(account_id) -> Optional[str]:
    """`ifrs-full_Assets`·`ifrs_Assets`·`dart_Assets` → `Assets`. 표준계정이 아니면 None."""
    if not isinstance(account_id, str):
        return None
    for prefix in TAXONOMY_PREFIXES:
        if account_id.startswith(prefix):
            return account_id[len(prefix):]
    return None


def _prefix_rank(account_id) -> int:
    for i, prefix in enumerate(TAXONOMY_PREFIXES):
        if isinstance(account_id, str) and account_id.startswith(prefix):
            return i
    return len(TAXONOMY_PREFIXES)


def _period_end(report_nm) -> Optional[str]:
    """제목의 `(YYYY.MM)` → 그 달 말일 `YYYYMMDD`. 제목이 없거나 꼴이 다르면 None."""
    if not isinstance(report_nm, str):
        return None
    m = _PERIOD_IN_TITLE.search(report_nm)
    if not m:
        return None
    year, month = int(m.group(1)), int(m.group(2))
    if not 1 <= month <= 12:
        return None
    end = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    return end.strftime("%Y%m%d")


def _is_amended(report_nm) -> Optional[bool]:
    """첫 제출본이 아닌가. 제목을 못 이었으면 None — 모르는 것을 '정정 아님' 으로 두지 않는다."""
    if not isinstance(report_nm, str):
        return None
    return report_nm.lstrip().startswith("[")


def _code_filter(alias: str, codes: Optional[List[str]]) -> Tuple[str, List[str]]:
    if codes is None:
        return "", []
    return f" AND {alias}.stock_code IN ({','.join('?' * len(codes))})", list(codes)


# ==================================================
# 2. 보고서 — 한 건당 한 줄, 언제 알았나까지
# ==================================================
def _reports(conn: sqlite3.Connection, *, codes: Optional[List[str]], rcept_upto: str,
             db_path=None) -> pd.DataFrame:
    """접수일이 `rcept_upto` 이하인 보고서. 칸: code · corp_code · bsns_year · reprt_code ·
    fs_div · rcept_no · rcept_dt · report_nm · known_at · report_order.

    SQL 의 `rcept_dt <= rcept_upto` 는 **넉넉한 상한**이다. `known_at` 은 늘 접수일보다 늦으므로
    여기서 빠진 보고서는 어차피 보일 수 없다. 정확한 자르기는 부르는 쪽이 `known_at` 으로 한다.

    ⚠️ 한 보고서 키(회사·연도·보고서·연결구분)에 접수번호가 둘 섞인 경우는 0건이다(실측
       2026-09-11). 섞이면 접수번호마다 따로 한 줄이 되고, 결산이 같으면 늦게 안 쪽이 이긴다.
    """
    조건, 인자 = _code_filter("f", codes)
    sql = f"""
      SELECT r.code, r.corp_code, r.bsns_year, r.reprt_code, r.fs_div,
             r.rcept_no, r.rcept_dt, d.report_nm
      FROM (SELECT f.stock_code AS code, f.corp_code, f.bsns_year, f.reprt_code, f.fs_div,
                   f.rcept_no, MIN(f.rcept_dt) AS rcept_dt
            FROM dart_financial f
            WHERE f.rcept_dt IS NOT NULL AND f.rcept_dt <> '' AND f.rcept_dt <= ?
              AND f.stock_code IS NOT NULL AND f.stock_code <> ''{조건}
            GROUP BY f.stock_code, f.corp_code, f.bsns_year, f.reprt_code, f.fs_div,
                     f.rcept_no) r
      LEFT JOIN dart_disclosure d ON d.rcept_no = r.rcept_no
    """
    frame = pd.read_sql_query(sql, conn, params=[rcept_upto, *인자])
    if frame.empty:
        return frame.assign(known_at=pd.Series(dtype="object"),
                            report_order=pd.Series(dtype="int64"))

    frame["code"] = frame["code"].astype(str)
    frame["bsns_year"] = frame["bsns_year"].astype(int)
    지도 = dart_known_at(frame["rcept_dt"].unique(), db_path=db_path)
    frame["known_at"] = frame["rcept_dt"].map(지도)
    # 달력 뒤라 다음 거래일을 모르면 행을 만들지 않는다 — 지어낸 시점은 곧 미래참조다.
    frame = frame[frame["known_at"].notna()].copy()
    frame["report_order"] = frame["reprt_code"].map(REPORT_ORDER).fillna(0).astype(int)
    return frame.reset_index(drop=True)


def _latest_per_code(reports: pd.DataFrame) -> pd.DataFrame:
    """종목마다 **결산이 가장 늦은** 보고서 하나. 결산이 같으면 늦게 안 쪽, 그다음 접수번호."""
    if reports.empty:
        return reports
    ordered = reports.sort_values(["code", "bsns_year", "report_order", "known_at", "rcept_no"])
    return ordered.groupby("code", sort=True).tail(1).reset_index(drop=True)


def _timeline(reports: pd.DataFrame) -> pd.DataFrame:
    """종목마다 *"그날까지 알게 된 것 중 결산이 가장 늦은 보고서"* 가 정해지는 시점들.

    칸: code · known_at · rcept_no. `merge_asof` 로 행에 붙일 오른쪽 표다.

    🔴 **왜 `merge_asof` 를 보고서 표에 바로 걸지 않나.** 그러면 *"가장 최근에 알게 된
       보고서"* 가 붙는다. 정정본은 옛 해를 늦게 드러내므로 2022-02 에 FY2020 자리를
       FY2015 가 차지한다. 알게 된 순서대로 훑으며 결산이 가장 늦은 것을 들고 간다.
    """
    rows: List[Tuple[str, str, str]] = []
    ordered = reports.sort_values(["code", "known_at"])
    for code, sub in ordered.groupby("code", sort=False):
        best: Optional[Tuple[int, int, str, str]] = None
        for known, same_day in sub.groupby("known_at", sort=True):
            for r in same_day.itertuples(index=False):
                key = (int(r.bsns_year), int(r.report_order), str(r.known_at), str(r.rcept_no))
                if best is None or key > best:
                    best = key
            rows.append((str(code), str(known), best[3]))
    return pd.DataFrame(rows, columns=["code", "known_at", "rcept_no"])


# ==================================================
# 3. 계정 — 접두어를 떼고 모은다
# ==================================================
def _account_lines(conn: sqlite3.Connection, *, codes: Optional[List[str]], rcept_upto: str,
                   rcept_nos: Iterable[str], representative_only: bool,
                   sj_div: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """고른 보고서들의 계정 줄. `representative_only` 면 넓은 표에 쓸 계정만 가져온다.

    `dart_financial` 에는 접수번호 인덱스가 없어서 **종목·접수일로 좁혀 읽고** 접수번호는
    파이썬에서 거른다. 표가 66만 행이라 "필요한 것만" 이 최적화가 아니라 조건이다.
    """
    wanted = set(map(str, rcept_nos))
    if not wanted:
        return pd.DataFrame(columns=["rcept_no", "sj_div", "account_id", "account_nm",
                                     "account_detail", "ord", "currency", "thstrm_nm",
                                     "thstrm_amount", "frmtrm_amount", "bfefrmtrm_amount"])

    조건, 인자 = _code_filter("f", codes)
    추가: List[str] = []
    if representative_only:
        ids = sorted({p + suffix for _, suffix in FINANCIAL_ACCOUNTS.values()
                      for p in TAXONOMY_PREFIXES})
        sjs = sorted({s for sjs_, _ in FINANCIAL_ACCOUNTS.values() for s in sjs_})
        조건 += f" AND f.account_id IN ({','.join('?' * len(ids))})"
        조건 += f" AND f.sj_div IN ({','.join('?' * len(sjs))})"
        추가 = ids + sjs
    elif sj_div:
        sjs = [str(s).upper() for s in sj_div]
        조건 += f" AND f.sj_div IN ({','.join('?' * len(sjs))})"
        추가 = sjs

    sql = f"""
      SELECT f.rcept_no, f.sj_div, f.account_id, f.account_nm, f.account_detail, f.ord,
             f.currency, f.thstrm_nm, f.thstrm_amount, f.frmtrm_amount, f.bfefrmtrm_amount
      FROM dart_financial f
      WHERE f.rcept_dt <= ?{조건}
    """
    frame = pd.read_sql_query(sql, conn, params=[rcept_upto, *인자, *추가])
    return frame[frame["rcept_no"].astype(str).isin(wanted)].reset_index(drop=True)


def _account_values(lines: pd.DataFrame) -> pd.DataFrame:
    """계정 줄 → 접수번호마다 대표 계정 아홉 칸. 없는 계정은 NaN 이다 (0 이 아니다).

    한 보고서에 후보가 여럿이면 **구분 순서 → 접두어 순서 → 세부항목 없음 → ord** 로 하나를
    고른다. 입력 순서에 기대지 않으므로 몇 번을 돌려도 같은 값이다.
    """
    out = pd.DataFrame(index=pd.Index(sorted(lines["rcept_no"].astype(str).unique()),
                                      name="rcept_no"))
    if lines.empty:
        for col in FINANCIAL_ACCOUNTS:
            out[col] = pd.Series(dtype="float64")
        return out

    work = lines.copy()
    work["rcept_no"] = work["rcept_no"].astype(str)
    work["account_key"] = work["account_id"].map(_account_key)
    work["_prefix_rank"] = work["account_id"].map(_prefix_rank)
    # 재무상태표·손익계산서의 세부항목 칸은 실측으로 '-' 다. 세부가 붙은 줄은 뒤로 민다.
    work["_has_detail"] = (~work["account_detail"].fillna("").isin(["", "-"])).astype(int)

    for col, (sjs, suffix) in FINANCIAL_ACCOUNTS.items():
        rank = {s: i for i, s in enumerate(sjs)}
        cand = work[(work["account_key"] == suffix) & work["sj_div"].isin(sjs)].copy()
        if cand.empty:
            out[col] = pd.Series(dtype="float64")
            continue
        cand["_sj_rank"] = cand["sj_div"].map(rank)
        cand = cand.sort_values(["rcept_no", "_sj_rank", "_prefix_rank", "_has_detail", "ord"])
        first = cand.drop_duplicates("rcept_no", keep="first").set_index("rcept_no")
        out[col] = pd.to_numeric(first["thstrm_amount"], errors="coerce").reindex(out.index)
    return out


def _wide(reports: pd.DataFrame, values: pd.DataFrame) -> pd.DataFrame:
    """보고서 표 + 계정 값 → `code` + `FINANCIAL_COLUMNS`."""
    if reports.empty:
        return _empty(["code", *FINANCIAL_COLUMNS])
    rcept = reports["rcept_no"].astype(str)
    out = pd.DataFrame({
        "code": reports["code"].astype(str).to_numpy(),
        "fin_bsns_year": reports["bsns_year"].astype(int).to_numpy(),
        "fin_reprt_code": reports["reprt_code"].to_numpy(),
        "fin_fs_div": reports["fs_div"].to_numpy(),
        "fin_period_end": reports["report_nm"].map(_period_end).to_numpy(),
        "fin_rcept_no": rcept.to_numpy(),
        "fin_rcept_dt": reports["rcept_dt"].to_numpy(),
        "fin_known_at": reports["known_at"].to_numpy(),
        "fin_amended": reports["report_nm"].map(_is_amended).to_numpy(),
        "fin_report_nm": reports["report_nm"].to_numpy(),
    })
    for col in FINANCIAL_ACCOUNTS:
        out[col] = values[col].reindex(rcept).to_numpy() if col in values else float("nan")
    return out.sort_values("code").reset_index(drop=True)


# ==================================================
# 4. 정문 — as_of 없이는 못 지난다
# ==================================================
def financial_as_of(bas_dd: AsOf, *, as_of: AsOf, codes: Optional[Iterable[str]] = None,
                    db_path=None) -> pd.DataFrame:
    """거래일 `bas_dd` 의 행에 보이는 재무 — 종목당 한 행.

    칸: `code` + `FINANCIAL_COLUMNS`(메타 9 · 계정 9). 그날까지 보고서가 하나도 안 보이는
    종목은 행이 없다. 빈 표는 오류가 아니라 *"그때는 몰랐다"* 다.
    """
    바스 = _row_day(bas_dd, as_of)
    코드 = _normalize_codes(codes)
    if 코드 == []:
        return _empty(["code", *FINANCIAL_COLUMNS])

    with closing(_connect(db_path)) as conn:
        reports = _reports(conn, codes=코드, rcept_upto=바스, db_path=db_path)
        reports = reports[reports["known_at"] <= 바스] if not reports.empty else reports
        pick = _latest_per_code(reports)
        lines = _account_lines(conn, codes=코드, rcept_upto=바스,
                               rcept_nos=pick["rcept_no"] if not pick.empty else [],
                               representative_only=True)
    return _wide(pick, _account_values(lines))


def financial_lines_as_of(bas_dd: AsOf, *, as_of: AsOf, codes: Optional[Iterable[str]] = None,
                          sj_div: Optional[Sequence[str]] = None,
                          db_path=None) -> pd.DataFrame:
    """`financial_as_of` 와 **같은 보고서**의 전 계정 줄 — 긴 표.

    넓은 표의 아홉 칸 밖의 계정이 필요할 때 쓴다. 보고서를 고르는 규칙이 같으므로 두 표를
    `fin_rcept_no` 로 이으면 어긋나지 않는다. 접두어 정규화는 `account_key` 에 해 두지만
    어떤 계정을 쓸지는 부르는 쪽이 고른다.
    """
    바스 = _row_day(bas_dd, as_of)
    코드 = _normalize_codes(codes)
    if 코드 == []:
        return _empty(FINANCIAL_LINE_COLUMNS)

    with closing(_connect(db_path)) as conn:
        reports = _reports(conn, codes=코드, rcept_upto=바스, db_path=db_path)
        reports = reports[reports["known_at"] <= 바스] if not reports.empty else reports
        pick = _latest_per_code(reports)
        if pick.empty:
            return _empty(FINANCIAL_LINE_COLUMNS)
        lines = _account_lines(conn, codes=코드, rcept_upto=바스, rcept_nos=pick["rcept_no"],
                               representative_only=False, sj_div=sj_div)
    if lines.empty:
        return _empty(FINANCIAL_LINE_COLUMNS)

    meta = pick.assign(rcept_no=pick["rcept_no"].astype(str)).set_index("rcept_no")
    lines = lines.assign(rcept_no=lines["rcept_no"].astype(str))
    out = pd.DataFrame({
        "code": lines["rcept_no"].map(meta["code"]).to_numpy(),
        "fin_rcept_no": lines["rcept_no"].to_numpy(),
        "fin_bsns_year": lines["rcept_no"].map(meta["bsns_year"]).to_numpy(),
        "fin_reprt_code": lines["rcept_no"].map(meta["reprt_code"]).to_numpy(),
        "fin_fs_div": lines["rcept_no"].map(meta["fs_div"]).to_numpy(),
        "fin_known_at": lines["rcept_no"].map(meta["known_at"]).to_numpy(),
        "sj_div": lines["sj_div"].to_numpy(),
        "account_id": lines["account_id"].to_numpy(),
        "account_key": lines["account_id"].map(_account_key).to_numpy(),
        "account_nm": lines["account_nm"].to_numpy(),
        "account_detail": lines["account_detail"].to_numpy(),
        "ord": lines["ord"].to_numpy(),
        "currency": lines["currency"].to_numpy(),
        "thstrm_nm": lines["thstrm_nm"].to_numpy(),
        "thstrm_amount": lines["thstrm_amount"].to_numpy(),
        "frmtrm_amount": lines["frmtrm_amount"].to_numpy(),
        "bfefrmtrm_amount": lines["bfefrmtrm_amount"].to_numpy(),
    })
    return out.sort_values(["code", "sj_div", "ord", "account_nm", "account_detail"]) \
              .reset_index(drop=True)


def attach_financial(frame: pd.DataFrame, *, as_of: AsOf, db_path=None) -> pd.DataFrame:
    """시세 표(`bas_dd`·`code` 가 있는 것)에 재무 18칸을 붙인다.

    행마다 **그 행의 날짜까지 알게 된 보고서 중 결산이 가장 늦은 것**을 붙인다. 보고서가
    아직 없는 행은 빈 칸이다 — 반출이 재무 때문에 죽어서는 안 되고, 빈 칸은 눈에 띈다.

    🔴 `as_of` 보다 뒤의 거래일이 든 표를 주면 세운다. `supply.price_series(as_of=...)` 로
       받은 표라면 그런 행이 없다.

    ⚠️ 입력 순서를 보존한다. `merge_asof` 는 정렬을 요구하므로 안에서 정렬했다가 되돌린다.
    """
    for col in ("bas_dd", "code"):
        if col not in frame.columns:
            raise ValueError(f"재무를 붙이려면 '{col}' 칸이 있어야 한다 — 시세 표를 넘겨라.")

    out = frame.copy()
    if out.empty:
        for col in FINANCIAL_COLUMNS:
            out[col] = pd.Series(dtype="object")
        return out

    days = out["bas_dd"].map(as_bas_dd)
    상한 = latest_known_day(as_of)
    if days.max() > 상한:
        raise ValueError(
            f"표에 as_of({to_kst(as_of).date()}) 시점에 아직 오지 않은 거래일이 있다 "
            f"(최대 {days.max()} > {상한}).\n"
            "  할 일: supply.price_series(as_of=...) 로 받은 표를 넘기거나 as_of 를 뒤로 옮긴다."
        )

    코드 = sorted(out["code"].astype(str).unique())
    with closing(_connect(db_path)) as conn:
        reports = _reports(conn, codes=코드, rcept_upto=상한, db_path=db_path)
        reports = reports[reports["known_at"] <= 상한] if not reports.empty else reports
        lines = _account_lines(conn, codes=코드, rcept_upto=상한,
                               rcept_nos=reports["rcept_no"] if not reports.empty else [],
                               representative_only=True)

    if reports.empty:
        for col in FINANCIAL_COLUMNS:
            out[col] = pd.Series([None] * len(out), index=out.index, dtype="object")
        return out

    table = _wide(reports, _account_values(lines)).set_index("fin_rcept_no")
    right = _timeline(reports)
    right["_key"] = right["known_at"].astype(int)

    left = pd.DataFrame({
        "code": out["code"].astype(str).to_numpy(),
        "_key": days.astype(int).to_numpy(),
        "_order": range(len(out)),
    })
    merged = pd.merge_asof(left.sort_values("_key"),
                           right[["_key", "code", "rcept_no"]].sort_values("_key"),
                           on="_key", by="code", direction="backward",
                           allow_exact_matches=True)
    rcept = merged.sort_values("_order")["rcept_no"]

    for col in FINANCIAL_COLUMNS:
        if col == "fin_rcept_no":
            out[col] = rcept.to_numpy()
        else:
            out[col] = table[col].reindex(rcept).to_numpy()
    return out


__all__ = [
    "FINANCIAL_ACCOUNTS",
    "FINANCIAL_COLUMNS",
    "FINANCIAL_LINE_COLUMNS",
    "FINANCIAL_META_COLUMNS",
    "REPORT_ORDER",
    "TAXONOMY_PREFIXES",
    "attach_financial",
    "financial_as_of",
    "financial_lines_as_of",
]
