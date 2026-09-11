"""공시 텍스트 — 그 행의 날짜에 **새로 알게 된** 공시와 제목 감성 확률.

## 무엇이 들어 있나 (실측 2026-09-11 · `main` `3fb42ee`)

    dart_disclosure  1,555,556행 · rcept_dt 20100104~20260904 · 유가 824,626 · 코스닥 730,930
                     종목코드 빈 값 0 · 종목·일당 평균 1.7건(p99 9건 · 최대 933건)
    text_signal         18,600행 · 고유 제목마다 한 줄 · 모델 1개(snunlp/KR-FinBert-SC @ f8586286)
                     text_sha = sha256(제목)[:16] (200/200 확인) · 공시 155만 행 100% 조인

## 🔴 이 표에는 날짜가 없다 — 시점은 공시에서 온다

`text_signal` 은 **고유 제목**마다 한 줄이다. 같은 제목이 2010년에도 2026년에도 나오므로
표 자체에는 *언제* 가 없다. 시점은 그 제목을 단 **공시의 접수일**에서 온다.

    dart_disclosure(rcept_dt · stock_code · report_nm) ─ 제목 ─ text_signal(p_neg · p_neu · p_pos)

`text_sha` 는 제목의 해시라, 제목으로 잇는 것과 해시로 잇는 것이 같다(인덱스가 제목에 있어
제목으로 잇는다). 점수를 2026-09-04 에 매긴 것은 미래참조가 아니다 — 점수는 **제목 문자열만의
함수**이고 다른 제목도 수익률도 보지 않는다. 모델 가중치는 리비전으로 고정돼 있다.

## 시점 규칙 — 접수일 다음 거래일 (재무와 같다 · HF 텍스트 반출과 같다)

`supply.clock.dart_known_at`. 행 T 에 **새로** 보이는 공시는 `known_at == T` 인 것이다.

    2024-03-12(화) 접수 → 20240313 행에 새로 보인다 → 20240314 시가 진입

## 종목 배정 — 공시의 `stock_code`

DART 공시 목록의 `stock_code` 는 **공시의 주인**(그 회사)이다. 제출인(`flr_nm`)이 증권사·
대주주여도 공시는 그 회사에 붙는다. 우선주는 따로 공시하지 않으므로 보통주 코드에만 붙는다.

## 🔴 `rm` 으로 거르지 않는다

공시 목록의 `rm` 칸에서 '정'(뒤에 정정됨)·'철'(철회됨)은 **나중에 붙는 표시**다. 그 공시가
나온 날에는 정정될지 철회될지 몰랐다. 이 칸으로 거르면 그 자체가 미래참조다. 해마다 '정' 이
6,725~15,601건, '철' 이 8~99건이다(실측).

## 🔴 신호는 없다 — 그래도 여는 이유

5일 방향과의 연관은 Cramér's V **0.026** 이다(2026-09-04 · `scripts/export_text_signal.py`).
제목에 결산연월이 박힌 정기보고서는 감성이 아니라 **시점**을 가리키므로, 제목을 그대로 피처로
쓰면 모델이 시점을 외운다. 이 문은 *"그날 무슨 공시가 났나"* 를 보는 이벤트 타임라인으로 연다.

    text_as_of("20240313", as_of="2024-03-14")          그날 새로 보인 공시 — 한 건당 한 행
    attach_text(price_frame, as_of="2024-08-31")        (종목, 일)마다 건수와 확률 평균 네 칸

창(5일·20일) 합계 같은 가공은 피처 파트의 몫이다.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from typing import Iterable, List, Optional, Sequence

import pandas as pd

from common.paths import krx_db_path
from supply.clock import (
    DART_KNOWN_RULE,
    AsOf,
    as_bas_dd,
    dart_known_at,
    latest_known_day,
    row_day,
    to_kst,
)

#: 시점 규칙의 이름 — 재무와 같은 DART 규칙이다.
TEXT_KNOWN_RULE = DART_KNOWN_RULE

#: 지금 DB 에 있는 유일한 감성 모델. `scripts/score_text_signal.MODEL_ID` 와 같아야 한다
#: (`tests/test_supply_text.py` 가 대조한다 — supply 는 scripts 를 import 하지 않는다).
DEFAULT_MODEL_ID = "snunlp/KR-FinBert-SC"

#: 점 조회가 내는 칸. **행이 0개여도 이 칸들은 있다.**
TEXT_COLUMNS = (
    "code", "rcept_no", "rcept_dt", "known_at", "report_nm", "text_sha",
    "model_id", "revision", "p_neg", "p_neu", "p_pos",
)

#: 붙이기가 내는 칸. `text_n` 은 그날 새로 보인 공시 수(수집 구간 밖이면 결측),
#: 확률 셋은 그 공시들의 평균(공시가 없거나 점수가 없으면 결측).
TEXT_DAILY_COLUMNS = ("text_n", "text_p_neg", "text_p_neu", "text_p_pos")

#: 접수일에서 다음 거래일까지 가장 긴 간격보다 넉넉히. 추석·설 연휴를 넘기고도 남는다.
_SPAN_DAYS = 15


# ==================================================
# 1. 작은 도구
# ==================================================
def _connect(db_path=None) -> sqlite3.Connection:
    """읽기만 한다. 경로를 인자로 받아 갈아 끼울 자리를 하나로 둔다."""
    conn = sqlite3.connect(db_path or krx_db_path(), timeout=60)
    conn.execute("PRAGMA busy_timeout=60000")
    return conn


def _empty(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})


def _normalize_codes(codes: Optional[Iterable[str]]) -> Optional[List[str]]:
    """종목코드를 **문자열 그대로**. 숫자로 바꾸지 않는다 — 5·6번째 자리에 영문이 있다."""
    if codes is None:
        return None
    if isinstance(codes, str):
        codes = [codes]
    return sorted({str(c).strip() for c in codes if str(c).strip()})


def _shift(day: str, days: int) -> str:
    return (pd.Timestamp(day) + pd.Timedelta(days=days)).strftime("%Y%m%d")


def _disclosures(conn: sqlite3.Connection, *, rcept_from: str, rcept_to: str,
                 codes: Optional[List[str]], model_id: str) -> pd.DataFrame:
    """접수일이 구간 안인 공시와 그 제목의 점수. 점수가 없는 공시도 남긴다(확률은 결측).

    `rm` 은 읽지 않는다 — 나중에 붙는 표시다(모듈 머리말).
    """
    조건, 인자 = "", []
    if codes is not None:
        조건 = f" AND d.stock_code IN ({','.join('?' * len(codes))})"
        인자 = list(codes)
    sql = f"""
      SELECT d.stock_code AS code, d.rcept_no, d.rcept_dt, d.report_nm,
             t.text_sha, t.model_id, t.revision, t.p_neg, t.p_neu, t.p_pos
      FROM dart_disclosure d
      LEFT JOIN text_signal t ON t.report_nm = d.report_nm AND t.model_id = ?
      WHERE d.rcept_dt BETWEEN ? AND ?
        AND d.stock_code IS NOT NULL AND d.stock_code <> ''{조건}
    """
    return pd.read_sql_query(sql, conn, params=[model_id, rcept_from, rcept_to, *인자])


def _with_known_at(frame: pd.DataFrame, db_path=None) -> pd.DataFrame:
    """`known_at` 을 붙인다. 달력 뒤라 다음 거래일을 모르면 행을 만들지 않는다."""
    if frame.empty:
        return frame.assign(known_at=pd.Series(dtype="object"))
    지도 = dart_known_at(frame["rcept_dt"].unique(), db_path=db_path)
    out = frame.assign(code=frame["code"].astype(str),
                       known_at=frame["rcept_dt"].map(지도))
    return out[out["known_at"].notna()].reset_index(drop=True)


# ==================================================
# 2. 정문 — as_of 없이는 못 지난다
# ==================================================
def text_as_of(bas_dd: AsOf, *, as_of: AsOf, start: Optional[AsOf] = None,
               codes: Optional[Iterable[str]] = None, model_id: str = DEFAULT_MODEL_ID,
               db_path=None) -> pd.DataFrame:
    """거래일 `bas_dd` 의 행에 **새로 보인** 공시 — 한 건당 한 행.

    `start` 를 주면 `start ~ bas_dd` 사이에 새로 보인 공시를 모두 준다(`known_at` 기준).
    칸: `TEXT_COLUMNS`. 순서는 known_at · code · rcept_dt · rcept_no.
    """
    끝 = row_day(bas_dd, as_of=as_of)
    처음 = as_bas_dd(start) if start is not None else 끝
    if 처음 > 끝:
        raise ValueError(f"start({처음}) 가 bas_dd({끝}) 보다 늦다.")
    코드 = _normalize_codes(codes)
    if 코드 == []:
        return _empty(TEXT_COLUMNS)

    with closing(_connect(db_path)) as conn:
        rows = _disclosures(conn, rcept_from=_shift(처음, -_SPAN_DAYS),
                            rcept_to=_shift(끝, -1), codes=코드, model_id=model_id)
    rows = _with_known_at(rows, db_path)
    rows = rows[(rows["known_at"] >= 처음) & (rows["known_at"] <= 끝)] if not rows.empty else rows
    if rows.empty:
        return _empty(TEXT_COLUMNS)
    return (rows.sort_values(["known_at", "code", "rcept_dt", "rcept_no"])[list(TEXT_COLUMNS)]
                .reset_index(drop=True))


def attach_text(frame: pd.DataFrame, *, as_of: AsOf, model_id: str = DEFAULT_MODEL_ID,
                db_path=None) -> pd.DataFrame:
    """시세 표(`bas_dd`·`code` 가 있는 것)에 그날 새로 보인 공시의 건수·확률 평균 네 칸을 붙인다.

    - `text_n` 공시가 없던 날은 **0** 이다 — 공시 목록은 날짜로 빠짐없이 받았으므로 "없었다" 는
      사실이다. 다만 **수집 구간 밖**의 행은 0 이 아니라 결측이다. 안 받은 날을 "공시 없음"
      으로 읽으면 조용히 틀린다.
    - 확률 셋은 그날 공시들의 평균이다. 공시가 없거나 그 제목에 점수가 없으면 결측이다.

    🔴 `as_of` 보다 뒤의 거래일이 든 표를 주면 세운다.
    ⚠️ 입력 순서를 보존한다.
    """
    for col in ("bas_dd", "code"):
        if col not in frame.columns:
            raise ValueError(f"공시 텍스트를 붙이려면 '{col}' 칸이 있어야 한다 — 시세 표를 넘겨라.")

    out = frame.copy()
    if out.empty:
        for col in TEXT_DAILY_COLUMNS:
            out[col] = pd.Series(dtype="object")
        return out

    days = out["bas_dd"].astype(str).str.replace("-", "", regex=False)
    상한 = latest_known_day(as_of)
    if days.max() > 상한:
        raise ValueError(
            f"표에 as_of({to_kst(as_of).date()}) 시점에 아직 오지 않은 거래일이 있다 "
            f"(최대 {days.max()} > {상한}).\n"
            "  할 일: supply.price_series(as_of=...) 로 받은 표를 넘기거나 as_of 를 뒤로 옮긴다."
        )

    코드 = sorted(out["code"].astype(str).unique())
    with closing(_connect(db_path)) as conn:
        범위 = conn.execute("SELECT MIN(rcept_dt), MAX(rcept_dt) FROM dart_disclosure").fetchone()
        rows = _disclosures(conn, rcept_from=_shift(days.min(), -_SPAN_DAYS),
                            rcept_to=_shift(days.max(), -1), codes=코드, model_id=model_id)
    rows = _with_known_at(rows, db_path)

    if rows.empty:
        daily = pd.DataFrame(columns=["code", "known_at", *TEXT_DAILY_COLUMNS])
    else:
        daily = (rows.groupby(["code", "known_at"], sort=False)
                     .agg(text_n=("rcept_no", "size"), text_p_neg=("p_neg", "mean"),
                          text_p_neu=("p_neu", "mean"), text_p_pos=("p_pos", "mean"))
                     .reset_index())

    left = pd.DataFrame({"code": out["code"].astype(str).to_numpy(),
                         "known_at": days.to_numpy(), "_order": range(len(out))})
    merged = (left.merge(daily, on=["code", "known_at"], how="left", validate="many_to_one")
                  .sort_values("_order"))

    # 수집 구간 — 첫 접수일의 다음 거래일 ~ 마지막 접수일의 다음 거래일. 이 안에서만 0 이 사실이다.
    if 범위[0] is None:
        covered = pd.Series(False, index=range(len(out)))
    else:
        지도 = dart_known_at([범위[0], 범위[1]], db_path=db_path)
        처음 = 지도.get(범위[0]) or 범위[0]
        끝 = 지도.get(범위[1]) or 범위[1]
        covered = pd.Series(((days >= 처음) & (days <= 끝)).to_numpy())

    n = pd.Series(pd.to_numeric(merged["text_n"], errors="coerce").fillna(0).to_numpy())
    out["text_n"] = n.where(covered).astype("Int64").to_numpy()
    for col in ("text_p_neg", "text_p_neu", "text_p_pos"):
        out[col] = pd.to_numeric(merged[col], errors="coerce").to_numpy()
    return out


__all__ = [
    "DEFAULT_MODEL_ID",
    "TEXT_COLUMNS",
    "TEXT_DAILY_COLUMNS",
    "TEXT_KNOWN_RULE",
    "attach_text",
    "text_as_of",
]
