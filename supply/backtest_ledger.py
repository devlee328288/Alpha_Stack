"""백테스트 원장 — 학습에 되먹일 수 있는 모양인지 재는 계약서.

강사님 09-04 응답 *"백테스팅 결과를 데이터 형식으로 받아 학습·예측에 되먹이는 설계"* 의
데이터 파트 몫이다. 백테스트 코드(`backtest/`)는 강민석 님 파트라 **여기서는 원장을 만들지
않는다.** 원장이 나오면 그것이 학습에 들어갈 수 있는 모양인지 **재기만** 한다.

## 왜 원장인가 — 축이 다르다

HF 에 먼저 올라간 백테스트 결과 4종은 **전략 단위 요약**이다(샤프 0.62 · 손익분기 비용 …).
발표에는 맞지만 모델이 배우는 단위는 `(종목, 날짜)` 다. "A전략이 샤프 0.62" 는 모델이 배울 수
있는 형태가 아니다. 그래서 이슈 #110 에서 **날짜마다 한 줄인 원장**에 칸 아홉을 여쭈었고
(PR #131 로 들어왔다), 이 파일이 그 아홉 칸의 계약이다.

    code                 종목 축. 지금은 상수 "KOSPI200" 이라도 종목으로 넓힐 때 그대로 쓴다
    p_up p_flat p_down   예측 확률. 라벨만 있으면 "맞았다/틀렸다" 뿐이지만 확률이 있으면
                         **어느 확신 구간이 실제로 돈이 됐나** 를 잴 수 있다
    realized_return_5d   그 예측의 실제 5거래일 수익률
    model_id model_rev   어느 모델·어느 버전의 예측인가
    run_id cost_rate     같은 원장에 여러 실행이 섞일 때 가른다

## 무엇을 재나 (붉으면 반출하지 않는다)

1. 칸 — 아홉 칸이 전부 있는가
2. 확률 — 셋 다 [0, 1] 이고 합이 1 인가 · 결측이 없는가
3. 시점 — 체결일이 예측일보다 **뒤**인가 (t 에 예측 · t+1 시가 체결)
4. 봉인 — 예측일이 홀드아웃(20240901~) 안에 있는 행이 없는가
5. 실현수익률 — 종가를 주면 **다시 계산해 대조**한다. 그리고 t+5 종가가 홀드아웃 안이면
   실현수익률이 비어 있어야 한다 (개발구간 예측이 봉인 구간 가격을 엿보면 안 된다)
6. 라벨 — `signal` 이 상승·중립·하락 셋 중 하나인가
7. 식별 — `code` · `model_id` · `model_rev` · `run_id` 가 비어 있지 않은가 · `cost_rate` ≥ 0

경고(노랑 · 막지는 않는다): 확률이 전 행에서 상수이면 학습에 정보가 없다 — 지금 랜덤 예측이
정확히 그 상태다(전 행 0.33/0.34/0.33). 한 `model_id` 에 `model_rev` 가 여럿이면 어느 행이
어느 가중치인지 모른다.

## 🔴 이 검사가 안 보는 축

**예측이 맞는지는 안 본다.** 여기서 재는 것은 *"값이 규격에 맞나"* 이지 *"그 값이 옳은가"*
가 아니다. 규격을 통과한 랜덤 예측은 여전히 랜덤이다. 되먹임 피처("최근 20거래일 몇 번
맞았나")는 `known_at` 을 체결일 다음 거래일로 잡아야 하는 별개 설계라 여기 없다(2차).

## 원장은 어디서 오나

`backtest/run_cost_sensitivity.py` 가 `run_backtest()` 의 `signal_log`(hold 포함 신호 전량 ·
17칸) · `trade_log`(체결만 · 22칸)을 모아 아래 두 저장소로 올린다. 행 수는 **거래일 수 − 1**
이다 — 마지막 거래일은 다음 날 체결이 없어 원장에 남지 않는다. 2026-09-06 실측으로 아직
올라가 있지 않다(코드만 머지). 올라오면 `scripts/verify_*` 처럼 반출 게이트에 건다.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from evaluation.horizon import HOLDOUT_START

#: HF 저장소 — `backtest/run_cost_sensitivity.py` 가 올리는 이름 그대로.
HF_EXECUTION_LOG = "qurious-quant/alphastack-backtest-execution-log"
HF_TRADE_LOG = "qurious-quant/alphastack-backtest-trade-log"

#: 이슈 #110 에서 여쭙고 PR #131 로 들어온 되먹임 칸 아홉. 순서도 원장과 같다.
FEEDBACK_COLUMNS = (
    "code", "p_up", "p_flat", "p_down", "realized_return_5d",
    "model_id", "model_rev", "run_id", "cost_rate",
)

#: `run_backtest()` 의 `signal_log` 기존 8칸 (PR #131 이전부터 있던 것).
SIGNAL_LOG_BASE = (
    "prediction_date", "execution_date", "signal", "consecutive_up", "consecutive_down",
    "requested_trade_ratio", "actual_trade_value", "position_ratio_after",
)

#: `trade_log` 기존 13칸. (평가파트 v1.0 에 "14칸" 으로 적혔던 것은 오기 — 실행으로 셌다.)
TRADE_LOG_BASE = (
    "prediction_date", "execution_date", "signal", "consecutive_up", "consecutive_down",
    "action", "requested_trade_ratio", "actual_trade_ratio", "price", "quantity",
    "trade_value", "cost", "position_ratio_after",
)

SIGNAL_LOG_COLUMNS = SIGNAL_LOG_BASE + FEEDBACK_COLUMNS   # 17
TRADE_LOG_COLUMNS = TRADE_LOG_BASE + FEEDBACK_COLUMNS     # 22

PROB_COLUMNS = ("p_up", "p_flat", "p_down")
LABELS = ("상승", "중립", "하락")

#: 확률 합이 1 에서 이만큼 벗어나면 붉다. 0.33+0.34+0.33 처럼 손으로 적은 값도 통과해야 한다.
PROB_TOL = 1e-6
#: 실현수익률 재계산 허용 오차 (부동소수 표기 차이만 허용).
RETURN_TOL = 1e-9
#: 실현수익률이 앞을 보는 거래일 수. 예측 대상 ADR(5거래일)과 같다.
RETURN_HORIZON = 5


def _to_ts(values) -> pd.Series:
    """문자열이든 Timestamp 든 `datetime64[ns]` 로. 못 읽는 값은 NaT 로 남긴다."""
    return pd.to_datetime(pd.Series(values).reset_index(drop=True), errors="coerce")


def _holdout_ts(holdout_start: str) -> pd.Timestamp:
    return pd.Timestamp(str(holdout_start))


def recompute_realized_return(prediction_dates: Sequence, close: pd.Series,
                              horizon: int = RETURN_HORIZON) -> pd.Series:
    """`close[t+h] / close[t] − 1` 을 종가 축 위에서 다시 계산한다.

    `close` 의 인덱스가 **거래일 축**이다(원장의 예측일 열이 아니다 — 원장에는 마지막
    거래일이 없어 그 축으로 세면 하나가 짧다). `t+h` 가 축 밖이면 NaN.
    """
    close = close.sort_index()
    pos = {d: i for i, d in enumerate(pd.to_datetime(close.index))}
    vals = close.to_numpy(dtype="float64")
    out: List[float] = []
    for d in _to_ts(prediction_dates):
        i = pos.get(d)
        if i is None or i + horizon >= len(vals) or vals[i] == 0:
            out.append(np.nan)
        else:
            out.append(vals[i + horizon] / vals[i] - 1.0)
    return pd.Series(out, dtype="float64")


def verify_ledger(df: pd.DataFrame, *, kind: str = "signal",
                  holdout_start: str = HOLDOUT_START,
                  close: Optional[pd.Series] = None) -> Dict:
    """원장 한 장을 재서 `{"rows", "problems", "warnings", "summary"}` 로 돌려준다.

    `problems` 가 비어 있지 않으면 **반출하지 않는다.** `warnings` 는 알리기만 한다.

    - `kind` — `"signal"`(hold 포함 전량) 또는 `"trade"`(체결만). 기대 칸이 다르다
    - `close` — 거래일 축의 종가(`index` = 날짜). 주면 실현수익률을 다시 계산해 대조하고
      봉인 구간 가격을 엿본 행을 잡는다. 안 주면 그 두 검사는 건너뛴다(`summary` 에 적는다)
    """
    if kind not in ("signal", "trade"):
        raise ValueError(f"kind 는 'signal' 또는 'trade' 여야 한다 (받은 값: {kind!r})")
    expected = SIGNAL_LOG_COLUMNS if kind == "signal" else TRADE_LOG_COLUMNS

    problems: List[str] = []
    warnings: List[str] = []
    summary: Dict = {"kind": kind, "holdout_start": str(holdout_start)}
    n = int(len(df))

    # 1. 칸
    missing_feedback = [c for c in FEEDBACK_COLUMNS if c not in df.columns]
    if missing_feedback:
        problems.append(f"되먹임 칸이 없다: {missing_feedback}")
    missing_all = [c for c in expected if c not in df.columns]
    summary["missing_columns"] = missing_all
    extra = [c for c in df.columns if c not in expected]
    summary["extra_columns"] = extra   # strategy · cost_key 처럼 모을 때 붙는 칸은 허용
    if n == 0:
        summary["note"] = "빈 원장 — 값 검사는 건너뜀"
        return {"rows": 0, "problems": problems, "warnings": warnings, "summary": summary}
    if missing_feedback:
        # 칸이 없으면 값 검사가 의미 없다. 여기서 멈춘다.
        return {"rows": n, "problems": problems, "warnings": warnings, "summary": summary}

    # 2. 확률
    probs = df[list(PROB_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    nan_rows = int(probs.isna().any(axis=1).sum())
    if nan_rows:
        problems.append(f"확률에 결측/비수치가 있다: {nan_rows:,}행")
    out_of_range = int(((probs < 0) | (probs > 1)).any(axis=1).sum())
    if out_of_range:
        problems.append(f"확률이 [0,1] 밖이다: {out_of_range:,}행")
    sums = probs.sum(axis=1)
    off = int((((sums - 1.0).abs() > PROB_TOL) & probs.notna().all(axis=1)).sum())
    if off:
        problems.append(f"확률 셋의 합이 1 이 아니다 (허용 {PROB_TOL}): {off:,}행")
    distinct = probs.dropna().drop_duplicates()
    summary["prob_distinct_rows"] = int(len(distinct))
    if len(distinct) == 1 and n > 1:
        warnings.append("확률이 전 행에서 상수다 — 학습에 정보가 없다 "
                        f"({distinct.iloc[0].round(4).tolist()})")

    # 3. 시점
    pred = _to_ts(df["prediction_date"]) if "prediction_date" in df.columns else None
    exe = _to_ts(df["execution_date"]) if "execution_date" in df.columns else None
    if pred is None or exe is None:
        problems.append("prediction_date / execution_date 칸이 없다")
    else:
        bad_dates = int(pred.isna().sum() + exe.isna().sum())
        if bad_dates:
            problems.append(f"날짜를 못 읽는 값이 있다: {bad_dates:,}개")
        not_after = int((exe <= pred).sum())
        if not_after:
            problems.append(f"체결일이 예측일보다 뒤가 아니다: {not_after:,}행")

        # 4. 봉인
        h = _holdout_ts(holdout_start)
        sealed = int((pred >= h).sum())
        summary["holdout_rows"] = sealed
        if sealed:
            problems.append(f"예측일이 홀드아웃({holdout_start}~) 안인 행: {sealed:,}")

        # 5. 실현수익률
        rr = pd.to_numeric(df["realized_return_5d"], errors="coerce").reset_index(drop=True)
        summary["realized_nan_rows"] = int(rr.isna().sum())
        if close is not None:
            close = close.copy()
            close.index = pd.to_datetime(close.index)
            recomputed = recompute_realized_return(pred, close)
            both = rr.notna() & recomputed.notna()
            mismatch = int(((rr - recomputed).abs() > RETURN_TOL)[both].sum())
            only_one = int((rr.notna() != recomputed.notna()).sum())
            if mismatch:
                problems.append(f"실현수익률이 종가로 다시 계산한 값과 다르다: {mismatch:,}행")
            if only_one:
                problems.append("실현수익률의 결측 자리가 종가로 계산한 것과 어긋난다: "
                                f"{only_one:,}행")
            # t+5 가 봉인 구간이면 값이 비어 있어야 한다
            idx = pd.to_datetime(close.sort_index().index)
            pos = {d: i for i, d in enumerate(idx)}
            peek = 0
            for d, v in zip(pred, rr, strict=True):
                i = pos.get(d)
                if i is None or pd.isna(v):
                    continue
                if i + RETURN_HORIZON < len(idx) and idx[i + RETURN_HORIZON] >= h:
                    peek += 1
            summary["holdout_peek_rows"] = peek
            if peek:
                problems.append(f"실현수익률이 홀드아웃 종가를 엿본 행: {peek:,}")
        else:
            summary["note"] = "close 를 안 줘서 실현수익률 재계산·봉인 엿보기 검사는 건너뜀"

    # 6. 라벨
    if "signal" in df.columns:
        bad_label = int((~df["signal"].isin(LABELS)).sum())
        if bad_label:
            problems.append(f"signal 이 {LABELS} 밖이다: {bad_label:,}행")
        summary["signal_counts"] = {k: int(v) for k, v in df["signal"].value_counts().items()}

    # 7. 식별
    for c in ("code", "model_id", "model_rev", "run_id"):
        s = df[c].astype("string")
        empty = int((s.isna() | (s.str.strip() == "")).sum())
        if empty:
            problems.append(f"{c} 가 비어 있다: {empty:,}행")
    cost = pd.to_numeric(df["cost_rate"], errors="coerce")
    if int((cost.isna() | (cost < 0)).sum()):
        problems.append("cost_rate 가 비었거나 음수다")
    revs = df.groupby("model_id")["model_rev"].nunique()
    multi = revs[revs > 1]
    if len(multi):
        warnings.append(f"model_id 하나에 model_rev 가 여럿이다: {multi.to_dict()}")
    summary["run_ids"] = int(df["run_id"].nunique())
    summary["model_ids"] = sorted(df["model_id"].dropna().astype(str).unique().tolist())

    return {"rows": n, "problems": problems, "warnings": warnings, "summary": summary}


def verify_execution_log(df: pd.DataFrame, **kw) -> Dict:
    """`signal_log`(hold 포함 전량) 용 — 17칸."""
    return verify_ledger(df, kind="signal", **kw)


def verify_trade_log(df: pd.DataFrame, **kw) -> Dict:
    """`trade_log`(체결만) 용 — 22칸. `action` 이 hold 인 행이 섞이면 붉다."""
    out = verify_ledger(df, kind="trade", **kw)
    if len(df) and "action" in df.columns:
        holds = int((df["action"] == "hold").sum())
        if holds:
            out["problems"].append(f"체결 원장에 hold 행이 섞였다: {holds:,}행")
    return out
