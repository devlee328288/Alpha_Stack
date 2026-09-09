"""백테스트 원장(강민석 님 파트)과 데이터 파트 **사이의 계약서**.

## 왜 이 파일이 따로 있는가

`backtest/run_backtest()` 가 내는 `signal_log` · `trade_log` 는 강민석 님 것이고, 그것을
`daily_price` · 라벨과 맞대어 *"어떤 예측이 실제로 돈이 됐나"* 를 내는 것은 데이터 파트 것이다.
두 쪽이 각자 초록이어도 **칸 이름 하나 · 날짜 순서 하나**가 어긋나면 되먹임은 조용히 틀린다.
원장이 HF 에 올라오기 전에(2026-09-06 실측 · 아직 없다) 그 사이를 못박아 둔다.

## 두 층으로 잰다

1. **규격 층** — `supply/backtest_ledger.py::verify_ledger` 가 손으로 만든 원장에서 아홉 규칙을
   잡아내는가. 이쪽은 백테스트 코드를 import 하지 않는다.
2. **종단 층** — `run_backtest()` 를 **실제로 돌려** 나온 원장이 그 검증기를 통과하는가.
   칸 수(17 · 22) · 행 수(거래일 − 1) · 체결일(다음 거래일) · 실현수익률(종가로 재계산) ·
   구버전 예측 함수 호환까지 코드에서 그대로 잰다.

## 🔴 `datasets` 대역

`backtest_strategies.py` 는 최상단에서 `from datasets import Dataset` 을 한다. `pyproject.toml` 은
`datasets` 를 일부러 넣지 않으므로(74행) 선언된 환경에서는 모듈이 import 되지 않는다.
여기서는 import 가 실패할 때만 빈 대역 모듈을 `sys.modules` 에 끼워 `run_backtest` 만 읽는다 —
업로드 경로(`Dataset.from_pandas`)는 이 시험이 부르지 않는다. 진짜 `datasets` 가 있으면 그대로 쓴다.
"""

from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pandas as pd
import pytest

from supply import backtest_ledger as bl

HOLDOUT = "20240901"

#: `backtest_strategies.py:713` 이 아직 `close[T+5]/close[T]` 로 원장을 적는다(#176).
#: 검증기는 이미 시가축이라 종단 층은 지금 어긋난다 — 그쪽이 고쳐지면 `strict` 가
#: xpass 로 알려주므로, 그때 이 표시를 지운다.
_원장이_아직_종가축 = pytest.mark.xfail(
    strict=True,
    reason="#176 — backtest_strategies.py:713 이 아직 종가축이다",
)


# ── 손으로 만드는 원장 ────────────────────────────────────────────────────────

def _days(start: str, n: int) -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def _price_walk(days: pd.DatetimeIndex, seed: int = 0) -> pd.Series:
    """거래일 축의 임의 가격 시계열. 1층은 이것을 **시가**로 쓴다 — 실현수익률이
    시가[T+1]→시가[T+6] 축이라 재계산 대조도 그 축에서 해야 한다."""
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, len(days))), index=days)


def _ledger(days: pd.DatetimeIndex, open_: pd.Series, *, probs=(0.5, 0.3, 0.2),
            kind: str = "signal") -> pd.DataFrame:
    """`run_backtest()` 가 내는 모양을 그대로 흉내 낸다 — 마지막 거래일은 행이 없다."""
    pred = days[:-1]
    exe = days[1:]
    rr = bl.recompute_realized_return(pred, open_)
    base = {
        "prediction_date": pred, "execution_date": exe,
        "signal": [bl.LABELS[i % 3] for i in range(len(pred))],
        "consecutive_up": 0, "consecutive_down": 0,
        "requested_trade_ratio": 0.0, "actual_trade_value": 0.0, "position_ratio_after": 0.0,
    }
    if kind == "trade":
        base = {**base, "action": "buy", "actual_trade_ratio": 0.0, "price": 1.0,
                "quantity": 1.0, "trade_value": 1.0, "cost": 0.0}
        base.pop("actual_trade_value")
    df = pd.DataFrame(base)
    df["code"] = "KOSPI200"
    df["p_up"], df["p_flat"], df["p_down"] = probs
    df["realized_return_5d"] = rr.to_numpy()
    df["model_id"] = "test-v0"
    df["model_rev"] = "v0"
    df["run_id"] = "run_test"
    df["cost_rate"] = 0.001
    cols = bl.SIGNAL_LOG_COLUMNS if kind == "signal" else bl.TRADE_LOG_COLUMNS
    return df[list(cols)]


@pytest.fixture
def 개발구간():
    days = _days("2024-06-03", 40)          # 전부 홀드아웃 앞
    open_ = _price_walk(days)
    return days, open_, _ledger(days, open_)


# ── 1. 규격 층 ────────────────────────────────────────────────────────────────

def test_되먹임_칸_아홉이_이슈_110과_같다():
    assert bl.FEEDBACK_COLUMNS == (
        "code", "p_up", "p_flat", "p_down", "realized_return_5d",
        "model_id", "model_rev", "run_id", "cost_rate",
    )
    assert len(bl.SIGNAL_LOG_COLUMNS) == 17
    assert len(bl.TRADE_LOG_COLUMNS) == 22
    assert len(set(bl.SIGNAL_LOG_COLUMNS)) == 17 and len(set(bl.TRADE_LOG_COLUMNS)) == 22


def test_규격에_맞는_원장은_문제가_없다(개발구간):
    days, open_, df = 개발구간
    r = bl.verify_execution_log(df, holdout_start=HOLDOUT, open_prices=open_)
    assert r["problems"] == [], r
    assert r["rows"] == len(days) - 1
    assert r["summary"]["holdout_rows"] == 0
    assert r["summary"]["holdout_peek_rows"] == 0
    assert r["summary"]["realized_nan_rows"] == 5        # 청산일 t+6 이 축 밖인 마지막 다섯 행
    assert r["summary"]["missing_columns"] == []


def test_시가를_안_주면_그_검사만_건너뛴다고_말한다(개발구간):
    _, _, df = 개발구간
    r = bl.verify_execution_log(df, holdout_start=HOLDOUT)
    assert r["problems"] == []
    assert "건너뜀" in r["summary"]["note"]


@pytest.mark.parametrize("칸", bl.FEEDBACK_COLUMNS)
def test_되먹임_칸이_하나라도_빠지면_잡는다(개발구간, 칸):
    _, open_, df = 개발구간
    r = bl.verify_execution_log(df.drop(columns=[칸]), holdout_start=HOLDOUT, open_prices=open_)
    assert any("되먹임 칸이 없다" in p and 칸 in p for p in r["problems"]), r["problems"]


def test_확률_합이_1이_아니면_잡는다(개발구간):
    _, open_, df = 개발구간
    bad = df.copy()
    bad.loc[bad.index[:3], "p_up"] = 0.9              # 0.9 + 0.3 + 0.2 = 1.4
    r = bl.verify_execution_log(bad, holdout_start=HOLDOUT, open_prices=open_)
    assert any("합이 1 이 아니다" in p and "3행" in p for p in r["problems"]), r["problems"]


def test_확률이_범위_밖이거나_비면_잡는다(개발구간):
    _, open_, df = 개발구간
    bad = df.copy()
    bad.loc[bad.index[0], "p_down"] = -0.2
    bad.loc[bad.index[1], "p_flat"] = np.nan
    r = bl.verify_execution_log(bad, holdout_start=HOLDOUT, open_prices=open_)
    assert any("[0,1] 밖" in p for p in r["problems"]), r["problems"]
    assert any("결측" in p for p in r["problems"]), r["problems"]


def test_손으로_적은_0_33_0_34_0_33_은_통과한다(개발구간):
    days, open_, _ = 개발구간
    df = _ledger(days, open_, probs=(0.33, 0.34, 0.33))   # 지금 predict_5d_after 의 값
    r = bl.verify_execution_log(df, holdout_start=HOLDOUT, open_prices=open_)
    assert r["problems"] == []


def test_확률이_전_행_상수면_경고만_한다(개발구간):
    _, open_, df = 개발구간
    r = bl.verify_execution_log(df, holdout_start=HOLDOUT, open_prices=open_)
    assert r["problems"] == []
    assert any("상수" in w for w in r["warnings"]), r["warnings"]


def test_체결일이_예측일보다_뒤가_아니면_잡는다(개발구간):
    _, open_, df = 개발구간
    bad = df.copy()
    bad.loc[bad.index[0], "execution_date"] = bad.loc[bad.index[0], "prediction_date"]   # 같은 날
    전날 = bad.loc[bad.index[1], "prediction_date"] - pd.Timedelta(days=1)
    bad.loc[bad.index[1], "execution_date"] = 전날
    r = bl.verify_execution_log(bad, holdout_start=HOLDOUT, open_prices=open_)
    assert any("뒤가 아니다" in p and "2행" in p for p in r["problems"]), r["problems"]


def test_예측일이_홀드아웃_안이면_잡는다():
    days = _days("2024-08-19", 20)                      # 08-19 ~ 09-13 · 봉인선을 넘는다
    open_ = _price_walk(days)
    df = _ledger(days, open_)
    r = bl.verify_execution_log(df, holdout_start=HOLDOUT, open_prices=open_)
    sealed = int((pd.to_datetime(df["prediction_date"]) >= pd.Timestamp(HOLDOUT)).sum())
    assert sealed > 0
    assert r["summary"]["holdout_rows"] == sealed
    assert any("홀드아웃" in p and "예측일" in p for p in r["problems"]), r["problems"]


def test_실현수익률이_홀드아웃_시가를_엿보면_잡는다():
    # 예측일은 전부 개발구간(≤ 08-30)인데 청산일 t+6 이 09-02 이후인 행이 생기게 자료를 만든다
    days = _days("2024-08-05", 20)                      # 08-05 ~ 08-30
    open_ = _price_walk(days)
    extra = pd.bdate_range("2024-09-02", periods=6)      # 봉인 구간 시가를 축에 붙인다
    open_full = pd.concat([open_, _price_walk(extra, seed=1)])
    df = _ledger(days, open_)
    # 뒤 몇 행의 실현수익률을 "봉인 구간 시가로 계산한 값" 으로 채운다 — 엿본 원장
    peeked = bl.recompute_realized_return(df["prediction_date"], open_full)
    df["realized_return_5d"] = peeked.to_numpy()
    assert df["realized_return_5d"].notna().all()
    r = bl.verify_execution_log(df, holdout_start=HOLDOUT, open_prices=open_full)
    assert r["summary"]["holdout_rows"] == 0            # 예측일만 보면 깨끗하다
    assert r["summary"]["holdout_peek_rows"] > 0        # 그런데 값이 봉인 구간을 봤다
    assert any("엿본" in p for p in r["problems"]), r["problems"]


def test_실현수익률이_시가와_다르면_잡는다(개발구간):
    _, open_, df = 개발구간
    bad = df.copy()
    bad.loc[bad.index[0], "realized_return_5d"] += 0.01
    r = bl.verify_execution_log(bad, holdout_start=HOLDOUT, open_prices=open_)
    assert any("다시 계산한 값과 다르다" in p and "1행" in p for p in r["problems"]), r["problems"]


def test_라벨이_셋_밖이면_잡는다(개발구간):
    _, open_, df = 개발구간
    bad = df.copy()
    bad.loc[bad.index[0], "signal"] = "UP"
    r = bl.verify_execution_log(bad, holdout_start=HOLDOUT, open_prices=open_)
    assert any("signal 이" in p for p in r["problems"]), r["problems"]


def test_식별_칸이_비면_잡는다(개발구간):
    _, open_, df = 개발구간
    bad = df.copy()
    bad.loc[bad.index[0], "run_id"] = ""
    bad.loc[bad.index[1], "model_id"] = None
    r = bl.verify_execution_log(bad, holdout_start=HOLDOUT, open_prices=open_)
    assert any("run_id 가 비어" in p for p in r["problems"]), r["problems"]
    assert any("model_id 가 비어" in p for p in r["problems"]), r["problems"]


def test_체결_원장에_hold_가_섞이면_잡는다(개발구간):
    days, open_, _ = 개발구간
    df = _ledger(days, open_, kind="trade")
    ok = bl.verify_trade_log(df, holdout_start=HOLDOUT, open_prices=open_)
    assert ok["problems"] == []
    df.loc[df.index[0], "action"] = "hold"
    r = bl.verify_trade_log(df, holdout_start=HOLDOUT, open_prices=open_)
    assert any("hold" in p for p in r["problems"]), r["problems"]


def test_빈_원장은_칸만_보고_멈춘다():
    r = bl.verify_execution_log(pd.DataFrame(columns=list(bl.SIGNAL_LOG_COLUMNS)))
    assert r["rows"] == 0 and r["problems"] == []
    assert "빈 원장" in r["summary"]["note"]


def test_모아_올릴_때_붙는_칸은_허용한다(개발구간):
    _, open_, df = 개발구간
    df = df.assign(strategy="A", cost_key="mid")         # run_cost_sensitivity 가 붙이는 둘
    r = bl.verify_execution_log(df, holdout_start=HOLDOUT, open_prices=open_)
    assert r["problems"] == []
    assert r["summary"]["extra_columns"] == ["strategy", "cost_key"]


# ── 2. 종단 층 — run_backtest 를 실제로 돌린다 ───────────────────────────────

@pytest.fixture(scope="module")
def 백테스트():
    """`backtest.backtest_strategies` 를 읽는다. `datasets` 가 없거나 깨져 있으면 대역을 끼운다."""
    inserted = False
    try:
        importlib.import_module("datasets")
    except Exception:                                    # ModuleNotFoundError · 메타데이터 결함
        stub = types.ModuleType("datasets")
        stub.Dataset = object
        sys.modules["datasets"] = stub
        inserted = True
    try:
        mod = importlib.import_module("backtest.backtest_strategies")
    except ImportError as e:
        pytest.skip(f"backtest 모듈을 읽을 수 없다: {e}")
    yield mod
    if inserted:
        sys.modules.pop("datasets", None)


def _market(days: pd.DatetimeIndex, close: pd.Series) -> pd.DataFrame:
    """시가·종가가 있는 시장 표. **시가는 전일 종가와 다르다** — 밤사이 갭이 있다.

    갭을 안 넣으면 `open[t+1] == close[t]` 라 시가축과 종가축이 **우연히 같아져서**,
    원장이 엉뚱한 축으로 적혀 있어도 이 시험이 못 잡는다. 실제 시장에는 갭이 있으므로
    갭을 넣는 쪽이 현실이기도 하다.
    """
    rng = np.random.default_rng(97)
    gap = 1.0 + rng.normal(0, 0.004, len(close))
    open_ = close.shift(1).fillna(close.iloc[0]).to_numpy() * gap
    return pd.DataFrame({"open": open_, "close": close.to_numpy()}, index=days)


def _predict_with_probs(date, market_data):
    i = market_data.index.get_loc(date)
    return bl.LABELS[i % 3], {"p_up": 0.5, "p_flat": 0.3, "p_down": 0.2}


def _predict_legacy(date, market_data):
    return "상승"                                         # 문자열만 주는 구버전


@pytest.fixture(scope="module")
def 종단_원장(백테스트):
    days = _days("2024-06-03", 60)
    market = _market(days, _price_walk(days))
    res = 백테스트.run_backtest(market, days[0], days[-1], _predict_with_probs,
                            strategy="A", model_id="contract-v0", run_id="run_contract")
    return days, market, res


def test_종단_signal_log_칸이_계약과_정확히_같다(종단_원장):
    _, _, res = 종단_원장
    assert list(res["signal_log"].columns) == list(bl.SIGNAL_LOG_COLUMNS)
    assert list(res["trade_log"].columns) == list(bl.TRADE_LOG_COLUMNS)


def test_종단_행_수는_거래일_빼기_하나(종단_원장):
    days, _, res = 종단_원장
    assert len(res["signal_log"]) == len(days) - 1


@_원장이_아직_종가축
def test_종단_원장이_검증기를_통과한다(종단_원장):
    _, market, res = 종단_원장
    r = bl.verify_execution_log(res["signal_log"], holdout_start=HOLDOUT,
                                open_prices=market["open"])
    assert r["problems"] == [], r
    t = bl.verify_trade_log(res["trade_log"], holdout_start=HOLDOUT,
                            open_prices=market["open"])
    assert t["problems"] == [], t


def test_종단_체결일은_바로_다음_거래일이다(종단_원장):
    days, _, res = 종단_원장
    pos = {d: i for i, d in enumerate(days)}
    s = res["signal_log"]
    gaps = [pos[e] - pos[p]
            for p, e in zip(s["prediction_date"], s["execution_date"], strict=True)]
    assert set(gaps) == {1}


@_원장이_아직_종가축
def test_종단_실현수익률은_시가_T1에서_T6_수익률이다(종단_원장):
    _, market, res = 종단_원장
    s = res["signal_log"]
    mine = bl.recompute_realized_return(s["prediction_date"], market["open"])
    both = s["realized_return_5d"].notna().to_numpy() & mine.notna().to_numpy()
    assert both.sum() == len(s) - 5
    assert np.allclose(s["realized_return_5d"].to_numpy()[both], mine.to_numpy()[both], atol=1e-12)


def test_종단_확률이_원장에_그대로_실린다(종단_원장):
    _, _, res = 종단_원장
    s = res["signal_log"]
    assert (s["p_up"] == 0.5).all() and (s["p_flat"] == 0.3).all() and (s["p_down"] == 0.2).all()
    assert (s["model_id"] == "contract-v0").all() and (s["run_id"] == "run_contract").all()


@_원장이_아직_종가축
def test_종단_구버전_예측함수도_기본_확률로_채워_통과한다(백테스트):
    days = _days("2024-06-03", 30)
    market = _market(days, _price_walk(days, seed=2))
    res = 백테스트.run_backtest(market, days[0], days[-1], _predict_legacy,
                            strategy="B")
    s = res["signal_log"]
    assert (s["signal"] == "상승").all()
    assert np.allclose(s[["p_up", "p_flat", "p_down"]].sum(axis=1), 1.0, atol=bl.PROB_TOL)
    r = bl.verify_execution_log(s, holdout_start=HOLDOUT, open_prices=market["open"])
    assert r["problems"] == [], r
    assert s["run_id"].str.match(r"^run_\d{8}_\d{6}_[0-9a-f]{8}$").all()   # 자동 생성 규칙


def test_종단_홀드아웃을_넘겨_돌리면_검증기가_막는다(백테스트):
    days = _days("2024-08-12", 30)                       # 08-12 ~ 09-20 · 봉인선을 넘는다
    market = _market(days, _price_walk(days, seed=3))
    res = 백테스트.run_backtest(market, days[0], days[-1], _predict_with_probs,
                            strategy="A")
    r = bl.verify_execution_log(res["signal_log"], holdout_start=HOLDOUT,
                                open_prices=market["open"])
    assert r["summary"]["holdout_rows"] > 0
    assert r["summary"]["holdout_peek_rows"] > 0        # 청산일 t+6 이 봉인 구간
    assert len(r["problems"]) >= 2
