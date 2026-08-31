"""공급 계층과 피처 계층 **사이의 계약서**.

## 왜 이 파일이 따로 있는가

두 계층에 이미 테스트가 있다. 그런데 둘 다 **자기 쪽만 본다.**

    tests/test_supply_*.py        supply 가 무엇을 주는지 — features 를 import 하지 않는다
    tests/test_features_*.py      features 가 무엇을 계산하는지 — 입력을 손으로 적는다
                                  (import 가 math·pytest·features 뿐이다. 실측)

그래서 **둘 다 초록인 채로 서로 안 맞을 수 있다.** 손으로 적은 `[1, 2, 3, 4, 5]` 로는
드러나지 않고 진짜 표를 물렸을 때만 드러나는 어긋남이 있다 — 빈 표의 dtype, 결측이
섞인 창, 인자 순서. 그 어긋남은 **예외를 내지 않는다.** 숫자가 그냥 조금 달라진다.

이 파일은 그 사이를 못박는다. `supply` 가 실제로 내보내는 표를 `features` 의 공개 함수
**14개 전부**에 물려서, 아래 여섯 가지가 참인지 본다.

    ① 길이       출력 길이 == 입력 길이. 행이 조용히 늘거나 줄지 않는다
    ② 정렬       과거 → 현재. 뒤집히면 이동평균·차분이 전부 뒤집힌 값을 낸다
    ③ dtype      numpy 계열(float64/int64). pandas nullable 은 결측이 드는 순간 터진다
    ④ 워밍업     NaN 은 **앞쪽 연속 구간에만**. 중간에 뚫리면 시간 순서가 깨진다
    ⑤ 빈 표      0행을 넣으면 0행이 나온다
    ⑥ 호출 방식  `as_of` 는 키워드 전용. 위치로 넘길 수 없다

## 왜 이 여섯 가지인가 — 수업 자료가 그대로 요구한다

`learning/09-finance-ml/3.회귀모델_주가예측.ipynb` 가 쓰는 네 줄이 근거다.

    samsung["Close_Lag1"] = samsung["Close"].shift(1)     # ② 정렬이 뒤집히면 미래를 당겨 온다
    data = samsung.dropna(subset = features)              # ④ NaN 이 앞에 몰려야 안전하다
    train_data = data.iloc[:train_size]                   # ① 길이가 어긋나면 분할점이 밀린다
    scaled_train = ss.fit_transform(x_train)              # ③ StandardScaler 는 numpy 를 받는다

`dropna` 가 특히 그렇다. NaN 이 앞쪽 워밍업 구간에만 있으면 앞을 자르는 것과 같아서
시간 순서가 보존된다. 그런데 **중간에 NaN 이 하나 끼면 그 행만 빠지고 앞뒤가 붙는다** —
`shift(1)` 이 가리키는 "어제" 가 어제가 아니게 되는데, 예외는 나지 않는다.

## 알려진 결함은 xfail 로 남긴다

계약을 세워 보니 `features/` 결함 4건(이슈 #17~#20)이 그대로 걸린다. `features/` 는
신장환 담당이라 여기서 고치지 않는다. 대신 `xfail(strict=True)` 로 표시한다.

    strict=True 라서 **고쳐지면 xpass 로 실패한다.** 고친 줄 모르고 지나갈 수가 없다.

reason 에 이슈 번호를 적어 두었으니 닫을 때 이 파일도 함께 손본다.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import pytest

import supply
from features import indicators, volatility, volume
from ingest.store import krx_index, krx_store

지수명 = "코스피 200"
종목코드 = "005930"

# 60거래일. 가장 긴 창(20)이 두 번 이상 들어가야 워밍업 뒤 구간이 남는다 —
# 40일만 주면 "NaN 아닌 값" 이 몇 개뿐이라 ④가 사실상 아무것도 못 잰다.
거래일수 = 60


def _평일들(n: int) -> List[str]:
    """`YYYYMMDD` 평일 n 개. 주말을 건너뛰는 이유는 `as_of` 경계가 달력을 보기 때문이다."""
    from datetime import date, timedelta

    out: List[str] = []
    day = date(2024, 1, 2)
    while len(out) < n:
        if day.weekday() < 5:
            out.append(day.strftime("%Y%m%d"))
        day += timedelta(days=1)
    return out


거래일 = _평일들(거래일수)


def _시세(i: int) -> Dict[str, float]:
    """i 번째 거래일의 가격 한 벌. **값이 움직여야 계약을 잴 수 있다.**

    전부 같은 값(1000, 1000, …)으로 채우면 표준편차가 0 이 되어 변동성 지표가
    통째로 0 이나 NaN 으로 나온다. 그러면 ④(워밍업 NaN 위치)가 "전부 NaN" 을 통과로
    읽어서 **아무것도 검사하지 못한다.** 그래서 추세 + 진동을 섞는다.

    난수를 쓰지 않는 이유는 실패가 재현돼야 하기 때문이다.
    """
    종가 = 1000.0 + i * 3.0 + 40.0 * math.sin(i / 4.0)
    시가 = 종가 - 2.0 * math.cos(i / 3.0)
    고가 = max(시가, 종가) + 5.0 + 3.0 * abs(math.sin(i / 5.0))
    저가 = min(시가, 종가) - 5.0 - 3.0 * abs(math.cos(i / 5.0))
    거래량 = 100_000 + 7_000 * i + 30_000 * abs(math.sin(i / 6.0))
    return {
        "open": round(시가, 2), "high": round(고가, 2), "low": round(저가, 2),
        "close": round(종가, 2), "volume": int(거래량),
    }


@pytest.fixture()
def 저장소(tmp_path, monkeypatch):
    """지수 1종·종목 1개가 든 임시 DB.

    ⚠️ `krx_index` 는 `krx_store.DB_PATH` 를 **import 시점에 자기 이름으로 가져간다.**
       한쪽만 갈아 끼우면 지수는 임시 DB, 종목은 다른 DB 를 보게 된다. 둘 다 바꾼다.
       (`tests/conftest.py` 가 진짜 DB 를 이미 막고 있지만, 막혔다는 사실에 기대어
        경로를 대충 두면 언젠가 그 안전장치만 남고 의도가 사라진다.)
    """
    db = tmp_path / "contract.db"
    monkeypatch.setattr(krx_store, "DB_PATH", db)
    monkeypatch.setattr(krx_index, "DB_PATH", db)
    krx_store.init_db()
    krx_index.init_db()

    지수행, 종목행 = [], []
    for i, day in enumerate(거래일):
        v = _시세(i)
        지수행.append((day, 지수명, "KOSPI", v["open"], v["high"], v["low"],
                     v["close"], 0.0, 0.0, v["volume"], v["volume"] * 100, 1_000_000))
        종목행.append((day, 종목코드, "삼성전자", "KOSPI", v["open"], v["high"], v["low"],
                     v["close"], 0.0, 0.0, v["volume"], v["volume"] * 100, 1_000_000, 100))

    with krx_store.connect() as conn:
        conn.executemany(
            "INSERT INTO index_price (bas_dd, index_name, index_class, open, high, low, "
            "close, change, change_rate, volume, value, market_cap) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", 지수행,
        )
        conn.executemany(
            "INSERT INTO daily_price (bas_dd, code, name, market, open, high, low, "
            "close, change, change_rate, volume, value, market_cap, listed_shares) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", 종목행,
        )
    return db


# ── 검사 대상 — features 의 공개 함수 14개 전부 ────────────────────────────
#
# ⚠️ **14개다.** `features/` 에 def 는 19개지만 5개(`_to_array` 3벌 · `_ewm_mean` ·
#    `_wilder_smooth`)는 밑줄로 시작하는 내부 헬퍼다. 계약은 밖에서 부르는 것에만 선다.
#    `test_공개함수를_하나도_빠뜨리지_않았다` 가 이 수를 실제 모듈과 맞춰 본다.
#
# 워밍업 은 "앞에서 몇 개가 NaN 이어야 하는가" 다. 창 크기와 다른 값이 섞여 있는데
# 그게 정상이다 — `historical_volatility` 는 수익률을 먼저 만들어 한 개를 더 잃고,
# `atr` 은 첫 행의 참범위를 고가-저가로 채워 한 개를 덜 잃는다.

호출 = Tuple[str, Callable[[Dict[str, pd.Series]], object], int]

CALLS: List[호출] = [
    ("sma",                   lambda d: indicators.sma(d["close"], window=20), 19),
    ("ema",                   lambda d: indicators.ema(d["close"], span=20), 19),
    ("rsi",                   lambda d: indicators.rsi(d["close"], window=14), 14),
    ("macd",                  lambda d: indicators.macd(d["close"]), 0),
    ("bollinger_bands",       lambda d: indicators.bollinger_bands(d["close"], window=20), 0),
    ("true_range",            lambda d: volatility.true_range(d["high"], d["low"],
                                                              d["close"]), 0),
    ("atr",                   lambda d: volatility.atr(d["high"], d["low"], d["close"],
                                                       window=14), 13),
    ("historical_volatility", lambda d: volatility.historical_volatility(d["close"],
                                                                        window=20), 20),
    ("parkinson_volatility",  lambda d: volatility.parkinson_volatility(d["high"], d["low"],
                                                                       window=20), 19),
    ("volume_sma",            lambda d: volume.volume_sma(d["volume"], window=20), 19),
    ("volume_ratio",          lambda d: volume.volume_ratio(d["volume"], window=20), 19),
    ("obv",                   lambda d: volume.obv(d["close"], d["volume"]), 0),
    ("vwap",                  lambda d: volume.vwap(d["close"], d["volume"], window=20), 19),
    ("volume_roc",            lambda d: volume.volume_roc(d["volume"], window=5), 5),
]

이름들 = [name for name, _, _ in CALLS]

#: 이름으로 호출을 찾는 표. 검사마다 `zip` 을 다시 엮으면 `CALLS` 의 순서가 바뀌었을 때
#: 이름과 함수가 어긋난 채로 계속 통과한다 — 한 곳에서 한 번만 엮는다.
함수표 = {name: fn for name, fn, _ in CALLS}


def _칸(df: pd.DataFrame) -> Dict[str, pd.Series]:
    return {c: df[c] for c in ("open", "high", "low", "close", "volume")}


def _배열들(결과) -> List[np.ndarray]:
    """반환이 배열이든 dict 든 tuple 이든 **배열 목록**으로 편다.

    `macd` 는 dict(macd/signal/hist), `bollinger_bands` 는 dict(mid/upper/lower/bandwidth)
    를 준다. 계약은 그 안의 모든 계열에 똑같이 걸린다.
    """
    if isinstance(결과, dict):
        return [np.asarray(v) for v in 결과.values()]
    if isinstance(결과, tuple):
        return [np.asarray(v) for v in 결과]
    return [np.asarray(결과)]


def _두_표(저장소) -> List[Tuple[str, pd.DataFrame]]:
    """계약이 걸리는 두 문. 종목도 지수와 **같은 계약**이다."""
    미래 = "2099-01-01"
    return [
        ("index_series", supply.index_series(지수명, as_of=미래)),
        ("price_series", supply.price_series(종목코드, as_of=미래)),
    ]


# ── 시험대 자체를 먼저 믿을 수 있게 한다 ───────────────────────────────────

def test_공개함수를_하나도_빠뜨리지_않았다():
    """🔴 이 검사가 없으면 계약이 **조용히 낡는다.**

    신장환이 함수를 하나 추가해도 위 `CALLS` 에 안 적으면 아무 일도 안 일어난다.
    새 함수는 계약 없이 들어오고, 그래도 이 파일은 초록이다. 그래서 실제 모듈의
    공개 함수 목록과 맞춰 본다 — 어긋나면 여기서 멈춘다.
    """
    import inspect

    실제: List[str] = []
    for 모듈 in (indicators, volatility, volume):
        for 이름, 대상 in inspect.getmembers(모듈, inspect.isfunction):
            # 다른 모듈에서 import 해 온 것은 그 모듈의 계약이 아니다
            if 대상.__module__ == 모듈.__name__ and not 이름.startswith("_"):
                실제.append(이름)

    assert sorted(실제) == sorted(이름들), (
        f"features 공개 함수가 바뀌었다. 계약에 없는 것: {sorted(set(실제) - set(이름들))} / "
        f"사라진 것: {sorted(set(이름들) - set(실제))}"
    )


def test_시험대가_움직이는_값을_준다(저장소):
    """고정값 표로는 ④가 아무것도 못 잰다 — `_시세` 의 진동이 살아 있는지 본다."""
    df = supply.index_series(지수명, as_of="2099-01-01")

    assert df["close"].std() > 1.0, "종가가 거의 안 움직인다 — 변동성 지표가 0 이 된다"
    assert (df["high"] >= df["close"]).all(), "고가가 종가보다 낮다 — 만들어 낸 표가 틀렸다"
    assert (df["low"] <= df["close"]).all(), "저가가 종가보다 높다 — 만들어 낸 표가 틀렸다"


# ── ① 길이 ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("이름", 이름들)
def test_계약1_출력_길이가_입력_길이와_같다(저장소, 이름):
    """행이 조용히 늘거나 줄면 `iloc[:train_size]` 의 분할점이 밀린다.

    수업 자료가 `train_size = int(len(data) * 0.8)` 으로 자르는데, 피처마다 길이가
    다르면 같은 인덱스가 서로 다른 날짜를 가리킨다. 그런데도 예외는 안 난다.
    """
    호출 = 함수표[이름]

    for 문, df in _두_표(저장소):
        for arr in _배열들(호출(_칸(df))):
            assert len(arr) == len(df), (
                f"{문} → {이름}: {len(df)}행을 넣었는데 {len(arr)}개가 나왔다"
            )


# ── ② 정렬 ────────────────────────────────────────────────────────────────

def test_계약2_공급이_과거에서_현재_순으로_준다(저장소):
    for 문, df in _두_표(저장소):
        assert df["bas_dd"].is_monotonic_increasing, f"{문} 이 과거→현재가 아니다"
        assert df["bas_dd"].iloc[0] == 거래일[0], f"{문} 의 첫 행이 가장 이른 날이 아니다"
        assert df["bas_dd"].iloc[-1] == 거래일[-1], f"{문} 의 끝 행이 가장 늦은 날이 아니다"


@pytest.mark.parametrize("이름", 이름들)
def test_계약2_방향이_뒤집히면_값이_달라진다(저장소, 이름):
    """🔴 **정렬이 계약인 이유를 여기서 증명한다.**

    "과거→현재" 를 지켜야 한다고 말만 해서는 부족하다. 뒤집었을 때 결과가 같다면
    그 함수에게 순서는 아무 의미가 없고, 계약 ②는 그 함수에 대해 빈 약속이다.
    뒤집었을 때 **달라져야** 순서가 실제로 뜻을 갖는다.

    바꿔 말하면 이 검사가 실패하는 날은, 누군가 `supply` 를 최근순으로 되돌려도
    아무 검사도 울리지 않게 된 날이다.
    """
    호출 = 함수표[이름]
    df = supply.index_series(지수명, as_of="2099-01-01")

    정 = _배열들(호출(_칸(df)))
    역 = _배열들(호출(_칸(df.iloc[::-1].reset_index(drop=True))))

    달라진_계열 = [
        not np.allclose(np.nan_to_num(a, nan=-9e9), np.nan_to_num(b[::-1], nan=-9e9))
        for a, b in zip(정, 역, strict=True)
    ]
    assert any(달라진_계열), (
        f"{이름}: 표를 뒤집어도 결과가 같다 — 이 함수에게는 정렬이 아무 뜻이 없다"
    )


# ── ③ dtype ───────────────────────────────────────────────────────────────

def test_계약3_공급이_numpy_계열_dtype으로_준다(저장소):
    """pandas nullable(`Int64`·`Float64`)로 바꾸면 features 가 결측에서 터진다.

    ⚠️ 지금은 지키고 있다. 이 검사는 **앞으로도 지키게** 하려고 둔다 —
       `astype("Int64")` 한 줄이면 조용히 넘어갈 수 있는 경계다.
       터지는 조건은 `test_④_결측이_들어오면` 아래 xfail 이 보여 준다.
    """
    for 문, df in _두_표(저장소):
        for 칸 in ("open", "high", "low", "close", "volume"):
            dtype = df[칸].dtype
            assert isinstance(dtype, np.dtype), (
                f"{문}.{칸} 이 pandas 확장 dtype({dtype})이다 — features 가 결측에서 터진다"
            )
            assert dtype.kind in "fiu", f"{문}.{칸} 이 수치가 아니다 ({dtype})"


@pytest.mark.parametrize("이름", 이름들)
def test_계약3_출력이_실수_배열이다(저장소, 이름):
    호출 = 함수표[이름]
    df = supply.index_series(지수명, as_of="2099-01-01")

    for arr in _배열들(호출(_칸(df))):
        assert arr.dtype.kind == "f", f"{이름}: 실수 배열이 아니다 ({arr.dtype})"


# ── ④ 워밍업 NaN 위치 ─────────────────────────────────────────────────────

@pytest.mark.parametrize("이름,워밍업", [(n, w) for n, _, w in CALLS])
def test_계약4_NaN은_앞쪽_연속구간에만_있다(저장소, 이름, 워밍업):
    """🔴 중간에 NaN 이 뚫리면 `dropna` 가 **시간 순서를 조용히 접는다.**

    수업 자료가 `data = samsung.dropna(subset = features)` 를 쓴다. NaN 이 앞에만
    있으면 이건 "앞을 자른다" 와 같아서 안전하다. 그런데 중간 한 행이 빠지면 그
    자리에서 앞뒤가 붙고, `Close_Lag1` 이 가리키는 "어제" 가 어제가 아니게 된다.
    붙은 자리는 표를 봐도 티가 안 난다 — 행이 하나 없을 뿐이다.
    """
    호출 = 함수표[이름]

    for 문, df in _두_표(저장소):
        for arr in _배열들(호출(_칸(df))):
            nan = np.isnan(arr)
            # 전부 NaN 은 여기서 볼 일이 아니다 — 순서를 접지 않는다.
            # 아래 `test_계약4_전부_NaN인_지표는_없다` 가 따로 본다.
            if not nan.any() or nan.all():
                continue
            첫_유효 = int(np.argmax(~nan))
            assert not nan[첫_유효:].any(), (
                f"{문} → {이름}: 첫 유효값({첫_유효}) 뒤에 NaN 이 또 있다 "
                f"— dropna 가 시간 순서를 접는다"
            )


@pytest.mark.parametrize("이름", 이름들)
def test_계약4_전부_NaN인_지표는_없다(저장소, 이름):
    """60거래일을 넣었는데 통째로 NaN 이 나오면 그 피처는 학습에 한 줄도 못 낸다.

    ④가 전부-NaN 을 통과로 읽어야 하는 것과는 별개로, 실제로 그런 일이 생기면
    `dropna(subset=features)` 가 **표를 통째로 비운다.** 창 크기를 잘못 키웠을 때
    조용히 그렇게 되므로 여기서 따로 막는다.
    """
    호출 = 함수표[이름]

    for 문, df in _두_표(저장소):
        for arr in _배열들(호출(_칸(df))):
            assert not np.isnan(arr).all(), (
                f"{문} → {이름}: {len(df)}행을 넣었는데 결과가 전부 NaN 이다"
            )


@pytest.mark.parametrize("이름,워밍업", [(n, w) for n, _, w in CALLS if w > 0])
def test_계약4_워밍업_길이가_실측과_같다(저장소, 이름, 워밍업):
    """워밍업 개수를 못박아 회귀를 잡는다.

    ⚠️ 이 숫자는 **2026-08-31 에 실제 표로 재서** 적었다. 창 크기에서 유추한 값이
       아니다 — `historical_volatility` 는 수익률을 먼저 만들어 하나를 더 잃고,
       `atr` 은 첫 참범위를 고가-저가로 채워 하나를 덜 잃는다. 유추했다면 둘 다 틀렸다.
    """
    호출 = 함수표[이름]
    df = supply.index_series(지수명, as_of="2099-01-01")

    for arr in _배열들(호출(_칸(df))):
        assert int(np.isnan(arr).sum()) == 워밍업, (
            f"{이름}: 워밍업 NaN 이 {워밍업}개여야 하는데 {int(np.isnan(arr).sum())}개다"
        )


@pytest.mark.xfail(strict=True, reason="이슈 #18 — 창 안의 결측을 0 으로 세고 나눈다")
def test_계약4_결측이_섞이면_그_창은_NaN이어야_한다(저장소):
    """🔴 지금은 **오염된 숫자**가 나온다. NaN 이 아니라 그럴듯한 값이라 더 위험하다.

    거래량 한 칸을 NaN 으로 만들면, 그 칸이 든 20일 창은 전부 NaN 이어야 한다.
    "모르는 값이 섞인 평균" 은 평균이 아니기 때문이다. 그런데 현재 구현은 결측을
    0 으로 세고 20 으로 나눠서, 창이 하나씩 지날 때마다 **평균을 낮게** 잡는다.
    낮아진 평균으로 나누는 `volume_ratio` 는 그만큼 **부풀어 오른다.**

    실측(2026-08-31, 코스피 200 4,093행): 109행이 0.9005 → 0.9469. 5.2% 부풀었다.
    NaN 이 하나 늘 뿐 나머지 19개 창은 조용히 틀린 값을 낸다.
    """
    df = supply.index_series(지수명, as_of="2099-01-01")
    칸 = _칸(df)
    더러운 = 칸["volume"].astype(float).copy()
    더러운.iloc[30] = np.nan

    깨끗 = volume.volume_ratio(칸["volume"], window=20)
    오염 = volume.volume_ratio(더러운, window=20)

    # 30 번 칸이 든 창은 30~49. 그 구간이 전부 NaN 이어야 한다
    창 = slice(30, 50)
    assert np.isnan(오염[창]).all(), (
        f"결측이 든 창이 숫자를 냈다 — 깨끗 {깨끗[35]:.4f} vs 오염 {오염[35]:.4f}"
    )


@pytest.mark.xfail(strict=True, reason="이슈 #20 — pd.NA 를 float() 에 그대로 넘긴다")
@pytest.mark.parametrize("이름", ["sma", "rsi", "volume_sma"])
def test_계약3_nullable_dtype에_결측이_들면_터진다(저장소, 이름):
    """🔴 `Int64` 자체는 통과한다. **결측이 들었을 때만** 터진다.

    `_to_array` 가 `None` 만 NaN 으로 바꾸고 `pd.NA` 는 그냥 `float()` 에 넘긴다.
        TypeError: float() argument must be a string or a real number, not 'NAType'

    지금 `supply` 는 numpy dtype 을 주므로 이 경로가 안 열려 있다. 하지만 저장소가
    `astype("Int64")` 로 바뀌는 날 — 결측이 있는 종목(거래정지 구멍)에서 즉시 죽는다.
    그래서 "언젠가 위험" 이 아니라 **이미 걸린 지뢰**로 적어 둔다.
    """
    df = supply.index_series(지수명, as_of="2099-01-01")
    종가 = df["close"].astype("Float64").copy()
    거래량 = df["volume"].astype("Int64").copy()
    종가.iloc[10] = pd.NA
    거래량.iloc[10] = pd.NA

    함수 = {
        "sma": lambda: indicators.sma(종가, window=20),
        "rsi": lambda: indicators.rsi(종가, window=14),
        "volume_sma": lambda: volume.volume_sma(거래량, window=20),
    }[이름]

    결과 = 함수()
    assert np.isnan(np.asarray(결과)).any(), "결측이 NaN 으로 전파돼야 한다"


# ── ⑤ 빈 표 ───────────────────────────────────────────────────────────────

def test_계약5_빈_표에도_칸이_남는다(저장소):
    """행이 0개여도 칸이 있어야 `df["close"]` 가 KeyError 로 터지지 않는다."""
    빈_지수 = supply.index_series(지수명, as_of="1990-01-01")
    빈_종목 = supply.price_series(종목코드, as_of="1990-01-01")

    assert len(빈_지수) == 0 and len(빈_종목) == 0
    for 칸 in ("open", "high", "low", "close", "volume"):
        assert 칸 in 빈_지수.columns, f"빈 지수 표에 {칸} 칸이 없다"
        assert 칸 in 빈_종목.columns, f"빈 종목 표에 {칸} 칸이 없다"


@pytest.mark.parametrize("이름", [n for n in 이름들 if n != "rsi"])
def test_계약5_빈_표를_넣으면_빈_결과가_나온다(저장소, 이름):
    """13개는 지킨다. `rsi` 만 어긋나서 아래 xfail 로 따로 뒀다."""
    호출 = 함수표[이름]
    빈 = supply.index_series(지수명, as_of="1990-01-01")

    for arr in _배열들(호출(_칸(빈))):
        assert len(arr) == 0, f"{이름}: 0행을 넣었는데 {len(arr)}개가 나왔다"


@pytest.mark.xfail(strict=True, reason="이슈 #17 — rsi 가 빈 입력에 길이 1 을 돌려준다")
def test_계약5_빈_표에_rsi도_빈_결과여야_한다(저장소):
    """🔴 `supply` 는 빈 표를 **정상적으로** 낸다 — 상장 전 구간, 아직 모르는 `as_of`.

    그때 다른 13개는 길이 0 을 주는데 `rsi` 만 길이 1(NaN)을 준다. 두 값을 나란히
    `pd.DataFrame` 에 담으면 **길이가 안 맞아 터지거나**, 한쪽이 브로드캐스트되어
    있지도 않은 행이 하나 생긴다.
    """
    빈 = supply.index_series(지수명, as_of="1990-01-01")

    assert len(indicators.rsi(빈["close"], window=14)) == 0


# ── ⑥ 호출 방식 ───────────────────────────────────────────────────────────

def test_계약6_as_of를_위치인자로_넘길_수_없다(저장소):
    """`as_of` 가 키워드 전용이라 **빠뜨릴 수가 없다.** 기본값이 "지금" 이면
    빠뜨린 코드가 조용히 미래를 본다.
    """
    with pytest.raises(TypeError):
        supply.index_series(지수명, "2024-03-01")   # type: ignore[misc]
    with pytest.raises(TypeError):
        supply.price_series(종목코드, "2024-03-01")  # type: ignore[misc]


def test_계약6_as_of를_아예_빼면_터진다(저장소):
    with pytest.raises(TypeError):
        supply.index_series(지수명)                  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        supply.price_series(종목코드)                # type: ignore[call-arg]


@pytest.mark.xfail(strict=True, reason="이슈 #19 — true_range 인자 순서를 검증하지 않는다")
def test_계약6_고가와_저가를_바꿔_넣으면_알아차린다(저장소):
    """🔴 `true_range(high, low, close)` 를 `(low, high, close)` 로 부르면 **조용히**
    다른 값이 나온다. 실측(코스피 200 4,093행) 평균 6.06 → 5.30.

    두 인자 다 `Sequence[float]` 라 타입으로는 못 잡는다. 값으로 잡아야 한다 —
    고가가 저가보다 낮은 표는 시장에 존재하지 않으므로 그건 입력 오류다.
    """
    df = supply.index_series(지수명, as_of="2099-01-01")
    칸 = _칸(df)

    with pytest.raises((ValueError, AssertionError)):
        volatility.true_range(칸["low"], 칸["high"], 칸["close"])


# ── 검사가 진짜로 잡는지 — 버그를 일부러 주입해 본다 ───────────────────────
#
# ⚠️ 새 검사를 만들면 **일부러 틀린 것을 넣어 실제로 잡는지 본다.** 이걸 안 하면
#    조건을 잘못 적어 언제나 통과하는 검사가 초록으로 남는다. 그런 검사는 없느니만
#    못하다 — 지켜지고 있다는 착각을 만든다.


def _계약_길이(arr: Sequence, 입력길이: int) -> bool:
    return len(arr) == 입력길이


def _계약_워밍업(arr: np.ndarray) -> bool:
    """NaN 이 앞쪽 연속 구간에만 있는가.

    🔴 **전부 NaN 을 따로 빼는 이유** — `np.argmax(~nan)` 은 참인 값이 하나도 없으면
    0 을 돌려준다. "0번째가 첫 유효값" 이라는 뜻이 아니라 **찾지 못했다**는 뜻인데,
    그걸 그대로 쓰면 `nan[0:].any()` 가 참이 되어 위반으로 읽는다.

    이 함정은 주입 시험(`test_주입_전부_NaN이면_계약4는_통과한다`)이 잡아냈다.
    전부 NaN 은 "전체가 워밍업" 이라 순서를 접지 않으므로 ④의 위반이 아니다 —
    다만 그것대로 쓸모없는 결과라서 `test_계약4_전부_NaN인_지표는_없다` 가 따로 본다.
    """
    nan = np.isnan(arr)
    if not nan.any() or nan.all():
        return True
    return not nan[int(np.argmax(~nan)):].any()


def test_주입_길이가_어긋나면_계약1이_잡는다():
    assert _계약_길이(np.zeros(60), 60)
    assert not _계약_길이(np.zeros(59), 60), "한 행이 사라졌는데 ① 이 통과했다"
    assert not _계약_길이(np.zeros(61), 60), "한 행이 늘었는데 ① 이 통과했다"


def test_주입_중간에_NaN을_뚫으면_계약4가_잡는다():
    정상 = np.concatenate([np.full(19, np.nan), np.arange(41.0)])
    assert _계약_워밍업(정상), "앞쪽 워밍업만 있는 배열을 ④ 가 잡아 버렸다"

    뚫림 = 정상.copy()
    뚫림[30] = np.nan
    assert not _계약_워밍업(뚫림), "중간에 NaN 을 뚫었는데 ④ 가 통과했다"

    끝 = 정상.copy()
    끝[-1] = np.nan
    assert not _계약_워밍업(끝), "마지막 행에 NaN 을 넣었는데 ④ 가 통과했다"


def test_주입_NaN이_하나도_없어도_계약4는_통과한다():
    """음성 시험 — 워밍업이 없는 함수(`obv`·`true_range`)를 ④ 가 실패시키면 안 된다."""
    assert _계약_워밍업(np.arange(60.0))


def test_주입_전부_NaN이면_계약4는_통과한다():
    """음성 시험 — 전부 NaN 인 배열에는 "첫 유효값 뒤" 가 없다. 여기서 IndexError 가
    나면 검사 자체가 깨진 것이다.
    """
    assert _계약_워밍업(np.full(60, np.nan))


def test_주입_뒤집어도_같은_함수는_계약2가_잡는다(저장소):
    """음성 시험의 반대편 — 순서를 무시하는 가짜 함수를 만들어 ② 가 잡는지 본다."""
    df = supply.index_series(지수명, as_of="2099-01-01")
    칸 = _칸(df)

    def 순서를_무시하는_지표(close: pd.Series) -> np.ndarray:
        # 정렬해 버리면 입력 순서가 결과에 남지 않는다 — 뒤집어도 같은 값이 나온다
        return np.sort(np.asarray(close, dtype=float))

    정 = 순서를_무시하는_지표(칸["close"])
    역 = 순서를_무시하는_지표(칸["close"].iloc[::-1])

    assert np.allclose(정, 역), "시험대가 잘못됐다 — 이 가짜 함수는 순서를 무시해야 한다"
