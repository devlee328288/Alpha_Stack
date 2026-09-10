"""횡단면 전처리(`features/preprocessing.py`) — 손계산 · 누수 가드 · 보호 칸 · 트리 불변성.

왜 손계산인가
------------
`test_features_indicators.py` 와 같은 이유다 — 라이브러리 결과를 라이브러리로 검산하면
같은 잘못을 공유한다(2026-09-08 TIL "검사와 대상이 같은 잘못을 공유하면 초록이 나온다").
아래 값은 전부 손으로 풀어 적었다.

    d1: x = [1, 2, 3, 4, 100]        median 3 · MAD 1 · mean 22 · std(ddof=1) 43.617
    d2: x = [10, 20, 30, 40, 50]

왜 누수 가드가 따로 있나
----------------------
횡단면 변환은 "그날 안에서" 라는 약속 하나로 누수를 막는다. 그 약속이 깨지면(전 표본
통계를 쓰면) 검증 구간의 값이 학습 구간의 피처를 바꾼다 — 그런데 에러가 안 난다. 그래서
**다른 날짜의 값을 바꿔도 오늘 결과가 같다** 를 네 함수 모두에 시험으로 못박고, 일부러
틀린 구현(전 표본 z-score)을 같은 시험에 넣어 시험이 실제로 잡는지도 확인한다.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from sklearn.tree import DecisionTreeClassifier

from features import preprocessing as P

D1 = [1.0, 2.0, 3.0, 4.0, 100.0]
D2 = [10.0, 20.0, 30.0, 40.0, 50.0]
STD_D1 = math.sqrt((21**2 + 20**2 + 19**2 + 18**2 + 78**2) / 4)     # 43.6176…


def _panel() -> pd.DataFrame:
    return pd.DataFrame({
        "bas_dd": ["d1"] * 5 + ["d2"] * 5,
        "code": list("abcde") * 2,
        "industry": ["A", "A", "A", "B", "B"] * 2,
        "x": D1 + D2,
        "c": [1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        "label_numeric": [0, 1, 0, 1, 1, 1, 0, 0, 1, 0],
    })


def _d1(out: pd.DataFrame) -> list[float]:
    return out["x"].iloc[:5].tolist()


# ══════════════════════════════════════════════════════════════════════════
# ① winsorize — 손계산
# ══════════════════════════════════════════════════════════════════════════
def test_MAD_3배는_100을_7점4478로_당기고_나머지는_그대로다():
    out = P.winsorize_cross_section(_panel(), ["x"])          # 기본 mad · k=3
    assert _d1(out) == pytest.approx([1, 2, 3, 4, 3 + 3 * 1.4826 * 1])


def test_sigma_방식은_평균_기준이라_극단값에_끌려_덜_당긴다():
    out = P.winsorize_cross_section(_panel(), ["x"], method="sigma", k=1.0)
    assert _d1(out) == pytest.approx([1, 2, 3, 4, 22 + STD_D1])


def test_분위_방식은_pandas_선형보간_분위로_자른다():
    out = P.winsorize_cross_section(_panel(), ["x"], method="quantile", limits=(0.2, 0.8))
    assert _d1(out) == pytest.approx([1.8, 2, 3, 4, 23.2])


def test_행은_하나도_사라지지_않는다():
    df = _panel()
    out = P.winsorize_cross_section(df, ["x"])
    assert len(out) == len(df)
    assert out.index.equals(df.index)


def test_MAD_가_0_이면_그날은_손대지_않는다():
    df = _panel()
    df.loc[df["bas_dd"] == "d1", "x"] = [5.0, 5.0, 5.0, 5.0, 9.0]
    out = P.winsorize_cross_section(df, ["x"])
    assert _d1(out) == [5, 5, 5, 5, 9]


def test_종목이_min_count_보다_적은_날은_손대지_않는다():
    df = _panel().iloc[[0, 4, 5, 6, 7, 8, 9]]     # d1 은 [1, 100] 둘뿐
    out = P.winsorize_cross_section(df, ["x"])
    assert out["x"].iloc[:2].tolist() == [1, 100]


# ══════════════════════════════════════════════════════════════════════════
# ② z-score — 손계산
# ══════════════════════════════════════════════════════════════════════════
def test_z_score_는_그날_평균과_ddof1_표준편차다():
    out = P.zscore_cross_section(_panel(), ["x"])
    assert _d1(out) == pytest.approx([(v - 22) / STD_D1 for v in D1])
    assert out["x"].iloc[5:].tolist() == pytest.approx(
        [(v - 30) / math.sqrt(1000 / 4) for v in D2])


def test_robust_z_는_중앙값과_MAD_다():
    out = P.zscore_cross_section(_panel(), ["x"], robust=True)
    assert _d1(out) == pytest.approx([(v - 3) / 1.4826 for v in D1])


def test_중심화만_하면_그날_평균만_빼고_나누지_않는다():
    """d1 평균 22 → [1−22, 2−22, 3−22, 4−22, 100−22]. `scale=False` 의 손계산이다."""
    out = P.zscore_cross_section(_panel(), ["x"], scale=False)
    assert _d1(out) == pytest.approx([-21, -20, -19, -18, 78])


def test_중심화만_하면_산포가_0_인_날도_살아_남는다():
    """나누지 않으므로 0 으로 나눌 일이 없다 — 그날은 전부 0 이 된다."""
    df = _panel()
    df.loc[df["bas_dd"] == "d1", "x"] = 7.0
    out = P.zscore_cross_section(df, ["x"], scale=False)
    assert _d1(out) == pytest.approx([0, 0, 0, 0, 0])
    assert P.zscore_cross_section(df, ["x"])["x"].iloc[:5].isna().all()   # z 는 NaN


def test_중심화만_한_결과의_날짜별_평균은_0_이다():
    """중립화가 절편 때문에 함께 하는 일이 바로 이것이다 — 그래서 따로 잰다."""
    df = _panel()
    out = P.zscore_cross_section(df, ["x", "c"], scale=False)
    assert out.groupby(df["bas_dd"]).mean().abs().to_numpy().max() < 1e-12


def test_전처리_한번에서_zscore_center_는_중심화만_한다():
    df = _panel()
    직접 = P.zscore_cross_section(df, ["x"], scale=False)
    한번에 = P.preprocess_cross_section(df, ["x"], winsorize=None, zscore="center")
    pd.testing.assert_frame_equal(직접, 한번에)


def test_모르는_zscore_값은_거부한다():
    with pytest.raises(ValueError, match="True · False · 'center'"):
        P.preprocess_cross_section(_panel(), ["x"], zscore="nope")


def test_산포가_0_이면_z_는_NaN_이다():
    df = _panel()
    df.loc[df["bas_dd"] == "d1", "x"] = 7.0
    out = P.zscore_cross_section(df, ["x"])
    assert out["x"].iloc[:5].isna().all()
    assert out["x"].iloc[5:].notna().all()


# ══════════════════════════════════════════════════════════════════════════
# ③ neutralize — 손계산
# ══════════════════════════════════════════════════════════════════════════
def test_업종만_주면_업종_안_평균을_뺀_것과_같다():
    """A = [1, 2, 3] 평균 2 · B = [4, 100] 평균 52 — Kakushadze `indneutralize`."""
    out = P.neutralize_cross_section(_panel(), ["x"], groups="industry")
    assert _d1(out) == pytest.approx([-1, 0, 1, -48, 48])


def test_연속_통제만_주면_절편_있는_단순회귀_잔차다():
    """d2: x = 60 − 10c 로 정확히 선형이라 잔차가 0 이어야 한다."""
    out = P.neutralize_cross_section(_panel(), ["x"], groups=None, controls=("c",))
    assert out["x"].iloc[5:].tolist() == pytest.approx([0, 0, 0, 0, 0], abs=1e-9)


def test_업종과_통제를_같이_주면_둘_다_회귀한다():
    df = _panel()
    df["c"] = [1.0, 2.0, 3.0, 1.0, 2.0, 1.0, 2.0, 3.0, 1.0, 2.0]
    df.loc[df["bas_dd"] == "d2", "x"] = [11.0, 12.0, 13.0, 21.0, 22.0]   # 업종 절편 + c
    out = P.neutralize_cross_section(df, ["x"], groups="industry", controls=("c",))
    assert out["x"].iloc[5:].tolist() == pytest.approx([0, 0, 0, 0, 0], abs=1e-9)


def test_자유도가_없는_날은_NaN_이다():
    df = _panel().iloc[[0, 1, 2, 5, 6, 7, 8, 9]]     # d1 은 A 셋뿐 → 더미 1개 + 여유 2 = 3 필요
    out = P.neutralize_cross_section(df, ["x"], groups="industry", controls=("c",))
    assert out["x"].iloc[:3].isna().all()            # 매개변수 2 (A 더미 + c) → 4 필요
    assert out["x"].iloc[3:].notna().all()


@pytest.mark.parametrize(
    ("groups", "controls"),
    [("industry", ()), (None, ("c",)), ("industry", ("c",))],
)
def test_어느_축을_지우든_그날_평균이_함께_0_이_된다(groups, controls):
    """설계행렬에 늘 절편(또는 더미 전부)이 있어서다 — 절편이 있는 회귀의 잔차는 평균이 0.

    2026-09-10 에 이것을 모르고 "시총 중립화" 를 음성 대조군으로 설계했다가, 시총이
    라벨과 무관(r=−0.0086)한데도 12폴드 정확도가 −1.94%p 나빠지는 것을 보고 알았다.
    이름이 연산의 절반만 말하고 있었다.
    """
    df = _panel()
    out = P.neutralize_cross_section(df, ["x"], groups=groups, controls=controls)
    assert out.groupby(df["bas_dd"]).mean().abs().to_numpy().max() < 1e-12


def test_중립화는_평균만_지우고_산포는_남긴다():
    """횡단면 z·순위와 갈리는 자리다 — 그 둘은 날짜별 산포까지 1 로 만든다."""
    df = _panel()
    중립 = P.neutralize_cross_section(df, ["x"], groups="industry")
    z = P.zscore_cross_section(df, ["x"])
    날짜별 = 중립["x"].groupby(df["bas_dd"]).std()
    assert 날짜별.max() / 날짜별.min() > 2                    # 날마다 산포가 다르다
    z날짜별 = z["x"].groupby(df["bas_dd"]).std()
    assert z날짜별.max() / z날짜별.min() == pytest.approx(1.0)


def test_통제_변수를_자기_자신에_회귀하면_거부한다():
    with pytest.raises(ValueError, match="자기 자신"):
        P.neutralize_cross_section(_panel(), ["x", "c"], groups=None, controls=("c",))


def test_축이_하나도_없으면_거부한다():
    with pytest.raises(ValueError, match="중립화 축"):
        P.neutralize_cross_section(_panel(), ["x"], groups=None, controls=())


# ══════════════════════════════════════════════════════════════════════════
# ④ rank — 손계산
# ══════════════════════════════════════════════════════════════════════════
def test_uniform_순위는_끝점을_피한_백분위다():
    out = P.rank_cross_section(_panel(), ["x"], method="uniform")
    assert _d1(out) == pytest.approx([0.1, 0.3, 0.5, 0.7, 0.9])


def test_gaussian_순위는_uniform_의_정규_역함수다():
    out = P.rank_cross_section(_panel(), ["x"])                 # 기본 gaussian
    assert _d1(out) == pytest.approx(norm.ppf([0.1, 0.3, 0.5, 0.7, 0.9]).tolist())
    assert abs(sum(_d1(out))) < 1e-12                           # 대칭


def test_signed_순위는_GKX_의_마이너스1에서_1_사이다():
    out = P.rank_cross_section(_panel(), ["x"], method="signed")
    assert _d1(out) == pytest.approx([2 * r / 6 - 1 for r in (1, 2, 3, 4, 5)])


def test_동률은_평균_순위다():
    df = _panel()
    df.loc[df["bas_dd"] == "d1", "x"] = [1.0, 2.0, 2.0, 4.0, 5.0]
    out = P.rank_cross_section(df, ["x"], method="uniform")
    assert _d1(out) == pytest.approx([0.1, 0.4, 0.4, 0.7, 0.9])


# ══════════════════════════════════════════════════════════════════════════
# 결측 — 채우지 않고 통계에서 뺀다
# ══════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("fn", [
    P.winsorize_cross_section, P.zscore_cross_section, P.rank_cross_section,
    lambda f, c: P.neutralize_cross_section(f, c, groups="industry"),
])
def test_결측은_결측으로_남고_나머지_통계는_결측_없이_계산된다(fn):
    df = _panel()
    df.loc[4, "x"] = np.nan                                   # d1 의 100 을 지운다
    out = fn(df, ["x"])
    assert math.isnan(out["x"].iloc[4])
    assert out["x"].iloc[:4].notna().all()
    # 100 이 빠졌으니 d1 의 통계는 [1, 2, 3, 4] 만으로 — z 라면 평균 2.5
    if fn is P.zscore_cross_section:
        assert out["x"].iloc[0] == pytest.approx((1 - 2.5) / math.sqrt(5 / 3))


# ══════════════════════════════════════════════════════════════════════════
# 누수 가드 — 다른 날짜는 오늘을 바꿀 수 없다
# ══════════════════════════════════════════════════════════════════════════
네_함수 = {
    "winsorize": lambda f: P.winsorize_cross_section(f, ["x"]),
    "zscore": lambda f: P.zscore_cross_section(f, ["x"]),
    "neutralize": lambda f: P.neutralize_cross_section(
        f, ["x"], groups="industry", controls=("c",)),
    "rank": lambda f: P.rank_cross_section(f, ["x"]),
}


@pytest.mark.parametrize("이름", list(네_함수))
def test_다른_날짜의_값을_바꿔도_오늘_결과는_같다(이름):
    fn = 네_함수[이름]
    before = fn(_panel())
    df = _panel()
    df.loc[df["bas_dd"] == "d2", "x"] = [1e6, -1e6, 0.0, 3.0, 7.0]   # d2 를 마구 바꾼다
    after = fn(df)
    pd.testing.assert_series_equal(before["x"].iloc[:5], after["x"].iloc[:5])


@pytest.mark.parametrize("이름", list(네_함수))
def test_미래_날짜를_덧붙여도_과거_결과는_같다(이름):
    fn = 네_함수[이름]
    before = fn(_panel())
    df = _panel()
    future = df[df["bas_dd"] == "d2"].copy()
    future["bas_dd"] = "d3"
    future["x"] = [999.0, 1.0, 2.0, 3.0, -50.0]
    after = fn(pd.concat([df, future], ignore_index=True))
    pd.testing.assert_series_equal(before["x"], after["x"].iloc[:10])


@pytest.mark.parametrize("이름", list(네_함수))
def test_행_순서를_섞어도_같은_행은_같은_값이다(이름):
    fn = 네_함수[이름]
    df = _panel()
    before = fn(df)
    shuffled = df.sample(frac=1.0, random_state=7)
    after = fn(shuffled)
    pd.testing.assert_series_equal(before["x"].loc[shuffled.index], after["x"])


def test_주입_전_표본_z_score_는_누수_시험이_잡는다():
    """일부러 틀린 구현 — 전 표본 평균·표준편차. 위 시험이 실제로 붉어지는지 본다."""
    def 틀린_z(f: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"x": (f["x"] - f["x"].mean()) / f["x"].std()}, index=f.index)

    before = 틀린_z(_panel())
    df = _panel()
    df.loc[df["bas_dd"] == "d2", "x"] = [1e6, -1e6, 0.0, 3.0, 7.0]
    after = 틀린_z(df)
    with pytest.raises(AssertionError):
        pd.testing.assert_series_equal(before["x"].iloc[:5], after["x"].iloc[:5])


def test_입력_표는_바뀌지_않는다():
    df = _panel()
    snapshot = df.copy()
    P.preprocess_cross_section(df, ["x"], neutralize={"groups": "industry"}, rank="gaussian")
    pd.testing.assert_frame_equal(df, snapshot)


# ══════════════════════════════════════════════════════════════════════════
# 보호 칸 — 라벨·수익률·가격·플래그는 받지 않는다
# ══════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("칸", ["label_numeric", "fwd_return_5d", "adj_close", "close",
                              "is_adj_suspect", "entry_adj_open", "market_cap"])
def test_보호_칸은_거부한다(칸):
    df = _panel()
    df[칸] = 1.0
    with pytest.raises(ValueError, match="전처리 대상이 아닌"):
        P.zscore_cross_section(df, [칸])


def test_없는_칸과_문자_칸은_거부한다():
    with pytest.raises(ValueError, match="표에 없습니다"):
        P.zscore_cross_section(_panel(), ["없음"])
    with pytest.raises(ValueError, match="숫자 칸만"):
        P.zscore_cross_section(_panel(), ["industry"])


def test_transformed_columns_는_보호_칸을_걸러_준다():
    cols = ["ret_5", "label_numeric", "adj_close", "hv_20", "is_halted", "close", "ret_5"]
    assert P.transformed_columns(cols) == ["ret_5", "hv_20"]


# ══════════════════════════════════════════════════════════════════════════
# 트리 — 전 표본 단조 변환은 무의미, 횡단면 변환은 뜻을 바꾼다
# ══════════════════════════════════════════════════════════════════════════
def test_전_표본_단조_변환은_트리에_무의미하지만_횡단면_변환은_다르다():
    """날짜마다 수준이 다른 피처. 라벨은 '그날 상위 절반인가'.

    깊이 1 그루터기는 원값·로그·전체 스케일링 어느 것으로도 같은 답을 내고(전 표본 단조
    변환 = 같은 분할), 날짜별 순위로 바꾸면 비로소 0.5 에서 갈라 전부 맞힌다.
    """
    rng = np.random.default_rng(0)
    rows = []
    for d, level in (("d1", 0.0), ("d2", 10.0), ("d3", 20.0)):
        x = level + rng.uniform(0, 1, 20)
        top = x >= np.median(x)
        rows.append(pd.DataFrame({"bas_dd": d, "x": x, "y": top.astype(int)}))
    df = pd.concat(rows, ignore_index=True)

    def 예측(x: pd.Series) -> np.ndarray:
        stump = DecisionTreeClassifier(max_depth=1, random_state=0)
        stump.fit(x.to_numpy()[:, None], df["y"])
        return stump.predict(x.to_numpy()[:, None])

    raw = 예측(df["x"])
    assert (예측(np.log1p(df["x"])) == raw).all()                  # 전 표본 로그
    assert (예측(df["x"] * 3.0 + 7.0) == raw).all()                 # 전 표본 스케일링
    assert (raw == df["y"]).mean() < 0.8                           # 수준이 섞여 못 가른다

    ranked = P.rank_cross_section(df, ["x"], method="uniform")["x"]
    assert (예측(ranked) == df["y"]).all()                          # 그날 안 상대 위치로 갈린다


# ══════════════════════════════════════════════════════════════════════════
# ⑤ 시계열 표준화 — 종목 안에서 시점을 비교한다 (이슈 #214 후보 B)
# ══════════════════════════════════════════════════════════════════════════
#: 종목 둘 × 여섯 날. a 는 1..6 으로 오르고, b 는 그 열 배다.
#: 창 3 · 준비 2 로 잡으면 손계산이 짧게 떨어진다.
#:
#:   a 의 2행째(x=2): 창 [1,2]   · mean 1.5 · std = sqrt(0.5)  → z = 0.5/sqrt(0.5) = sqrt(0.5)
#:   a 의 3행째(x=3): 창 [1,2,3] · mean 2   · std 1                    → z = +1
#:   a 의 4행째(x=4): 창 [2,3,4] · mean 3   · std 1                    → z = +1
#:   a 의 1행째:      창 [1] · 준비 2 미만                              → NaN
_TS_기대_a = [math.nan, math.sqrt(0.5), 1.0, 1.0, 1.0, 1.0]


def _시계열_패널() -> pd.DataFrame:
    """종목이 섞여 들어오고 행 순서도 뒤죽박죽인 표. 함수가 스스로 정렬해야 한다."""
    행 = []
    for i, d in enumerate(["d1", "d2", "d3", "d4", "d5", "d6"], start=1):
        행.append({"bas_dd": d, "code": "a", "x": float(i), "industry": "A"})
        행.append({"bas_dd": d, "code": "b", "x": float(i) * 10.0, "industry": "B"})
    return pd.DataFrame(행).sample(frac=1.0, random_state=3).reset_index(drop=True)


def _날짜순(out: pd.DataFrame, df: pd.DataFrame, code: str) -> np.ndarray:
    """한 종목의 결과를 날짜 오름차순으로 편다."""
    골라 = df["code"] == code
    값 = out.loc[골라, "x"].to_numpy()
    return 값[np.argsort(df.loc[골라, "bas_dd"].to_numpy())]


def test_시계열_z_는_그_종목의_최근_창_평균과_ddof1_표준편차다():
    df = _시계열_패널()
    out = P.standardize_time_series(df, ["x"], window=3, min_periods=2)
    a = _날짜순(out, df, "a")
    assert math.isnan(a[0])
    assert a[1:].tolist() == pytest.approx(_TS_기대_a[1:])


def test_시계열_z_는_종목마다_따로_센다():
    """b 는 a 의 열 배지만 z 는 같다 — 종목 고정효과(수준·배율)가 지워진다."""
    df = _시계열_패널()
    out = P.standardize_time_series(df, ["x"], window=3, min_periods=2)
    np.testing.assert_allclose(_날짜순(out, df, "a")[1:], _날짜순(out, df, "b")[1:])


def test_준비구간은_NaN_이고_min_periods_가_그_길이를_정한다():
    df = _시계열_패널()
    긴준비 = P.standardize_time_series(df, ["x"], window=6, min_periods=4)
    a = _날짜순(긴준비, df, "a")
    assert np.isnan(a[:3]).all() and np.isfinite(a[3:]).all()


# ── 🔴 인과성 — 이 세 시험이 후보 B 의 존재 근거다 ──────────────────────────
def test_뒤_행을_잘라도_앞_행의_z_는_같다():
    """t+1 이후를 한 줄도 안 본다. 전 구간 통계를 쓰는 구현이면 여기서 깨진다."""
    df = _시계열_패널().sort_values(["code", "bas_dd"], kind="mergesort").reset_index(drop=True)
    전체 = P.standardize_time_series(df, ["x"], window=3, min_periods=2)
    앞부분 = df[df["bas_dd"] <= "d4"]
    잘라서 = P.standardize_time_series(앞부분, ["x"], window=3, min_periods=2)
    pd.testing.assert_frame_equal(전체.loc[잘라서.index], 잘라서)


def test_미래_날짜를_덧붙여도_과거_z_는_같다():
    df = _시계열_패널()
    before = P.standardize_time_series(df, ["x"], window=3, min_periods=2)
    미래 = pd.DataFrame([
        {"bas_dd": "d7", "code": "a", "x": 1e6, "industry": "A"},
        {"bas_dd": "d7", "code": "b", "x": -1e6, "industry": "B"},
    ])
    after = P.standardize_time_series(
        pd.concat([df, 미래], ignore_index=True), ["x"], window=3, min_periods=2
    )
    pd.testing.assert_frame_equal(before, after.iloc[: len(df)])


def test_전_구간_표준화를_넣으면_인과성_시험이_잡는다():
    """일부러 틀린 구현 — 시험이 실제로 무엇을 막는지 확인한다."""

    def 전_구간_z(f: pd.DataFrame) -> pd.DataFrame:
        g = f.groupby("code")["x"]
        return ((f["x"] - g.transform("mean")) / g.transform("std")).to_frame("x")

    df = _시계열_패널().sort_values(["code", "bas_dd"], kind="mergesort").reset_index(drop=True)
    전체 = 전_구간_z(df)
    앞부분 = df[df["bas_dd"] <= "d4"]
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(전체.loc[앞부분.index], 전_구간_z(앞부분))


def test_행_순서를_섞어도_시계열_z_는_같다():
    df = _시계열_패널()
    before = P.standardize_time_series(df, ["x"], window=3, min_periods=2)
    섞 = df.sample(frac=1.0, random_state=11)
    after = P.standardize_time_series(섞, ["x"], window=3, min_periods=2)
    pd.testing.assert_frame_equal(before.loc[섞.index], after)


def test_같은_종목_같은_날이_두_번이면_거부한다():
    df = pd.concat([_시계열_패널(), _시계열_패널().iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="두 번 이상"):
        P.standardize_time_series(df, ["x"], window=3, min_periods=2)


def test_시계열_표준화도_보호_칸을_거부한다():
    df = _시계열_패널()
    df["label_numeric"] = 0
    with pytest.raises(ValueError, match="전처리 대상이 아닌"):
        P.standardize_time_series(df, ["label_numeric"])


def test_산포가_0_인_구간은_NaN_이다():
    df = pd.DataFrame({
        "bas_dd": ["d1", "d2", "d3", "d4"],
        "code": ["a"] * 4,
        "x": [5.0, 5.0, 5.0, 9.0],
    })
    out = P.standardize_time_series(df, ["x"], window=3, min_periods=2)
    assert math.isnan(out["x"].iloc[1]) and math.isnan(out["x"].iloc[2])
    assert np.isfinite(out["x"].iloc[3])


# ══════════════════════════════════════════════════════════════════════════
# ⑥ 날짜 수준 복원 (이슈 #214 후보 C)
# ══════════════════════════════════════════════════════════════════════════
def test_복원한_평균과_표준편차는_그날_전체에_같은_값이다():
    out = P.restore_date_level(_panel(), ["x"])
    assert list(out.columns) == ["cs_mean_x", "cs_std_x"]
    assert out["cs_mean_x"].iloc[:5].nunique() == 1
    assert out["cs_mean_x"].iloc[0] == pytest.approx(22.0)          # d1 평균 손계산
    assert out["cs_std_x"].iloc[0] == pytest.approx(STD_D1)
    assert out["cs_mean_x"].iloc[5] == pytest.approx(30.0)          # d2 평균


def test_복원_칸_이름을_미리_알_수_있다():
    assert P.date_level_columns(["hv_20", "atr_ratio"]) == [
        "cs_mean_hv_20", "cs_std_hv_20", "cs_mean_atr_ratio", "cs_std_atr_ratio",
    ]
    assert P.date_level_columns(["x"], ("median",)) == ["cs_median_x"]


def test_복원은_원래_칸을_돌려주지_않는다():
    """①~④ 와 규약이 다르다 — 새 칸만 준다. 부르는 쪽이 붙인다."""
    out = P.restore_date_level(_panel(), ["x"])
    assert "x" not in out.columns


def test_날짜_수준_복원도_다른_날짜의_값에_흔들리지_않는다():
    before = P.restore_date_level(_panel(), ["x"])
    df = _panel()
    df.loc[df["bas_dd"] == "d2", "x"] = [1e6, -1e6, 0.0, 3.0, 7.0]
    after = P.restore_date_level(df, ["x"])
    pd.testing.assert_frame_equal(before.iloc[:5], after.iloc[:5])


def test_종목이_min_count_보다_적은_날은_복원값이_NaN_이다():
    df = pd.DataFrame({"bas_dd": ["d1", "d1", "d2"], "code": list("abc"),
                       "x": [1.0, 2.0, 3.0]})
    out = P.restore_date_level(df, ["x"], min_count=3)
    assert out["cs_mean_x"].isna().all()


def test_모르는_통계는_거부한다():
    with pytest.raises(ValueError, match="모르는 날짜 수준 통계"):
        P.restore_date_level(_panel(), ["x"], stats=("mode",))


# ══════════════════════════════════════════════════════════════════════════
# 축 가르기 · keep_raw (이슈 #214 후보 D)
# ══════════════════════════════════════════════════════════════════════════
def test_축을_가르면_입력_순서를_지킨다():
    처리, 원값 = P.split_by_axis(["rsi_14", "hv_20", "sma_gap_5_20", "atr_ratio"])
    assert 처리 == ["rsi_14", "sma_gap_5_20"]
    assert 원값 == ["hv_20", "atr_ratio"]


def test_없는_칸을_보존_목록에_넣어도_멈추지_않는다():
    """조합마다 칸이 다르므로, 없는 이름은 조용히 무시한다."""
    처리, 원값 = P.split_by_axis(["rsi_14"], keep_raw=("hv_20", "bb_bandwidth"))
    assert 처리 == ["rsi_14"] and 원값 == []


def test_keep_raw_칸은_한_자리도_안_바뀐다():
    df = _panel()
    out = P.preprocess_cross_section(df, ["x", "c"], keep_raw=("x",))
    pd.testing.assert_series_equal(out["x"], df["x"])
    assert not np.allclose(out["c"], df["c"])                        # 나머지는 처리된다
    assert list(out.columns) == ["x", "c"]                           # 칸 순서는 요청 그대로


def test_보존_목록의_기본값은_변동성_축이다():
    assert P.VOLATILITY_AXIS == ("hv_20", "atr_ratio", "hv_regime", "bb_bandwidth")


def test_전처리_한번에가_시계열_경로도_같은_값을_낸다():
    df = _시계열_패널()
    직접 = P.standardize_time_series(df, ["x"], window=3, min_periods=2)
    한번에 = P.preprocess_cross_section(
        df, ["x"], winsorize=None, zscore=False,
        time_series=dict(window=3, min_periods=2),
    )
    pd.testing.assert_frame_equal(직접, 한번에)


def test_전처리_한번에가_중립화_경로도_같은_값을_낸다():
    df = _panel()
    직접 = P.neutralize_cross_section(df, ["x"], groups="industry")
    한번에 = P.preprocess_cross_section(
        df, ["x"], winsorize=None, zscore=False,
        neutralize=dict(groups="industry", controls=()),
    )
    pd.testing.assert_frame_equal(직접, 한번에)


# ══════════════════════════════════════════════════════════════════════════
# log 시총 — ③ 중립화의 크기 통제 변수
# ══════════════════════════════════════════════════════════════════════════
def test_log_시총은_자연로그다():
    df = pd.DataFrame({"market_cap": [1.0, math.e, math.e**2, math.e**10]})
    assert P.log_size_column(df).tolist() == pytest.approx([0, 1, 2, 10])


def test_log_시총은_0_이하와_결측을_채우지_않고_결측으로_둔다():
    """`clip(lower=1)` 로 막으면 log 가 0 이 되어 **가장 작은 회사**로 보인다 — 오류를
    그럴듯한 값으로 바꾸는 것이라 하지 않는다. 중립화가 그 행을 회귀에서 뺀다."""
    df = pd.DataFrame({"market_cap": [1e12, 0.0, -3.0, None]})
    out = P.log_size_column(df)
    assert out.iloc[0] == pytest.approx(math.log(1e12))
    assert out.iloc[1:].isna().all()


def test_log_시총은_칸_이름을_SIZE_CONTROL_로_준다():
    out = P.log_size_column(pd.DataFrame({"market_cap": [1e12]}))
    assert out.name == P.SIZE_CONTROL == "log_cap"


def test_시총_칸이_없으면_무엇을_해야_하는지_말하며_거부한다():
    with pytest.raises(ValueError, match="붙이고 다시"):
        P.log_size_column(pd.DataFrame({"code": ["005930"]}))


def test_시총_자체는_보호_칸이라_피처로_못_쓴다():
    """`market_cap` 을 그냥 통제 변수로 쓰지 않고 `log_cap` 을 따로 만드는 이유."""
    assert "market_cap" in P.PROTECTED_NAMES
    assert P.SIZE_CONTROL not in P.PROTECTED_NAMES


def test_log_는_순서를_바꾸지_않는다():
    """중립화 잔차를 읽을 때 "큰 회사가 큰 값" 이라는 방향이 유지된다는 뜻이다."""
    df = pd.DataFrame({"market_cap": [1.8e11, 8.5e12, 5.4e14]})
    out = P.log_size_column(df)
    assert out.is_monotonic_increasing


# ══════════════════════════════════════════════════════════════════════════
# ④ 순위 — 세 방식 중 둘은 왜 별도 시행이 아닌가
# ══════════════════════════════════════════════════════════════════════════
def _한날(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "bas_dd": ["d1"] * n,
        "x": [float(i) for i in range(n)],
    })


def _표준화(s: pd.Series) -> np.ndarray:
    v = s.to_numpy(dtype=float)
    return (v - v.mean()) / v.std(ddof=0)


def test_uniform_과_signed_는_같은_날_안에서는_표준화하면_같아진다():
    """모델(`build_logistic_baseline`)이 `StandardScaler` 를 앞에 둔다는 사실의 결과다.

    n 이 같으면 `uniform=(r−0.5)/n` 과 `signed=2r/(n+1)−1` 은 서로 **선형변환**이고,
    선형변환은 표준화가 지운다. 그래서 순위 세 방식 중 이 둘에 시행을 따로 쓸 이유가
    없다 — 2026-09-10 판이 `gaussian` 하나만 돌린 근거다.
    """
    df = _한날(5)
    u = P.rank_cross_section(df, ["x"], method="uniform")["x"]
    s = P.rank_cross_section(df, ["x"], method="signed")["x"]
    assert u.tolist() == pytest.approx([0.1, 0.3, 0.5, 0.7, 0.9])
    assert s.tolist() == pytest.approx([-2 / 3, -1 / 3, 0, 1 / 3, 2 / 3])
    assert _표준화(u) == pytest.approx(_표준화(s))


def test_gaussian_은_비선형이라_표준화해도_두_방식과_다르다():
    df = _한날(5)
    u = P.rank_cross_section(df, ["x"], method="uniform")["x"]
    g = P.rank_cross_section(df, ["x"], method="gaussian")["x"]
    assert _표준화(g) != pytest.approx(_표준화(u))
    assert g.tolist() == pytest.approx([norm.ppf(v) for v in (0.1, 0.3, 0.5, 0.7, 0.9)])


def test_날짜마다_종목_수가_다르면_uniform_과_signed_도_어긋난다():
    """"같아진다" 는 n 이 같을 때다. 우리 패널은 하루 43~50종으로 날마다 다르므로
    두 방식이 정확히 같지는 않다 — "거의" 를 "정확히" 로 적지 않으려고 함께 못박는다."""
    df = pd.DataFrame({
        "bas_dd": ["d1"] * 5 + ["d2"] * 3,
        "x": [0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0],
    })
    u = P.rank_cross_section(df, ["x"], method="uniform")["x"]
    s = P.rank_cross_section(df, ["x"], method="signed")["x"]
    assert _표준화(u) != pytest.approx(_표준화(s))


def test_순위_앞의_z_score_는_결과를_바꾸지_않는다():
    """z 는 그날 안 순서를 보존하는 단조 변환이라 순위가 그대로다."""
    df = _panel()
    순위만 = P.preprocess_cross_section(
        df, ["x"], winsorize=None, zscore=False, rank="gaussian")
    z_뒤순위 = P.preprocess_cross_section(
        df, ["x"], winsorize=None, zscore=True, rank="gaussian")
    pd.testing.assert_frame_equal(순위만, z_뒤순위)


def test_순위_앞의_winsorize_는_동률을_만들어_결과를_바꾼다():
    """"순위를 켜면 앞 단계가 무의미하다" 는 z 에만 맞다. winsorize 는 극단 둘 이상을
    같은 경계로 당겨 **동률**을 만들고, 동률은 평균 순위가 되어 값이 달라진다.
    2026-09-10 판이 조건 사전에서 `winsorize=None` 을 명시적으로 적은 이유다."""
    df = pd.DataFrame({
        "bas_dd": ["d1"] * 5,
        "x": [1.0, 2.0, 3.0, 100.0, 200.0],          # median 3 · MAD 1 → 상한 7.4478
    })
    순위만 = P.preprocess_cross_section(
        df, ["x"], winsorize=None, zscore=False, rank="gaussian")["x"]
    winsor_뒤 = P.preprocess_cross_section(
        df, ["x"], winsorize="mad", zscore=False, rank="gaussian")["x"]
    assert 순위만.iloc[3] != pytest.approx(순위만.iloc[4])
    assert winsor_뒤.iloc[3] == pytest.approx(winsor_뒤.iloc[4])     # 동률 → 평균 순위
