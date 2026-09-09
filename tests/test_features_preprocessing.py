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
