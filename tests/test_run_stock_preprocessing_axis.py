"""③ 중립화 · ④ 순위 러너(`scripts/run_stock_preprocessing_axis.py`)의 사전 등록을 못박는다.

## 왜 시험이 사전 등록을 지키나

사전 등록은 docstring 에 적혀 있지만, 글은 코드가 바뀌어도 안 바뀐다. 여기서 잡는 것은
**나중에 조용히 어긋날 수 있는 것들**이다.

- 조건 사전이 `winsorize`·`zscore` 를 **명시적으로** 끄는가 — 기본값이 각각 `"mad"` 와
  `True` 라서 키를 빼면 세 단계가 묶여 걸리고, 나빠졌을 때 원인을 못 가른다.
- 판정 벌이 실재하고, `full` 에 빌려 온 B 가 섞이지 않는가 — 섞이면 표본이 조용히
  151,555행으로 줄어 "판정은 full" 이라는 사전 등록이 거짓이 된다.
- 시행 수가 09-09 까지의 7 에 이번 셋을 더한 값인가 — N 이 틀리면 Bonferroni 문턱이
  틀리고, 문턱이 틀리면 판정이 틀린다.

무거운 것(패널 로드 · 12폴드)은 부르지 않는다. 조건 사전과 순수 함수만 본다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features import preprocessing as P
from features.stock_model_dataset import StockModelDataset
from scripts import run_stock_preprocessing_axis as A
from scripts.run_stock_preprocessing_ablation import RAW
from scripts.run_stock_preprocessing_candidates import CONDITIONS as CANDIDATE_CONDITIONS


# ══════════════════════════════════════════════════════════════════════════
# 사전 등록 — 조건 사전
# ══════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("이름", list(A.AXIS_CONDITIONS))
def test_조건은_winsorize_와_zscore_를_명시적으로_끈다(이름):
    options = A.AXIS_CONDITIONS[이름]
    assert options["winsorize"] is None, f"{이름}: 기본값 'mad' 가 조용히 걸린다"
    assert options["zscore"] is False, f"{이름}: 기본값 True 가 조용히 걸린다"


@pytest.mark.parametrize("이름", list(A.AXIS_CONDITIONS))
def test_조건은_중립화와_순위_중_하나만_켠다(이름):
    """한 번에 하나만 바꾼다 — 둘을 함께 켜면 나빠졌을 때 어느 쪽인지 모른다."""
    options = A.AXIS_CONDITIONS[이름]
    켜진것 = [키 for 키 in ("neutralize", "rank") if options.get(키)]
    assert len(켜진것) == 1, f"{이름}: {켜진것}"


@pytest.mark.parametrize("이름", list(A.AXIS_CONDITIONS))
def test_조건은_시계열_표준화를_켜지_않는다(이름):
    """⑤ 는 09-09 에 B 로 이미 쟀다. 다시 켜면 같은 시행을 두 번 세게 된다."""
    assert A.AXIS_CONDITIONS[이름].get("time_series") is None


def test_시총_중립화는_log_cap_을_통제_변수로_쓴다():
    """`market_cap` 은 보호 칸이라 그대로는 못 쓴다."""
    옵션 = A.AXIS_CONDITIONS["F_neutralize_size"]["neutralize"]
    assert 옵션["controls"] == (P.SIZE_CONTROL,)
    assert 옵션["groups"] is None                       # 크기 축만 — 업종은 E 가 본다


def test_업종_중립화는_통제_변수_없이_업종만_본다():
    옵션 = A.AXIS_CONDITIONS["E_neutralize_industry"]["neutralize"]
    assert 옵션["groups"] == "industry"
    assert 옵션["controls"] == ()


def test_순위는_gaussian_하나다():
    """`uniform` 과 `signed` 는 StandardScaler 뒤에서 서로 같아진다
    (`test_features_preprocessing.py::test_uniform_과_signed_는_…`). 시행을 셋 쓸 이유가 없다."""
    순위조건 = [o for o in A.AXIS_CONDITIONS.values() if o.get("rank")]
    assert [o["rank"] for o in 순위조건] == ["gaussian"]


# ══════════════════════════════════════════════════════════════════════════
# 사전 등록 — 표본과 시행 수
# ══════════════════════════════════════════════════════════════════════════
def test_판정_벌은_실재하고_빌려온_조건이_섞이지_않는다():
    assert A.VERDICT_SAMPLE_SET in A.SAMPLE_SETS
    assert A.BORROWED not in A.SAMPLE_SETS[A.VERDICT_SAMPLE_SET]


def test_대조_벌에는_빌려온_B_가_들어_있다():
    """09-09 와 같은 151,555행을 만들려면 B 가 있어야 한다 — B 만이 결측을 만든다."""
    assert A.BORROWED in A.SAMPLE_SETS["common"]
    assert set(A.AXIS_CONDITIONS) <= set(A.SAMPLE_SETS["common"])


def test_빌려온_조건은_09_09_사전에서_그대로_가져온다():
    """여기서 다시 적으면 두 판의 B 가 조용히 갈라진다."""
    assert A.BORROWED in CANDIDATE_CONDITIONS
    assert A._conditions_for("common")[A.BORROWED] is CANDIDATE_CONDITIONS[A.BORROWED]


def test_원값은_전처리를_걸지_않는다():
    for 벌 in A.SAMPLE_SETS:
        assert A._conditions_for(벌)[RAW] is None


def test_시행_수는_09_09_까지의_일곱에_이번_셋을_더한_값이다():
    assert A.TRIALS_BEFORE == 7                          # #210 둘 + A·A′·B·C·D 다섯
    assert len(A.AXIS_CONDITIONS) == 3
    assert 0.05 / (A.TRIALS_BEFORE + len(A.AXIS_CONDITIONS)) == pytest.approx(0.005)


# ══════════════════════════════════════════════════════════════════════════
# 탐색 조건 — 채택 후보와 섞이지 않는가
# ══════════════════════════════════════════════════════════════════════════
def test_탐색_조건은_판정_벌에_섞이지_않는다():
    """섞이면 공통 표본이 함께 줄어 "판정은 full 159,900행" 이 조용히 깨진다."""
    판정벌 = set(A.SAMPLE_SETS[A.VERDICT_SAMPLE_SET])
    assert not (판정벌 & set(A.EXPLORATORY_CONDITIONS))


def test_탐색_조건은_채택_후보와_이름이_겹치지_않는다():
    assert not (set(A.AXIS_CONDITIONS) & set(A.EXPLORATORY_CONDITIONS))


@pytest.mark.parametrize("이름", list(A.EXPLORATORY_CONDITIONS))
def test_탐색_조건도_winsorize_를_명시적으로_끈다(이름):
    assert A.EXPLORATORY_CONDITIONS[이름]["winsorize"] is None


def test_D_는_그날_평균만_뺀다():
    """모든 중립화가 절편 때문에 공통으로 하는 일 하나만 떼어 낸 조건이다."""
    옵션 = A.EXPLORATORY_CONDITIONS["D_center_only"]
    assert 옵션["zscore"] == "center"
    assert "neutralize" not in 옵션 and "rank" not in 옵션


def test_R_과_R2_는_서로_다른_섞은_칸을_쓴다():
    """진짜 업종을 쓰면 대조군이 아니라 E 를 두 번 돌리는 것이 된다. 그리고 둘이 같은
    칸을 쓰면 자유도를 맞춘 대조(R2)와 안 맞춘 대조(R)를 가를 수 없다."""
    R = A.EXPLORATORY_CONDITIONS["R_neutralize_shuffled_industry"]["neutralize"]
    R2 = A.EXPLORATORY_CONDITIONS["R2_neutralize_shuffled_within_day"]["neutralize"]
    assert R["groups"] == A.SHUFFLED_INDUSTRY
    assert R2["groups"] == A.WITHIN_DAY_SHUFFLED_INDUSTRY
    assert len({R["groups"], R2["groups"], "industry"}) == 3


def test_판정은_탐색_조건이_원값을_넘어도_채택하지_않는다():
    """탐색 조건은 Bonferroni 의 N 에 없어 문턱이 보호하지 않는다."""
    보고 = {"paired_t_vs_raw": {
        "D_center_only": {"mean_difference": 0.05, "p_value": 1e-9},
        "E_neutralize_industry": {"mean_difference": -0.01, "p_value": 0.5},
    }}
    판정 = A._verdict(보고, 0.005)
    assert 판정["accepted"] == []
    assert 판정["nominal_best"] == "E_neutralize_industry"   # 후보 중에서만 고른다


# ══════════════════════════════════════════════════════════════════════════
# 순수 함수
# ══════════════════════════════════════════════════════════════════════════
def _업종패널() -> pd.DataFrame:
    return pd.DataFrame({
        "bas_dd": ["d1", "d1", "d1", "d1", "d2", "d2", "d2", "d2"],
        "code": ["a", "b", "c", "d"] * 2,
        "industry": ["금속", "금속", "화학", "건설"] * 2,
        "x": [1.0, 2, 3, 4, 5, 6, 7, 8],
    })


def test_섞은_업종은_종목마다_하나이고_업종별_종목_수가_보존된다():
    ds = StockModelDataset(frame=_업종패널(), feature_columns=("x",))
    frame = A.with_shuffled_industry(ds).frame
    종목별 = frame.groupby("code")[A.SHUFFLED_INDUSTRY].nunique()
    assert (종목별 == 1).all()                                   # 종목 단위로 섞었다
    원래 = sorted(frame.groupby("code")["industry"].first().value_counts().tolist())
    섞음 = sorted(frame.groupby("code")[A.SHUFFLED_INDUSTRY].first().value_counts().tolist())
    assert 원래 == 섞음                                          # 자유도가 같아야 대조군이다


def test_R2_는_그날_업종_분포를_그대로_두어_자유도가_같다():
    """R2 의 존재 이유다. 그날 라벨의 **다중집합**이 같으면 더미 수가 같고, 더미 수가
    같아야 E 와 견줘 업종 정보의 몫을 가를 수 있다. R(종목 단위)은 이것이 안 된다 —
    실측에서 하루 업종이 10 → 16 으로 늘었다."""
    frame = A.with_shuffled_industry(
        StockModelDataset(frame=_업종패널(), feature_columns=("x",))
    ).frame
    for _, sub in frame.groupby("bas_dd"):
        assert sorted(sub["industry"]) == sorted(sub[A.WITHIN_DAY_SHUFFLED_INDUSTRY])


@pytest.mark.parametrize("칸", ["SHUFFLED_INDUSTRY", "WITHIN_DAY_SHUFFLED_INDUSTRY"])
def test_섞은_업종은_시드가_같으면_같다(칸):
    """다시 돌렸을 때 값이 달라지면 대조군 구실을 못 한다."""
    이름 = getattr(A, 칸)
    ds = StockModelDataset(frame=_업종패널(), feature_columns=("x",))
    첫번째 = A.with_shuffled_industry(ds).frame[이름]
    두번째 = A.with_shuffled_industry(ds).frame[이름]
    pd.testing.assert_series_equal(첫번째, 두번째)
    assert A.SHUFFLE_SEED == 20260910


def test_섞은_업종을_붙여도_입력과_피처_목록은_안_바뀐다():
    ds = StockModelDataset(frame=_업종패널(), feature_columns=("x",))
    나온것 = A.with_shuffled_industry(ds)
    assert 나온것.feature_columns == ("x",)
    assert A.SHUFFLED_INDUSTRY not in ds.frame.columns


def test_시총_칸을_붙여도_피처_목록은_안_바뀐다():
    """`log_cap` 은 회귀의 설명변수일 뿐 모델 입력이 아니다."""
    frame = pd.DataFrame({"bas_dd": ["d1"], "x": [1.0], "market_cap": [1e12]})
    ds = StockModelDataset(frame=frame, feature_columns=("x",))
    나온것 = A.with_size_control(ds)
    assert 나온것.feature_columns == ("x",)
    assert P.SIZE_CONTROL in 나온것.frame.columns
    assert P.SIZE_CONTROL not in ds.frame.columns        # 입력은 안 바뀐다


def _연결_패널() -> pd.DataFrame:
    """업종 셋을 hv 순서대로 두고 비중립 비율도 같은 순서로 만든다 → Spearman ρ = 1."""
    행 = []
    for 업종, hv, 비중립수 in (("A", 0.010, 1), ("B", 0.020, 5), ("C", 0.030, 9)):
        for i in range(10):
            행.append({
                "bas_dd": f"d{i}",
                "code": f"{업종}{i:02d}",
                "industry": 업종,
                "hv_20": hv,
                "label_numeric": 1 if i < 비중립수 else 0,
                "market_cap": 1e12 * (i + 1),
            })
    frame = pd.DataFrame(행)
    frame[P.SIZE_CONTROL] = P.log_size_column(frame)
    return frame


def test_축_라벨_연결은_업종_수준과_비중립을_이어_잰다():
    링크 = A.axis_label_link(_연결_패널())
    업종 = 링크["industry_axis_erased_by_E"]["min_rows_0"]
    assert 업종["groups"] == 3
    assert 업종["spearman_rho"] == pytest.approx(1.0)
    assert 업종["nonneutral_min"] == pytest.approx(0.1)
    assert 업종["nonneutral_max"] == pytest.approx(0.9)


def test_축_라벨_연결은_문턱을_고르지_않고_여러_개로_잰다():
    """하나만 적으면 그 값이 유리해서 고른 것인지 알 수 없다."""
    링크 = A.axis_label_link(_연결_패널())
    for 축 in ("stock_axis_erased_by_B", "industry_axis_erased_by_E"):
        assert set(링크[축]) == {f"min_rows_{n}" for n in A.LINK_MIN_ROWS}


def test_그룹이_셋_미만이면_상관을_적지_않고_비운다():
    """못 재면 NULL 로 두고 그룹 수를 함께 남긴다 — 억지로 채우지 않는다."""
    링크 = A.axis_label_link(_연결_패널())
    빈칸 = 링크["industry_axis_erased_by_E"]["min_rows_500"]
    assert 빈칸["spearman_rho"] is None
    assert 빈칸["groups"] == 0
    assert 빈칸["why_empty"] == "groups_below_3"


def test_IC_는_그날_완전_단조면_1_이고_역순이면_마이너스1_이다():
    """05 노트북 §6.1 과 같은 정의인지 손계산으로 확인한다."""
    frame = pd.DataFrame({
        "bas_dd": ["d1"] * 4 + ["d2"] * 4,
        "x": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
        "fwd": [10.0, 20.0, 30.0, 40.0, 40.0, 30.0, 20.0, 10.0],
    })
    표 = A.ic_table(frame[["x"]], frame["fwd"], frame["bas_dd"])
    assert 표.loc["x", "dates"] == 2
    assert 표.loc["x", "ic_mean"] == pytest.approx(0.0)      # (+1 + −1) / 2
    assert 표.loc["x", "icir"] == pytest.approx(0.0)


def test_IC_는_피처_결측_행을_빼되_대상_순위는_전체에서_매긴다():
    """05 노트북 §6.1 정의의 미묘한 자리다 — **대상의 순위와 평균은 결측을 가르기 전에**
    계산된다(`tr = target.groupby(dates).rank()` 가 마스크보다 앞에 있다). 그래서 남은
    행이 완전 단조여도 IC 가 1 이 아니다.

        fwd 순위 [1,2,3,4] · 평균 2.5 (넷 다 센다)
        x 순위   [1,2,3]   · 평균 2   (셋만 센다)
        분자 1.5 + 0 + 0.5 = 2.0 · 분모 √(2 × 2.75) = 2.3452 → 0.8528

    우리 보고서는 **공통 표본이라 피처 결측이 0** 이어서 이 성질에 흔들리지 않는다. 다만
    이 함수를 결측 있는 자료에 그대로 쓰면 값이 달라진다는 것을 못박아 둔다 — 정의를
    고치면 05 노트북의 0.1050 → 0.1144 와 대조가 깨지므로 고치지 않고 적어만 둔다.
    """
    frame = pd.DataFrame({
        "bas_dd": ["d1"] * 4,
        "x": [1.0, 2.0, 3.0, np.nan],
        "fwd": [10.0, 20.0, 30.0, 40.0],
    })
    표 = A.ic_table(frame[["x"]], frame["fwd"], frame["bas_dd"])
    assert 표.loc["x", "ic_mean"] == pytest.approx(2.0 / np.sqrt(2 * 2.75))
    assert 표.loc["x", "ic_mean"] == pytest.approx(0.8528, abs=1e-4)


def test_상수_입력이면_상관을_비우고_왜인지_남긴다():
    """날짜별 hv 평균이 모든 날 같으면 순위상관이 정의되지 않는다 — 경고를 흘리지 않고
    `why_empty` 에 이유를 적는다."""
    링크 = A.axis_label_link(_연결_패널())
    날짜축 = 링크["date_axis_erased_by_cross_sectional_z"]["min_rows_0"]
    assert 날짜축["spearman_rho"] is None
    assert 날짜축["why_empty"] == "constant_input"
