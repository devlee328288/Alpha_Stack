"""횡단면 전처리 — 날짜 안에서 끝나는 네 단계 (조립 층 · pandas).

    ① winsorize    극단값을 **그날 횡단면의 산포** 기준으로 당긴다   (기본 MAD 3배)
    ② zscore       그날 평균·표준편차로 표준화한다
    ③ neutralize   업종 더미·log 시총 같은 축에 회귀해 **잔차만** 남긴다
    ④ rank         그날 순위로 바꾼다                                 (uniform · gaussian · signed)

네 함수 모두 같은 규약을 지킨다.

- **입력과 같은 인덱스, 요청한 칸만** 담은 새 표를 돌려준다. 입력은 바꾸지 않는다.
- 계산은 `date_col` 로 묶은 **그날 안에서만** 한다. 다른 날짜의 값이 오늘 결과를 바꾸는
  일은 없다 — 시험 `test_다른_날짜의_값을_바꿔도_오늘_결과는_같다` 가 못박는다.
- 결측은 결측으로 남긴다. 채우지 않고, 그날의 통계에서 뺀다.
- **라벨·수익률·가격·플래그는 받지 않는다.** `PROTECTED_PREFIXES` 에 걸리면 거부한다.

왜 "날짜 안에서" 인가 — 누수
----------------------------
`StandardScaler().fit(X_all)` 처럼 전 구간 통계를 쓰면 검증 구간의 평균·분산이 학습에
흘러든다(`features/__init__.py` 경고 2). 횡단면 변환은 다르다 — **그날 종목들끼리의**
평균·순위라서 미래를 볼 수 없고, 그래서 학습·검증을 가르기 전에 패널 조립 단계에서
해도 된다. 시계열 방향 스케일링(폴드 안 `StandardScaler`)은 그대로 `models/` 의
Pipeline 이 맡는다. 두 방향을 섞지 않는다.

트리 모델에 무엇이 무의미하고 무엇이 유효한가
------------------------------------------
결정나무는 "x ≤ 임계값" 으로 나누므로 **전 표본에 같은 단조 변환**(로그 · 전체 구간
스케일링 · 전 표본 분위 winsorize)을 걸어도 같은 분할을 찾는다 — LightGBM ·
RandomForest 에 `StandardScaler` 가 필요 없는 이유다. 그런데 **횡단면 변환은 그것이
아니다.** 그날의 평균·산포·순위로 바꾸면 "절대 수준" 이 "그날 안에서의 상대 위치" 로
뜻이 바뀌고, 날짜가 다른 두 행의 순서가 뒤집힌다(변동성이 큰 날의 +3% 는 낮은 순위,
조용한 날의 +3% 는 높은 순위). 그래서 ①②④ 도 트리에 유효하다 — 이슈 #155 의 횡단면
순위(조합 F)가 그 예다. 시험 `test_전_표본_단조_변환은_트리에_무의미하지만_횡단면_변환은_다르다`
가 두 사실을 함께 못박는다. 로지스틱 회귀에는 넷 다 유효하다.

왜 분위(1%/99%)가 아니라 MAD 가 기본인가
--------------------------------------
분위는 **종목 수에 대한 비율**이다. MSCI 품질지수 방법론(2014)이 5/95 분위를 200종
예시로 설명하는데, 우리 후보군은 날짜별 46~50종이라 1% 는 0.49종이다 — 양끝 한 개씩만
닿고 이동은 중앙 0.1~0.2σ 에 그친다(2026-09-09 실측 · 175,914행). 표본이 작을 때는
**산포**로 자르는 것이 맞다. `median ± k × 1.4826 × MAD` 는 qlib `RobustZScoreNorm`
(microsoft/qlib `79633dd`) 과 같은 정의이고 1.4826 은 정규분포에서 MAD 를 σ 로 맞추는
상수다(Wikipedia "Median absolute deviation"). 분위 방식은 학술 관행(국내 재무 논문의
"상하위 1% 윈저화")과 비교하려고 옵션으로 남긴다.

참고한 원문
----------
- Barra USE4 (MSCI 2011) — 기술자를 평균 ±3σ 로 winsorize 한 뒤 시총가중 평균 0 ·
  동일가중 표준편차 1 로 표준화
- Gu·Kelly·Xiu, "Empirical Asset Pricing via Machine Learning" (RFS 2020) — 월별
  횡단면 순위를 [−1, 1] 로 (`rank(method="signed")`)
- Kakushadze, "101 Formulaic Alphas" (2016) — `rank(x)` 횡단면 순위,
  `indneutralize(x, g)` 그룹 안 평균 빼기
- qlib `processor.py` — `CSZScoreNorm` · `CSRankNorm` · `RobustZScoreNorm`
"""

from __future__ import annotations

from typing import Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

#: 정규분포에서 MAD 를 표준편차로 맞추는 상수 — 1 / Φ⁻¹(3/4).
MAD_TO_SIGMA = 1.4826

#: 이 접두사로 시작하는 칸은 전처리 대상이 아니다. 라벨과 답, 진입·청산 가격, 원가격,
#: 공급 층 판정 플래그를 변환하면 "무엇을 맞히는가" 자체가 바뀐다.
PROTECTED_PREFIXES: tuple[str, ...] = (
    "label", "fwd_return", "realized", "entry_", "exit_", "adj_", "is_",
)
#: 이름이 정확히 이것인 칸도 받지 않는다 — 원가격·거래량·시총·등락률.
PROTECTED_NAMES: frozenset[str] = frozenset(
    {"open", "high", "low", "close", "volume", "value", "market_cap", "change_rate"}
)

#: 그날 종목이 이보다 적으면 통계를 만들지 않는다 — 셋 미만이면 MAD 도 표준편차도 뜻이 없다.
DEFAULT_MIN_COUNT = 3

#: 시계열 표준화(⑤)의 기본 창과 준비구간. 250거래일은 약 1년이고, 60 은 석 달이다.
#: 60 을 고른 이유는 **표본이 줄지 않기 때문**이다 — 이 패널에서 가장 늦게 값이 나오는
#: 피처(`hv_regime`)가 269행을 먹으므로, 60행짜리 준비구간은 그 안에 통째로 들어간다.
DEFAULT_TS_WINDOW = 250
DEFAULT_TS_MIN_PERIODS = 60

#: 날짜 수준을 되돌릴 때 새 칸에 붙이는 머리말. `cs_mean_hv_20` 처럼 읽힌다.
DATE_LEVEL_PREFIX = "cs_"

#: 라벨이 **절대 밴드**(±2%)라서 수준을 지우면 안 되는 축. 2026-09-09 실측 —
#: `hv_20` 십분위와 "라벨 ≠ 중립" 확률의 Spearman ρ 가 1.0000 이었다(47.6% → 75.2%).
#: 변동성이 큰 종목·날일수록 ±2% 를 벗어나므로, 이 넷의 **수준 자체가 라벨의 절반**이다.
#: 값이 아니라 이름 목록이므로, 조합에 그 칸이 없으면 그냥 걸리지 않는다.
VOLATILITY_AXIS: tuple[str, ...] = ("hv_20", "atr_ratio", "hv_regime", "bb_bandwidth")

#: 중립화(③)의 크기 통제 변수 이름. 시총 자체는 보호 칸(`PROTECTED_NAMES`)이라 피처로도
#: 통제 변수로도 그냥 쓰지 않고, `log_size_column` 이 만든 이 칸을 쓴다.
SIZE_CONTROL = "log_cap"

WinsorMethod = Literal["mad", "quantile", "sigma"]
RankMethod = Literal["uniform", "gaussian", "signed"]
DateLevelStat = Literal["mean", "std", "median"]

__all__ = [
    "DATE_LEVEL_PREFIX",
    "DEFAULT_TS_MIN_PERIODS",
    "DEFAULT_TS_WINDOW",
    "MAD_TO_SIGMA",
    "PROTECTED_NAMES",
    "PROTECTED_PREFIXES",
    "SIZE_CONTROL",
    "VOLATILITY_AXIS",
    "date_level_columns",
    "log_size_column",
    "neutralize_cross_section",
    "preprocess_cross_section",
    "rank_cross_section",
    "restore_date_level",
    "split_by_axis",
    "standardize_time_series",
    "transformed_columns",
    "winsorize_cross_section",
    "zscore_cross_section",
]


# ══════════════════════════════════════════════════════════════════════════
# 공통 — 입력 검사
# ══════════════════════════════════════════════════════════════════════════
def _check(frame: pd.DataFrame, columns: Sequence[str], date_col: str) -> list[str]:
    """칸 목록을 확정한다. 비어 있거나, 없거나, 숫자가 아니거나, 보호 대상이면 거부한다."""
    cols = list(dict.fromkeys(columns))
    if not cols:
        raise ValueError("전처리할 칸이 비어 있습니다.")
    if date_col not in frame.columns:
        raise ValueError(f"날짜 칸이 없습니다: {date_col!r}")
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise ValueError(f"전처리할 칸이 표에 없습니다: {missing}")
    protected = [
        c for c in cols
        if c in PROTECTED_NAMES or c.startswith(PROTECTED_PREFIXES)
    ]
    if protected:
        raise ValueError(
            f"전처리 대상이 아닌 칸입니다: {protected} — 라벨·수익률·가격·플래그는 손대지 않습니다."
        )
    not_numeric = [c for c in cols if not pd.api.types.is_numeric_dtype(frame[c])]
    if not_numeric:
        raise ValueError(f"숫자 칸만 전처리합니다: {not_numeric}")
    if frame[date_col].isna().any():
        raise ValueError("날짜 칸에 결측이 있습니다.")
    return cols


def _values(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """계산용 float64 복사본. 입력은 건드리지 않는다."""
    return frame.loc[:, cols].astype("float64")


# ══════════════════════════════════════════════════════════════════════════
# ① winsorize
# ══════════════════════════════════════════════════════════════════════════
def winsorize_cross_section(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    date_col: str = "bas_dd",
    method: WinsorMethod = "mad",
    k: float = 3.0,
    limits: tuple[float, float] = (0.01, 0.99),
    min_count: int = DEFAULT_MIN_COUNT,
) -> pd.DataFrame:
    """그날 횡단면의 극단값을 경계로 **당긴다**(자르지 않는다 — 행은 하나도 안 사라진다).

    method
        ``"mad"``       median ± k × 1.4826 × MAD           (기본 · 소표본에 맞다)
        ``"sigma"``     mean ± k × std                       (Barra USE4 · k=3)
        ``"quantile"``  [limits[0], limits[1]] 분위          (학술 관행 · 1%/99%)

    MAD 가 0 이면(그날 값이 절반 넘게 같으면) 경계를 만들 수 없어 그대로 둔다. 종목이
    `min_count` 보다 적은 날도 그대로 둔다.
    """
    cols = _check(frame, columns, date_col)
    if k <= 0:
        raise ValueError("k 는 0 보다 커야 합니다.")
    lo_q, hi_q = limits
    if not (0.0 <= lo_q < hi_q <= 1.0):
        raise ValueError(f"limits 는 0 ≤ 하한 < 상한 ≤ 1 이어야 합니다: {limits}")
    if method not in ("mad", "quantile", "sigma"):
        raise ValueError(f"모르는 winsorize 방식입니다: {method!r}")

    x = _values(frame, cols)
    g = x.groupby(frame[date_col], sort=False)
    n = g.transform("count")

    if method == "mad":
        center = g.transform("median")
        scale = (x - center).abs().groupby(frame[date_col], sort=False).transform("median")
        scale = scale * MAD_TO_SIGMA * k
    elif method == "sigma":
        center = g.transform("mean")
        scale = g.transform("std") * k
    else:
        lower = g.transform(lambda s: s.quantile(lo_q))
        upper = g.transform(lambda s: s.quantile(hi_q))

    if method in ("mad", "sigma"):
        lower = center - scale
        upper = center + scale
        # 산포가 0 이면 경계가 한 점으로 무너진다 — 그날은 손대지 않는다.
        usable = scale.gt(0) & n.ge(min_count)
    else:
        usable = n.ge(min_count)

    out = x.clip(lower=lower, upper=upper, axis=None)
    out = out.where(usable, x)
    out.index = frame.index
    return out


# ══════════════════════════════════════════════════════════════════════════
# ② z-score
# ══════════════════════════════════════════════════════════════════════════
def zscore_cross_section(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    date_col: str = "bas_dd",
    robust: bool = False,
    scale: bool = True,
    min_count: int = DEFAULT_MIN_COUNT,
) -> pd.DataFrame:
    """그날 평균·표준편차(또는 중앙값·MAD)로 표준화한다.

    표준편차는 `ddof=1` 이다 — qlib `CSZScoreNorm` · sklearn 과 다를 수 있는데 sklearn 은
    `ddof=0` 을 쓴다. 이 프로젝트는 pandas 기본(ddof=1)을 그대로 두고, 값이 어느 쪽인지
    여기 적어 둔다. 산포가 0 이거나 종목이 `min_count` 보다 적은 날은 NaN 이다.

    ## `scale=False` — 중심화만 (그날 평균 빼기)

    표준화는 **두 가지**를 한다. 그날 평균을 빼고(중심화), 그날 산포로 나눈다(척도화).
    `scale=False` 는 앞의 하나만 한다. 산포로 나누지 않으므로 산포가 0 인 날도 값이
    남는다(전부 0 이 된다).

    이 갈래가 필요한 이유는 2026-09-10 실측이다 — **모든 중립화(③)가 설계행렬에 절편이나
    그룹 더미를 넣으므로 그날 평균을 함께 지운다.** `groups=None, controls=("log_cap",)`
    로 잰 "시총 중립화" 의 잔차가 그냥 그날 평균을 뺀 값과 상관 0.9566~0.9831 이었고,
    시총이 **추가로** 설명한 몫은 4.46% 뿐이었다. 그러니 중립화의 손실을 "그 축을 지운
    값" 으로 읽으려면 **중심화만 한 조건과 견줘야** 한다. 그게 이 인자다.
    """
    cols = _check(frame, columns, date_col)
    x = _values(frame, cols)
    g = x.groupby(frame[date_col], sort=False)
    n = g.transform("count")
    if robust:
        center = g.transform("median")
        분모 = (x - center).abs().groupby(frame[date_col], sort=False).transform("median")
        분모 = 분모 * MAD_TO_SIGMA
    else:
        center = g.transform("mean")
        분모 = g.transform("std")
    if not scale:
        # 나누지 않으므로 산포가 0 인 날도 살린다 — 막는 것은 종목 수 조건 하나다.
        return (x - center).where(n.ge(min_count)).set_axis(frame.index)
    out = (x - center) / 분모
    out = out.where(분모.gt(0) & n.ge(min_count))
    out.index = frame.index
    return out


# ══════════════════════════════════════════════════════════════════════════
# ③ neutralize — 회귀 잔차
# ══════════════════════════════════════════════════════════════════════════
def log_size_column(
    frame: pd.DataFrame, *, cap_col: str = "market_cap"
) -> pd.Series:
    """시총을 자연로그로 바꾼 한 칸을 만든다 — 중립화(③)의 크기 통제 변수.

    ## 왜 로그인가

    시총은 왜도가 극심하다. 2026-09-10 조합 K 패널 실측에서 최소 1,799억 · 중앙값 8.5조 ·
    최대 543조로 **3,020배** 벌어져 있다. 원값으로 회귀하면 대형주 몇 종목이 기울기를
    혼자 정하고 나머지 40여 종의 잔차는 거의 원값 그대로 남는다. 로그를 씌우면 그
    3,020배가 8.0 차이가 되어 종목들이 같은 자에 놓인다. Fama–French 의 `SMB` 부터
    중국 A주 다요인 관행까지 크기 축을 로그로 쓰는 이유가 이것이다.

    ## 0 이하는 채우지 않고 결측으로 둔다

    시총이 0 이하인 것은 자료 오류다. `clip(lower=1)` 로 막으면 log 가 0 이 되어 **가장
    작은 회사**처럼 보이는데, 그것은 사실이 아니라 오류를 그럴듯한 값으로 바꾼 것이다.
    결측으로 두면 `neutralize_cross_section` 이 그 행을 회귀에서 빼고 잔차도 결측으로
    남긴다 — 어디가 비었는지 셀 수 있다.
    """
    if cap_col not in frame.columns:
        raise ValueError(
            f"시총 칸이 없습니다: {cap_col!r} — 패널에 그 칸을 붙이고 다시 부르십시오."
        )
    cap = pd.to_numeric(frame[cap_col], errors="coerce").astype("float64")
    return pd.Series(
        np.log(cap.where(cap > 0)), index=frame.index, name=SIZE_CONTROL
    )


def _design_matrix(
    sub: pd.DataFrame, groups: str | None, controls: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """그날의 설계행렬 X 와 "X 가 완전한 행" 마스크를 만든다.

    그룹이 있으면 그룹 더미 전부(절편 없이), 없으면 절편 하나. 거기에 연속 통제 변수.
    """
    parts: list[np.ndarray] = []
    ok = np.ones(len(sub), dtype=bool)
    if groups is not None:
        codes, _ = pd.factorize(sub[groups], sort=True, use_na_sentinel=True)
        ok &= codes >= 0
        n_levels = int(codes.max()) + 1 if len(codes) else 0
        dummies = np.zeros((len(sub), max(n_levels, 0)), dtype="float64")
        rows = np.arange(len(sub))[codes >= 0]
        dummies[rows, codes[codes >= 0]] = 1.0
        parts.append(dummies)
    else:
        parts.append(np.ones((len(sub), 1), dtype="float64"))
    for c in controls:
        v = sub[c].to_numpy(dtype="float64")
        ok &= np.isfinite(v)
        parts.append(v[:, None])
    return np.hstack(parts), ok


def neutralize_cross_section(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    date_col: str = "bas_dd",
    groups: str | None = "industry",
    controls: Sequence[str] = (),
    min_count: int | None = None,
) -> pd.DataFrame:
    """그날 안에서 피처를 `groups` 더미와 `controls` 에 회귀하고 **잔차**를 돌려준다.

    - `groups="industry"`, `controls=()`         → 업종 안 평균 빼기 (Kakushadze `indneutralize`)
    - `groups="industry"`, `controls=("log_cap",)` → 중국 A주 관행(업종 + log 시총 회귀 잔차)
    - `groups=None`, `controls=("log_cap",)`       → 그날 평균 + 시총 (아래 🔴)

    ## 🔴 어느 축을 주든 **그날 평균이 함께 지워진다**

    설계행렬에는 늘 절편이 들어간다 — `groups` 가 있으면 더미 전부(그것이 절편 노릇을
    한다), 없으면 `1` 한 칸이다. 절편이 있는 최소제곱의 잔차는 **평균이 정확히 0** 이다.
    그러니 이 함수는 어떤 축을 지우든 **날짜 수준을 먼저 지운다.**

    이름이 그 절반만 말한다는 것을 2026-09-10 에 실측으로 확인했다. 조합 K 패널
    159,900행에서 `groups=None, controls=("log_cap",)`("시총 중립화")의 잔차와, 그냥 그날
    평균만 뺀 값(`zscore_cross_section(..., scale=False)`)의 상관이 **0.9566~0.9831** 이고
    시총이 **추가로** 설명한 몫은 평균 **4.46%** 뿐이었다. 12폴드 정확도로도 원값 대비
    −1.94%p 로 나빠졌는데, 시총은 라벨(절대 ±2% 밴드)과 점이연 상관 −0.0086 으로 거의
    무관하다 — 나빠진 것은 시총이 아니라 **날짜 수준을 지웠기 때문**이다.

    그래서 중립화의 손실을 "그 축을 지운 값" 으로 읽으면 안 된다. 축의 몫만 보려면
    `zscore="center"` 조건과 견줘야 한다(전처리 v1.4 §2).

    다만 **산포는 남는다** — 평균만 빼고 나누지 않으므로 날짜별 표준편차의 최대/최소가
    원값 7.55배에서 7.73배로 그대로다(같은 실측). 횡단면 z 와 순위는 1.000 으로 산포까지
    지운다. 세 연산이 지우는 것이 서로 다르다.

    설명변수에 결측이 있는 행은 잔차도 NaN 이다. 그날의 완전한 행 수가 매개변수 수보다
    `min_count`(기본: 매개변수 수 + 2) 만큼 넘지 않으면 그날 전체가 NaN 이다 — 자유도가
    없는 회귀의 잔차는 0 에 가까운 잡음일 뿐이다.

    피처 하나마다 그날의 완전한 행만 골라 최소제곱을 푼다. 후보 50종 × 3,600일 규모에서
    몇 초, 전 시장 900종이면 몇 분이다.
    """
    cols = _check(frame, columns, date_col)
    if groups is None and not controls:
        raise ValueError("중립화 축이 없습니다 — groups 나 controls 중 하나는 있어야 합니다.")
    if groups is not None and groups not in frame.columns:
        raise ValueError(f"그룹 칸이 없습니다: {groups!r}")
    for c in controls:
        if c not in frame.columns:
            raise ValueError(f"통제 칸이 없습니다: {c!r}")
        if not pd.api.types.is_numeric_dtype(frame[c]):
            raise ValueError(f"통제 칸은 숫자여야 합니다: {c!r}")
    overlap = set(cols) & set(controls)
    if overlap:
        raise ValueError(f"통제 변수를 자기 자신에 회귀할 수 없습니다: {sorted(overlap)}")

    y_all = _values(frame, cols).to_numpy()
    out = np.full_like(y_all, np.nan)
    positions = np.arange(len(frame))
    for _, idx in frame.groupby(date_col, sort=False).indices.items():
        sub = frame.iloc[idx]
        X, ok = _design_matrix(sub, groups, controls)
        p = X.shape[1]
        need = p + 2 if min_count is None else max(min_count, p + 1)
        for j in range(len(cols)):
            y = y_all[idx, j]
            m = ok & np.isfinite(y)
            if int(m.sum()) < need:
                continue
            beta, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
            resid = y[m] - X[m] @ beta
            out[positions[idx][m], j] = resid
    result = pd.DataFrame(out, columns=cols, index=frame.index)
    return result


# ══════════════════════════════════════════════════════════════════════════
# ④ rank
# ══════════════════════════════════════════════════════════════════════════
def rank_cross_section(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    date_col: str = "bas_dd",
    method: RankMethod = "gaussian",
    min_count: int = DEFAULT_MIN_COUNT,
) -> pd.DataFrame:
    """그날 순위로 바꾼다. 동률은 평균 순위다.

    method
        ``"uniform"``   (순위 − 0.5) / n          ∈ (0, 1)      — 백분위. 기존 `*_rank` 칸과
                                                                 같은 뜻이나 끝점을 피한다
        ``"gaussian"``  Φ⁻¹(uniform)              ≈ N(0, 1)     — 정규 순위 (기본)
        ``"signed"``    2 × 순위 / (n + 1) − 1     ∈ (−1, 1)     — Gu·Kelly·Xiu (2020)

    결측은 순위에서 빠지고 결측으로 남는다(GKX 는 0 으로 채우지만 여기서는 채우지 않는다 —
    채울지는 모델 쪽이 정한다). 종목이 `min_count` 보다 적은 날은 NaN 이다.
    """
    cols = _check(frame, columns, date_col)
    if method not in ("uniform", "gaussian", "signed"):
        raise ValueError(f"모르는 순위 방식입니다: {method!r}")
    x = _values(frame, cols)
    g = x.groupby(frame[date_col], sort=False)
    r = g.rank(method="average")
    n = g.transform("count")
    if method == "signed":
        out = 2.0 * r / (n + 1.0) - 1.0
    else:
        u = (r - 0.5) / n
        out = u if method == "uniform" else pd.DataFrame(
            norm.ppf(u.to_numpy()), columns=cols, index=u.index
        )
    out = out.where(n.ge(min_count))
    out.index = frame.index
    return out


# ══════════════════════════════════════════════════════════════════════════
# ⑤ 시계열 표준화 — 종목이 자기 최근 이력 대비 얼마나 다른가
# ══════════════════════════════════════════════════════════════════════════
def standardize_time_series(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    date_col: str = "bas_dd",
    by: str = "code",
    window: int = DEFAULT_TS_WINDOW,
    min_periods: int = DEFAULT_TS_MIN_PERIODS,
    robust: bool = False,
) -> pd.DataFrame:
    """각 종목의 **자기 최근 `window` 행** 평균·표준편차로 표준화한다.

        z_t = (x_t − mean(x_{t−w+1 … t})) / std(x_{t−w+1 … t})

    ①~④ 가 **날짜 안**에서 종목을 비교한다면, 이것은 **종목 안**에서 시점을 비교한다.
    방향이 직각이라 지우는 것도 다르다.

    ## 왜 이 함수가 필요한가 — 날짜별 z 는 라벨이 쓰는 것을 지운다

    우리 라벨은 **절대 ±2% 밴드**다. 변동성이 높은 날은 종목을 가리지 않고 밴드를 더
    벗어난다(2026-09-09 실측 · 날짜별 평균 `hv_20` ↔ 그날 비중립 비율 ρ = 0.3244 ·
    3,343일). 그런데 날짜별 z-score 는 **정의상 그날 평균을 0 으로 만들어** 그 정보를
    통째로 지운다. 실제로 조합 K 12폴드에서 기준선 대비 정확도가 +2.43%p → +0.84%p 로
    떨어졌다(이슈 #210·#214).

    시계열 표준화는 그 수준을 남긴다 — 시장 전체가 시끄러우면 **모든 종목의 z 가 함께**
    오르므로 날짜별 평균이 0 이 되지 않는다. 대신 종목마다 다른 기준 수준(삼성전자와
    소형주의 `hv_20` 평균이 애초에 다르다)은 지운다. 그건 라벨과 무관한 종목 고정효과다.

    ## 🔴 인과성 — 이 함수의 존재 이유

    `rolling(window)` 은 **t 를 포함하되 t+1 이후는 절대 보지 않는다.** 그래서 `shift` 가
    따로 필요 없다. 뒤 행을 잘라내도 앞 행의 z 는 한 자리도 바뀌지 않으며, 그것을
    `test_뒤_행을_잘라도_앞_행의_z_는_같다` 가 못박는다. 전 구간 `StandardScaler` 와
    다른 점이 정확히 이것이다.

    ## ⚠️ 창은 달력이 아니라 **행**을 센다

    `window=250` 은 "250거래일" 이 아니라 **"이 표에 그 종목이 나타난 250행"** 이다.
    후보군 패널처럼 종목이 들락날락하는 표에서는 250행이 250거래일보다 긴 기간을 덮는다.
    전 종목 일별 패널에 걸면 둘이 같아진다. 어느 쪽이든 미래를 보지 않는 것은 같다.

    준비구간(`min_periods` 미만)은 NaN 이다. 기본 60 을 고른 이유는 상수
    `DEFAULT_TS_MIN_PERIODS` 주석 참조 — 표본이 줄지 않는다.

    `robust=True` 면 중앙값과 MAD 를 쓴다. 산포가 0 인 구간(값이 내내 같은 종목)은 NaN 이다.
    """
    cols = _check(frame, columns, date_col)
    if by not in frame.columns:
        raise ValueError(f"종목 칸이 없습니다: {by!r}")
    if window < 2:
        raise ValueError(f"window 는 2 이상이어야 합니다: {window}")
    if not (1 <= min_periods <= window):
        raise ValueError(f"min_periods 는 1 이상 window 이하여야 합니다: {min_periods}")
    if frame[by].isna().any():
        raise ValueError(f"종목 칸에 결측이 있습니다: {by!r}")
    if frame.duplicated([by, date_col]).any():
        # 같은 종목·같은 날이 두 행이면 창 안의 순서가 정해지지 않는다. 그 표는
        # 어느 행이 먼저인지에 따라 결과가 달라지므로 계산하지 않고 멈춘다.
        raise ValueError(f"같은 ({by}, {date_col}) 행이 두 번 이상 있습니다.")

    # 🔴 정렬은 계산용 사본에서만 한다. 입력이 날짜순이 아니어도 결과는 같아야 하고,
    #    돌려주는 표는 입력과 같은 인덱스·같은 순서다.
    order = frame.sort_values([by, date_col], kind="mergesort").index
    x = _values(frame.loc[order], cols)
    groups = frame.loc[order, by]
    g = x.groupby(groups, sort=False)
    if robust:
        center = g.rolling(window, min_periods=min_periods).median().droplevel(0)
        deviation = (x - center).abs()
        scale = (
            deviation.groupby(groups, sort=False)
            .rolling(window, min_periods=min_periods)
            .median()
            .droplevel(0)
            * MAD_TO_SIGMA
        )
    else:
        center = g.rolling(window, min_periods=min_periods).mean().droplevel(0)
        scale = g.rolling(window, min_periods=min_periods).std().droplevel(0)
    out = (x - center) / scale
    out = out.where(scale.gt(0))
    return out.reindex(frame.index)


# ══════════════════════════════════════════════════════════════════════════
# ⑥ 날짜 수준 복원 — z 가 지운 것을 피처로 돌려준다
# ══════════════════════════════════════════════════════════════════════════
def date_level_columns(
    columns: Iterable[str],
    stats: Sequence[DateLevelStat] = ("mean", "std"),
    *,
    prefix: str = DATE_LEVEL_PREFIX,
) -> list[str]:
    """`restore_date_level` 이 만들 칸 이름. 부르는 쪽이 피처 목록을 미리 세울 때 쓴다."""
    return [f"{prefix}{stat}_{col}" for col in dict.fromkeys(columns) for stat in stats]


def restore_date_level(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    date_col: str = "bas_dd",
    stats: Sequence[DateLevelStat] = ("mean", "std"),
    prefix: str = DATE_LEVEL_PREFIX,
    min_count: int = DEFAULT_MIN_COUNT,
) -> pd.DataFrame:
    """날짜별 z-score 가 지운 **그날의 수준**을 새 칸으로 되돌린다.

    `zscore_cross_section` 은 그날 평균을 0 · 표준편차를 1 로 만든다. 즉 지워지는 것은
    정확히 **그날의 평균과 산포** 둘이다. 그것을 명시적 피처로 되돌리면, 모델이 상대
    위치(z)와 절대 수준(그날 평균)을 **함께** 볼 수 있다.

    이게 맞는 설명인지 재는 것이 이 함수의 목적이다 — 지운 것을 돌려줘서 회복하면
    "z 가 해친 것은 날짜 수준" 이 맞고, 회복하지 않으면 다른 원인이 있다(이슈 #214 후보 C).

    ## 미래를 보지 않는다

    통계는 **그날 행만으로** 만든다. 다른 날짜의 값이 오늘 결과를 바꾸지 않는 것은
    ①~④ 와 같은 규약이고, 같은 누수 시험이 이 함수에도 걸린다.

    ## 돌려주는 것은 **새 칸만** 이다

    ①~④ 는 "요청한 칸을 같은 이름으로" 돌려주지만, 이 함수는 `cs_mean_hv_20` 처럼
    **새 이름의 칸만** 돌려준다. 원래 칸을 덮어쓰지 않으므로 부르는 쪽이 그대로 옆에
    붙이면 된다(`pd.concat`). 피처 목록도 함께 늘려야 한다 —
    `date_level_columns()` 가 그 이름을 미리 준다.

    한 날짜의 종목이 `min_count` 보다 적으면 그날은 NaN 이다.
    """
    cols = _check(frame, columns, date_col)
    unknown = [s for s in stats if s not in ("mean", "std", "median")]
    if unknown:
        raise ValueError(f"모르는 날짜 수준 통계입니다: {unknown}")
    if not stats:
        raise ValueError("되돌릴 통계가 비어 있습니다.")
    x = _values(frame, cols)
    g = x.groupby(frame[date_col], sort=False)
    n = g.transform("count")
    parts: dict[str, pd.Series] = {}
    for stat in stats:
        got = g.transform(stat)
        for col in cols:
            parts[f"{prefix}{stat}_{col}"] = got[col].where(n[col].ge(min_count))
    out = pd.DataFrame(parts, index=frame.index)
    return out.loc[:, date_level_columns(cols, stats, prefix=prefix)]


# ══════════════════════════════════════════════════════════════════════════
# 축 가르기 — 수준을 보존할 칸과 상대 위치로 바꿀 칸
# ══════════════════════════════════════════════════════════════════════════
def split_by_axis(
    columns: Iterable[str],
    keep_raw: Iterable[str] = VOLATILITY_AXIS,
) -> tuple[list[str], list[str]]:
    """`(처리할 칸, 원값으로 둘 칸)` 으로 가른다. 입력 순서를 지킨다.

    같은 표 안에 성질이 다른 두 종류가 섞여 있다는 것이 전제다.

    - **수준이 뜻인 칸** — `hv_20`·`atr_ratio` 처럼 값의 크기 자체가 라벨과 이어진다.
      횡단면 z 를 걸면 그 크기가 사라진다.
    - **이미 상대값인 칸** — `rsi_14`·`sma_gap_5_20` 은 종목 안에서 이미 정규화돼 있어
      횡단면 처리로 잃을 수준이 적다.

    `keep_raw` 에 있지만 `columns` 에 없는 이름은 **조용히 무시**한다 — 조합마다 칸이
    다르므로, 없는 칸을 넣었다고 멈추면 조합을 바꿀 때마다 목록을 고쳐야 한다.
    """
    보존 = set(keep_raw)
    cols = list(dict.fromkeys(columns))
    return [c for c in cols if c not in 보존], [c for c in cols if c in 보존]


# ══════════════════════════════════════════════════════════════════════════
# 한 번에 — 순서는 ① → ② → ⑤ → ③ → ④
# ══════════════════════════════════════════════════════════════════════════
def preprocess_cross_section(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    date_col: str = "bas_dd",
    winsorize: WinsorMethod | None = "mad",
    zscore: bool | Literal["center"] = True,
    neutralize: Mapping[str, object] | None = None,
    rank: RankMethod | None = None,
    winsorize_kwargs: Mapping[str, object] | None = None,
    time_series: Mapping[str, object] | None = None,
    keep_raw: Iterable[str] | None = None,
) -> pd.DataFrame:
    """단계들을 표준 순서로 건다. 끄고 싶은 단계는 `None`/`False`.

        ① winsorize → ② 횡단면 z → ⑤ 시계열 z → ③ 중립화 → ④ 순위

    `zscore` 는 `True`(평균 빼고 산포로 나눔) · `"center"`(그날 평균만 뺌) · `False`
    셋이다. `"center"` 는 중립화가 절편 때문에 함께 지우는 것을 따로 재려고 있다
    (`zscore_cross_section` 의 `scale` 설명 참조).

    `neutralize` 는 `neutralize_cross_section` 의 키워드 인자 dict
    (예: ``{"groups": "industry", "controls": ("log_cap",)}``). 순위(④)를 켜면 그 앞
    단계의 결과 순서가 그대로 순위가 되므로, ①·② 는 ④ 앞에서는 아무것도 바꾸지 않는다 —
    ④ 를 켤 때 의미가 있는 것은 ③ 뿐이다.

    `time_series` 는 `standardize_time_series` 의 키워드 인자 dict
    (예: ``{"window": 250, "min_periods": 60}``). 종목 축이라 날짜 축(②)과 방향이 직각이고,
    **둘을 함께 켜면 두 번 표준화된다** — 뜻이 없지는 않지만 해석이 어려워지므로 보통은
    한쪽만 켠다.

    `keep_raw` 에 든 칸은 **모든 단계를 건너뛰고 원값 그대로** 나온다. 라벨이 절대
    밴드라 수준이 곧 정보인 축(`VOLATILITY_AXIS`)을 지키려고 있다. 돌려주는 표의 칸
    순서·이름은 `keep_raw` 와 무관하게 요청한 그대로다.

    이 함수는 예나 지금이나 **요청한 칸만** 돌려준다. 날짜 수준을 피처로 되돌리는
    `restore_date_level` 이 새 칸을 만드는 것과 다르다 — 그쪽은 부르는 쪽이 직접 붙인다.
    """
    if zscore not in (True, False, "center"):
        raise ValueError(
            f"zscore 는 True · False · 'center' 중 하나입니다: {zscore!r}"
        )
    cols = _check(frame, columns, date_col)
    처리, 원값 = split_by_axis(cols, keep_raw or ())
    work = frame.loc[:, [date_col, *cols]].copy()
    extra = []
    if neutralize:
        for key in ("groups",):
            v = neutralize.get(key)
            if isinstance(v, str):
                extra.append(v)
        extra.extend(neutralize.get("controls", ()) or ())
    if time_series is not None:
        extra.append(str(time_series.get("by", "code")))
    for c in extra:
        if c not in work.columns:
            work[c] = frame[c]

    if 처리:
        if winsorize is not None:
            work[처리] = winsorize_cross_section(
                work, 처리, date_col=date_col, method=winsorize,
                **(winsorize_kwargs or {}),
            )
        if zscore:
            work[처리] = zscore_cross_section(
                work, 처리, date_col=date_col, scale=zscore != "center",
            )
        if time_series is not None:
            work[처리] = standardize_time_series(
                work, 처리, date_col=date_col, **time_series
            )
        if neutralize:
            work[처리] = neutralize_cross_section(
                work, 처리, date_col=date_col, **neutralize
            )
        if rank is not None:
            work[처리] = rank_cross_section(work, 처리, date_col=date_col, method=rank)
    if 원값:
        # 원값 칸은 사본에서도 한 번도 안 건드렸으므로 그대로 두면 된다. 다만 "그대로"
        # 라는 사실을 코드로 남겨 둔다 — 나중에 위 블록을 고칠 때 여기가 눈에 걸리게.
        work[원값] = frame.loc[:, 원값]
    out = work.loc[:, cols]
    out.index = frame.index
    return out


def transformed_columns(columns: Iterable[str]) -> list[str]:
    """도움 함수 — 주어진 이름 중 전처리 대상이 될 수 있는 것만 남긴다 (보호 칸 제외)."""
    return [
        c for c in dict.fromkeys(columns)
        if c not in PROTECTED_NAMES and not c.startswith(PROTECTED_PREFIXES)
    ]
