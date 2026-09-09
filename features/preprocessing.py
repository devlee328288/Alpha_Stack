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

WinsorMethod = Literal["mad", "quantile", "sigma"]
RankMethod = Literal["uniform", "gaussian", "signed"]

__all__ = [
    "MAD_TO_SIGMA",
    "PROTECTED_NAMES",
    "PROTECTED_PREFIXES",
    "neutralize_cross_section",
    "preprocess_cross_section",
    "rank_cross_section",
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
    min_count: int = DEFAULT_MIN_COUNT,
) -> pd.DataFrame:
    """그날 평균·표준편차(또는 중앙값·MAD)로 표준화한다.

    표준편차는 `ddof=1` 이다 — qlib `CSZScoreNorm` · sklearn 과 다를 수 있는데 sklearn 은
    `ddof=0` 을 쓴다. 이 프로젝트는 pandas 기본(ddof=1)을 그대로 두고, 값이 어느 쪽인지
    여기 적어 둔다. 산포가 0 이거나 종목이 `min_count` 보다 적은 날은 NaN 이다.
    """
    cols = _check(frame, columns, date_col)
    x = _values(frame, cols)
    g = x.groupby(frame[date_col], sort=False)
    n = g.transform("count")
    if robust:
        center = g.transform("median")
        scale = (x - center).abs().groupby(frame[date_col], sort=False).transform("median")
        scale = scale * MAD_TO_SIGMA
    else:
        center = g.transform("mean")
        scale = g.transform("std")
    out = (x - center) / scale
    out = out.where(scale.gt(0) & n.ge(min_count))
    out.index = frame.index
    return out


# ══════════════════════════════════════════════════════════════════════════
# ③ neutralize — 회귀 잔차
# ══════════════════════════════════════════════════════════════════════════
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
    - `groups=None`, `controls=("log_cap",)`       → 시총만

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
# 한 번에 — 순서는 ① → ② → ③ → ④
# ══════════════════════════════════════════════════════════════════════════
def preprocess_cross_section(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    date_col: str = "bas_dd",
    winsorize: WinsorMethod | None = "mad",
    zscore: bool = True,
    neutralize: Mapping[str, object] | None = None,
    rank: RankMethod | None = None,
    winsorize_kwargs: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """네 단계를 표준 순서로 건다. 끄고 싶은 단계는 `None`/`False`.

    `neutralize` 는 `neutralize_cross_section` 의 키워드 인자 dict
    (예: ``{"groups": "industry", "controls": ("log_cap",)}``). 순위(④)를 켜면 그 앞
    단계의 결과 순서가 그대로 순위가 되므로, ①·② 는 ④ 앞에서는 아무것도 바꾸지 않는다 —
    ④ 를 켤 때 의미가 있는 것은 ③ 뿐이다.
    """
    cols = _check(frame, columns, date_col)
    work = frame.loc[:, [date_col, *cols]].copy()
    extra = []
    if neutralize:
        for key in ("groups",):
            v = neutralize.get(key)
            if isinstance(v, str):
                extra.append(v)
        extra.extend(neutralize.get("controls", ()) or ())
    for c in extra:
        if c not in work.columns:
            work[c] = frame[c]

    if winsorize is not None:
        work[cols] = winsorize_cross_section(
            work, cols, date_col=date_col, method=winsorize, **(winsorize_kwargs or {})
        )
    if zscore:
        work[cols] = zscore_cross_section(work, cols, date_col=date_col)
    if neutralize:
        work[cols] = neutralize_cross_section(work, cols, date_col=date_col, **neutralize)
    if rank is not None:
        work[cols] = rank_cross_section(work, cols, date_col=date_col, method=rank)
    out = work.loc[:, cols]
    out.index = frame.index
    return out


def transformed_columns(columns: Iterable[str]) -> list[str]:
    """도움 함수 — 주어진 이름 중 전처리 대상이 될 수 있는 것만 남긴다 (보호 칸 제외)."""
    return [
        c for c in dict.fromkeys(columns)
        if c not in PROTECTED_NAMES and not c.startswith(PROTECTED_PREFIXES)
    ]
