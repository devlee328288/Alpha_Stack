"""③ 중립화 · ④ 순위를 09-09 의 A~D 와 **같은 자리**에서 잰다 (전처리 v1.4).

## 남아 있던 둘

`preprocess_cross_section` 의 네 단계 중 ①②(winsorize · 횡단면 z)와 ⑤(시계열 z)는
2026-09-09 에 조합 K · LogisticRegression · 12폴드로 다 쟀고 **전부 기각**했다
(`run_stock_preprocessing_candidates.py` · 전처리 v1.3). 켜 보지 않은 것이 ③ 중립화와
④ 순위다. 이 러너가 그 둘을 같은 자리에서 잰다.

## 왜 또 따로 만드나

09-09 러너가 #210(오준영 님 파일)을 안 건드리려고 조건 사전만 따로 뒀던 것과 같은
이유다. 그 보고서(`reports/stock_preprocessing_candidates.json`)는 **전처리 v1.3 의
근거로 고정**되어 있어 덮어쓰면 안 된다. 조립·검증·결측 처리·시행 원장은 그쪽 함수를
그대로 **가져다 쓴다** — 두 번 쓰면 두 판의 숫자가 조용히 갈라진다.

## 사전 등록 — 결과를 보기 전에 정한 것

- **후보 셋.** E 업종 중립화 · F 시총 중립화 · H 순위(gaussian).
- **판정 축**: [ADR 0007](../docs/decisions/0007-최종-모델-선정과-홀드아웃.md) 그대로
  기준선 대비 accuracy → Macro F1 → 승리 폴드. **이번 실행의 원값을 넘어야 채택**한다.
- 🔴 **IC·ICIR 로 판정하지 않는다.** 노트북
  `02-품질·전처리/05.횡단면-전처리는-날짜-안에서-끝난다.ipynb` §6.1 이 ICIR 0.1050 →
  0.1144 를 근거로 *"③ 중립화 = 업종 기본"* 이라고 적었고, 같은 노트북 §8.1 이
  *"3분류 정확도는 0.3938 → 0.3731 로 떨어졌다 · 이 노트북은 그 가설을 검증하지
  않았다"* 라고 달아 두었다. 이 러너가 그 숙제를 푼다. 두 축이 다른 답을 내는 것 자체가
  결과이므로 **IC·ICIR 도 같이 싣되 판정에는 쓰지 않는다.**
- **시행 N = 10.** 09-09 까지 7(#210 둘 + A·A′·B·C·D 다섯) + 이번 셋. Bonferroni α = 0.005.
- **두 표본을 내고, 판정은 `full` 로 한다.** ③④ 는 결측을 만들지 않아 159,900행이 그대로
  남는다. `common` 은 B 를 끼워 09-09(151,555행)와 이어붙이려고 5.22% 를 일부러 버린
  표본이라 **대조로만** 싣는다. 어느 쪽으로 판정하는지를 돌리기 전에 못박는 이유는,
  두 벌을 낸 뒤 유리한 쪽을 고르면 그것이 곧 표본 선택이기 때문이다.
- **한 번에 하나만 바꾼다.** 라벨·피처 목록·모델·폴드·표본 고정. ③ 앞에 winsorize·z 를
  켜지 않는다(09-09 의 B 와 같은 방식) — 나빠졌을 때 원인이 중립화 하나로 좁혀진다.

## 돌리기 전에 적어 두는 예상 — 그리고 그 근거

09-09 에 B(`ts_zscore_causal`)를 *"종목 고정효과라 라벨과 무관하다"* 라고 설계했다가,
기각된 뒤에야 종목 평균 `hv_20` 과 그 종목 비중립 비율의 Spearman ρ 가 **0.9413** 인
것을 쟀다. 통계 용어가 그럴듯해서 확인을 건너뛴 것이다. 그래서 이번에는 **돌리기 전에**
잰다(`axis_label_link`). 2026-09-10 실측:

    E 가 지우는 업종 수준   업종 평균 hv_20 ↔ 업종 비중립     ρ = 0.9429   → 나빠질 것
    F 가 지우는 시총 수준   log 시총        ↔ 비중립          r = −0.0086  → 무해할 것
    H 가 지우는 것          날짜·업종·종목 수준 전부                       → 가장 나쁠 것

세 예상이 서로 다르다. **F 는 음성 대조군이다** — "지우는 축이 라벨과 이어진 만큼
나빠진다" 가 설명이라면, 라벨과 이어지지 않은 축을 지울 때는 안 나빠져야 한다. 셋이 다
맞으면 설명이 선 것이고, F 까지 나빠지면 설명이 틀렸고 중립화 자체(자유도 손실 등)가
원인이다.

    python scripts/run_stock_preprocessing_axis.py            # 두 벌 다
    python scripts/run_stock_preprocessing_axis.py --plan     # 무엇을 돌릴지만
    python scripts/run_stock_preprocessing_axis.py --sample full
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.walk_forward import expanding_group_splits  # noqa: E402
from features.preprocessing import SIZE_CONTROL, log_size_column  # noqa: E402
from features.stock_model_dataset import (  # noqa: E402
    STOCK_COMBINATION_FEATURES,
    StockModelDataset,
    align_stock_feature_datasets,
)
from models.experiment import MODEL_BUILDERS  # noqa: E402
from models.stock_experiment import (  # noqa: E402
    LABEL_HORIZON,
    MIN_TRAIN_DATES,
    N_FOLDS,
    VALID_DATES,
    StockExperimentResult,
    evaluate_stock_models,
)
from scripts.run_stock_model_experiment import (  # noqa: E402
    DAILY_PATH,
    INDEX_PATH,
    _json_default,
    _sha256,
    load_stock_model_dataset,
)
from scripts.run_stock_preprocessing_ablation import (  # noqa: E402
    COMBINATION,
    MODEL_NAME,
    RAW,
    _condition_report,
)
from scripts.run_stock_preprocessing_candidates import (  # noqa: E402
    CONDITIONS as CANDIDATE_CONDITIONS,
)
from scripts.run_stock_preprocessing_candidates import (  # noqa: E402
    _append_trials,
    _paired_t,
    build_condition_dataset,
    common_sample,
)

REPORT_PATH = ROOT / "reports" / "stock_preprocessing_axis.json"
REPORT_RELATIVE = "reports/stock_preprocessing_axis.json"

#: 09-09 까지 센 시행 수. #210 의 둘(원값 · winsor+z)과 A·A′·B·C·D 다섯이다.
TRIALS_BEFORE = 7

#: 이번 판의 조건 셋. 값은 `preprocess_cross_section` 의 키워드 인자 그대로다.
#:
#: 🔴 `winsorize`·`zscore` 를 **명시적으로 끈다.** 기본값이 `winsorize="mad"`,
#:    `zscore=True` 라서 생략하면 세 단계가 묶여 걸리고, 나빠졌을 때 원인을 못 가른다.
AXIS_CONDITIONS: dict[str, dict[str, object] | None] = {
    # E — Kakushadze(2016) `indneutralize`. 그날 업종 안 평균을 뺀 잔차만 남긴다.
    "E_neutralize_industry": dict(
        winsorize=None, zscore=False,
        neutralize=dict(groups="industry", controls=()),
    ),
    # F — 크기 축만. **음성 대조군**이다(위 예상 참조).
    "F_neutralize_size": dict(
        winsorize=None, zscore=False,
        neutralize=dict(groups=None, controls=(SIZE_CONTROL,)),
    ),
    # H — qlib `CSRankNorm` 과 같은 목적. Φ⁻¹ 라 값의 수준이 통째로 사라진다.
    "H_rank_gaussian": dict(winsorize=None, zscore=False, rank="gaussian"),
}

#: 업종 라벨을 섞어 넣는 두 칸과 시드. R 과 R2 가 각각 쓴다.
#:
#: 🔴 **둘이 필요한 이유** — 우리 후보군은 *"매 거래일 업종 시총 상위 10 × 업종별 상위 5"*
#: 규칙(계획서 v5.0)으로 뽑혀 **하루 업종이 정확히 10개**다. 종목 단위로 섞으면 그 구조가
#: 깨져 하루 업종이 **16개**로 흩어지고(2026-09-10 실측 · 중앙값), 더미가 늘어 E 보다
#: 자유도를 더 쓴다. 그래서 "자유도만 같고 정보는 없는" 대조군이 되지 못한다.
#: 날짜 안에서 치환하면 하루 업종 분포(10 × 5)가 **정확히** 보존된다.
SHUFFLED_INDUSTRY = "industry_shuffled"
WITHIN_DAY_SHUFFLED_INDUSTRY = "industry_shuffled_within_day"
SHUFFLE_SEED = 20260910

#: 🔵 **탐색(exploratory) 조건.** 채택 후보가 아니다 — 좋아지기를 기대하고 돌리는 것이
#: 아니라, E·F·H 의 손실이 **무엇 때문인지**를 가르려고 돌린다. 그래서 Bonferroni 의
#: N 에 넣지 않는다(임상시험 보고의 primary / exploratory 구분과 같다). 대신 돌린 사실과
#: 결과는 전부 싣고, 이것으로는 채택하지 않는다는 것을 보고서에 못박는다.
#:
#: 2026-09-10 실측이 이 둘을 필요하게 만들었다 — **모든 중립화가 설계행렬의 절편·더미
#: 때문에 그날 평균을 함께 지운다.** "시총 중립화" 로 이름 붙인 F 의 잔차가 그냥 그날
#: 평균을 뺀 값과 상관 0.9566~0.9831 이었고 시총이 더 설명한 몫은 4.46% 뿐이었다.
EXPLORATORY_CONDITIONS: dict[str, dict[str, object] | None] = {
    # D — 그날 평균만 뺀다. 모든 중립화가 공통으로 하는 일 하나만 떼어 낸 조건이다.
    #     F − D 가 시총 축의 몫, E − D 가 업종 축 + 자유도의 몫이다.
    "D_center_only": dict(winsorize=None, zscore="center"),
    # R — 업종을 **종목 단위로** 섞어서 중립화한다. 종목-업종 대응은 유지되지만 하루
    #     업종 수가 10 → 16 으로 늘어 **E 보다 자유도를 더 쓴다**(위 상수 설명). 그래서
    #     이것만으로는 업종 정보의 몫을 못 가른다. 그래도 남긴다 — "자유도를 더 쓰면 더
    #     나빠진다" 를 보여 주는 축이고, R2 와 나란히 놓으면 자유도의 값이 읽힌다.
    "R_neutralize_shuffled_industry": dict(
        winsorize=None, zscore=False,
        neutralize=dict(groups=SHUFFLED_INDUSTRY, controls=()),
    ),
    # R2 — 업종을 **그날 안에서** 치환해 중립화한다. 하루 업종 분포(10 × 5)가 정확히
    #      보존되므로 자유도가 E 와 **똑같다**. 지우는 것은 아무 정보도 아니다.
    #      **자유도를 맞춘 음성 대조군**이라 E − R2 가 업종 정보의 몫이다.
    "R2_neutralize_shuffled_within_day": dict(
        winsorize=None, zscore=False,
        neutralize=dict(groups=WITHIN_DAY_SHUFFLED_INDUSTRY, controls=()),
    ),
}

#: `common` 벌에만 끼우는 09-09 조건. **새 시행이 아니다** — 이미 센 B 를 같은 표본에
#: 두어 ρ 대 손실 그림을 한 자리에서 그리려는 것이고, 09-09 의 −2.81%p 가 재현되는지
#: 보는 대조이기도 하다.
BORROWED = "B_ts_zscore_causal"

#: 🔴 탐색 조건은 **판정 벌에 섞지 않는다.** 공통 표본은 조건 하나가 결측을 만들면 모든
#: 조건의 표본을 함께 줄이므로, 탐색 조건을 판정 벌에 넣으면 "판정은 full 159,900행" 이라는
#: 사전 등록이 조용히 깨질 수 있다. 별도 벌로 두면 판정 벌이 절대 흔들리지 않고, E·F·H 가
#: 두 벌에서 같은 값을 내는지도 함께 확인된다.
SAMPLE_SETS: dict[str, tuple[str, ...]] = {
    "full": tuple(AXIS_CONDITIONS),
    "mechanism": (*AXIS_CONDITIONS, *EXPLORATORY_CONDITIONS),
    "common": (BORROWED, *AXIS_CONDITIONS),
}

#: 판정에 쓰는 벌. 돌리기 전에 정한다 (위 사전 등록).
VERDICT_SAMPLE_SET = "full"

#: 축-라벨 연결을 잴 때 쓰는 최소 행 수들. **문턱을 고르지 않고 여러 개로 재서 다 싣는다** —
#: 하나만 적으면 그 값이 유리해서 고른 것인지 알 수 없다.
LINK_MIN_ROWS = (0, 100, 250, 500)

def with_shuffled_industry(
    dataset: StockModelDataset, *, seed: int = SHUFFLE_SEED
) -> StockModelDataset:
    """업종 라벨을 무작위로 섞은 칸 **둘**을 붙인다 — E 의 음성 대조군.

    E 가 나빠진 것이 (가) 업종 정보를 지워서인지 (나) 더미를 세우느라 자유도를 잃어서인지
    가른다. 섞은 라벨로 중립화하면 지우는 것이 아무 정보도 아니므로 **(나)만** 남는다.

    ## 섞는 방식이 둘인 이유

    | 칸 | 섞는 단위 | 종목-업종 대응 | 하루 업종 수 |
    |---|---|---|---|
    | `industry_shuffled` (R) | 종목 | 유지 (한 종목 = 한 가짜 업종) | 10 → **16** |
    | `industry_shuffled_within_day` (R2) | 그날 안 | 깨짐 (날마다 다름) | **10 그대로** |

    처음에는 R 하나만 두었다. 업종은 종목의 속성이니 종목 단위로 섞는 것이 구조를
    보존한다고 보았기 때문이다. **그런데 자유도가 안 맞았다** — 우리 후보군은 *"업종 시총
    상위 10 × 업종별 상위 5"* 규칙으로 뽑혀 하루 업종이 **정확히 10개**인데, 종목 단위로
    섞으니 그 짝이 흩어져 하루 16개 업종이 되었다(2026-09-10 실측). 더미가 늘면 자유도를
    더 쓰므로 E 와 견줄 수 없다.

    R2 는 **그날 안에서만 치환**한다. 그날 업종 라벨의 다중집합이 그대로라 하루 업종 수가
    10 으로 보존되고, 자유도가 E 와 똑같아진다. 대신 같은 종목이 날마다 다른 가짜 업종에
    들어가는데, 대조군의 목적이 "자유도는 같게 · 정보는 없게" 이므로 그 목적에는 맞는다.

    둘을 다 남긴다 — R2 로 업종 정보의 몫을 재고, R 로 자유도를 더 썼을 때 얼마나 더
    나빠지는지 읽는다.

    시드는 고정한다 — 다시 돌렸을 때 다른 값이 나오면 대조군 구실을 못 한다.
    """
    frame = dataset.frame.copy()
    종목업종 = frame.groupby("code")["industry"].agg(lambda s: s.mode().iat[0])
    rng = np.random.default_rng(seed)
    섞은값 = rng.permutation(종목업종.to_numpy())
    매핑 = dict(zip(종목업종.index, 섞은값, strict=True))
    frame[SHUFFLED_INDUSTRY] = frame["code"].map(매핑)

    # R2 — 그날 안에서만 치환한다. 하루 업종 분포가 그대로라 **자유도가 E 와 같다.**
    # 날짜를 정렬해 도는 이유는 재현 때문이다 — 난수를 순서대로 쓰므로 도는 순서가
    # 달라지면 다른 배정이 나온다.
    값 = frame["industry"].to_numpy().copy()
    자리 = np.arange(len(frame))
    for _, idx in frame.groupby("bas_dd", sort=True).indices.items():
        위치 = 자리[idx]
        값[위치] = rng.permutation(값[위치])
    frame[WITHIN_DAY_SHUFFLED_INDUSTRY] = 값
    return StockModelDataset(frame=frame, feature_columns=dataset.feature_columns)


def with_size_control(dataset: StockModelDataset) -> StockModelDataset:
    """패널에 `log_cap` 칸을 붙인다 — F 의 통제 변수.

    피처 목록은 건드리지 않으므로 **모델에는 들어가지 않는다.** 회귀의 설명변수로만
    쓰인다. 시총 자체(`market_cap`)는 보호 칸이라 그대로는 못 쓴다.
    """
    frame = dataset.frame.copy()
    frame[SIZE_CONTROL] = log_size_column(frame)
    return StockModelDataset(frame=frame, feature_columns=dataset.feature_columns)


def axis_label_link(frame: pd.DataFrame) -> dict[str, object]:
    """각 후보가 **지우는 축**이 라벨과 얼마나 이어져 있는지 잰다 — 돌리기 전에.

    라벨은 절대 ±2% 밴드라 "그 축의 수준이 높으면 밴드를 더 자주 벗어나는가" 가 곧
    "그 축이 라벨을 설명하는가" 다. 변동성 대리로 `hv_20` 을 쓴다 — 전처리 v1.3 §1.1 에서
    `hv_20` 십분위와 비중립 확률의 Spearman ρ 가 1.0000(47.6% → 75.2%)이었다.

    ## 이 값은 대상이 만들 때 쓴 것이 아니다

    중립화는 `hv_20` 을 업종 더미에 회귀할 뿐, 라벨(`label_numeric`)을 보지 않는다.
    그러니 여기서 재는 "축 ↔ 라벨" 은 대상과 독립된 증거다. 만약 라벨에서 파생한 값으로
    라벨과의 상관을 쟀다면 그것은 대조가 아니라 항등식이다.
    """
    비중립 = frame["label_numeric"].ne(0).astype(int)
    work = pd.DataFrame({
        "비중립": 비중립.to_numpy(),
        "hv_20": frame["hv_20"].to_numpy(dtype=float),
        "code": frame["code"].to_numpy(),
        "industry": frame["industry"].to_numpy() if "industry" in frame else None,
        "bas_dd": frame["bas_dd"].to_numpy(),
        SIZE_CONTROL: (
            frame[SIZE_CONTROL].to_numpy(dtype=float)
            if SIZE_CONTROL in frame else np.full(len(frame), np.nan)
        ),
    })

    def _group_link(key: str) -> dict[str, object]:
        g = work.dropna(subset=["hv_20"]).groupby(key)
        t = g.agg(행=("비중립", "size"), 비중립=("비중립", "mean"), hv=("hv_20", "mean"))
        out: dict[str, object] = {}
        for 최소 in LINK_MIN_ROWS:
            keep = t[t["행"] >= 최소] if 최소 else t
            # 못 재면 비워 두고 왜인지 남긴다 — 억지로 숫자를 채우지 않는다. 그룹이 둘
            # 이하면 순위상관이 늘 ±1 이고, 한쪽이 상수면 아예 정의되지 않는다.
            막힘 = (
                "groups_below_3" if len(keep) < 3
                else "constant_input" if keep["hv"].nunique() < 2
                or keep["비중립"].nunique() < 2
                else None
            )
            if 막힘:
                out[f"min_rows_{최소}"] = {
                    "groups": int(len(keep)), "spearman_rho": None, "why_empty": 막힘,
                }
                continue
            r = stats.spearmanr(keep["hv"], keep["비중립"])
            out[f"min_rows_{최소}"] = {
                "groups": int(len(keep)),
                "spearman_rho": float(r.statistic),
                "p_value": float(r.pvalue),
                "nonneutral_min": float(keep["비중립"].min()),
                "nonneutral_max": float(keep["비중립"].max()),
            }
        return out

    링크: dict[str, object] = {
        "note": (
            "각 축의 수준(hv_20 평균)과 그 축의 비중립 비율 사이 Spearman rho. "
            "높을수록 그 축을 지우면 라벨 정보를 지우는 것이다."
        ),
        "stock_axis_erased_by_B": _group_link("code"),
        "industry_axis_erased_by_E": _group_link("industry"),
        "date_axis_erased_by_cross_sectional_z": _group_link("bas_dd"),
    }

    # 시총은 연속 축이라 그룹이 아니라 점이연 상관과 십분위로 잰다.
    ok = np.isfinite(work[SIZE_CONTROL].to_numpy())
    if int(ok.sum()) > 100:
        pb = stats.pointbiserialr(work.loc[ok, "비중립"], work.loc[ok, SIZE_CONTROL])
        십분위 = pd.qcut(work.loc[ok, SIZE_CONTROL], 10, labels=False, duplicates="drop")
        비율 = work.loc[ok].groupby(십분위)["비중립"].mean()
        r = stats.spearmanr(비율.index, 비율)
        링크["size_axis_erased_by_F"] = {
            "point_biserial_r": float(pb.statistic),
            "p_value": float(pb.pvalue),
            "decile_spearman_rho": float(r.statistic),
            "decile_nonneutral": [float(v) for v in 비율],
            "rows": int(ok.sum()),
        }
    링크["rank_erases_everything"] = (
        "H(순위)는 축 하나로 요약되지 않는다 — 그날 안 순서만 남기므로 날짜 수준·업종 "
        "수준·종목 수준이 한꺼번에 사라진다. 위 셋을 동시에 지운다고 읽으면 된다."
    )
    return 링크


def ic_table(
    features: pd.DataFrame, target: pd.Series, dates: pd.Series
) -> pd.DataFrame:
    """날짜별 스피어만 IC 의 평균과 ICIR.

    노트북 `02-품질·전처리/05.횡단면-전처리는-날짜-안에서-끝난다.ipynb` §6.1 의 정의를
    **그대로** 쓴다. 정의가 한 글자라도 다르면 그 노트북의 0.1050 → 0.1144 와 대조가
    되지 않는다.
    """
    tr = target.groupby(dates).rank()
    mt = tr.groupby(dates).transform("mean")
    out: dict[str, dict[str, float]] = {}
    for f in features.columns:
        x = features[f].astype(float)
        ok = x.notna() & target.notna()
        xr = x[ok].groupby(dates[ok]).rank()
        mx = xr.groupby(dates[ok]).transform("mean")
        t_ok, mt_ok = tr[ok], mt[ok]
        num = ((xr - mx) * (t_ok - mt_ok)).groupby(dates[ok]).sum()
        den = np.sqrt(((xr - mx) ** 2).groupby(dates[ok]).sum()
                      * ((t_ok - mt_ok) ** 2).groupby(dates[ok]).sum())
        ic = (num / den).replace([np.inf, -np.inf], np.nan).dropna()
        out[f] = {
            "ic_mean": float(ic.mean()),
            "icir": float(ic.mean() / ic.std()) if ic.std() else float("nan"),
            "dates": int(len(ic)),
        }
    return pd.DataFrame(out).T


def _ic_summary(dataset: StockModelDataset) -> dict[str, float]:
    """조건 하나의 |IC| 평균과 |ICIR| 평균 — 05 노트북과 같은 요약 방식."""
    cols = [c for c in dataset.feature_columns if c in dataset.frame.columns]
    표 = ic_table(
        dataset.frame.loc[:, cols],
        dataset.frame["fwd_return_5d"],
        dataset.frame["bas_dd"],
    )
    return {
        "abs_ic_mean": float(표["ic_mean"].abs().mean()),
        "abs_icir_mean": float(표["icir"].abs().mean()),
        "features": int(len(표)),
    }


def _conditions_for(sample_set: str) -> dict[str, dict[str, object] | None]:
    """그 벌에서 쓸 조건 사전. 빌려 온 B 는 09-09 사전에서 **그대로** 가져온다."""
    전체 = {**AXIS_CONDITIONS, **EXPLORATORY_CONDITIONS,
           BORROWED: CANDIDATE_CONDITIONS[BORROWED]}
    return {RAW: None, **{name: 전체[name] for name in SAMPLE_SETS[sample_set]}}


def run_sample_set(
    sample_set: str, base_dataset: StockModelDataset, *, run_id: str,
    source: dict[str, object],
) -> dict[str, object]:
    """한 벌을 돈다 — 조건 조립 → 공통 표본 → 분할 재생성 → 12폴드."""
    조건사전 = _conditions_for(sample_set)
    돌릴것 = [RAW, *SAMPLE_SETS[sample_set]]
    조건표 = {
        name: (
            base_dataset if name == RAW
            else build_condition_dataset(base_dataset, name, conditions=조건사전)
        )
        for name in 돌릴것
    }
    조건표, 표본보고 = common_sample(조건표)
    원인 = (
        f" · 원인 {표본보고['dropped_by_condition']}" if 표본보고["rows_dropped"] else ""
    )
    print(f"\n── 벌 {sample_set} · 공통 표본 {표본보고['rows_before']:,} → "
          f"{표본보고['rows_after']:,}행 ({표본보고['dropped_ratio']:.2%} 뺌){원인}",
          flush=True)

    # 🔴 자른 뒤 분할을 다시 만든다 — 행 번호가 밀리면 엉뚱한 행이 학습에 들어간다.
    splits = expanding_group_splits(
        조건표[RAW].groups, n_folds=N_FOLDS, min_train=MIN_TRAIN_DATES,
        horizon=VALID_DATES, gap=LABEL_HORIZON, label_horizon=LABEL_HORIZON,
    )

    results: dict[str, StockExperimentResult] = {}
    for name, dataset in 조건표.items():
        print(f"  {name} ({len(dataset.feature_columns)}칸) …", end="", flush=True)
        results[name] = evaluate_stock_models(
            dataset, model_builders={MODEL_NAME: MODEL_BUILDERS[MODEL_NAME]},
            outer_splits=splits,
        )
        요약 = results[name].outer_results
        print(f" Accuracy {요약['accuracy'].mean():.4f} · 기준선 대비 "
              f"{요약['accuracy_minus_training_majority_baseline'].mean():+.4f}", flush=True)
        _append_trials(
            run_id=run_id, condition=f"{sample_set}-{name}", result=results[name],
            source=source, feature_columns=dataset.feature_columns,
            report_path=REPORT_RELATIVE,
            extra_experiment={"sample_set": sample_set, "issue": "preprocessing-v1.4"},
        )

    원값_폴드 = results[RAW].outer_results
    잘린표 = 조건표[RAW].frame
    return {
        "experiment": {
            "combination": COMBINATION,
            "model": MODEL_NAME,
            "rows": int(len(잘린표)),
            "dates": int(잘린표["bas_dd"].nunique()),
            "codes": int(잘린표["code"].nunique()),
            "first_date": str(잘린표["bas_dd"].min()),
            "last_date": str(잘린표["bas_dd"].max()),
            "folds": N_FOLDS,
        },
        "common_sample": 표본보고,
        "conditions": {
            name: {
                **_condition_report(result),
                "feature_count": len(조건표[name].feature_columns),
                "options": 조건사전.get(name),
                "ic": _ic_summary(조건표[name]),
            }
            for name, result in results.items()
        },
        "paired_t_vs_raw": {
            name: _paired_t(result.outer_results, 원값_폴드)
            for name, result in results.items() if name != RAW
        },
    }


def _verdict(보고: dict[str, object], alpha: float) -> dict[str, object]:
    """판정 벌에서 원값을 뜻있게 넘은 조건이 있는지 센다. 규칙은 사전 등록 그대로.

    🔴 **채택 후보만 본다.** 탐색 조건이 우연히 원값을 넘어도 채택하지 않는다 — 그것들은
    "좋아지는가" 를 묻는 조건이 아니라 "무엇 때문에 나빠지는가" 를 가르는 조건이고,
    Bonferroni 의 N 에도 안 들어가 있어 문턱이 그것들을 보호하지 않는다.
    """
    비교 = {
        name: v for name, v in 보고["paired_t_vs_raw"].items() if name in AXIS_CONDITIONS
    }
    넘은것 = [
        name for name, v in 비교.items()
        if v["mean_difference"] > 0 and v["p_value"] < alpha
    ]
    명목1위 = max(비교, key=lambda n: 비교[n]["mean_difference"]) if 비교 else None
    return {
        "sample_set": VERDICT_SAMPLE_SET,
        "bonferroni_alpha": alpha,
        "accepted": 넘은것,
        "nominal_best": 명목1위,
        "nominal_best_difference": (
            비교[명목1위]["mean_difference"] if 명목1위 else None
        ),
        "decision": (
            "채택 없음 — 전처리 없음(원값) 유지" if not 넘은것
            else f"채택 후보: {넘은것}"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", choices=[*SAMPLE_SETS, "both"], default="both")
    parser.add_argument("--plan", action="store_true", help="돌리지 않고 계획만")
    args = parser.parse_args()

    벌목록 = list(SAMPLE_SETS) if args.sample == "both" else [args.sample]
    시행수 = TRIALS_BEFORE + len(AXIS_CONDITIONS)
    alpha = 0.05 / 시행수

    if args.plan:
        print(f"조합 {COMBINATION} · {MODEL_NAME} · {N_FOLDS}폴드")
        print("채택 후보 (Bonferroni N 에 든다)")
        for name, options in AXIS_CONDITIONS.items():
            print(f"  {name:30s} {options}")
        print("탐색 조건 (기제 분해 · N 에 안 든다 · 이것으로 채택하지 않는다)")
        for name, options in EXPLORATORY_CONDITIONS.items():
            print(f"  {name:30s} {options}")
        print(f"\n벌: {벌목록} · 판정은 {VERDICT_SAMPLE_SET}")
        for 벌 in 벌목록:
            print(f"  {벌:8s} raw + {list(SAMPLE_SETS[벌])}")
        print(f"\n시행 N = {시행수} (09-09 까지 {TRIALS_BEFORE} + 이번 "
              f"{len(AXIS_CONDITIONS)}) · Bonferroni α = {alpha:.5f}")
        print(f"보고서: {REPORT_RELATIVE}")
        return 0

    datasets = {
        name: load_stock_model_dataset(features)
        for name, features in STOCK_COMBINATION_FEATURES.items()
    }
    base = with_shuffled_industry(
        with_size_control(align_stock_feature_datasets(datasets)[COMBINATION])
    )
    print(f"조합 {COMBINATION} · {len(base.frame):,}행 · "
          f"{base.frame['code'].nunique()}종 · {base.frame['bas_dd'].nunique()}일")
    섞은칸 = [SHUFFLED_INDUSTRY, WITHIN_DAY_SHUFFLED_INDUSTRY]
    하루업종 = base.frame.groupby("bas_dd")[["industry", *섞은칸]].nunique()
    print(f"하루 업종 수 중앙 — 진짜 {하루업종['industry'].median():.0f}", end="")
    for 칸, 표 in zip(섞은칸, ("R 종목단위", "R2 날짜안"), strict=True):
        같은자리 = float((base.frame["industry"] == base.frame[칸]).mean())
        print(f" · {표} {하루업종[칸].median():.0f} (원래와 같은 자리 {같은자리:.2%})", end="")
    print("  ← 자유도가 같아야 업종 정보의 몫을 가른다")

    링크 = axis_label_link(base.frame)
    업종 = 링크["industry_axis_erased_by_E"]["min_rows_500"]
    시총 = 링크["size_axis_erased_by_F"]
    print(f"돌리기 전 축-라벨 연결 — 업종 ρ = {업종['spearman_rho']:.4f} "
          f"(업종 {업종['groups']}개) · 시총 r = {시총['point_biserial_r']:+.4f}")

    generated_at = datetime.now(timezone.utc).isoformat()
    run_id = "stock-preprocessing-axis-" + generated_at.replace(":", "").replace("-", "")
    source = {
        "repo": "qurious-quant/alphastack-krx-dev",
        "daily_path": "full/daily_price_dev.parquet",
        "daily_sha256": _sha256(DAILY_PATH),
        "index_path": "full/index_price_dev.parquet",
        "index_sha256": _sha256(INDEX_PATH),
    }
    벌보고 = {
        벌: run_sample_set(벌, base, run_id=run_id, source=source) for 벌 in 벌목록
    }

    report = {
        "generated_at_utc": generated_at,
        "run_id": run_id,
        "source": source,
        "preregistration": {
            "candidates": list(AXIS_CONDITIONS),
            "exploratory": list(EXPLORATORY_CONDITIONS),
            "exploratory_note": (
                "탐색 조건은 채택 후보가 아니다 — E·F·H 의 손실이 무엇 때문인지 가르려고 "
                "돌린다. Bonferroni 의 N 에 넣지 않으므로 이것으로는 채택하지 않는다. "
                "D 는 그날 평균만 빼고, R 은 업종을 종목 단위로 섞어 중립화한다"
                f"(시드 {SHUFFLE_SEED})."
            ),
            "acceptance": "ADR 0007 — 기준선 대비 accuracy → macro F1 → 승리 폴드",
            "must_beat": "이번 실행의 원값(raw) — 같은 공통 표본에서",
            "verdict_sample_set": VERDICT_SAMPLE_SET,
            "trials_before": TRIALS_BEFORE,
            "trials_total": 시행수,
            "bonferroni_alpha": alpha,
            "ic_is_not_the_axis": (
                "IC·ICIR 은 05 노트북과의 대조로만 싣는다. 판정 축은 accuracy 다."
            ),
            "borrowed_condition": (
                f"{BORROWED} 는 09-09 에 이미 센 시행이라 N 에 다시 세지 않는다. "
                "common 벌의 표본을 09-09 와 같게 만들려고 끼운다."
            ),
        },
        "axis_label_link": 링크,
        "sample_sets": 벌보고,
    }
    if VERDICT_SAMPLE_SET in 벌보고:
        report["verdict"] = _verdict(벌보고[VERDICT_SAMPLE_SET], alpha)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(f"\n결과 저장: {REPORT_RELATIVE}")

    for 벌, 보고 in 벌보고.items():
        표 = []
        for name, rep in 보고["conditions"].items():
            s = rep["model_summary"]
            비교 = 보고["paired_t_vs_raw"].get(name)
            표.append({
                "조건": name,
                "종류": (
                    "후보" if name in AXIS_CONDITIONS
                    else "탐색" if name in EXPLORATORY_CONDITIONS
                    else "기준" if name == RAW else "빌림"
                ),
                "기준선대비": round(float(s["accuracy_minus_training_majority_baseline"]), 4),
                "원값대비": round(비교["mean_difference"], 4) if 비교 else None,
                # 09-09 v1.3 §9 표와 같은 열이다 — **원값**을 이긴 폴드 수. 기준선(다수
                # 클래스)을 이긴 폴드 수와 다르므로 둘을 함께 적는다.
                "원값이김": f"{비교['wins']}/{비교['folds']}" if 비교 else "—",
                "기준선이김": f"{int(s['baseline_win_folds'])}/{int(s['folds'])}",
                "p": round(비교["p_value"], 4) if 비교 else None,
                "|IC|": round(rep["ic"]["abs_ic_mean"], 4),
                "|ICIR|": round(rep["ic"]["abs_icir_mean"], 4),
            })
        print(f"\n── 벌 {벌} · {보고['experiment']['rows']:,}행 ──")
        print(pd.DataFrame(표).to_string(index=False))

    print(f"\nBonferroni(N={시행수}) 문턱 α = {alpha:.5f}")
    if "verdict" in report:
        print(f"판정({VERDICT_SAMPLE_SET}): {report['verdict']['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
