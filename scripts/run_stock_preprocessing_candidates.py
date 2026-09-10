"""라벨에 맞는 전처리 후보 넷(A~D)을 원값·#210 과 같은 자리에서 잰다 (이슈 #214).

## 왜 이 스크립트가 따로 있나

`run_stock_preprocessing_ablation.py`(오준영 님)가 원값과 `winsorize+z-score` 둘을
이미 쟀고(#210), 그 결과가 **기준선 대비 정확도 +2.43%p → +0.84%p** 였다. 떨어진 이유가
*"우리 라벨이 절대 ±2% 밴드라 변동성 **수준**이 라벨의 절반을 설명하는데, 날짜별 z 가 그
수준을 지운다"* 라는 것이 #214 의 진단이고, 그 진단에서 나온 대안 넷이 여기 있다.

조건을 오준영 님 파일에 끼우는 것이 원래 계획이었지만(#214 여쭙는 것 1), 그 결정이 아직
안 났다. **남의 파일을 말없이 고치지 않기 위해** 조건 등록만 이쪽에 두고, 실험·보고
형식은 그 파일의 함수를 그대로 가져다 쓴다. 결정이 나면 이 조건 사전을 그쪽으로 옮기고
이 파일을 지우면 된다 — 그때 결과가 달라지지 않도록 **분할·표본·모델·라벨을 전부 같은
경로에서 가져온다.**

## 후보 다섯

    A   winsorize_only              날짜별 MAD 3배만 · z 없음
    A'  winsorize_non_vol           A 에서 변동성 축만 뺀 것  ← 조건부 후보 (아래)
    B   ts_zscore_causal            종목별 250행 인과적 z (준비 60)
    C   cs_zscore_plus_date_level   v1.1 z + 지운 날짜 수준을 피처로 복원
    D   selective_vol_raw           변동성 축 원값 · 나머지만 횡단면

## 사전 등록 — 결과를 보기 전에 정한 것

- **채택 기준**: ADR 0007 그대로. 기준선 대비 Accuracy → Macro F1 → 승리 폴드.
  **원값을 넘어야 채택**한다. 넘는 것이 없으면 "전처리 없음" 을 유지한다.
- **시행**: #210 의 둘 + 여기서 돌린 조건 수. 최고가 원값을 넘어도 **Bonferroni 문턱과
  12폴드 대응표본 t** 를 함께 적는다. *"이겼다"* 가 아니라 *"몇 개 중 이겼고 p 가 얼마"* 로.
- **한 번에 하나만**: 라벨·피처 목록·모델·폴드·표본을 전부 고정하고 전처리 조건만 바꾼다.
- 한 번에 다 돌린다. 신장환 님 의견(#214) — A·B 를 먼저 보고 C·D 를 정하면
  *"데이터를 보고 시행 범위를 정한 것"* 이라 N 을 정직하게 적을 수 없다.
- **A′ 는 조건부로 미리 적어 둔 후보다.** #214 §3 에 *"A 가 원값보다 나쁘면 winsorize
  자체가 문제(극단 변동성 = 사건)라 변동성 축을 뺀 A′ 로 간다"* 라고 썼고, 첫 실행에서
  A 가 원값에 0.33%p 뒤져 그 조건이 걸렸다. 결과를 보고 만든 후보가 아니라 **조건이
  결과로 충족된** 후보이며, 그래도 시행 수에는 똑같이 한 자리로 센다.

## 🔴 두 가지를 실측이 뒤집었다 — 적어 둔다

**① B 는 표본을 줄인다.** v1.2 명세서에 *"준비구간 60 은 `hv_regime` 269 안이라 표본이 안
줄어든다"* 라고 적었는데 **틀렸다.** 이 패널은 전 종목 일별 표가 아니라 **후보군 패널**이라
종목마다 행이 드문드문 있고, 창은 달력이 아니라 **행**을 센다. 그래서 종목마다 앞 59행이
NaN 이 되고 전체의 5.2%(8,345행)가 빈다.

**② 그래서 공통 표본을 쓴다.** 조건마다 표본이 다르면 전처리 효과가 아니라 표본 차이를
재게 된다. 모든 조건에서 값이 있는 행만 남기고, **원값도 같이 잘라** 짝지어 비교한다.
잘린 행 수는 보고서 `common_sample` 에 그대로 실린다.

## C 만 피처 개수가 다르다 — 그것을 숨기지 않는다

C 는 날짜 수준을 되돌리므로 피처가 14 → 22 로 는다(변동성 축 4칸 × 평균·표준편차).
"피처가 늘어서 좋아졌나, 수준을 되돌려서 좋아졌나" 가 섞이므로, 보고서에 C 의
`feature_count` 를 따로 싣는다. 나머지는 14칸 그대로다.

    python scripts/run_stock_preprocessing_candidates.py             # 전부
    python scripts/run_stock_preprocessing_candidates.py --only A B  # 골라서
    python scripts/run_stock_preprocessing_candidates.py --plan      # 무엇을 돌릴지만
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.walk_forward import expanding_group_splits  # noqa: E402
from features.preprocessing import (  # noqa: E402
    VOLATILITY_AXIS,
    date_level_columns,
    preprocess_cross_section,
    restore_date_level,
)
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
    TRIALS_PATH,
    _json_default,
    _sha256,
    load_stock_model_dataset,
)

# 🔴 #210 과 **같은 자리**에서 재려고 그쪽 상수·보고 함수를 그대로 가져온다.
#    여기서 다시 정의하면 조합이나 모델이 조용히 갈라질 수 있다.
from scripts.run_stock_preprocessing_ablation import (  # noqa: E402
    COMBINATION,
    MODEL_NAME,
    RAW,
    _condition_report,
)

REPORT_PATH = ROOT / "reports" / "stock_preprocessing_candidates.json"

#: #210 이 이미 잰 두 조건. 여기서는 다시 돌리지 않고 그 보고서에서 읽어 비교 기준으로만 쓴다.
BASELINE_REPORT_PATH = ROOT / "reports" / "stock_preprocessing_ablation.json"

#: 조건 사전. 값은 `preprocess_cross_section` 의 키워드 인자 그대로다.
#: `None` 은 "전처리를 걸지 않는다".
#:
#: 🔴 키를 **전부 명시**한다. `preprocess_cross_section` 의 기본값이
#:    `winsorize="mad"`, `zscore=True` 라서, B 처럼 시계열만 쓰고 싶은 조건에서
#:    키를 생략하면 **횡단면 z 가 조용히 함께 걸린다.**
CONDITIONS: dict[str, dict[str, object] | None] = {
    "A_winsorize_only": dict(winsorize="mad", zscore=False),
    # A′ — **사전 등록된 조건부 후보**다. "A 가 원값보다 나쁘면 winsorize 자체가 문제(극단
    # 변동성 = 사건)이니 변동성 축을 빼고 다시 잰다" 를 #214 §3 에 미리 적어 두었고,
    # 2026-09-09 실측에서 A 가 원값에 0.33%p 뒤져 그 조건이 실제로 걸렸다.
    "Aprime_winsorize_non_vol": dict(
        winsorize="mad", zscore=False, keep_raw=VOLATILITY_AXIS
    ),
    "B_ts_zscore_causal": dict(
        winsorize=None, zscore=False, time_series=dict(window=250, min_periods=60)
    ),
    "C_cs_zscore_plus_date_level": dict(winsorize="mad", zscore=True),
    "D_selective_vol_raw": dict(winsorize="mad", zscore=True, keep_raw=VOLATILITY_AXIS),
}

#: C 만 칸이 는다. 되돌리는 축은 변동성 넷이고, 지워지는 통계가 평균과 산포 둘이라 8칸이다.
DATE_LEVEL_STATS = ("mean", "std")

_별칭 = {"A": "A_winsorize_only", "A'": "Aprime_winsorize_non_vol",
        "AP": "Aprime_winsorize_non_vol", "B": "B_ts_zscore_causal",
        "C": "C_cs_zscore_plus_date_level", "D": "D_selective_vol_raw"}


def build_condition_dataset(
    dataset: StockModelDataset,
    condition: str,
    *,
    conditions: Mapping[str, dict[str, object] | None] | None = None,
) -> StockModelDataset:
    """조건 하나를 조합 K 피처에만 건다. 키·라벨·행은 고정한다.

    `conditions` 를 주면 그 사전에서 조건을 찾는다. 생략하면 이 파일의 `CONDITIONS` 다 —
    **A~D 를 돌리는 기존 동작은 그대로**이고, 다음 판의 러너
    (`run_stock_preprocessing_axis.py`)가 자기 조건 사전을 끼울 자리만 열어 둔 것이다.
    조립 순서·검증·결측 처리를 두 번 쓰면 두 판의 숫자가 조용히 갈라진다.

    C 는 날짜 수준 8칸을 **덧붙이므로** 피처 목록이 함께 늘어난다. 나머지는 칸 이름·개수가
    그대로다.

    ## 결측을 여기서 막지 않고 뒤로 넘긴다

    B 는 준비구간(`min_periods`)만큼 앞머리가 NaN 이다 — 그것이 인과성의 대가다. 예전에는
    여기서 예외를 세웠는데, 그러면 B 를 아예 못 잰다. 대신 **모든 조건에서 값이 있는 행**만
    남기는 공통 표본을 `main()` 이 만든다. 이 함수는 만든 것을 그대로 돌려주고, 어디에
    구멍이 났는지는 그쪽에서 센다.
    """
    options = (CONDITIONS if conditions is None else conditions)[condition]
    frame = dataset.frame.copy()
    columns = list(dataset.feature_columns)
    if options is None:
        return StockModelDataset(frame=frame, feature_columns=tuple(columns))

    transformed = preprocess_cross_section(frame, columns, **options)
    if transformed.shape != dataset.x.shape or not transformed.index.equals(frame.index):
        raise RuntimeError(f"{condition}: 전처리가 조합 K 의 행 또는 인덱스를 바꿨습니다.")
    frame.loc[:, columns] = transformed.to_numpy(dtype=float)

    if condition == "C_cs_zscore_plus_date_level":
        축 = [c for c in VOLATILITY_AXIS if c in columns]
        수준 = restore_date_level(dataset.frame, 축, stats=DATE_LEVEL_STATS)
        frame = pd.concat([frame, 수준], axis=1)
        columns += date_level_columns(축, DATE_LEVEL_STATS)
    return StockModelDataset(frame=frame, feature_columns=tuple(columns))


def _finite_mask(dataset: StockModelDataset) -> np.ndarray:
    """그 조건에서 **모든 피처가 유한한** 행."""
    values = dataset.frame.loc[:, list(dataset.feature_columns)].to_numpy(dtype=float)
    return np.isfinite(values).all(axis=1)


def common_sample(
    datasets: dict[str, StockModelDataset],
) -> tuple[dict[str, StockModelDataset], dict[str, object]]:
    """모든 조건에서 값이 있는 행만 남긴다. **표본 차이가 성능 차이로 새지 않게.**

    ## 왜 필요한가

    조건마다 결측이 다르면 "전처리 효과" 가 아니라 "누구를 뺐나" 를 재게 된다. B 는
    종목마다 앞 `min_periods − 1` 행이 NaN 이고(2026-09-09 실측 · 8,345행 · 전체의 5.2%),
    나머지 조건은 구멍이 없다. 그대로 두면 B 만 다른 표본으로 채점된다.

    ## 무엇을 잃는가 — 숨기지 않는다

    빠지는 것은 **각 종목이 이 패널에 나타난 첫 59행**이라 이른 폴드에 몰린다. 그래서
    돌려주는 보고에 조건별 결측 행 수와 잘린 뒤 폴드별 행 수를 함께 싣는다. #210 의
    원값 숫자(159,900행)와 직접 비교하지 말고, **같이 잘린 원값**과 비교해야 한다.
    """
    masks = {name: _finite_mask(ds) for name, ds in datasets.items()}
    공통 = np.logical_and.reduce(list(masks.values()))
    남김 = {
        name: StockModelDataset(
            frame=ds.frame.loc[공통].reset_index(drop=True),
            feature_columns=ds.feature_columns,
        )
        for name, ds in datasets.items()
    }
    보고 = {
        "rows_before": int(len(공통)),
        "rows_after": int(공통.sum()),
        "rows_dropped": int((~공통).sum()),
        "dropped_ratio": float((~공통).mean()),
        "dropped_by_condition": {
            name: int((~m).sum()) for name, m in masks.items() if not m.all()
        },
    }
    return 남김, 보고


def _append_trials(
    *, run_id: str, condition: str, result: StockExperimentResult,
    source: dict[str, object], feature_columns: tuple[str, ...],
    report_path: str = "reports/stock_preprocessing_candidates.json",
    extra_experiment: Mapping[str, object] | None = None,
) -> None:
    """실제로 끝난 fit 을 시행 원장에 남긴다. 사전 등록의 '시행 6' 을 세는 근거다.

    `report_path` 와 `extra_experiment` 는 다음 판의 러너가 자기 보고서를 가리키고
    자기 표본 이름(`sample_set`)을 남기려고 열어 둔 자리다. 원장 형식을 두 번 쓰면
    나중에 시행을 셀 때 두 판이 같은 자로 세어지지 않는다.
    """
    common = {
        "schema_version": 1,
        "status": "success",
        "track": "stock",
        "run": run_id,
        "report_path": report_path,
        "dataset_source": source,
        "experiment": {
            "combination": COMBINATION,
            "model": MODEL_NAME,
            "condition": condition,
            "feature_columns": list(feature_columns),
            "label": "T+1_adj_open_to_T+6_adj_open_absolute_2pct_band",
            **(extra_experiment or {}),
        },
        "audit": {"origin": "direct_execution", "coverage": "all_completed_fits"},
    }
    records: list[dict[str, object]] = []
    for row in result.inner_results.to_dict(orient="records"):
        weight = row["class_weight"]
        weight_id = "none" if weight is None else str(weight)
        records.append({
            **common,
            "trial_id": f"{run_id}-{condition}-fold-{int(row['fold']):02d}-inner-{weight_id}",
            "fit": {
                "phase": "inner_class_weight_selection",
                "fold": int(row["fold"]),
                "class_weight": weight,
                "train_rows": int(row["inner_train_rows"]),
                "valid_rows": int(row["inner_valid_rows"]),
            },
            "metrics": {
                key: row[key]
                for key in ("accuracy", "macro_f1", "down_recall", "core_harmonic_mean")
            },
        })
    for row in result.outer_results.to_dict(orient="records"):
        records.append({
            **common,
            "trial_id": f"{run_id}-{condition}-fold-{int(row['fold']):02d}-outer",
            "fit": {
                "phase": "outer_oos_evaluation",
                "fold": int(row["fold"]),
                "class_weight": row["selected_class_weight"],
                "train_rows": int(row["train_rows"]),
                "valid_rows": int(row["valid_rows"]),
                "train_end": row["train_end"],
                "valid_start": row["valid_start"],
                "valid_end": row["valid_end"],
            },
            "metrics": {
                key: row[key]
                for key in (
                    "accuracy", "macro_f1", "down_recall", "core_harmonic_mean",
                    "balanced_accuracy", "mcc", "pr_auc_macro_ovr",
                    "accuracy_minus_training_majority_baseline",
                )
            },
        })
    TRIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRIALS_PATH.open("a", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")


def _paired_t(조건_폴드: pd.DataFrame, 원값_폴드: pd.DataFrame) -> dict[str, object]:
    """같은 12폴드에서 조건과 원값을 **짝지어** 비교한다.

    폴드마다 표본·기간이 다르므로 독립 표본 t 는 맞지 않는다. 같은 폴드의 두 값이
    한 쌍이다. 지표는 기준선 대비 accuracy — ADR 0007 의 1순위 축이다.
    """
    칸 = "accuracy_minus_training_majority_baseline"
    a = 조건_폴드.sort_values("fold")[칸].to_numpy(dtype=float)
    b = 원값_폴드.sort_values("fold")[칸].to_numpy(dtype=float)
    if len(a) != len(b):
        raise RuntimeError(f"폴드 수가 다릅니다: {len(a)} vs {len(b)}")
    차 = a - b
    result = stats.ttest_rel(a, b)
    return {
        "metric": 칸,
        "folds": int(len(a)),
        "mean_difference": float(차.mean()),
        "std_difference": float(차.std(ddof=1)),
        "wins": int((차 > 0).sum()),
        "t_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def _baseline_conditions() -> dict[str, dict[str, object]]:
    """#210 보고서에서 원값·winsor+z 요약을 읽는다. 없으면 빈 사전."""
    if not BASELINE_REPORT_PATH.exists():
        return {}
    report = json.loads(BASELINE_REPORT_PATH.read_text(encoding="utf-8"))
    return report.get("conditions", {})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=[],
                        help="A A' B C D 또는 조건 이름. 생략하면 전부")
    parser.add_argument("--plan", action="store_true", help="돌리지 않고 계획만 보여 준다")
    args = parser.parse_args()

    고른것 = [_별칭.get(k, k) for k in args.only] or list(CONDITIONS)
    모르는 = [k for k in 고른것 if k not in CONDITIONS]
    if 모르는:
        print(f"🔴 모르는 조건: {모르는}\n   쓸 수 있는 것: {list(CONDITIONS)} (A~D 로도 됩니다)")
        return 2

    if args.plan:
        print(f"조합 {COMBINATION} · {MODEL_NAME} · {N_FOLDS}폴드")
        for name in 고른것:
            print(f"  {name:32s} {CONDITIONS[name]}")
        print(f"\n보고서: {REPORT_PATH.relative_to(ROOT)}")
        print("비교 기준: #210 의 원값 (기준선 대비 +2.43%p · 10/12) — 넘어야 채택")
        return 0

    datasets = {
        name: load_stock_model_dataset(features)
        for name, features in STOCK_COMBINATION_FEATURES.items()
    }
    raw_dataset = align_stock_feature_datasets(datasets)[COMBINATION]
    builder = {MODEL_NAME: MODEL_BUILDERS[MODEL_NAME]}

    # 원값을 여기서도 다시 돌린다 — #210 보고서의 숫자를 그대로 믿지 않고,
    # **이번 실행의 원값**과 짝지어 비교하기 위해서다. 두 값이 어긋나면 그것 자체가 신호다.
    돌릴것 = [RAW, *고른것]
    조건표 = {
        condition: (
            raw_dataset if condition == RAW
            else build_condition_dataset(raw_dataset, condition)
        )
        for condition in 돌릴것
    }
    조건표, 표본보고 = common_sample(조건표)
    if 표본보고["rows_dropped"]:
        print(f"공통 표본: {표본보고['rows_before']:,} → {표본보고['rows_after']:,}행 "
              f"({표본보고['dropped_ratio']:.2%} 뺌) · 원인 {표본보고['dropped_by_condition']}",
              flush=True)

    # 🔴 분할은 **자른 뒤** 다시 만든다. 자르기 전 분할을 그대로 쓰면 행 번호가 밀려
    #    엉뚱한 행이 학습·검증에 들어간다.
    splits = expanding_group_splits(
        조건표[RAW].groups,
        n_folds=N_FOLDS,
        min_train=MIN_TRAIN_DATES,
        horizon=VALID_DATES,
        gap=LABEL_HORIZON,
        label_horizon=LABEL_HORIZON,
    )

    results: dict[str, StockExperimentResult] = {}
    features_used: dict[str, tuple[str, ...]] = {}
    for condition, dataset in 조건표.items():
        features_used[condition] = dataset.feature_columns
        print(f"조합 {COMBINATION} {MODEL_NAME} · {condition} "
              f"({len(dataset.feature_columns)}칸) 시작", flush=True)
        results[condition] = evaluate_stock_models(
            dataset, model_builders=builder, outer_splits=splits
        )
        요약 = results[condition].outer_results
        print(f"  → Accuracy {요약['accuracy'].mean():.4f} · 기준선 대비 "
              f"{요약['accuracy_minus_training_majority_baseline'].mean():+.4f}", flush=True)

    generated_at = datetime.now(timezone.utc).isoformat()
    run_id = "stock-preprocessing-candidates-" + generated_at.replace(":", "").replace("-", "")
    source = {
        "repo": "qurious-quant/alphastack-krx-dev",
        "daily_path": "full/daily_price_dev.parquet",
        "daily_sha256": _sha256(DAILY_PATH),
        "index_path": "full/index_price_dev.parquet",
        "index_sha256": _sha256(INDEX_PATH),
    }
    condition_reports = {
        name: {
            **_condition_report(result),
            "feature_count": len(features_used[name]),
            "feature_columns": list(features_used[name]),
            "options": CONDITIONS.get(name),
        }
        for name, result in results.items()
    }
    for name, result in results.items():
        _append_trials(run_id=run_id, condition=name, result=result,
                       source=source, feature_columns=features_used[name])

    원값_폴드 = results[RAW].outer_results
    비교 = {
        name: _paired_t(result.outer_results, 원값_폴드)
        for name, result in results.items() if name != RAW
    }
    # #210 의 둘 + 이번에 실제로 돌린 조건(원값 재실행 제외). A′ 가 걸리면 자연히 7 이 된다.
    시행수 = 2 + len([n for n in results if n != RAW])
    잘린표 = 조건표[RAW].frame
    report = {
        "generated_at_utc": generated_at,
        "run_id": run_id,
        "issue": 214,
        "source": source,
        "experiment": {
            "combination": COMBINATION,
            "model": MODEL_NAME,
            "rows": int(len(잘린표)),
            "rows_before_common_sample": int(len(raw_dataset.frame)),
            "dates": int(잘린표["bas_dd"].nunique()),
            "first_date": str(잘린표["bas_dd"].min()),
            "last_date": str(잘린표["bas_dd"].max()),
            "fixed": ["rows", "labels", "model", "outer_splits"],
            "changed_only": "preprocessing condition",
            "label_policy": "절대 ±2% 유지 · 라벨·선정 기준(ADR 0006·0007) 그대로",
            "folds": N_FOLDS,
            "minimum_initial_train_dates": MIN_TRAIN_DATES,
            "valid_dates_per_fold": VALID_DATES,
            "gap_dates": LABEL_HORIZON,
        },
        "common_sample": 표본보고,
        "preregistration": {
            "acceptance": "ADR 0007 — 기준선 대비 accuracy → macro F1 → 승리 폴드",
            "must_beat": "이번 실행의 원값(raw) — 같은 공통 표본에서",
            "trials_total": 시행수,
            "bonferroni_alpha": 0.05 / 시행수,
            "note": (
                "#210 의 두 조건과 이번 조건들을 한 가족으로 센다. "
                "A′ 는 'A 가 원값보다 나쁘면' 이라는 조건으로 #214 에 미리 적힌 후보다."
            ),
        },
        "conditions": condition_reports,
        "paired_t_vs_raw": 비교,
        "prior_run_210": _baseline_conditions(),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(f"\n결과 저장: {REPORT_PATH.relative_to(ROOT)}")

    표 = []
    for name, rep in condition_reports.items():
        s = rep["model_summary"]
        표.append({
            "조건": name,
            "칸": rep["feature_count"],
            "accuracy": round(float(s["accuracy"]), 4),
            "기준선대비": round(float(s["accuracy_minus_training_majority_baseline"]), 4),
            "macro_f1": round(float(s["macro_f1"]), 4),
            "승리폴드": int(s["baseline_win_folds"]),
            "p(원값대비)": (round(비교[name]["p_value"], 4) if name in 비교 else None),
        })
    print(pd.DataFrame(표).to_string(index=False))
    print(f"\nBonferroni(N={시행수}) 문턱 α = {0.05 / 시행수:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
