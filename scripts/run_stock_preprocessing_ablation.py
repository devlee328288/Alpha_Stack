"""조합 K에서 횡단면 winsorize+z-score의 효과만 12폴드로 분리해 비교한다."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.walk_forward import expanding_group_splits  # noqa: E402
from features.preprocessing import preprocess_cross_section  # noqa: E402
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
    _model_summary,
    _oos_diagnostics,
    _sha256,
    load_stock_model_dataset,
)

REPORT_PATH = ROOT / "reports" / "stock_preprocessing_ablation.json"
COMBINATION = "K"
MODEL_NAME = "LogisticRegression"
RAW = "raw"
WINSOR_ZSCORE = "cross_section_winsor_mad_zscore"


def build_preprocessed_dataset(dataset: StockModelDataset) -> StockModelDataset:
    """키·라벨·행은 고정하고 조합 K 입력 피처에만 기본 횡단면 전처리를 적용한다."""

    frame = dataset.frame.copy()
    transformed = preprocess_cross_section(frame, dataset.feature_columns)
    if transformed.shape != dataset.x.shape or not transformed.index.equals(frame.index):
        raise RuntimeError("횡단면 전처리가 조합 K의 행 또는 인덱스를 바꿨습니다.")
    if not np.isfinite(transformed.to_numpy(dtype=float)).all():
        raise RuntimeError("횡단면 전처리 뒤 조합 K 피처에 결측 또는 무한대가 생겼습니다.")
    frame.loc[:, list(dataset.feature_columns)] = transformed.to_numpy(dtype=float)
    return StockModelDataset(frame=frame, feature_columns=dataset.feature_columns)


def _append_trials(
    *,
    run_id: str,
    condition: str,
    result: StockExperimentResult,
    source: dict[str, object],
) -> None:
    """실제로 끝난 전처리 ablation의 내부·외부 fit을 시행 원장에 추가한다."""

    records: list[dict[str, object]] = []
    common = {
        "schema_version": 1,
        "status": "success",
        "track": "stock",
        "run": run_id,
        "report_path": "reports/stock_preprocessing_ablation.json",
        "dataset_source": source,
        "experiment": {
            "combination": COMBINATION,
            "model": MODEL_NAME,
            "condition": condition,
            "feature_columns": list(STOCK_COMBINATION_FEATURES[COMBINATION]),
            "label": "T+1_adj_open_to_T+6_adj_open_absolute_2pct_band",
        },
        "audit": {"origin": "direct_execution", "coverage": "all_completed_fits"},
    }
    for row in result.inner_results.to_dict(orient="records"):
        weight = row["class_weight"]
        weight_id = "none" if weight is None else str(weight)
        records.append(
            {
                **common,
                "trial_id": (
                    f"{run_id}-{condition}-fold-{int(row['fold']):02d}-inner-{weight_id}"
                ),
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
            }
        )
    for row in result.outer_results.to_dict(orient="records"):
        weight = row["selected_class_weight"]
        records.append(
            {
                **common,
                "trial_id": f"{run_id}-{condition}-fold-{int(row['fold']):02d}-outer",
                "fit": {
                    "phase": "outer_oos_evaluation",
                    "fold": int(row["fold"]),
                    "class_weight": weight,
                    "train_rows": int(row["train_rows"]),
                    "valid_rows": int(row["valid_rows"]),
                    "train_end": row["train_end"],
                    "valid_start": row["valid_start"],
                    "valid_end": row["valid_end"],
                },
                "metrics": {
                    key: row[key]
                    for key in (
                        "accuracy",
                        "macro_f1",
                        "down_recall",
                        "core_harmonic_mean",
                        "balanced_accuracy",
                        "mcc",
                        "pr_auc_macro_ovr",
                        "accuracy_minus_training_majority_baseline",
                    )
                },
            }
        )
    TRIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRIALS_PATH.open("a", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")


def _condition_report(result: StockExperimentResult) -> dict[str, object]:
    summary = _model_summary(result.outer_results)[0]
    return {
        "model_summary": summary,
        "oos_diagnostics": _oos_diagnostics(result.oos_predictions),
        "selected_class_weight_counts": (
            result.outer_results["selected_class_weight"]
            .fillna("None")
            .value_counts()
            .to_dict()
        ),
        "outer_fold_results": result.outer_results.to_dict(orient="records"),
        "inner_trial_count": int(len(result.inner_results)),
        "outer_fit_count": int(len(result.outer_results)),
    }


def main() -> None:
    """A~K 공통 표본·같은 분할에서 전처리만 바꿔 LogisticRegression을 비교한다."""

    datasets = {
        name: load_stock_model_dataset(features)
        for name, features in STOCK_COMBINATION_FEATURES.items()
    }
    raw_dataset = align_stock_feature_datasets(datasets)[COMBINATION]
    preprocessed_dataset = build_preprocessed_dataset(raw_dataset)
    splits = expanding_group_splits(
        raw_dataset.groups,
        n_folds=N_FOLDS,
        min_train=MIN_TRAIN_DATES,
        horizon=VALID_DATES,
        gap=LABEL_HORIZON,
        label_horizon=LABEL_HORIZON,
    )
    builder = {MODEL_NAME: MODEL_BUILDERS[MODEL_NAME]}
    results: dict[str, StockExperimentResult] = {}
    for condition, dataset in (
        (RAW, raw_dataset),
        (WINSOR_ZSCORE, preprocessed_dataset),
    ):
        print(f"조합 K LogisticRegression · {condition} 시작", flush=True)
        results[condition] = evaluate_stock_models(
            dataset,
            model_builders=builder,
            outer_splits=splits,
        )
        score = results[condition].outer_results["accuracy"].mean()
        print(f"조합 K LogisticRegression · {condition} 완료 · Accuracy {score:.4f}")

    generated_at = datetime.now(timezone.utc).isoformat()
    run_id = "stock-preprocessing-ablation-" + generated_at.replace(":", "").replace("-", "")
    source = {
        "repo": "qurious-quant/alphastack-krx-dev",
        "daily_path": "full/daily_price_dev.parquet",
        "daily_sha256": _sha256(DAILY_PATH),
        "index_path": "full/index_price_dev.parquet",
        "index_sha256": _sha256(INDEX_PATH),
    }
    condition_reports = {
        condition: _condition_report(result) for condition, result in results.items()
    }
    for condition, result in results.items():
        _append_trials(run_id=run_id, condition=condition, result=result, source=source)

    raw_summary = condition_reports[RAW]["model_summary"]
    processed_summary = condition_reports[WINSOR_ZSCORE]["model_summary"]
    comparison_metrics = (
        "accuracy",
        "macro_f1",
        "down_recall",
        "core_harmonic_mean",
        "balanced_accuracy",
        "mcc",
        "pr_auc_macro_ovr",
        "accuracy_minus_training_majority_baseline",
    )
    report = {
        "generated_at_utc": generated_at,
        "run_id": run_id,
        "issue": 197,
        "source": source,
        "experiment": {
            "combination": COMBINATION,
            "model": MODEL_NAME,
            "rows": int(len(raw_dataset.frame)),
            "dates": int(raw_dataset.frame["bas_dd"].nunique()),
            "first_date": str(raw_dataset.frame["bas_dd"].min()),
            "last_date": str(raw_dataset.frame["bas_dd"].max()),
            "features": list(raw_dataset.feature_columns),
            "fixed": ["rows", "labels", "features", "model", "outer_splits"],
            "changed_only": "cross-sectional MAD winsorize followed by z-score",
            "label_policy": "절대 ±2% 유지; 횡단면 라벨·순위·업종 중립화 미사용",
            "folds": 12,
            "minimum_initial_train_dates": 750,
            "valid_dates_per_fold": 60,
            "gap_dates": 5,
        },
        "conditions": condition_reports,
        "processed_minus_raw": {
            metric: float(processed_summary[metric] - raw_summary[metric])
            for metric in comparison_metrics
        },
        "decision": "이 보고서는 제한적 ablation이며 최종 모델 선정 기준을 자동 변경하지 않는다.",
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(f"결과 저장: {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
