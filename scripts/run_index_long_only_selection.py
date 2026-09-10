"""KOSPI200 기존 100개 후보를 long-only 상승 목적에 맞춰 다시 평가한다.

기존 ``model_sweep.json``은 후보 목록과 과거 요약값만 제공한다. 상승 임계값을 전체
OOS에서 고르는 누수를 피하기 위해 각 후보를 다시 학습하고, 외부 폴드의 학습구간 안에
있는 마지막 60거래일에서만 class weight와 상승 임계값을 선택한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.walk_forward import expanding_splits  # noqa: E402
from features.model_dataset import ModelDataset, build_model_dataset  # noqa: E402
from models.experiment import (  # noqa: E402
    LABEL_HORIZON,
    MIN_TRAIN_SIZE,
    N_FOLDS,
    VALID_SIZE,
)
from models.index_long_only import (  # noqa: E402
    DEFAULT_UP_THRESHOLDS,
    evaluate_long_only_thresholds,
    restrict_dataset_to_dates,
    summarize_long_only_result,
)
from supply.hf_model_data import load_hf_index_prices, validate_index_snapshot  # noqa: E402

INPUT_PATH = ROOT / "reports" / "model_sweep.json"
OUTPUT_PATH = ROOT / "reports" / "index_long_only_selection.json"
PREDICTION_PATH = ROOT / "data" / "raw" / "index_long_only_oos_predictions.parquet"
TRIALS_PATH = ROOT / "reports" / "trials.jsonl"
LOCAL_HF_ROOT = ROOT / "data" / "raw" / "hf_snapshot"
EXPECTED_CANDIDATES = 100
MIN_BASELINE_WIN_FOLDS = 7
MULTIPLE_TEST_COUNT = 100
BONFERRONI_ALPHA = 0.05 / MULTIPLE_TEST_COUNT
DEFLATED_SHARPE_THRESHOLD = 0.8115


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if pd.isna(value):
        return None
    raise TypeError(f"JSON으로 변환할 수 없는 값입니다: {type(value).__name__}")


def _optional_class_weight(value: object) -> object:
    """DataFrame이 선택값 ``None``을 NaN으로 바꿔도 다시 ``None``으로 복원한다."""

    if isinstance(value, (float, np.floating)) and pd.isna(value):
        return None
    return value


def _json_clean(value: object, *, field_name: str | None = None) -> object:
    """선택적 class weight만 JSON ``null``로 바꾸고 지표의 비정상값은 숨기지 않는다."""

    if isinstance(value, dict):
        return {
            str(key): _json_clean(item, field_name=str(key)) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_clean(item, field_name=field_name) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        if field_name in {"class_weight", "selected_class_weight"}:
            return None
        raise ValueError(f"{field_name or '값'}에 유한하지 않은 수치가 있습니다: {value!r}")
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _candidate_id(experiment: dict[str, Any]) -> str:
    returns = experiment.get("return_features", [])
    variant = "base" if not returns else "-".join(str(value) for value in returns)
    return "-".join(
        (
            str(experiment["combination"]).lower(),
            variant.lower(),
            str(experiment["model"]).lower(),
        )
    )


def _candidate_manifest(report: dict[str, Any]) -> list[dict[str, Any]]:
    experiments = report.get("experiments")
    if not isinstance(experiments, list) or len(experiments) != EXPECTED_CANDIDATES:
        raise ValueError(
            f"KOSPI200 기존 후보는 정확히 {EXPECTED_CANDIDATES}개여야 합니다: "
            f"{0 if not isinstance(experiments, list) else len(experiments)}"
        )
    manifest = [dict(record["experiment"]) for record in experiments]
    identifiers = [_candidate_id(experiment) for experiment in manifest]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("KOSPI200 후보 조합·수익률 피처·모델 키가 중복됩니다.")
    return manifest


def _build_common_datasets(
    index_prices: pd.DataFrame,
    experiments: list[dict[str, Any]],
) -> tuple[dict[tuple[str, tuple[str, ...]], ModelDataset], tuple[str, ...]]:
    datasets: dict[tuple[str, tuple[str, ...]], ModelDataset] = {}
    for experiment in experiments:
        key = (
            str(experiment["combination"]),
            tuple(str(value) for value in experiment.get("return_features", [])),
        )
        if key not in datasets:
            datasets[key] = build_model_dataset(
                index_prices,
                key[0],
                return_features=key[1],
            )

    common = set.intersection(
        *(set(dataset.frame["bas_dd"].astype(str)) for dataset in datasets.values())
    )
    common_dates = tuple(sorted(common))
    if len(common_dates) < MIN_TRAIN_SIZE + LABEL_HORIZON + VALID_SIZE:
        raise RuntimeError(
            "모든 후보의 공통 거래일로 12폴드 평가를 구성할 수 없습니다: "
            f"{len(common_dates)}일"
        )
    restricted = {
        key: restrict_dataset_to_dates(dataset, common_dates)
        for key, dataset in datasets.items()
    }
    return restricted, common_dates


def _trial_records(
    *,
    run_id: str,
    source: dict[str, object],
    experiment: dict[str, Any],
    result: object,
) -> list[dict[str, object]]:
    candidate_id = _candidate_id(experiment)
    common = {
        "schema_version": 1,
        "status": "success",
        "track": "kospi200",
        "run": "kospi200_long_only_threshold_selection",
        "report_path": "reports/index_long_only_selection.json",
        "dataset_source": source,
        "experiment": {
            **experiment,
            "threshold_policy": "inner_pr_auc_then_delta_sharpe_threshold",
            "threshold_candidates": list(DEFAULT_UP_THRESHOLDS),
        },
        "audit": {
            "origin": "direct_execution",
            "holdout_used": False,
            "coverage": "all_completed_fits",
        },
    }
    records: list[dict[str, object]] = []
    for row in result.inner_results.to_dict(orient="records"):
        fold = int(row["fold"])
        class_weight = _optional_class_weight(row["class_weight"])
        weight_id = "none" if class_weight is None else str(class_weight)
        records.append(
            {
                **common,
                "trial_id": f"{run_id}-{candidate_id}-fold-{fold:02d}-inner-{weight_id}",
                "fit": {
                    "phase": "inner_threshold_selection",
                    "fold": fold,
                    "class_weight": class_weight,
                    "train_rows": int(row["inner_train_size"]),
                    "valid_rows": int(row["inner_valid_size"]),
                    "train_end": row["inner_train_end"],
                    "valid_start": row["inner_valid_start"],
                    "valid_end": row["inner_valid_end"],
                    "selected_up_threshold": row["threshold"],
                },
                "metrics": {
                    key: row[key]
                    for key in (
                        "inner_pr_auc_up",
                        "up_precision",
                        "up_recall",
                        "buy_signals",
                        "opportunity_rate",
                        "delta_sharpe_net",
                    )
                },
            }
        )
    for row in result.outer_results.to_dict(orient="records"):
        fold = int(row["fold"])
        class_weight = _optional_class_weight(row["selected_class_weight"])
        records.append(
            {
                **common,
                "trial_id": f"{run_id}-{candidate_id}-fold-{fold:02d}-outer",
                "fit": {
                    "phase": "outer_oos_evaluation",
                    "fold": fold,
                    "class_weight": class_weight,
                    "train_rows": int(row["train_size"]),
                    "valid_rows": int(row["valid_size"]),
                    "train_end": row["train_end"],
                    "valid_start": row["valid_start"],
                    "valid_end": row["valid_end"],
                    "selected_up_threshold": row["selected_up_threshold"],
                },
                "metrics": {
                    key: row[key]
                    for key in (
                        "accuracy",
                        "macro_f1",
                        "pr_auc_up",
                        "up_precision",
                        "up_recall",
                        "buy_signals",
                        "opportunity_rate",
                        "delta_sharpe_net",
                        "training_majority_baseline_accuracy",
                        "accuracy_minus_training_majority_baseline",
                    )
                },
            }
        )
    return records


def _append_trials(records: list[dict[str, object]]) -> None:
    TRIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRIALS_PATH.open("a", encoding="utf-8") as stream:
        for record in records:
            stream.write(
                json.dumps(
                    _json_clean(record),
                    ensure_ascii=False,
                    default=_json_default,
                    allow_nan=False,
                )
                + "\n"
            )


def _rank_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for candidate in candidates:
        summary = candidate["summary"]
        summary["passes_operational_gate"] = bool(
            float(summary["accuracy_minus_training_majority_baseline"]) > 0.0
            and int(summary["baseline_win_folds"]) >= MIN_BASELINE_WIN_FOLDS
        )

    # 게이트 탈락 후보도 누락하지 않고 전체 순위에 남긴다. 게이트 통과 여부를 먼저 두고,
    # 이동원님 제안대로 상승 PR-AUC → 비용 차감 ΔSharpe → 상승 Precision 순으로 정렬한다.
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            bool(candidate["summary"]["passes_operational_gate"]),
            float(candidate["summary"]["pr_auc_up"]),
            float(candidate["summary"]["delta_sharpe_net_median"]),
            float(candidate["summary"]["up_precision"]),
            float(candidate["summary"]["accuracy"]),
            _candidate_id(candidate["experiment"]),
        ),
        reverse=True,
    )
    for rank, candidate in enumerate(ranked, start=1):
        candidate["rank"] = rank
    return ranked


def selection_audit(ranked: list[dict[str, Any]]) -> dict[str, Any]:
    """최종 1위의 관문 해석·다중 시도·전처리 계약을 보고서에 고정한다."""

    if not ranked:
        raise ValueError("선정 감사를 계산할 후보가 없습니다.")
    fold_differences = np.asarray(
        [
            row["accuracy_minus_training_majority_baseline"]
            for row in ranked[0]["outer_fold_results"]
        ],
        dtype=float,
    )
    if len(fold_differences) != N_FOLDS:
        raise ValueError(
            f"최종 후보의 외부 폴드가 {N_FOLDS}개가 아닙니다: {len(fold_differences)}"
        )
    test = ttest_1samp(fold_differences, 0.0, alternative="greater")
    raw_p_value = float(test.pvalue)
    return {
        "operational_gate_interpretation": (
            "7/12는 통계적 유의성 문턱이 아니라 저성능 후보를 거르는 최소 운영 안정성 관문"
        ),
        "ranking_sensitivity": (
            "Accuracy 기반 운영 관문을 제거하고 상승 PR-AUC만 우선하면 조합 E "
            "LogisticRegression이 1위로 바뀔 수 있음"
        ),
        "multiple_testing": {
            "candidate_count": MULTIPLE_TEST_COUNT,
            "bonferroni_alpha": BONFERRONI_ALPHA,
            "winner_accuracy_paired_t": {
                "pairs": N_FOLDS,
                "alternative": "accuracy_minus_training_majority_baseline > 0",
                "mean_difference": float(fold_differences.mean()),
                "t_statistic": float(test.statistic),
                "p_value_one_sided": raw_p_value,
                "bonferroni_adjusted_p_value": min(
                    raw_p_value * MULTIPLE_TEST_COUNT, 1.0
                ),
                "passes_bonferroni": raw_p_value < BONFERRONI_ALPHA,
            },
            "deflated_sharpe_threshold_n_100": DEFLATED_SHARPE_THRESHOLD,
            "interpretation": (
                "개발구간 1위는 100회 탐색 뒤의 결과이며 통계적 우위가 확정됐다는 뜻이 아님"
            ),
        },
        "preprocessing": {
            "logistic_regression": "StandardScaler then LogisticRegression",
            "fit_scope": "each_inner_or_outer_training_fold_only",
            "implementation": "sklearn Pipeline",
            "full_period_scaling": False,
        },
    }


def _write_report(
    *,
    run_id: str,
    source: dict[str, object],
    common_dates: tuple[str, ...],
    splits: list[tuple[np.ndarray, np.ndarray]],
    candidates: list[dict[str, Any]],
    complete: bool,
) -> None:
    ranked = _rank_candidates(candidates)
    schedule = [
        {
            "fold": fold,
            "train_rows": len(train),
            "valid_rows": len(valid),
            "train_end": common_dates[int(train[-1])],
            "valid_start": common_dates[int(valid[0])],
            "valid_end": common_dates[int(valid[-1])],
        }
        for fold, (train, valid) in enumerate(splits, start=1)
    ]
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "status": "complete" if complete else "partial",
        "holdout_used": False,
        "source": source,
        "candidate_count": len(ranked),
        "expected_candidate_count": EXPECTED_CANDIDATES,
        "common_oos_contract": {
            "common_date_start": common_dates[0],
            "common_date_end": common_dates[-1],
            "common_date_rows": len(common_dates),
            "oos_validation_start": schedule[0]["valid_start"],
            "oos_validation_end": schedule[-1]["valid_end"],
            "oos_rows_per_candidate": sum(int(row["valid_rows"]) for row in schedule),
            "folds": N_FOLDS,
            "min_train": MIN_TRAIN_SIZE,
            "valid_size": VALID_SIZE,
            "gap": LABEL_HORIZON,
            "schedule": schedule,
        },
        "threshold_policy": {
            "candidates": list(DEFAULT_UP_THRESHOLDS),
            "selection_scope": "each_outer_fold_training_data_only",
            "class_weight_order": "inner_pr_auc_up_then_delta_sharpe",
            "threshold_order": (
                "delta_sharpe_net_then_precision_times_opportunity_then_precision"
            ),
            "prediction_rule": (
                "p_up >= threshold이면 상승, 아니면 p_down과 p_neutral 중 큰 클래스"
            ),
        },
        "selection_policy": {
            "status": "pending_issue_216_statistical_review",
            "gate": (
                "accuracy_minus_training_majority_baseline > 0 and "
                f"baseline_win_folds >= {MIN_BASELINE_WIN_FOLDS}/{N_FOLDS}"
            ),
            "ranking": ["pr_auc_up", "delta_sharpe_net_median", "up_precision"],
            "down_recall": "report_only",
            **selection_audit(ranked),
        },
        "provisional_winner": ranked[0] if complete and ranked else None,
        "candidates": ranked,
    }
    OUTPUT_PATH.write_text(
        json.dumps(
            _json_clean(report),
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def _load_index_snapshot(refresh_hf: bool) -> tuple[pd.DataFrame, dict[str, object]]:
    if refresh_hf:
        snapshot = load_hf_index_prices()
        return snapshot.frame, {
            "repo": "qurious-quant/alphastack-krx-dev",
            "repo_sha": snapshot.repo_sha,
            "index_sha256": snapshot.file_sha256,
            "generated_at": snapshot.generated_at,
            "dev_end": snapshot.dev_end,
        }

    manifest_path = LOCAL_HF_ROOT / "MANIFEST.json"
    index_path = LOCAL_HF_ROOT / "full" / "index_price_dev.parquet"
    if not manifest_path.is_file() or not index_path.is_file():
        raise FileNotFoundError(
            "로컬 HF 스냅샷이 없습니다. 먼저 전체 데이터를 내려받거나 --refresh-hf를 사용하세요."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frame, file_sha = validate_index_snapshot(index_path, manifest)
    return frame, {
        "repo": "qurious-quant/alphastack-krx-dev",
        "repo_sha": "local_snapshot_verified_by_manifest",
        "index_sha256": file_sha,
        "generated_at": str(manifest.get("generated_at", "")),
        "dev_end": str(manifest.get("dev_end", "")),
    }


def main(candidate_limit: int | None = None, *, refresh_hf: bool = False) -> int:
    source_report = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    experiments = _candidate_manifest(source_report)
    if candidate_limit is not None:
        if not 1 <= candidate_limit <= len(experiments):
            raise ValueError("candidate-limit은 1~100이어야 합니다.")
        experiments = experiments[:candidate_limit]

    index_prices, source = _load_index_snapshot(refresh_hf)
    datasets, common_dates = _build_common_datasets(index_prices, experiments)
    splits = expanding_splits(
        len(common_dates),
        n_folds=N_FOLDS,
        min_train=MIN_TRAIN_SIZE,
        horizon=VALID_SIZE,
        gap=LABEL_HORIZON,
        label_horizon=LABEL_HORIZON,
    )
    if len(splits) != N_FOLDS:
        raise RuntimeError(f"공통 외부 폴드가 {N_FOLDS}개가 아닙니다: {len(splits)}")

    run_id = datetime.now(timezone.utc).strftime("index-long-only-%Y%m%dT%H%M%S.%f%z")
    candidates: list[dict[str, Any]] = []
    prediction_parts: list[pd.DataFrame] = []
    total = len(experiments)
    for number, experiment in enumerate(experiments, start=1):
        key = (
            str(experiment["combination"]),
            tuple(str(value) for value in experiment.get("return_features", [])),
        )
        model_name = str(experiment["model"])
        print(f"[{number:03d}/{total:03d}] {_candidate_id(experiment)}", flush=True)
        result = evaluate_long_only_thresholds(datasets[key], model_name, splits)
        summary = summarize_long_only_result(result)
        candidates.append(
            {
                "experiment": experiment,
                "feature_columns": list(datasets[key].feature_columns),
                "summary": summary,
                "inner_selections": result.inner_results.to_dict(orient="records"),
                "outer_fold_results": result.outer_results.to_dict(orient="records"),
            }
        )
        predictions = result.oos_predictions.copy()
        predictions.insert(0, "candidate_id", _candidate_id(experiment))
        prediction_parts.append(predictions)
        _append_trials(
            _trial_records(
                run_id=run_id,
                source=source,
                experiment=experiment,
                result=result,
            )
        )
        _write_report(
            run_id=run_id,
            source=source,
            common_dates=common_dates,
            splits=splits,
            candidates=candidates,
            complete=False,
        )

    all_predictions = pd.concat(prediction_parts, ignore_index=True)
    PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_predictions.to_parquet(PREDICTION_PATH, index=False)
    complete = candidate_limit is None
    _write_report(
        run_id=run_id,
        source=source,
        common_dates=common_dates,
        splits=splits,
        candidates=candidates,
        complete=complete,
    )
    print(f"후보 {len(candidates)}개 평가 완료: {OUTPUT_PATH}")
    print(f"전체 OOS 확률 로컬 저장: {PREDICTION_PATH}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-limit", type=int)
    parser.add_argument("--refresh-hf", action="store_true")
    arguments = parser.parse_args()
    raise SystemExit(main(arguments.candidate_limit, refresh_hf=arguments.refresh_hf))
