"""HF 개발본으로 개별종목 후보·패널·4모델 OOS 실험을 재현한다."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.horizon import HOLDOUT_START  # noqa: E402
from evaluation.multiple_comparison import compare_accuracy_to_baseline  # noqa: E402
from evaluation.walk_forward import expanding_group_splits  # noqa: E402
from features.stock_model_dataset import (  # noqa: E402
    ALL_STOCK_FEATURE_COLUMNS,
    STOCK_COMBINATION_FEATURES,
    STOCK_FEATURE_COLUMNS,
    StockModelDataset,
    align_stock_feature_datasets,
    build_sector_stock_model_dataset,
    select_stock_feature_dataset,
)
from models.experiment import (  # noqa: E402
    MODEL_BUILDERS,
    classification_probability_metrics,
)
from models.selection import (  # noqa: E402
    final_model_selection_key,
    rank_final_model_candidates,
)
from models.stock_experiment import (  # noqa: E402
    LABEL_HORIZON,
    MIN_TRAIN_DATES,
    N_FOLDS,
    VALID_DATES,
    evaluate_stock_models,
    fold_classification_baselines,
)
from models.stock_ranking import add_probability_ranks  # noqa: E402
from supply.adj_quality import attach_adjustment_quality  # noqa: E402
from supply.stock_training_universe import build_sector_candidate_frame  # noqa: E402

HF_ROOT = ROOT / "data" / "raw" / "hf_snapshot"
DAILY_PATH = HF_ROOT / "full" / "daily_price_dev.parquet"
INDEX_PATH = HF_ROOT / "full" / "index_price_dev.parquet"
LOCAL_OOS_PATH = ROOT / "data" / "raw" / "stock_model_oos.parquet"
PANEL_CACHE_PATH = ROOT / "data" / "raw" / "stock_model_panel.parquet"
PANEL_CACHE_META_PATH = ROOT / "data" / "raw" / "stock_model_panel.meta.json"
REPORT_PATH = ROOT / "reports" / "stock_model_experiment.json"
SWEEP_REPORT_PATH = ROOT / "reports" / "stock_feature_combinations.json"
TRIALS_PATH = ROOT / "reports" / "trials.jsonl"

DAILY_COLUMNS = (
    "bas_dd",
    "code",
    "name",
    "market",
    "market_cap",
    "value",
    "industry",
    "adj_open",
    "adj_high",
    "adj_low",
    "adj_close",
    "close",
    "change_rate",
    "volume",
)
INDEX_COLUMNS = ("bas_dd", "index_name", "index_class", "market_cap", "close")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if pd.isna(value):
        return None
    raise TypeError(f"JSON으로 바꿀 수 없는 값입니다: {type(value).__name__}")


def _panel_cache_signature() -> dict[str, object]:
    """HF 원천과 패널 생성 코드가 같을 때만 재사용할 캐시 서명을 만든다."""

    code_paths = (
        ROOT / "features" / "stock_model_dataset.py",
        ROOT / "supply" / "adj_quality.py",
        ROOT / "supply" / "stock_training_universe.py",
    )
    digest = hashlib.sha256()
    for path in code_paths:
        digest.update(path.read_bytes())
    return {
        "daily_sha256": _sha256(DAILY_PATH),
        "index_sha256": _sha256(INDEX_PATH),
        "panel_code_sha256": digest.hexdigest(),
        "features": list(ALL_STOCK_FEATURE_COLUMNS),
        "adjustment_quality_policy": "exclude_only_is_adj_suspect_gap_over_1pct_point",
    }


def _read_cached_panel(signature: dict[str, object]) -> StockModelDataset | None:
    """서명이 정확히 같은 로컬 중간 패널만 읽는다."""

    if not PANEL_CACHE_PATH.exists() or not PANEL_CACHE_META_PATH.exists():
        return None
    metadata = json.loads(PANEL_CACHE_META_PATH.read_text(encoding="utf-8"))
    if metadata != signature:
        return None
    frame = pd.read_parquet(PANEL_CACHE_PATH)
    required = {"bas_dd", "label_numeric", *ALL_STOCK_FEATURE_COLUMNS}
    if required - set(frame.columns):
        return None
    if frame.empty or (frame["bas_dd"].astype("string") >= HOLDOUT_START).any():
        return None
    return StockModelDataset(frame=frame, feature_columns=ALL_STOCK_FEATURE_COLUMNS)


def _write_panel_cache(dataset: StockModelDataset, signature: dict[str, object]) -> None:
    """HF 원천이 아닌 재생성 가능한 중간 패널을 data/raw에만 저장한다."""

    PANEL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.frame.to_parquet(PANEL_CACHE_PATH, index=False)
    PANEL_CACHE_META_PATH.write_text(
        json.dumps(signature, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _model_summary(outer_results: pd.DataFrame) -> list[dict[str, object]]:
    candidate_metrics = (
        "accuracy",
        "macro_f1",
        "down_recall",
        "core_harmonic_mean",
        "balanced_accuracy",
        "mcc",
        "pr_auc_down",
        "pr_auc_neutral",
        "pr_auc_up",
        "pr_auc_macro_ovr",
        "training_majority_baseline_accuracy",
        "validation_majority_oracle_accuracy",
        "accuracy_minus_training_majority_baseline",
    )
    metrics = tuple(metric for metric in candidate_metrics if metric in outer_results.columns)
    rows: list[dict[str, object]] = []
    for model, group in outer_results.groupby("model", sort=False):
        row: dict[str, object] = {
            "model": model,
            "folds": int(len(group)),
            "baseline_win_folds": int(
                (group["accuracy_minus_training_majority_baseline"] > 0.0).sum()
            ),
        }
        row["baseline_win_rate"] = row["baseline_win_folds"] / row["folds"]
        for metric in metrics:
            row[metric] = float(group[metric].mean())
            row[f"{metric}_fold_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return sorted(rows, key=final_model_selection_key, reverse=True)


def _oos_diagnostics(predictions: pd.DataFrame) -> dict[str, object]:
    """저장된 OOS 확률에서 확률 지표와 혼동행렬을 다시 계산한다."""

    required = {
        "label_numeric",
        "predicted",
        "p_down",
        "p_neutral",
        "p_up",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"OOS 진단 열이 없습니다: {sorted(missing)}")
    return classification_probability_metrics(
        predictions["label_numeric"].to_numpy(dtype=int),
        predictions["predicted"].to_numpy(dtype=int),
        predictions[["p_down", "p_neutral", "p_up"]].to_numpy(dtype=float),
    )


def _enrich_outer_results_with_oos(
    outer: pd.DataFrame,
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """폴드 행과 모델 전체에 OOS 불균형 분류 지표를 붙인다."""

    enriched = outer.copy()
    scalar_metrics = (
        "balanced_accuracy",
        "mcc",
        "pr_auc_down",
        "pr_auc_neutral",
        "pr_auc_up",
        "pr_auc_macro_ovr",
    )
    overall: dict[str, object] = {}
    for model, model_predictions in predictions.groupby("model", sort=False):
        overall[str(model)] = _oos_diagnostics(model_predictions)
        for fold, fold_predictions in model_predictions.groupby("fold", sort=True):
            diagnostics = _oos_diagnostics(fold_predictions)
            mask = enriched["model"].eq(model) & enriched["fold"].eq(fold)
            if int(mask.sum()) != 1:
                raise ValueError(f"{model} {fold}폴드 결과 행을 하나로 찾지 못했습니다.")
            for metric in scalar_metrics:
                enriched.loc[mask, metric] = float(diagnostics[metric])
    return enriched, overall


def _final_selection_report(report: dict[str, object]) -> dict[str, object]:
    """모든 조합·모델을 사전등록 규칙으로 한 번에 비교한다."""

    candidates = []
    for combination, item in report["combinations"].items():
        for summary in item["model_summary"]:
            candidates.append(
                {
                    "combination": combination,
                    "feature_columns": item["features"],
                    **summary,
                }
            )
    ranked = rank_final_model_candidates(candidates)
    return {
        "decision": "ADR 0007",
        "primary": "accuracy_minus_training_majority_baseline",
        "secondary": "macro_f1",
        "tertiary": "baseline_win_folds",
        "position_policy": "long_only_{0,+1}",
        "holdout_policy": "선정 고정 뒤 한 번만 평가하고 재선정하지 않는다",
        "selected": ranked[0],
        "candidates": ranked,
    }


def _multiple_comparison_report(report: dict[str, object]) -> dict[str, object]:
    """A~H 32개 후보를 같은 폴드 기준선과 비교하고 가족오류율을 통제한다."""

    parts = []
    for combination, item in report["combinations"].items():
        part = pd.DataFrame(item["outer_fold_results"])
        part.insert(0, "combination", combination)
        parts.append(part)
    return compare_accuracy_to_baseline(pd.concat(parts, ignore_index=True))


def _fold_baseline_rows(dataset: StockModelDataset) -> list[dict[str, object]]:
    """저장된 실험과 같은 날짜 분할에서 비누수 기준선과 사후 분포를 계산한다."""

    splits = expanding_group_splits(
        dataset.groups,
        n_folds=N_FOLDS,
        min_train=MIN_TRAIN_DATES,
        horizon=VALID_DATES,
        gap=LABEL_HORIZON,
        label_horizon=LABEL_HORIZON,
    )
    rows: list[dict[str, object]] = []
    for fold, (train, valid) in enumerate(splits, start=1):
        rows.append(
            {
                "fold": fold,
                "train_end": str(dataset.groups[train][-1]),
                "valid_start": str(dataset.groups[valid][0]),
                "valid_end": str(dataset.groups[valid][-1]),
                **fold_classification_baselines(dataset.y[train], dataset.y[valid]),
            }
        )
    return rows


def refresh_saved_report_baselines() -> None:
    """재학습 없이 기존 A~H OOS의 기준선·진단지표·선정 순위를 갱신한다."""

    if not SWEEP_REPORT_PATH.exists():
        raise FileNotFoundError(f"기존 A~H 보고서가 없습니다: {SWEEP_REPORT_PATH}")
    report = json.loads(SWEEP_REPORT_PATH.read_text(encoding="utf-8"))
    datasets = {
        name: load_stock_model_dataset(features)
        for name, features in STOCK_COMBINATION_FEATURES.items()
    }
    datasets = align_stock_feature_datasets(datasets)

    for combination, dataset in datasets.items():
        combination_report = report["combinations"][combination]
        baselines = _fold_baseline_rows(dataset)
        by_fold = {int(row["fold"]): row for row in baselines}
        outer = pd.DataFrame(combination_report["outer_fold_results"])
        for column in baselines[0]:
            if column not in {"fold", "train_end", "valid_start", "valid_end"}:
                outer[column] = outer["fold"].map(
                    {fold: row[column] for fold, row in by_fold.items()}
                )
        outer["accuracy_minus_training_majority_baseline"] = (
            outer["accuracy"] - outer["training_majority_baseline_accuracy"]
        )
        oos_path = ROOT / combination_report["local_oos_path"]
        if not oos_path.exists():
            raise FileNotFoundError(f"기존 OOS 확률 파일이 없습니다: {oos_path}")
        oos = pd.read_parquet(oos_path)
        outer, overall_diagnostics = _enrich_outer_results_with_oos(outer, oos)
        combination_report["outer_fold_results"] = outer.to_dict(orient="records")
        combination_report["model_summary"] = _model_summary(outer)
        combination_report["oos_diagnostics"] = overall_diagnostics
        combination_report["fold_baselines"] = baselines
        combination_report["baseline_summary"] = {
            "training_majority_baseline_accuracy_mean": float(
                np.mean([row["training_majority_baseline_accuracy"] for row in baselines])
            ),
            "validation_majority_oracle_accuracy_mean": float(
                np.mean([row["validation_majority_oracle_accuracy"] for row in baselines])
            ),
            "interpretation": (
                "학습 최빈 기준선만 모델 비교에 사용하며, validation oracle은 "
                "검증 정답 분포를 설명하는 사후 통계다."
            ),
        }
        # 기존 키는 과거 노트북 호환을 위해 남기되 사후 통계라는 새 이름도 함께 둔다.
        combination_report["oos_validation_majority_oracle_accuracy"] = combination_report[
            "majority_baseline_accuracy_on_oos"
        ]

    winners = [
        {"combination": combination, **item["model_summary"][0]}
        for combination, item in report["combinations"].items()
    ]
    winners.sort(key=final_model_selection_key, reverse=True)
    report["combination_winners"] = winners
    report["final_selection"] = _final_selection_report(report)
    report["multiple_comparison"] = _multiple_comparison_report(report)
    report["baseline_enriched_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["validation"]["baseline_policy"] = {
        "comparison": "outer training-window majority class applied to validation",
        "descriptive_only": "validation-window majority oracle",
        "decision": "ADR 0006",
    }
    report["validation"]["selection"] = (
        "ADR 0007: 기준선 대비 Accuracy → Macro F1 → 기준선 승리 폴드 수"
    )
    report["validation"]["multiple_comparison"] = (
        "32개 후보의 폴드 Accuracy 차이를 단측 Wilcoxon으로 검정하고 Holm 보정"
    )
    SWEEP_REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(f"A~H 기존 결과에 폴드별 기준선을 보강했습니다: {SWEEP_REPORT_PATH.relative_to(ROOT)}")


def load_stock_model_dataset(
    feature_columns: tuple[str, ...] = STOCK_FEATURE_COLUMNS,
) -> StockModelDataset:
    """HF full parquet에서 후보·라벨을 만들고 요청한 피처 조합만 선택한다."""

    unknown = set(feature_columns) - set(ALL_STOCK_FEATURE_COLUMNS)
    if unknown:
        raise ValueError(f"아직 계산하지 않는 개별종목 피처입니다: {sorted(unknown)}")
    if not feature_columns or len(set(feature_columns)) != len(feature_columns):
        raise ValueError("개별종목 피처 조합은 비어 있거나 중복될 수 없습니다.")
    for path in (DAILY_PATH, INDEX_PATH):
        if not path.exists():
            raise FileNotFoundError(f"HF 최신본을 먼저 내려받아야 합니다: {path}")

    signature = _panel_cache_signature()
    base_dataset = _read_cached_panel(signature)
    if base_dataset is None:
        daily = pd.read_parquet(
            DAILY_PATH,
            columns=list(DAILY_COLUMNS),
            filters=[("market", "==", "KOSPI")],
        )
        indices = pd.read_parquet(
            INDEX_PATH,
            columns=list(INDEX_COLUMNS),
            filters=[("index_class", "==", "KOSPI")],
        )
        candidates = build_sector_candidate_frame(daily, indices)
        candidates = attach_adjustment_quality(candidates, daily)
        quality_summary = dict(candidates.attrs["adjustment_quality"])
        # 수익률 크기만으로 행을 지우지 않는다. KRX 등락률과 1%p 넘게 어긋난
        # 수정주가 의심 행만 제외하고, KRX와 일치하는 실제 극단 사건은 남긴다.
        candidates = candidates.loc[~candidates["is_adj_suspect"]].copy()
        base_dataset = build_sector_stock_model_dataset(
            daily,
            candidates,
            index_prices=indices,
            feature_columns=ALL_STOCK_FEATURE_COLUMNS,
            drop_incomplete_features=False,
        )
        base_dataset.frame.attrs["adjustment_quality"] = quality_summary
        _write_panel_cache(base_dataset, signature)

    return select_stock_feature_dataset(base_dataset.frame, tuple(feature_columns))


def _append_trials(
    *,
    run_id: str,
    combination: str,
    features: tuple[str, ...],
    source: dict[str, object],
    inner_results: pd.DataFrame,
    outer_results: pd.DataFrame,
) -> None:
    """실제로 끝난 fit을 조합·모델·폴드 단위로 원장에 추가한다."""

    records: list[dict[str, object]] = []
    common = {
        "schema_version": 1,
        "status": "success",
        "track": "stock",
        "run": run_id,
        "report_path": "reports/stock_feature_combinations.json",
        "dataset_source": source,
        "experiment": {"combination": combination, "feature_columns": list(features)},
        "audit": {"origin": "direct_execution", "coverage": "all_completed_fits"},
    }
    for row in inner_results.to_dict(orient="records"):
        weight = row["class_weight"]
        weight_id = "none" if weight is None else str(weight)
        records.append(
            {
                **common,
                "trial_id": (
                    f"{run_id}-{combination}-{row['model']}-fold-{int(row['fold']):02d}"
                    f"-inner-{weight_id}"
                ),
                "fit": {
                    "phase": "inner_class_weight_selection",
                    "model": row["model"],
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
    for row in outer_results.to_dict(orient="records"):
        records.append(
            {
                **common,
                "trial_id": (
                    f"{run_id}-{combination}-{row['model']}-fold-{int(row['fold']):02d}-outer"
                ),
                "fit": {
                    "phase": "outer_evaluation",
                    "model": row["model"],
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
                    for key in ("accuracy", "macro_f1", "down_recall", "core_harmonic_mean")
                },
            }
        )
    with TRIALS_PATH.open("a", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")


def main() -> None:
    """HF 공통 패널에서 A~H를 같은 날짜·종목 조건으로 비교한다."""

    source = {
        "repo": "qurious-quant/alphastack-krx-dev",
        "daily_path": "full/daily_price_dev.parquet",
        "daily_sha256": _sha256(DAILY_PATH),
        "index_path": "full/index_price_dev.parquet",
        "index_sha256": _sha256(INDEX_PATH),
        "holdout_start": HOLDOUT_START,
    }
    print("[1/4] HF 공통 종목 패널과 A~H 피처 준비", flush=True)
    datasets = {
        name: load_stock_model_dataset(features)
        for name, features in STOCK_COMBINATION_FEATURES.items()
    }
    datasets = align_stock_feature_datasets(datasets)
    first_dataset = next(iter(datasets.values()))
    common_dates = set(first_dataset.frame["bas_dd"].unique())
    common_rows = len(first_dataset.frame)
    if len(common_dates) < 750 + 5 + 60:
        raise ValueError("A~H 공통 날짜·종목 표본으로 12폴드 평가를 만들 수 없습니다.")
    quality_summary = dict(
        next(iter(datasets.values())).frame.attrs.get("adjustment_quality", {})
    )
    if not quality_summary:
        raise RuntimeError("수정주가 품질 판정 요약이 종목 패널에 기록되지 않았습니다.")
    print(
        f"      공통 날짜·종목 {common_rows:,}행 · 거래일 {len(common_dates):,}일 · 조합별 행 "
        + ", ".join(f"{name} {len(dataset.frame):,}" for name, dataset in datasets.items()),
        flush=True,
    )

    generated_at = datetime.now(timezone.utc).isoformat()
    run_id = "stock-feature-combinations-" + generated_at.replace(":", "").replace("-", "")
    combination_reports: dict[str, object] = {}
    print("[2/4] 조합 A~H × 4모델 · 날짜 그룹 expanding 12폴드", flush=True)
    for combination, dataset in datasets.items():
        model_results = []
        for model_name, builder in MODEL_BUILDERS.items():
            print(f"      조합{combination} {model_name} 시작", flush=True)
            result = evaluate_stock_models(dataset, model_builders={model_name: builder})
            model_results.append(result)
            _append_trials(
                run_id=run_id,
                combination=combination,
                features=dataset.feature_columns,
                source=source,
                inner_results=result.inner_results,
                outer_results=result.outer_results,
            )
            score = result.outer_results["core_harmonic_mean"].mean()
            print(f"      조합{combination} {model_name} 완료 · {score:.4f}", flush=True)

        inner = pd.concat([result.inner_results for result in model_results], ignore_index=True)
        outer = pd.concat([result.outer_results for result in model_results], ignore_index=True)
        oos = pd.concat([result.oos_predictions for result in model_results], ignore_index=True)
        ranked = add_probability_ranks(oos)
        oos_path = LOCAL_OOS_PATH.with_name(f"stock_model_oos_combination_{combination}.parquet")
        ranked.to_parquet(oos_path, index=False)
        overall_diagnostics = {
            str(model): _oos_diagnostics(model_oos)
            for model, model_oos in oos.groupby("model", sort=False)
        }
        selected_weights = (
            outer.assign(selected_class_weight=outer["selected_class_weight"].fillna("None"))
            .groupby(["model", "selected_class_weight"], sort=False)
            .size()
            .rename("folds")
            .reset_index()
            .to_dict(orient="records")
        )
        actual = oos.drop_duplicates(["bas_dd", "code"])["label_numeric"]
        fold_baselines = _fold_baseline_rows(dataset)
        combination_reports[combination] = {
            "features": list(dataset.feature_columns),
            "panel": {
                "model_rows": int(len(dataset.frame)),
                "dates": int(dataset.frame["bas_dd"].nunique()),
                "stocks": int(dataset.frame["code"].nunique()),
                "first_date": str(dataset.frame["bas_dd"].min()),
                "last_date": str(dataset.frame["bas_dd"].max()),
            },
            "label_distribution": {
                str(key): int(value)
                for key, value in dataset.frame["label"].value_counts().items()
            },
            "majority_baseline_accuracy_on_oos": (
                int(actual.value_counts().max()) / int(len(actual))
            ),
            "oos_validation_majority_oracle_accuracy": (
                int(actual.value_counts().max()) / int(len(actual))
            ),
            "fold_baselines": fold_baselines,
            "baseline_summary": {
                "training_majority_baseline_accuracy_mean": float(
                    np.mean(
                        [
                            row["training_majority_baseline_accuracy"]
                            for row in fold_baselines
                        ]
                    )
                ),
                "validation_majority_oracle_accuracy_mean": float(
                    np.mean(
                        [
                            row["validation_majority_oracle_accuracy"]
                            for row in fold_baselines
                        ]
                    )
                ),
                "interpretation": (
                    "학습 최빈 기준선만 모델 비교에 사용하며, validation oracle은 "
                    "검증 정답 분포를 설명하는 사후 통계다."
                ),
            },
            "model_summary": _model_summary(outer),
            "oos_diagnostics": overall_diagnostics,
            "selected_class_weight_counts": selected_weights,
            "inner_results": inner.to_dict(orient="records"),
            "outer_fold_results": outer.to_dict(orient="records"),
            "inner_trial_count": int(len(inner)),
            "outer_fit_count": int(len(outer)),
            "local_oos_path": str(oos_path.relative_to(ROOT)).replace("\\", "/"),
        }

    winners = [
        {
            "combination": combination,
            **report["model_summary"][0],
        }
        for combination, report in combination_reports.items()
    ]
    winners.sort(key=final_model_selection_key, reverse=True)
    report = {
        "generated_at_utc": generated_at,
        "run_id": run_id,
        "source": source,
        "candidate_rule": {
            "sector_count": 10,
            "stocks_per_sector": 5,
            "max_candidates_per_day": 50,
        },
        "data_quality_policy": {
            "absolute_return_filter": False,
            "method": "supply.adj_quality.attach_adjustment_quality",
            "removed_candidate_rows": int(
                quality_summary["candidate_suspect_rows"]
            ),
            "reason": (
                "KRX 등락률과 수정주가 일간수익률이 1%p 넘게 어긋난 "
                "is_adj_suspect 행만 제외한다."
            ),
            "adjustment_quality": quality_summary,
        },
        "validation": {
            "window": "expanding",
            "folds": 12,
            "minimum_initial_train_dates": 750,
            "valid_dates_per_fold": 60,
            "gap_dates": 5,
            "class_weight_candidates": [None, "balanced"],
            "selection": "ADR 0007: 기준선 대비 Accuracy → Macro F1 → 기준선 승리 폴드 수",
            "common_dates_across_combinations": len(common_dates),
            "common_rows_across_combinations": common_rows,
            "common_row_keys": ["bas_dd", "code"],
            "baseline_policy": {
                "comparison": "outer training-window majority class applied to validation",
                "descriptive_only": "validation-window majority oracle",
                "decision": "ADR 0006",
            },
        },
        "combinations": combination_reports,
        "combination_winners": winners,
        "fit_count": sum(
            int(item["inner_trial_count"]) + int(item["outer_fit_count"])
            for item in combination_reports.values()
        ),
    }
    report["final_selection"] = _final_selection_report(report)
    report["multiple_comparison"] = _multiple_comparison_report(report)
    report["validation"]["multiple_comparison"] = (
        "32개 후보의 폴드 Accuracy 차이를 단측 Wilcoxon으로 검정하고 Holm 보정"
    )
    SWEEP_REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(f"[3/4] A~H 리포트 저장: {SWEEP_REPORT_PATH.relative_to(ROOT)}", flush=True)
    print("[4/4] 조합별 1위", flush=True)
    for row in winners:
        print(
            f"      조합{row['combination']} {row['model']} · 조화평균 "
            f"{row['core_harmonic_mean']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh-baselines-only",
        action="store_true",
        help="기존 모델을 다시 학습하지 않고 A~H 보고서의 폴드별 기준선만 갱신합니다.",
    )
    arguments = parser.parse_args()
    if arguments.refresh_baselines_only:
        refresh_saved_report_baselines()
    else:
        main()
