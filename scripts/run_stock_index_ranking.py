"""KOSPI200 최우수 OOS 방향과 개별종목 최우수 OOS 확률을 결합한다."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from features.model_dataset import build_model_dataset  # noqa: E402
from models.experiment import MODEL_BUILDERS, evaluate_nested_class_weights  # noqa: E402
from models.stock_experiment import evaluate_stock_models  # noqa: E402
from models.stock_ranking import (  # noqa: E402
    aligned_index_splits,
    aligned_panel_splits,
    build_common_validation_schedule,
    select_for_index_direction,
    summarize_direction_ranking,
    summarize_random_ranking_baseline,
)
from scripts.run_stock_model_experiment import load_stock_model_dataset  # noqa: E402

HF_INDEX_PATH = ROOT / "data" / "raw" / "hf_snapshot" / "full" / "index_price_dev.parquet"
STOCK_REPORT_PATH = ROOT / "reports" / "stock_feature_combinations.json"
LOCAL_RANKING_PATH = ROOT / "data" / "raw" / "stock_index_direction_ranking.parquet"
REPORT_PATH = ROOT / "reports" / "stock_index_ranking.json"
TRIALS_PATH = ROOT / "reports" / "trials.jsonl"

INDEX_COMBINATION = "E"
INDEX_RETURN_FEATURES = ("five_day_return",)
INDEX_MODEL = "RandomForest"


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


def _direction_summary(ranking: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    names = {-1: "하락", 0: "중립", 1: "상승"}
    for label in (-1, 0, 1):
        selected = ranking.loc[ranking["index_predicted"].eq(label)]
        rows.append(
            {
                "index_direction": names[label],
                "rows": int(len(selected)),
                "dates": int(selected["bas_dd"].nunique()),
                "direction_hit_rate": float(
                    selected["label_numeric"].eq(label).mean()
                ),
                "mean_selected_probability": float(
                    selected["selected_probability"].mean()
                ),
            }
        )
    return rows


def _append_alignment_trials(
    *,
    run_id: str,
    track: str,
    experiment: dict[str, object],
    source: dict[str, object],
    inner_results: pd.DataFrame,
    outer_results: pd.DataFrame,
) -> None:
    """지수·종목 공통 OOS 일정을 만들며 실제로 수행한 fit을 모두 기록한다."""

    records: list[dict[str, object]] = []

    def _row_size(row: dict[str, object], *keys: str) -> int:
        """지수·종목 평가기가 서로 다르게 붙인 크기 열을 한 형식으로 읽는다."""

        for key in keys:
            if key in row:
                return int(row[key])
        raise KeyError(f"학습·검증 크기 열을 찾지 못했습니다: {keys}")

    common = {
        "schema_version": 1,
        "status": "success",
        "track": track,
        "run": run_id,
        "report_path": "reports/stock_index_ranking.json",
        "dataset_source": source,
        "experiment": experiment,
        "audit": {"origin": "direct_execution", "coverage": "all_completed_fits"},
    }
    for row in inner_results.to_dict(orient="records"):
        weight = row["class_weight"]
        weight_id = "none" if weight is None else str(weight)
        records.append(
            {
                **common,
                "trial_id": (
                    f"{run_id}-{track}-{row['model']}-fold-{int(row['fold']):02d}"
                    f"-inner-{weight_id}"
                ),
                "fit": {
                    "phase": "inner_class_weight_selection",
                    "model": row["model"],
                    "fold": int(row["fold"]),
                    "class_weight": weight,
                    "train_rows": _row_size(row, "inner_train_rows", "inner_train_size"),
                    "valid_rows": _row_size(row, "inner_valid_rows", "inner_valid_size"),
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
                "trial_id": f"{run_id}-{track}-{row['model']}-fold-{int(row['fold']):02d}-outer",
                "fit": {
                    "phase": "outer_evaluation",
                    "model": row["model"],
                    "fold": int(row["fold"]),
                    "class_weight": row["selected_class_weight"],
                    "train_rows": _row_size(row, "train_rows", "train_size"),
                    "valid_rows": _row_size(row, "valid_rows", "valid_size"),
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
    """두 OOS 예측의 공통 날짜에서 방향별 Top 5 종목 랭킹을 만든다."""

    for path in (HF_INDEX_PATH, STOCK_REPORT_PATH):
        if not path.exists():
            raise FileNotFoundError(f"선행 산출물이 없습니다: {path}")

    stock_report = json.loads(STOCK_REPORT_PATH.read_text(encoding="utf-8"))
    stock_winner = stock_report["combination_winners"][0]
    stock_combination = str(stock_winner["combination"])
    stock_model = str(stock_winner["model"])
    stock_features = tuple(stock_report["combinations"][stock_combination]["features"])
    index_sha = _sha256(HF_INDEX_PATH)
    if stock_report["source"]["index_sha256"] != index_sha:
        raise RuntimeError("개별종목 OOS와 현재 HF 지수 Parquet의 리비전이 다릅니다.")

    print("[1/4] KOSPI200 조합E + 5Day Return RandomForest OOS 재현", flush=True)
    index_prices = pd.read_parquet(HF_INDEX_PATH)
    index_dataset = build_model_dataset(
        index_prices,
        INDEX_COMBINATION,
        return_features=INDEX_RETURN_FEATURES,
    )
    stock_dataset = load_stock_model_dataset(stock_features)
    schedule = build_common_validation_schedule(
        index_dataset.frame["bas_dd"],
        stock_dataset.frame["bas_dd"],
    )
    index_splits = aligned_index_splits(
        index_dataset.frame["bas_dd"],
        schedule,
    )
    index_result = evaluate_nested_class_weights(
        index_dataset,
        model_names=(INDEX_MODEL,),
        outer_splits=index_splits,
    )
    index_oos = index_result.oos_predictions.loc[:, ["bas_dd", "actual", "predicted"]]

    print(
        f"[2/4] 같은 OOS 날짜로 개별종목 조합{stock_combination} {stock_model} 재현",
        flush=True,
    )
    stock_splits = aligned_panel_splits(
        stock_dataset.frame["bas_dd"],
        schedule,
    )
    stock_result = evaluate_stock_models(
        stock_dataset,
        model_builders={stock_model: MODEL_BUILDERS[stock_model]},
        outer_splits=stock_splits,
    )
    stock_oos = stock_result.oos_predictions
    ranking = select_for_index_direction(stock_oos, index_oos, top_n=5)
    LOCAL_RANKING_PATH.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_parquet(LOCAL_RANKING_PATH, index=False)

    print("[3/4] Top 1·3·5 방향 적중률 계산", flush=True)
    cutoff_summary = summarize_direction_ranking(
        stock_oos,
        index_oos,
        cutoffs=(1, 3, 5),
    )
    random_summary = summarize_random_ranking_baseline(
        stock_oos,
        index_oos,
        cutoffs=(1, 3, 5),
    )
    overall_random = random_summary.loc[random_summary["index_predicted"].isna()]
    random_by_cutoff = overall_random.set_index("top_n")["random_direction_hit_rate"]
    cutoff_summary["random_direction_hit_rate"] = cutoff_summary["top_n"].map(
        random_by_cutoff
    )
    cutoff_summary["hit_rate_lift_over_random"] = (
        cutoff_summary["direction_hit_rate"]
        - cutoff_summary["random_direction_hit_rate"]
    )
    baseline = stock_oos.merge(
        index_oos.loc[:, ["bas_dd", "predicted"]].rename(
            columns={"predicted": "index_predicted"}
        ),
        on="bas_dd",
        how="inner",
        validate="many_to_one",
    )
    baseline_hit_rate = float(
        baseline["label_numeric"].eq(baseline["index_predicted"]).mean()
    )

    outer_metrics = (
        index_result.outer_results[
            ["accuracy", "macro_f1", "down_recall", "core_harmonic_mean"]
        ]
        .mean()
        .to_dict()
    )
    generated_at = datetime.now(timezone.utc).isoformat()
    run_id = "stock-index-ranking-" + generated_at.replace(":", "").replace("-", "")
    source = {
        "repo": stock_report["source"]["repo"],
        "index_path": stock_report["source"]["index_path"],
        "index_sha256": index_sha,
        "daily_path": stock_report["source"]["daily_path"],
        "daily_sha256": stock_report["source"]["daily_sha256"],
        "holdout_start": stock_report["source"]["holdout_start"],
    }
    _append_alignment_trials(
        run_id=run_id,
        track="index",
        experiment={
            "combination": INDEX_COMBINATION,
            "return_features": list(INDEX_RETURN_FEATURES),
        },
        source=source,
        inner_results=index_result.inner_results,
        outer_results=index_result.outer_results,
    )
    _append_alignment_trials(
        run_id=run_id,
        track="stock",
        experiment={
            "combination": stock_combination,
            "feature_columns": list(stock_features),
        },
        source=source,
        inner_results=stock_result.inner_results,
        outer_results=stock_result.outer_results,
    )
    report = {
        "generated_at_utc": generated_at,
        "run_id": run_id,
        "source": {
            **source,
        },
        "data_quality_policy": stock_report["data_quality_policy"],
        "index_model": {
            "combination": INDEX_COMBINATION,
            "return_features": list(INDEX_RETURN_FEATURES),
            "model": INDEX_MODEL,
            "outer_fold_metrics": outer_metrics,
            "selected_class_weights": (
                index_result.outer_results["selected_class_weight"]
                .fillna("None")
                .value_counts()
                .to_dict()
            ),
            "prediction_distribution": {
                str(key): int(value)
                for key, value in index_oos["predicted"].value_counts().items()
            },
        },
        "stock_model": {
            "combination": stock_combination,
            "model": stock_model,
            "feature_columns": list(stock_features),
            "outer_fold_metrics": (
                stock_result.outer_results[
                    ["accuracy", "macro_f1", "down_recall", "core_harmonic_mean"]
                ]
                .mean()
                .to_dict()
            ),
            "selected_class_weights": (
                stock_result.outer_results["selected_class_weight"]
                .fillna("None")
                .value_counts()
                .to_dict()
            ),
            "fold_baseline_summary": {
                "training_majority_baseline_accuracy_mean": float(
                    stock_result.outer_results[
                        "training_majority_baseline_accuracy"
                    ].mean()
                ),
                "validation_majority_oracle_accuracy_mean": float(
                    stock_result.outer_results[
                        "validation_majority_oracle_accuracy"
                    ].mean()
                ),
                "accuracy_minus_training_majority_baseline_mean": float(
                    stock_result.outer_results[
                        "accuracy_minus_training_majority_baseline"
                    ].mean()
                ),
            },
        },
        "common_oos": {
            "dates": int(ranking["bas_dd"].nunique()),
            "first_date": str(ranking["bas_dd"].min()),
            "last_date": str(ranking["bas_dd"].max()),
            "top5_rows": int(len(ranking)),
        },
        "all_candidate_direction_hit_rate": baseline_hit_rate,
        "ranking_summary": cutoff_summary.to_dict(orient="records"),
        "random_ranking_baseline": (
            random_summary.astype(object)
            .where(random_summary.notna(), None)
            .to_dict(orient="records")
        ),
        "top5_by_index_direction": _direction_summary(ranking),
        "local_ranking_path": str(LOCAL_RANKING_PATH.relative_to(ROOT)).replace(
            "\\", "/"
        ),
        "note": (
            "이 결과는 분류 확률 랭킹 평가다. 상승·중립·하락을 실제 포지션으로 바꾸는 "
            "규칙이 확정되지 않아 거래비용 수익률 백테스트에는 사용하지 않았다."
        ),
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )

    print(f"[4/4] 결과 저장: {REPORT_PATH.relative_to(ROOT)}", flush=True)
    print(f"      전체 후보 방향 적중률: {baseline_hit_rate:.4f}", flush=True)
    for row in report["ranking_summary"]:
        print(
            f"      Top {row['top_n']}: 적중률 {row['direction_hit_rate']:.4f} · "
            f"평균 선택확률 {row['mean_selected_probability']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
