"""KOSPI200 최우수 OOS 방향과 개별종목 최우수 OOS 확률을 결합한다."""

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

from evaluation.arima_companion import evaluate_arima_companion  # noqa: E402
from evaluation.baseline import fold_multiclass_baseline_predictions  # noqa: E402
from evaluation.stock_backtest import run_overlapping_stock_backtest  # noqa: E402
from features.model_dataset import build_model_dataset  # noqa: E402
from models.experiment import (  # noqa: E402
    MODEL_BUILDERS,
    classification_metrics,
    classification_probability_metrics,
    evaluate_nested_class_weights,
)
from models.final_models import load_winning_models  # noqa: E402
from models.stock_experiment import evaluate_stock_models  # noqa: E402
from models.stock_ranking import (  # noqa: E402
    aligned_index_splits,
    aligned_panel_splits,
    build_common_validation_schedule,
    build_full_stock_prediction_output,
    select_for_index_direction,
    summarize_direction_ranking,
    summarize_full_stock_prediction_output,
    summarize_random_ranking_baseline,
)
from scripts.run_stock_model_experiment import load_stock_model_dataset  # noqa: E402

HF_INDEX_PATH = ROOT / "data" / "raw" / "hf_snapshot" / "full" / "index_price_dev.parquet"
HF_DAILY_PATH = ROOT / "data" / "raw" / "hf_snapshot" / "full" / "daily_price_dev.parquet"
STOCK_REPORT_PATH = ROOT / "reports" / "stock_feature_combinations.json"
INDEX_REPORT_PATH = ROOT / "reports" / "model_sweep.json"
LOCAL_FULL_OUTPUT_PATH = ROOT / "data" / "raw" / "stock_prediction_output.parquet"
LOCAL_RANKING_PATH = ROOT / "data" / "raw" / "stock_index_direction_ranking.parquet"
LOCAL_BACKTEST_DAILY_PATH = ROOT / "data" / "raw" / "stock_ranking_backtest_daily.parquet"
LOCAL_BACKTEST_TRADE_PATH = ROOT / "data" / "raw" / "stock_ranking_backtest_trades.parquet"
REPORT_PATH = ROOT / "reports" / "stock_index_ranking.json"
TRIALS_PATH = ROOT / "reports" / "trials.jsonl"

STOCK_BACKTEST_COSTS = (0.0, 0.0028, 0.0043)
STOCK_BACKTEST_CUTOFFS = (1, 3, 5)


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


def _index_baseline_report(
    labels: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    model_outer_results: pd.DataFrame,
) -> dict[str, object]:
    """KOSPI200과 같은 12폴드에서 세 고정 기준선을 비교한다."""

    fold_rows = []
    actual_parts: dict[str, list[np.ndarray]] = {}
    predicted_parts: dict[str, list[np.ndarray]] = {}
    model_accuracy = model_outer_results.set_index("fold")["accuracy"]
    for fold, (train, valid) in enumerate(splits, start=1):
        predictions = fold_multiclass_baseline_predictions(labels, train, valid)
        for name, predicted in predictions.items():
            actual = labels[valid]
            accuracy = float(np.mean(predicted == actual))
            fold_rows.append(
                {
                    "fold": fold,
                    "baseline": name,
                    "accuracy": accuracy,
                    "model_wins": bool(float(model_accuracy.loc[fold]) > accuracy),
                }
            )
            actual_parts.setdefault(name, []).append(actual)
            predicted_parts.setdefault(name, []).append(predicted)

    summary = []
    rows = pd.DataFrame(fold_rows)
    for name in ("always_up", "majority_class", "previous_direction"):
        metrics = classification_metrics(
            np.concatenate(actual_parts[name]),
            np.concatenate(predicted_parts[name]),
        )
        summary.append(
            {
                "baseline": name,
                **metrics,
                "model_win_folds": int(
                    rows.loc[rows["baseline"].eq(name), "model_wins"].sum()
                ),
                "folds": len(splits),
                "operational": name != "previous_direction",
            }
        )
    return {
        "summary": summary,
        "fold_results": fold_rows,
        "warning": (
            "previous_direction은 겹치는 5일 라벨의 직전 정답이 예측 시점에 아직 "
            "확정되지 않으므로 설명용이며 모델 선정에 사용하지 않는다."
        ),
    }


def _arima_report(
    dataset: object,
    splits: list[tuple[np.ndarray, np.ndarray]],
    model_outer_results: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame]:
    """KOSPI200 지도학습과 같은 학습 종료점·12폴드에서 ARIMA를 비교한다."""

    result = evaluate_arima_companion(
        dataset.opens,
        dataset.signal_positions,
        dataset.y,
        splits,
        neutral_band=0.01,
        horizon=5,
    )
    predictions = result.predictions
    metrics = classification_metrics(
        predictions["actual"].to_numpy(dtype=int),
        predictions["predicted"].to_numpy(dtype=int),
    )
    model_accuracy = model_outer_results.set_index("fold")["accuracy"]
    fold_results = []
    for fold, group in predictions.groupby("fold", sort=True):
        accuracy = float(group["actual"].eq(group["predicted"]).mean())
        fold_results.append(
            {
                "fold": int(fold),
                "accuracy": accuracy,
                "model_wins": bool(float(model_accuracy.loc[fold]) > accuracy),
            }
        )
    return (
        {
            "metrics": metrics,
            "model_win_folds": int(sum(row["model_wins"] for row in fold_results)),
            "folds": len(fold_results),
            "fold_results": fold_results,
            "orders": result.fold_results.to_dict(orient="records"),
            "fit_trials": int(len(result.trial_results)),
            "contract": (
                "각 외부 폴드의 지도학습 종료일까지만 적합하고, 검증 60거래일 동안 "
                "실제 시가로 갱신하지 않은 채 T+1~T+6 수익률을 예측한다."
            ),
        },
        result.trial_results,
    )


def _append_arima_trials(
    *,
    run_id: str,
    source: dict[str, object],
    trials: pd.DataFrame,
) -> None:
    """ARIMA 차수 탐색과 외부 평가의 실제 적합 호출을 장부에 남긴다."""

    records = []
    for row in trials.to_dict(orient="records"):
        phase = str(row["phase"])
        fold = int(row["fold"])
        order_id = f"{int(row['p'])}-{int(row['d'])}-{int(row['q'])}"
        metrics = {
            key: row[key]
            for key in ("aic", "bic", "accuracy")
            if key in row and pd.notna(row[key])
        }
        records.append(
            {
                "schema_version": 1,
                "status": "success",
                "track": "index_arima",
                "run": run_id,
                "report_path": "reports/stock_index_ranking.json",
                "dataset_source": source,
                "experiment": {"benchmark": "ARIMA", "neutral_band": 0.01},
                "audit": {"origin": "direct_execution", "coverage": "all_completed_fits"},
                "trial_id": f"{run_id}-index-arima-fold-{fold:02d}-{phase}-{order_id}",
                "fit": {
                    "phase": phase,
                    "model": "ARIMA",
                    "fold": fold,
                    "p": int(row["p"]),
                    "d": int(row["d"]),
                    "q": int(row["q"]),
                    "train_rows": int(row["train_rows"]),
                    "valid_rows": int(row["valid_rows"]),
                    "selected": bool(row["selected"]),
                },
                "metrics": metrics,
            }
        )
    with TRIALS_PATH.open("a", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")


def _run_stock_backtests(ranking: pd.DataFrame) -> dict[str, object]:
    """저장된 랭킹으로 비용 전·후 Top-K 5슬리브 성과를 만든다."""

    daily_prices = pd.read_parquet(
        HF_DAILY_PATH,
        columns=["bas_dd", "code", "market", "adj_open"],
        filters=[("market", "==", "KOSPI")],
    )
    needed_codes = set(ranking["code"].astype("string").str.zfill(6))
    needed_start = str(ranking["entry_bas_dd"].min())
    needed_end = str(ranking["exit_bas_dd"].max())
    daily_prices = daily_prices.loc[
        daily_prices["code"].astype("string").str.zfill(6).isin(needed_codes)
        & daily_prices["bas_dd"].astype("string").between(needed_start, needed_end),
        ["bas_dd", "code", "adj_open"],
    ].copy()

    scenarios = []
    daily_parts = []
    trade_parts = []
    track_rows = []
    for cost in STOCK_BACKTEST_COSTS:
        for cutoff in STOCK_BACKTEST_CUTOFFS:
            backtest = run_overlapping_stock_backtest(
                ranking,
                daily_prices,
                top_n=cutoff,
                round_trip_cost=cost,
            )
            scenarios.append(backtest.summary)
            daily_parts.append(
                backtest.daily_returns.assign(top_n=cutoff, round_trip_cost=cost)
            )
            trade_parts.append(
                backtest.trade_log.assign(top_n=cutoff, round_trip_cost=cost)
            )
            track_rows.extend(
                backtest.track_summary.assign(
                    top_n=cutoff,
                    round_trip_cost=cost,
                ).to_dict(orient="records")
            )
    pd.concat(daily_parts, ignore_index=True).to_parquet(
        LOCAL_BACKTEST_DAILY_PATH,
        index=False,
    )
    pd.concat(trade_parts, ignore_index=True).to_parquet(
        LOCAL_BACKTEST_TRADE_PATH,
        index=False,
    )
    return {
        "scenarios": scenarios,
        "track_scenarios": track_rows,
        "daily_path": str(LOCAL_BACKTEST_DAILY_PATH.relative_to(ROOT)).replace("\\", "/"),
        "trade_path": str(LOCAL_BACKTEST_TRADE_PATH.relative_to(ROOT)).replace("\\", "/"),
    }


def refresh_saved_backtests() -> None:
    """모델 재학습 없이 저장된 랭킹의 비용 시나리오만 다시 계산한다."""

    if not LOCAL_RANKING_PATH.exists() or not REPORT_PATH.exists():
        raise FileNotFoundError("먼저 KOSPI200-종목 OOS 랭킹을 생성해야 합니다.")
    ranking = pd.read_parquet(LOCAL_RANKING_PATH)
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    report["stock_backtest"] = _run_stock_backtests(ranking)
    report["backtest_refreshed_at_utc"] = datetime.now(timezone.utc).isoformat()
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(f"비용 전·후 백테스트를 갱신했습니다: {REPORT_PATH.relative_to(ROOT)}")


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
    """두 트랙의 1위 모델로 최대 50종목 기본 출력과 부가 확률 분위 결과를 만든다."""

    for path in (HF_INDEX_PATH, INDEX_REPORT_PATH, STOCK_REPORT_PATH):
        if not path.exists():
            raise FileNotFoundError(f"선행 산출물이 없습니다: {path}")

    stock_report = json.loads(STOCK_REPORT_PATH.read_text(encoding="utf-8"))
    if "final_selection" not in stock_report:
        raise RuntimeError("ADR 0007 최종 선정 결과가 없습니다. 평가 보고서를 먼저 갱신하세요.")
    index_winner, stock_winner = load_winning_models(
        INDEX_REPORT_PATH,
        STOCK_REPORT_PATH,
    )
    index_combination = index_winner.combination
    index_model = index_winner.model
    index_return_features = index_winner.return_features
    stock_combination = stock_winner.combination
    stock_model = stock_winner.model
    stock_features = stock_winner.feature_columns
    index_sha = _sha256(HF_INDEX_PATH)
    if stock_report["source"]["index_sha256"] != index_sha:
        raise RuntimeError("개별종목 OOS와 현재 HF 지수 Parquet의 리비전이 다릅니다.")

    print(
        f"[1/4] KOSPI200 1위 조합{index_combination} {index_model} OOS 재현",
        flush=True,
    )
    index_prices = pd.read_parquet(HF_INDEX_PATH)
    index_dataset = build_model_dataset(
        index_prices,
        index_combination,
        return_features=index_return_features,
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
        model_names=(index_model,),
        outer_splits=index_splits,
    )
    index_oos = index_result.oos_predictions.loc[
        :, ["fold", "bas_dd", "actual", "predicted", "p_down", "p_neutral", "p_up"]
    ]
    index_baselines = _index_baseline_report(
        index_dataset.y,
        index_splits,
        index_result.outer_results,
    )
    index_arima, arima_trials = _arima_report(
        index_dataset,
        index_splits,
        index_result.outer_results,
    )

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

    # 1차 기본 산출물은 업종 10×종목 5 후보 전량이다. 확률 Top 1·3·5는 아래에서
    # 별도로 만드는 부가 실험이며 이 표를 대신하거나 행을 제거하지 않는다.
    full_output = build_full_stock_prediction_output(stock_oos, index_oos)
    full_output_summary = summarize_full_stock_prediction_output(full_output)
    LOCAL_FULL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    full_output.to_parquet(LOCAL_FULL_OUTPUT_PATH, index=False)

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
    index_direction_comparison = stock_oos.merge(
        index_oos.loc[:, ["bas_dd", "predicted"]].rename(
            columns={"predicted": "index_predicted"}
        ),
        on="bas_dd",
        how="inner",
        validate="many_to_one",
    )
    index_direction_match_rate = float(
        index_direction_comparison["label_numeric"]
        .eq(index_direction_comparison["index_predicted"])
        .mean()
    )
    # 기존 콘솔의 "방향 적중률"은 종목 모델 정확도가 아니라 지수 방향 일치율이다.
    # 이름을 유지해야 하는 과거 로그와의 비교용 별칭이며 기본 적중률로 사용하지 않는다.
    baseline_hit_rate = index_direction_match_rate

    print("      T+1~T+6 수정 시가 · 5슬리브 · 비용 전/0.28%/0.43%", flush=True)
    stock_backtest = _run_stock_backtests(ranking)

    index_diagnostics = classification_probability_metrics(
        index_oos["actual"].to_numpy(dtype=int),
        index_oos["predicted"].to_numpy(dtype=int),
        index_oos[["p_down", "p_neutral", "p_up"]].to_numpy(dtype=float),
    )
    outer_metrics = {
        key: value
        for key, value in index_diagnostics.items()
        if key != "confusion_matrix"
    }
    stock_diagnostics = classification_probability_metrics(
        stock_oos["label_numeric"].to_numpy(dtype=int),
        stock_oos["predicted"].to_numpy(dtype=int),
        stock_oos[["p_down", "p_neutral", "p_up"]].to_numpy(dtype=float),
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
            "combination": index_combination,
            "return_features": list(index_return_features),
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
    _append_arima_trials(
        run_id=run_id,
        source=source,
        trials=arima_trials,
    )
    report = {
        "generated_at_utc": generated_at,
        "run_id": run_id,
        "source": {
            **source,
        },
        "data_quality_policy": stock_report["data_quality_policy"],
        "index_model": {
            "combination": index_combination,
            "return_features": list(index_return_features),
            "model": index_model,
            "selection": {
                "metric": index_winner.selection_metric,
                "value": index_winner.selection_value,
            },
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
            "classification_baselines": index_baselines,
            "arima_companion": index_arima,
        },
        "stock_model": {
            "combination": stock_combination,
            "model": stock_model,
            "feature_columns": list(stock_features),
            "oos_metrics": stock_diagnostics,
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
            "selection": stock_report["final_selection"],
        },
        "common_oos": {
            "dates": int(full_output["bas_dd"].nunique()),
            "first_date": str(full_output["bas_dd"].min()),
            "last_date": str(full_output["bas_dd"].max()),
            "full_candidate_rows": int(len(full_output)),
            "supplementary_top5_rows": int(len(ranking)),
        },
        "full_prediction_summary": full_output_summary,
        "all_candidate_stock_hit_rate": full_output_summary["stock_hit_rate"],
        "all_candidate_index_direction_match_rate": index_direction_match_rate,
        "probability_quantile_summary": cutoff_summary.to_dict(orient="records"),
        "random_ranking_baseline": (
            random_summary.astype(object)
            .where(random_summary.notna(), None)
            .to_dict(orient="records")
        ),
        "top5_by_index_direction": _direction_summary(ranking),
        "output_contract": {
            "keys": ["fold", "bas_dd", "code"],
            "index_prediction": [
                "index_prediction",
                "index_p_down",
                "index_p_neutral",
                "index_p_up",
            ],
            "stock_prediction": [
                "stock_prediction",
                "p_down",
                "p_neutral",
                "p_up",
                "actual_label",
                "stock_hit",
            ],
            "display_order": [
                "sector_market_cap_rank",
                "industry_stock_rank",
                "code",
            ],
            "hit_rates": ["sector_hit_rate", "daily_hit_rate"],
            "supplementary_probability_ranking": [
                "industry_index_name",
                "selected_probability",
                "index_direction_rank",
            ],
            "execution": [
                "entry_bas_dd",
                "entry_adj_open",
                "exit_bas_dd",
                "exit_adj_open",
            ],
            "position_policy": (
                "KOSPI200 상승 예측과 개별종목 상승 예측이 모두 성립한 종목만 매수, "
                "그 외는 현금"
            ),
        },
        "stock_backtest": stock_backtest,
        "local_full_output_path": str(LOCAL_FULL_OUTPUT_PATH.relative_to(ROOT)).replace(
            "\\", "/"
        ),
        "local_ranking_path": str(LOCAL_RANKING_PATH.relative_to(ROOT)).replace(
            "\\", "/"
        ),
        "note": (
            "업종 10×종목 5의 최대 50종목은 모두 기본 출력에 남긴다. Top 1·3·5는 "
            "확률 분위 부가 실험으로 분리한다. 실제 매수는 지수와 종목이 모두 상승으로 "
            "예측된 경우에만 수행한다."
        ),
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )

    print(f"[4/4] 결과 저장: {REPORT_PATH.relative_to(ROOT)}", flush=True)
    print(f"      전체 후보 방향 적중률: {baseline_hit_rate:.4f}", flush=True)
    print(
        f"      전체 후보 종목 적중률 {full_output_summary['stock_hit_rate']:.4f}",
        flush=True,
    )
    if full_output_summary["up_precision"] is not None:
        print(
            f"      상승 클래스 Precision {full_output_summary['up_precision']:.4f}",
            flush=True,
        )
    for row in report["probability_quantile_summary"]:
        print(
            f"      Top {row['top_n']}: 적중률 {row['direction_hit_rate']:.4f} · "
            f"평균 선택확률 {row['mean_selected_probability']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backtest-only",
        action="store_true",
        help="모델을 다시 학습하지 않고 저장된 랭킹의 비용 백테스트만 갱신합니다.",
    )
    arguments = parser.parse_args()
    if arguments.backtest_only:
        refresh_saved_backtests()
    else:
        main()
