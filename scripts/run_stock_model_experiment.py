"""HF 개발본으로 개별종목 후보·패널·4모델 OOS 실험을 재현한다."""

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

from evaluation.horizon import HOLDOUT_START  # noqa: E402
from features.stock_model_dataset import (  # noqa: E402
    STOCK_FEATURE_COLUMNS,
    StockModelDataset,
    build_sector_stock_model_dataset,
)
from models.experiment import MODEL_BUILDERS  # noqa: E402
from models.stock_experiment import evaluate_stock_models  # noqa: E402
from models.stock_ranking import add_probability_ranks  # noqa: E402
from supply.stock_training_universe import (  # noqa: E402
    build_sector_candidate_frame,
    filter_extreme_adjusted_returns,
)

HF_ROOT = ROOT / "data" / "raw" / "hf_snapshot"
DAILY_PATH = HF_ROOT / "full" / "daily_price_dev.parquet"
INDEX_PATH = HF_ROOT / "full" / "index_price_dev.parquet"
LOCAL_OOS_PATH = ROOT / "data" / "raw" / "stock_model_oos.parquet"
PANEL_CACHE_PATH = ROOT / "data" / "raw" / "stock_model_panel.parquet"
PANEL_CACHE_META_PATH = ROOT / "data" / "raw" / "stock_model_panel.meta.json"
REPORT_PATH = ROOT / "reports" / "stock_model_experiment.json"

DAILY_COLUMNS = (
    "bas_dd",
    "code",
    "name",
    "market",
    "market_cap",
    "industry",
    "adj_open",
    "adj_high",
    "adj_low",
    "adj_close",
    "volume",
)
INDEX_COLUMNS = ("bas_dd", "index_name", "index_class", "market_cap")


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
        ROOT / "supply" / "stock_training_universe.py",
        Path(__file__).resolve(),
    )
    digest = hashlib.sha256()
    for path in code_paths:
        digest.update(path.read_bytes())
    return {
        "daily_sha256": _sha256(DAILY_PATH),
        "index_sha256": _sha256(INDEX_PATH),
        "panel_code_sha256": digest.hexdigest(),
        "features": list(STOCK_FEATURE_COLUMNS),
    }


def _read_cached_panel(signature: dict[str, object]) -> StockModelDataset | None:
    """서명이 정확히 같은 로컬 중간 패널만 읽는다."""

    if not PANEL_CACHE_PATH.exists() or not PANEL_CACHE_META_PATH.exists():
        return None
    metadata = json.loads(PANEL_CACHE_META_PATH.read_text(encoding="utf-8"))
    if metadata != signature:
        return None
    frame = pd.read_parquet(PANEL_CACHE_PATH)
    required = {"bas_dd", "label_numeric", *STOCK_FEATURE_COLUMNS}
    if required - set(frame.columns):
        return None
    if frame.empty or (frame["bas_dd"].astype("string") >= HOLDOUT_START).any():
        return None
    return StockModelDataset(frame=frame)


def _write_panel_cache(dataset: StockModelDataset, signature: dict[str, object]) -> None:
    """HF 원천이 아닌 재생성 가능한 중간 패널을 data/raw에만 저장한다."""

    PANEL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.frame.to_parquet(PANEL_CACHE_PATH, index=False)
    PANEL_CACHE_META_PATH.write_text(
        json.dumps(signature, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _model_summary(outer_results: pd.DataFrame) -> list[dict[str, object]]:
    metrics = ("accuracy", "macro_f1", "down_recall", "core_harmonic_mean")
    rows: list[dict[str, object]] = []
    for model, group in outer_results.groupby("model", sort=False):
        row: dict[str, object] = {"model": model, "folds": int(len(group))}
        for metric in metrics:
            row[metric] = float(group[metric].mean())
            row[f"{metric}_fold_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return sorted(rows, key=lambda item: float(item["core_harmonic_mean"]), reverse=True)


def load_stock_model_dataset(
    feature_columns: tuple[str, ...] = STOCK_FEATURE_COLUMNS,
) -> StockModelDataset:
    """HF full parquet에서 후보·라벨을 만들고 요청한 피처 조합만 선택한다."""

    unknown = set(feature_columns) - set(STOCK_FEATURE_COLUMNS)
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
        candidates = filter_extreme_adjusted_returns(candidates, daily)
        base_dataset = build_sector_stock_model_dataset(daily, candidates)
        _write_panel_cache(base_dataset, signature)

    return StockModelDataset(
        frame=base_dataset.frame,
        feature_columns=tuple(feature_columns),
    )


def main() -> None:
    """고정된 HF 경로만 읽어 실험하고 요약 JSON을 쓴다."""

    for path in (DAILY_PATH, INDEX_PATH):
        if not path.exists():
            raise FileNotFoundError(f"HF 최신본을 먼저 내려받아야 합니다: {path}")

    print("[1/5] HF KOSPI 종목·지수 parquet 읽기", flush=True)
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
    print(f"      종목 {len(daily):,}행 · 지수 {len(indices):,}행", flush=True)

    print("[2/5] 업종 상위 10 × 업종별 상위 5 후보 선정", flush=True)
    candidates = build_sector_candidate_frame(daily, indices)
    candidate_rule = dict(candidates.attrs["candidate_rule"])
    candidates = filter_extreme_adjusted_returns(candidates, daily)
    extreme_rule = dict(candidates.attrs["extreme_return_filter"])
    print(
        f"      후보 {len(candidates):,}행 · {candidates['bas_dd'].nunique():,}거래일 · "
        f"극단값 제거 {extreme_rule['removed_candidate_rows']}행",
        flush=True,
    )

    print("[3/5] 종목 전체 시계열 피처와 T+1→T+6 라벨 생성", flush=True)
    dataset = build_sector_stock_model_dataset(daily, candidates)
    panel_rule = dict(dataset.frame.attrs["stock_panel"])
    label_counts = dataset.frame["label"].value_counts().to_dict()
    print(
        f"      학습표 {len(dataset.frame):,}행 · {panel_rule['dates']:,}거래일 · "
        f"{panel_rule['stocks']:,}종목",
        flush=True,
    )

    print("[4/5] 날짜 그룹 expanding 12폴드 · 4모델 학습", flush=True)
    results = []
    for model_name, builder in MODEL_BUILDERS.items():
        print(f"      {model_name} 시작", flush=True)
        result = evaluate_stock_models(dataset, model_builders={model_name: builder})
        results.append(result)
        score = result.outer_results["core_harmonic_mean"].mean()
        print(f"      {model_name} 완료 · 조화평균 {score:.4f}", flush=True)

    inner_results = pd.concat([result.inner_results for result in results], ignore_index=True)
    outer_results = pd.concat([result.outer_results for result in results], ignore_index=True)
    oos_predictions = pd.concat(
        [result.oos_predictions for result in results], ignore_index=True
    )
    ranked_predictions = add_probability_ranks(oos_predictions)
    LOCAL_OOS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ranked_predictions.to_parquet(LOCAL_OOS_PATH, index=False)

    selected_weights = (
        outer_results.assign(
            selected_class_weight=outer_results["selected_class_weight"].fillna("None")
        )
        .groupby(["model", "selected_class_weight"], sort=False)
        .size()
        .rename("folds")
        .reset_index()
        .to_dict(orient="records")
    )
    actual = oos_predictions.drop_duplicates(["bas_dd", "code"])["label_numeric"]
    majority_count = int(actual.value_counts().max())
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repo": "qurious-quant/alphastack-krx-dev",
            "daily_path": "full/daily_price_dev.parquet",
            "daily_sha256": _sha256(DAILY_PATH),
            "index_path": "full/index_price_dev.parquet",
            "index_sha256": _sha256(INDEX_PATH),
            "holdout_start": HOLDOUT_START,
        },
        "candidate_rule": candidate_rule,
        "extreme_return_filter": extreme_rule,
        "panel": panel_rule,
        "features": list(STOCK_FEATURE_COLUMNS),
        "label_distribution": {str(key): int(value) for key, value in label_counts.items()},
        "validation": {
            "window": "expanding",
            "folds": 12,
            "minimum_initial_train_dates": 750,
            "valid_dates_per_fold": 60,
            "gap_dates": 5,
            "class_weight_candidates": [None, "balanced"],
            "selection": "Accuracy·Macro F1·하락 Recall 조화평균 최대",
        },
        "majority_baseline_accuracy_on_oos": majority_count / int(len(actual)),
        "model_summary": _model_summary(outer_results),
        "selected_class_weight_counts": selected_weights,
        "outer_fold_results": outer_results.to_dict(orient="records"),
        "inner_trial_count": int(len(inner_results)),
        "outer_fit_count": int(len(outer_results)),
        "local_oos_path": str(LOCAL_OOS_PATH.relative_to(ROOT)).replace("\\", "/"),
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(f"[5/5] 결과 저장: {REPORT_PATH.relative_to(ROOT)}", flush=True)
    for row in report["model_summary"]:
        print(
            f"      {row['model']}: Acc {row['accuracy']:.4f} · "
            f"Macro F1 {row['macro_f1']:.4f} · 하락 Recall {row['down_recall']:.4f} · "
            f"조화평균 {row['core_harmonic_mean']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
