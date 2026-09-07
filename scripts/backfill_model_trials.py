"""보존된 모델 리포트와 노트북에서 과거 ``fit`` 원장을 복원한다.

이 스크립트가 세는 단위는 계획서의 F-13과 같은 최상위 모델 ``fit`` 1회다.
계측 코드가 없던 시기의 실패·중간 재실행은 사후에 증명할 수 없으므로, 보존된
완료 리포트로 확인되는 성공 호출만 기록한다. 그 한계는 모든 행의 ``audit``에 남긴다.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_model_notebooks import MODELS, VARIANTS, notebook_targets  # noqa: E402

DEFAULT_OUTPUT = ROOT / "reports" / "trials.jsonl"
EXPECTED_COUNTS = {
    "kospi200_window_selection": 192,
    "kospi200_class_weight_tuning": 144,
    "kospi200_combination_sweep": 3456,
    "kospi200_market_internals": 144,
    "kospi200_combined_market_internals": 144,
    "stock_combination_a": 144,
}
AUDIT = {
    "origin": "historical_backfill",
    "coverage": "verified_successful_fits_only",
    "limitation": "계측 전 실패 호출과 보존되지 않은 중간 재실행은 복원할 수 없음",
}


def _load_json(relative_path: str) -> dict[str, Any]:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def _clean_json(value: Any) -> Any:
    """NaN을 JSON 표준의 null로 바꾸고 pandas 값을 기본 파이썬 값으로 바꾼다."""

    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return _clean_json(value.item())
    return value


def _record(
    *,
    trial_id: str,
    run: str,
    track: str,
    report_path: str,
    report: dict[str, Any],
    fit: dict[str, Any],
    experiment: dict[str, Any],
    metrics: dict[str, Any] | None = None,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    created_at = report.get("created_at_utc", report.get("generated_at_utc"))
    result = {
        "schema_version": 1,
        "trial_id": trial_id,
        "status": "success",
        "track": track,
        "run": run,
        "report_created_at_utc": created_at,
        "report_path": report_path,
        "dataset_source": report.get("source", {}),
        "experiment": experiment,
        "fit": fit,
        "metrics": metrics,
        "evidence": evidence or {},
        "audit": AUDIT,
    }
    return _clean_json(result)


def _window_trials() -> list[dict[str, Any]]:
    path = "reports/window_selection.json"
    report = _load_json(path)
    rows = report["fold_results"]
    records = []
    for row in rows:
        window = row["window"]
        model = row["model"]
        fold = int(row["fold"])
        metrics = {
            key: row[key]
            for key in (
                "accuracy",
                "macro_f1",
                "down_recall",
                "core_harmonic_mean",
                "strategy_sharpe_net",
                "buyhold_sharpe_net",
                "delta_sharpe_net",
                "all_cash",
            )
        }
        records.append(
            _record(
                trial_id=f"kospi200-window-{window}-{model}-fold-{fold:02d}",
                run="kospi200_window_selection",
                track="kospi200",
                report_path=path,
                report=report,
                experiment={**report["representative"], "window": window},
                fit={
                    "phase": "window_evaluation",
                    "model": model,
                    "fold": fold,
                    "class_weight": report["rules"]["class_weight"],
                    "train_size": row["train_size"],
                    "valid_size": row["valid_size"],
                    "train_end": row["train_end"],
                    "valid_start": row["valid_start"],
                    "valid_end": row["valid_end"],
                },
                metrics=metrics,
            )
        )
    return records


def _nested_trials_from_detailed_report(
    *,
    report_path: str,
    run: str,
    track: str,
    experiment: dict[str, Any],
) -> list[dict[str, Any]]:
    """inner와 outer 결과가 모두 보존된 class-weight 리포트를 복원한다."""

    report = _load_json(report_path)
    records = []
    for row in report["inner_results"]:
        model = row["model"]
        fold = int(row["fold"])
        weight = row["class_weight"]
        weight_id = "none" if _clean_json(weight) is None else str(weight)
        metrics = {
            key: row[key]
            for key in ("accuracy", "macro_f1", "down_recall", "core_harmonic_mean")
        }
        records.append(
            _record(
                trial_id=f"{run}-{model}-fold-{fold:02d}-inner-{weight_id}",
                run=run,
                track=track,
                report_path=report_path,
                report=report,
                experiment=experiment,
                fit={
                    "phase": "inner_class_weight_selection",
                    "model": model,
                    "fold": fold,
                    "class_weight": weight,
                    "train_size": row["inner_train_size"],
                    "valid_size": row["inner_valid_size"],
                    "train_end": row["inner_train_end"],
                    "valid_start": row["inner_valid_start"],
                    "valid_end": row["inner_valid_end"],
                },
                metrics=metrics,
            )
        )
    for row in report["outer_results"]:
        model = row["model"]
        fold = int(row["fold"])
        metrics = {
            key: row[key]
            for key in (
                "accuracy",
                "macro_f1",
                "down_recall",
                "core_harmonic_mean",
                "strategy_sharpe_net",
                "buyhold_sharpe_net",
                "delta_sharpe_net",
                "all_cash",
            )
        }
        records.append(
            _record(
                trial_id=f"{run}-{model}-fold-{fold:02d}-outer",
                run=run,
                track=track,
                report_path=report_path,
                report=report,
                experiment=experiment,
                fit={
                    "phase": "outer_evaluation",
                    "model": model,
                    "fold": fold,
                    "class_weight": row["selected_class_weight"],
                    "train_size": row["train_size"],
                    "valid_size": row["valid_size"],
                    "train_end": row["train_end"],
                    "valid_start": row["valid_start"],
                    "valid_end": row["valid_end"],
                },
                metrics=metrics,
            )
        )
    return records


def _notebook_outer_rows(path: Path) -> list[dict[str, Any]]:
    """실행 노트북에 보존된 12개 outer-fold 표를 읽는다."""

    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = [
        cell
        for cell in notebook["cells"]
        if "nested = evaluate_nested_class_weights" in "".join(cell.get("source", []))
    ]
    if len(cells) != 1 or len(cells[0].get("outputs", [])) < 1:
        raise RuntimeError(f"outer-fold 실행 결과를 찾을 수 없습니다: {path}")
    html = "".join(cells[0]["outputs"][0]["data"]["text/html"])
    frame = pd.read_html(StringIO(html))[0].drop(columns=["Unnamed: 0"], errors="ignore")
    if len(frame) != 12:
        raise RuntimeError(f"outer-fold 결과가 12개가 아닙니다: {path} ({len(frame)})")
    return frame.to_dict(orient="records")


def _variant_for(return_features: list[str]) -> str:
    configured = tuple(return_features)
    for variant, (_suffix, features, _description) in VARIANTS.items():
        if configured == tuple(features):
            return variant
    raise ValueError(f"알 수 없는 수익률 피처 조합입니다: {return_features}")


def _sweep_notebook_lookup() -> dict[tuple[str, str, str], Path]:
    lookup = {}
    for combination, variant, directory in notebook_targets():
        for model, filename in MODELS.items():
            lookup[(combination, variant, model)] = directory / filename
    return lookup


def _combination_sweep_trials() -> list[dict[str, Any]]:
    path = "reports/model_sweep.json"
    report = _load_json(path)
    notebooks = _sweep_notebook_lookup()
    records = []
    for item in report["experiments"]:
        experiment = item["experiment"]
        combination = experiment["combination"]
        model = experiment["model"]
        variant = _variant_for(experiment["return_features"])
        notebook_path = notebooks[(combination, variant, model)]
        outer_rows = _notebook_outer_rows(notebook_path)
        base_id = f"kospi200-sweep-{combination}-{variant}-{model}"
        common_experiment = {
            "combination": combination,
            "return_features": experiment["return_features"],
            "feature_columns": item["summary"]["feature_columns"],
            "variant": variant,
        }
        evidence = {
            "notebook_path": notebook_path.relative_to(ROOT).as_posix(),
            "outer_metrics_precision_decimals": 4,
        }
        for row in outer_rows:
            fold = int(row["fold"])
            inner_train_size = int(row["train_size"]) - 65
            for weight, weight_id in ((None, "none"), ("balanced", "balanced")):
                records.append(
                    _record(
                        trial_id=f"{base_id}-fold-{fold:02d}-inner-{weight_id}",
                        run="kospi200_combination_sweep",
                        track="kospi200",
                        report_path=path,
                        report=report,
                        experiment=common_experiment,
                        fit={
                            "phase": "inner_class_weight_selection",
                            "model": model,
                            "fold": fold,
                            "class_weight": weight,
                            "train_size": inner_train_size,
                            "valid_size": 60,
                            "gap": 5,
                        },
                        evidence=evidence,
                    )
                )
            metrics = {
                key: row[key]
                for key in (
                    "accuracy",
                    "macro_f1",
                    "down_recall",
                    "core_harmonic_mean",
                    "delta_sharpe_net",
                    "all_cash",
                )
            }
            records.append(
                _record(
                    trial_id=f"{base_id}-fold-{fold:02d}-outer",
                    run="kospi200_combination_sweep",
                    track="kospi200",
                    report_path=path,
                    report=report,
                    experiment=common_experiment,
                    fit={
                        "phase": "outer_evaluation",
                        "model": model,
                        "fold": fold,
                        "class_weight": row["selected_class_weight"],
                        "train_size": row["train_size"],
                        "valid_size": 60,
                        "train_end": str(row["train_end"]),
                        "valid_start": str(row["valid_start"]),
                        "valid_end": str(row["valid_end"]),
                    },
                    metrics=metrics,
                    evidence=evidence,
                )
            )
    return records


def _nested_trials_from_outer_report(
    *, report_path: str, run: str, track: str
) -> list[dict[str, Any]]:
    """inner 점수는 없고 outer 결과와 fit 횟수만 보존된 리포트를 복원한다."""

    report = _load_json(report_path)
    outer_rows = report.get("outer_results", report.get("outer_fold_results"))
    if outer_rows is None:
        raise ValueError(f"outer 결과가 없습니다: {report_path}")
    dataset = report.get("dataset", report.get("panel", {}))
    features = report.get("features", dataset.get("feature_columns", []))
    experiment = {
        "combination": dataset.get("combination", "A" if track == "stock" else None),
        "feature_columns": features,
    }
    records = []
    for row in outer_rows:
        model = row["model"]
        fold = int(row["fold"])
        base_id = f"{run}-{model}-fold-{fold:02d}"
        for weight, weight_id in ((None, "none"), ("balanced", "balanced")):
            records.append(
                _record(
                    trial_id=f"{base_id}-inner-{weight_id}",
                    run=run,
                    track=track,
                    report_path=report_path,
                    report=report,
                    experiment=experiment,
                    fit={
                        "phase": "inner_class_weight_selection",
                        "model": model,
                        "fold": fold,
                        "class_weight": weight,
                        "valid_size": 60,
                        "gap": 5,
                    },
                )
            )
        metrics = {
            key: row[key]
            for key in ("accuracy", "macro_f1", "down_recall", "core_harmonic_mean")
        }
        for optional in (
            "strategy_sharpe_net",
            "buyhold_sharpe_net",
            "delta_sharpe_net",
            "all_cash",
        ):
            if optional in row:
                metrics[optional] = row[optional]
        train_size = row.get("train_size", row.get("train_rows"))
        valid_size = row.get("valid_size", row.get("valid_rows"))
        records.append(
            _record(
                trial_id=f"{base_id}-outer",
                run=run,
                track=track,
                report_path=report_path,
                report=report,
                experiment=experiment,
                fit={
                    "phase": "outer_evaluation",
                    "model": model,
                    "fold": fold,
                    "class_weight": row["selected_class_weight"],
                    "train_size": train_size,
                    "valid_size": valid_size,
                    "train_end": row["train_end"],
                    "valid_start": row["valid_start"],
                    "valid_end": row["valid_end"],
                },
                metrics=metrics,
            )
        )
    return records


def build_historical_trials() -> list[dict[str, Any]]:
    """현재 저장소의 보존 결과로 확인되는 성공 fit 4,224개를 만든다."""

    records = []
    records.extend(_window_trials())
    class_weight_path = "reports/class_weight_tuning.json"
    class_weight_report = _load_json(class_weight_path)
    records.extend(
        _nested_trials_from_detailed_report(
            report_path=class_weight_path,
            run="kospi200_class_weight_tuning",
            track="kospi200",
            experiment=class_weight_report["representative"],
        )
    )
    records.extend(_combination_sweep_trials())
    records.extend(
        _nested_trials_from_outer_report(
            report_path="reports/market_feature_experiment.json",
            run="kospi200_market_internals",
            track="kospi200",
        )
    )
    records.extend(
        _nested_trials_from_outer_report(
            report_path="reports/combined_market_feature_experiment.json",
            run="kospi200_combined_market_internals",
            track="kospi200",
        )
    )
    records.extend(
        _nested_trials_from_outer_report(
            report_path="reports/stock_model_experiment.json",
            run="stock_combination_a",
            track="stock",
        )
    )
    _validate_records(records)
    return records


def _validate_records(records: list[dict[str, Any]]) -> None:
    counts = Counter(record["run"] for record in records)
    if counts != Counter(EXPECTED_COUNTS):
        raise RuntimeError(f"fit 횟수가 보존 리포트와 다릅니다: {dict(counts)}")
    trial_ids = [record["trial_id"] for record in records]
    if len(trial_ids) != len(set(trial_ids)):
        duplicates = [key for key, count in Counter(trial_ids).items() if count > 1]
        raise RuntimeError(f"trial_id가 중복됩니다: {duplicates[:5]}")
    for record in records:
        json.dumps(record, ensure_ascii=False, allow_nan=False)


def write_trials(records: list[dict[str, Any]], output: Path) -> None:
    """검증을 끝낸 뒤 전체 JSONL을 한 번에 교체한다."""

    _validate_records(records)
    text = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
        for record in records
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    records = build_historical_trials()
    write_trials(records, args.output)
    counts = Counter(record["run"] for record in records)
    print(f"성공 fit {len(records):,}회 기록: {args.output}")
    for run, count in counts.items():
        print(f"- {run}: {count:,}회")
    print("주의: 계측 전 실패 호출과 보존되지 않은 중간 재실행은 복원 범위에 없습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
