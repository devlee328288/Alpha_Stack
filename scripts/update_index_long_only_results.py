"""KOSPI200 long-only 100후보 결과로 best result 폴더와 문서를 갱신한다."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import nbformat

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.final_models import select_best_long_only_index_model  # noqa: E402
from scripts.build_model_notebooks import (  # noqa: E402
    COMBINATION_DIRS,
    MODELS,
    VARIANTS,
    experiment_directory,
)
from scripts.run_index_long_only_selection import selection_audit  # noqa: E402
from scripts.update_best_model_results import FEATURE_DESCRIPTIONS  # noqa: E402
from scripts.update_final_model_docs import update_final_model_docs  # noqa: E402

REPORT_PATH = ROOT / "reports" / "index_long_only_selection.json"
BEST_ROOT = ROOT / "notebooks" / "04-모델" / "KOSPI200" / "실험" / "조합별 best result"
RANK_PREFIXES = ("⭐", "❤️", "♡")


def _variant_name(record: dict[str, Any]) -> str:
    returns = tuple(record["experiment"].get("return_features", []))
    for name, (_suffix, configured, _description) in VARIANTS.items():
        if returns == configured:
            return name
    raise ValueError(f"알 수 없는 수익률 피처 구성입니다: {returns}")


def _variant_label(record: dict[str, Any]) -> str:
    suffix = VARIANTS[_variant_name(record)][0]
    return "기본 조합" if not suffix else suffix.removeprefix(" + ")


def select_best_results(
    candidates: list[dict[str, Any]], combination: str
) -> dict[str, dict[str, Any]]:
    """한 조합에서 모델별로 보고서 순위가 가장 높은 피처 변형을 고른다."""

    selected: dict[str, dict[str, Any]] = {}
    for model_name in MODELS:
        matches = [
            record
            for record in candidates
            if record["experiment"]["combination"] == combination
            and record["experiment"]["model"] == model_name
        ]
        if not matches:
            raise ValueError(f"조합 {combination} {model_name} long-only 결과가 없습니다.")
        selected[model_name] = min(matches, key=lambda record: int(record["rank"]))
    return selected


def _result_notebook(
    combination: str,
    model_name: str,
    selected: dict[str, Any],
    candidates: list[dict[str, Any]],
):
    variant = _variant_name(selected)
    source_filename = MODELS[model_name]
    source_path = experiment_directory(combination, variant)
    source_link = (
        Path("../..") / source_path.relative_to(BEST_ROOT.parent) / source_filename
    ).as_posix()
    rows = []
    for record in sorted(candidates, key=lambda item: int(item["rank"])):
        summary = record["summary"]
        marker = " **(선정)**" if record is selected else ""
        rows.append(
            f"| {_variant_label(record)}{marker} | {record['rank']} | "
            f"{summary['passes_operational_gate']} | {summary['accuracy']:.4f} | "
            f"{summary['accuracy_minus_training_majority_baseline']:+.4f} | "
            f"{summary['baseline_win_folds']}/{summary['folds']} | "
            f"{summary['pr_auc_up']:.4f} | {summary['up_precision']:.4f} | "
            f"{summary['up_recall']:.4f} | {summary['buy_signals']} | "
            f"{summary['delta_sharpe_net_median']:.4f} |"
        )
    summary = selected["summary"]
    accuracy_delta = summary["accuracy_minus_training_majority_baseline"]
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            f"# 조합{combination} · {model_name} long-only best result\n\n"
            f"선정 실험: **{_variant_label(selected)}**  \n"
            "선정 기준: 운영 기준선 게이트 → 상승 PR-AUC → 비용 차감 ΔSharpe → "
            "상승 Precision  \n"
            f"[피처·모델 원본 노트북]({source_link})"
        ),
        nbformat.v4.new_markdown_cell(
            "## 피처 변형 비교\n\n"
            "| 실험 | 전체 순위 | 게이트 | Accuracy | 기준선 대비 | 승리 폴드 | "
            "상승 PR-AUC | 상승 Precision | 상승 Recall | 매수 신호 | ΔSharpe 중앙값 |\n"
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
            + "\n".join(rows)
        ),
        nbformat.v4.new_markdown_cell(
            "## 선정 결과\n\n"
            "| 지표 | 값 |\n|---|---:|\n"
            f"| Accuracy | {summary['accuracy']:.4f} |\n"
            f"| 학습 최빈 기준선 | {summary['training_majority_baseline_accuracy']:.4f} |\n"
            f"| 기준선 대비 Accuracy | {accuracy_delta:+.4f} |\n"
            f"| 기준선 승리 | {summary['baseline_win_folds']}/{summary['folds']} |\n"
            f"| Macro F1 | {summary['macro_f1']:.4f} |\n"
            f"| 상승 PR-AUC | {summary['pr_auc_up']:.4f} |\n"
            f"| 상승 Precision | {summary['up_precision']:.4f} |\n"
            f"| 상승 Recall | {summary['up_recall']:.4f} |\n"
            f"| 매수 신호 | {summary['buy_signals']} |\n"
            f"| 비용 차감 ΔSharpe 폴드 중앙값 | {summary['delta_sharpe_net_median']:.4f} |\n"
            f"| 상승 임계값 중앙값 | {summary['selected_up_threshold_median']:.4f} |\n"
            f"| 상승 임계값 범위 | {summary['selected_up_threshold_min']:.4f}~"
            f"{summary['selected_up_threshold_max']:.4f} |"
        ),
        nbformat.v4.new_markdown_cell(
            "## 해석 범위\n\n"
            "각 외부 폴드의 상승 임계값과 class weight는 그 폴드의 학습구간 안에 둔 "
            "내부 검증에서만 골랐다. 이 값은 개발구간 공통 OOS 결과이며 봉인 홀드아웃은 "
            "사용하지 않았다. 이슈 #203 확정 전까지는 잠정 비교 결과다."
        ),
    ]
    return notebook


def _feature_markdown(
    combination: str,
    selected: dict[str, dict[str, Any]],
    overall_best: str,
    marker: str,
    source: dict[str, Any],
) -> str:
    base_features = [
        feature
        for feature in next(iter(selected.values()))["feature_columns"]
        if feature not in {"daily_return", "five_day_return"}
    ]
    feature_lines = "\n".join(
        f"- `{feature}`: {FEATURE_DESCRIPTIONS.get(feature, '공통 피처 함수로 계산')}"
        for feature in base_features
    )
    result_lines = []
    result_header = (
        "| 모델 | 선정 실험 | 전체 순위 | 게이트 | Accuracy | 기준선 대비 | 승리 폴드 | "
        "상승 PR-AUC | 상승 Precision | ΔSharpe 중앙값 |"
    )
    for model_name, record in selected.items():
        summary = record["summary"]
        model_marker = f"{marker} " if model_name == overall_best else ""
        result_lines.append(
            f"| {model_marker}{model_name} | {_variant_label(record)} | {record['rank']} | "
            f"{summary['passes_operational_gate']} | {summary['accuracy']:.4f} | "
            f"{summary['accuracy_minus_training_majority_baseline']:+.4f} | "
            f"{summary['baseline_win_folds']}/{summary['folds']} | "
            f"{summary['pr_auc_up']:.4f} | {summary['up_precision']:.4f} | "
            f"{summary['delta_sharpe_net_median']:.4f} |"
        )
    return f"""# 조합{combination} long-only 피처 선정

## 기본 피처

{feature_lines}

`daily_return`과 `five_day_return` 추가 여부까지 포함한 모든 정의된 변형을 비교했습니다.

## 조합 내 모델별 최우수 결과

{result_header}
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(result_lines)}

`{marker}`는 조합{combination} 안에서 새 long-only 기준 순위가 가장 높은 모델입니다.
순위는 운영 기준선 게이트를 먼저 적용한 뒤 상승 PR-AUC, 비용 차감 ΔSharpe 폴드 중앙값,
상승 Precision 순으로 정했습니다. 하락 Recall은 보고만 하고 선정축에는 넣지 않았습니다.

## 데이터와 평가 범위

- HF 로컬 스냅샷 생성시각: `{source.get('generated_at', '')}`
- `full/index_price_dev.parquet` SHA-256: `{source['index_sha256']}`
- 홀드아웃 사용: 없음
- 공통 expanding 12폴드 · 최초 학습 750거래일 · 검증 60거래일 · gap 5거래일
- 상승 임계값·class weight: 각 외부 학습구간 안의 내부 60거래일에서만 선택
- 이슈 #203 팀 확정 전 잠정 결과
"""


def _strip_rank_prefix(name: str) -> str:
    for prefix in RANK_PREFIXES:
        if name.startswith(prefix):
            return name.removeprefix(prefix)
    return name


def sync_long_only_best_results(
    report: dict[str, Any],
    *,
    best_root: Path = BEST_ROOT,
) -> dict[str, Path]:
    """완료 보고서의 조합별 순위대로 기존 일곱 폴더를 이름·내용까지 동기화한다."""

    select_best_long_only_index_model(report)
    candidates = report["candidates"]
    selected_by_combination = {
        combination: select_best_results(candidates, combination)
        for combination in COMBINATION_DIRS
    }
    ranked_combinations = sorted(
        COMBINATION_DIRS,
        key=lambda combination: min(
            int(record["rank"]) for record in selected_by_combination[combination].values()
        ),
    )
    combination_markers = {
        combination: marker
        for combination, marker in zip(ranked_combinations[:3], RANK_PREFIXES, strict=True)
    }

    current: dict[str, Path] = {}
    for directory in best_root.iterdir():
        if not directory.is_dir():
            continue
        plain = _strip_rank_prefix(directory.name)
        if plain.startswith("조합") and plain[2:] in COMBINATION_DIRS:
            combination = plain[2:]
            if combination in current:
                raise RuntimeError(f"조합 {combination} best result 폴더가 중복됐습니다.")
            current[combination] = directory
    missing = set(COMBINATION_DIRS) - set(current)
    if missing:
        raise RuntimeError(f"best result 폴더가 없습니다: {sorted(missing)}")

    written: dict[str, Path] = {}
    for combination in COMBINATION_DIRS:
        marker = combination_markers.get(combination, "")
        target = best_root / f"{marker}조합{combination}"
        directory = current[combination]
        if directory != target:
            if target.exists():
                raise RuntimeError(f"새 best result 폴더 이름이 이미 존재합니다: {target}")
            directory.rename(target)
            directory = target

        selected = selected_by_combination[combination]
        overall_best = min(selected, key=lambda model: int(selected[model]["rank"]))
        removable_names = {
            f"{prefix}{filename}"
            for prefix in (*RANK_PREFIXES, "")
            for filename in MODELS.values()
        }
        for path in directory.iterdir():
            if path.is_file() and path.name in removable_names:
                path.unlink()
        for model_name, record in selected.items():
            filename = MODELS[model_name]
            if model_name == overall_best:
                filename = f"{marker or '⭐'}{filename}"
            variants = [
                candidate
                for candidate in candidates
                if candidate["experiment"]["combination"] == combination
                and candidate["experiment"]["model"] == model_name
            ]
            nbformat.write(
                _result_notebook(combination, model_name, record, variants),
                directory / filename,
            )
        (directory / "피처선정.md").write_text(
            _feature_markdown(
                combination,
                selected,
                overall_best,
                marker or "⭐",
                report["source"],
            ),
            encoding="utf-8",
        )
        written[combination] = directory
    return written


def main() -> int:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    report["selection_policy"].update(selection_audit(report["candidates"]))
    report["selection_policy"]["status"] = "pending_issue_216_statistical_review"
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    written = sync_long_only_best_results(report)
    update_final_model_docs(ROOT)
    for combination in written:
        # Windows CP949 콘솔은 하트 이모지가 든 실제 폴더 경로를 출력하지 못한다.
        print(f"조합 {combination}: best result 갱신")
    print("KOSPI200 최종모델 README도 long-only 잠정 1위로 갱신했습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
