"""개별종목 4모델 결과 노트북과 비교 노트북을 같은 형식으로 만든다."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_ROOT = ROOT / "notebooks" / "04-모델" / "개별종목" / "실험"
COMBINATION_DIRECTORY = "조합A_trend_momentum_volatility_volume_returns"
OUTPUT = EXPERIMENT_ROOT / COMBINATION_DIRECTORY
BEST_OUTPUT = EXPERIMENT_ROOT / "조합별 best result" / "조합A"
BASE_OUTPUT = EXPERIMENT_ROOT / "기본모델"
REPORT = ROOT / "reports" / "stock_model_experiment.json"

MODEL_FILES = {
    "LogisticRegression": "01.LogisticRegression.ipynb",
    "RandomForest": "02.RandomForest.ipynb",
    "XGBoost": "03.XGBoost.ipynb",
    "LightGBM": "04.LightGBM.ipynb",
}
MODEL_BUILDERS = {
    "LogisticRegression": ("models.logistic", "build_logistic_baseline"),
    "RandomForest": ("models.random_forest", "build_random_forest_baseline"),
    "XGBoost": ("models.xgboost", "build_xgboost_baseline"),
    "LightGBM": ("models.lightgbm", "build_lightgbm_baseline"),
}
KERNEL_METADATA = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
}


def _best_notebook(model_name: str, summary: dict[str, object]) -> nbformat.NotebookNode:
    """조합별 best-result에 둘 모델 요약 노트북을 만든다."""

    metric_row = (
        f"| {summary['accuracy']:.4f} | {summary['macro_f1']:.4f} | "
        f"{summary['down_recall']:.4f} | **{summary['core_harmonic_mean']:.4f}** |"
    )
    markdown = f"""# 개별종목 — {model_name}

## 실험 목적

KOSPI200 방향이 상승·보합·하락 중 어디인지 정해졌을 때, 같은 방향일 확률이 높은
개별종목을 찾기 위한 3분류 모델입니다.

후보는 매 거래일 **KOSPI 세부 업종지수 시가총액 상위 10개 × 업종별 KOSPI 보통주
시가총액 상위 5개**로 먼저 고정합니다. 업종의 미래 방향을 따로 예측하는 구조는 아닙니다.

## 공통 조건

| 항목 | 값 |
|---|---|
| 원천 | HF `full/daily_price_dev.parquet`, `full/index_price_dev.parquet` |
| 홀드아웃 | `20240901` 이후 접근 금지 |
| 라벨 | T일 판단 → T+1 `adj_open` 진입 → T+6 `adj_open` 평가, 종목 ±2% |
| 외부 검증 | 날짜 그룹 expanding 12폴드 |
| 최초 학습 | 750거래일 |
| 검증·gap | 폴드당 60거래일 · 직전 5거래일 제거 |
| class weight | 각 외부 폴드 내부에서 `None`과 `balanced` 재비교 |
| 선정 지표 | Accuracy·Macro F1·하락 Recall 조화평균 |

## OOS 결과

| Accuracy | Macro F1 | 하락 Recall | 핵심지표 조화평균 |
|---:|---:|---:|---:|
{metric_row}

아래 셀은 저장된 실측 리포트에서 이 모델의 폴드 결과와 class weight 선택 횟수를 다시
읽습니다. 학습 구현은 `models/stock_experiment.py`, 피처·라벨은
`features/stock_model_dataset.py`가 정본입니다.
"""
    code = f"""import json
from pathlib import Path

import pandas as pd

ROOT = Path.cwd()
while ROOT.parent != ROOT and not (ROOT / "reports" / "stock_model_experiment.json").exists():
    ROOT = ROOT.parent
report = json.loads((ROOT / "reports" / "stock_model_experiment.json").read_text(encoding="utf-8"))
model_name = {model_name!r}

folds = pd.DataFrame(report["outer_fold_results"])
display(folds.loc[folds["model"].eq(model_name)].reset_index(drop=True))

weights = pd.DataFrame(report["selected_class_weight_counts"])
display(weights.loc[weights["model"].eq(model_name)].reset_index(drop=True))
"""
    return nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_markdown_cell(markdown), nbformat.v4.new_code_cell(code)],
        metadata=KERNEL_METADATA,
    )


def _base_model_notebook(model_name: str) -> nbformat.NotebookNode:
    """피처 조합과 무관한 모델 생성 코드만 담은 노트북을 만든다."""

    module_name, builder_name = MODEL_BUILDERS[model_name]
    markdown = f"""# 개별종목 기본모델 — {model_name}

이 노트북은 피처·후보·분할을 정하지 않고 {model_name} 생성 코드만 제공합니다.
실제 생성 함수는 `models/`에 한 번만 두고 기본모델과 조합 노트북이 함께 가져옵니다.
"""
    code = f"""# 기본모델은 데이터에 손대지 않고 학습되지 않은 모델 생성 함수만 제공합니다.
from {module_name} import {builder_name}

MODEL_NAME = {model_name!r}
MODEL_BUILDER = {builder_name}

# class weight는 외부 폴드 안의 내부 검증에서 None과 balanced 중 다시 선택합니다.
default_model = MODEL_BUILDER(class_weight=None)
balanced_model = MODEL_BUILDER(class_weight="balanced")
print("기본모델:", MODEL_NAME)
print("생성 함수:", MODEL_BUILDER.__name__)
"""
    return nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_markdown_cell(markdown), nbformat.v4.new_code_cell(code)],
        metadata=KERNEL_METADATA,
    )


def _combination_notebook(
    model_name: str,
    feature_columns: tuple[str, ...],
) -> nbformat.NotebookNode:
    """기본모델을 불러오고 조합A 피처만 주입하는 실행 노트북을 만든다."""

    base_filename = MODEL_FILES[model_name]
    module_name, builder_name = MODEL_BUILDERS[model_name]
    markdown = f"""# 개별종목 조합A — {model_name}

`기본모델/{base_filename}`과 같은 `{module_name}.{builder_name}`을 가져오고
조합A 피처 12개를 주입합니다. 기본모델 코드는 `models/`에 한 번만 존재합니다.
후보·라벨·날짜 그룹 12폴드 실행은 모든 조합이 같은 공통 함수를 사용합니다.
"""
    setup_code = f"""# 1. 기본모델을 가져옵니다.
import sys
from pathlib import Path

import pandas as pd
from IPython.display import display

project_root = Path.cwd().resolve()
while project_root != project_root.parent and not (project_root / "pyproject.toml").is_file():
    project_root = project_root.parent
if not (project_root / "pyproject.toml").is_file():
    raise RuntimeError("프로젝트 루트를 찾지 못했습니다.")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from {module_name} import {builder_name}  # noqa: E402

MODEL_NAME = {model_name!r}
MODEL_BUILDER = {builder_name}
"""
    feature_code = f"""# 2. 조합A의 피처 값만 지정합니다.
FEATURE_COLUMNS = {feature_columns!r}

from models.stock_experiment import evaluate_stock_models  # noqa: E402
from models.stock_ranking import add_probability_ranks  # noqa: E402
from scripts.run_stock_model_experiment import load_stock_model_dataset  # noqa: E402

dataset = load_stock_model_dataset(FEATURE_COLUMNS)
print("학습 기간:", dataset.frame["bas_dd"].min(), "~", dataset.frame["bas_dd"].max())
print("학습 행·종목:", len(dataset.frame), dataset.frame["code"].nunique())
print("조합A 피처:", list(dataset.feature_columns))

result = evaluate_stock_models(
    dataset,
    model_builders={{MODEL_NAME: MODEL_BUILDER}},
)
fold_columns = [
    "fold",
    "selected_class_weight",
    "train_dates",
    "valid_start",
    "valid_end",
    "accuracy",
    "macro_f1",
    "down_recall",
    "core_harmonic_mean",
]
display(result.outer_results.loc[:, fold_columns].round(4))

metric_columns = ["accuracy", "macro_f1", "down_recall", "core_harmonic_mean"]
display(result.outer_results.loc[:, metric_columns].mean().to_frame("OOS 폴드 평균").round(4))

ranked = add_probability_ranks(result.oos_predictions)
cache_path = project_root / "data" / "raw" / f"stock_model_oos_{{MODEL_NAME}}.parquet"
ranked.to_parquet(cache_path, index=False)
print("OOS 확률·랭킹 저장:", cache_path)
"""
    return nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell(markdown),
            nbformat.v4.new_code_cell(setup_code),
            nbformat.v4.new_code_cell(feature_code),
        ],
        metadata=KERNEL_METADATA,
    )


def _comparison_notebook(report: dict[str, object]) -> nbformat.NotebookNode:
    rows = []
    for rank, summary in enumerate(report["model_summary"], start=1):
        marker = " ⭐" if rank == 1 else ""
        rows.append(
            f"| {rank} | {summary['model']}{marker} | {summary['accuracy']:.4f} | "
            f"{summary['macro_f1']:.4f} | {summary['down_recall']:.4f} | "
            f"**{summary['core_harmonic_mean']:.4f}** |"
        )
    table = "\n".join(rows)
    markdown = f"""# 개별종목 4모델 비교

## 후보·평가 흐름

`KOSPI200 3분류 → 시총 기준 업종 Top 10 → 업종별 시총 Top 5 → 최대 50종목의
3분류 확률 → 지수 예측 클래스에 해당하는 확률로 랭킹` 순서입니다.

## OOS 비교

| 순위 | 모델 | Accuracy | Macro F1 | 하락 Recall | 핵심지표 조화평균 |
|---:|---|---:|---:|---:|---:|
{table}

별표는 합의한 핵심지표 조화평균이 가장 높은 모델입니다. Accuracy 단독 순위가 아니라
하락 Recall까지 함께 반영하므로, 중립이나 상승 한쪽으로만 쏠린 모델을 피합니다.

## 데이터 규모

- 모델 행: {report['panel']['model_rows']:,}행
- 거래일: {report['panel']['dates']:,}일
- 후보에 포함된 종목: {report['panel']['stocks']:,}종목
- 극단 수정수익률 제거: {report['extreme_return_filter']['removed_candidate_rows']}행
- OOS 최빈 클래스 Accuracy 기준선: {report['majority_baseline_accuracy_on_oos']:.4f}
"""
    code = """import json
from pathlib import Path

import pandas as pd

ROOT = Path.cwd()
while ROOT.parent != ROOT and not (ROOT / "reports" / "stock_model_experiment.json").exists():
    ROOT = ROOT.parent
report = json.loads((ROOT / "reports" / "stock_model_experiment.json").read_text(encoding="utf-8"))
comparison = pd.DataFrame(report["model_summary"])
display(comparison)
"""
    return nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_markdown_cell(markdown), nbformat.v4.new_code_cell(code)],
        metadata=KERNEL_METADATA,
    )


def main() -> None:
    """리포트를 읽어 모델별 4개와 비교 1개 노트북을 만든다."""

    report = json.loads(REPORT.read_text(encoding="utf-8"))
    summaries = {item["model"]: item for item in report["model_summary"]}
    feature_columns = tuple(report["features"])

    BASE_OUTPUT.mkdir(parents=True, exist_ok=True)
    for model_name, filename in MODEL_FILES.items():
        nbformat.write(_base_model_notebook(model_name), BASE_OUTPUT / filename)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for model_name, filename in MODEL_FILES.items():
        notebook = _combination_notebook(model_name, feature_columns)
        nbformat.write(notebook, OUTPUT / filename)
    nbformat.write(_comparison_notebook(report), OUTPUT / "05.모델비교.ipynb")

    BEST_OUTPUT.mkdir(parents=True, exist_ok=True)
    winner = report["model_summary"][0]["model"]
    for model_name, filename in MODEL_FILES.items():
        best_filename = f"⭐{filename}" if model_name == winner else filename
        notebook = _best_notebook(model_name, summaries[model_name])
        nbformat.write(notebook, BEST_OUTPUT / best_filename)


if __name__ == "__main__":
    main()
