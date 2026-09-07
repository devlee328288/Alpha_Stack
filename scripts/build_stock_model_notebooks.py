"""개별종목 4모델 결과 노트북과 비교 노트북을 같은 형식으로 만든다."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_ROOT = ROOT / "notebooks" / "04-모델" / "개별종목" / "실험"
BASE_OUTPUT = EXPERIMENT_ROOT / "기본모델"
REPORT = ROOT / "reports" / "stock_feature_combinations.json"

COMBINATION_DIRECTORIES = {
    "A": "조합A_trend_momentum_volatility_volume_returns",
    "B": "조합B_trend_momentum_high_distance",
    "C": "조합C_short_reversal_intraday",
    "D": "조합D_volatility_liquidity",
    "E": "조합E_sector_market_relative_strength",
    "F": "조합F_cross_sectional_ranks",
}
COMBINATION_TITLES = {
    "A": "추세·모멘텀·변동성·거래량·수익률",
    "B": "5거래일 단기 반전·고점 거리",
    "C": "단기 반전·장중 위치",
    "D": "변동성·유동성",
    "E": "업종·시장 상대강도",
    "F": "당일 후보군 횡단면 순위",
}

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

FEATURE_DESCRIPTIONS = {
    "sma_gap_5_20": "5일·20일 수정종가 이동평균의 상대 거리",
    "sma_gap_20_60": "20일·60일 수정종가 이동평균의 상대 거리",
    "rsi_14": "14일 상승·하락 강도",
    "macd_hist_ratio": "MACD 히스토그램을 수정종가로 나눈 값",
    "bb_bandwidth": "20일 볼린저밴드 폭",
    "bb_position": "볼린저밴드 안에서 수정종가의 위치",
    "atr_ratio": "14일 ATR을 수정종가로 나눈 상대 변동성",
    "hv_20": "20일 수정종가 로그수익률 변동성",
    "vol_ratio_20": "당일 거래량을 20일 평균 거래량으로 나눈 값",
    "obv_slope_20": "20일 OBV 변화를 평균 거래량으로 정규화한 값",
    "daily_return": "수정종가 1거래일 수익률",
    "five_day_return": "수정종가 5거래일 수익률",
    "ret_1": "수정종가 1거래일 수익률",
    "ret_5": "수정종가 5거래일 수익률",
    "ret_20": "수정종가 20거래일 수익률",
    "dist_high_20": "현재 수정종가와 최근 20일 최고가의 상대 거리",
    "dist_high_60": "현재 수정종가와 최근 60일 최고가의 상대 거리",
    "overnight_gap": "전일 수정종가 대비 당일 수정시가 수익률",
    "intraday_return": "당일 수정시가 대비 수정종가 수익률",
    "close_location": "당일 수정고가·저가 범위 안의 수정종가 위치",
    "volume_z_20": "로그 거래량의 20일 z-score",
    "range_1": "당일 수정고가·저가 범위를 수정종가로 나눈 값",
    "range_20": "range_1의 20일 평균",
    "turnover_20": "거래대금/시가총액 비율의 20일 평균",
    "log_amihud_20": "수익률/거래대금 비유동성의 20일 평균 로그값",
    "sector_ret_5": "해당 날짜 업종지수의 5거래일 수익률",
    "sector_ret_20": "해당 날짜 업종지수의 20거래일 수익률",
    "relative_ret_5_sector": "종목 5일 수익률에서 업종 5일 수익률을 뺀 값",
    "relative_ret_20_sector": "종목 20일 수익률에서 업종 20일 수익률을 뺀 값",
    "relative_ret_5_market": "종목 5일 수익률에서 KOSPI200 5일 수익률을 뺀 값",
    "sector_hv_20": "업종지수 20일 로그수익률 변동성",
    "sector_beta_60": "종목 수익률의 업종지수 대비 60일 베타",
    "ret_5_rank": "당일 후보군 안 5일 수익률 백분위 순위",
    "sector_relative_rank": "당일 후보군 안 업종 대비 상대강도 백분위 순위",
    "turnover_rank": "당일 후보군 안 20일 회전율 백분위 순위",
    "hv_20_rank": "당일 후보군 안 20일 변동성 백분위 순위",
    "market_cap_percentile": "당일 후보군 안 시가총액 백분위 순위",
    "sector_market_cap_rank": "당일 업종지수 시가총액 순위",
    "industry_stock_rank": "당일 같은 업종 안 종목 시가총액 순위",
}


def _best_notebook(
    combination: str,
    model_name: str,
    summary: dict[str, object],
) -> nbformat.NotebookNode:
    """조합별 best-result에 둘 모델 요약 노트북을 만든다."""

    metric_row = (
        f"| {summary['accuracy']:.4f} | {summary['macro_f1']:.4f} | "
        f"{summary['down_recall']:.4f} | **{summary['core_harmonic_mean']:.4f}** |"
    )
    markdown = f"""# 개별종목 조합{combination} — {model_name}

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
while ROOT.parent != ROOT and not (ROOT / "reports" / "stock_feature_combinations.json").exists():
    ROOT = ROOT.parent
report_path = ROOT / "reports" / "stock_feature_combinations.json"
report = json.loads(report_path.read_text(encoding="utf-8"))
combination = {combination!r}
model_name = {model_name!r}

combination_report = report["combinations"][combination]
folds = pd.DataFrame(combination_report["outer_fold_results"])
display(folds.loc[folds["model"].eq(model_name)].reset_index(drop=True))

weights = pd.DataFrame(combination_report["selected_class_weight_counts"])
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
    combination: str,
    model_name: str,
    feature_columns: tuple[str, ...],
) -> nbformat.NotebookNode:
    """기본모델과 조합별 피처, 저장된 실측 결과를 연결한다."""

    base_filename = MODEL_FILES[model_name]
    module_name, builder_name = MODEL_BUILDERS[model_name]
    feature_literal = "(\n" + "".join(
        f"    {feature!r},\n" for feature in feature_columns
    ) + ")"
    markdown = f"""# 개별종목 조합{combination} — {model_name}

`기본모델/{base_filename}`과 같은 `{module_name}.{builder_name}`을 가져오고
조합{combination} 피처를 주입합니다. 기본모델 코드는 `models/`에 한 번만 존재합니다.
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
    feature_code = f"""# 2. 조합{combination}의 피처 값만 지정합니다.
import json

COMBINATION = {combination!r}
FEATURE_COLUMNS = {feature_literal}

report_path = project_root / "reports" / "stock_feature_combinations.json"
report = json.loads(report_path.read_text(encoding="utf-8"))
combination_report = report["combinations"][COMBINATION]
panel = combination_report["panel"]
print("학습 기간:", panel["first_date"], "~", panel["last_date"])
print("학습 행·종목:", panel["model_rows"], panel["stocks"])
print(f"조합{{COMBINATION}} 피처:", FEATURE_COLUMNS)

folds = pd.DataFrame(combination_report["outer_fold_results"])
model_folds = folds.loc[folds["model"].eq(MODEL_NAME)].reset_index(drop=True)
fold_columns = [
    "fold",
    "selected_class_weight",
    "train_dates",
    "valid_start",
    "valid_end",
    "accuracy",
    "training_majority_baseline_accuracy",
    "accuracy_minus_training_majority_baseline",
    "macro_f1",
    "down_recall",
    "core_harmonic_mean",
]
display(model_folds.loc[:, fold_columns].round(4))

metric_columns = [
    "accuracy",
    "training_majority_baseline_accuracy",
    "accuracy_minus_training_majority_baseline",
    "macro_f1",
    "down_recall",
    "core_harmonic_mean",
]
display(model_folds.loc[:, metric_columns].mean().to_frame("OOS 폴드 평균").round(4))

# 24개 노트북이 각각 중복 학습하지 않도록 실제 fit은 공통 실행기에서 한 번 수행합니다.
print("재실행 명령: python scripts/run_stock_model_experiment.py")
"""
    return nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell(markdown),
            nbformat.v4.new_code_cell(setup_code),
            nbformat.v4.new_code_cell(feature_code),
        ],
        metadata=KERNEL_METADATA,
    )


def _comparison_notebook(
    combination: str,
    report: dict[str, object],
    data_quality_policy: dict[str, object],
) -> nbformat.NotebookNode:
    rows = []
    for rank, summary in enumerate(report["model_summary"], start=1):
        marker = " ⭐" if rank == 1 else ""
        rows.append(
            f"| {rank} | {summary['model']}{marker} | {summary['accuracy']:.4f} | "
            f"{summary['training_majority_baseline_accuracy']:.4f} | "
            f"{summary['accuracy_minus_training_majority_baseline']:+.4f} | "
            f"{summary['macro_f1']:.4f} | {summary['down_recall']:.4f} | "
            f"**{summary['core_harmonic_mean']:.4f}** |"
        )
    table = "\n".join(rows)
    quality = data_quality_policy["adjustment_quality"]
    table_header = (
        "| 순위 | 모델 | Accuracy | 학습 최빈 기준선 | 기준선 대비 | "
        "Macro F1 | 하락 Recall | 핵심지표 조화평균 |"
    )
    training_baseline = report["baseline_summary"][
        "training_majority_baseline_accuracy_mean"
    ]
    validation_oracle = report["baseline_summary"][
        "validation_majority_oracle_accuracy_mean"
    ]
    markdown = f"""# 개별종목 조합{combination} 4모델 비교

## 후보·평가 흐름

`KOSPI200 3분류 → 시총 기준 업종 Top 10 → 업종별 시총 Top 5 → 최대 50종목의
3분류 확률 → 지수 예측 클래스에 해당하는 확률로 랭킹` 순서입니다.

## OOS 비교

{table_header}
|---:|---|---:|---:|---:|---:|---:|---:|
{table}

별표는 합의한 핵심지표 조화평균이 가장 높은 모델입니다. Accuracy 단독 순위가 아니라
하락 Recall까지 함께 반영하므로, 중립이나 상승 한쪽으로만 쏠린 모델을 피합니다.

## 데이터 규모

- 모델 행: {report['panel']['model_rows']:,}행
- 거래일: {report['panel']['dates']:,}일
- 후보에 포함된 종목: {report['panel']['stocks']:,}종목
- 수정주가 품질 의심 제거: {data_quality_policy['removed_candidate_rows']:,}행
- KRX 등락률과 일치해 보존한 실제 극단 사건: {quality['candidate_extreme_rows']:,}행
- 폴드별 학습 최빈 Accuracy 기준선 평균: {training_baseline:.4f}
- 검증 최빈 Accuracy 사후 참고값 평균: {validation_oracle:.4f}

학습 최빈 기준선만 모델 비교에 사용합니다. 검증 최빈 값은 해당 폴드의 정답 분포를 본
`oracle` 통계이므로 성능 판정에는 사용하지 않습니다.
"""
    code = f"""import json
from pathlib import Path

import pandas as pd

ROOT = Path.cwd()
while ROOT.parent != ROOT and not (ROOT / "reports" / "stock_feature_combinations.json").exists():
    ROOT = ROOT.parent
report_path = ROOT / "reports" / "stock_feature_combinations.json"
report = json.loads(report_path.read_text(encoding="utf-8"))
comparison = pd.DataFrame(report["combinations"][{combination!r}]["model_summary"])
display(comparison)
"""
    return nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_markdown_cell(markdown), nbformat.v4.new_code_cell(code)],
        metadata=KERNEL_METADATA,
    )


def _feature_selection_markdown(
    combination: str,
    feature_columns: tuple[str, ...],
    winner: dict[str, object] | None = None,
) -> str:
    """조합 폴더와 best-result 폴더에 둘 피처 설명을 만든다."""

    lines = [
        f"# 조합{combination} 피처 선정",
        "",
        "## 종속변수",
        "",
        "- T일 종가까지의 정보로 판단",
        "- T+1 `adj_open` 진입 → T+6 `adj_open` 평가",
        "- 수익률이 +2% 초과면 상승, -2% 미만이면 하락, 나머지는 중립",
        "",
        "## 독립변수",
        "",
    ]
    lines.extend(
        f"- `{feature}`: {FEATURE_DESCRIPTIONS[feature]}" for feature in feature_columns
    )
    lines.extend(
        [
            "",
            "## 선정 이유",
            "",
            f"{COMBINATION_TITLES[combination]} 관점이 5거래일 뒤 개별종목 방향을 얼마나 "
            "설명하는지 다른 조합과 같은 후보·날짜·폴드에서 비교합니다.",
            "모든 이동창은 종목의 전체 과거 시계열에서 먼저 계산하고, 횡단면 순위는 그날 "
            "확정된 최대 50개 후보만 사용합니다.",
        ]
    )
    if winner is not None:
        lines.extend(
            [
                "",
                "## 선정 결과",
                "",
                "- 선정 기준: Accuracy·Macro F1·하락 Recall의 폴드별 조화평균",
                (
                    f"- 최종 1위: **{winner['model']}**, 조화평균 "
                    f"**{winner['core_harmonic_mean']:.4f}**"
                ),
                "- 별표가 붙은 노트북이 이 조합의 4모델 중 최종 1위입니다.",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    """A~F 각각에 모델 4개·비교·피처 문서와 best-result를 만든다."""

    report = json.loads(REPORT.read_text(encoding="utf-8"))

    BASE_OUTPUT.mkdir(parents=True, exist_ok=True)
    for model_name, filename in MODEL_FILES.items():
        nbformat.write(_base_model_notebook(model_name), BASE_OUTPUT / filename)

    for combination, directory in COMBINATION_DIRECTORIES.items():
        combination_report = report["combinations"][combination]
        summaries = {
            item["model"]: item for item in combination_report["model_summary"]
        }
        feature_columns = tuple(combination_report["features"])
        output = EXPERIMENT_ROOT / directory
        output.mkdir(parents=True, exist_ok=True)
        for model_name, filename in MODEL_FILES.items():
            notebook = _combination_notebook(combination, model_name, feature_columns)
            nbformat.write(notebook, output / filename)
        nbformat.write(
            _comparison_notebook(
                combination,
                combination_report,
                report["data_quality_policy"],
            ),
            output / "05.모델비교.ipynb",
        )
        (output / "피처선정.md").write_text(
            _feature_selection_markdown(combination, feature_columns),
            encoding="utf-8",
        )

        best_output = EXPERIMENT_ROOT / "조합별 best result" / f"조합{combination}"
        best_output.mkdir(parents=True, exist_ok=True)
        winner = combination_report["model_summary"][0]
        # 1위 모델이 바뀌면 이전의 일반 파일과 별표 파일이 함께 남을 수 있다.
        # 이 폴더의 ipynb는 전부 이 생성기의 산출물이므로 4개를 새로 맞춘다.
        for old_notebook in best_output.glob("*.ipynb"):
            old_notebook.unlink()
        for model_name, filename in MODEL_FILES.items():
            best_filename = f"⭐{filename}" if model_name == winner["model"] else filename
            notebook = _best_notebook(combination, model_name, summaries[model_name])
            nbformat.write(notebook, best_output / best_filename)
        (best_output / "피처선정.md").write_text(
            _feature_selection_markdown(combination, feature_columns, winner),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
