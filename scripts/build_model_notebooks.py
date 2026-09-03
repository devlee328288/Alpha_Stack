"""A~F의 96개 모델 노트북과 24개 비교 노트북을 같은 형식으로 만든다."""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EXPERIMENT_ROOT = ROOT / "notebooks" / "04-모델" / "실험"
COMBINATION_DIRS = {
    "A": "조합A_rsi14_bb_bandwidth_hv20_vol_ratio20",
    "B": "조합B_sma_gap_5_20_macd_hist_ratio_rsi14_hv20",
    "C": "조합C_sma_gap_macd_rsi_bb_hv_volume",
    "D": "조합D_option2_volatility_focus",
    "E": "조합E_option5_volatility_only",
    "F": "조합F_option6_maximum",
}
VARIANTS = {
    "base": ("", (), "기본 조합"),
    "daily": (" + Daily_Return", ("daily_return",), "Daily_Return 추가"),
    "five_day": (" + 5Day_Return", ("five_day_return",), "5Day_Return 추가"),
    "both": (
        " + Daily_Return + 5Day_Return",
        ("daily_return", "five_day_return"),
        "Daily_Return·5Day_Return 추가",
    ),
}
MODELS = {
    "LogisticRegression": "01.LogisticRegression.ipynb",
    "RandomForest": "02.RandomForest.ipynb",
    "XGBoost": "03.XGBoost.ipynb",
    "LightGBM": "04.LightGBM.ipynb",
}


def experiment_directory(combination: str, variant: str) -> Path:
    """조합과 수익률 변형에 해당하는 실제 노트북 폴더를 돌려준다."""

    base = EXPERIMENT_ROOT / COMBINATION_DIRS[combination]
    suffix = VARIANTS[variant][0]
    return base if not suffix else base / f"조합{combination}{suffix}"


def notebook_targets() -> list[tuple[str, str, Path]]:
    """정해진 24개 실험 폴더를 순서대로 열거한다."""

    return [
        (combination, variant, experiment_directory(combination, variant))
        for combination in COMBINATION_DIRS
        for variant in VARIANTS
    ]


def _model_notebook(combination: str, variant: str, model_name: str):
    suffix, return_features, description = VARIANTS[variant]
    title = f"조합{combination}{suffix} — {model_name}"
    cache_name = f"{combination}_{variant}_{model_name}.json"
    feature_literal = repr(return_features)
    markdown = f"""# {title}

## 실험 목적

HF의 `full/index_price_dev.parquet`에서 `코스피 200`만 선택해 {description} 실험을 수행한다.
피처와 미래 5거래일 방향 라벨은 parquet 원시값으로 다시 계산한다.

## 공통 조건

| 항목 | 값 |
|---|---|
| 데이터 | HF 비공개 저장소의 고정 리비전 parquet |
| 홀드아웃 시작 | `20240901` — 이후 행 접근 금지 |
| 외부 검증 | expanding walk-forward 12폴드 |
| 최초 외부 학습 | 750거래일 |
| 외부 검증 | 폴드당 60거래일 |
| 학습·검증 gap | 5거래일 |
| 클래스 가중치 | 외부 폴드마다 내부 60일 검증으로 `None`/`balanced` 선택 |
| 가중치 선정 기준 | Accuracy·Macro F1·하락 Recall 조화평균 |
| 매매 평가 | 상승만 다음 시가 진입, 5슬리브·5거래일 보유, 왕복비용 0.05% |
"""
    setup_code = f'''# 어느 폴더에서 실행해도 프로젝트 모듈을 찾도록 루트를 확인합니다.
import json
import sys
from pathlib import Path

import pandas as pd
from IPython.display import Markdown, display

project_root = Path.cwd().resolve()
while project_root != project_root.parent and not (project_root / "pyproject.toml").is_file():
    project_root = project_root.parent
if not (project_root / "pyproject.toml").is_file():
    raise RuntimeError("프로젝트 루트를 찾지 못했습니다.")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from features.model_dataset import build_model_dataset  # noqa: E402
from models.experiment import evaluate_nested_class_weights  # noqa: E402
from models.notebook_experiment import summarize_notebook_experiment  # noqa: E402
from supply.hf_model_data import load_hf_index_prices  # noqa: E402

COMBINATION = "{combination}"
RETURN_FEATURES = {feature_literal}
MODEL_NAME = "{model_name}"
CACHE_NAME = "{cache_name}"

# MANIFEST와 parquet를 같은 HF 커밋에서 받고 SHA-256·홀드아웃 경계를 검사합니다.
snapshot = load_hf_index_prices()
dataset = build_model_dataset(
    snapshot.frame,
    COMBINATION,
    return_features=RETURN_FEATURES,
)
print("HF 리비전:", snapshot.repo_sha)
print("지수 parquet SHA-256:", snapshot.file_sha256)
print("학습 가능 기간:", dataset.frame["bas_dd"].min(), "~", dataset.frame["bas_dd"].max())
print("학습 가능 행:", len(dataset.frame))
print("피처:", list(dataset.feature_columns))
'''
    train_code = '''# 12개 외부 폴드 각각에서 과거 데이터만으로 가중치를 고르고 OOS를 평가합니다.
nested = evaluate_nested_class_weights(dataset, model_names=(MODEL_NAME,))
result = summarize_notebook_experiment(dataset, nested, MODEL_NAME)

fold_columns = [
    "fold",
    "selected_class_weight",
    "train_size",
    "train_end",
    "valid_start",
    "valid_end",
    "accuracy",
    "macro_f1",
    "down_recall",
    "core_harmonic_mean",
    "delta_sharpe_net",
    "all_cash",
]
display(result.fold_results.loc[:, fold_columns].round(4))
display(result.weight_counts)
'''
    result_code = '''# 전체 720개 OOS 예측을 합쳐 분류 성능과 혼동행렬을 계산합니다.
summary = result.summary
summary_table = pd.Series(
    {
        "전체 OOS 표본": summary["oos_rows"],
        "Accuracy": summary["accuracy"],
        "Macro F1": summary["macro_f1"],
        "하락 Recall": summary["down_recall"],
        "핵심지표 조화평균": summary["core_harmonic_mean"],
        "Balanced Accuracy": summary["balanced_accuracy"],
        "최빈 클래스 기준선": summary["majority_accuracy"],
        "ΔSharpe_net 폴드 중앙값": summary["delta_sharpe_net_median"],
        "전 기간 현금 폴드": summary["all_cash_folds"],
    },
    name="결과",
)
display(summary_table.to_frame().round(4))
display(result.confusion)
display(result.class_report.round(4))

# 뒤의 05.모델비교 노트북은 네 모델의 이 실행 결과만 읽습니다.
payload = {
    "source": {
        "repo_sha": snapshot.repo_sha,
        "index_sha256": snapshot.file_sha256,
        "dev_end": snapshot.dev_end,
    },
    "experiment": {
        "combination": COMBINATION,
        "return_features": list(RETURN_FEATURES),
        "model": MODEL_NAME,
    },
    "summary": summary,
    "weight_counts": result.weight_counts.to_dict(orient="records"),
}
cache_path = project_root / "data" / "raw" / "model_results" / CACHE_NAME
cache_path.parent.mkdir(parents=True, exist_ok=True)
cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print("비교용 실행 결과 저장:", cache_path)
'''
    interpretation = '''# 05-백테스트·성과 형식으로 핵심 결과를 함께 해석합니다.
display(Markdown(f"""
## 결과 해석

- 전체 OOS Accuracy: `{summary['accuracy']:.4f}`
- 전체 OOS Macro F1: `{summary['macro_f1']:.4f}`
- 하락 Recall: `{summary['down_recall']:.4f}`
- 세 핵심지표 조화평균: `{summary['core_harmonic_mean']:.4f}`
- 최빈 클래스 Accuracy 기준선: `{summary['majority_accuracy']:.4f}`
- 비용 차감 ΔSharpe 폴드 중앙값: `{summary['delta_sharpe_net_median']:.4f}`
- 전 기간 현금 보유 폴드: `{summary['all_cash_folds']}`개
- 예측 개수: 하락 `{summary['predicted_down']}` · 중립
  `{summary['predicted_neutral']}` · 상승 `{summary['predicted_up']}`

이 값은 봉인 홀드아웃이 아닌 개발구간의 12폴드 OOS 결과다. 클래스 가중치는 각 폴드의
외부 검증값을 보지 않고, 그보다 앞선 내부 60거래일의 조화평균으로만 선택했다.
"""))
'''
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        nbformat.v4.new_markdown_cell(markdown),
        nbformat.v4.new_code_cell(setup_code),
        nbformat.v4.new_markdown_cell(
            "## 클래스 가중치 재튜닝\n\n외부 검증 60일은 가중치 선정에 사용하지 않는다."
        ),
        nbformat.v4.new_code_cell(train_code),
        nbformat.v4.new_markdown_cell("## OOS 핵심지표와 백테스트 성과"),
        nbformat.v4.new_code_cell(result_code),
        nbformat.v4.new_code_cell(interpretation),
    ]
    notebook.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata.language_info = {"name": "python", "version": "3.12"}
    return notebook


def _comparison_notebook(combination: str, variant: str):
    suffix, return_features, description = VARIANTS[variant]
    return_literal = repr(return_features)
    markdown = f"""# 조합{combination}{suffix} 모델 4종 비교

{description} 조건에서 실행한 네 모델의 전체 OOS 지표를 비교한다. 각 숫자는 같은 HF
parquet 리비전과 같은 12개 expanding 검증 구간에서 나온 실행 결과다.
"""
    code = f'''import json
from pathlib import Path

import pandas as pd
from IPython.display import Markdown, display

project_root = Path.cwd().resolve()
while project_root != project_root.parent and not (project_root / "pyproject.toml").is_file():
    project_root = project_root.parent
if not (project_root / "pyproject.toml").is_file():
    raise RuntimeError("프로젝트 루트를 찾지 못했습니다.")

combination = "{combination}"
variant = "{variant}"
return_features = {return_literal}
model_names = {tuple(MODELS)}
payloads = []
for model_name in model_names:
    path = project_root / "data" / "raw" / "model_results" / (
        f"{{combination}}_{{variant}}_{{model_name}}.json"
    )
    if not path.is_file():
        raise FileNotFoundError(f"먼저 01~04 모델 노트북을 실행해야 합니다: {{path}}")
    payloads.append(json.loads(path.read_text(encoding="utf-8")))

# 서로 다른 데이터나 피처 실험의 숫자가 한 표에 섞이면 즉시 중단합니다.
source_keys = {{
    (item["source"]["repo_sha"], item["source"]["index_sha256"])
    for item in payloads
}}
experiment_keys = {{
    (
        item["experiment"]["combination"],
        tuple(item["experiment"]["return_features"]),
    )
    for item in payloads
}}
if len(source_keys) != 1 or experiment_keys != {{(combination, return_features)}}:
    raise RuntimeError("네 모델의 HF 출처 또는 피처 구성이 서로 다릅니다.")

rows = []
for item in payloads:
    summary = item["summary"]
    rows.append(
        {{
            "모델": summary["model"],
            "Accuracy": summary["accuracy"],
            "Macro F1": summary["macro_f1"],
            "하락 Recall": summary["down_recall"],
            "핵심지표 조화평균": summary["core_harmonic_mean"],
            "ΔSharpe_net 중앙값": summary["delta_sharpe_net_median"],
            "현금 폴드": summary["all_cash_folds"],
        }}
    )
comparison_df = pd.DataFrame(rows).sort_values(
    "핵심지표 조화평균", ascending=False, kind="stable"
).reset_index(drop=True)
display(comparison_df.round(4))

best = comparison_df.iloc[0]
display(Markdown(f"""
## 자동 비교 결과

- 세 핵심지표 조화평균 1위: **{{best['모델']}}** `{{best['핵심지표 조화평균']:.4f}}`
- Accuracy: `{{best['Accuracy']:.4f}}`
- Macro F1: `{{best['Macro F1']:.4f}}`
- 하락 Recall: `{{best['하락 Recall']:.4f}}`
- 비용 차감 ΔSharpe 폴드 중앙값: `{{best['ΔSharpe_net 중앙값']:.4f}}`

순위는 합의한 기준인 Accuracy·Macro F1·하락 Recall 조화평균으로 정했다.
"""))
'''
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        nbformat.v4.new_markdown_cell(markdown),
        nbformat.v4.new_code_cell(code),
        nbformat.v4.new_markdown_cell(
            "## 개별 실험\n\n"
            "- [Logistic Regression](01.LogisticRegression.ipynb)\n"
            "- [RandomForest](02.RandomForest.ipynb)\n"
            "- [XGBoost](03.XGBoost.ipynb)\n"
            "- [LightGBM](04.LightGBM.ipynb)"
        ),
    ]
    notebook.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata.language_info = {"name": "python", "version": "3.12"}
    return notebook


def main() -> int:
    written = 0
    for combination, variant, directory in notebook_targets():
        directory.mkdir(parents=True, exist_ok=True)
        for model_name, filename in MODELS.items():
            nbformat.write(_model_notebook(combination, variant, model_name), directory / filename)
            written += 1
        nbformat.write(_comparison_notebook(combination, variant), directory / "05.모델비교.ipynb")
        written += 1
    print(f"노트북 {written}개 생성 완료: 모델 96개 + 비교 24개")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
