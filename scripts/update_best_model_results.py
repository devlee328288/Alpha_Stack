"""96개 실행 결과에서 조합별 모델 best 요약과 피처 설명을 갱신한다."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nbformat

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from features.model_dataset import COMBINATION_FEATURES  # noqa: E402
from scripts.build_model_notebooks import (  # noqa: E402
    COMBINATION_DIRS,
    MODELS,
    VARIANTS,
    experiment_directory,
)

CACHE_DIR = ROOT / "data" / "raw" / "model_results"
BEST_ROOT = ROOT / "notebooks" / "04-모델" / "실험" / "조합별 best result"
SWEEP_REPORT = ROOT / "reports" / "model_sweep.json"

FEATURE_DESCRIPTIONS = {
    "rsi_14": "14일 상승·하락 강도",
    "bb_bandwidth": "20일 볼린저밴드 폭",
    "hv_20": "20일 로그수익률 표준편차",
    "vol_ratio_20": "현재 거래량을 20일 평균 거래량으로 나눈 값",
    "sma_gap_5_20": "5일·20일 단순이동평균 간 상대 거리",
    "macd_hist_ratio": "MACD 히스토그램을 종가로 나눈 값",
    "bb_position": "볼린저밴드 안에서 종가가 위치한 비율",
    "sma_gap_20_60": "20일·60일 단순이동평균 간 상대 거리",
    "atr_ratio": "14일 ATR을 종가로 나눈 변동성",
    "hv_regime": "20일 변동성을 과거 250일 평균 변동성과 비교한 값",
    "obv_slope_20": "20일 OBV 변화를 평균 거래량으로 정규화한 값",
    "macd_hist_atr": "MACD 히스토그램을 14일 ATR로 나눈 값",
    "daily_return": "현재 종가를 직전 거래일 종가와 비교한 1거래일 수익률",
    "five_day_return": "현재 종가를 5거래일 전 종가와 비교한 5거래일 수익률",
}


def load_results() -> list[dict[str, Any]]:
    """정확히 96개의 실행 캐시를 읽고 HF 출처가 하나인지 검사한다."""

    paths = sorted(CACHE_DIR.glob("*.json"))
    if len(paths) != 96:
        raise RuntimeError(f"모델 실행 결과가 96개가 아닙니다: {len(paths)}")
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    sources = {
        (record["source"]["repo_sha"], record["source"]["index_sha256"])
        for record in records
    }
    if len(sources) != 1:
        raise RuntimeError(f"96개 결과의 HF 출처가 서로 다릅니다: {sources}")
    return records


def select_best_results(
    records: list[dict[str, Any]], combination: str
) -> dict[str, dict[str, Any]]:
    """같은 조합·모델의 네 피처 변형 중 조화평균 최대 결과를 고른다."""

    selected: dict[str, dict[str, Any]] = {}
    variant_order = {name: index for index, name in enumerate(VARIANTS)}
    for model_name in MODELS:
        candidates = [
            record
            for record in records
            if record["experiment"]["combination"] == combination
            and record["experiment"]["model"] == model_name
        ]
        if len(candidates) != len(VARIANTS):
            raise RuntimeError(
                f"조합 {combination} {model_name} 결과가 4개가 아닙니다: {len(candidates)}"
            )
        selected[model_name] = max(
            candidates,
            key=lambda record: (
                float(record["summary"]["core_harmonic_mean"]),
                -variant_order[_variant_name(record)],
            ),
        )
    return selected


def _variant_name(record: dict[str, Any]) -> str:
    returns = tuple(record["experiment"]["return_features"])
    for name, (_suffix, configured, _description) in VARIANTS.items():
        if returns == configured:
            return name
    raise ValueError(f"알 수 없는 수익률 피처 구성입니다: {returns}")


def _variant_label(variant: str) -> str:
    suffix = VARIANTS[variant][0]
    return "기본 조합" if not suffix else suffix.removeprefix(" + ")


def _source_link(combination: str, variant: str, filename: str) -> str:
    directory = experiment_directory(combination, variant)
    return (Path("../..") / directory.relative_to(BEST_ROOT.parent) / filename).as_posix()


def _result_notebook(
    combination: str,
    model_name: str,
    selected: dict[str, Any],
    candidates: list[dict[str, Any]],
):
    variant = _variant_name(selected)
    summary = selected["summary"]
    filename = MODELS[model_name]
    rows = []
    for record in sorted(candidates, key=lambda item: list(VARIANTS).index(_variant_name(item))):
        item = record["summary"]
        marker = " **(선정)**" if record is selected else ""
        rows.append(
            "| "
            f"{_variant_label(_variant_name(record))}{marker} | "
            f"{item['accuracy']:.4f} | {item['macro_f1']:.4f} | "
            f"{item['down_recall']:.4f} | **{item['core_harmonic_mean']:.4f}** |"
        )
    count_rows = "\n".join(
        f"| `{item['클래스 가중치']}` | {item['선택 폴드 수']} |"
        for item in selected["weight_counts"]
    )
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            f"# 조합{combination} · {model_name} best result\n\n"
            f"선정 실험: **{_variant_label(variant)}**  \n"
            "선정 기준: 전체 OOS Accuracy·Macro F1·하락 Recall 조화평균  \n"
            f"[실행된 원본 노트북]({_source_link(combination, variant, filename)})"
        ),
        nbformat.v4.new_markdown_cell(
            "## 선정 결과\n\n"
            "| 실험 | Accuracy | Macro F1 | 하락 Recall | 핵심지표 조화평균 |\n"
            "|---|---:|---:|---:|---:|\n" + "\n".join(rows)
        ),
        nbformat.v4.new_markdown_cell(
            "## 최종 OOS 핵심지표\n\n"
            "| 지표 | 결과 |\n|---|---:|\n"
            f"| Accuracy | {summary['accuracy']:.4f} |\n"
            f"| Macro F1 | {summary['macro_f1']:.4f} |\n"
            f"| 하락 Recall | {summary['down_recall']:.4f} |\n"
            f"| **핵심지표 조화평균** | **{summary['core_harmonic_mean']:.4f}** |\n"
            f"| Balanced Accuracy | {summary['balanced_accuracy']:.4f} |\n"
            f"| 최빈 클래스 Accuracy 기준선 | {summary['majority_accuracy']:.4f} |\n"
            f"| OOS 표본 | {summary['oos_rows']} |"
        ),
        nbformat.v4.new_markdown_cell(
            "## 05-백테스트·성과\n\n"
            "| 지표 | 결과 |\n|---|---:|\n"
            f"| 비용 차감 ΔSharpe 폴드 중앙값 | {summary['delta_sharpe_net_median']:.4f} |\n"
            f"| 비용 차감 ΔSharpe 폴드 평균 | {summary['delta_sharpe_net_mean']:.4f} |\n"
            f"| 전 기간 현금 보유 폴드 | {summary['all_cash_folds']} |\n\n"
            "상승 예측만 다음 거래일 시가에 20% 슬리브로 진입하고 정확히 5거래일 보유한다. "
            "중립·하락 예측은 현금이며 왕복비용은 0.05%다. 전 기간 현금인 폴드의 전략 "
            "Sharpe는 평가상 0으로 기록했다."
        ),
        nbformat.v4.new_markdown_cell(
            "## 폴드별 클래스 가중치 선택\n\n"
            "| 클래스 가중치 | 선택 폴드 수 |\n|---|---:|\n" + count_rows +
            "\n\n각 외부 폴드에서 마지막 내부 60거래일의 조화평균으로만 선택했다."
        ),
        nbformat.v4.new_markdown_cell(
            "## 해석 범위\n\n"
            "이 결과는 봉인 홀드아웃이 아닌 개발구간의 12폴드 OOS 평가다. 같은 개발 "
            "OOS에서 피처 변형을 비교했으므로 최종 채택 전 별도 홀드아웃 검증이 필요하다."
        ),
    ]
    return notebook


def _feature_markdown(
    combination: str,
    selected: dict[str, dict[str, Any]],
    overall_best: str,
) -> str:
    feature_lines = "\n".join(
        f"- `{feature}`: {FEATURE_DESCRIPTIONS[feature]}"
        for feature in COMBINATION_FEATURES[combination]
    )
    result_lines = []
    for model_name, record in selected.items():
        summary = record["summary"]
        marker = "⭐ " if model_name == overall_best else ""
        result_lines.append(
            f"| {marker}{model_name} | {_variant_label(_variant_name(record))} | "
            f"{summary['accuracy']:.4f} | {summary['macro_f1']:.4f} | "
            f"{summary['down_recall']:.4f} | **{summary['core_harmonic_mean']:.4f}** | "
            f"{summary['delta_sharpe_net_median']:.4f} |"
        )
    source = next(iter(selected.values()))["source"]
    return f"""# 조합{combination} 피처 선정

## 종속변수

- `fwd_return_5d`: 예측일 다음 거래일 시가 `open(t+1)`부터 `open(t+6)`까지의 수익률
- `label`: `fwd_return_5d`가 `+1%` 초과면 상승, `-1%` 미만이면 하락, 나머지는 중립
- 숫자 라벨: 하락 `-1` · 중립 `0` · 상승 `1`

## 기본 피처

{feature_lines}

## 추가 수익률 피처

- `Daily_Return`: 현재 종가와 직전 거래일 종가의 수익률
- `5Day_Return`: 현재 종가와 5거래일 전 종가의 수익률
- 각 모델에서 기본·Daily·5Day·두 수익률 동시 추가의 네 실험을 비교했다.

## 선정 결과

| 모델 | 선정 실험 | Accuracy | Macro F1 | 하락 Recall | 핵심지표 조화평균 | ΔSharpe 중앙값 |
|---|---|---:|---:|---:|---:|---:|
{"\n".join(result_lines)}

별표는 조합{combination}의 네 모델 중 핵심지표 조화평균이 가장 높은 모델이다. 모델별
선정도 같은 기준을 사용하며, 어느 한 지표가 0이면 조화평균도 0으로 처리한다.
클래스 가중치는 고정하지 않고 각 외부 폴드의 과거 내부 검증에서 다시 선택했다.

## 데이터와 평가 범위

- HF 리비전: `{source['repo_sha']}`
- `full/index_price_dev.parquet` SHA-256: `{source['index_sha256']}`
- 홀드아웃 시작: `20240901` — 이후 데이터는 사용하지 않음
- expanding walk-forward 12폴드 · 최초 학습 750거래일 · 검증 60거래일 · gap 5거래일
- 매매 성과: 상승만 5슬리브로 5거래일 보유 · 왕복비용 0.05% · 현금 Sharpe 0
"""


def main() -> int:
    records = load_results()
    report_combinations = {}
    for combination in COMBINATION_DIRS:
        selected = select_best_results(records, combination)
        overall_best = max(
            selected,
            key=lambda model_name: float(selected[model_name]["summary"]["core_harmonic_mean"]),
        )
        target = BEST_ROOT / f"조합{combination}"
        target.mkdir(parents=True, exist_ok=True)
        for old_notebook in target.glob("*.ipynb"):
            old_notebook.unlink()
        for model_name, record in selected.items():
            candidates = [
                item
                for item in records
                if item["experiment"]["combination"] == combination
                and item["experiment"]["model"] == model_name
            ]
            filename = MODELS[model_name]
            if model_name == overall_best:
                filename = f"⭐{filename}"
            nbformat.write(
                _result_notebook(combination, model_name, record, candidates),
                target / filename,
            )
        (target / "피처선정.md").write_text(
            _feature_markdown(combination, selected, overall_best),
            encoding="utf-8",
        )
        report_combinations[combination] = {
            "overall_best_model": overall_best,
            "models": {
                model: {
                    "variant": _variant_name(record),
                    "summary": record["summary"],
                    "weight_counts": record["weight_counts"],
                }
                for model, record in selected.items()
            },
        }
        print(f"조합 {combination}: {overall_best} 최우수")

    SWEEP_REPORT.write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "source": records[0]["source"],
                "selection": "Accuracy·Macro F1·하락 Recall 조화평균 최대",
                "experiments": records,
                "best_by_combination": report_combinations,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"전체 결과 저장: {SWEEP_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
