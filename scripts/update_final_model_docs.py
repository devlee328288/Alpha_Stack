"""평가 보고서의 현재 1위로 `04-모델/최종모델` 문서를 다시 만든다.

문서는 사람이 복사해 적지 않는다. KOSPI200·개별종목 평가 보고서가 갱신될 때 이 함수를
호출하면 모델명, 피처, OOS 지표와 선정 기준 비교가 같은 리비전으로 따라간다.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.final_models import (  # noqa: E402
    compare_index_model_selection_policies,
    load_winning_models,
    preferred_index_report_path,
    select_best_long_only_index_model,
    select_best_stock_model,
)

INDEX_REPORT_RELATIVE = Path("reports/model_sweep.json")
LONG_ONLY_INDEX_REPORT_RELATIVE = Path("reports/index_long_only_selection.json")
STOCK_REPORT_RELATIVE = Path("reports/stock_feature_combinations.json")
RANKING_REPORT_RELATIVE = Path("reports/stock_index_ranking.json")
COMPARISON_REPORT_RELATIVE = Path("reports/index_model_selection_comparison.json")
FINAL_ROOT_RELATIVE = Path("notebooks/04-모델/최종모델")
GENERATED_NOTICE = (
    "<!-- 이 파일은 scripts/update_final_model_docs.py가 생성합니다. "
    "직접 수정하지 마세요. -->"
)

FEATURE_DESCRIPTIONS = {
    "atr_ratio": "14일 평균 실제 변동폭(ATR)을 종가로 나눈 값 — 최근 가격 변동성의 상대 크기",
    "bb_bandwidth": "20일 볼린저밴드의 폭 — 값이 클수록 최근 가격 변동성이 큼",
    "hv_regime": "20일 변동성이 최근 250일 평균 변동성의 몇 배인지 나타내는 시장 국면",
    "five_day_return": "현재 종가가 5거래일 전보다 얼마나 올랐거나 내렸는지 나타내는 수익률",
    "relative_ret_5_market": "종목의 5일 수익률에서 KOSPI200 5일 수익률을 뺀 시장 대비 성과",
    "sma_gap_5_20": "5일 이동평균과 20일 이동평균의 상대 차이 — 단기 추세",
    "sma_gap_20_60": "20일 이동평균과 60일 이동평균의 상대 차이 — 중기 추세",
    "rsi_14": "최근 14일 상승·하락 강도의 균형을 0~100으로 나타낸 과매수·과매도 지표",
    "macd_hist_ratio": "MACD와 신호선의 차이를 종가로 나눈 값 — 추세 변화의 방향과 강도",
    "bb_position": "20일 볼린저밴드 안에서 현재 종가가 어디에 있는지 나타내는 위치",
    "hv_20": "최근 20일 로그수익률의 표준편차로 계산한 과거 변동성",
    "vol_ratio_20": "당일 거래량을 최근 20일 평균 거래량으로 나눈 값 — 평소 대비 거래량",
    "obv_slope_20": "가격 방향에 따라 누적한 거래량(OBV)의 최근 20일 변화 방향",
    "daily_return": "현재 종가가 직전 거래일보다 얼마나 변했는지 나타내는 1일 수익률",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"보고서 최상위 값은 객체여야 합니다: {path}")
    return value


def _index_record(report: dict[str, Any], combination: str, model: str, returns: list[str]):
    matches = [
        record
        for record in report["experiments"]
        if record["experiment"].get("combination") == combination
        and record["experiment"].get("model") == model
        and record["experiment"].get("return_features", []) == returns
    ]
    if len(matches) != 1:
        raise ValueError(f"KOSPI200 1위 실험이 정확히 한 건이 아닙니다: {len(matches)}")
    return matches[0]


def _long_only_index_record(
    report: dict[str, Any], combination: str, model: str, returns: list[str]
) -> dict[str, Any]:
    matches = [
        record
        for record in report["candidates"]
        if record["experiment"].get("combination") == combination
        and record["experiment"].get("model") == model
        and record["experiment"].get("return_features", []) == returns
    ]
    if len(matches) != 1:
        raise ValueError(f"KOSPI200 long-only 1위 실험이 정확히 한 건이 아닙니다: {len(matches)}")
    return matches[0]


def _feature_lines(features: tuple[str, ...], *, describe: bool = False) -> str:
    if not describe:
        return "\n".join(f"- `{feature}`" for feature in features)
    return "\n".join(
        f"- `{feature}`: {FEATURE_DESCRIPTIONS.get(feature, '발표 전 의미 설명을 추가해야 함')}"
        for feature in features
    )


def _variant_label(return_features: tuple[str, ...]) -> str:
    if not return_features:
        return "기본"
    labels = {
        "daily_return": "Daily Return",
        "five_day_return": "5Day Return",
    }
    return " + ".join(labels.get(feature, feature) for feature in return_features)


def _render_root(index_winner, stock_winner, *, index_selected: bool = False) -> str:
    index_variant = _variant_label(index_winner.return_features)
    index_status = "확정" if index_selected else "잠정"
    index_basis = (
        "이슈 #216 개발구간 선정 규칙"
        if index_selected
        else "이슈 #203 선정 기준 확인 중"
    )
    index_row = (
        f"| KOSPI200 | 조합 {index_winner.combination} {index_winner.model} + "
        f"{index_variant} | {index_status} | {index_basis} |"
    )
    return f"""{GENERATED_NOTICE}
# 최종모델

평가 보고서가 선택한 KOSPI200·개별종목 모델을 한곳에서 확인합니다. 실행 코드는 조합
노트북을 복사하지 않고 [`models/final_models.py`](../../../models/final_models.py)에서
보고서를 읽어 현재 1위를 선택합니다.

| 트랙 | 현재 모델 | 상태 | 근거 |
|---|---|---|---|
{index_row}
| 개별종목 | 조합 {stock_winner.combination} {stock_winner.model} | 확정 | ADR 0007 |

- [KOSPI200 선정 모델](KOSPI200/README.md)
- [개별종목 최종 모델](개별종목/README.md)

새 피처 조합은 기존 네 모델로 같은 개발구간 OOS 평가를 마친 뒤 보고서에 추가합니다.
선정 기준이 같으면 최종 산출 코드는 바꾸지 않습니다. KOSPI200 선정 기준 자체가 바뀌면
선택 함수의 정책과 문서를 한 번 맞춰야 합니다.

봉인 홀드아웃 `20240901` 이후 데이터는 모델 선택이나 이 문서 생성에 사용하지 않습니다.
"""


def _render_index(index_winner, record: dict[str, Any], comparison: dict[str, Any]) -> str:
    summary = record["summary"]
    rows = []
    for policy in comparison["policies"]:
        winner = policy["winner"]
        variant = _variant_label(tuple(winner["return_features"]))
        distribution = "/".join(
            str(winner[key])
            for key in ("predicted_down", "predicted_neutral", "predicted_up")
        )
        rows.append(
            f"| {policy['description']} | 조합 {winner['combination']} {variant}·"
            f"{winner['model']} | {winner['accuracy']:.4f} | {winner['macro_f1']:.4f} | "
            f"{winner['down_recall']:.4f} | {distribution} |"
        )
    return f"""{GENERATED_NOTICE}
# KOSPI200 현재 잠정 모델

## 상태

현재 운영 중인 `Accuracy·Macro F1·하락 Recall` 조화평균 기준의 잠정 1위입니다.
long-only `{{0, +1}}` 목적에 맞는 최종 선정 기준은 이슈 #203에서 확인 중이므로 아직 최종
확정으로 표시하지 않습니다.

| 항목 | 값 |
|---|---|
| 조합 | {index_winner.combination} + {_variant_label(index_winner.return_features)} |
| 모델 | {index_winner.model} |
| 피처 수 | {len(index_winner.feature_columns)} |
| 선정 지표 | {index_winner.selection_metric} |
| 선정값 | {index_winner.selection_value:.4f} |

## 사용 피처

{_feature_lines(index_winner.feature_columns)}

## 개발구간 OOS 결과

| 지표 | 값 |
|---|---:|
| Accuracy | {summary['accuracy']:.4f} |
| Macro F1 | {summary['macro_f1']:.4f} |
| 하락 Recall | {summary['down_recall']:.4f} |
| 3지표 조화평균 | {summary['core_harmonic_mean']:.4f} |
| Balanced Accuracy | {summary['balanced_accuracy']:.4f} |
| MCC | {summary['mcc']:.4f} |
| 상승 Recall | {summary['up_recall']:.4f} |
| 상승 PR-AUC | {summary['pr_auc_up']:.4f} |

공통 조건은 expanding 12폴드, 최초 학습 750거래일, 검증 60거래일, gap 5입니다.
OOS 예측은 {summary['oos_rows']}행이고 예측 분포는 하락 {summary['predicted_down']}·보합
{summary['predicted_neutral']}·상승 {summary['predicted_up']}입니다.

## 선정 기준별 잠정 1위

| 기준 | 조합·모델 | Accuracy | Macro F1 | 하락 Recall | 예측 분포 하/보/상 |
|---|---|---:|---:|---:|---:|
{chr(10).join(rows)}

Accuracy 우선 후보의 보합 예측 비중처럼 클래스 편향을 함께 확인한 뒤 기준을 확정해야 합니다.
`majority_accuracy` 사용 행은 현재 KOSPI200 보고서에 저장된 값의 단순 비교이며, 개별종목
ADR 0007의 폴드별 학습구간 최빈 기준선과 같은 값이라고 간주하지 않습니다.

정책별 전체 실측값은
[`reports/index_model_selection_comparison.json`](../../../../reports/index_model_selection_comparison.json)에
기록합니다. 지수 파일 SHA-256은 `{record['source']['index_sha256']}`입니다.
"""


def _render_index_long_only(
    index_winner,
    record: dict[str, Any],
    report: dict[str, Any],
) -> str:
    summary = record["summary"]
    threshold_range = (
        f"{summary['selected_up_threshold_min']:.4f}~"
        f"{summary['selected_up_threshold_max']:.4f}"
    )
    combination_header = (
        "| 순위 | 조합·모델 | 게이트 | Accuracy | 기준선 대비 | 승리 폴드 | "
        "상승 PR-AUC | 상승 Precision | ΔSharpe 중앙값 |"
    )
    rows = []
    seen_combinations: set[str] = set()
    for candidate in sorted(report["candidates"], key=lambda item: int(item["rank"])):
        combination = str(candidate["experiment"]["combination"])
        if combination in seen_combinations:
            continue
        seen_combinations.add(combination)
        item = candidate["summary"]
        rows.append(
            f"| {len(rows) + 1} | 조합 {combination} "
            f"{_variant_label(tuple(candidate['experiment'].get('return_features', [])))}·"
            f"{candidate['experiment']['model']} | {item['passes_operational_gate']} | "
            f"{item['accuracy']:.4f} | {item['accuracy_minus_training_majority_baseline']:+.4f} | "
            f"{item['baseline_win_folds']}/{item['folds']} | {item['pr_auc_up']:.4f} | "
            f"{item['up_precision']:.4f} | {item['delta_sharpe_net_median']:.4f} |"
        )
    contract = report["common_oos_contract"]
    policy = report["selection_policy"]
    multiple = policy["multiple_testing"]
    paired_t = multiple["winner_accuracy_paired_t"]
    return f"""{GENERATED_NOTICE}
# KOSPI200 long-only 개발구간 최종 선정 모델

## 상태

KOSPI200이 상승으로 예측된 경우만 매수하는 운영 목적에 맞춰 기존 100개 후보를 전부
같은 12폴드에서 다시 평가했습니다. 이슈 #216에서 합의한 선정 규칙에 따라 조합 C
LogisticRegression을 개발구간 최종 모델로 고정했으며, 봉인 홀드아웃은 사용하지 않았습니다.

| 항목 | 값 |
|---|---|
| 조합 | {index_winner.combination} + {_variant_label(index_winner.return_features)} |
| 모델 | {index_winner.model} |
| 피처 수 | {len(index_winner.feature_columns)} |
| 선정 규칙 | 운영 기준선 게이트 → 상승 PR-AUC → 비용 차감 ΔSharpe → 상승 Precision |
| 선정값(상승 PR-AUC) | {index_winner.selection_value:.4f} |

## 사용 피처

{_feature_lines(index_winner.feature_columns)}

## 개발구간 공통 OOS 결과

| 지표 | 값 |
|---|---:|
| Accuracy | {summary['accuracy']:.4f} |
| 학습구간 최빈 기준선 Accuracy | {summary['training_majority_baseline_accuracy']:.4f} |
| 기준선 대비 Accuracy | {summary['accuracy_minus_training_majority_baseline']:+.4f} |
| 기준선 승리 폴드 | {summary['baseline_win_folds']}/{summary['folds']} |
| Macro F1 | {summary['macro_f1']:.4f} |
| 상승 PR-AUC | {summary['pr_auc_up']:.4f} |
| 상승 Precision | {summary['up_precision']:.4f} |
| 상승 Recall | {summary['up_recall']:.4f} |
| 매수 신호 | {summary['buy_signals']} / 720 |
| 비용 차감 ΔSharpe 폴드 중앙값 | {summary['delta_sharpe_net_median']:.4f} |
| 내부 선택 상승 임계값 중앙값 | {summary['selected_up_threshold_median']:.4f} |
| 내부 선택 상승 임계값 범위 | {threshold_range} |

상승 임계값과 `None`/`balanced` 클래스 가중치는 각 외부 폴드의 학습구간 안에 둔 내부
60거래일 검증에서만 골랐습니다. 외부 검증 60일의 정답이나 확률로 임계값을 고르지 않았습니다.

`7/12`는 통계적 유의성 문턱이 아니라 저성능 후보를 거르는 최소 운영 안정성 관문입니다.
이 관문을 제거하고 상승 PR-AUC만 우선하면 조합 E LogisticRegression이 1위로 바뀔 수 있습니다.
조합 C는 관문 통과, 더 많은 매수 기회와 더 높은 상승 Recall을 함께 고려한 개발구간
선정 모델이며, 통계적 우위가 입증된 모델로 해석하지 않습니다.

## 다중 시도와 통계적 한계

- 후보 수: {multiple['candidate_count']}개
- Bonferroni 문턱: α = {multiple['bonferroni_alpha']:.4f}
- 조합 C의 폴드별 `Accuracy - 학습 최빈 기준선` 대응표본 단측 t검정:
  `t = {paired_t['t_statistic']:.4f}`, `p = {paired_t['p_value_one_sided']:.4f}`
- Bonferroni 보정 p값: {paired_t['bonferroni_adjusted_p_value']:.4f} — 통과하지 못함
- ΔSharpe의 다중 시도 기준: `SR*(N=100) = {multiple['deflated_sharpe_threshold_n_100']:.4f}`

따라서 조합 C는 개발구간 운영 기준에 따른 선택이지, 100개 후보 전체에 대해 통계적 우위가
확정됐다는 뜻은 아닙니다. 선정 모델과 규칙을 고정한 뒤 봉인 홀드아웃을 한 번만 개봉하며,
그 결과로 모델·피처·임계값을 다시 선택하지 않습니다.

## 전처리 누수 확인

LogisticRegression은 `StandardScaler → LogisticRegression`의 sklearn `Pipeline`입니다.
각 내부·외부 폴드에서 해당 학습 행으로만 `fit`하고 검증 행에는 `transform`만 적용하므로,
전체 기간을 먼저 표준화하는 누수는 없습니다. `hv_20`과 `vol_ratio_20`도 각 폴드
학습구간의 평균·표준편차만 사용해 변환됩니다.

## 조합별 개발구간 순위

{combination_header}
|---:|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

공통 피처 계산 가능 거래일은 `{contract['common_date_start']}`부터
`{contract['common_date_end']}`까지 {contract['common_date_rows']}행이고, 실제 공통 OOS 검증은
`{contract['oos_validation_start']}`부터 `{contract['oos_validation_end']}`까지 후보당
{contract['oos_rows_per_candidate']}행입니다. expanding 12폴드·최초 학습 750일·검증 60일·gap 5를
모든 후보에 똑같이 적용했습니다. 전체 결과는
[`reports/index_long_only_selection.json`](../../../../reports/index_long_only_selection.json)에
기록합니다. 지수 파일 SHA-256은 `{report['source']['index_sha256']}`입니다.
"""


def _ranking_summary(
    ranking_report: dict[str, Any] | None,
    index_winner,
    stock_winner,
) -> str:
    if not ranking_report:
        return "현재 두 1위 모델로 다시 만든 결합 출력 보고서가 없습니다."
    index_model = ranking_report.get("index_model", {})
    stock_model = ranking_report.get("stock_model", {})
    matches = (
        index_model.get("combination") == index_winner.combination
        and index_model.get("model") == index_winner.model
        and stock_model.get("combination") == stock_winner.combination
        and stock_model.get("model") == stock_winner.model
    )
    if not matches:
        return (
            "현재 결합 출력 보고서는 새 1위 조합과 일치하지 않습니다. "
            "`scripts/run_stock_index_ranking.py`를 다시 실행해야 합니다."
        )
    summary = ranking_report["full_prediction_summary"]
    return (
        f"공통 OOS {summary['dates']}거래일의 기본 출력은 {summary['rows']:,}행이며 전체 "
        f"적중률은 {summary['stock_hit_rate']:.4f}, 상승 Precision은 "
        f"{summary['up_precision']:.4f}, 실제 매수 신호는 {summary['buy_signal_rows']:,}건입니다."
    )


def _render_stock(
    stock_winner,
    selected: dict[str, Any],
    ranking_summary: str,
) -> str:
    return f"""{GENERATED_NOTICE}
# 개별종목 최종 모델

ADR 0007의 `학습 최빈 기준선 대비 Accuracy → Macro F1 → 기준선 승리 폴드 수` 순서로
선택한 개발구간 1위입니다.

| 항목 | 값 |
|---|---|
| 조합 | {stock_winner.combination} |
| 모델 | {stock_winner.model} |
| 피처 수 | {len(stock_winner.feature_columns)} |
| 기준선 승리 | {selected['baseline_win_folds']}/{selected['folds']} |

## 사용 피처

{_feature_lines(stock_winner.feature_columns, describe=True)}

## 개발구간 OOS 결과

| 지표 | 값 |
|---|---:|
| Accuracy | {selected['accuracy']:.4f} |
| 학습 최빈 기준선 Accuracy | {selected['training_majority_baseline_accuracy']:.4f} |
| 기준선 대비 Accuracy | {selected['accuracy_minus_training_majority_baseline']:+.4f} |
| Macro F1 | {selected['macro_f1']:.4f} |
| 하락 Recall | {selected['down_recall']:.4f} |
| 기존 3지표 조화평균 | {selected['core_harmonic_mean']:.4f} |
| Balanced Accuracy | {selected['balanced_accuracy']:.4f} |
| MCC | {selected['mcc']:.4f} |
| Macro PR-AUC | {selected['pr_auc_macro_ovr']:.4f} |
| 기준선 승리 폴드 | {selected['baseline_win_folds']}/{selected['folds']} |

공통 조건은 날짜 그룹 expanding 12폴드, 최초 학습 750거래일, 검증 60거래일,
gap 5입니다. 중립대는 `±2.0%`, 라벨은 T일 정보로 예측한
`T+1 adj_open → T+6 adj_open` 수익률입니다.

## 기본 출력과 매수 조건

매 거래일 업종 시가총액 상위 10개 × 업종별 보통주 시가총액 상위 5개, 최대 50종목을
전부 출력합니다. 종목별 예측·확률·실제 라벨·적중 여부와 업종별·전체 적중률을 남깁니다.
실제 매수는 KOSPI200과 개별종목이 모두 상승으로 예측된 경우만 허용합니다.

{ranking_summary}

Top 1·3·5는 기본 후보를 줄이는 규칙이 아니라 부가 실험입니다.
"""


def update_final_model_docs(root: Path = ROOT) -> dict[str, Path]:
    """현재 평가 보고서로 비교 JSON과 최종모델 README 세 개를 원자적으로 갱신한다."""

    index_path = root / INDEX_REPORT_RELATIVE
    preferred_index_path = preferred_index_report_path(root / "reports")
    stock_path = root / STOCK_REPORT_RELATIVE
    index_report = _read_json(index_path)
    stock_report = _read_json(stock_path)
    long_only_report = (
        _read_json(preferred_index_path)
        if preferred_index_path.name == LONG_ONLY_INDEX_REPORT_RELATIVE.name
        else None
    )
    if long_only_report is None:
        index_winner, stock_winner = load_winning_models(index_path, stock_path)
        index_record = _index_record(
            index_report,
            index_winner.combination,
            index_winner.model,
            list(index_winner.return_features),
        )
    else:
        index_sha = long_only_report.get("source", {}).get("index_sha256")
        stock_index_sha = stock_report.get("source", {}).get("index_sha256")
        if index_sha and stock_index_sha and index_sha != stock_index_sha:
            raise ValueError("KOSPI200과 개별종목 평가 보고서의 지수 데이터 SHA-256이 다릅니다.")
        index_winner = select_best_long_only_index_model(long_only_report)
        stock_winner = select_best_stock_model(stock_report)
        index_record = _long_only_index_record(
            long_only_report,
            index_winner.combination,
            index_winner.model,
            list(index_winner.return_features),
        )
    selected = stock_report["final_selection"]["selected"]

    comparison = compare_index_model_selection_policies(index_report)
    comparison["source_report"] = INDEX_REPORT_RELATIVE.as_posix()
    comparison["source"] = index_report.get("source", {})
    comparison_path = root / COMPARISON_REPORT_RELATIVE
    existing_comparison = _read_json(comparison_path) if comparison_path.exists() else None
    if existing_comparison is not None:
        comparable_existing = {
            key: value
            for key, value in existing_comparison.items()
            if key != "generated_at_utc"
        }
    else:
        comparable_existing = None
    if comparable_existing == comparison:
        comparison = existing_comparison
    else:
        comparison["generated_at_utc"] = datetime.now(timezone.utc).isoformat()

    ranking_path = root / RANKING_REPORT_RELATIVE
    ranking_report = _read_json(ranking_path) if ranking_path.exists() else None
    final_root = root / FINAL_ROOT_RELATIVE
    index_readme = final_root / "KOSPI200" / "README.md"
    stock_readme = final_root / "개별종목" / "README.md"
    root_readme = final_root / "README.md"
    for path in (index_readme, stock_readme, root_readme, comparison_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    comparison_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    root_readme.write_text(
        _render_root(index_winner, stock_winner, index_selected=long_only_report is not None),
        encoding="utf-8",
    )
    index_markdown = (
        _render_index(index_winner, index_record, comparison)
        if long_only_report is None
        else _render_index_long_only(index_winner, index_record, long_only_report)
    )
    index_readme.write_text(index_markdown, encoding="utf-8")
    stock_readme.write_text(
        _render_stock(
            stock_winner,
            selected,
            _ranking_summary(ranking_report, index_winner, stock_winner),
        ),
        encoding="utf-8",
    )
    return {
        "root_readme": root_readme,
        "index_readme": index_readme,
        "stock_readme": stock_readme,
        "comparison_report": comparison_path,
    }


def main() -> int:
    """CLI에서 최종모델 문서를 갱신한다."""

    paths = update_final_model_docs()
    for name, path in paths.items():
        print(f"{name}: {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
