import json
import shutil
from pathlib import Path

from scripts.update_final_model_docs import update_final_model_docs

ROOT = Path(__file__).resolve().parents[1]


def _copy_reports(target: Path) -> None:
    reports = target / "reports"
    reports.mkdir(parents=True)
    for filename in (
        "model_sweep.json",
        "stock_feature_combinations.json",
        "stock_index_ranking.json",
    ):
        shutil.copyfile(ROOT / "reports" / filename, reports / filename)


def test_평가보고서의_새_1위와_피처가_최종모델_readme에_자동반영된다(tmp_path):
    _copy_reports(tmp_path)
    first_paths = update_final_model_docs(tmp_path)

    first_index = first_paths["index_readme"].read_text(encoding="utf-8")
    first_stock = first_paths["stock_readme"].read_text(encoding="utf-8")
    assert "조합 | E + 5Day Return" in first_index
    assert "조합 | K" in first_stock
    assert "scripts/update_final_model_docs.py가 생성합니다" in first_index

    index_path = tmp_path / "reports" / "model_sweep.json"
    index_report = json.loads(index_path.read_text(encoding="utf-8"))
    index_report["experiments"][0]["summary"]["core_harmonic_mean"] = 1.0
    index_path.write_text(
        json.dumps(index_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    stock_path = tmp_path / "reports" / "stock_feature_combinations.json"
    stock_report = json.loads(stock_path.read_text(encoding="utf-8"))
    selected = stock_report["final_selection"]["selected"]
    selected["combination"] = "Z"
    selected["model"] = "RandomForest"
    selected["feature_columns"] = ["new_feature"]
    stock_path.write_text(
        json.dumps(stock_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    second_paths = update_final_model_docs(tmp_path)
    second_index = second_paths["index_readme"].read_text(encoding="utf-8")
    second_stock = second_paths["stock_readme"].read_text(encoding="utf-8")

    assert "조합 | A + 기본" in second_index
    assert "조합 | Z" in second_stock
    assert "모델 | RandomForest" in second_stock
    assert "`new_feature`" in second_stock
    assert "결합 출력 보고서는 새 1위 조합과 일치하지 않습니다" in second_stock


def test_완료된_long_only_보고서가_있으면_kospi200_선정문서에_우선반영한다(tmp_path):
    _copy_reports(tmp_path)
    stock_report = json.loads(
        (tmp_path / "reports" / "stock_feature_combinations.json").read_text(encoding="utf-8")
    )
    report = {
        "status": "complete",
        "holdout_used": False,
        "expected_candidate_count": 1,
        "source": {
            "index_sha256": stock_report["source"]["index_sha256"],
            "generated_at": "now",
        },
            "common_oos_contract": {
            "common_date_start": "20140101",
            "common_date_end": "20240831",
            "common_date_rows": 3000,
            "oos_validation_start": "20140217",
            "oos_validation_end": "20240831",
                "oos_rows_per_candidate": 720,
            },
            "selection_policy": {
                "multiple_testing": {
                    "candidate_count": 100,
                    "bonferroni_alpha": 0.0005,
                    "winner_accuracy_paired_t": {
                        "t_statistic": 1.0092,
                        "p_value_one_sided": 0.1673,
                        "bonferroni_adjusted_p_value": 1.0,
                    },
                    "deflated_sharpe_threshold_n_100": 0.8115,
                }
            },
            "candidates": [
            {
                "rank": 1,
                "experiment": {"combination": "C", "model": "LogisticRegression"},
                "feature_columns": ["rsi_14"],
                "summary": {
                    "passes_operational_gate": True,
                    "accuracy": 0.45,
                    "training_majority_baseline_accuracy": 0.40,
                    "accuracy_minus_training_majority_baseline": 0.05,
                    "baseline_win_folds": 7,
                    "folds": 12,
                    "macro_f1": 0.41,
                    "pr_auc_up": 0.40,
                    "up_precision": 0.42,
                    "up_recall": 0.50,
                    "buy_signals": 300,
                    "delta_sharpe_net_median": 0.20,
                    "selected_up_threshold_median": 0.35,
                    "selected_up_threshold_min": 0.20,
                    "selected_up_threshold_max": 0.50,
                },
            }
        ],
    }
    path = tmp_path / "reports" / "index_long_only_selection.json"
    path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")

    paths = update_final_model_docs(tmp_path)
    index_markdown = paths["index_readme"].read_text(encoding="utf-8")
    root_markdown = paths["root_readme"].read_text(encoding="utf-8")

    assert "조합 | C + 기본" in index_markdown
    assert "조합 C LogisticRegression + 기본" in root_markdown
    assert "개발구간 최종 선정 모델" in index_markdown
    assert "| 확정 | 이슈 #216 개발구간 선정 규칙 |" in root_markdown
