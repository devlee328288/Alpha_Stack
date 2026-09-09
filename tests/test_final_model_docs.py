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
