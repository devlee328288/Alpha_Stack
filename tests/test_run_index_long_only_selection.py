import numpy as np
import pytest

from scripts.run_index_long_only_selection import (
    _json_clean,
    _optional_class_weight,
    selection_audit,
)


def test_가중치_미사용값만_json_null로_복원한다():
    assert _optional_class_weight(np.nan) is None
    assert _json_clean({"class_weight": np.nan}) == {"class_weight": None}


def test_성능지표의_nan은_null로_숨기지_않고_중단한다():
    with pytest.raises(ValueError, match="accuracy"):
        _json_clean({"metrics": {"accuracy": np.nan}})


def test_선정감사는_관문을_유의성으로_오해하지_않게_기록한다():
    differences = [0.01] * 7 + [-0.01] * 5
    candidate = {
        "outer_fold_results": [
            {"accuracy_minus_training_majority_baseline": value}
            for value in differences
        ]
    }

    audit = selection_audit([candidate])

    assert "통계적 유의성 문턱이 아니라" in audit["operational_gate_interpretation"]
    assert audit["multiple_testing"]["candidate_count"] == 100
    assert audit["multiple_testing"]["bonferroni_alpha"] == pytest.approx(0.0005)
    assert audit["preprocessing"]["implementation"] == "sklearn Pipeline"
    assert audit["preprocessing"]["full_period_scaling"] is False
