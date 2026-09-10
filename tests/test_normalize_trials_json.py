import json

import pytest

from scripts.normalize_trials_json import normalize_trials_text


def _line(trial_id: str, class_weight: str) -> str:
    return (
        '{"trial_id": "'
        + trial_id
        + '", "fit": {"class_weight": '
        + class_weight
        + "}}"
    )


def test_과거_class_weight_nan과_trial_id만_표준_json으로_고친다():
    text = _line("run-fold-01-inner-nan", "NaN") + "\n"

    normalized, counts = normalize_trials_text(text)
    record = json.loads(normalized)

    assert record["trial_id"] == "run-fold-01-inner-none"
    assert record["fit"]["class_weight"] is None
    assert counts["class_weight_nan_to_null"] == 1
    assert counts["inner_nan_to_none"] == 1


def test_성능지표_nan은_메타데이터처럼_고치지_않는다():
    text = '{"trial_id": "run", "metrics": {"accuracy": NaN}}\n'

    with pytest.raises(ValueError, match="class_weight 이외"):
        normalize_trials_text(text)


def test_정규화로_trial_id가_겹치면_중단한다():
    text = "\n".join(
        (
            _line("run-inner-nan", "NaN"),
            _line("run-inner-none", "null"),
        )
    )

    with pytest.raises(ValueError, match="중복"):
        normalize_trials_text(text)
