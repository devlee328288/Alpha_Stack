import json
from collections import Counter

from scripts.backfill_model_trials import EXPECTED_COUNTS, build_historical_trials


def test_보존된_성공_fit을_한_행씩_복원한다():
    records = build_historical_trials()

    assert len(records) == 4224
    assert Counter(record["run"] for record in records) == Counter(EXPECTED_COUNTS)
    assert len({record["trial_id"] for record in records}) == len(records)


def test_KOSPI200_A부터_F까지_모든_조합과_수익률변형을_기록한다():
    records = [
        record
        for record in build_historical_trials()
        if record["run"] == "kospi200_combination_sweep"
    ]

    assert {record["experiment"]["combination"] for record in records} == set("ABCDEF")
    assert {record["experiment"]["variant"] for record in records} == {
        "base",
        "daily",
        "five_day",
        "both",
    }
    assert Counter(record["fit"]["phase"] for record in records) == {
        "inner_class_weight_selection": 2304,
        "outer_evaluation": 1152,
    }


def test_JSONL에_NaN을_쓰지_않는다():
    records = build_historical_trials()

    encoded = [json.dumps(record, ensure_ascii=False, allow_nan=False) for record in records]
    assert all('"origin": "historical_backfill"' in line for line in encoded)
    assert all('"status": "success"' in line for line in encoded)
