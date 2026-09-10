# dashboard/dash_config.py
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Defaults:
    DATASET = "KOSPI200"
    MODEL = "LightGBM"
    FEATURE_SET = "F"
    START = date(2018, 1, 1)
    END = date(2024, 8, 22)
    COST = 0.0010
    SEED = 42


DATASETS = ["KOSPI200"]

MODELS = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"]

FEATURE_SETS = ["A", "B", "C", "D", "E", "F"]

FEATURE_SET_LABELS = {
    "A": "A · Technical (4)",
    "B": "B · Momentum (4)",
    "C": "C · Tech + Momentum (6)",
    "D": "D · Trend + Vol (6)",
    "E": "E · Vol only (3)",
    "F": "F · Full (8)",
}

HORIZON = 5
