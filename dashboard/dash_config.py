# dashboard/dash_config.py
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Defaults:
    DATASET = "KOSPI200"
    MODEL = "LightGBM"
    FEATURE_SET = "F"          # 기본값: F (8피처)
    START = date(2018, 1, 1)
    END = date(2024, 8, 22)
    COST = 0.0010
    SEED = 42


DATASETS = ["KOSPI200"]

MODELS = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"]

# 실제 엔진의 COMBINATION_FEATURES 키
FEATURE_SETS = ["A", "B", "C", "D", "E", "F"]

# 사이드바 라벨 (표시용)
FEATURE_SET_LABELS = {
    "A": "A · Technical (4)",
    "B": "B · Momentum (4)",
    "C": "C · Tech + Momentum (6)",
    "D": "D · Trend + Vol (6)",
    "E": "E · Vol only (3)",
    "F": "F · Full (8)",
}

HORIZON = 5
