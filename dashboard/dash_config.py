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

# ── Scope / Ticker (V2 개별종목 확장 대비) ──────────────────
SCOPES = ["MARKET", "STOCK"]     # STOCK 은 V2 에서 활성화

DEFAULT_SCOPE = "MARKET"
DEFAULT_TICKER = "KOSPI200"

# MARKET scope 의 ticker 들
MARKET_TICKERS = ["KOSPI200"]

# STOCK scope 의 ticker 들 (V2 에서 채워짐)
STOCK_TICKERS: list[str] = []    # 예: ["005930", "000660", ...]

# scope 별 ticker 소스
def tickers_for(scope: str) -> list[str]:
    if scope == "MARKET":
        return MARKET_TICKERS
    if scope == "STOCK":
        return STOCK_TICKERS
    return []