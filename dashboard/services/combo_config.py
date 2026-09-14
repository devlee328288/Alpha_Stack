# dashboard/services/combo_config.py
"""
Scope별 피처 조합 매핑.

- MARKET   : 조합 C (KOSPI200 지수, 6 features)
- UNIVERSE : 조합 K (개별종목 panel, 14 features)
- STOCK    : 조합 K (개별종목 panel, 14 features)
"""

from __future__ import annotations

# ── 조합 K (개별종목, 14 features) ─────────────────────
_K_FEATURES = (
    "atr_ratio",
    "bb_bandwidth",
    "hv_regime",
    "five_day_return",
    "relative_ret_5_market",
    "sma_gap_5_20",
    "sma_gap_20_60",
    "rsi_14",
    "macd_hist_ratio",
    "bb_position",
    "hv_20",
    "vol_ratio_20",
    "obv_slope_20",
    "daily_return",
)

# ── 조합 C (KOSPI200 지수, 6 features) ─────────────────
_C_FEATURES = (
    "sma_gap_5_20",
    "macd_hist_ratio",
    "rsi_14",
    "bb_position",
    "hv_20",
    "vol_ratio_20",
)


SCOPE_COMBOS: dict[str, dict] = {
    "MARKET": {
        "combination": "C",
        "return_features": (),
        "label": "COMBINATION C",
        "features": _C_FEATURES,
    },
    "UNIVERSE": {
        "combination": "K",
        "return_features": (),
        "label": "COMBINATION K",
        "features": _K_FEATURES,
    },
    "STOCK": {
        "combination": "K",
        "return_features": (),
        "label": "COMBINATION K",
        "features": _K_FEATURES,
    },
}

_DEFAULT = SCOPE_COMBOS["MARKET"]


def get_combo(scope: str) -> dict:
    return SCOPE_COMBOS.get(scope, _DEFAULT)


def combination_of(scope: str) -> str:
    return str(get_combo(scope)["combination"])


def return_features_of(scope: str) -> tuple:
    return tuple(get_combo(scope)["return_features"])


def combo_label(scope: str) -> str:
    return str(get_combo(scope)["label"])


def features_of(scope: str) -> tuple:
    return tuple(get_combo(scope).get("features", ()))


def n_features_of(scope: str) -> int:
    return len(features_of(scope))
