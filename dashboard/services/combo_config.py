# dashboard/services/combo_config.py
"""Scope별 피처 조합 매핑. MARKET/UNIVERSE = C, STOCK = K (다음 턴)."""

from __future__ import annotations

SCOPE_COMBOS: dict[str, dict] = {
    "MARKET": {"combination": "C", "return_features": (), "label": "COMBINATION C"},
    "UNIVERSE": {"combination": "C", "return_features": (), "label": "COMBINATION C"},
    "STOCK": {"combination": "K", "return_features": (), "label": "COMBINATION K"},
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
