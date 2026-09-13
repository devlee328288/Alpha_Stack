# dashboard/state.py
import streamlit as st
from dash_config import DATASETS, FEATURE_SETS, MODELS, Defaults

# ═══════════════════════════════════════════════════════════
# UNIVERSE (V2)
# ═══════════════════════════════════════════════════════════
_UNIVERSE_DEFAULTS = {
    "_universe_baseline_results": {},   # {ticker: baseline_result}
    "_universe_model_results": {},      # {ticker: {model: result}}
    "_universe_meta": {},               # {ticker: {code, name, ...}}
}

_DEFAULTS = {
    "dataset": Defaults.DATASET,
    "model": Defaults.MODEL,
    "feature_set": Defaults.FEATURE_SET,
    "start": Defaults.START,
    "end": Defaults.END,
    "cost": Defaults.COST,
    "seed": Defaults.SEED,
    "scope": "MARKET",       # MARKET | STOCK (V2: UNIVERSE)
    "ticker": "KOSPI200",    # scope=MARKET 이면 KOSPI200, STOCK 이면 종목코드
    "latest_model_result": None,
    "latest_backtest": None,
    "experiments": [],
}

_VALID = {
    "dataset": DATASETS,
    "model": MODELS,
    "feature_set": FEATURE_SETS,
}


def init_state():
    for k, v in _DEFAULTS.items():
        st.session_state.setdefault(k, v)
    for k, v in _UNIVERSE_DEFAULTS.items():
        st.session_state.setdefault(k, v)
    for k, valid in _VALID.items():
        if st.session_state.get(k) not in valid:
            st.session_state[k] = valid[0]


def ctx() -> dict:
    return {k: st.session_state[k] for k in
            ["dataset", "model", "feature_set", "start", "end", "cost", "seed",
             "scope", "ticker"]}


def fingerprint() -> str:
    c = ctx()
    return f"{c['scope']}|{c['ticker']}"


# ═══════════════════════════════════════════════════════════
# 결과 캐시 (scope:ticker 별 nested dict)
# ═══════════════════════════════════════════════════════════
def result_key(scope: str, ticker: str) -> str:
    return f"{scope}:{ticker}"


def get_result(kind: str, scope: str, ticker: str):
    """
    kind : "baseline" | "model_lab" | "bt" | "cost_grid" | "cost_be"
    """
    all_r = st.session_state.get(f"_{kind}_results", {})
    return all_r.get(result_key(scope, ticker))


def set_result(kind: str, scope: str, ticker: str, value) -> None:
    key = f"_{kind}_results"
    all_r = st.session_state.get(key, {})
    all_r[result_key(scope, ticker)] = value
    st.session_state[key] = all_r


def has_result(kind: str, scope: str, ticker: str) -> bool:
    v = get_result(kind, scope, ticker)
    if v is None:
        return False
    if hasattr(v, "empty") and hasattr(v, "shape"):
        try:
            return not bool(v.empty)
        except Exception:
            return False
    if hasattr(v, "__len__"):
        try:
            return len(v) > 0
        except Exception:
            return False
    return bool(v)


def list_result_keys() -> dict:
    """System 페이지에서 표시할 결과 키 요약."""
    out = {}
    for kind in ("baseline", "model_lab", "bt", "cost_grid", "cost_be"):
        all_r = st.session_state.get(f"_{kind}_results", {})
        if isinstance(all_r, dict) and all_r:
            out[kind] = list(all_r.keys())
    return out

# ═══════════════════════════════════════════════════════════
# UNIVERSE 결과 접근자
# ═══════════════════════════════════════════════════════════
def universe_get(kind: str) -> dict:
    """
    kind : "baseline" | "model"
    Returns {ticker: result}
    """
    key = f"_universe_{kind}_results"
    return st.session_state.get(key, {}) or {}


def universe_set(kind: str, ticker: str, value) -> None:
    key = f"_universe_{kind}_results"
    all_r = dict(st.session_state.get(key, {}) or {})
    all_r[ticker] = value
    st.session_state[key] = all_r


def universe_clear(kind: str | None = None) -> None:
    if kind is None:
        for k in ("baseline", "model"):
            st.session_state[f"_universe_{k}_results"] = {}
        st.session_state["_universe_meta"] = {}
    else:
        st.session_state[f"_universe_{kind}_results"] = {}