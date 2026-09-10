# dashboard/state.py
import streamlit as st
from dash_config import DATASETS, FEATURE_SETS, MODELS, Defaults

_DEFAULTS = {
    "dataset": Defaults.DATASET,
    "model": Defaults.MODEL,
    "feature_set": Defaults.FEATURE_SET,
    "start": Defaults.START,
    "end": Defaults.END,
    "cost": Defaults.COST,
    "seed": Defaults.SEED,
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
    # 세션에 남은 옛 선택값이 지금 선택지에 없으면 리셋
    for k, valid in _VALID.items():
        if st.session_state.get(k) not in valid:
            st.session_state[k] = valid[0]


def ctx() -> dict:
    return {k: st.session_state[k] for k in
            ["dataset", "model", "feature_set", "start", "end", "cost", "seed"]}


def fingerprint() -> str:
    c = ctx()
    return f"{c['dataset']}|{c['model']}|{c['feature_set']}|{c['start']}|{c['end']}|{c['seed']}"
