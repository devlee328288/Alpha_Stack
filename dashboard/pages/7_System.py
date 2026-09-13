import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

st.set_page_config(page_title="System · AlphaStack", layout="wide")

import theme
from theme import (
    metric_row, section_header, top_strip, panel, show_table,
    status_dot,
)
theme.inject()

from components import sidebar_controls
from services import system_service

sidebar_controls()

# ═══════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════
st.markdown('<div class="as-title">System</div>', unsafe_allow_html=True)

sys_all = system_service.collect_all()

_ok_engines = sum(1 for v in sys_all["engines"].values() if v.get("ok"))
_n_engines = len(sys_all["engines"])

top_strip(
    ["REPO", "ALPHASTACK",
     f"ENGINES {_ok_engines}/{_n_engines}",
     sys_all["collected_at"]],
    status_text="READY" if _ok_engines == _n_engines else "PARTIAL",
    status_tone="up" if _ok_engines == _n_engines else "warn",
)

# ═══════════════════════════════════════════════════════════
# ① HF SNAPSHOT
# ═══════════════════════════════════════════════════════════
section_header("HF SNAPSHOT")

hf = sys_all["hf"]
_hf_tone = "up" if hf["ok"] else "warn"

with panel(
    hf["repo_id"],
    status_text=("OK" if hf["ok"] else "FAIL"),
    status_tone=_hf_tone,
):
    rows = [
        ("repo_sha", hf["repo_sha"]),
        ("dev_end", hf["dev_end"]),
        ("index_sha256", hf["index_sha256"]),
        ("source", hf["source"]),
        ("cache_dir", hf["cache_dir"]),
        ("cache_exists", "yes" if hf["cache_exists"] else "no"),
    ]
    if hf.get("err"):
        rows.append(("error", hf["err"]))

    for k, v in rows:
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;'
            f'padding:4px 0;border-bottom:1px solid var(--border-subtle);">'
            f'<span style="font-family:var(--font-ui);font-size:11px;'
            f'text-transform:uppercase;letter-spacing:0.06em;'
            f'color:var(--text-muted);">{k}</span>'
            f'<span style="font-family:var(--font-num);font-size:11px;'
            f'color:var(--text-primary);">{v}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

st.caption(
    "dev_end / index_sha256 은 Baseline 또는 Model Lab 을 RUN 한 후 "
    "session_state 에서 자동으로 채워집니다."
)

# ═══════════════════════════════════════════════════════════
# ② ENGINE STATUS
# ═══════════════════════════════════════════════════════════
section_header("ENGINE STATUS")

engine_rows = []
for name, info in sys_all["engines"].items():
    engine_rows.append({
        "SERVICE": name,
        "STATUS": "● real" if info.get("ok") else "○ fail",
        "DETAIL": (info.get("err") or "")[:60],
        "EXTRA": ", ".join(
            f"{k}={v}" for k, v in info.items()
            if k not in ("ok", "err") and isinstance(v, bool)
        ),
    })

with panel(f"{_n_engines} SERVICES", status_text=f"{_ok_engines}/{_n_engines} OK",
           status_tone="up" if _ok_engines == _n_engines else "warn"):
    show_table(pd.DataFrame(engine_rows))

# ═══════════════════════════════════════════════════════════
# ③ PACKAGES
# ═══════════════════════════════════════════════════════════
section_header("PACKAGES")

pkg_df = pd.DataFrame(sys_all["packages"])
_missing = pkg_df[pkg_df["version"] == "(not installed)"].shape[0]

with panel(
    f"{len(pkg_df)} PACKAGES",
    status_text=f"{_missing} missing" if _missing else "ALL INSTALLED",
    status_tone="warn" if _missing else "up",
):
    show_table(pkg_df)

# ═══════════════════════════════════════════════════════════
# ④ SESSION
# ═══════════════════════════════════════════════════════════
section_header("SESSION")

sess = sys_all["session"]

metric_row([
    dict(label="SCOPE", value=sess["scope"]),
    dict(label="TICKER", value=sess["ticker"]),
    dict(label="SESSION KEYS", value=f"{sess['n_session_keys']}"),
    dict(label="RESULT KEYS", value=f"{len(sess['result_keys'])}"),
], cols=4)

if sess["result_keys"]:
    section_header("RESULT KEYS")
    rk_df = pd.DataFrame([
        {"key": k, "n_entries": n} for k, n in sess["result_keys"]
    ])
    with panel():
        show_table(rk_df, num_cols=["n_entries"], precision=0)

if sess["all_underscore_keys"]:
    with st.expander("전체 underscore session keys"):
        for k in sess["all_underscore_keys"]:
            v = st.session_state.get(k)
            t = type(v).__name__
            size = len(v) if hasattr(v, "__len__") else "—"
            st.markdown(f"- `{k}` · type=`{t}` · size=`{size}`")

# ═══════════════════════════════════════════════════════════
# ⑤ CACHE CLEAR (개발용)
# ═══════════════════════════════════════════════════════════
section_header("DEVELOPER")

with panel("CACHE / STATE"):
    c1, c2, c3 = st.columns(3, gap="small")
    with c1:
        if st.button("Clear service cache", use_container_width=True):
            try:
                from services import (
                    baseline_service, model_service,
                    backtest_service, hf_service,
                )
                baseline_service.clear_cache()
                model_service.clear_cache()
                backtest_service.clear_cache()
                hf_service.clear_cache()
                st.success("service 캐시 초기화 완료")
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")
    with c2:
        if st.button("Clear session results", use_container_width=True):
            for k in ["_baseline_results", "_model_lab_results",
                      "_bt_results", "_cost_grid", "_cost_be"]:
                st.session_state.pop(k, None)
            st.success("session 결과 초기화 완료")
    with c3:
        if st.button("Hard refresh", use_container_width=True):
            st.rerun()