import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Model Lab · AlphaStack", layout="wide")

import theme
from theme import (
    metric_row,
    section_header,
    top_strip,
    panel,
    show_table,
)

theme.inject()

from components import sidebar_controls
from services import model_service, comparison_service
from state import ctx, get_result, set_result

sidebar_controls()
c = ctx()
scope = c["scope"]
ticker = c["ticker"]

# ── UNIVERSE scope 는 전용 페이지에서 ──
if scope == "UNIVERSE":
    st.markdown(
        '<div class="as-title">전용 페이지로 이동</div>', unsafe_allow_html=True
    )
    st.info(
        "**UNIVERSE scope 는 Universe 페이지에서만 사용합니다.**  \n"
        "사이드바에서 **UNIVERSE** 버튼을 다시 누르거나, "
        "페이지 목록의 **Universe** 를 여세요."
    )
    if st.button("▶  Universe 페이지로", type="primary"):
        st.switch_page("pages/8_Universe.py")
    st.stop()

# ═══════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════
st.markdown('<div class="as-title">Model Lab</div>', unsafe_allow_html=True)

_err = model_service.engine_status()
if _err:
    top_strip([scope, ticker], status_text="IMPORT FAIL", status_tone="warn")
    st.error(f"repo engine import 실패: {_err}")
    st.stop()

top_strip(
    [scope, ticker, "COMBINATION E", "RETURN · 5DAY", "12-FOLD EXPANDING"],
    status_text="READY",
    status_tone="neutral",
)

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════
section_header("CONFIG")
with panel():
    c1, c2 = st.columns([3, 1], gap="small")
    with c1:
        selected = st.multiselect(
            "Models",
            list(model_service.MODELS),
            default=list(model_service.MODELS),
        )
    with c2:
        st.write("")
        run = st.button("▶  RUN", type="primary", use_container_width=True)

    # ── Label Source ──────────────────────────────
    c1, c2 = st.columns([3, 2], gap="small")
    with c1:
        label_source = st.radio(
            "Label Source",
            ["fwd_return", "adaptive"],
            format_func=lambda x: {
                "fwd_return": "fwd_return ±1%",
                "adaptive": "Adaptive 6-param",
            }[x],
            horizontal=True,
            key="_ml_label_source",
        )
    with c2:
        if label_source == "adaptive":
            _bl = get_result("baseline", scope, ticker)
            if _bl is None:
                st.warning("⚠️ Baseline 페이지에서 **RUN BASELINE** 필요")
            else:
                st.caption(f"{_bl['total_folds']}폴드 per-fold 파라미터로 라벨")
        else:
            st.caption("라벨: `fwd_return ± 1%` (T+1→T+6)")

# ═══════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════
if run and selected:
    import traceback
    from services import baseline_service

    adaptive_labels = None

    if label_source == "adaptive":
        _bl = get_result("baseline", scope, ticker)
        if _bl is None:
            st.error(
                "Baseline 결과 없음. Baseline 페이지에서 RUN BASELINE 먼저 실행하세요."
            )
            st.stop()
        _fold_details = _bl.get("fold_details", [])
        if not _fold_details:
            st.error("Baseline의 fold_details 없음.")
            st.stop()
        with st.spinner("Adaptive 라벨 생성 중…"):
            try:
                _ad = baseline_service.make_adaptive_labels_from_baseline(
                    scope,
                    ticker,
                    _fold_details,
                )
                adaptive_labels = _ad["labels"]
                st.caption(f"Adaptive 라벨: {_ad['n_valid']}/{_ad['n_total']} 유효")
            except Exception as e:
                st.error(f"Adaptive 라벨 생성 실패: {type(e).__name__}: {e}")
                with st.expander("Traceback", expanded=True):
                    st.code(traceback.format_exc())
                st.stop()

    progress = st.progress(0.0)
    status = st.empty()

    def _cb(i, total, name):
        if total > 0:
            progress.progress(min(i / total, 1.0))
        status.caption(f"[{i}/{total}] {name}")

    try:
        with st.spinner(f"학습 · {label_source} · {scope}:{ticker}…"):
            results = model_service.run_all_models(
                scope=scope,
                ticker=ticker,
                models=tuple(selected),
                label_source=label_source,
                adaptive_labels=adaptive_labels,
                progress_cb=_cb,
            )

        slot_map = {
            "fwd_return": "model_lab",
            "adaptive": "model_lab_adaptive",
        }
        slot = slot_map[label_source]
        set_result(slot, scope, ticker, results)
        progress.progress(1.0)
        status.caption(f"완료 · {len(results)} 모델 · {label_source}")
        st.rerun()
    except Exception as e:
        st.error(f"실행 실패: {type(e).__name__}: {e}")
        with st.expander("Traceback", expanded=True):
            st.code(traceback.format_exc())

# ═══════════════════════════════════════════════════════════
# RESULT · 두 source 다 읽음
# ═══════════════════════════════════════════════════════════
results_fwd = get_result("model_lab", scope, ticker)
results_ad = get_result("model_lab_adaptive", scope, ticker)

_all_slots = {
    "fwd_return": results_fwd,
    "adaptive": results_ad,
}
_available = [k for k, v in _all_slots.items() if v]

if not _available:
    st.caption(f"{scope}:{ticker} 결과 없음. RUN 을 누르세요.")
    st.stop()

# ── 라벨 소스 비교 (2개 이상 있을 때) ────────────
if len(_available) >= 2:
    section_header("LABEL SOURCE · COMPARISON")
    st.caption(
        "ℹ️ **fwd_return** = 미래 5일 수익률 라벨 (예측 과제) ·  "
        "**adaptive** = per-fold 파라미터 라벨 (판정 재현 과제)."
    )

    _cmp_rows = []
    for src in _available:
        _k = comparison_service.build_summary_kpis(_all_slots[src])
        _cmp_rows.append(
            {
                "SOURCE": src,
                "BEST": _k["best_model"],
                "ACC": _k["best_accuracy"],
                "MACRO F1": _k["best_macro_f1"],
                "DOWN REC": _k["best_down_recall"],
                "HARMONIC": _k["best_harmonic"],
                "ΔSHARPE": _k["best_delta_sharpe"],
            }
        )
    _cmp = pd.DataFrame(_cmp_rows)
    with panel(f"{len(_available)} 라벨 소스 · 각 best model 성능"):
        show_table(
            _cmp,
            num_cols=["ACC", "MACRO F1", "DOWN REC", "HARMONIC", "ΔSHARPE"],
            precision=4,
        )

# ── VIEW 소스 선택 ─────────────────────────────
if len(_available) > 1:
    section_header("VIEW")
    _view = st.radio(
        "결과 소스",
        _available,
        format_func=lambda x: {
            "fwd_return": "fwd_return 결과",
            "adaptive": "Adaptive 결과",
        }[x],
        horizontal=True,
        key="_ml_view_src",
    )
else:
    _view = _available[0]

results = _all_slots[_view]

# ═══════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════
kpi = comparison_service.build_summary_kpis(results)
cmp_df = comparison_service.build_comparison_table(results)

section_header(f"SUMMARY · {scope}:{ticker} · {_view}")
metric_row(
    [
        dict(label="BEST MODEL", value=kpi["best_model"], tone="up", accent=True),
        dict(label="HARMONIC", value=f"{kpi['best_harmonic']:.4f}", tone="up"),
        dict(label="ACCURACY", value=f"{kpi['best_accuracy']:.4f}"),
        dict(label="MACRO F1", value=f"{kpi['best_macro_f1']:.4f}"),
        dict(label="DOWN RECALL", value=f"{kpi['best_down_recall']:.4f}"),
        dict(
            label="ΔSHARPE",
            value=f"{kpi['best_delta_sharpe']:+.4f}",
            tone="up" if kpi["best_delta_sharpe"] > 0 else "down",
        ),
    ],
    cols=6,
)

# ═══════════════════════════════════════════════════════════
# MODEL COMPARISON
# ═══════════════════════════════════════════════════════════
section_header(f"MODEL COMPARISON · {len(cmp_df)} MODELS")
with panel(
    "12-FOLD OOS · SORT BY HARMONIC (ACC · F1 · DOWN RECALL 조화평균)",
    status_text=f"BEST · {kpi['best_model']}",
    status_tone="up",
):
    st.caption(
        "⚠️ **HARMONIC = ACC·Macro F1·하락 Recall의 조화평균**  \n"
        "세 지표 중 하나라도 낮으면 전체가 급락 → **하락 예측을 못 하는 모델은 후순위**."
    )
    display = cmp_df.rename(columns=comparison_service.COMPARISON_COLUMNS)
    show_table(
        display,
        num_cols=[
            "ACC",
            "MACRO F1",
            "DOWN RECALL",
            "HARMONIC",
            "BAL ACC",
            "MAJORITY",
            "ΔSHARPE",
        ],
        precision=4,
        highlight_row=0,
    )

# ═══════════════════════════════════════════════════════════
# MODEL DETAIL
# ═══════════════════════════════════════════════════════════
section_header("MODEL DETAIL")
pick = st.selectbox("Inspect model", list(results.keys()), index=0)
r = results[pick]
s = r["summary"]

metric_row(
    [
        dict(label="OOS ROWS", value=f"{s['oos_rows']}"),
        dict(label="ACC", value=f"{s['accuracy']:.4f}"),
        dict(label="MACRO F1", value=f"{s['macro_f1']:.4f}"),
        dict(label="DOWN REC", value=f"{s['down_recall']:.4f}"),
        dict(label="NEUTRAL REC", value=f"{s['neutral_recall']:.4f}"),
        dict(label="UP REC", value=f"{s['up_recall']:.4f}"),
    ],
    cols=6,
)

fold_df = pd.DataFrame(r["fold_results"])
if not fold_df.empty:
    section_header("FOLD RESULTS")
    with panel(f"{len(fold_df)} FOLDS"):
        show_table(
            fold_df,
            num_cols=[
                "accuracy",
                "macro_f1",
                "down_recall",
                "core_harmonic_mean",
                "delta_sharpe_net",
                "strategy_sharpe_net",
                "buyhold_sharpe_net",
            ],
            precision=4,
        )

wc = pd.DataFrame(r["weight_counts"])
if not wc.empty:
    section_header("CLASS WEIGHT SELECTION")
    with panel():
        show_table(wc, num_cols=["선택 폴드 수"], precision=0)


# ═══════════════════════════════════════════════════════════
# ADAPTIVE LABELS (per-fold 파라미터로 라벨 생성)
# ═══════════════════════════════════════════════════════════
def make_adaptive_labels_from_baseline(scope: str, ticker: str, fold_details):
    """
    Baseline 의 fold_details (per-fold 파라미터) 로 전체 df 라벨 생성.

    각 row 의 시점에 따라 "가장 최근에 학습된 fold" 의 파라미터를 사용.
    Walk-forward 정신에 맞음 (미래 정보 누설 없음).

    Returns
    -------
    dict : {"labels": list[float], "n_valid": int, "n_total": int}
        labels 값은 {0.0=하락, 1.0=중립, 2.0=상승} 또는 NaN
    """
    import numpy as np
    import pandas as pd
    from services import data_loader
    from step5_optimize_6params import make_baseline_labels

    _IMPORT_ERR = None

    if _IMPORT_ERR:
        raise RuntimeError(f"repo engine import 실패: {_IMPORT_ERR}")

    df = data_loader.load_market_data(scope, ticker)
    n = len(df)

    fold_df = pd.DataFrame(fold_details)
    if fold_df.empty:
        raise ValueError("fold_details 비어있음")
    if "train_end" not in fold_df.columns:
        raise ValueError("train_end 컬럼 없음")

    fold_df = fold_df.sort_values("train_end").reset_index(drop=True)
    train_ends = fold_df["train_end"].astype(int).tolist()

    # 각 row 에 어느 fold 파라미터를 배정할지
    fold_assign = np.full(n, -1, dtype=int)
    for i, te in enumerate(train_ends):
        next_te = train_ends[i + 1] if i + 1 < len(train_ends) else n
        fold_assign[te:next_te] = i
    # 앞부분 (train_end 이전) 은 -1 유지 → NaN

    # fold 별 라벨 생성
    labels = np.full(n, np.nan, dtype=float)
    for i in range(len(fold_df)):
        mask = fold_assign == i
        if not mask.any():
            continue
        row = fold_df.iloc[i]
        try:
            params = {
                "alpha_up": float(row["alpha_up"]),
                "alpha_down": float(row["alpha_down"]),
                "beta_up": float(row["beta_up"]),
                "beta_down": float(row["beta_down"]),
                "vol_period": int(round(float(row["vol_period"]))),
                "volume_period": int(round(float(row["volume_period"]))),
            }
            fold_labels = make_baseline_labels(df, params)
            labels[mask] = fold_labels[mask]
        except Exception:
            continue

    valid_mask = np.isfinite(labels)
    return {
        "labels": labels.tolist(),
        "n_valid": int(valid_mask.sum()),
        "n_total": n,
    }
