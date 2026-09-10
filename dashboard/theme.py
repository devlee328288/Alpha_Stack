"""
AlphaStack Terminal (A안) — 디자인 시스템.
CSS 주입 + 컴포넌트 헬퍼.

사용법:
    import theme
    theme.inject()                      # 페이지 최상단에서 1회
    theme.metric_panel("SHARPE", "1.08", "+0.24 vs base", tone="up")
"""

from __future__ import annotations

import streamlit as st

# ─────────────────────────────────────────────────────────────
# 1. CSS (디자인 토큰 + 컴포넌트 스타일)
# ─────────────────────────────────────────────────────────────

CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  /* 배경 계층 */
  --bg-base:      #0a0c10;
  --bg-panel:     #11141a;
  --bg-elevated:  #161a22;
  --bg-hover:     #1c2028;

  /* 경계선 */
  --border-subtle: rgba(255,255,255,0.05);
  --border-base:   rgba(255,255,255,0.09);
  --border-strong: rgba(255,255,255,0.16);

  /* 텍스트 */
  --text-primary:   #e6e8ec;
  --text-secondary: #9aa0a6;
  --text-muted:     #5c6470;
  --text-inverse:   #0a0c10;

  /* 시맨틱 */
  --color-up:      #4ade80;
  --color-down:    #ff5c5c;
  --color-neutral: #9aa0a6;
  --color-warn:    #fbbf24;
  --color-accent:  #7fd1ff;
  --color-focus:   #a78bfa;

  /* 차트 */
  --chart-grid: rgba(255,255,255,0.06);
  --chart-1:    #7fd1ff;
  --chart-2:    #4ade80;
  --chart-3:    #9aa0a6;
  --chart-4:    #fbbf24;

  /* 타이포 */
  --font-ui:  'Inter', -apple-system, 'Segoe UI', sans-serif;
  --font-num: 'JetBrains Mono', 'SF Mono', Consolas, monospace;

  /* 스페이싱 */
  --sp-1: 2px;  --sp-2: 4px;  --sp-3: 8px;  --sp-4: 12px;
  --sp-5: 16px; --sp-6: 24px; --sp-7: 32px;

  --radius: 2px;
  --radius-chip: 3px;
}

/* ── 전역 ─────────────────────────────────── */
html, body, [class*="css"] {
  font-family: var(--font-ui);
  font-size: 13px;
  color: var(--text-primary);
}

.stApp, [data-testid="stAppViewContainer"] {
  background: var(--bg-base);
  color: var(--text-primary);
}

[data-testid="stHeader"] {
  background: transparent;
  height: 0;
}

/* 페이지 패딩 압축 */
.block-container {
  padding: 12px 20px 24px 20px !important;
  max-width: 1600px;
}

/* 헤딩 */
h1, h2, h3, h4 {
  font-family: var(--font-ui);
  color: var(--text-primary);
  font-weight: 600;
  letter-spacing: -0.01em;
}
h1 { font-size: 22px !important; }
h2 { font-size: 16px !important; }
h3 { font-size: 14px !important; }

/* 링크 */
a { color: var(--color-accent); text-decoration: none; }
a:hover { text-decoration: underline; }

/* 스크롤바 (선택) */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
  background: rgba(255,255,255,0.08);
  border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.16); }

/* ── 사이드바 ─────────────────────────────── */
[data-testid="stSidebar"] {
  background: var(--bg-elevated);
  border-right: 1px solid var(--border-subtle);
  width: 220px !important;
  min-width: 220px !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 8px; }

[data-testid="stSidebarNav"] {
  padding-top: 4px;
}
[data-testid="stSidebarNav"] ul { padding: 0; }
[data-testid="stSidebarNav"] li { margin: 0; }

[data-testid="stSidebarNav"] a {
  font-size: 12px !important;
  color: var(--text-secondary) !important;
  padding: 8px 12px 8px 14px !important;
  border-radius: 0;
  border-left: 2px solid transparent;
  transition: none;
}
[data-testid="stSidebarNav"] a:hover {
  background: var(--bg-hover);
  color: var(--text-primary) !important;
}
[data-testid="stSidebarNav"] a[aria-current="page"] {
  background: var(--bg-hover);
  border-left: 2px solid var(--color-accent);
  color: var(--text-primary) !important;
  font-weight: 500;
}

/* 사이드바 본문 텍스트 (커스텀 메타) */
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] label {
  font-size: 12px;
  color: var(--text-secondary);
}
[data-testid="stSidebar"] hr {
  border-color: var(--border-subtle);
  margin: 8px 0;
}

/* ── st.metric (fallback용 — 가능하면 metric_panel 사용) ── */
[data-testid="stMetric"] {
  background: var(--bg-panel);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius);
  padding: 10px 12px;
}
[data-testid="stMetricLabel"] {
  font-size: 10px !important;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-secondary) !important;
}
[data-testid="stMetricValue"] {
  font-family: var(--font-num) !important;
  font-size: 20px !important;
  font-weight: 600;
  color: var(--text-primary) !important;
  letter-spacing: -0.01em;
}
[data-testid="stMetricDelta"] {
  font-family: var(--font-num) !important;
  font-size: 11px !important;
}

/* ── 컨테이너(border=True) = Panel ────────── */
[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--bg-panel);
  border: 1px solid var(--border-base) !important;
  border-radius: var(--radius) !important;
}

/* 패널 컨테이너 자체는 패딩 0 */
[data-testid="stVerticalBlockBorderWrapper"] > div {
  padding: 0 !important;
}

/* 패널 본문 전체에 좌우/하단 패딩 */
[data-testid="stVerticalBlockBorderWrapper"] > div > div[data-testid="stVerticalBlock"] {
  padding: 12px 10px 12px 10px;
}

/* 헤더만 좌우로 -10px 당겨서 full-width로 (우리 클래스로 정확히 지정) */
[data-testid="stVerticalBlockBorderWrapper"] .as-panel-header {
  margin-left: -10px;
  margin-right: -10px;
}

/* ── 데이터프레임 ─────────────────────────── */
[data-testid="stDataFrame"] {
  background: var(--bg-panel);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius);
}
[data-testid="stDataFrame"] * {
  font-family: var(--font-num) !important;
  font-size: 12px !important;
}

/* ── 탭 ───────────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] {
  gap: 0;
  border-bottom: 1px solid var(--border-base);
}
[data-testid="stTabs"] [role="tab"] {
  font-size: 11px !important;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-secondary) !important;
  padding: 8px 14px !important;
  border-radius: 0 !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
  color: var(--text-primary) !important;
  background: transparent;
  border-bottom: 2px solid var(--color-accent) !important;
}
[data-testid="stTabs"] [role="tab"]:hover {
  color: var(--text-primary) !important;
}

/* ── 버튼 ─────────────────────────────────── */
.stButton > button {
  font-family: var(--font-ui);
  font-size: 12px;
  font-weight: 500;
  letter-spacing: 0.02em;
  border-radius: var(--radius);
  border: 1px solid var(--border-base);
  background: var(--bg-elevated);
  color: var(--text-primary);
  padding: 6px 14px;
  transition: none;
}
.stButton > button:hover {
  background: var(--bg-hover);
  border-color: var(--border-strong);
  color: var(--text-primary);
}
.stButton > button[kind="primary"] {
  background: var(--color-accent);
  color: var(--text-inverse);
  border-color: var(--color-accent);
}
.stButton > button[kind="primary"]:hover {
  background: #a5dcff;
  border-color: #a5dcff;
}

/* ── 입력 위젯 ────────────────────────────── */
.stSelectbox > div > div,
.stNumberInput > div > div,
.stTextInput > div > div {
  background: var(--bg-elevated) !important;
  border-color: var(--border-base) !important;
  border-radius: var(--radius) !important;
  font-size: 12px;
}
.stSlider [data-baseweb="slider"] div[role="slider"] {
  background: var(--color-accent);
}

/* ── Plotly 컨테이너 배경 투명 ─────────────── */
.js-plotly-plot .plotly,
.js-plotly-plot .plotly .main-svg {
  background: transparent !important;
}

/* ═══════════════════════════════════════════
   A안 커스텀 컴포넌트
   ═══════════════════════════════════════════ */

/* ── Metric Panel ─────────────────────────── */
.as-mp {
  background: var(--bg-panel);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius);
  padding: 10px 12px;
  min-height: 68px;
  box-sizing: border-box;
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 2px;
}
.as-mp--accent { border-left: 2px solid var(--color-accent); }
.as-mp--hero {
  padding: 16px 20px;
  min-height: 96px;
}
.as-mp--compact {
  padding: 6px 8px;
  min-height: 0;
}
.as-mp--empty {
  background: transparent;
  border-color: transparent;
  visibility: hidden;
}

.as-mp-label {
  font-family: var(--font-ui);
  font-size: 10px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-secondary);
  line-height: 1.2;
}
.as-mp-value {
  font-family: var(--font-num);
  font-size: 24px;
  font-weight: 600;
  letter-spacing: -0.01em;
  color: var(--text-primary);
  line-height: 1.15;
}
.as-mp--hero .as-mp-value   { font-size: 40px; font-weight: 600; }
.as-mp--compact .as-mp-value { font-size: 16px; }

.as-mp-delta {
  font-family: var(--font-num);
  font-size: 11px;
  color: var(--text-secondary);
  line-height: 1.2;
}
.as-mp-delta--up     { color: var(--color-up); }
.as-mp-delta--down   { color: var(--color-down); }
.as-mp-delta--warn   { color: var(--color-warn); }
.as-mp-delta--accent { color: var(--color-accent); }
.as-mp-delta--neutral{ color: var(--text-secondary); }

/* ── Section Header ───────────────────────── */
.as-section {
  display: flex;
  align-items: center;
  gap: 12px;
  margin: 16px 0 8px 0;
  color: var(--text-secondary);
  font-size: 11px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.1em;
}
.as-section::after {
  content: "";
  flex: 1;
  height: 1px;
  background: var(--border-base);
}

/* ── Panel Header (컨테이너 내부 상단) ────── */
.as-panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 12px;
  border-bottom: 1px solid var(--border-subtle);
  margin-bottom: 8px;
}
.as-panel-title {
  font-size: 11px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text-secondary);
}
.as-panel-status {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 11px;
  color: var(--text-secondary);
}

/* ── Status Dot ───────────────────────────── */
.as-dot {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--text-muted);
  vertical-align: middle;
}
.as-dot--up      { background: var(--color-up); }
.as-dot--down    { background: var(--color-down); }
.as-dot--warn    { background: var(--color-warn); }
.as-dot--neutral { background: var(--color-neutral); }
.as-dot--accent  { background: var(--color-accent); }
.as-dot--hollow  { background: transparent; border: 1px solid var(--text-muted); }

.as-status-line {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: var(--text-secondary);
  line-height: 1.6;
}
.as-status-line .as-status-val {
  font-family: var(--font-num);
  color: var(--text-primary);
}

/* ── Signal Chip ──────────────────────────── */
.as-chip {
  display: inline-block;
  font-family: var(--font-num);
  font-size: 11px;
  font-weight: 500;
  padding: 2px 8px;
  border-radius: var(--radius-chip);
  line-height: 1.5;
  white-space: nowrap;
}
.as-chip--up {
  background: rgba(74,222,128,0.15);
  color: #4ade80;
  border: 1px solid rgba(74,222,128,0.4);
}
.as-chip--down {
  background: rgba(255,92,92,0.15);
  color: #ff5c5c;
  border: 1px solid rgba(255,92,92,0.4);
}
.as-chip--neutral {
  background: rgba(154,160,166,0.15);
  color: #9aa0a6;
  border: 1px solid rgba(154,160,166,0.4);
}
.as-chip--warn {
  background: rgba(251,191,36,0.15);
  color: #fbbf24;
  border: 1px solid rgba(251,191,36,0.4);
}
.as-chip--accent {
  background: rgba(127,209,255,0.15);
  color: #7fd1ff;
  border: 1px solid rgba(127,209,255,0.4);
}

/* ── Top Strip ────────────────────────────── */
.as-topstrip {
  font-size: 11px;
  color: var(--text-secondary);
  letter-spacing: 0.02em;
  padding: 6px 0 10px 0;
  border-bottom: 1px solid var(--border-subtle);
  margin-bottom: 12px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}
.as-topstrip .as-ts-left {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}
.as-topstrip .as-ts-sep {
  color: var(--text-muted);
}
.as-topstrip .as-ts-right {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: var(--text-secondary);
}

/* ── Page title ───────────────────────────── */
.as-title {
  font-size: 22px;
  font-weight: 600;
  letter-spacing: -0.01em;
  color: var(--text-primary);
  margin: 0 0 6px 0;
}
.as-title-sub {
  font-size: 11px;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin: 0 0 8px 0;
}

/* ── Verdict list ─────────────────────────── */
.as-verdict {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 4px 0;
}
.as-verdict-row {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: var(--text-primary);
  line-height: 1.5;
}
.as-verdict-mark {
  font-family: var(--font-num);
  font-size: 12px;
  width: 14px;
  text-align: center;
  flex-shrink: 0;
}
.as-verdict-mark--pass { color: var(--color-up); }
.as-verdict-mark--fail { color: var(--color-down); }
.as-verdict-mark--warn { color: var(--color-warn); }
.as-verdict-mark--info { color: var(--text-secondary); }

/* Streamlit의 element-container 간격 압축 */
[data-testid="element-container"] { margin-bottom: 0; }
[data-testid="stVerticalBlock"] { gap: 8px; }

/* ── Responsive fallback (Phase 3) ────────── */
@media (max-width: 1200px) {
  .block-container { padding: 10px 14px 20px 14px !important; }
  .as-mp-value           { font-size: 20px; }
  .as-mp--hero .as-mp-value { font-size: 32px; }
  .as-title              { font-size: 20px; }
}

@media (max-width: 900px) {
  .block-container { padding: 8px 10px 16px 10px !important; }
  .as-mp           { min-height: 60px; padding: 8px 10px; }
  .as-mp-value     { font-size: 18px; }
  .as-mp--hero .as-mp-value { font-size: 28px; }
  .as-mp--hero     { min-height: 76px; padding: 12px 14px; }
  .as-title        { font-size: 18px; }
  .as-mp-label     { font-size: 9px; }
  .as-topstrip     { font-size: 10px; flex-wrap: wrap; }
}

@media (max-width: 640px) {
  [data-testid="stSidebar"] { width: 200px !important; min-width: 200px !important; }
  .as-mp-value     { font-size: 16px; }
  .as-mp--hero .as-mp-value { font-size: 24px; }
}
"""


def inject() -> None:
    """모든 페이지 최상단에서 1회 호출. CSS 주입."""
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# 2. 컴포넌트 헬퍼 (저수준 HTML)
# ─────────────────────────────────────────────────────────────

def _html(html: str) -> None:
    st.markdown(html, unsafe_allow_html=True)


def metric_panel(
    label: str,
    value: str,
    delta: str | None = None,
    tone: str = "neutral",
    variant: str = "",
    accent: bool = False,
) -> None:
    """
    A안 메트릭 패널.

    tone    : neutral | up | down | warn | accent
    variant : "" | "hero" | "compact"
    accent  : 좌측 2px accent line
    """
    cls = "as-mp"
    if variant:
        cls += f" as-mp--{variant}"
    if accent:
        cls += " as-mp--accent"

    delta_html = ""
    if delta:
        delta_html = f'<div class="as-mp-delta as-mp-delta--{tone}">{delta}</div>'

    _html(
        f'<div class="{cls}">'
        f'  <div class="as-mp-label">{label}</div>'
        f'  <div class="as-mp-value">{value}</div>'
        f'  {delta_html}'
        f'</div>'
    )


def metric_row(metrics: list[dict], cols: int = 4, fill_empty: bool = True) -> None:
    """
    메트릭 한 행. metrics 각 항목은 metric_panel의 kwargs.

    Example:
        theme.metric_row([
            dict(label="ACC",  value="58.4",  delta="+2.1", tone="up"),
            dict(label="BAL",  value="54.8"),
            ...
        ], cols=4)
    """
    columns = st.columns(cols, gap="small")
    for i, col in enumerate(columns):
        with col:
            if i < len(metrics):
                metric_panel(**metrics[i])
            elif fill_empty:
                _html('<div class="as-mp as-mp--empty"></div>')


def metric_grid(metrics: list[dict], cols: int = 4) -> None:
    """여러 행으로 metric 배치."""
    for i in range(0, len(metrics), cols):
        metric_row(metrics[i:i + cols], cols=cols)


def section_header(title: str) -> None:
    """'MODEL PERFORMANCE ─────' 스타일."""
    _html(f'<div class="as-section">{title}</div>')


def panel_header(title: str, status_text: str | None = None,
                 status_tone: str = "neutral") -> None:
    """
    st.container(border=True) 안쪽 첫 요소로 사용.
    우측에 상태 dot + 라벨 표시.
    """
    status_html = ""
    if status_text:
        status_html = (
            f'<span class="as-panel-status">'
            f'  <span class="as-dot as-dot--{status_tone}"></span>'
            f'  {status_text}'
            f'</span>'
        )
    _html(
        f'<div class="as-panel-header">'
        f'  <span class="as-panel-title">{title}</span>'
        f'  {status_html}'
        f'</div>'
    )


def status_dot(tone: str = "neutral") -> str:
    """dot HTML 문자열 리턴 (직접 조립용)."""
    return f'<span class="as-dot as-dot--{tone}"></span>'


def status_line(label: str, value: str, tone: str = "neutral") -> None:
    """'Data Ingestion    ● OK    KOSPI200 · HF' 한 줄."""
    _html(
        f'<div class="as-status-line">'
        f'  <span>{label}</span>'
        f'  <span class="as-status-val">{status_dot(tone)} {value}</span>'
        f'</div>'
    )


def signal_chip(signal: str, prob: float | None = None) -> str:
    """
    '<span class="as-chip as-chip--up">UP 72%</span>' 리턴.
    tone은 signal에서 자동 추론.
    """
    s = signal.upper()
    if s in ("UP", "BUY", "LONG"):
        cls, tone = "as-chip--up", "up"
    elif s in ("DOWN", "SELL", "SHORT"):
        cls, tone = "as-chip--down", "down"
    elif s in ("WARN", "CAUTION"):
        cls, tone = "as-chip--warn", "warn"
    else:
        cls, tone = "as-chip--neutral", "neutral"

    text = f"{s} {round(prob * 100)}%" if prob is not None else s
    return f'<span class="as-chip {cls}">{text}</span>'


def top_strip(parts: list[str], status_text: str = "LIVE",
              status_tone: str = "up") -> None:
    """
    페이지 타이틀 아래 컨텍스트 한 줄.
    Example:
        theme.top_strip(["KOSPI200","LGBM","F","2018-2024","COST 0.10%"])
    """
    left = '<span class="as-ts-sep">·</span>'.join(
        f'<span>{p}</span>' for p in parts
    )
    right = (
        f'<span class="as-dot as-dot--{status_tone}"></span>'
        f'<span>{status_text}</span>'
    )
    _html(
        f'<div class="as-topstrip">'
        f'  <div class="as-ts-left">{left}</div>'
        f'  <div class="as-ts-right">{right}</div>'
        f'</div>'
    )


def verdict_row(mark: str, text: str, tone: str = "info") -> None:
    """
    '✓ Prediction skill PASS' 스타일 한 줄.
    mark: '✓' | '✕' | '△' | '·' 등
    tone: pass | fail | warn | info
    """
    tone_map = {
        "pass": "as-verdict-mark--pass",
        "fail": "as-verdict-mark--fail",
        "warn": "as-verdict-mark--warn",
        "info": "as-verdict-mark--info",
    }
    cls = tone_map.get(tone, "as-verdict-mark--info")
    _html(
        f'<div class="as-verdict-row">'
        f'  <span class="as-verdict-mark {cls}">{mark}</span>'
        f'  <span>{text}</span>'
        f'</div>'
    )


def panel(title: str | None = None, status_text: str | None = None,
          status_tone: str = "neutral"):
    """
    컨텍스트 매니저. st.container(border=True) + 헤더.

    Example:
        with theme.panel("EQUITY CURVE", "LIVE", "up"):
            st.plotly_chart(fig, use_container_width=True)
    """
    from contextlib import contextmanager

    @contextmanager
    def _cm():
        container = st.container(border=True)
        with container:
            if title:
                panel_header(title, status_text, status_tone)
            yield container

    return _cm()


# ─────────────────────────────────────────────────────────────
# 3. 포맷 헬퍼
# ─────────────────────────────────────────────────────────────

def fmt_pct(x: float, digits: int = 1, signed: bool = False) -> str:
    if x is None:
        return "—"
    s = f"{x * 100:+.{digits}f}%" if signed else f"{x * 100:.{digits}f}%"
    return s


def fmt_num(x: float, digits: int = 2, signed: bool = False) -> str:
    if x is None:
        return "—"
    s = f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"
    return s


def tone_from_sign(x: float) -> str:
    if x is None:
        return "neutral"
    if x > 0:
        return "up"
    if x < 0:
        return "down"
    return "neutral"


# ─────────────────────────────────────────────────────────────
# 4. DataFrame Styler (숫자 우측 정렬 + 부호 색상 + 행 강조)
# ─────────────────────────────────────────────────────────────

def styled_df(
    df,
    precision: int = 2,
    num_cols: list[str] | None = None,
    pct_cols: list[str] | None = None,
    signed_color: bool = True,
    highlight_row=None,
    highlight_color: str = "rgba(74,222,128,0.15)",
):
    """
    A안 데이터프레임용 pandas Styler.

    num_cols      : 소수점 precision 자리, 부호 색상(mint/coral)
    pct_cols      : 0.584 → '+58.40%'
    signed_color  : False 면 색 안 입히고 정렬만
    highlight_row : 최고 성능 행 인덱스 → 배경 + 좌측 accent bar
    """
    import pandas as pd

    num_cols = num_cols or []
    pct_cols = pct_cols or []

    fmt = {}
    for c in num_cols:
        if c in df.columns:
            fmt[c] = (lambda p: lambda x: f"{x:.{p}f}"
                      if isinstance(x, (int, float)) and pd.notna(x) else "—")(precision)
    for c in pct_cols:
        if c in df.columns:
            fmt[c] = lambda x: (f"{x*100:+.2f}%"
                                if isinstance(x, (int, float)) and pd.notna(x) else "—")

    styler = df.style.format(fmt, na_rep="—")

    def _num_cell(v):
        base = ("font-family:'JetBrains Mono',monospace;"
                "font-size:12px;text-align:right;")
        if not isinstance(v, (int, float)) or pd.isna(v):
            return base + "color:#5c6470;"
        if signed_color:
            if v > 0:
                return base + "color:#4ade80;"
            if v < 0:
                return base + "color:#ff5c5c;"
        return base + "color:#e6e8ec;"

    for c in list(num_cols) + list(pct_cols):
        if c not in df.columns:
            continue
        try:
            styler = styler.map(_num_cell, subset=[c])
        except AttributeError:      # pandas < 2.1
            styler = styler.applymap(_num_cell, subset=[c])

    if highlight_row is not None:
        def _hl(row):
            if row.name == highlight_row:
                return [f"background:{highlight_color};"
                        f"border-left:2px solid #7fd1ff;"] * len(row)
            return [""] * len(row)
        styler = styler.apply(_hl, axis=1)

    return styler


# ─────────────────────────────────────────────────────────────
# 5. 자동화 헬퍼 (Phase 3)
# ─────────────────────────────────────────────────────────────

def auto_top_strip(
    c: dict,
    extras: list[str] | None = None,
    status_text: str | None = None,
    status_tone: str = "neutral",
) -> None:
    """
    ctx() dict에서 자동으로 top strip 구성.

    Example:
        theme.auto_top_strip(c, extras=[f"TEST {len(res.test_index)}d"],
                             status_text="LIVE", status_tone="up")
    """
    parts = []
    if c.get("dataset"):
        parts.append(str(c["dataset"]).upper())
    if c.get("model"):
        parts.append(f"MODEL {str(c['model']).upper()}")
    if c.get("feature_set"):
        parts.append(f"FEATURE {str(c['feature_set']).upper()}")
    if c.get("start") and c.get("end"):
        parts.append(f"{c['start']} – {c['end']}")
    if c.get("cost") is not None:
        try:
            parts.append(f"COST {float(c['cost']):.2%}")
        except Exception:
            pass
    if extras:
        parts.extend(extras)

    top_strip(
        parts,
        status_text=status_text or "LIVE",
        status_tone=status_tone,
    )


def sidebar_header(subtitle: str = "QUANT RESEARCH TERMINAL") -> None:
    """사이드바 최상단 브랜딩."""
    st.sidebar.markdown(
        f'<div style="padding:14px 14px 10px 14px;">'
        f'  <div style="font-family:var(--font-ui);font-size:14px;'
        f'              font-weight:600;letter-spacing:0.14em;'
        f'              color:var(--text-primary);line-height:1;">'
        f'    ALPHASTACK'
        f'  </div>'
        f'  <div style="font-family:var(--font-ui);font-size:9px;'
        f'              font-weight:500;letter-spacing:0.18em;'
        f'              color:var(--text-muted);margin-top:4px;">'
        f'    {subtitle}'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def sidebar_meta(rows: list[tuple[str, str]]) -> None:
    """
    사이드바 컨텍스트 메타. 상단에 1px 경계선 자동.

    Example:
        theme.sidebar_meta([
            ("DATASET", c["dataset"]),
            ("MODEL",   c["model"]),
            ("FEATURE", c["feature_set"]),
            ("PERIOD",  f"{c['start']} — {c['end']}"),
            ("COST",    f"{c['cost']:.2%}"),
        ])
    """
    html = ('<div style="padding:8px 14px 10px 14px;'
            'border-top:1px solid var(--border-subtle);">')
    for k, v in rows:
        html += (
            f'<div style="display:flex;justify-content:space-between;'
            f'align-items:baseline;padding:3px 0;gap:8px;">'
            f'  <span style="font-family:var(--font-ui);font-size:9px;'
            f'               font-weight:500;letter-spacing:0.1em;'
            f'               color:var(--text-muted);text-transform:uppercase;'
            f'               flex-shrink:0;">{k}</span>'
            f'  <span style="font-family:var(--font-num);font-size:11px;'
            f'               color:var(--text-primary);text-align:right;'
            f'               overflow:hidden;text-overflow:ellipsis;'
            f'               white-space:nowrap;">{v}</span>'
            f'</div>'
        )
    html += '</div>'
    st.sidebar.markdown(html, unsafe_allow_html=True)


def sidebar_section(label: str) -> None:
    """사이드바 소제목 — '───────── LABELS' 톤."""
    st.sidebar.markdown(
        f'<div style="padding:10px 14px 4px 14px;font-family:var(--font-ui);'
        f'            font-size:9px;font-weight:500;letter-spacing:0.14em;'
        f'            color:var(--text-muted);text-transform:uppercase;">'
        f'  {label}'
        f'</div>',
        unsafe_allow_html=True,
    )
