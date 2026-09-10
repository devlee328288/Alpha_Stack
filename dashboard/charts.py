import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# ═══════════════════════════════════════════════════════════
# A안 Terminal Plotly 템플릿 등록 (전역)
# ═══════════════════════════════════════════════════════════
TERMINAL_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        font=dict(
            family="JetBrains Mono, Inter, sans-serif",
            size=11,
            color="#e6e8ec",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=28, b=8),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.06)",
            zerolinecolor="rgba(255,255,255,0.09)",
            linecolor="rgba(255,255,255,0.09)",
            tickfont=dict(size=10, color="#9aa0a6"),
            title=dict(font=dict(size=10, color="#9aa0a6")),
            showline=True,
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.06)",
            zerolinecolor="rgba(255,255,255,0.09)",
            linecolor="rgba(255,255,255,0.09)",
            tickfont=dict(size=10, color="#9aa0a6"),
            title=dict(font=dict(size=10, color="#9aa0a6")),
        ),
        legend=dict(
            orientation="h",
            y=1.08, x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10, color="#9aa0a6"),
        ),
        colorway=["#7fd1ff", "#4ade80", "#9aa0a6", "#fbbf24",
                  "#a78bfa", "#ff5c5c"],
        hoverlabel=dict(
            bgcolor="#161a22",
            bordercolor="rgba(255,255,255,0.16)",
            font=dict(
                family="JetBrains Mono, Inter, sans-serif",
                size=11,
                color="#e6e8ec",
            ),
        ),
    )
)
pio.templates["terminal"] = TERMINAL_TEMPLATE
pio.templates.default = "terminal"


# ═══════════════════════════════════════════════════════════
# _LAYOUT — 터미널 템플릿 위에 덧씌울 최소값만
# (template / bg / axis 그리드는 템플릿이 처리)
# ═══════════════════════════════════════════════════════════
_LAYOUT = dict(
    margin=dict(l=10, r=10, t=36, b=10),
    height=340,
)


def price_line(df: pd.DataFrame, title: str = "Price") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["close"], name="Close",
        line=dict(width=1.4, color="#7fd1ff"),
    ))
    fig.update_layout(title=title, **_LAYOUT)
    return fig


def equity_curve(series_map: dict, title: str = "Equity") -> go.Figure:
    fig = go.Figure()
    for name, s in series_map.items():
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, name=name,
            line=dict(width=1.6),
        ))
    fig.update_layout(title=title, **_LAYOUT)
    return fig


def drawdown_area(dd: pd.Series, title: str = "Drawdown") -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=dd.index, y=dd.values,
        fill="tozeroy",
        line=dict(color="#ff5c5c", width=1.2),
        fillcolor="rgba(255,92,92,0.15)",
        name="DD",
    ))
    fig.update_layout(title=title, **_LAYOUT)
    return fig


def proba_stack(df: pd.DataFrame,
                title: str = "Prediction Probability") -> go.Figure:
    fig = go.Figure()
    for col, color, fill in [
        ("P(Down)",    "#ff5c5c", "rgba(255,92,92,0.35)"),
        ("P(Neutral)", "#9aa0a6", "rgba(154,160,166,0.35)"),
        ("P(Up)",      "#4ade80", "rgba(74,222,128,0.35)"),
    ]:
        if col in df:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col], name=col,
                stackgroup="one",
                line=dict(width=0.4, color=color),
                fillcolor=fill,
            ))
    fig.update_layout(title=title, **_LAYOUT)
    return fig


def bar(series: pd.Series, title: str = "",
        color: str = "#7fd1ff") -> go.Figure:
    fig = go.Figure(go.Bar(
        x=series.index, y=series.values,
        marker_color=color,
    ))
    fig.update_layout(title=title, **_LAYOUT)
    return fig


def confusion_heatmap(cm: pd.DataFrame,
                      title: str = "Confusion Matrix") -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=cm.values,
        x=cm.columns,
        y=cm.index,
        colorscale=[
            [0.0, "#0e1420"],
            [0.4, "#1e4a6e"],
            [1.0, "#7fd1ff"],
        ],
        showscale=False,
        text=cm.values,
        texttemplate="%{text}",
        textfont=dict(size=13, color="#e6e8ec",
                      family="JetBrains Mono, monospace"),
        hovertemplate="actual=%{y}, pred=%{x}<br>count=%{z}<extra></extra>",
    ))
    fig.update_layout(title=title, **_LAYOUT)
    return fig
