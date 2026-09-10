import pandas as pd
import plotly.graph_objects as go

_LAYOUT = dict(
    template="plotly_dark",
    margin=dict(l=10, r=10, t=36, b=10),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=340,
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
)


def price_line(df: pd.DataFrame, title: str = "Price") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["close"], name="Close",
                             line=dict(width=1.4, color="#7fd1ff")))
    fig.update_layout(title=title, **_LAYOUT)
    return fig


def equity_curve(series_map: dict, title: str = "Equity") -> go.Figure:
    fig = go.Figure()
    for name, s in series_map.items():
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=name, line=dict(width=1.6)))
    fig.update_layout(title=title, **_LAYOUT)
    return fig


def drawdown_area(dd: pd.Series, title: str = "Drawdown") -> go.Figure:
    fig = go.Figure(go.Scatter(x=dd.index, y=dd.values, fill="tozeroy",
                               line=dict(color="#ff6b6b", width=1.2), name="DD"))
    fig.update_layout(title=title, **_LAYOUT)
    return fig


def proba_stack(df: pd.DataFrame, title: str = "Prediction Probability") -> go.Figure:
    fig = go.Figure()
    for col, color in [("P(Down)", "#ff6b6b"), ("P(Neutral)", "#9aa0a6"), ("P(Up)", "#4ade80")]:
        if col in df:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col,
                                     stackgroup="one", line=dict(width=0.4, color=color)))
    fig.update_layout(title=title, **_LAYOUT)
    return fig


def bar(series: pd.Series, title: str = "", color: str = "#7fd1ff") -> go.Figure:
    fig = go.Figure(go.Bar(x=series.index, y=series.values, marker_color=color))
    fig.update_layout(title=title, **_LAYOUT)
    return fig


def confusion_heatmap(cm: pd.DataFrame, title: str = "Confusion Matrix") -> go.Figure:
    fig = go.Figure(go.Heatmap(z=cm.values, x=cm.columns, y=cm.index,
                               colorscale="Blues", showscale=False,
                               text=cm.values, texttemplate="%{text}"))
    fig.update_layout(title=title, **_LAYOUT)
    return fig
