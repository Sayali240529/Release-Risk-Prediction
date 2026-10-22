"""Plotly gauge: Green 0-35%, Yellow 35-70%, Red 70-100%."""
import plotly.graph_objects as go

try:
    from model.policy import LOW_THRESHOLD, HIGH_THRESHOLD
except ImportError:
    LOW_THRESHOLD, HIGH_THRESHOLD = 0.35, 0.70


def risk_gauge(score: float) -> go.Figure:
    pct = round(score * 100, 1)
    lo, hi = LOW_THRESHOLD * 100, HIGH_THRESHOLD * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 48}},
        title={"text": "Release Risk Score"},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar": {"color": "#222222", "thickness": 0.25},
            "steps": [
                {"range": [0, lo], "color": "#2ecc71"},
                {"range": [lo, hi], "color": "#f1c40f"},
                {"range": [hi, 100], "color": "#e74c3c"},
            ],
            "threshold": {"line": {"color": "black", "width": 4}, "thickness": 0.8, "value": pct},
        },
    ))
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=60, b=10))
    return fig
