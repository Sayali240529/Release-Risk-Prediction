"""Horizontal bar chart of SHAP values for the current release."""
import plotly.graph_objects as go

try:
    from model.explain import readable
except ImportError:
    def readable(name):
        return name.replace("_", " ").capitalize()


def shap_bar_chart(result: dict, top_n: int = 8) -> go.Figure:
    """Uses result['shap_all'] if present, else falls back to result['top_3_factors']."""
    if result.get("shap_all"):
        items = [(readable(k), v) for k, v in result["shap_all"].items()]
    else:
        items = list(result.get("top_3_factors", []))

    items = sorted(items, key=lambda t: abs(t[1]), reverse=True)[:top_n]
    items.reverse()  # biggest bar at the top
    labels = [i[0] for i in items]
    vals = [i[1] for i in items]

    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker_color=["#e74c3c" if v > 0 else "#2ecc71" for v in vals],
        text=[f"{v:+.2f}" for v in vals], textposition="outside",
    ))
    fig.update_layout(
        title="What is driving this score? (red = raises risk, green = lowers it)",
        xaxis_title="SHAP value (log-odds impact)",
        height=max(260, 55 * len(items) + 120),
        margin=dict(l=10, r=30, t=60, b=40),
    )
    return fig
