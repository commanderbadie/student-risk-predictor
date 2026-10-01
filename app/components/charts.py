"""
app/components/charts.py — Plotly chart factory functions for the Streamlit app.

All functions return a plotly Figure object ready to pass to st.plotly_chart().
Color palette stays consistent with the dark navy/teal theme.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Theme constants ───────────────────────────────────────────────────────────
BG       = "#0D1B2A"
SURFACE  = "#1A2E45"
TEAL     = "#00BCD4"
TEXT     = "#E0E0E0"
MUTED    = "#90A4AE"
RISK_COLORS = {"Low": "#4CAF50", "Medium": "#FF9800", "High": "#F44336"}

LAYOUT_BASE = dict(
    paper_bgcolor=BG,
    plot_bgcolor=SURFACE,
    font=dict(color=TEXT, family="sans-serif"),
    margin=dict(l=40, r=20, t=50, b=40),
)


def _axis_style(**kwargs):
    return dict(
        gridcolor="#263A52",
        zerolinecolor="#263A52",
        tickfont=dict(color=MUTED),
        title_font=dict(color=TEXT),
        **kwargs,
    )


# ── Probability gauge ─────────────────────────────────────────────────────────

def probability_gauge(probabilities: dict) -> go.Figure:
    """Three horizontal bar chart showing class probabilities."""
    classes = ["High", "Medium", "Low"]
    values  = [probabilities.get(c, 0) * 100 for c in classes]
    colors  = [RISK_COLORS[c] for c in classes]

    fig = go.Figure(go.Bar(
        x=values,
        y=classes,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="auto",
        textfont=dict(color=TEXT),
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Risk Probability", font=dict(color=TEXT)),
        xaxis=dict(range=[0, 100], title="Probability (%)", **_axis_style()),
        yaxis=_axis_style(),
        height=220,
    )
    return fig


# ── Feature importance bar ────────────────────────────────────────────────────

def feature_importance_chart(feature_names: list, importances: list, top_n: int = 12) -> go.Figure:
    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1])[-top_n:]
    names, vals = zip(*pairs)

    fig = go.Figure(go.Bar(
        x=list(vals),
        y=list(names),
        orientation="h",
        marker=dict(color=list(vals), colorscale=[[0, "#1A2E45"], [1, TEAL]]),
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=f"Top {top_n} Feature Importances", font=dict(color=TEXT)),
        xaxis=dict(title="Importance", **_axis_style()),
        yaxis=_axis_style(),
        height=380,
    )
    return fig


# ── Confusion matrix ──────────────────────────────────────────────────────────

def confusion_matrix_chart(cm: np.ndarray, labels: list) -> go.Figure:
    annotations = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            annotations.append(dict(
                x=j, y=i,
                text=str(cm[i, j]),
                showarrow=False,
                font=dict(color=TEXT, size=14),
            ))

    fig = go.Figure(go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale="Blues",
        showscale=True,
        zmin=0,
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        annotations=annotations,
        title=dict(text="Confusion Matrix", font=dict(color=TEXT)),
        xaxis=dict(title="Predicted", **_axis_style()),
        yaxis=dict(title="Actual", autorange="reversed", **_axis_style()),
        height=380,
    )
    return fig


# ── ROC curve ─────────────────────────────────────────────────────────────────

def roc_curve_chart(fpr_dict: dict, tpr_dict: dict, auc_dict: dict) -> go.Figure:
    """
    fpr_dict / tpr_dict / auc_dict: {class_name: array/float}
    """
    colors_map = {"High": "#F44336", "Medium": "#FF9800", "Low": "#4CAF50"}
    fig = go.Figure()
    for cls in fpr_dict:
        fig.add_trace(go.Scatter(
            x=fpr_dict[cls], y=tpr_dict[cls],
            mode="lines",
            name=f"{cls} (AUC={auc_dict[cls]:.2f})",
            line=dict(color=colors_map.get(cls, TEAL), width=2),
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(dash="dash", color=MUTED, width=1),
        showlegend=False,
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="ROC Curves (One-vs-Rest)", font=dict(color=TEXT)),
        xaxis=dict(title="False Positive Rate", range=[0, 1], **_axis_style()),
        yaxis=dict(title="True Positive Rate", range=[0, 1.02], **_axis_style()),
        legend=dict(bgcolor=SURFACE, font=dict(color=TEXT)),
        height=400,
    )
    return fig


# ── Distribution plot ─────────────────────────────────────────────────────────

def distribution_chart(df: pd.DataFrame, col: str, color_by: str | None = None) -> go.Figure:
    if color_by and color_by in df.columns:
        fig = px.histogram(
            df, x=col, color=color_by,
            color_discrete_map=RISK_COLORS,
            barmode="overlay",
            opacity=0.75,
            nbins=30,
        )
    else:
        fig = px.histogram(df, x=col, nbins=30, color_discrete_sequence=[TEAL])

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=f"Distribution: {col}", font=dict(color=TEXT)),
        xaxis=_axis_style(title=col),
        yaxis=_axis_style(title="Count"),
        legend=dict(bgcolor=SURFACE, font=dict(color=TEXT)),
        height=320,
    )
    return fig


# ── Correlation heatmap ───────────────────────────────────────────────────────

def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    corr = df[numeric_cols].corr()

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale="RdBu",
        zmin=-1, zmax=1,
        colorbar=dict(tickfont=dict(color=TEXT)),
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Feature Correlation Heatmap", font=dict(color=TEXT)),
        xaxis=dict(tickangle=-45, tickfont=dict(color=MUTED, size=10)),
        yaxis=dict(autorange="reversed", tickfont=dict(color=MUTED, size=10)),
        height=520,
    )
    return fig


# ── Risk breakdown bar ────────────────────────────────────────────────────────

def risk_breakdown_chart(df: pd.DataFrame, group_col: str) -> go.Figure:
    counts = df.groupby([group_col, "risk_level"]).size().reset_index(name="count")
    fig = px.bar(
        counts, x=group_col, y="count", color="risk_level",
        color_discrete_map=RISK_COLORS,
        barmode="group",
    )
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=f"Risk Level by {group_col}", font=dict(color=TEXT)),
        xaxis=_axis_style(title=group_col),
        yaxis=_axis_style(title="Count"),
        legend=dict(bgcolor=SURFACE, font=dict(color=TEXT), title_text="Risk"),
        height=340,
    )
    return fig


# ── Donut chart ───────────────────────────────────────────────────────────────

def risk_donut(counts: dict) -> go.Figure:
    labels = list(counts.keys())
    values = list(counts.values())
    colors = [RISK_COLORS.get(l, TEAL) for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color=BG, width=2)),
        textfont=dict(color=TEXT),
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Risk Distribution", font=dict(color=TEXT)),
        legend=dict(bgcolor=SURFACE, font=dict(color=TEXT)),
        height=320,
    )
    return fig


# ── Scatter: actual vs predicted GPA ─────────────────────────────────────────

def actual_vs_predicted(y_true, y_pred) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred,
        mode="markers",
        marker=dict(color=TEAL, opacity=0.55, size=6),
        name="Predictions",
    ))
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx],
        mode="lines",
        line=dict(color="#FF7043", dash="dash", width=1.5),
        name="Perfect fit",
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Actual vs Predicted GPA", font=dict(color=TEXT)),
        xaxis=dict(title="Actual GPA", **_axis_style()),
        yaxis=dict(title="Predicted GPA", **_axis_style()),
        height=380,
    )
    return fig
