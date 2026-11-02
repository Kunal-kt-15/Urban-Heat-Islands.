"""
Interactive Plotly Charts for Streamlit Dashboard.
Includes land-cover donuts, scenario comparison bars, training convergence curves,
confusion matrices, and actual vs predicted regression scatter plots.
"""

from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config


def plot_landcover_distribution_chart(percentages: Dict[str, float]) -> go.Figure:
    """
    Creates donut chart displaying the four land-cover class percentages.
    """
    labels = [config.CLASS_NAMES[0], config.CLASS_NAMES[1], config.CLASS_NAMES[2], config.CLASS_NAMES[3]]
    values = [
        percentages.get("impervious_pct", 0.0),
        percentages.get("vegetation_pct", 0.0),
        percentages.get("bare_soil_pct", 0.0),
        percentages.get("water_pct", 0.0)
    ]
    colors = [config.CLASS_HEX_COLORS[0], config.CLASS_HEX_COLORS[1], config.CLASS_HEX_COLORS[2], config.CLASS_HEX_COLORS[3]]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker=dict(colors=colors, line=dict(color="#1E1E1E", width=2)),
                textinfo="label+percent",
                hoverinfo="label+value+percent",
                textfont=dict(size=12, color="white")
            )
        ]
    )
    fig.update_layout(
        title=dict(text="Land-Cover Composition", font=dict(size=16, color="white")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="white"), orientation="h", yanchor="bottom", y=-0.2),
        margin=dict(l=20, r=20, t=40, b=40),
        height=320
    )
    return fig


def plot_mitigation_scenarios_chart(df_scenarios: pd.DataFrame) -> go.Figure:
    """
    Creates dual-axis chart showing Temperature Reduction (°C) and Land-Cover Shift
    across intervention levels (10% -> 50%).
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Predicted Temperature Reduction (°C)", "Land-Cover Shift (% Area)"),
        specs=[[{"type": "xy"}, {"type": "xy"}]]
    )

    # 1. Cooling curve
    fig.add_trace(
        go.Bar(
            x=df_scenarios["Scenario"],
            y=df_scenarios["Predicted Cooling (°C)"],
            name="Cooling Reduction (°C)",
            marker=dict(color="#2A9D8F", line=dict(color="#264653", width=1.5)),
            text=df_scenarios["Predicted Cooling (°C)"].apply(lambda x: f"-{x:.2f} °C"),
            textposition="outside"
        ),
        row=1, col=1
    )

    # 2. Shift in Impervious vs Vegetation
    fig.add_trace(
        go.Scatter(
            x=df_scenarios["Scenario"],
            y=df_scenarios["Modified Impervious %"],
            mode="lines+markers",
            name="Impervious %",
            line=dict(color="#E63946", width=3)
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=df_scenarios["Scenario"],
            y=df_scenarios["Modified Vegetation %"],
            mode="lines+markers",
            name="Vegetation %",
            line=dict(color="#2A9D8F", width=3)
        ),
        row=1, col=2
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,24,33,0.6)",
        font=dict(color="white"),
        height=360,
        margin=dict(l=30, r=30, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25)
    )
    fig.update_yaxes(title_text="Predicted Reduction (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Percentage (%)", row=1, col=2)
    return fig


def plot_training_history_charts(
    unet_history: Dict[str, Any],
    reg_history: Dict[str, Any]
) -> Tuple[go.Figure, go.Figure]:
    """
    Creates training progress charts for U-Net (Loss, mIoU) and Regression ANN (Loss, RMSE).
    """
    # U-Net Chart
    epochs_u = list(range(1, len(unet_history.get("train_loss", [])) + 1))
    fig_u = make_subplots(rows=1, cols=2, subplot_titles=("U-Net Loss Convergence", "Validation mIoU"))

    if epochs_u:
        fig_u.add_trace(
            go.Scatter(x=epochs_u, y=unet_history["train_loss"], name="Train Loss", line=dict(color="#E76F51", width=2)),
            row=1, col=1
        )
        fig_u.add_trace(
            go.Scatter(x=epochs_u, y=unet_history["val_loss"], name="Val Loss", line=dict(color="#F4A261", dash="dot")),
            row=1, col=1
        )
        fig_u.add_trace(
            go.Scatter(x=epochs_u, y=unet_history["val_miou"], name="Val mIoU", line=dict(color="#2A9D8F", width=2)),
            row=1, col=2
        )

    fig_u.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,24,33,0.6)",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    # Regression Chart
    epochs_r = list(range(1, len(reg_history.get("train_loss", [])) + 1))
    fig_r = make_subplots(rows=1, cols=2, subplot_titles=("Regression Loss (MSE)", "Validation RMSE (°C)"))

    if epochs_r:
        fig_r.add_trace(
            go.Scatter(x=epochs_r, y=reg_history["train_loss"], name="Train Loss", line=dict(color="#457B9D", width=2)),
            row=1, col=1
        )
        fig_r.add_trace(
            go.Scatter(x=epochs_r, y=reg_history["val_loss"], name="Val Loss", line=dict(color="#A8DADC", dash="dot")),
            row=1, col=1
        )
        fig_r.add_trace(
            go.Scatter(x=epochs_r, y=reg_history["val_rmse"], name="Val RMSE", line=dict(color="#E63946", width=2)),
            row=1, col=2
        )

    fig_r.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,24,33,0.6)",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig_u, fig_r


def plot_confusion_matrix_chart(cm: np.ndarray) -> go.Figure:
    """
    Renders normalized confusion matrix heatmap for semantic segmentation.
    """
    class_labels = [config.CLASS_NAMES[i] for i in range(config.NUM_CLASSES)]
    # Row normalize to percentages
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    norm_cm = (cm / row_sums) * 100.0

    fig = go.Figure(
        data=go.Heatmap(
            z=norm_cm,
            x=class_labels,
            y=class_labels,
            text=[[f"{norm_cm[r, c]:.1f}%<br>({cm[r, c]})" for c in range(len(class_labels))] for r in range(len(class_labels))],
            texttemplate="%{text}",
            colorscale="Viridis",
            colorbar=dict(title="Accuracy %")
        )
    )
    fig.update_layout(
        title=dict(text="Segmentation Confusion Matrix (Normalized)", font=dict(color="white")),
        xaxis=dict(title="Predicted Class", tickfont=dict(color="white")),
        yaxis=dict(title="Actual Class", tickfont=dict(color="white"), autorange="reversed"),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=380,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig


def plot_actual_vs_predicted_scatter(y_actual: np.ndarray, y_predicted: np.ndarray) -> go.Figure:
    """
    Scatter plot of actual vs predicted LST anomaly with identity 1:1 diagonal.
    """
    actual_flat = y_actual.flatten()
    pred_flat = y_predicted.flatten()

    fig = go.Figure()

    # Scatter points
    fig.add_trace(
        go.Scatter(
            x=actual_flat,
            y=pred_flat,
            mode="markers",
            name="Samples",
            marker=dict(color="#457B9D", size=7, opacity=0.75, line=dict(color="#1D3557", width=1))
        )
    )

    # 1:1 Diagonal line
    min_val = min(float(np.min(actual_flat)), float(np.min(pred_flat))) - 0.5
    max_val = max(float(np.max(actual_flat)), float(np.max(pred_flat))) + 0.5

    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Perfect Prediction (1:1)",
            line=dict(color="#E63946", dash="dash", width=2)
        )
    )

    fig.update_layout(
        title=dict(text="Actual vs. Predicted LST Anomaly", font=dict(color="white")),
        xaxis=dict(title="Actual LST Anomaly (°C)", range=[min_val, max_val]),
        yaxis=dict(title="Predicted LST Anomaly (°C)", range=[min_val, max_val]),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,24,33,0.6)",
        height=380,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig
