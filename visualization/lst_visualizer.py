"""
Thermal Anomaly Heatmap and Spatial Temperature Differential Visualization.
Renders continuous Land Surface Temperature (LST) anomaly fields and mitigation delta maps.
"""

from typing import Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.ndimage as ndimage

import config


def render_lst_heatmap(
    lst_array: np.ndarray,
    title: str = "Predicted Land Surface Temperature (LST) Anomaly",
    colormap: str = "inferno",
    vmin: float = -2.0,
    vmax: float = 10.0
) -> plt.Figure:
    """
    Renders high-contrast spatial heatmap of LST anomaly distribution.
    """
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#0E1117")
    ax.set_facecolor("#0E1117")

    im = ax.imshow(lst_array, cmap=colormap, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("LST Anomaly (°C)", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=10)
    ax.axis("off")
    plt.tight_layout()
    return fig


def render_lst_difference_map(
    orig_lst: np.ndarray,
    mod_lst: np.ndarray,
    title: str = "Predicted Temperature Reduction (°C)"
) -> plt.Figure:
    """
    Renders cooling delta map (Original - Modified).
    Positive values (greens/blues) represent cooling, zeros (neutral grey).
    """
    diff = orig_lst - mod_lst

    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#0E1117")
    ax.set_facecolor("#0E1117")

    max_delta = max(0.5, float(np.nanmax(np.abs(diff))))
    im = ax.imshow(diff, cmap="YlGn_r", vmin=0.0, vmax=max_delta)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Estimated Cooling (°C)", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=10)
    ax.axis("off")
    plt.tight_layout()
    return fig
