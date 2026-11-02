"""
Segmentation Map Colorization and Satellite Overlay Utilities.
Enforces consistent scientific color coding across all visualizations:
- Impervious Surface: Red (#E63946)
- Vegetation: Green (#2A9D8F)
- Bare Soil: Yellow/Gold (#E9C46A)
- Water: Deep Blue (#264653)
"""

from typing import Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import config


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Maps 2D class indices (0..3) to an RGB image array (H, W, 3) in uint8 [0..255].
    """
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, rgb in config.CLASS_RGB_COLORS.items():
        matching = (mask == class_id)
        colored[matching] = rgb

    return colored


def create_segmentation_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.45
) -> np.ndarray:
    """
    Blends colorized segmentation mask onto optical satellite image.

    Args:
        image: Satellite image array (3, H, W) or (H, W, 3) in [0..1] or uint8.
        mask: 2D class mask (H, W).
        alpha: Transparency factor for segmentation overlay (0.0: image only, 1.0: mask only).

    Returns:
        Blended RGB uint8 image of shape (H, W, 3).
    """
    # Normalize image to uint8 (H, W, 3)
    if image.ndim == 3 and image.shape[0] == 3:
        img_hwc = np.transpose(image, (1, 2, 0))
    else:
        img_hwc = image.copy()

    if img_hwc.dtype != np.uint8:
        if np.nanmax(img_hwc) <= 1.0:
            img_uint8 = (np.clip(img_hwc, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            img_uint8 = np.clip(img_hwc, 0, 255).astype(np.uint8)
    else:
        img_uint8 = img_hwc

    color_mask = colorize_mask(mask)
    blended = (1.0 - alpha) * img_uint8.astype(np.float32) + alpha * color_mask.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def create_segmentation_legend_figure() -> plt.Figure:
    """
    Generates a standalone Matplotlib legend figure for dashboard display.
    """
    fig, ax = plt.subplots(figsize=(8, 0.8), facecolor="none")
    ax.axis("off")

    patches = [
        mpatches.Patch(color=hex_c, label=f"{name}")
        for cid, (name, hex_c) in enumerate(zip(config.CLASS_NAMES.values(), config.CLASS_HEX_COLORS.values()))
    ]

    ax.legend(
        handles=patches,
        loc="center",
        ncol=4,
        frameon=True,
        facecolor="#1E1E1E",
        edgecolor="#444444",
        fontsize=10,
        labelcolor="white"
    )
    plt.tight_layout()
    return fig
