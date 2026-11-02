"""
Data preprocessing, tensor transformations, and feature normalization routines.
"""

from typing import Dict, Tuple, Union
import numpy as np
import torch
import config


def normalize_satellite_image(
    image: np.ndarray,
    scale_type: str = "minmax"
) -> np.ndarray:
    """
    Normalizes multi-spectral or RGB satellite image array to [0.0, 1.0].
    Handles float and uint8/uint16 imagery cleanly.

    Args:
        image: np.ndarray of shape (C, H, W) or (H, W, C).
        scale_type: 'minmax' or 'standardize'.
    Returns:
        Normalized np.ndarray of float32.
    """
    img_float = image.astype(np.float32)

    if scale_type == "minmax":
        # Check if already 0-1
        max_val = np.nanmax(img_float)
        min_val = np.nanmin(img_float)
        if max_val > min_val:
            normalized = (img_float - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(img_float)
        return np.clip(normalized, 0.0, 1.0)

    elif scale_type == "standardize":
        mean = np.nanmean(img_float)
        std = np.nanstd(img_float)
        if std > 1e-6:
            return (img_float - mean) / std
        return img_float - mean

    return img_float


def compute_landcover_percentages(
    mask: Union[np.ndarray, torch.Tensor],
    num_classes: int = config.NUM_CLASSES
) -> Dict[str, float]:
    """
    Computes exact percentage of each land cover class from a 2D segmentation mask.

    Classes:
        0: Impervious Surface
        1: Vegetation
        2: Bare Soil
        3: Water

    Returns:
        Dict with keys ['impervious_pct', 'vegetation_pct', 'bare_soil_pct', 'water_pct']
        Percentages sum to ~100.0%.
    """
    if isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = np.asarray(mask)

    total_pixels = mask_np.size
    if total_pixels == 0:
        return {
            "impervious_pct": 0.0,
            "vegetation_pct": 0.0,
            "bare_soil_pct": 0.0,
            "water_pct": 0.0,
        }

    percentages = {}
    key_mapping = {
        0: "impervious_pct",
        1: "vegetation_pct",
        2: "bare_soil_pct",
        3: "water_pct",
    }

    for class_id in range(num_classes):
        count = np.sum(mask_np == class_id)
        pct = (float(count) / float(total_pixels)) * 100.0
        percentages[key_mapping.get(class_id, f"class_{class_id}_pct")] = round(pct, 3)

    return percentages


def assemble_feature_vector(
    landcover_pcts: Dict[str, float],
    elevation_m: float,
    humidity_pct: float,
    solar_irradiance_wm2: float
) -> np.ndarray:
    """
    Assembles a 7-dimensional feature array in the strictly defined order:
    [impervious_pct, vegetation_pct, bare_soil_pct, water_pct, elevation_m, humidity_pct, solar_irradiance_wm2]

    Returns:
        1D np.ndarray of shape (7,) with float32.
    """
    features = [
        float(landcover_pcts.get("impervious_pct", 0.0)),
        float(landcover_pcts.get("vegetation_pct", 0.0)),
        float(landcover_pcts.get("bare_soil_pct", 0.0)),
        float(landcover_pcts.get("water_pct", 0.0)),
        float(elevation_m),
        float(humidity_pct),
        float(solar_irradiance_wm2)
    ]
    return np.array(features, dtype=np.float32)
