"""
Urban Heat Island (UHI) Mitigation Simulation Engine.
Simulates targeted urban planning interventions by converting impervious surfaces into vegetation,
recalculating land-cover distributions, and estimating model-predicted temperature reductions.
"""

from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import scipy.ndimage as ndimage
import torch

import config
from models.regression_ann import LSTRegressionANN
from utils.preprocessing import compute_landcover_percentages, assemble_feature_vector
from utils.logging_utils import setup_logger

logger = setup_logger(__name__)


def simulate_mitigation(
    mask: np.ndarray,
    metadata: Dict[str, float],
    conversion_percentage: float,
    regression_model: LSTRegressionANN,
    scaler: Optional[Any] = None,
    contiguous: bool = True,
    seed: int = config.RANDOM_SEED
) -> Dict[str, Any]:
    """
    Simulates urban greening by converting a specific fraction of impervious pixels into vegetation.

    Args:
        mask: 2D numpy array (H, W) of original land-cover classes (0..3).
        metadata: Dict containing 'elevation_m', 'humidity_pct', 'solar_irradiance_wm2'.
        conversion_percentage: Fraction of IMPERVIOUS pixels to convert (e.g. 0.20 for 20%).
        regression_model: Trained LSTRegressionANN model.
        scaler: StandardScaler used during regression training.
        contiguous: If True, selects realistic contiguous green corridors/parks; otherwise uniform random.
        seed: Random seed for reproducibility.

    Returns:
        Structured dictionary with before/after statistics and temperature reduction.
    """
    if not (0.0 <= conversion_percentage <= 1.0):
        raise ValueError(f"conversion_percentage must be between 0.0 and 1.0, got {conversion_percentage}")

    modified_mask = mask.copy()
    h, w = mask.shape
    total_pixels = h * w

    # 1. Identify Impervious Pixels (Class 0)
    impervious_indices = np.argwhere(mask == 0)
    total_impervious_pixels = len(impervious_indices)

    # 2. Number of pixels to convert (relative to impervious total, NOT entire image)
    num_to_convert = int(round(total_impervious_pixels * conversion_percentage))

    if total_impervious_pixels > 0 and num_to_convert > 0:
        rng = np.random.default_rng(seed)

        if contiguous:
            # Contiguous Intervention Strategy:
            # Grow greening clusters outward from selected seed impervious points
            urban_binary = (mask == 0).astype(np.float32)
            # Distance transform to locate dense urban core / cluster seeds
            dist = ndimage.distance_transform_edt(urban_binary)
            noise = rng.uniform(0.0, 1.0, size=(h, w))
            # Combined score prioritizes dense impervious zones with randomized patchiness
            priority = dist + 1.5 * ndimage.gaussian_filter(noise, sigma=6.0)
            priority[mask != 0] = -np.inf  # STRICTLY preserve non-impervious pixels

            flat_priority = priority.flatten()
            # Top-k impervious pixels with highest priority
            convert_flat_indices = np.argpartition(flat_priority, -num_to_convert)[-num_to_convert:]
            np.put(modified_mask, convert_flat_indices, 1)  # Class 1: Vegetation

        else:
            # Uniform Random Selection
            selected_choice = rng.choice(total_impervious_pixels, size=num_to_convert, replace=False)
            selected_coords = impervious_indices[selected_choice]
            for r, c in selected_coords:
                modified_mask[r, c] = 1  # Vegetation

    # 3. Verify Integrity: Ensure NO pixels outside Class 0 were modified
    non_impervious_original = (mask != 0)
    assert np.array_equal(mask[non_impervious_original], modified_mask[non_impervious_original]), (
        "Integrity check failed: Non-impervious pixels were modified during simulation!"
    )

    # 4. Land Cover Distributions Before and After
    orig_pcts = compute_landcover_percentages(mask)
    mod_pcts = compute_landcover_percentages(modified_mask)

    elev = float(metadata.get("elevation_m", 200.0))
    hum = float(metadata.get("humidity_pct", 50.0))
    solar = float(metadata.get("solar_irradiance_wm2", 650.0))

    # 5. Regression Prediction Before
    orig_raw = assemble_feature_vector(orig_pcts, elev, hum, solar).reshape(1, -1)
    orig_scaled = scaler.transform(orig_raw) if scaler is not None else orig_raw

    # 6. Regression Prediction After
    mod_raw = assemble_feature_vector(mod_pcts, elev, hum, solar).reshape(1, -1)
    mod_scaled = scaler.transform(mod_raw) if scaler is not None else mod_raw

    regression_model.eval()
    with torch.no_grad():
        device = next(regression_model.parameters()).device
        t_orig = torch.from_numpy(orig_scaled).float().to(device)
        t_mod = torch.from_numpy(mod_scaled).float().to(device)

        orig_lst = float(regression_model(t_orig).item())
        mod_lst = float(regression_model(t_mod).item())

    # Predicted Temperature Reduction = Original - Modified
    temp_reduction = orig_lst - mod_lst

    return {
        "conversion_percentage": conversion_percentage,
        "conversion_pct_display": f"{int(conversion_percentage * 100)}%",
        "pixels_converted": num_to_convert,
        "total_impervious_pixels": total_impervious_pixels,
        "original_mask": mask,
        "modified_mask": modified_mask,
        "original_landcover_pcts": orig_pcts,
        "modified_landcover_pcts": mod_pcts,
        "original_lst_anomaly": round(orig_lst, 3),
        "modified_lst_anomaly": round(mod_lst, 3),
        "predicted_temperature_reduction": round(temp_reduction, 3),
        "contiguous_mode": contiguous
    }


class MitigationSimulator:
    """
    Object-oriented wrapper around the mitigation simulation workflow.
    """

    def __init__(self, regression_model: LSTRegressionANN, scaler: Optional[Any] = None):
        self.regression_model = regression_model
        self.scaler = scaler

    def run(
        self,
        mask: np.ndarray,
        metadata: Dict[str, float],
        conversion_pct: float,
        contiguous: bool = True
    ) -> Dict[str, Any]:
        return simulate_mitigation(
            mask=mask,
            metadata=metadata,
            conversion_percentage=conversion_pct,
            regression_model=self.regression_model,
            scaler=self.scaler,
            contiguous=contiguous
        )
