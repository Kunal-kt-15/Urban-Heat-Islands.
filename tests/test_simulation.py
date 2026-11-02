"""
Unit tests for UHI mitigation simulation and land-cover calculations.
"""

import pytest
import numpy as np
import torch

from models.regression_ann import LSTRegressionANN
from simulation.mitigation_engine import simulate_mitigation
from utils.preprocessing import compute_landcover_percentages


def test_landcover_percentages_sum_to_100():
    """Verify that calculated land cover percentages sum to approximately 100%."""
    # Create mask with known class counts
    mask = np.zeros((100, 100), dtype=np.int64)
    mask[:40, :] = 0  # 40% impervious
    mask[40:70, :] = 1  # 30% veg
    mask[70:90, :] = 2  # 20% soil
    mask[90:, :] = 3  # 10% water

    pcts = compute_landcover_percentages(mask)

    total_sum = sum(pcts.values())
    assert abs(total_sum - 100.0) < 0.01
    assert pcts["impervious_pct"] == 40.0
    assert pcts["vegetation_pct"] == 30.0
    assert pcts["bare_soil_pct"] == 20.0
    assert pcts["water_pct"] == 10.0


def test_simulation_converts_only_impervious_pixels():
    """Verify that simulate_mitigation NEVER modifies non-impervious pixels."""
    h, w = 100, 100
    mask = np.zeros((h, w), dtype=np.int64)
    # Impervious = 50%, Veg = 20%, Soil = 20%, Water = 10%
    mask[:50, :] = 0
    mask[50:70, :] = 1
    mask[70:90, :] = 2
    mask[90:, :] = 3

    original_non_urban = (mask != 0)
    original_non_urban_values = mask[original_non_urban].copy()

    meta = {"elevation_m": 200.0, "humidity_pct": 50.0, "solar_irradiance_wm2": 600.0}
    reg_model = LSTRegressionANN(in_features=7, hidden_layers=[16])

    # Convert 30% of impervious
    res = simulate_mitigation(
        mask=mask,
        metadata=meta,
        conversion_percentage=0.30,
        regression_model=reg_model,
        contiguous=True
    )

    modified_mask = res["modified_mask"]

    # Check non-impervious pixels were untouched
    assert np.array_equal(modified_mask[original_non_urban], original_non_urban_values)

    # Check that converted pixels were strictly changed from 0 to 1
    orig_urban = (mask == 0)
    new_urban_values = modified_mask[orig_urban]
    # Each previously urban pixel is either still 0 or now 1
    for val in np.unique(new_urban_values):
        assert val in [0, 1], f"Impervious pixel was converted to unexpected class: {val}"


def test_simulation_conversion_count():
    """Verify conversion percentage is relative to impervious count, not total image size."""
    h, w = 100, 100  # 10,000 total pixels
    mask = np.ones((h, w), dtype=np.int64)  # Start with all vegetation (class 1)
    mask[:20, :] = 0  # Exactly 2,000 impervious pixels

    meta = {"elevation_m": 200.0, "humidity_pct": 50.0, "solar_irradiance_wm2": 600.0}
    reg_model = LSTRegressionANN(in_features=7, hidden_layers=[16])

    # 25% conversion of 2,000 impervious pixels = 500 pixels
    res = simulate_mitigation(
        mask=mask,
        metadata=meta,
        conversion_percentage=0.25,
        regression_model=reg_model,
        contiguous=False
    )

    assert res["pixels_converted"] == 500
    assert np.sum(res["modified_mask"] == 0) == 1500
    assert np.sum(res["modified_mask"] == 1) == 8500
