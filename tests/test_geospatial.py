"""
Unit tests for geospatial preprocessing, patch extraction, and raster alignment.
"""

import pytest
import numpy as np
from data_pipeline.patch_extractor import PatchExtractor
from data_pipeline.raster_aligner import RasterAligner
from data_pipeline.geospatial_processor import GeospatialProcessor


def test_patch_extraction_and_reconstruction_2d():
    """Verify 2D raster tiling and seamless reconstruction."""
    h, w = 512, 512
    arr = np.random.randint(0, 4, size=(h, w)).astype(np.float32)

    extractor = PatchExtractor(patch_size=256, stride=256)
    patches, coords, orig_shape = extractor.extract_patches(arr)

    assert patches.shape == (4, 256, 256)
    assert len(coords) == 4
    assert orig_shape == (h, w)

    reconstructed = extractor.reconstruct_from_patches(patches, coords, orig_shape)
    assert reconstructed.shape == (h, w)
    assert np.allclose(arr, reconstructed)


def test_categorical_resampling_integrity():
    """Verify that categorical resampling preserves integer class IDs without creating fractional classes."""
    aligner = RasterAligner()
    # 2D discrete mask with class IDs [0, 1, 2, 3]
    mask = np.random.choice([0, 1, 2, 3], size=(128, 128)).astype(np.int64)

    resampled = aligner.resample_array(mask, target_shape=(256, 256), is_categorical=True)

    assert resampled.shape == (256, 256)
    unique_classes = np.unique(resampled)
    for c in unique_classes:
        assert c in [0, 1, 2, 3], f"Unexpected interpolated class found: {c}"


def test_grid_cell_generation():
    """Verify GeoPandas bounding box grid creation."""
    proc = GeospatialProcessor()
    bounds = (77.0, 28.0, 77.04, 28.04)  # 0.04 deg box -> 4x4 = 16 cells of 0.01 deg
    gdf = proc.create_grid_cells(bounds, cell_size_deg=0.01)

    assert len(gdf) == 16
    assert "cell_id" in gdf.columns
    assert "geometry" in gdf.columns
