"""
Geospatial Raster Alignment and Resampling Engine.
Ensures identical CRS, spatial extent, and pixel grid resolution across multi-sensor rasters
(e.g., aligning high-res RGB/NIR with coarser Landsat thermal or DEM elevation products).
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
import scipy.ndimage as ndimage

try:
    import rasterio
    from rasterio.warp import reproject, Resampling, calculate_default_transform
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

import config
from utils.logging_utils import setup_logger

logger = setup_logger(__name__)


class RasterAligner:
    """
    Harmonizes spatial rasters through coordinate reprojection and explicit categorical/continuous resampling.
    """

    def __init__(self):
        self.has_rasterio = HAS_RASTERIO

    def resample_array(
        self,
        array: np.ndarray,
        target_shape: Tuple[int, int],
        is_categorical: bool = False
    ) -> np.ndarray:
        """
        Resamples a 2D or 3D numpy array to target (H, W).
        Uses nearest-neighbor interpolation for categorical masks (preserving integer class IDs),
        and bilinear/spline interpolation for continuous optical or thermal surfaces.
        """
        orig_shape = array.shape[-2:]
        if orig_shape == target_shape:
            return array

        zoom_factors = (target_shape[0] / orig_shape[0], target_shape[1] / orig_shape[1])

        if array.ndim == 2:
            order = 0 if is_categorical else 1
            resampled = ndimage.zoom(array, zoom_factors, order=order)
            return resampled.astype(array.dtype)
        elif array.ndim == 3:
            c = array.shape[0]
            out = np.zeros((c, target_shape[0], target_shape[1]), dtype=array.dtype)
            order = 0 if is_categorical else 1
            for i in range(c):
                out[i] = ndimage.zoom(array[i], zoom_factors, order=order)
            return out
        else:
            raise ValueError(f"Array must be 2D or 3D, got ndim={array.ndim}")

    def align_geotiffs(
        self,
        source_path: Path,
        reference_path: Path,
        output_path: Path,
        is_categorical: bool = False
    ) -> None:
        """
        Reprojects and matches source_path GeoTIFF to exact CRS, transform,
        and pixel grid dimensions of reference_path GeoTIFF.
        """
        if not HAS_RASTERIO:
            raise RuntimeError("Rasterio is required for GeoTIFF file reprojection.")

        source_path = Path(source_path)
        reference_path = Path(reference_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        resampling_mode = Resampling.nearest if is_categorical else Resampling.bilinear

        with rasterio.open(reference_path) as ref:
            dst_crs = ref.crs
            dst_transform = ref.transform
            dst_width = ref.width
            dst_height = ref.height
            dst_meta = ref.meta.copy()

        with rasterio.open(source_path) as src:
            dst_meta.update({
                "count": src.count,
                "dtype": src.dtypes[0]
            })

            with rasterio.open(output_path, "w", **dst_meta) as dst:
                for band in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, band),
                        destination=rasterio.band(dst, band),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        resampling=resampling_mode
                    )

        logger.info("Aligned %s with reference %s -> %s", source_path.name, reference_path.name, output_path.name)
