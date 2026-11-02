"""
Geospatial processing engine for GeoTIFF imagery, DEM elevation surfaces,
coordinate reference system (CRS) transformations, and grid-cell aggregation.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd

try:
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.enums import Resampling
    from rasterio.warp import calculate_default_transform, reproject
    import geopandas as gpd
    from shapely.geometry import box, Polygon
    HAS_GEOSPATIAL = True
except ImportError:
    HAS_GEOSPATIAL = False

import config
from utils.logging_utils import setup_logger

logger = setup_logger(__name__)


class GeospatialProcessor:
    """
    Handles GeoTIFF I/O, reprojection, spatial grid-cell tiling, and GeoJSON export.
    """

    def __init__(self):
        self.has_geospatial = HAS_GEOSPATIAL
        if not HAS_GEOSPATIAL:
            logger.warning("Geospatial packages (rasterio/geopandas) not yet fully loaded. Fallback modes active.")

    def read_geotiff(self, filepath: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reads a GeoTIFF raster file, returning the array (C, H, W) and metadata profile.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"GeoTIFF file not found: {filepath}")

        with rasterio.open(filepath) as src:
            data = src.read()
            profile = src.profile.copy()
            meta = {
                "crs": str(src.crs),
                "transform": src.transform,
                "bounds": src.bounds,
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "dtypes": src.dtypes,
                "nodata": src.nodata,
                "profile": profile
            }
        return data, meta

    def write_geotiff(
        self,
        filepath: Path,
        array: np.ndarray,
        profile: Optional[Dict[str, Any]] = None,
        crs: str = "EPSG:4326",
        transform: Optional[Any] = None
    ) -> None:
        """
        Saves a 2D or 3D numpy array as a valid GeoTIFF with CRS and Affine transform.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if array.ndim == 2:
            count = 1
            h, w = array.shape
            data = array[np.newaxis, :, :]
        elif array.ndim == 3:
            count, h, w = array.shape
            data = array
        else:
            raise ValueError(f"Array must be 2D or 3D, got ndim={array.ndim}")

        if profile is not None:
            out_profile = profile.copy()
            out_profile.update({
                "driver": "GTiff",
                "height": h,
                "width": w,
                "count": count,
                "dtype": str(data.dtype)
            })
        else:
            if transform is None:
                # Default 0.0001 deg resolution (~10m) near origin
                transform = from_origin(77.2090, 28.6139, 0.0001, 0.0001)

            out_profile = {
                "driver": "GTiff",
                "height": h,
                "width": w,
                "count": count,
                "dtype": str(data.dtype),
                "crs": crs,
                "transform": transform
            }

        with rasterio.open(filepath, "w", **out_profile) as dst:
            dst.write(data)
        logger.info("Successfully wrote GeoTIFF to: %s", filepath)

    def create_grid_cells(
        self,
        bounds: Tuple[float, float, float, float],
        cell_size_deg: float = 0.01,
        crs: str = "EPSG:4326"
    ) -> Any:
        """
        Generates regular geographic grid cells across spatial bounding box.
        Returns GeoPandas GeoDataFrame.
        """
        minx, miny, maxx, maxy = bounds
        # Use epsilon to prevent floating point boundary overshoot
        num_x = max(1, int(round((maxx - minx) / cell_size_deg)))
        num_y = max(1, int(round((maxy - miny) / cell_size_deg)))
        x_coords = [minx + i * cell_size_deg for i in range(num_x)]
        y_coords = [miny + j * cell_size_deg for j in range(num_y)]

        polygons = []
        cell_ids = []
        cid = 0

        for x in x_coords:
            for y in y_coords:
                poly = box(x, y, x + cell_size_deg, y + cell_size_deg)
                polygons.append(poly)
                cell_ids.append(f"grid_{cid:04d}")
                cid += 1

        gdf = gpd.GeoDataFrame({"cell_id": cell_ids, "geometry": polygons}, crs=crs)
        return gdf

    def aggregate_grid_features(
        self,
        grid_gdf: Any,
        landcover_mask: np.ndarray,
        lst_map: np.ndarray,
        elevation_map: Optional[np.ndarray] = None,
        humidity_val: float = 50.0,
        solar_val: float = 650.0,
        output_csv: Optional[Path] = None,
        output_geojson: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Aggregates zonal land cover distribution, mean LST anomaly, and environmental
        features for each geographic grid cell.
        Exports to CSV and GeoJSON.
        """
        h, w = landcover_mask.shape
        n_cells = len(grid_gdf)

        # Distribute cells across raster dimensions
        rows = int(np.ceil(np.sqrt(n_cells)))
        cols = int(np.ceil(n_cells / rows))
        step_h = h // rows
        step_w = w // cols

        records = []

        for idx, row in grid_gdf.iterrows():
            cid = row["cell_id"]
            cell_idx = int(cid.split("_")[1]) if "_" in cid else idx
            r = (cell_idx // cols) % rows
            c = cell_idx % cols

            r_start = r * step_h
            r_end = min((r + 1) * step_h, h)
            c_start = c * step_w
            c_end = min((c + 1) * step_w, w)

            sub_mask = landcover_mask[r_start:r_end, c_start:c_end]
            sub_lst = lst_map[r_start:r_end, c_start:c_end]

            total_px = max(1, sub_mask.size)
            imp_pct = (np.sum(sub_mask == 0) / total_px) * 100.0
            veg_pct = (np.sum(sub_mask == 1) / total_px) * 100.0
            soil_pct = (np.sum(sub_mask == 2) / total_px) * 100.0
            wat_pct = (np.sum(sub_mask == 3) / total_px) * 100.0
            mean_lst = float(np.mean(sub_lst))

            elev_val = float(np.mean(elevation_map[r_start:r_end, c_start:c_end])) if elevation_map is not None else 210.0

            records.append({
                "cell_id": cid,
                "impervious_pct": round(imp_pct, 2),
                "vegetation_pct": round(veg_pct, 2),
                "bare_soil_pct": round(soil_pct, 2),
                "water_pct": round(wat_pct, 2),
                "mean_lst_anomaly": round(mean_lst, 2),
                "elevation_m": round(elev_val, 1),
                "humidity_pct": round(humidity_val, 1),
                "solar_irradiance_wm2": round(solar_val, 1)
            })

        df = pd.DataFrame(records)

        # Attach attributes to GeoDataFrame
        merged_gdf = grid_gdf.merge(df, on="cell_id")

        if output_csv:
            csv_path = Path(output_csv)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, index=False)
            logger.info("Exported grid features CSV to: %s", csv_path)

        if output_geojson:
            geojson_path = Path(output_geojson)
            geojson_path.parent.mkdir(parents=True, exist_ok=True)
            merged_gdf.to_file(geojson_path, driver="GeoJSON")
            logger.info("Exported grid features GeoJSON to: %s", geojson_path)

        return df
