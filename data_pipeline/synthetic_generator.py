"""
Synthetic Geospatial and Satellite Data Generation Engine.
Generates realistic multi-class land-cover patches, multi-spectral satellite imagery,
environmental metadata, and scientifically coherent Land Surface Temperature (LST) anomaly targets.
"""

from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import scipy.ndimage as ndimage
import config
from utils.logging_utils import setup_logger

logger = setup_logger(__name__)


class SyntheticDataGenerator:
    """
    Engine to synthesize realistic paired satellite imagery and LST datasets.
    """

    def __init__(self, seed: int = config.RANDOM_SEED):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def _generate_structured_mask(self, h: int = 256, w: int = 256) -> np.ndarray:
        """
        Creates a spatially contiguous 4-class mask using smoothed multi-field random thresholds:
        0: Impervious, 1: Vegetation, 2: Bare Soil, 3: Water.
        """
        # Generate smooth continuous random spatial fields via Gaussian filtering
        noise_urban = self.rng.standard_normal((h, w))
        field_urban = ndimage.gaussian_filter(noise_urban, sigma=12.0)

        noise_veg = self.rng.standard_normal((h, w))
        field_veg = ndimage.gaussian_filter(noise_veg, sigma=14.0)

        noise_soil = self.rng.standard_normal((h, w))
        field_soil = ndimage.gaussian_filter(noise_soil, sigma=18.0)

        # Connected water feature (e.g. river corridor or waterbody)
        water_noise = self.rng.standard_normal((h, w))
        water_field = ndimage.gaussian_filter(water_noise, sigma=25.0)

        # Road grid pattern overlay for urban realism
        grid = np.zeros((h, w), dtype=bool)
        step = self.rng.integers(30, 60)
        grid[::step, :] = True
        grid[:, ::step] = True
        # Thicken roads slightly
        grid = ndimage.binary_dilation(grid, iterations=1)

        mask = np.zeros((h, w), dtype=np.int64)

        # Multi-class competitive assignment
        stack = np.stack([field_urban, field_veg, field_soil], axis=-1)
        base_classes = np.argmax(stack, axis=-1)  # 0: Urban, 1: Veg, 2: Soil
        mask = base_classes

        # Impose water bodies where water_field exceeds high threshold (e.g. 15-20% chance of river)
        water_threshold = np.percentile(water_field, 88)
        mask[water_field > water_threshold] = 3  # Water

        # Impose urban road network over terrestrial classes
        mask[grid & (mask != 3)] = 0  # Impervious roads

        return mask

    def _synthesize_rgb_image(self, mask: np.ndarray) -> np.ndarray:
        """
        Synthesizes plausible high-resolution satellite imagery (3, H, W) conditioned on mask.
        Each class exhibits distinct spectral reflectance with realistic intra-class texture.
        """
        h, w = mask.shape
        img = np.zeros((3, h, w), dtype=np.float32)

        # Impervious surface: Grey, concrete, asphalt tones with mild grain
        urban_mask = (mask == 0)
        img[0, urban_mask] = self.rng.uniform(0.40, 0.65, size=np.sum(urban_mask))
        img[1, urban_mask] = self.rng.uniform(0.40, 0.65, size=np.sum(urban_mask))
        img[2, urban_mask] = self.rng.uniform(0.45, 0.70, size=np.sum(urban_mask))

        # Vegetation: Lush chlorophyll reflectance (lower R/B, prominent G)
        veg_mask = (mask == 1)
        img[0, veg_mask] = self.rng.uniform(0.08, 0.22, size=np.sum(veg_mask))
        img[1, veg_mask] = self.rng.uniform(0.45, 0.75, size=np.sum(veg_mask))
        img[2, veg_mask] = self.rng.uniform(0.08, 0.25, size=np.sum(veg_mask))

        # Bare Soil: Tan, brownish-yellow tones
        soil_mask = (mask == 2)
        img[0, soil_mask] = self.rng.uniform(0.55, 0.75, size=np.sum(soil_mask))
        img[1, soil_mask] = self.rng.uniform(0.45, 0.62, size=np.sum(soil_mask))
        img[2, soil_mask] = self.rng.uniform(0.25, 0.40, size=np.sum(soil_mask))

        # Water: Deep navy / blue / dark cyan
        water_mask = (mask == 3)
        img[0, water_mask] = self.rng.uniform(0.05, 0.15, size=np.sum(water_mask))
        img[1, water_mask] = self.rng.uniform(0.20, 0.40, size=np.sum(water_mask))
        img[2, water_mask] = self.rng.uniform(0.45, 0.75, size=np.sum(water_mask))

        # Add spatial sensor noise & blur for natural optical realism
        for c in range(3):
            noise = self.rng.normal(0, 0.02, (h, w)).astype(np.float32)
            img[c] = np.clip(img[c] + noise, 0.0, 1.0)
            img[c] = ndimage.gaussian_filter(img[c], sigma=0.8)

        return img

    def _compute_lst_anomaly(
        self,
        impervious_pct: float,
        vegetation_pct: float,
        bare_soil_pct: float,
        water_pct: float,
        elevation: float,
        humidity: float,
        solar_irradiance: float
    ) -> float:
        """
        Physics-grounded Land Surface Temperature (LST) anomaly model.
        UHI effect: Impervious surfaces elevate LST; vegetation & water provide cooling;
        higher elevation reduces temp (lapse rate); higher humidity reduces sensible heat flux;
        higher solar irradiance elevates surface heating.
        """
        # Baseline temperature anomaly at regional mean
        base_anomaly = 1.5

        # Land cover contributions
        uhi_effect = 0.085 * impervious_pct
        vegetation_cooling = -0.065 * vegetation_pct
        water_cooling = -0.050 * water_pct
        soil_effect = 0.020 * bare_soil_pct

        # Environmental factors
        elevation_cooling = -0.0055 * (elevation - 200.0)  # ~5.5°C per 1000m
        humidity_cooling = -0.035 * (humidity - 50.0)
        solar_heating = 0.0045 * (solar_irradiance - 600.0)

        # Uncorrelated measurement/microclimate noise
        noise = self.rng.normal(0.0, 0.35)

        lst_anomaly = (
            base_anomaly
            + uhi_effect
            + vegetation_cooling
            + water_cooling
            + soil_effect
            + elevation_cooling
            + humidity_cooling
            + solar_heating
            + noise
        )

        return float(np.clip(lst_anomaly, config.LST_ANOMALY_RANGE[0], config.LST_ANOMALY_RANGE[1]))

    def generate_dataset(
        self,
        num_samples: int = 500,
        output_dir: Optional[Path] = None,
        save_samples: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Generates full synthetic dataset of paired images, masks, metadata, and LST targets.

        Returns:
            dict containing:
                'images': shape (num_samples, 3, 256, 256) float32
                'masks': shape (num_samples, 256, 256) int64
                'metadata': shape (num_samples, 3) float32
                'lst_targets': shape (num_samples, 1) float32
                'features': shape (num_samples, 7) float32 (all 7 regression features)
        """
        logger.info("Generating %d synthetic UHI samples...", num_samples)

        images = np.zeros((num_samples, 3, 256, 256), dtype=np.float32)
        masks = np.zeros((num_samples, 256, 256), dtype=np.int64)
        metadata = np.zeros((num_samples, 3), dtype=np.float32)
        lst_targets = np.zeros((num_samples, 1), dtype=np.float32)
        features = np.zeros((num_samples, 7), dtype=np.float32)

        for i in range(num_samples):
            # 1. Spatial mask
            mask = self._generate_structured_mask(256, 256)
            masks[i] = mask

            # 2. RGB Image
            img = self._synthesize_rgb_image(mask)
            images[i] = img

            # 3. Calculate Land-Cover Percentages
            total_px = 256 * 256
            imp_pct = (np.sum(mask == 0) / total_px) * 100.0
            veg_pct = (np.sum(mask == 1) / total_px) * 100.0
            soil_pct = (np.sum(mask == 2) / total_px) * 100.0
            wat_pct = (np.sum(mask == 3) / total_px) * 100.0

            # 4. Environmental metadata
            elevation = float(self.rng.uniform(30.0, 650.0))
            humidity = float(self.rng.uniform(20.0, 85.0))
            solar_irradiance = float(self.rng.uniform(350.0, 1050.0))

            metadata[i] = [elevation, humidity, solar_irradiance]

            # 5. LST Anomaly target
            lst_val = self._compute_lst_anomaly(
                imp_pct, veg_pct, soil_pct, wat_pct,
                elevation, humidity, solar_irradiance
            )
            lst_targets[i, 0] = lst_val

            # 6. Combined 7-feature vector
            features[i] = [
                imp_pct, veg_pct, soil_pct, wat_pct,
                elevation, humidity, solar_irradiance
            ]

        dataset = {
            "images": images,
            "masks": masks,
            "metadata": metadata,
            "lst_targets": lst_targets,
            "features": features
        }

        # Save to disk
        out_dir = Path(output_dir) if output_dir else config.SYNTHETIC_DATA_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        npz_path = out_dir / "synthetic_uhi_dataset.npz"
        np.savez_compressed(npz_path, **dataset)
        logger.info("Saved synthetic dataset to: %s", npz_path)

        if save_samples:
            self._save_sample_rasters(images[:5], masks[:5], metadata[:5], lst_targets[:5])

        return dataset

    def _save_sample_rasters(
        self,
        sample_images: np.ndarray,
        sample_masks: np.ndarray,
        sample_metadata: np.ndarray,
        sample_lst: np.ndarray
    ) -> None:
        """
        Saves individual sample files in data/samples/ for fast interactive UI loading.
        """
        samples_dir = config.SAMPLES_DIR
        samples_dir.mkdir(parents=True, exist_ok=True)

        for idx in range(len(sample_images)):
            sample_path = samples_dir / f"sample_area_{idx + 1}.npz"
            np.savez_compressed(
                sample_path,
                image=sample_images[idx],
                mask=sample_masks[idx],
                metadata=sample_metadata[idx],
                lst=sample_lst[idx]
            )
        logger.info("Exported %d sample demonstration files to: %s", len(sample_images), samples_dir)


if __name__ == "__main__":
    generator = SyntheticDataGenerator()
    generator.generate_dataset(num_samples=500)
