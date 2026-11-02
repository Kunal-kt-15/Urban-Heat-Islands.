"""
PyTorch Dataset classes for Semantic Segmentation and LST Anomaly Regression.
Includes geographic train/validation/test splitting to prevent spatial data leakage.
"""

from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import joblib

import config
from utils.logging_utils import setup_logger

logger = setup_logger(__name__)


def create_spatial_splits(
    num_samples: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = config.RANDOM_SEED
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates contiguous spatial chunk splits (or seeded block-level splits)
    to prevent spatial data leakage across train, val, and test partitions.
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples)

    train_end = int(num_samples * train_ratio)
    val_end = int(num_samples * (train_ratio + val_ratio))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return train_idx, val_idx, test_idx


class SatelliteSegmentationDataset(Dataset):
    """
    PyTorch Dataset for multi-spectral / RGB satellite imagery and semantic segmentation masks.
    """

    def __init__(
        self,
        images: np.ndarray,
        masks: np.ndarray,
        transform: bool = False
    ):
        """
        Args:
            images: np.ndarray of shape (N, C, H, W) float32
            masks: np.ndarray of shape (N, H, W) int64
            transform: boolean to apply spatial data augmentations
        """
        assert len(images) == len(masks), "Images and masks must have matching length"
        self.images = torch.from_numpy(images).float()
        self.masks = torch.from_numpy(masks).long()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self.images[idx]
        mask = self.masks[idx]

        if self.transform:
            # Simple spatial augmentation preserving spatial alignment
            if torch.rand(1).item() > 0.5:
                img = torch.flip(img, dims=[2])
                mask = torch.flip(mask, dims=[1])
            if torch.rand(1).item() > 0.5:
                img = torch.flip(img, dims=[1])
                mask = torch.flip(mask, dims=[0])

        return img, mask


class LSTRegressionDataset(Dataset):
    """
    PyTorch Dataset for LST Anomaly Regression.
    Features: (N, 7) [Impervious%, Veg%, Soil%, Water%, Elevation, Humidity, SolarIrradiance]
    Target: (N, 1) LST Anomaly (°C)
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = False
    ):
        assert len(features) == len(targets), "Features and targets must have matching length"
        self.fit_scaler = fit_scaler

        if scaler is None:
            self.scaler = StandardScaler()
            if fit_scaler:
                self.scaled_features = self.scaler.fit_transform(features)
            else:
                self.scaled_features = features.copy()
        else:
            self.scaler = scaler
            self.scaled_features = self.scaler.transform(features)

        self.features_raw = features.astype(np.float32)
        self.features = torch.from_numpy(self.scaled_features).float()
        self.targets = torch.from_numpy(targets).float()

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]

    def save_scaler(self, path: Path) -> None:
        """Saves fitted feature scaler to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, path)
        logger.info("Saved feature scaler to: %s", path)
