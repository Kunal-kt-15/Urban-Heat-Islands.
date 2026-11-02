"""
Prediction utilities for standalone U-Net segmentation and LST Regression inference.
Supports numpy arrays, PyTorch tensors, and batch processing.
"""

from typing import Union, Tuple, Dict, Any, Optional
import numpy as np
import torch
import config
from models.unet import UNet
from models.regression_ann import LSTRegressionANN
from utils.device import get_device
from utils.preprocessing import normalize_satellite_image, compute_landcover_percentages


def predict_single_image(
    image: np.ndarray,
    model: UNet,
    device: Optional[torch.device] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Runs segmentation inference on a single satellite image (3, H, W) or (H, W, 3).

    Returns:
        pred_mask: 2D np.ndarray of shape (H, W) with class IDs (0..3)
        probabilities: 3D np.ndarray of shape (num_classes, H, W)
        percentages: Dict of land-cover class percentages
    """
    if device is None:
        device = get_device()

    model.eval()
    model.to(device)

    # Format shape to (3, H, W)
    if image.ndim == 3 and image.shape[-1] == 3:
        image = np.transpose(image, (2, 0, 1))

    # Normalize
    norm_img = normalize_satellite_image(image)
    tensor_img = torch.from_numpy(norm_img).float().unsqueeze(0).to(device)

    with torch.no_grad():
        probs = model.predict_proba(tensor_img).squeeze(0).cpu().numpy()
        pred_mask = np.argmax(probs, axis=0)

    percentages = compute_landcover_percentages(pred_mask)
    return pred_mask, probs, percentages


def run_batch_inference(
    images: np.ndarray,
    model: UNet,
    batch_size: int = 8,
    device: Optional[torch.device] = None
) -> np.ndarray:
    """
    Runs batch segmentation inference across an array of images (N, C, H, W).
    Returns predicted masks array (N, H, W).
    """
    if device is None:
        device = get_device()

    model.eval()
    model.to(device)

    n_samples = len(images)
    pred_masks = []

    with torch.no_grad():
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch = images[start_idx:end_idx]
            batch_tensor = torch.from_numpy(batch).float().to(device)
            logits = model(batch_tensor)
            batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
            pred_masks.append(batch_preds)

    return np.concatenate(pred_masks, axis=0)
