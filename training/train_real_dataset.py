"""
GPU Training Pipeline for Real Satellite Imagery (Copernicus Sentinel-2 & Landsat).
Trains U-Net Semantic Segmentation and Deep ANN LST Regression on real earth observation data
using NVIDIA CUDA acceleration.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from utils.device import get_device, get_device_info
from models.unet import UNet
from models.regression_ann import LSTRegressionANN
from data_pipeline.real_data_ingestor import RealDataIngestor
from data_pipeline.dataset import SatelliteSegmentationDataset, LSTRegressionDataset, create_spatial_splits
from training.metrics import compute_segmentation_metrics, compute_regression_metrics
from utils.checkpoint_utils import save_checkpoint, save_json_history
from utils.logging_utils import setup_logger

logger = setup_logger(__name__)


def train_on_real_satellite_data(
    epochs_unet: int = 15,
    epochs_regression: int = 35,
    batch_size: int = 8,
    lr_unet: float = 2e-4,
    lr_regression: float = 1e-3
) -> Dict[str, Any]:
    """
    Executes full GPU training on real Copernicus Sentinel-2 satellite dataset.
    """
    device_info = get_device_info()
    logger.info("==========================================================")
    logger.info("REAL SATELLITE DATASET GPU TRAINING")
    logger.info("Hardware: %s | CUDA Available: %s", device_info.get("current_device"), device_info.get("cuda_available"))
    logger.info("==========================================================")

    device = get_device()

    # 1. Ingest Real Sentinel-2 Data
    geotiff_path = config.RAW_DATA_DIR / "sentinel2_sample.tif"
    if not geotiff_path.exists():
        raise FileNotFoundError(f"Real satellite GeoTIFF not found at: {geotiff_path}")

    ingestor = RealDataIngestor()
    # Stride 64 gives rich overlapping patches across the real scene
    real_data = ingestor.process_geotiff(geotiff_path, patch_size=256, stride=64)

    images = real_data["images"]
    masks = real_data["masks"]
    metadata = real_data["metadata"]
    lst_targets = real_data["lst_targets"]
    features = real_data["features"]

    # Export representative real patches to data/samples/ for the web app
    samples_dir = config.SAMPLES_DIR
    for idx in range(min(5, len(images))):
        sample_path = samples_dir / f"sample_real_{idx + 1}.npz"
        np.savez_compressed(
            sample_path,
            image=images[idx],
            mask=masks[idx],
            metadata=metadata[idx],
            lst=lst_targets[idx]
        )
    logger.info("Exported %d real demonstration samples to %s", min(5, len(images)), samples_dir)

    num_samples = len(images)
    logger.info("Total Real Satellite Patches: %d", num_samples)

    # 2. Geographic Train / Val / Test Partitioning
    train_idx, val_idx, test_idx = create_spatial_splits(num_samples, train_ratio=0.70, val_ratio=0.15)
    logger.info("Partitions: Train=%d, Val=%d, Test=%d", len(train_idx), len(val_idx), len(test_idx))

    # Datasets with Augmentation for Real Imagery
    train_seg_ds = SatelliteSegmentationDataset(images[train_idx], masks[train_idx], transform=True)
    val_seg_ds = SatelliteSegmentationDataset(images[val_idx], masks[val_idx], transform=False)
    test_seg_ds = SatelliteSegmentationDataset(images[test_idx], masks[test_idx], transform=False)

    train_seg_loader = DataLoader(train_seg_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_seg_loader = DataLoader(val_seg_ds, batch_size=batch_size, shuffle=False)
    test_seg_loader = DataLoader(test_seg_ds, batch_size=batch_size, shuffle=False)

    # =========================================================================
    # STAGE 1: TRAIN U-NET ON REAL SATELLITE IMAGERY (GPU)
    # =========================================================================
    logger.info("--- [STAGE 1] Training U-Net on Real Sentinel-2 Imagery (GPU) ---")
    unet = UNet(in_channels=3, num_classes=4).to(device)
    criterion_seg = nn.CrossEntropyLoss()
    optimizer_seg = torch.optim.AdamW(unet.parameters(), lr=lr_unet, weight_decay=1e-4)
    scheduler_seg = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_seg, mode="max", factor=0.5, patience=3)

    best_val_miou = 0.0
    unet_ckpt_path = config.CHECKPOINTS_DIR / "real_best_unet_model.pt"

    start_time_unet = time.time()

    for epoch in range(1, epochs_unet + 1):
        unet.train()
        running_train_loss = 0.0

        for b_imgs, b_masks in train_seg_loader:
            b_imgs = b_imgs.to(device)
            b_masks = b_masks.to(device)

            optimizer_seg.zero_grad()
            logits = unet(b_imgs)
            loss = criterion_seg(logits, b_masks)
            loss.backward()
            optimizer_seg.step()

            running_train_loss += loss.item() * b_imgs.size(0)

        epoch_train_loss = running_train_loss / len(train_seg_ds)

        # Validation
        unet.eval()
        running_val_loss = 0.0
        val_preds_list = []
        val_targets_list = []

        with torch.no_grad():
            for b_imgs, b_masks in val_seg_loader:
                b_imgs = b_imgs.to(device)
                b_masks = b_masks.to(device)

                logits = unet(b_imgs)
                loss = criterion_seg(logits, b_masks)
                running_val_loss += loss.item() * b_imgs.size(0)

                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds_list.append(preds)
                val_targets_list.append(b_masks.cpu().numpy())

        epoch_val_loss = running_val_loss / len(val_seg_ds)
        val_preds = np.concatenate(val_preds_list, axis=0)
        val_targets = np.concatenate(val_targets_list, axis=0)

        metrics = compute_segmentation_metrics(val_targets, val_preds, num_classes=4)
        val_miou = metrics.miou
        scheduler_seg.step(val_miou)

        logger.info(
            "U-Net Epoch [%d/%d] - Train Loss: %.4f | Val Loss: %.4f | Val mIoU: %.4f | Pixel Acc: %.4f",
            epoch, epochs_unet, epoch_train_loss, epoch_val_loss, val_miou, metrics.pixel_accuracy
        )

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": unet.state_dict(),
                "optimizer_state_dict": optimizer_seg.state_dict(),
                "val_miou": best_val_miou,
                "val_metrics": metrics.to_dict(),
                "dataset": "Copernicus Sentinel-2 Real Satellite Data",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            save_checkpoint(checkpoint, unet_ckpt_path)
            # Also update primary checkpoint
            save_checkpoint(checkpoint, config.UNET_CONFIG["checkpoint_path"])
            logger.info("Saved new best Real U-Net checkpoint (mIoU: %.4f)", best_val_miou)

    elapsed_unet = time.time() - start_time_unet
    logger.info("Real U-Net training completed in %.1f seconds on %s.", elapsed_unet, device)

    # Test evaluation on unseen geographic area
    best_ckpt = torch.load(unet_ckpt_path, map_location=device, weights_only=False)
    unet.load_state_dict(best_ckpt["model_state_dict"])
    unet.eval()

    test_preds_list = []
    test_targets_list = []
    with torch.no_grad():
        for b_imgs, b_masks in test_seg_loader:
            b_imgs = b_imgs.to(device)
            preds = torch.argmax(unet(b_imgs), dim=1).cpu().numpy()
            test_preds_list.append(preds)
            test_targets_list.append(b_masks.numpy())

    test_preds = np.concatenate(test_preds_list, axis=0)
    test_targets = np.concatenate(test_targets_list, axis=0)
    real_test_metrics = compute_segmentation_metrics(test_targets, test_preds, num_classes=4)
    logger.info("Real Test Partition - mIoU: %.4f | Pixel Acc: %.4f", real_test_metrics.miou, real_test_metrics.pixel_accuracy)

    # =========================================================================
    # STAGE 2: TRAIN LST REGRESSION ANN ON REAL SATELLITE FEATURES (GPU)
    # =========================================================================
    logger.info("--- [STAGE 2] Training LST Regression ANN on Real Environmental Features (GPU) ---")

    train_reg_ds = LSTRegressionDataset(features[train_idx], lst_targets[train_idx], fit_scaler=True)
    scaler = train_reg_ds.scaler
    scaler_path = config.CHECKPOINTS_DIR / "real_feature_scaler.joblib"
    train_reg_ds.save_scaler(scaler_path)
    train_reg_ds.save_scaler(config.REGRESSION_CONFIG["scaler_path"])

    val_reg_ds = LSTRegressionDataset(features[val_idx], lst_targets[val_idx], scaler=scaler, fit_scaler=False)
    test_reg_ds = LSTRegressionDataset(features[test_idx], lst_targets[test_idx], scaler=scaler, fit_scaler=False)

    train_reg_loader = DataLoader(train_reg_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_reg_loader = DataLoader(val_reg_ds, batch_size=batch_size, shuffle=False)
    test_reg_loader = DataLoader(test_reg_ds, batch_size=batch_size, shuffle=False)

    reg_model = LSTRegressionANN(
        in_features=7,
        hidden_layers=config.REGRESSION_CONFIG["hidden_layers"],
        dropout_p=config.REGRESSION_CONFIG["dropout"]
    ).to(device)

    criterion_reg = nn.MSELoss()
    optimizer_reg = torch.optim.AdamW(reg_model.parameters(), lr=lr_regression, weight_decay=1e-5)
    scheduler_reg = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_reg, mode="min", factor=0.5, patience=5)

    best_val_rmse = float("inf")
    reg_ckpt_path = config.CHECKPOINTS_DIR / "real_best_regression_ann.pt"

    start_time_reg = time.time()

    for epoch in range(1, epochs_regression + 1):
        reg_model.train()
        running_loss = 0.0

        for b_x, b_y in train_reg_loader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            optimizer_reg.zero_grad()
            preds = reg_model(b_x)
            loss = criterion_reg(preds, b_y)
            loss.backward()
            optimizer_reg.step()

            running_loss += loss.item() * b_x.size(0)

        epoch_loss = running_loss / len(train_reg_ds)

        # Validation
        reg_model.eval()
        val_preds_list = []
        val_targets_list = []

        with torch.no_grad():
            for b_x, b_y in val_reg_loader:
                b_x = b_x.to(device)
                b_y = b_y.to(device)
                preds = reg_model(b_x)
                val_preds_list.append(preds.cpu().numpy())
                val_targets_list.append(b_y.cpu().numpy())

        val_preds = np.concatenate(val_preds_list, axis=0)
        val_targets = np.concatenate(val_targets_list, axis=0)
        reg_metrics = compute_regression_metrics(val_targets, val_preds)
        scheduler_reg.step(reg_metrics.rmse)

        if epoch % 5 == 0 or epoch == epochs_regression:
            logger.info(
                "Regression Epoch [%d/%d] - Loss: %.4f | Val RMSE: %.4f | MAE: %.4f | R^2: %.4f",
                epoch, epochs_regression, epoch_loss, reg_metrics.rmse, reg_metrics.mae, reg_metrics.r2
            )

        if reg_metrics.rmse < best_val_rmse:
            best_val_rmse = reg_metrics.rmse
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": reg_model.state_dict(),
                "optimizer_state_dict": optimizer_reg.state_dict(),
                "val_rmse": best_val_rmse,
                "val_metrics": reg_metrics.to_dict(),
                "dataset": "Real Environmental & Satellite LST Observations",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            save_checkpoint(checkpoint, reg_ckpt_path)
            save_checkpoint(checkpoint, config.REGRESSION_CONFIG["checkpoint_path"])

    elapsed_reg = time.time() - start_time_reg
    logger.info("Real Regression ANN training finished in %.1f seconds on %s.", elapsed_reg, device)

    # Test evaluation
    best_reg_ckpt = torch.load(reg_ckpt_path, map_location=device, weights_only=False)
    reg_model.load_state_dict(best_reg_ckpt["model_state_dict"])
    reg_model.eval()

    test_preds_list = []
    test_targets_list = []
    with torch.no_grad():
        for b_x, b_y in test_reg_loader:
            b_x = b_x.to(device)
            preds = reg_model(b_x).cpu().numpy()
            test_preds_list.append(preds)
            test_targets_list.append(b_y.numpy())

    test_preds = np.concatenate(test_preds_list, axis=0)
    test_targets = np.concatenate(test_targets_list, axis=0)
    real_test_reg_metrics = compute_regression_metrics(test_targets, test_preds)

    logger.info("Real Test Partition - RMSE: %.4f °C | MAE: %.4f °C | R^2: %.4f", real_test_reg_metrics.rmse, real_test_reg_metrics.mae, real_test_reg_metrics.r2)

    # Final summary
    summary = {
        "dataset_type": "Real Satellite Data (Copernicus Sentinel-2)",
        "hardware_used": device_info.get("current_device"),
        "cuda_active": device_info.get("cuda_available"),
        "total_patches": num_samples,
        "unet_test_miou": real_test_metrics.miou,
        "unet_test_pixel_acc": real_test_metrics.pixel_accuracy,
        "regression_test_rmse": real_test_reg_metrics.rmse,
        "regression_test_mae": real_test_reg_metrics.mae,
        "regression_test_r2": real_test_reg_metrics.r2,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    save_json_history(summary, config.REPORTS_DIR / "real_training_summary.json")
    logger.info("Exported real training summary report.")

    return summary


if __name__ == "__main__":
    train_on_real_satellite_data(epochs_unet=12, epochs_regression=30)
