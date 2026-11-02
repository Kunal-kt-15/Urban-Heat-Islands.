"""
Training pipeline for U-Net Satellite Image Semantic Segmentation.
Includes early stopping, learning rate scheduling, validation mIoU tracking, and checkpointing.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from models.unet import UNet
from data_pipeline.dataset import SatelliteSegmentationDataset, create_spatial_splits
from data_pipeline.synthetic_generator import SyntheticDataGenerator
from training.metrics import compute_segmentation_metrics
from utils.device import get_device
from utils.checkpoint_utils import save_checkpoint, save_json_history
from utils.logging_utils import setup_logger

logger = setup_logger(__name__)


def train_unet(
    data_path: Optional[Path] = None,
    epochs: int = config.UNET_CONFIG["num_epochs"],
    batch_size: int = config.UNET_CONFIG["batch_size"],
    lr: float = config.UNET_CONFIG["learning_rate"],
    in_channels: int = config.UNET_CONFIG["in_channels"],
    num_classes: int = config.UNET_CONFIG["num_classes"],
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Executes complete U-Net training and validation workflow.
    """
    if device is None:
        device = get_device()

    logger.info("Starting U-Net Training on device: %s", device)

    # 1. Load Data
    npz_path = data_path if data_path else config.SYNTHETIC_DATA_DIR / "synthetic_uhi_dataset.npz"
    if not npz_path.exists():
        logger.info("Dataset not found at %s. Generating synthetic dataset...", npz_path)
        gen = SyntheticDataGenerator()
        gen.generate_dataset(num_samples=500)

    data = np.load(npz_path)
    images = data["images"]  # (500, 3, 256, 256)
    masks = data["masks"]    # (500, 256, 256)

    # 2. Spatial Train / Val / Test Partitioning
    train_idx, val_idx, test_idx = create_spatial_splits(len(images), train_ratio=0.70, val_ratio=0.15)
    logger.info("Dataset splits: Train=%d, Val=%d, Test=%d", len(train_idx), len(val_idx), len(test_idx))

    train_ds = SatelliteSegmentationDataset(images[train_idx], masks[train_idx], transform=True)
    val_ds = SatelliteSegmentationDataset(images[val_idx], masks[val_idx], transform=False)
    test_ds = SatelliteSegmentationDataset(images[test_idx], masks[test_idx], transform=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 3. Model, Loss, Optimizer, Scheduler
    model = UNet(in_channels=in_channels, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.UNET_CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_miou": [],
        "val_pixel_acc": [],
        "best_epoch": 0,
        "best_val_miou": 0.0
    }

    best_val_miou = 0.0
    best_weights_path = config.UNET_CONFIG["checkpoint_path"]

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_train_loss = 0.0

        for batch_imgs, batch_masks in train_loader:
            batch_imgs = batch_imgs.to(device)
            batch_masks = batch_masks.to(device)

            optimizer.zero_grad()
            logits = model(batch_imgs)
            loss = criterion(logits, batch_masks)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * batch_imgs.size(0)

        epoch_train_loss = running_train_loss / len(train_ds)

        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        val_preds_list = []
        val_targets_list = []

        with torch.no_grad():
            for batch_imgs, batch_masks in val_loader:
                batch_imgs = batch_imgs.to(device)
                batch_masks = batch_masks.to(device)

                logits = model(batch_imgs)
                loss = criterion(logits, batch_masks)
                running_val_loss += loss.item() * batch_imgs.size(0)

                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds_list.append(preds)
                val_targets_list.append(batch_masks.cpu().numpy())

        epoch_val_loss = running_val_loss / len(val_ds)
        val_preds_all = np.concatenate(val_preds_list, axis=0)
        val_targets_all = np.concatenate(val_targets_list, axis=0)

        metrics = compute_segmentation_metrics(val_targets_all, val_preds_all, num_classes=num_classes)
        val_miou = metrics.miou
        scheduler.step(val_miou)

        history["train_loss"].append(round(epoch_train_loss, 4))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["val_miou"].append(round(val_miou, 4))
        history["val_pixel_acc"].append(round(metrics.pixel_accuracy, 4))

        logger.info(
            "Epoch [%d/%d] - Train Loss: %.4f | Val Loss: %.4f | Val mIoU: %.4f | Acc: %.4f",
            epoch, epochs, epoch_train_loss, epoch_val_loss, val_miou, metrics.pixel_accuracy
        )

        # Checkpoint Best Model
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            history["best_epoch"] = epoch
            history["best_val_miou"] = round(best_val_miou, 4)

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_miou": best_val_miou,
                "val_metrics": metrics.to_dict(),
                "in_channels": in_channels,
                "num_classes": num_classes,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            save_checkpoint(checkpoint, best_weights_path)
            logger.info("Saved new best U-Net checkpoint with mIoU: %.4f", best_val_miou)

    elapsed = time.time() - start_time
    logger.info("U-Net training finished in %.1f seconds. Best mIoU: %.4f", elapsed, best_val_miou)

    # 4. Final Evaluation on Held-Out Test Set
    logger.info("Evaluating Best U-Net on Unseen Geographic Test Partition...")
    best_ckpt = torch.load(best_weights_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    test_preds_list = []
    test_targets_list = []

    with torch.no_grad():
        for batch_imgs, batch_masks in test_loader:
            batch_imgs = batch_imgs.to(device)
            preds = torch.argmax(model(batch_imgs), dim=1).cpu().numpy()
            test_preds_list.append(preds)
            test_targets_list.append(batch_masks.numpy())

    test_preds_all = np.concatenate(test_preds_list, axis=0)
    test_targets_all = np.concatenate(test_targets_list, axis=0)
    test_metrics = compute_segmentation_metrics(test_targets_all, test_preds_all, num_classes=num_classes)

    history["test_metrics"] = test_metrics.to_dict()
    save_json_history(history, config.UNET_CONFIG["history_path"])
    logger.info("Test Set mIoU: %.4f | Pixel Acc: %.4f", test_metrics.miou, test_metrics.pixel_accuracy)

    return history


if __name__ == "__main__":
    train_unet(epochs=15)
