"""
Training pipeline for Land Surface Temperature (LST) Anomaly Deep ANN Regression.
Includes feature scaling, validation RMSE/R^2 tracking, early stopping, and checkpointing.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from models.regression_ann import LSTRegressionANN
from data_pipeline.dataset import LSTRegressionDataset, create_spatial_splits
from data_pipeline.synthetic_generator import SyntheticDataGenerator
from training.metrics import compute_regression_metrics
from utils.device import get_device
from utils.checkpoint_utils import save_checkpoint, save_json_history
from utils.logging_utils import setup_logger

logger = setup_logger(__name__)


def train_regression(
    data_path: Optional[Path] = None,
    epochs: int = config.REGRESSION_CONFIG["num_epochs"],
    batch_size: int = config.REGRESSION_CONFIG["batch_size"],
    lr: float = config.REGRESSION_CONFIG["learning_rate"],
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Executes training and validation for LST Anomaly Regression ANN.
    """
    if device is None:
        device = get_device()

    logger.info("Starting LST Regression Training on device: %s", device)

    # 1. Load Data
    npz_path = data_path if data_path else config.SYNTHETIC_DATA_DIR / "synthetic_uhi_dataset.npz"
    if not npz_path.exists():
        logger.info("Dataset not found at %s. Generating synthetic dataset...", npz_path)
        gen = SyntheticDataGenerator()
        gen.generate_dataset(num_samples=500)

    data = np.load(npz_path)
    features = data["features"]      # (500, 7)
    lst_targets = data["lst_targets"]  # (500, 1)

    # 2. Spatial Train / Val / Test Partitioning
    train_idx, val_idx, test_idx = create_spatial_splits(len(features), train_ratio=0.70, val_ratio=0.15)
    logger.info("Regression splits: Train=%d, Val=%d, Test=%d", len(train_idx), len(val_idx), len(test_idx))

    # Feature Scaling on Training Partition ONLY (Prevent Data Leakage)
    train_ds = LSTRegressionDataset(features[train_idx], lst_targets[train_idx], fit_scaler=True)
    scaler = train_ds.scaler
    train_ds.save_scaler(config.REGRESSION_CONFIG["scaler_path"])

    val_ds = LSTRegressionDataset(features[val_idx], lst_targets[val_idx], scaler=scaler, fit_scaler=False)
    test_ds = LSTRegressionDataset(features[test_idx], lst_targets[test_idx], scaler=scaler, fit_scaler=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 3. Model, Loss, Optimizer, Scheduler
    model = LSTRegressionANN(
        in_features=config.TOTAL_REGRESSION_FEATURES,
        hidden_layers=config.REGRESSION_CONFIG["hidden_layers"],
        dropout_p=config.REGRESSION_CONFIG["dropout"]
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=config.REGRESSION_CONFIG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_rmse": [],
        "val_mae": [],
        "val_r2": [],
        "best_epoch": 0,
        "best_val_rmse": float("inf")
    }

    best_val_rmse = float("inf")
    best_weights_path = config.REGRESSION_CONFIG["checkpoint_path"]

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * batch_x.size(0)

        epoch_train_loss = running_train_loss / len(train_ds)

        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        val_preds_list = []
        val_targets_list = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                running_val_loss += loss.item() * batch_x.size(0)

                val_preds_list.append(preds.cpu().numpy())
                val_targets_list.append(batch_y.cpu().numpy())

        epoch_val_loss = running_val_loss / len(val_ds)
        val_preds_all = np.concatenate(val_preds_list, axis=0)
        val_targets_all = np.concatenate(val_targets_list, axis=0)

        metrics = compute_regression_metrics(val_targets_all, val_preds_all)
        scheduler.step(metrics.rmse)

        history["train_loss"].append(round(epoch_train_loss, 4))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["val_rmse"].append(round(metrics.rmse, 4))
        history["val_mae"].append(round(metrics.mae, 4))
        history["val_r2"].append(round(metrics.r2, 4))

        if epoch % 5 == 0 or epoch == epochs:
            logger.info(
                "Epoch [%d/%d] - Train Loss: %.4f | Val RMSE: %.4f | MAE: %.4f | R^2: %.4f",
                epoch, epochs, epoch_train_loss, metrics.rmse, metrics.mae, metrics.r2
            )

        # Checkpoint Best Model
        if metrics.rmse < best_val_rmse:
            best_val_rmse = metrics.rmse
            history["best_epoch"] = epoch
            history["best_val_rmse"] = round(best_val_rmse, 4)

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_rmse": best_val_rmse,
                "val_metrics": metrics.to_dict(),
                "in_features": config.TOTAL_REGRESSION_FEATURES,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            save_checkpoint(checkpoint, best_weights_path)

    elapsed = time.time() - start_time
    logger.info("Regression ANN training finished in %.1f seconds. Best RMSE: %.4f", elapsed, best_val_rmse)

    # 4. Final Evaluation on Held-Out Test Set
    logger.info("Evaluating Best Regression ANN on Unseen Geographic Test Partition...")
    best_ckpt = torch.load(best_weights_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    test_preds_list = []
    test_targets_list = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            preds = model(batch_x).cpu().numpy()
            test_preds_list.append(preds)
            test_targets_list.append(batch_y.numpy())

    test_preds_all = np.concatenate(test_preds_list, axis=0)
    test_targets_all = np.concatenate(test_targets_list, axis=0)
    test_metrics = compute_regression_metrics(test_targets_all, test_preds_all)

    history["test_metrics"] = test_metrics.to_dict()
    save_json_history(history, config.REGRESSION_CONFIG["history_path"])
    logger.info("Test Set RMSE: %.4f °C | MAE: %.4f °C | R^2: %.4f", test_metrics.rmse, test_metrics.mae, test_metrics.r2)

    return history


if __name__ == "__main__":
    train_regression(epochs=35)
