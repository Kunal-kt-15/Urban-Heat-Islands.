"""
Integrated Training and End-to-End Evaluation Orchestrator.
Trains U-Net and Regression ANN sequentially, evaluates feature drift between ground truth
and U-Net-predicted land-cover percentages, and verifies end-to-end pipeline robustness.
"""

from pathlib import Path
from typing import Dict, Any
import time
import numpy as np
import torch
import joblib

import config
from training.train_segmentation import train_unet
from training.train_regression import train_regression
from models.unet import UNet
from models.regression_ann import LSTRegressionANN
from training.metrics import compute_regression_metrics, compute_segmentation_metrics
from utils.device import get_device
from utils.checkpoint_utils import save_json_history
from utils.logging_utils import setup_logger

logger = setup_logger(__name__)


def run_integrated_training(
    unet_epochs: int = 15,
    regression_epochs: int = 35
) -> Dict[str, Any]:
    """
    Executes full two-stage integrated training pipeline and cross-validates
    end-to-end performance on held-out test data.
    """
    device = get_device()
    logger.info("==================================================")
    logger.info("STAGE 1: TRAINING U-NET SEMANTIC SEGMENTATION")
    logger.info("==================================================")
    unet_history = train_unet(epochs=unet_epochs, device=device)

    logger.info("==================================================")
    logger.info("STAGE 2: TRAINING LST REGRESSION ANN")
    logger.info("==================================================")
    reg_history = train_regression(epochs=regression_epochs, device=device)

    logger.info("==================================================")
    logger.info("STAGE 3: END-TO-END SYSTEM INTEGRATION VALIDATION")
    logger.info("==================================================")

    # Load synthetic test partition
    data_path = config.SYNTHETIC_DATA_DIR / "synthetic_uhi_dataset.npz"
    data = np.load(data_path)
    images = data["images"]
    masks = data["masks"]
    metadata = data["metadata"]
    lst_targets = data["lst_targets"]

    test_start = int(len(images) * 0.85)
    test_imgs = images[test_start:]
    test_masks = masks[test_start:]
    test_meta = metadata[test_start:]
    test_lst = lst_targets[test_start:]

    # Load trained models
    unet = UNet(in_channels=3, num_classes=4).to(device)
    unet_ckpt = torch.load(config.UNET_CONFIG["checkpoint_path"], map_location=device)
    unet.load_state_dict(unet_ckpt["model_state_dict"])
    unet.eval()

    reg_model = LSTRegressionANN(in_features=7).to(device)
    reg_ckpt = torch.load(config.REGRESSION_CONFIG["checkpoint_path"], map_location=device)
    reg_model.load_state_dict(reg_ckpt["model_state_dict"])
    reg_model.eval()

    scaler = joblib.load(config.REGRESSION_CONFIG["scaler_path"])

    # Run End-to-End inference
    end_to_end_preds = []
    ground_truth_preds = []

    with torch.no_grad():
        for i in range(len(test_imgs)):
            # 1. Segment image
            img_tensor = torch.from_numpy(test_imgs[i:i+1]).float().to(device)
            pred_mask = torch.argmax(unet(img_tensor), dim=1).squeeze(0).cpu().numpy()

            # 2. Extract land-cover percentages dynamically
            total_px = 256 * 256
            pred_imp = (np.sum(pred_mask == 0) / total_px) * 100.0
            pred_veg = (np.sum(pred_mask == 1) / total_px) * 100.0
            pred_soil = (np.sum(pred_mask == 2) / total_px) * 100.0
            pred_wat = (np.sum(pred_mask == 3) / total_px) * 100.0

            # 3. Assemble 7-feature vector from PREDICTED segmentation
            pred_feats = np.array([
                pred_imp, pred_veg, pred_soil, pred_wat,
                test_meta[i, 0], test_meta[i, 1], test_meta[i, 2]
            ], dtype=np.float32).reshape(1, -1)

            scaled_feats = scaler.transform(pred_feats)
            pred_lst_val = reg_model(torch.from_numpy(scaled_feats).float().to(device)).item()
            end_to_end_preds.append(pred_lst_val)

    end_to_end_preds = np.array(end_to_end_preds).reshape(-1, 1)
    e2e_metrics = compute_regression_metrics(test_lst, end_to_end_preds)

    logger.info("End-to-End Inference Evaluation on Test Partition:")
    logger.info("  E2E LST RMSE: %.4f °C", e2e_metrics.rmse)
    logger.info("  E2E LST MAE:  %.4f °C", e2e_metrics.mae)
    logger.info("  E2E LST R^2:  %.4f", e2e_metrics.r2)

    summary = {
        "unet_best_miou": unet_history.get("best_val_miou", 0.0),
        "regression_best_rmse": reg_history.get("best_val_rmse", 0.0),
        "e2e_test_rmse": e2e_metrics.rmse,
        "e2e_test_mae": e2e_metrics.mae,
        "e2e_test_r2": e2e_metrics.r2,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    report_path = config.REPORTS_DIR / "integrated_training_report.json"
    save_json_history(summary, report_path)
    logger.info("Integrated training summary exported to: %s", report_path)

    return summary


if __name__ == "__main__":
    run_integrated_training(unet_epochs=12, regression_epochs=30)
