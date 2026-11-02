"""
Comprehensive evaluation metrics for Semantic Segmentation and LST Regression.
Calculates mIoU, per-class IoU, confusion matrices, RMSE, MAE, and R^2 score.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import confusion_matrix, r2_score, mean_squared_error, mean_absolute_error
import config


@dataclass
class SegmentationMetrics:
    miou: float
    per_class_iou: Dict[str, float]
    pixel_accuracy: float
    precision_macro: float
    recall_macro: float
    confusion_matrix: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        return {
            "miou": round(self.miou, 4),
            "per_class_iou": {k: round(v, 4) for k, v in self.per_class_iou.items()},
            "pixel_accuracy": round(self.pixel_accuracy, 4),
            "precision_macro": round(self.precision_macro, 4),
            "recall_macro": round(self.recall_macro, 4),
            "confusion_matrix": self.confusion_matrix.tolist()
        }


@dataclass
class RegressionMetrics:
    rmse: float
    mae: float
    r2: float
    residuals_mean: float
    residuals_std: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rmse": round(self.rmse, 4),
            "mae": round(self.mae, 4),
            "r2": round(self.r2, 4),
            "residuals_mean": round(self.residuals_mean, 4),
            "residuals_std": round(self.residuals_std, 4)
        }


def compute_segmentation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = config.NUM_CLASSES
) -> SegmentationMetrics:
    """
    Computes rigorous multi-class segmentation metrics:
    - Per-Class IoU = TP / (TP + FP + FN)
    - Mean IoU (mIoU)
    - Pixel Accuracy = (TP + TN) / Total
    - Macro-averaged Precision and Recall
    - Confusion Matrix (num_classes x num_classes)
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=list(range(num_classes)))

    per_class_iou = {}
    precisions = []
    recalls = []

    for c in range(num_classes):
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp

        denom_iou = tp + fp + fn
        iou = tp / denom_iou if denom_iou > 0 else 1.0

        class_name = config.CLASS_NAMES.get(c, f"class_{c}")
        per_class_iou[class_name] = float(iou)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)

    miou = float(np.mean(list(per_class_iou.values())))
    pixel_acc = float(np.sum(np.diag(cm)) / max(1, np.sum(cm)))
    precision_macro = float(np.mean(precisions))
    recall_macro = float(np.mean(recalls))

    return SegmentationMetrics(
        miou=miou,
        per_class_iou=per_class_iou,
        pixel_accuracy=pixel_acc,
        precision_macro=precision_macro,
        recall_macro=recall_macro,
        confusion_matrix=cm
    )


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> RegressionMetrics:
    """
    Computes rigorous regression quality metrics for LST Anomaly:
    - RMSE (°C)
    - MAE (°C)
    - R^2 Score
    - Residual statistics
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    rmse = float(np.sqrt(mean_squared_error(y_true_flat, y_pred_flat)))
    mae = float(mean_absolute_error(y_true_flat, y_pred_flat))
    r2 = float(r2_score(y_true_flat, y_pred_flat))

    residuals = y_true_flat - y_pred_flat
    res_mean = float(np.mean(residuals))
    res_std = float(np.std(residuals))

    return RegressionMetrics(
        rmse=rmse,
        mae=mae,
        r2=r2,
        residuals_mean=res_mean,
        residuals_std=res_std
    )
