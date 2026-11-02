"""
Integrated AI Pipeline for Urban Heat Island (UHI) Analysis.
Connects U-Net Semantic Segmentation and Deep ANN LST Regression into a unified inference flow.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import torch
import joblib

import config
from models.unet import UNet
from models.regression_ann import LSTRegressionANN
from utils.device import get_device
from utils.preprocessing import normalize_satellite_image, compute_landcover_percentages, assemble_feature_vector
from utils.logging_utils import setup_logger

logger = setup_logger(__name__)


class IntegratedUHIPipeline:
    """
    Unified end-to-end AI pipeline executing:
    Image Ingestion -> U-Net Segmentation -> Class Statistics ->
    Feature Assembly -> Regression ANN -> LST Anomaly Prediction.
    """

    def __init__(
        self,
        unet_weights_path: Optional[Path] = None,
        regression_weights_path: Optional[Path] = None,
        scaler_path: Optional[Path] = None,
        device: Optional[torch.device] = None
    ):
        self.device = device if device is not None else get_device()
        self.unet_weights_path = Path(unet_weights_path) if unet_weights_path else config.UNET_CONFIG["checkpoint_path"]
        self.regression_weights_path = Path(regression_weights_path) if regression_weights_path else config.REGRESSION_CONFIG["checkpoint_path"]
        self.scaler_path = Path(scaler_path) if scaler_path else config.REGRESSION_CONFIG["scaler_path"]

        self.unet: Optional[UNet] = None
        self.regression_ann: Optional[LSTRegressionANN] = None
        self.scaler = None
        self.is_loaded = False

    def load_models(self) -> None:
        """
        Loads pre-trained U-Net weights, Regression ANN weights, and feature scaler.
        """
        logger.info("Loading UHI integrated pipeline models...")

        # 1. Load U-Net
        self.unet = UNet(
            in_channels=config.UNET_CONFIG["in_channels"],
            num_classes=config.UNET_CONFIG["num_classes"]
        ).to(self.device)

        if not self.unet_weights_path.exists():
            raise FileNotFoundError(f"U-Net checkpoint missing at: {self.unet_weights_path}. Train the model first.")

        unet_ckpt = torch.load(self.unet_weights_path, map_location=self.device, weights_only=False)
        self.unet.load_state_dict(unet_ckpt.get("model_state_dict", unet_ckpt))
        self.unet.eval()

        # 2. Load Regression ANN
        self.regression_ann = LSTRegressionANN(
            in_features=config.TOTAL_REGRESSION_FEATURES,
            hidden_layers=config.REGRESSION_CONFIG["hidden_layers"],
            dropout_p=config.REGRESSION_CONFIG["dropout"]
        ).to(self.device)

        if not self.regression_weights_path.exists():
            raise FileNotFoundError(f"Regression checkpoint missing at: {self.regression_weights_path}. Train the model first.")

        reg_ckpt = torch.load(self.regression_weights_path, map_location=self.device, weights_only=False)
        self.regression_ann.load_state_dict(reg_ckpt.get("model_state_dict", reg_ckpt))
        self.regression_ann.eval()

        # 3. Load Scaler
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Feature scaler missing at: {self.scaler_path}. Run training first.")

        self.scaler = joblib.load(self.scaler_path)
        self.is_loaded = True
        logger.info("Integrated UHI Pipeline initialized successfully on %s.", self.device)

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Validates, normalizes, and shapes input image array to (1, C, H, W) torch tensor.
        """
        if image.ndim == 2:
            # Grayscale to RGB
            image = np.stack([image, image, image], axis=0)
        elif image.ndim == 3 and image.shape[-1] == 3:
            # (H, W, 3) -> (3, H, W)
            image = np.transpose(image, (2, 0, 1))

        norm_img = normalize_satellite_image(image)
        tensor = torch.from_numpy(norm_img).float().unsqueeze(0).to(self.device)
        return tensor

    def segment_image(self, image_tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs U-Net segmentation.
        Returns:
            mask: (H, W) class indices [0..3]
            probabilities: (4, H, W) softmax probabilities
        """
        if not self.is_loaded:
            self.load_models()

        with torch.no_grad():
            logits = self.unet(image_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            mask = np.argmax(probs, axis=0)

        return mask, probs

    def calculate_landcover_percentages(self, mask: np.ndarray) -> Dict[str, float]:
        """
        Calculates exact land cover proportions summing to ~100%.
        """
        return compute_landcover_percentages(mask)

    def prepare_regression_features(
        self,
        landcover_pcts: Dict[str, float],
        elevation_m: float,
        humidity_pct: float,
        solar_irradiance_wm2: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assembles 7-dimensional feature vector and scales it using fitted scaler.
        Returns:
            raw_features: (1, 7) np.ndarray
            scaled_features: (1, 7) np.ndarray
        """
        raw = assemble_feature_vector(
            landcover_pcts=landcover_pcts,
            elevation_m=elevation_m,
            humidity_pct=humidity_pct,
            solar_irradiance_wm2=solar_irradiance_wm2
        ).reshape(1, -1)

        scaled = self.scaler.transform(raw)
        return raw, scaled

    def predict_lst(self, scaled_features: np.ndarray) -> float:
        """
        Passes scaled features through Regression ANN to predict LST Anomaly (°C).
        """
        if not self.is_loaded:
            self.load_models()

        tensor_x = torch.from_numpy(scaled_features).float().to(self.device)
        with torch.no_grad():
            pred = self.regression_ann(tensor_x)
            lst_anomaly = float(pred.item())
        return round(lst_anomaly, 3)

    def analyze_geographic_region(
        self,
        image: np.ndarray,
        elevation_m: float = 210.0,
        humidity_pct: float = 55.0,
        solar_irradiance_wm2: float = 680.0
    ) -> Dict[str, Any]:
        """
        Executes complete end-to-end analysis on an optical satellite patch.

        Returns structured dictionary containing:
            - segmentation_mask: (H, W)
            - class_probabilities: (4, H, W)
            - landcover_percentages: Dict[str, float]
            - environmental_metadata: Dict[str, float]
            - raw_features: (7,)
            - predicted_lst_anomaly: float (°C)
            - heat_risk_level: str ("Low", "Moderate", "High", "Critical")
        """
        img_tensor = self.preprocess_image(image)
        mask, probs = self.segment_image(img_tensor)
        landcover_pcts = self.calculate_landcover_percentages(mask)

        raw_feats, scaled_feats = self.prepare_regression_features(
            landcover_pcts=landcover_pcts,
            elevation_m=elevation_m,
            humidity_pct=humidity_pct,
            solar_irradiance_wm2=solar_irradiance_wm2
        )

        predicted_lst = self.predict_lst(scaled_feats)

        # Categorize heat risk level
        if predicted_lst < 1.0:
            heat_risk = "Low"
        elif predicted_lst < 3.5:
            heat_risk = "Moderate"
        elif predicted_lst < 6.0:
            heat_risk = "High"
        else:
            heat_risk = "Critical Urban Heat Hotspot"

        return {
            "segmentation_mask": mask,
            "class_probabilities": probs,
            "landcover_percentages": landcover_pcts,
            "environmental_metadata": {
                "elevation_m": elevation_m,
                "humidity_pct": humidity_pct,
                "solar_irradiance_wm2": solar_irradiance_wm2
            },
            "raw_features": raw_feats.squeeze(0),
            "predicted_lst_anomaly": predicted_lst,
            "heat_risk_level": heat_risk
        }
