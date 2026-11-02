"""
Configuration settings for AI-Powered Urban Heat Island (UHI) Mitigation System.
Centralizes paths, hyperparameters, geospatial specifications, and class definitions.
"""

from pathlib import Path
import os
import torch

# ---------------------------------------------------------
# DIRECTORY PATHS
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
SAMPLES_DIR = DATA_DIR / "samples"

MODELS_DIR = BASE_DIR / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

OUTPUTS_DIR = BASE_DIR / "outputs"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
VISUALIZATIONS_DIR = OUTPUTS_DIR / "visualizations"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# Ensure essential directories exist
for directory in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SYNTHETIC_DATA_DIR,
    SAMPLES_DIR,
    CHECKPOINTS_DIR,
    PREDICTIONS_DIR,
    VISUALIZATIONS_DIR,
    REPORTS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# LAND COVER CLASSES & COLOR CODING
# ---------------------------------------------------------
CLASS_NAMES = {
    0: "Impervious Surface",
    1: "Vegetation",
    2: "Bare Soil",
    3: "Water"
}

NUM_CLASSES = len(CLASS_NAMES)

# Distinct scientific visualization palette
CLASS_HEX_COLORS = {
    0: "#E63946",  # Red - Impervious / Urban
    1: "#2A9D8F",  # Green - Vegetation
    2: "#E9C46A",  # Yellow/Gold - Bare Soil
    3: "#264653"   # Deep Blue - Water
}

CLASS_RGB_COLORS = {
    0: (230, 57, 70),
    1: (42, 157, 143),
    2: (233, 196, 106),
    3: (38, 70, 83)
}

# ---------------------------------------------------------
# ENVIRONMENTAL METADATA SPECIFICATIONS
# ---------------------------------------------------------
# Features: [Impervious%, Vegetation%, BareSoil%, Water%, Elevation, Humidity, SolarIrradiance]
ENVIRONMENTAL_FEATURES = [
    "elevation",        # Elevation in meters (DEM)
    "humidity",         # Baseline Relative Humidity (%)
    "solar_irradiance"  # Solar Irradiance (W/m^2)
]

FEATURE_NAMES = [
    "impervious_pct",
    "vegetation_pct",
    "bare_soil_pct",
    "water_pct",
    "elevation_m",
    "humidity_pct",
    "solar_irradiance_wm2"
]

TOTAL_REGRESSION_FEATURES = len(FEATURE_NAMES)  # 7

# Plausible ranges for synthetic and normalization baselines
FEATURE_RANGES = {
    "elevation_m": (10.0, 800.0),
    "humidity_pct": (15.0, 95.0),
    "solar_irradiance_wm2": (200.0, 1100.0)
}

# LST Anomaly Reference Baseline (°C deviation relative to regional mean)
LST_ANOMALY_RANGE = (-5.0, 12.0)

# ---------------------------------------------------------
# MODEL HYPERPARAMETERS
# ---------------------------------------------------------
RANDOM_SEED = 42

# U-Net Segmentation
UNET_CONFIG = {
    "in_channels": 3,          # 3 for RGB, 4 for RGB+NIR
    "num_classes": 4,
    "init_features": 32,
    "patch_size": 256,
    "batch_size": 8,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "num_epochs": 20,
    "checkpoint_path": CHECKPOINTS_DIR / "best_unet_model.pt",
    "history_path": CHECKPOINTS_DIR / "unet_history.json"
}

# LST Regression ANN
REGRESSION_CONFIG = {
    "in_features": TOTAL_REGRESSION_FEATURES,
    "hidden_layers": [128, 64, 32],
    "dropout": 0.15,
    "batch_size": 16,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "num_epochs": 40,
    "checkpoint_path": CHECKPOINTS_DIR / "best_regression_ann.pt",
    "scaler_path": CHECKPOINTS_DIR / "feature_scaler.joblib",
    "history_path": CHECKPOINTS_DIR / "regression_history.json"
}

# ---------------------------------------------------------
# SIMULATION SCENARIOS
# ---------------------------------------------------------
DEFAULT_INTERVENTION_SCENARIOS = [0.10, 0.20, 0.30, 0.40, 0.50]

# ---------------------------------------------------------
# HARDWARE / DEVICE MANAGEMENT
# ---------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
