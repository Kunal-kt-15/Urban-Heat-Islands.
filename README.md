# AI-Powered Urban Heat Island (UHI) Mitigation Platform
### Semantic Segmentation & Deep Learning Regression for Microclimate Intervention

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5%20CUDA-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 1. Project Abstract
Rapid worldwide urbanization has drastically altered local surface energy budgets. Pervious vegetative cover is continually displaced by impervious surfaces (asphalt, concrete, roofing), producing pronounced **Urban Heat Islands (UHI)**. This project presents an integrated, modular, research-oriented deep learning and geospatial intelligence platform that pairs:
1. **High-Resolution Satellite Image Semantic Segmentation (U-Net)** to classify surface composition into four discrete land-cover classes (Impervious Surface, Vegetation, Bare Soil, Water).
2. **Land Surface Temperature Anomaly Regression (Deep ANN)** to predict microclimatic thermal elevation using land-cover fractions coupled with environmental metadata (elevation, humidity, solar irradiance).
3. **Urban Heat Island Mitigation Simulator** providing counterfactual scenario forecasting (10% to 50% conversion of impervious surfaces into vegetation canopy).

---

## 2. Problem Statement & Research Objectives
Conventional urban microclimate studies depend heavily on sparse meteorological stations or retrospective satellite thermal imaging. Urban planners lack interactive, predictive simulation tools capable of estimating the quantitative cooling impact of green infrastructure interventions prior to costly physical investments.

### Key Questions Addressed:
- What percentage of a given urban sector is composed of impervious materials vs. vegetation canopy?
- What is the estimated Land Surface Temperature (LST) anomaly index for that sector?
- How much cooling (LST anomaly reduction) is forecast if 10%, 20%, 30%, or 50% of impervious surfaces are retrofitted into vegetation corridors?
- How do environmental forcing factors (elevation, baseline humidity, solar irradiance) modulate the urban heat island effect?

---

## 3. System Architecture & Workflow

```
+-------------------------------------------------------------------------+
|                         SATELLITE SCENE / PATCH                         |
|                     (3-Band RGB or 4-Band RGB+NIR)                      |
+------------------------------------+------------------------------------+
                                     |
                                     v
                       [ Geospatial Preprocessing ]
                       - Radiometric Normalization
                       - 256x256 Tiling & Alignment
                                     |
                                     v
                        [ Phase 1: U-Net Model ]
                     Encoder-Decoder Skip Connections
                     Multi-Class Softmax Classification
                                     |
            +------------------------+------------------------+
            |                        |                        |
            v                        v                        v
       Class 0: Urban           Class 1: Veg             Class 2: Soil / Class 3: Water
       (Impervious %)          (Vegetation %)           (Bare Soil % / Water %)
            |                        |                        |
            +------------------------+------------------------+
                                     |
                                     v
                       [ Dynamic Land-Cover Stats ]
                         Sum of Fractions = 100%
                                     |
                                     +<--- [ Environmental Metadata ]
                                     |     (Elevation, Humidity, Solar)
                                     v
                       [ 7D Combined Feature Vector ]
                                     |
                                     v
                      [ Phase 2: LST Regression ANN ]
                        Dense [128 -> 64 -> 32]
                         BatchNorm1d + Dropout
                                     |
                                     v
                     [ Predicted LST Anomaly (°C) ]
                                     |
                                     v
                   [ UHI Mitigation Simulation Engine ]
                 - Contiguous Greening Scenario Modeling
                 - 10% - 50% Impervious-to-Vegetation Shift
                 - Counterfactual LST Reduction Delta Map
```

---

## 4. Land-Cover Classification Taxonomy
The segmentation network enforces four mutually exclusive classes with standardized scientific visualization color codes:

| Class ID | Land-Cover Category | Description | Hex Color | Display Badge |
|---|---|---|---|---|
| **0** | **Impervious Surface** | Asphalt roads, concrete pavements, commercial & residential rooftops | `#E63946` | Red |
| **1** | **Vegetation** | Tree canopy, manicured lawns, urban parks, nature reserves | `#2A9D8F` | Green |
| **2** | **Bare Soil** | Exposed earth, construction sites, fallow fields | `#E9C46A` | Yellow/Gold |
| **3** | **Water** | Rivers, canals, urban reservoirs, lakes | `#264653` | Deep Blue |

---

## 5. Directory Structure

```
uhi_mitigation_project/
├── app.py                      # Multi-page interactive Streamlit dashboard
├── config.py                   # Central hyperparameters, paths, and configurations
├── requirements.txt            # Python dependencies
├── pytest.ini                  # Pytest configuration
├── run_tests.py                # Automated unit and integration test runner
├── README.md                   # Comprehensive project documentation
│
├── data/
│   ├── raw/                    # Raw external satellite and thermal rasters
│   ├── processed/              # Resampled, cropped, and aligned GeoTIFFs
│   ├── synthetic/              # 500-sample paired dataset (.npz)
│   └── samples/                # Pre-packaged demonstration zones (.npz)
│
├── models/
│   ├── __init__.py
│   ├── unet.py                 # Encoder-Decoder U-Net with skip connections
│   ├── regression_ann.py       # 7D -> 1D Deep Multilayer Perceptron
│   └── checkpoints/            # Saved weights (.pt) and training histories (.json)
│
├── data_pipeline/
│   ├── __init__.py
│   ├── synthetic_generator.py  # Spatially contiguous multi-class landscape synthesizer
│   ├── dataset.py              # PyTorch datasets with spatial split logic
│   ├── patch_extractor.py      # 256x256 geospatial patch tiling & reconstruction
│   ├── geospatial_processor.py # GeoTIFF I/O, CRS transforms, and grid-cell aggregator
│   └── raster_aligner.py       # Nearest-neighbor and bilinear multi-sensor alignment
│
├── training/
│   ├── __init__.py
│   ├── train_segmentation.py   # U-Net trainer with AdamW, scheduler, and mIoU tracking
│   ├── train_regression.py     # LST ANN trainer with feature scaling and RMSE tracking
│   ├── train_integrated.py     # Sequential end-to-end training and drift validation
│   └── metrics.py              # mIoU, per-class IoU, confusion matrix, RMSE, MAE, R²
│
├── inference/
│   ├── __init__.py
│   ├── integrated_pipeline.py  # IntegratedUHIPipeline end-to-end orchestration class
│   └── prediction.py           # Standalone and batch inference functions
│
├── simulation/
│   ├── __init__.py
│   ├── mitigation_engine.py    # simulate_mitigation() with contiguous urban greening
│   └── scenario_analysis.py    # 10% - 50% multi-scenario comparison tables
│
├── visualization/
│   ├── __init__.py
│   ├── segmentation_visualizer.py # RGB colorization and satellite overlay blending
│   ├── lst_visualizer.py       # Continuous thermal anomaly heatmaps and delta maps
│   └── dashboard_charts.py     # Plotly donuts, scenario curves, and scatter plots
│
├── utils/
│   ├── __init__.py
│   ├── device.py               # Automatic CUDA / CPU detection and diagnostics
│   ├── preprocessing.py        # Normalization, land-cover percentages, and feature packing
│   ├── logging_utils.py        # Centralized structured logging
│   └── checkpoint_utils.py     # PyTorch checkpoint and JSON history I/O
│
├── tests/
│   ├── __init__.py
│   ├── test_unet.py            # U-Net tensor dimensions and softmax validation
│   ├── test_regression.py      # ANN dimensions and single-vector handling
│   ├── test_geospatial.py      # Patch tiling, categorical resampling, and grid cells
│   └── test_simulation.py      # Land-cover conservation and greening constraint tests
│
└── outputs/
    ├── predictions/            # Exported inference outputs
    ├── visualizations/         # Saved figures and heatmaps
    └── reports/                # Zonal CSV and GeoJSON exports
```

---

## 6. Installation & Environment Setup

### Prerequisites
- Python 3.11 or Python 3.12 (64-bit)
- NVIDIA GPU with CUDA 12+ (optional, automatically uses CPU if unavailable)
- Windows PowerShell or Terminal

### Step 1: Create and Activate Virtual Environment
```powershell
# Navigate to project root
cd "C:\Projects\Kunal Project"

# Create virtual environment with Python 3.11
py -3.11 -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

### Step 2: Install PyTorch with CUDA Acceleration
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Install Remaining Requirements
```powershell
pip install -r requirements.txt
```

---

## 7. Model Training & Pipeline Execution

### Train Integrated Pipeline (Two-Stage Workflow)
To generate the 500-sample synthetic dataset, train the U-Net, train the regression ANN, and evaluate end-to-end performance:
```powershell
.\venv\Scripts\python.exe -m training.train_integrated
```

### Train Segmentation Network Individually
```powershell
.\venv\Scripts\python.exe -m training.train_segmentation
```

### Train Regression ANN Individually
```powershell
.\venv\Scripts\python.exe -m training.train_regression
```

---

## 8. Launching the Web Applications

### A. Dedicated Modern Web Application (FastAPI + Modern HTML5/CSS3)
Launch the state-of-the-art custom web platform:
```powershell
.\venv\Scripts\python.exe -m uvicorn web_app.server:app --host 0.0.0.0 --port 8080
```
- **Live Local URL:** [http://localhost:8080](http://localhost:8080)
- **Features:** Custom design system (Midnight Obsidian & Cyber Emerald), interactive U-Net satellite vision, real-time thermal slider calculations, contiguous greening simulator with live before/after map render, Chart.js multi-scenario graphs, and GPU hardware telemetry.

### B. Interactive Streamlit Dashboard (Alternative)
```powershell
.\venv\Scripts\streamlit.exe run app.py
```
- **Local URL:** [http://localhost:8501](http://localhost:8501)

---

## 9. Automated Testing
Run the complete unit and integration test suite:
```powershell
.\venv\Scripts\python.exe run_tests.py
```
Or directly via pytest:
```powershell
.\venv\Scripts\pytest.exe tests/ -v
```

---

## 10. Scientific Limitations & Responsible AI Notice
1. **LST vs. Air Temperature:** Models predict Land Surface Temperature (LST) anomaly indices, not 2-meter ambient air temperature. LST exhibits higher daytime amplitudes than air temperature.
2. **Synthetic Data Notice:** The synthetic dataset engine creates realistic spatial patterns for software validation. It does not replace ground-truth multi-spectral imagery (e.g., Sentinel-2) and calibrated thermal products (e.g., Landsat-8/9 TIRS).
3. **Counterfactual Greening:** Simulated cooling represents model-inferred scenario estimates under specified environmental conditions. Real-world cooling depends additionally on tree canopy species, soil moisture availability, wind corridors, and three-dimensional urban canyon geometry.
