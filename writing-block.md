# MASTER DEVELOPMENT PROMPT
## Project: AI-Powered Urban Heat Island (UHI) Mitigation Using Semantic Segmentation and Deep Learning Regression

### 1. ROLE AND RESPONSIBILITY

Act as a Senior AI Research Scientist, Deep Learning Engineer, Geospatial Data Scientist, and Full-Stack Python Developer.

Your responsibility is to design, develop, test, and deliver a complete, modular, research-oriented, production-quality final-year academic project.

Do not create a basic demonstration or static dashboard. Build a functional AI-powered Urban Heat Island (UHI) analysis and mitigation platform integrating:

1. Satellite Image Semantic Segmentation using U-Net.
2. Land Surface Temperature (LST) Prediction using a Deep Feedforward Neural Network.
3. Geospatial Analysis using Rasterio, GeoPandas, and NumPy.
4. Interactive Urban Heat Island Mitigation Simulation.
5. Visualization of land-cover distributions and predicted temperature changes.

The entire system must work as one integrated application.

---

### 2. PROJECT OBJECTIVE

Build an AI-powered platform that identifies urban land-cover patterns from satellite imagery and predicts Land Surface Temperature (LST) anomalies based on land-cover composition and environmental conditions.

The system must help urban planners understand how converting impervious surfaces into vegetation can potentially reduce urban heat.

The platform must answer these questions:

- How much of a selected urban region is covered by impervious surfaces?
- How much vegetation, bare soil, and water exists?
- What is the predicted LST anomaly for the selected region?
- Which regions have higher predicted heat levels?
- What happens when 10%, 20%, 30%, or 50% of impervious surfaces are converted into vegetation?
- How much temperature reduction does the model predict after an intervention?

Important scientific distinction: The model predicts an LST anomaly, not necessarily absolute air temperature. Do not present simulated model predictions as experimentally proven real-world temperature reductions.

---

# 3. TECHNOLOGY STACK

Use the following technologies:

### Programming and Deep Learning
- Python 3.11+
- PyTorch
- Torchvision
- NumPy
- Pandas
- Scikit-learn

### Geospatial Processing
- Rasterio
- GeoPandas
- Shapely
- PyProj
- GDAL
- Xarray

### Visualization and Dashboard
- Streamlit
- Plotly
- Matplotlib
- Seaborn

### Data Management
- CSV
- GeoTIFF
- NumPy NPZ
- JSON
- GeoJSON

### Development
- VS Code / Antigravity IDE
- Modular Python architecture
- Virtual environment
- Git-ready project structure

Use compatible package versions and provide a requirements.txt file.

---

# 4. SYSTEM ARCHITECTURE

Develop two integrated AI pipelines.

## PHASE 1: SEMANTIC SEGMENTATION USING U-NET

### Objective

Train a U-Net model to identify four land-cover classes from high-resolution satellite imagery.

### Input

Satellite imagery with RGB and Near Infrared (NIR) bands where available.

Support:
- RGB imagery: 3 channels.
- RGB + NIR imagery: 4 channels.

Do not silently discard NIR information. Make the input channel count configurable.

### Output classes

Use the following fixed class mapping:

| Class ID | Land Cover | Description |
|---|---|---|
| 0 | Impervious Surface | Asphalt, roads, concrete, rooftops |
| 1 | Vegetation | Trees, grass, parks, vegetation |
| 2 | Bare Soil | Exposed soil and undeveloped land |
| 3 | Water | Rivers, lakes, ponds |

### Model architecture

Implement a standard encoder-decoder U-Net.

Requirements:
- Configurable input channels.
- Four output classes by default.
- Double convolution blocks.
- ReLU activation.
- Max pooling.
- Transposed convolution.
- Skip connections.
- Batch normalization.
- Dropout where appropriate.
- Softmax probabilities during inference.

Use CrossEntropyLoss for training.

### Dataset preparation

Implement:
- Satellite GeoTIFF loading.
- CRS validation.
- Raster alignment.
- Resampling.
- Image normalization.
- NoData handling.
- 256 × 256 image patch extraction.
- Corresponding segmentation mask extraction.
- Train/validation/test splitting.

Ensure that image and mask transformations preserve spatial alignment.

Avoid random patch-level train/test splitting when patches originate from the same geographic area. Prefer geographic-region-based splitting to reduce spatial data leakage.

### Evaluation

Calculate:
- Mean Intersection over Union (mIoU).
- Per-class IoU.
- Pixel accuracy.
- Precision.
- Recall.
- Confusion matrix.

Save:
- Best model checkpoint.
- Training history.
- Validation metrics.
- Segmentation prediction images.

---

## PHASE 2: LAND SURFACE TEMPERATURE REGRESSION

### Objective

Build a Deep Feedforward Neural Network that predicts the LST anomaly index using land-cover composition and environmental metadata.

### Input features

The model must receive seven features:

**Land-cover distribution:**
1. Impervious surface percentage.
2. Vegetation percentage.
3. Bare soil percentage.
4. Water percentage.

**Environmental metadata:**
5. Elevation.
6. Baseline humidity.
7. Solar irradiance.

The four land-cover percentages must be extracted dynamically from the U-Net segmentation predictions.

The percentages must sum to approximately 100%.

### Regression model

Implement a configurable ANN architecture:

- Input layer: 7 features.
- Hidden layer 1: 128 neurons.
- Hidden layer 2: 64 neurons.
- Hidden layer 3: 32 neurons.
- ReLU activation.
- Batch normalization.
- Dropout.
- Output layer: 1 neuron.

Use MSELoss for regression.

Include:
- Feature scaling.
- Target normalization where appropriate.
- Training and validation.
- Model checkpointing.
- Prediction and inverse scaling.

### Regression evaluation

Calculate:
- RMSE.
- MAE.
- R² score.
- Actual vs predicted scatter plot.
- Residual distribution.

Clearly document the LST anomaly target's units and reference baseline.

---

# 5. SYNTHETIC DATA GENERATION ENGINE

The application must run even when real satellite imagery and labeled datasets are unavailable.

Create a synthetic dataset generator using NumPy and PyTorch.

Generate 500 paired samples with:

- Images: [500, 3, 256, 256].
- Segmentation masks: [500, 256, 256].
- Environmental metadata: [500, 3].
- LST anomaly targets: [500, 1].

Use realistic spatial patterns rather than completely random independent pixels.

Examples:
- Vegetation clusters.
- Connected water regions.
- Impervious urban blocks.
- Bare soil patches.

Generate plausible environmental metadata ranges and temperature relationships.

The synthetic temperature targets should have a scientifically motivated relationship with land-cover composition and environmental variables, with controlled noise.

Ensure that synthetic training and validation data are separated reproducibly.

Add a fixed random seed.

Important: Synthetic data is for testing the software pipeline and demonstrating model functionality. Do not claim that synthetic training establishes real-world prediction accuracy.

---

# 6. INTEGRATED AI PIPELINE

Build a unified pipeline connecting the two models.

The complete execution flow must be:

STEP 1:
Load satellite imagery.

STEP 2:
Perform geospatial preprocessing, normalization, and patch extraction.

STEP 3:
Pass imagery through the trained U-Net.

STEP 4:
Obtain pixel-level segmentation predictions.

STEP 5:
Calculate the percentage of pixels belonging to each of the four land-cover classes.

STEP 6:
Load environmental metadata corresponding to the same geographic grid cell.

STEP 7:
Concatenate the four land-cover percentages and three environmental features.

STEP 8:
Pass the seven-dimensional feature vector into the trained regression ANN.

STEP 9:
Predict the LST anomaly.

STEP 10:
Display the segmentation map, land-cover statistics, and predicted temperature information.

Implement this workflow in a reusable Python class called:

IntegratedUHIPipeline

Provide methods such as:
- load_models()
- preprocess_image()
- segment_image()
- calculate_landcover_percentages()
- prepare_regression_features()
- predict_lst()
- analyze_geographic_region()

Ensure correct tensor dimensions, device management, model evaluation mode, and inference without gradient calculation.

---

# 7. UHI MITIGATION SIMULATION ENGINE

This is one of the most important features of the project.

Develop a dedicated simulation engine.

### Objective

Allow users to simulate urban planning interventions by converting a specified percentage of impervious surface pixels into vegetation.

### Function

Create:

simulate_mitigation(mask, metadata, conversion_percentage, regression_model)

### Functionality

1. Accept an original segmentation mask.
2. Identify all pixels belonging to class 0 (Impervious Surface).
3. Calculate the number of pixels corresponding to the requested conversion percentage.
4. Convert the selected pixels from class 0 into class 1 (Vegetation).
5. Preserve the other land-cover classes.
6. Recalculate the four land-cover percentages.
7. Concatenate the new distribution with the environmental metadata.
8. Feed the modified features into the trained regression ANN.
9. Compare original and modified LST anomaly predictions.
10. Calculate the predicted temperature difference.

### Simulation scenarios

Support:
- 10% conversion.
- 20% conversion.
- 30% conversion.
- 40% conversion.
- 50% conversion.

Allow custom percentages as well.

### Results

Display:

Original LST anomaly.
Modified LST anomaly.
Predicted temperature difference.
Original impervious surface percentage.
New impervious surface percentage.
Original vegetation percentage.
New vegetation percentage.

Use:

Predicted Temperature Reduction = Original Predicted LST Anomaly - Modified Predicted LST Anomaly.

A positive value indicates a predicted reduction.

Make sure the conversion percentage is relative to the original impervious pixels, not the total image pixels.

Use reproducible pixel selection and optionally support spatially contiguous conversion rather than randomly distributed isolated pixels.

Never assume that a predicted decrease is guaranteed. Report it as a model-based scenario estimate.

---

# 8. GEOSPATIAL ANALYSIS AND PROCESSING

Create a reusable geospatial preprocessing module.

Support:
- GeoTIFF reading and writing.
- Raster CRS transformation.
- Raster reprojection.
- Raster alignment.
- Resampling.
- Spatial cropping.
- Raster masking.
- Pixel resolution inspection.
- NoData processing.
- Vector boundary overlays.
- Grid-cell generation.

### Satellite and LST alignment

Support high-resolution RGB/NIR imagery and Landsat-derived thermal products.

Ensure:
- Matching coordinate reference systems.
- Consistent geographic boundaries.
- Correct pixel-to-geographic transformations.
- Consistent grid-cell indexing.
- Explicit resampling methods for categorical and continuous data.

Use nearest-neighbor resampling for categorical segmentation labels and suitable continuous interpolation for imagery or continuous temperature surfaces.

Do not treat interpolation of Landsat LST as an increase in actual thermal spatial resolution.

### Grid-cell aggregation

Create configurable geographic grid cells.

For each cell, calculate:
- Impervious surface percentage.
- Vegetation percentage.
- Bare soil percentage.
- Water percentage.
- Mean LST anomaly.
- Elevation.
- Humidity.
- Solar irradiance.

Export the resulting feature dataset as CSV and GeoJSON.

---

# 9. INTERACTIVE STREAMLIT DASHBOARD

Build a professional, modern, research-oriented dashboard.

The dashboard must have the following sections.

### PAGE 1: PROJECT OVERVIEW

Display:
- Project title.
- Project description.
- System architecture.
- Dataset information.
- Model information.
- Project workflow.

### PAGE 2: SATELLITE IMAGE ANALYSIS

Allow users to:
- Upload GeoTIFF imagery.
- Select RGB/NIR channels.
- Run semantic segmentation.
- View original satellite imagery.
- View predicted segmentation.
- View color-coded land-cover classes.
- View class distribution percentages.

Use this color mapping consistently:
- Impervious Surface: Red.
- Vegetation: Green.
- Bare Soil: Yellow.
- Water: Blue.

Include a legend.

### PAGE 3: LST PREDICTION

Display:
- Elevation input.
- Baseline humidity input.
- Solar irradiance input.
- Four extracted land-cover percentages.
- Predicted LST anomaly.

Allow users to inspect the input features and prediction.

### PAGE 4: UHI MITIGATION SIMULATOR

Provide an interactive slider for the percentage of impervious surfaces to convert into vegetation.

Display:
- Original segmentation map.
- Modified segmentation map.
- Before-and-after land-cover distribution.
- Original predicted LST anomaly.
- Modified predicted LST anomaly.
- Predicted temperature reduction.

Provide comparison charts for 10%, 20%, 30%, 40%, and 50% intervention scenarios.

### PAGE 5: MODEL PERFORMANCE

Display:
- Training and validation loss.
- Segmentation mIoU.
- Per-class IoU.
- Regression RMSE.
- Regression MAE.
- Regression R².
- Confusion matrix.
- Actual vs predicted scatter plot.

Clearly identify whether metrics come from synthetic or real-world test data.

### PAGE 6: GEOSPATIAL VISUALIZATION

Display:
- Geographic boundaries.
- Land-cover distribution maps.
- LST anomaly maps.
- High-heat urban zones.
- Potential vegetation intervention zones.

Support map visualization using appropriate geospatial visualization libraries.

---

# 10. PROJECT DIRECTORY STRUCTURE

Create a clean modular architecture.

Use the following structure:

uhi_mitigation_project/
│
├── app.py
├── requirements.txt
├── README.md
├── config.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── synthetic/
│   └── samples/
│
├── models/
│   ├── __init__.py
│   ├── unet.py
│   ├── regression_ann.py
│   └── checkpoints/
│
├── data_pipeline/
│   ├── __init__.py
│   ├── geospatial_processor.py
│   ├── raster_aligner.py
│   ├── patch_extractor.py
│   ├── dataset.py
│   └── synthetic_generator.py
│
├── training/
│   ├── train_segmentation.py
│   ├── train_regression.py
│   ├── train_integrated.py
│   └── metrics.py
│
├── inference/
│   ├── integrated_pipeline.py
│   └── prediction.py
│
├── simulation/
│   ├── mitigation_engine.py
│   └── scenario_analysis.py
│
├── visualization/
│   ├── segmentation_visualizer.py
│   ├── lst_visualizer.py
│   └── dashboard_charts.py
│
├── utils/
│   ├── device.py
│   ├── preprocessing.py
│   ├── logging_utils.py
│   └── checkpoint_utils.py
│
├── tests/
│   ├── test_unet.py
│   ├── test_regression.py
│   ├── test_geospatial.py
│   └── test_simulation.py
│
└── outputs/
    ├── predictions/
    ├── visualizations/
    └── reports/

You may add additional files if required, but preserve modular separation of responsibilities.

---

# 11. TRAINING AND OPTIMIZATION REQUIREMENTS

Implement separate training pipelines for segmentation and regression.

### Segmentation training

- CrossEntropyLoss.
- AdamW optimizer.
- Learning rate scheduler.
- Mini-batch training.
- Validation loop.
- Early stopping.
- Best checkpoint saving.
- Reproducible random seeds.

### Regression training

- MSELoss.
- AdamW optimizer.
- Feature normalization.
- Validation loop.
- Early stopping.
- Best checkpoint saving.

### Integrated inference

Connect the independently trained models.

Do not train the regression model on arbitrary random land-cover percentages if the production inference pipeline uses U-Net-generated features. Ensure consistency between regression training features and actual U-Net outputs.

During training, use ground-truth segmentation distributions initially as a baseline, but also evaluate regression robustness using predicted segmentation distributions.

Avoid claiming that a jointly optimized model is being trained unless the computational graph and gradient flow actually support it. A two-stage training pipeline is acceptable and preferred for this project.

---

# 12. SCIENTIFIC VALIDATION AND DATA QUALITY

Treat scientific reliability as a core requirement.

Implement:
- Spatially separated training, validation, and test regions.
- Data leakage prevention.
- Input validation.
- Missing-data handling.
- Model uncertainty or confidence reporting where practical.
- Reproducibility.
- Proper evaluation on held-out geographic regions.

Important scientific limitations:

1. Satellite imagery and LST must represent compatible observation dates or clearly documented time windows.
2. Landsat thermal observations have coarser native spatial resolution than high-resolution RGB imagery.
3. Humidity and solar irradiance must have documented temporal and spatial units.
4. Elevation, humidity, and solar irradiance must be normalized consistently.
5. Synthetic labels must not be presented as real satellite ground truth.
6. Model predictions must not be described as causal proof of cooling.
7. A genuine real-world model requires independently validated segmentation masks and measured or satellite-derived LST targets.

Add a data provenance report describing the source, resolution, date, CRS, and preprocessing history of every real dataset.

---

# 13. AUTOMATED TESTING

Create unit tests for:

- U-Net input/output dimensions.
- Regression ANN input/output dimensions.
- Synthetic dataset generation.
- Raster alignment.
- Patch extraction.
- Land-cover percentage calculation.
- Integrated inference.
- Mitigation simulation.
- Checkpoint loading.

Verify the following:

Input image: [B, 3, 256, 256].
U-Net output: [B, 4, 256, 256].
Segmentation mask: [B, 256, 256].
Land-cover features: [B, 4].
Environmental features: [B, 3].
Regression input: [B, 7].
Regression output: [B, 1].

All percentages must be valid and sum to approximately 100%.

The simulation must never convert pixels outside the impervious surface class.

---

# 14. DOCUMENTATION AND FINAL DELIVERABLES

Generate a comprehensive README.md containing:

- Project abstract.
- Problem statement.
- Objectives.
- Existing system limitations.
- Proposed system.
- System architecture.
- Dataset requirements.
- Installation instructions.
- Training instructions.
- Inference instructions.
- Dashboard execution.
- Mitigation simulation instructions.
- Evaluation methodology.
- Scientific limitations.
- Future enhancements.

Also generate:
- requirements.txt.
- Sample synthetic dataset.
- Trained synthetic demonstration checkpoints.
- Model architecture diagrams.
- Training metric plots.
- Sample segmentation outputs.
- Sample mitigation comparison report.
- API/function documentation where applicable.

Include exact commands to install dependencies and launch the project on Windows.

---

# 15. ANTIGRAVITY EXECUTION STRATEGY

Follow these development phases in order.

### PHASE A — PROJECT INITIALIZATION

1. Inspect the existing workspace.
2. Create the project directory structure.
3. Configure the Python environment.
4. Create requirements.txt.
5. Verify installed dependencies.
6. Create the configuration module.

### PHASE B — DATA PIPELINE

1. Implement the synthetic dataset generator.
2. Generate 500 synthetic paired samples.
3. Implement raster loading and preprocessing.
4. Implement image/mask patch extraction.
5. Implement geospatial alignment.
6. Test tensor dimensions and data consistency.

### PHASE C — MODEL DEVELOPMENT

1. Implement U-Net.
2. Implement Regression ANN.
3. Run architecture tests.
4. Verify model input/output dimensions.

### PHASE D — MODEL TRAINING

1. Train U-Net on synthetic demonstration data.
2. Evaluate segmentation metrics.
3. Train ANN using corresponding land-cover and environmental features.
4. Evaluate regression metrics.
5. Save model checkpoints.

### PHASE E — INTEGRATION

1. Connect the U-Net and regression ANN.
2. Implement dynamic land-cover feature extraction.
3. Implement integrated LST inference.
4. Test end-to-end predictions.

### PHASE F — MITIGATION ENGINE

1. Implement impervious-to-vegetation conversion.
2. Implement intervention percentages.
3. Calculate modified land-cover distributions.
4. Run modified ANN predictions.
5. Generate before-and-after comparisons.

### PHASE G — DASHBOARD

1. Build the Streamlit interface.
2. Add image upload.
3. Add segmentation visualization.
4. Add LST prediction.
5. Add mitigation simulation.
6. Add performance visualization.
7. Add geospatial visualization.

### PHASE H — VALIDATION AND DELIVERY

1. Run all automated tests.
2. Fix runtime errors.
3. Test synthetic demonstration workflow.
4. Verify checkpoint loading.
5. Verify dashboard execution.
6. Generate documentation.
7. Provide a final project completion report.

---

# 16. STRICT DEVELOPMENT RULES

Follow these rules throughout development:

1. Do not put the entire application into a single Python file.
2. Do not create placeholder functions for core functionality.
3. Do not use hardcoded model predictions.
4. Do not use fabricated evaluation metrics.
5. Do not silently ignore missing dependencies.
6. Do not assume that all satellite imagery has the same CRS or resolution.
7. Do not mix segmentation class IDs.
8. Do not use regression features with inconsistent ordering.
9. Do not train and evaluate on overlapping geographic regions.
10. Do not claim real-world scientific validity from synthetic data.
11. Keep code modular, reusable, documented, and testable.
12. Handle CPU and CUDA devices automatically.
13. Make all paths configurable.
14. Include meaningful logging and error handling.
15. Ensure the complete synthetic demonstration works without downloading external datasets.

### MOST IMPORTANT EXECUTION INSTRUCTION

Do not just generate a project plan or explain how to build the project.

Actually implement the complete project inside the currently opened Antigravity workspace.

Work phase by phase. Create the files, implement the code, execute the scripts, inspect errors, fix them, and verify the integrated application.

Do not stop after generating the folder structure.

If a real-world dataset is unavailable, complete and test the entire synthetic demonstration pipeline first. Keep the real-data ingestion functionality ready for later dataset integration.

At the end, provide:

- Project implementation summary.
- Files created.
- Models implemented.
- Training results actually obtained.
- Tests executed and their results.
- Exact commands to run the project.
- Any remaining limitations or dataset requirements.

The final deliverable must be a working, integrated, academically defensible Urban Heat Island Analysis and Mitigation platform.