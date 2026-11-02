# Slide-by-Slide Presentation Summary & Faculty Defense Guide

**Project Title:** AI-Powered Urban Heat Island (UHI) Mitigation Using Semantic Segmentation and Deep Learning Regression  
**Presenter:** Kunal (Roll No: 24215215)  
**Project Guide / Mentor:** Dr. Ramesh Chandra Poonia  
**Institution:** Christ University Delhi NCR  
**Academic Year:** 2026–2027  
**Total Slides:** 12 (16:9 Widescreen)  
**Target Duration:** 10–12 Minutes (~50–60 seconds per slide)  

---

## Slide 1: Title Slide
- **Slide Title:** AI-Powered Urban Heat Island (UHI) Mitigation Using Semantic Segmentation and Deep Learning Regression
- **Subtitle:** An Intelligent Satellite-Based Approach for Urban Heat Analysis and Mitigation Simulation
- **Key Content:**
  - Presenter: Kunal (Roll No: 24215215)
  - Department of Computer Science & Engineering, Christ University Delhi NCR
  - Project Guide: Dr. Ramesh Chandra Poonia
  - Academic Year: 2026–2027
- **Visual Assets:**
  - Modern aerial city grid visualization with subtle dual-layer thermal contour overlay.
- **Presenter Focus:**
  - Greet the panel confidently. State the title, problem context, and overarching AI motivation.

---

## Slide 2: Understanding the Urban Heat Island Effect
- **Slide Title:** Understanding the Urban Heat Island Effect
- **Subtitle:** The Physical Phenomenon, Environmental Drivers, and Problem Statement
- **Key Bullets:**
  - **Urban Microclimate Elevation:** Dense metropolitan zones experience surface temperatures 3°C to 8°C higher than surrounding rural baselines.
  - **Thermal Mass Retention:** Concrete, asphalt, and dense roofing absorb shortwave solar radiation during daylight and continuously re-radiate longwave heat overnight.
  - **Vegetative Depletion:** Loss of tree canopies and green spaces drastically reduces natural evaporative cooling and transpiration.
  - **Cascading Impacts:** Severe public health vulnerability, surge in electrical cooling grid demands, and elevated smog and emissions.
  - **Core Problem Statement:** Existing municipal workflows lack an integrated, automated AI tool to map land cover and quantitatively simulate greening interventions prior to capital expenditure.
- **Visual Assets:**
  - High-contrast comparative card: High-Albedo Vegetated Buffer (Evaporative cooling, porous soil, lower LST) vs. Low-Albedo Urban Concrete (High thermal mass, dark asphalt, heat retention).
- **Presenter Focus:**
  - Articulate why this problem matters socially and economically, leading cleanly into the project objectives.

---

## Slide 3: Project Objectives
- **Slide Title:** Project Objectives
- **Subtitle:** Five Core Milestones for Intelligent UHI Assessment and Mitigation
- **Five Key Objectives:**
  1. **Satellite Imagery Ingestion:** Process 4-band multispectral imagery (RGB + NIR) with GPU-accelerated geospatial normalization.
  2. **Semantic Land-Cover Classification:** Deploy a customized U-Net to segment land surfaces into 4 distinct physical classes.
  3. **Land-Cover Distribution Extraction:** Calculate precise percentage fractions for impervious surfaces, vegetation, bare soil, and water bodies.
  4. **LST Anomaly Neural Regression:** Predict localized Land Surface Temperature anomalies using land-cover fractions and atmospheric features.
  5. **Greening Intervention Simulation:** Interactively reassign impervious pixels to vegetation (10%–50%) and compute simulated cooling outcomes.
- **Visual Assets:**
  - 5-stage horizontal workflow card progression with numbered cyan badges.
- **Presenter Focus:**
  - Enumerate the 5 objectives crisply. Evaluators want to verify that every objective has a concrete technical outcome.

---

## Slide 4: Data Sources and Technology Stack
- **Slide Title:** Data Sources and Technology Stack
- **Subtitle:** Multi-Modal Geospatial Data and Production-Grade AI Frameworks
- **Data Sources:**
  - **Copernicus Sentinel-2:** 10m multispectral surface reflectance bands (B02 Blue, B03 Green, B04 Red, B08 NIR).
  - **Landsat-Derived LST:** Thermal infrared sensor observations calibrated for Land Surface Temperature anomalies.
  - **Topographical & Climate Metadata:** Digital Elevation Model (DEM), baseline relative humidity, and incoming solar irradiance ($W/m^2$).
- **Software Stack:**
  - **Language & Runtime:** Python 3.11 with NumPy, SciPy, and Pandas.
  - **Deep Learning:** PyTorch 2.5.1 + CUDA 12.1 on host NVIDIA GeForce RTX 3050 Laptop GPU.
  - **Geospatial Processing:** Rasterio, GeoPandas, GDAL, and Shapely.
  - **Serving & UI:** FastAPI backend, Plotly, Matplotlib, and modern vanilla CSS/JS dashboard.
- **Presenter Focus:**
  - Emphasize CUDA GPU acceleration on host hardware and highlight that both synthetic benchmarking and real Sentinel-2 GeoTIFF data were engineered.

---

## Slide 5: Proposed System Architecture
- **Slide Title:** Proposed System Architecture
- **Subtitle:** End-to-End Three-Tier Workflow: Ingestion, Classification, and Thermal Prediction
- **Workflow Steps:**
  1. **Satellite Data Ingestion:** Multi-band GeoTIFF scenes (Sentinel-2 / synthetic benchmarks).
  2. **Geospatial Preprocessing:** Radiometric normalization, zero-mean unit-variance, sliding-window $256 \times 256$ tile extraction.
  3. **Phase 1: U-Net Semantic Segmentation:** 4-channel input ($256 \times 256 \times 4$) $\rightarrow$ Contracting/Expanding paths $\rightarrow$ 4-class segmentation mask.
  4. **Feature Aggregation:** Histogram calculation of class percentages (Impervious, Vegetation, Soil, Water).
  5. **Environmental Metadata Concatenation:** Merges 4 land-cover percentages with Elevation, Humidity, and Solar Irradiance ($1 \times 7$ vector).
  6. **Phase 2: Regression ANN:** 3-layer deep feedforward neural network $\rightarrow$ Continuous scalar predicted LST anomaly (°C).
  7. **Intervention Feedback Loop:** Selective pixel reassignment $\rightarrow$ Re-inference $\rightarrow$ Real-time mitigated temperature difference.
- **Visual Assets:**
  - High-resolution architecture workflow diagram (`slide_architecture_diagram.png`) embedded directly on the slide.
- **Presenter Focus:**
  - Walk the panel through the three functional blocks: Geospatial Preprocessing, U-Net Segmentation, and ANN Regression.

---

## Slide 6: Phase 1 – Semantic Segmentation Using U-Net
- **Slide Title:** Land-Cover Classification Using U-Net
- **Subtitle:** Pixel-Level Multi-Class Segmentation with Skip-Connection Architecture
- **Core Architecture Details:**
  - **Encoder (Contracting Path):** 4 downsampling stages with dual $3 \times 3$ convolutions, ReLU activations, and $2 \times 2$ max pooling (feature depth: $64 \rightarrow 128 \rightarrow 256 \rightarrow 512$).
  - **Skip Connections:** Direct feature concatenation between corresponding encoder and decoder stages to preserve fine-grained building edges and road boundaries.
  - **Decoder (Expanding Path):** 4 upsampling stages with transposed convolutions restoring spatial resolution to $256 \times 256 \times 4$.
  - **4 Standard Land-Cover Classes:**
    - Class 0: **Impervious Surfaces** (Red - `#EF4444`)
    - Class 1: **Vegetation** (Green - `#10B981`)
    - Class 2: **Bare Soil** (Yellow - `#F59E0B`)
    - Class 3: **Water Bodies** (Blue - `#3B82F6`)
- **Visual Assets:**
  - Side-by-side comparative visualization of RGB satellite input and predicted 4-class segmentation mask with colored class legends.
- **Presenter Focus:**
  - Clearly explain why U-Net was chosen: skip connections solve the spatial resolution loss problem inherent in standard CNNs.

---

## Slide 7: Phase 2 – LST Prediction Using ANN
- **Slide Title:** Land Surface Temperature Prediction
- **Subtitle:** Deep Feedforward Neural Regression with 7 Multi-Modal Inputs
- **Feature Specification (7-Dimensional Input Vector):**
  - **4 Land-Cover Percentages:** Impervious Surface %, Vegetation %, Bare Soil %, Water %.
  - **3 Environmental Metadata:** Elevation (m), Baseline Humidity (%), Solar Irradiance ($W/m^2$).
- **Network Topology:**
  - **Input Layer:** 7 features with Batch Normalization.
  - **Hidden Layer 1:** 128 Neurons, LeakyReLU ($\alpha = 0.1$), BatchNorm, Dropout ($p = 0.20$).
  - **Hidden Layer 2:** 64 Neurons, LeakyReLU, BatchNorm, Dropout ($p = 0.20$).
  - **Hidden Layer 3:** 32 Neurons, LeakyReLU, BatchNorm.
  - **Output Layer:** 1 Linear Neuron $\rightarrow$ Predicted LST Anomaly ($\Delta T$ in °C).
- **Physical Rationale:**
  - Impervious density correlates positively with thermal retention; vegetative coverage produces evaporative cooling; solar irradiance and elevation modulate the baseline equilibrium.
- **Presenter Focus:**
  - Highlight the regularization choices (Batch Normalization and Dropout) that ensure robust generalizability.

---

## Slide 8: Integrated AI Workflow
- **Slide Title:** Integration of Both Deep Learning Models
- **Subtitle:** Synchronous Execution Pipeline from Raw Pixels to Thermal Decision Support
- **6-Step Integration Pipeline:**
  - Step 1: Input multispectral satellite imagery tile ($256 \times 256 \times 4$).
  - Step 2: GPU U-Net inference outputs discrete pixel classification matrix.
  - Step 3: Spatial histogram computation calculates exact percentage fractions for all 4 classes.
  - Step 4: Concatenate land-cover fractions with atmospheric and topographical metadata into a unified 7-feature tensor.
  - Step 5: Regression ANN computes the baseline thermal anomaly in $< 5$ milliseconds.
  - Step 6: Web dashboard renders the segmentation map, land-cover pie chart, and temperature anomaly gauge.
- **Key Architectural Benefit:**
  - Decoupled modular design allows either model to be upgraded or fine-tuned independently without re-training the entire pipeline.
- **Presenter Focus:**
  - Reiterate that while the models are structurally independent, they are coupled through an automated execution pipeline.

---

## Slide 9: UHI Mitigation Simulation
- **Slide Title:** Simulating Urban Greening Interventions
- **Subtitle:** Interactive What-If Greening Scenarios for Data-Driven Urban Policy
- **Simulation Mechanism:**
  - User selects targeted greening conversion percentage: **10%, 20%, 30%, 40%, or 50%**.
  - Algorithm indexes all Class 0 (impervious concrete) pixels and reclassifies a designated proportion to Class 1 (vegetation).
  - Recalculates modified land-cover distribution vector: $[P_{imp}', P_{veg}', P_{soil}, P_{water}]$.
  - Re-executes the regression ANN to obtain the modified anomaly: $\text{LST}_{mitigated}$.
  - Computes simulated cooling outcome:
    $$\Delta T_{\text{reduction}} = \text{LST}_{\text{baseline}} - \text{LST}_{\text{mitigated}}$$
- **Scientific Caveat:**
  - Explicitly defined as an empirical model simulation based on trained correlations, not a guaranteed microclimate forecast.
- **Visual Assets:**
  - Before-and-after segmentation visualization showing conversion of concrete zones into simulated urban parks and green roofs.
- **Presenter Focus:**
  - Highlight this slide as the project's primary practical deliverable for urban planners and city municipalities.

---

## Slide 10: Model Performance and Evaluation
- **Slide Title:** Model Performance and Evaluation
- **Subtitle:** Empirical Metrics on Synthetic Benchmarks and Real Sentinel-2 Imagery
- **Section A: Phase 1 Segmentation (U-Net):**
  - **Synthetic Benchmark (500 scenes):**
    - Mean IoU: **0.9947** | Pixel Accuracy: **99.74%**
    - Impervious IoU: 0.9931 | Vegetation IoU: 0.9962 | Soil IoU: 0.9950 | Water IoU: 0.9945
  - **Real Copernicus Sentinel-2 Tiles (144 patches):**
    - Mean IoU: **0.6899** | Pixel Accuracy: **91.98%**
    - Demonstrates strong generalization across noisy real-world terrain.
- **Section B: Phase 2 Regression (ANN):**
  - **Synthetic Benchmark:**
    - RMSE: **0.4596 °C** | MAE: **0.3649 °C** | $R^2$ Score: **0.9167**
  - **Real Sentinel-2 Tiles:**
    - RMSE: **1.1800 °C** | MAE: **1.0903 °C**
    - Confirms that the ANN accurately models the nonlinear thermal penalty of impervious coverage.
- **Visual Assets:**
  - High-contrast 4-metric bar chart (`slide_model_metrics_chart.png`) comparing synthetic vs. real Sentinel-2 performance.
- **Presenter Focus:**
  - Strictly present true empirical results without fabrication. Clearly explain the difference between controlled synthetic training and real satellite testing.

---

## Slide 11: Applications and Future Enhancements
- **Slide Title:** Applications and Future Enhancements
- **Subtitle:** Real-World Deployments and Research Scalability Roadmap
- **Municipal Applications:**
  - **Smart City Planning:** Pinpoint thermal hotspots for prioritizing urban tree canopies and green corridor construction.
  - **Urban Heat Risk Management:** Identify vulnerable demographic wards exposed to extreme surface temperatures during heatwaves.
  - **Building Code Optimization:** Quantify cool roof and permeable pavement benefits for municipal green compliance.
  - **Microclimate Monitoring:** Continuous tracking of urban vegetation health versus impervious expansion.
- **Future Research Scope:**
  - **Live Cloud Ingestion:** Integrate Google Earth Engine API for automated multi-temporal satellite scene ingestion.
  - **3D Urban Morphology:** Incorporate LiDAR canopy height models to account for urban canyon solar shading.
  - **Atmospheric Physics Coupling:** Incorporate computational fluid dynamics (CFD) wind circulation and evapotranspiration models.
- **Presenter Focus:**
  - Show breadth of vision by detailing how this work can be extended to cloud-scale satellite monitoring and 3D urban physics.

---

## Slide 12: Conclusion and Q&A
- **Slide Title:** Conclusion
- **Subtitle:** Summary of Key Contributions and Acknowledgements
- **Core Summary:**
  - Demonstrated an integrated deep learning framework combining pixel-level semantic segmentation (U-Net) with multi-variable neural regression (ANN) to assess and mitigate Urban Heat Islands.
  - Delivered an interactive web simulation platform enabling urban planners to model temperature reduction across greening intensities from 10% to 50%.
- **Key Project Contributions:**
  1. Automated 4-band satellite image preprocessing and patch extraction.
  2. High-accuracy 4-class semantic segmentation ($91.98\%$ real pixel accuracy).
  3. Validated regression ANN modeling surface temperature anomalies.
  4. Real-time scenario simulation engine with modern interactive visualization.
- **Closing & Acknowledgments:**
  - Sincere gratitude to project guide **Dr. Ramesh Chandra Poonia** and the evaluators of **Christ University Delhi NCR**.
  - Invitation to live demonstration and interactive Q&A session.
- **Presenter Focus:**
  - Conclude with poise. Offer to run a live demonstration of the web app on `http://localhost:8080`.

---

## Anticipated Faculty Q&A Preparation

| Question | Recommended Answer |
|---|---|
| **Q1: Why use U-Net instead of Mask R-CNN or YOLO?** | U-Net is designed specifically for dense pixel-level semantic segmentation where every pixel belongs to an environmental class (impervious, vegetation, soil, water), rather than detecting discrete bounding-box objects. Its skip connections preserve critical spatial boundaries of rooftops and roads. |
| **Q2: Why not predict LST directly from the satellite image end-to-end?** | An end-to-end image-to-LST model acts as an uninterpretable black box. By first extracting physical land-cover percentages, we create an interpretable intermediate representation. This allows urban planners to explicitly simulate greening interventions by modifying land-cover proportions and seeing the direct simulated thermal response. |
| **Q3: How do you account for seasonal temperature variations?** | Our ANN incorporates solar irradiance ($W/m^2$) and relative baseline humidity as explicit input features, which reflect seasonal solar angles and atmospheric conditions, rather than assuming static climate baselines. |
| **Q4: Were real satellite images tested?** | Yes. We integrated Copernicus Sentinel-2 Level-2A 4-band GeoTIFF imagery (144 patches) aligned with Landsat LST observations, achieving 91.98% pixel accuracy and 1.18°C RMSE, alongside our 500-scene synthetic benchmark. |
| **Q5: Can this tool run on a standard laptop?** | Yes. Our inference pipeline is optimized with PyTorch. It runs efficiently on consumer GPUs (such as the NVIDIA RTX 3050 Laptop GPU in our testing environment) and requires under 50ms per $256 \times 256$ tile. |
