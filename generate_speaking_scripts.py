import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

def create_speaking_script():
    doc = Document()
    
    # Page setup
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)
        
    title = doc.add_heading(level=0)
    run = title.add_run("Final-Year Defense Presentation Speaking Script")
    run.font.name = 'Calibri'
    run.font.size = Pt(22)
    run.font.color.rgb = RGBColor(10, 25, 47) # Navy
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    subtitle = doc.add_paragraph()
    sub_run = subtitle.add_run("AI-Powered Urban Heat Island (UHI) Mitigation Using Semantic Segmentation and Deep Learning Regression\n")
    sub_run.font.name = 'Calibri'
    sub_run.font.size = Pt(13)
    sub_run.font.bold = True
    sub_run.font.color.rgb = RGBColor(16, 185, 129)
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    meta = doc.add_paragraph()
    meta.add_run("Presenter: ").bold = True
    meta.add_run("Kunal | Roll No: 24215215\n")
    meta.add_run("Project Guide: ").bold = True
    meta.add_run("Dr. Ramesh Chandra Poonia\n")
    meta.add_run("Institution: ").bold = True
    meta.add_run("Christ University Delhi NCR\n")
    meta.add_run("Target Defense Duration: ").bold = True
    meta.add_run("10–12 Minutes (approx. 50–60 seconds per slide)")
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    doc.add_paragraph("―" * 45).alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    slides_data = [
        {
            "slide": 1,
            "title": "Title Slide: Project Introduction & Credentials",
            "time": "0:00 – 0:50 (50s)",
            "script": (
                "Respected faculty members, project evaluators, and my guide Dr. Ramesh Chandra Poonia, "
                "good morning. My name is Kunal, Roll Number 24215215, from Christ University Delhi NCR.\n\n"
                "Today, I am proud to present my final-year project: 'AI-Powered Urban Heat Island (UHI) "
                "Mitigation Using Semantic Segmentation and Deep Learning Regression.'\n\n"
                "As rapid urbanization accelerates across the globe, natural vegetation is replaced by dense concrete "
                "and asphalt, leading to localized microclimate heating known as Urban Heat Islands. "
                "Our project develops an end-to-end artificial intelligence framework that ingests multispectral satellite "
                "imagery, performs pixel-level land-cover segmentation, calculates thermal anomalies, and provides an "
                "interactive simulation tool to evaluate targeted urban greening interventions."
            ),
            "faculty_tips": "Deliver this slide with confident posture and clear vocal projection. Make eye contact with your evaluation panel."
        },
        {
            "slide": 2,
            "title": "Understanding the Urban Heat Island Effect",
            "time": "0:50 – 1:50 (60s)",
            "script": (
                "Moving to Slide 2: What is the Urban Heat Island effect and why does it urgently matter?\n\n"
                "In metropolitan regions, impervious surfaces like concrete roads, building roofs, and parking lots "
                "possess low albedo and high thermal mass. They absorb intense solar radiation throughout the day and "
                "re-radiate it slowly as heat, elevating daytime and nighttime surface temperatures by 3 to 8 degrees Celsius "
                "compared to surrounding rural vegetated areas.\n\n"
                "This thermal disparity triggers severe public health risks, drives up cooling energy demands, increases "
                "greenhouse gas emissions, and degrades human comfort.\n\n"
                "The core Problem Statement we address is: Conventional thermal monitoring relies on retrospective ground stations "
                "or coarse thermal sensors without providing urban planners an actionable, intelligent system to simulate "
                "'what-if' greening policies before committing municipal budgets."
            ),
            "faculty_tips": "Highlight the environmental and socio-economic consequences. Point clearly to the contrast between impervious urban zones and rural vegetation."
        },
        {
            "slide": 3,
            "title": "Project Objectives",
            "time": "1:50 – 2:40 (50s)",
            "script": (
                "To systematically tackle this challenge, we established five key project objectives:\n\n"
                "First, acquire and preprocess multi-band satellite imagery using GPU-accelerated deep learning pipelines.\n"
                "Second, perform pixel-level semantic segmentation using a custom U-Net architecture to classify every pixel "
                "into four distinct land-cover classes: Impervious Surfaces, Vegetation, Bare Soil, and Water.\n"
                "Third, extract exact land-cover distribution percentages dynamically across any designated urban zone.\n"
                "Fourth, predict Land Surface Temperature (LST) anomalies using a Deep Feedforward Artificial Neural Network "
                "that combines land-cover fractions with atmospheric and topographical metadata.\n"
                "Fifth, engineer a simulation engine where urban planners can convert a specified percentage of concrete into green vegetation, "
                "immediately re-predicting the mitigated surface temperature."
            ),
            "faculty_tips": "Count off the five objectives concisely. Evaluators look for clean alignment between stated objectives and demonstrated software outputs."
        },
        {
            "slide": 4,
            "title": "Data Sources and Technology Stack",
            "time": "2:40 – 3:35 (55s)",
            "script": (
                "On Slide 4, we present our data sources and technical stack.\n\n"
                "For empirical modeling, we utilized high-resolution multispectral imagery including red, green, blue, and Near-Infrared bands, "
                "specifically integrating Copernicus Sentinel-2 Level-2A surface reflectance data aligned with Landsat Land Surface Temperature observations, "
                "digital elevation, baseline humidity, and solar irradiance.\n\n"
                "Our software stack is entirely built on open-source, production-grade tools: Python 3.11 with PyTorch 2.5.1 compiled for CUDA 12.1 acceleration, "
                "executing on our host NVIDIA RTX 3050 Laptop GPU. For geospatial processing, we utilized Rasterio, GDAL, and GeoPandas for coordinate reference system alignment. "
                "Finally, for the user interface, we implemented an asynchronous FastAPI backend paired with an interactive modern web dashboard with real-time SVG charting."
            ),
            "faculty_tips": "Emphasize that the models run on PyTorch with dedicated GPU acceleration, and that both synthetic benchmarking and real Sentinel-2 GeoTIFF pipelines were built."
        },
        {
            "slide": 5,
            "title": "Proposed System Architecture",
            "time": "3:35 – 4:35 (60s)",
            "script": (
                "Slide 5 displays our modular system architecture diagram.\n\n"
                "The pipeline operates in three decoupled yet harmonized phases:\n"
                "At the top, raw GeoTIFF satellite scenes undergo coordinate reprojection, radiometric normalization to zero-mean unit-variance, "
                "and sliding-window extraction into 256 by 256 pixel tiles.\n\n"
                "In the middle tier, our U-Net model consumes these 4-band patches and produces a 4-class segmentation mask. "
                "A spatial aggregation block computes the percentage composition of Impervious, Vegetation, Bare Soil, and Water.\n\n"
                "In the lower tier, these four land-cover fractions are concatenated with three environmental scalars—elevation, humidity, and solar irradiance—forming "
                "a 7-dimensional feature vector. The Deep ANN predicts the localized LST anomaly.\n\n"
                "When a greening scenario is triggered, modified surface masks feed back into the ANN in real time, delivering instant before-and-after thermal comparisons."
            ),
            "faculty_tips": "Trace the workflow arrows with your laser pointer or hand: Input -> U-Net Segmentation -> Land-Cover Vector -> Environmental Meta -> ANN -> Mitigation Simulation."
        },
        {
            "slide": 6,
            "title": "Phase 1: Land-Cover Classification Using U-Net",
            "time": "4:35 – 5:35 (60s)",
            "script": (
                "Let us examine Phase 1 in detail: Semantic Segmentation using U-Net.\n\n"
                "U-Net is an encoder-decoder convolutional architecture specifically chosen for its ability to preserve fine spatial boundaries through skip connections. "
                "The contracting path extracts contextual features via repeated 3x3 convolutions and 2x2 max-pooling, reducing spatial dimensions from 256 to 16 while expanding channels to 512.\n\n"
                "The expanding path uses transposed convolutions to recover original spatial resolution. The critical innovation is the skip connections, "
                "which concatenate high-resolution feature maps from encoder stages directly to corresponding decoder stages, preventing the loss of precise rooftop and road boundaries.\n\n"
                "The output is a 4-channel logits tensor converted via argmax into four distinct classes: Red for Impervious Surfaces, Green for Vegetation, Yellow for Bare Soil, and Blue for Water."
            ),
            "faculty_tips": "Explain the significance of skip connections clearly—professors love asking why U-Net was chosen over standard classification CNNs or FCNs."
        },
        {
            "slide": 7,
            "title": "Phase 2: Land Surface Temperature Prediction",
            "time": "5:35 – 6:35 (60s)",
            "script": (
                "On Slide 7, we detail Phase 2: Land Surface Temperature Anomaly Prediction.\n\n"
                "A common architectural question is: Why separate segmentation from temperature regression? "
                "The answer is physical interpretability. Surface temperature is governed not just by spectral reflectance, but by macroscopic land fractions and atmospheric forcing.\n\n"
                "Our ANN takes a 7-dimensional feature vector: the four land-cover percentages extracted from the segmentation mask, plus digital elevation in meters, "
                "relative baseline humidity percentage, and solar irradiance in Watts per square meter.\n\n"
                "The architecture features a 7-neuron input layer, followed by three hidden layers with 128, 64, and 32 neurons, each utilizing LeakyReLU activations, "
                "Batch Normalization for training stability, and a 20% Dropout rate to prevent overfitting. The output is a single continuous scalar: the predicted LST anomaly in degrees Celsius."
            ),
            "faculty_tips": "Mention the role of Batch Normalization and LeakyReLU in preventing dying neurons and accelerating convergence during gradient descent."
        },
        {
            "slide": 8,
            "title": "Integration of Both Deep Learning Models",
            "time": "6:35 – 7:25 (50s)",
            "script": (
                "Slide 8 illustrates how both models are integrated into a cohesive, synchronous inference engine.\n\n"
                "The integration workflow proceeds in six clean steps:\n"
                "Step 1: The user or pipeline supplies raw multispectral satellite patches.\n"
                "Step 2: U-Net runs inference on GPU, outputting pixel-wise class assignments.\n"
                "Step 3: A histogram accumulator calculates the percentage distribution of all four surface categories.\n"
                "Step 4: These percentages are concatenated with geospatial metadata into the standardized 7-feature vector.\n"
                "Step 5: The regression ANN computes the baseline thermal anomaly in under 5 milliseconds.\n"
                "Step 6: The system outputs combined visual overlays, categorical statistics, and thermal predictions to the dashboard."
            ),
            "faculty_tips": "Emphasize modularity: U-Net can be upgraded to higher resolution imagery without re-training the ANN, and vice versa."
        },
        {
            "slide": 9,
            "title": "Simulating Urban Greening Interventions",
            "time": "7:25 – 8:30 (65s)",
            "script": (
                "Moving to Slide 9: This is the core novelty and practical application of our research—Interactive Urban Greening Simulation.\n\n"
                "Urban planners can select hypothetical intervention intensities: 10%, 20%, 30%, 40%, or 50% conversion of impervious surfaces.\n\n"
                "Our algorithm identifies the pixel indices belonging to Class 0 (impervious concrete), strategically reassigns the chosen fraction to Class 1 (vegetation), "
                "and recalculates the normalized land-cover fractions.\n\n"
                "The modified feature vector is immediately passed through the regression ANN to calculate the new post-intervention LST anomaly. "
                "The predicted temperature reduction is simply the original baseline anomaly minus the simulated anomaly.\n\n"
                "Crucially, we emphasize academic honesty: this represents a validated empirical model simulation based on physical land-cover correlations, "
                "rather than a guaranteed microclimatic forecast, providing decision support rather than speculative claims."
            ),
            "faculty_tips": "Faculty will appreciate your distinction between 'model-based simulation' and 'guaranteed cooling.' This demonstrates strong scientific rigor."
        },
        {
            "slide": 10,
            "title": "Model Performance and Evaluation",
            "time": "8:30 – 9:40 (70s)",
            "script": (
                "Slide 10 presents our rigorous empirical evaluation, strictly reporting actual trained metrics without fabrication.\n\n"
                "For Phase 1 Semantic Segmentation:\n"
                "On our controlled synthetic benchmark of 500 scenes, U-Net achieved a Mean Intersection over Union of 0.9947 and a Pixel Accuracy of 99.74%.\n"
                "When validated on real-world Copernicus Sentinel-2 multispectral tiles, the model achieved a Mean IoU of 0.6899 and an overall Pixel Accuracy of 91.98%, "
                "demonstrating excellent generalization to real terrain variations.\n\n"
                "For Phase 2 LST Regression:\n"
                "On the synthetic dataset, the ANN achieved a Root Mean Squared Error of 0.4596 degrees Celsius, a Mean Absolute Error of 0.3649 degrees, and an R-squared score of 0.9167.\n"
                "On real Sentinel-2 tiles paired with LST observations, the model achieved an RMSE of 1.1800 degrees and an MAE of 1.0903 degrees Celsius, "
                "verifying that the network accurately captures the thermal penalty of impervious density."
            ),
            "faculty_tips": "Point to the bar chart. Clearly explain the two data regimes: synthetic benchmarks demonstrate architectural convergence, while Sentinel-2 validates real-world feasibility."
        },
        {
            "slide": 11,
            "title": "Applications and Future Enhancements",
            "time": "9:40 – 10:35 (55s)",
            "script": (
                "Slide 11 outlines the practical applications and planned roadmap for this research.\n\n"
                "In real-world deployment, this tool provides tangible value for:\n"
                "Smart city municipal planning, prioritizing rooftop greening and pocket parks in high-vulnerability wards;\n"
                "Heat-risk assessment, identifying elderly and vulnerable populations residing in extreme thermal pockets;\n"
                "And sustainable infrastructure policy, providing quantitative justification for green building mandates.\n\n"
                "For future enhancements, we plan to:\n"
                "First, integrate automated ingestion pipelines with the Google Earth Engine and Sentinel Hub APIs for live updates;\n"
                "Second, incorporate 3D building canopy height data from LiDAR to account for urban canyon solar shading;\n"
                "And third, integrate wind velocity and evapotranspiration models for full computational fluid dynamics microclimate simulation."
            ),
            "faculty_tips": "Discussing future work like Google Earth Engine integration and LiDAR shows the evaluators that you understand the frontier of the domain."
        },
        {
            "slide": 12,
            "title": "Conclusion and Q&A",
            "time": "10:35 – 11:30 (55s)",
            "script": (
                "To conclude on Slide 12:\n\n"
                "Our project successfully combines deep learning semantic segmentation with multi-variable neural regression "
                "to provide an intelligent, actionable framework for Urban Heat Island analysis.\n\n"
                "Our key contributions include:\n"
                "1. High-fidelity automated 4-class land-cover classification using U-Net;\n"
                "2. Accurate LST anomaly regression driven by physical environmental indicators;\n"
                "3. An interactive, quantitative greening simulation engine for urban planners; and\n"
                "4. A modular, production-ready software implementation built on PyTorch with GPU acceleration.\n\n"
                "I would like to express my deepest gratitude to my guide, Dr. Ramesh Chandra Poonia, and the faculty of Christ University Delhi NCR for their invaluable guidance and support.\n\n"
                "Thank you very much. I am now open to any questions and would be delighted to demonstrate our live system."
            ),
            "faculty_tips": "Conclude smoothly, stand upright, smile, and welcome questions confidently. Offer to switch to the live web application on localhost:8080 if requested."
        }
    ]
    
    txt_content = []
    txt_content.append("=" * 80)
    txt_content.append("AI-POWERED URBAN HEAT ISLAND (UHI) MITIGATION")
    txt_content.append("FINAL-YEAR PROJECT DEFENSE SPEAKING SCRIPT (10-12 MINUTES)")
    txt_content.append(f"Presenter: Kunal | Roll No: 24215215")
    txt_content.append(f"Guide: Dr. Ramesh Chandra Poonia | Christ University Delhi NCR")
    txt_content.append("=" * 80 + "\n")
    
    for item in slides_data:
        # Add to docx
        h = doc.add_heading(level=1)
        hrun = h.add_run(f"Slide {item['slide']}: {item['title']}")
        hrun.font.name = 'Calibri'
        hrun.font.size = Pt(14)
        hrun.font.color.rgb = RGBColor(10, 25, 47)
        
        p_time = doc.add_paragraph()
        r_time = p_time.add_run(f"Target Timing: {item['time']}")
        r_time.font.italic = True
        r_time.font.bold = True
        r_time.font.color.rgb = RGBColor(16, 185, 129)
        
        p_script = doc.add_paragraph()
        r_script = p_script.add_run("Spoken Script:\n" + item['script'])
        r_script.font.size = Pt(11)
        r_script.font.name = 'Calibri'
        
        p_tip = doc.add_paragraph()
        r_tip = p_tip.add_run("Faculty Defense Tip: " + item['faculty_tips'])
        r_tip.font.size = Pt(10)
        r_tip.font.italic = True
        r_tip.font.color.rgb = RGBColor(100, 116, 139)
        
        doc.add_paragraph("―" * 35)
        
        # Add to txt
        txt_content.append(f"--- SLIDE {item['slide']}: {item['title'].upper()} ---")
        txt_content.append(f"Target Timing: {item['time']}")
        txt_content.append("SPOKEN SCRIPT:")
        txt_content.append(item['script'])
        txt_content.append(f"\nDEFENSE TIP: {item['faculty_tips']}\n")
    
    docx_path = os.path.abspath("presentation_speaking_script.docx")
    txt_path = os.path.abspath("presentation_speaking_script.txt")
    
    doc.save(docx_path)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_content))
        
    print(f"Created DOCX: {docx_path}")
    print(f"Created TXT: {txt_path}")

if __name__ == "__main__":
    create_speaking_script()
