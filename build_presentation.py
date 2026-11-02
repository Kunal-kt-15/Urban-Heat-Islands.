"""
PowerPoint Presentation Generator for Final Year Academic Project.
Project: AI-Powered Urban Heat Island (UHI) Mitigation Using Semantic Segmentation and Deep Learning Regression
Presenter: Kunal (Roll No: 24215215)
Guide: Dr. Ramesh Chandra Poonia
Institution: Christ University Delhi NCR
"""

from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

import config

# Color Palette Constants
COLOR_BG_NAVY = RGBColor(11, 19, 43)        # #0B132B Dark Navy
COLOR_CARD_DARK = RGBColor(15, 23, 42)      # #0F172A Slate Card
COLOR_CARD_BORDER = RGBColor(51, 65, 85)    # #334155
COLOR_EMERALD = RGBColor(16, 185, 129)      # #10B981 Forest Green / Emerald
COLOR_CYAN = RGBColor(56, 189, 248)         # #38BDF8 Cyan Accent
COLOR_WHITE = RGBColor(255, 255, 255)       # Pure White
COLOR_LIGHT_GREY = RGBColor(203, 213, 225)  # #CBD5E1 Light Grey
COLOR_MUTED = RGBColor(148, 163, 184)       # #94A3B8 Muted Slate
COLOR_CRIMSON = RGBColor(230, 57, 70)       # #E63946 Red Urban
COLOR_GOLD = RGBColor(233, 196, 106)        # #E9C46A Yellow Soil


def apply_dark_background(slide):
    """Sets widescreen dark navy background."""
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = COLOR_BG_NAVY


def add_slide_header(slide, title_text, category_tag="ACADEMIC PROJECT DEFENSE | FINAL YEAR"):
    """Adds unified slide header with title and category tag."""
    # Tag
    tag_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11), Inches(0.3))
    tf_tag = tag_box.text_frame
    tf_tag.word_wrap = True
    p_tag = tf_tag.paragraphs[0]
    p_tag.text = category_tag.upper()
    p_tag.font.name = "Arial"
    p_tag.font.size = Pt(9.5)
    p_tag.font.bold = True
    p_tag.font.color.rgb = COLOR_CYAN

    # Title
    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.65), Inches(11.7), Inches(0.65))
    tf_title = title_box.text_frame
    tf_title.word_wrap = True
    p_title = tf_title.paragraphs[0]
    p_title.text = title_text
    p_title.font.name = "Arial"
    p_title.font.size = Pt(22)
    p_title.font.bold = True
    p_title.font.color.rgb = COLOR_WHITE


def add_content_card(slide, left, top, width, height, border_color=COLOR_CARD_BORDER, bg_color=COLOR_CARD_DARK):
    """Adds a modern rounded rectangle container card."""
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg_color
    shape.line.color.rgb = border_color
    shape.line.width = Pt(1.2)
    return shape


def add_bullet_points(tf, points, font_size=12, text_color=COLOR_LIGHT_GREY, bold_title=True):
    """Populates text frame with crisp, readable bullet points."""
    tf.word_wrap = True
    has_initial_text = len(tf.paragraphs[0].text.strip()) > 0
    for i, pt_text in enumerate(points):
        p = tf.add_paragraph() if (i > 0 or has_initial_text) else tf.paragraphs[0]
        p.space_after = Pt(6)
        p.font.name = "Arial"
        p.font.size = Pt(font_size)

        if ":" in pt_text and bold_title:
            parts = pt_text.split(":", 1)
            run1 = p.add_run()
            run1.text = "• " + parts[0] + ":"
            run1.font.bold = True
            run1.font.color.rgb = COLOR_WHITE
            run2 = p.add_run()
            run2.text = parts[1]
            run2.font.bold = False
            run2.font.color.rgb = text_color
        else:
            run = p.add_run()
            run.text = "• " + pt_text
            run.font.bold = False
            run.font.color.rgb = text_color


def build_powerpoint():
    prs = Presentation()
    # 16:9 Widescreen dimensions
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # =========================================================================
    # SLIDE 1: TITLE SLIDE
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    apply_dark_background(s1)

    # Title Background Image
    bg_img_path = config.VISUALIZATIONS_DIR / "slide_title_bg.png"
    if bg_img_path.exists():
        s1.shapes.add_picture(str(bg_img_path), Inches(0), Inches(0), Inches(13.333), Inches(7.5))

    # Dark overlay box for contrast
    overlay = add_content_card(s1, Inches(0.8), Inches(0.7), Inches(11.7), Inches(6.1), border_color=COLOR_EMERALD)

    # Category Pill
    tag_box = s1.shapes.add_textbox(Inches(1.2), Inches(1.0), Inches(10), Inches(0.4))
    p_tag = tag_box.text_frame.paragraphs[0]
    p_tag.text = "B.TECH FINAL YEAR PROJECT DEFENSE | 2026–2027"
    p_tag.font.name = "Arial"
    p_tag.font.size = Pt(11)
    p_tag.font.bold = True
    p_tag.font.color.rgb = COLOR_CYAN

    # Title & Subtitle
    title_box = s1.shapes.add_textbox(Inches(1.2), Inches(1.4), Inches(10.8), Inches(1.8))
    tf_t = title_box.text_frame
    tf_t.word_wrap = True
    p_t = tf_t.paragraphs[0]
    p_t.text = "AI-Powered Urban Heat Island (UHI) Mitigation\nUsing Semantic Segmentation and Deep Learning Regression"
    p_t.font.name = "Arial"
    p_t.font.size = Pt(26)
    p_t.font.bold = True
    p_t.font.color.rgb = COLOR_WHITE

    p_sub = tf_t.add_paragraph()
    p_sub.text = "An Intelligent Satellite-Based Framework for Urban Land-Cover Analysis and Greening Simulation"
    p_sub.space_before = Pt(8)
    p_sub.font.name = "Arial"
    p_sub.font.size = Pt(13)
    p_sub.font.color.rgb = COLOR_EMERALD

    # Metadata Grid (Presenter & Institutional details)
    meta_box = s1.shapes.add_textbox(Inches(1.2), Inches(3.7), Inches(10.8), Inches(2.7))
    tf_m = meta_box.text_frame
    tf_m.word_wrap = True

    details = [
        ("Presented by", "Kunal"),
        ("Roll Number", "24215215"),
        ("Department", "Department of Computer Science & Engineering / Data Science"),
        ("Institution", "Christ University Delhi NCR"),
        ("Project Guide", "Dr. Ramesh Chandra Poonia"),
        ("Academic Year", "2026–2027")
    ]

    for label, val in details:
        p = tf_m.add_paragraph()
        p.space_after = Pt(4)
        run1 = p.add_run()
        run1.text = f"{label}: "
        run1.font.bold = True
        run1.font.size = Pt(11.5)
        run1.font.color.rgb = COLOR_MUTED
        run2 = p.add_run()
        run2.text = val
        run2.font.bold = True
        run2.font.size = Pt(11.5)
        run2.font.color.rgb = COLOR_WHITE

    # =========================================================================
    # SLIDE 2: INTRODUCTION & PROBLEM STATEMENT
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    apply_dark_background(s2)
    add_slide_header(s2, "Understanding the Urban Heat Island (UHI) Effect", "Introduction & Problem Statement")

    # Left Card: Phenomena & Drivers
    card_left = add_content_card(s2, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.3))
    tb_l = s2.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(5.2), Inches(4.8))
    tf_l = tb_l.text_frame
    tf_l.word_wrap = True

    p_h1 = tf_l.paragraphs[0]
    p_h1.text = "The Urban Microclimate Challenge"
    p_h1.font.bold = True
    p_h1.font.size = Pt(15)
    p_h1.font.color.rgb = COLOR_WHITE
    p_h1.space_after = Pt(12)

    intro_points = [
        "Thermal Disparity: Densely built metropolitan areas exhibit significantly higher surface temperatures than surrounding rural fringes.",
        "Heat Absorption: Dense asphalt, concrete, and commercial rooftops absorb and retain high solar thermal radiation.",
        "Vegetation Deficit: Depletion of natural tree canopies and green corridors suppresses evaporative cooling and moisture flux.",
        "Environmental Impact: Heightened surface heat accelerates building cooling energy demands, degrades air quality, and intensifies heat-wave health risks."
    ]
    add_bullet_points(tf_l, intro_points, font_size=11.5)

    # Right Card: Formal Problem Statement
    card_right = add_content_card(s2, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.3), border_color=COLOR_EMERALD)
    tb_r = s2.shapes.add_textbox(Inches(7.0), Inches(1.7), Inches(5.3), Inches(4.8))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True

    p_h2 = tf_r.paragraphs[0]
    p_h2.text = "Core Problem Statement"
    p_h2.font.bold = True
    p_h2.font.size = Pt(15)
    p_h2.font.color.rgb = COLOR_EMERALD
    p_h2.space_after = Pt(12)

    prob_points = [
        "Lack of Predictive Tools: Urban planners lack unified, data-driven platforms capable of linking high-resolution satellite imagery directly to thermal forecasts.",
        "Non-Quantified Interventions: City agencies invest in green infrastructure without prior quantitative estimations of microclimatic cooling yields.",
        "Coupled Multi-Variable Complexity: Real-world surface temperatures depend on non-linear interactions among land-cover, elevation, humidity, and solar irradiance.",
        "Project Solution: A dual-stage AI architecture that segments optical satellite scenes, predicts Land Surface Temperature anomalies, and interactively simulates greening policies."
    ]
    add_bullet_points(tf_r, prob_points, font_size=11.5)

    # =========================================================================
    # SLIDE 3: PROJECT OBJECTIVES
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    apply_dark_background(s3)
    add_slide_header(s3, "Project Objectives & Scope", "Research Goals")

    objectives = [
        ("01", "Satellite Image Segmentation", "Deploy deep convolutional neural networks (U-Net) to process high-resolution satellite imagery into pixel-level land-cover classifications."),
        ("02", "Four-Class Surface Taxonomy", "Accurately categorize urban landscapes into four standardized classes: Impervious Surfaces, Vegetation Canopy, Bare Soil, and Water Bodies."),
        ("03", "Dynamic Land-Cover Metrics", "Dynamically compute regional land-cover percentage distributions that strictly sum to 100% for microclimate feature engineering."),
        ("04", "LST Anomaly Regression", "Develop a Deep Feedforward Artificial Neural Network (ANN) coupling 4 land-cover features with 3 environmental forcing factors to predict thermal anomalies."),
        ("05", "Interactive Mitigation Simulation", "Build a simulation engine enabling planners to model 10%–50% greening interventions and quantify counterfactual cooling reductions before physical execution.")
    ]

    card_y = 1.5
    for num, title, desc in objectives:
        add_content_card(s3, Inches(0.8), Inches(card_y), Inches(11.7), Inches(0.95))
        tb = s3.shapes.add_textbox(Inches(1.0), Inches(card_y + 0.05), Inches(11.3), Inches(0.85))
        tf = tb.text_frame
        tf.word_wrap = True

        p = tf.paragraphs[0]
        run_num = p.add_run()
        run_num.text = f"[{num}] "
        run_num.font.bold = True
        run_num.font.size = Pt(13)
        run_num.font.color.rgb = COLOR_CYAN

        run_title = p.add_run()
        run_title.text = f"{title}: "
        run_title.font.bold = True
        run_title.font.size = Pt(12)
        run_title.font.color.rgb = COLOR_WHITE

        run_desc = p.add_run()
        run_desc.text = desc
        run_desc.font.size = Pt(11)
        run_desc.font.color.rgb = COLOR_LIGHT_GREY

        card_y += 1.08

    # =========================================================================
    # SLIDE 4: DATASETS AND TECHNOLOGY STACK
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    apply_dark_background(s4)
    add_slide_header(s4, "Data Sources and Technology Stack", "System Infrastructure")

    # Left: Data Sources
    add_content_card(s4, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.3))
    tb_d = s4.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(5.2), Inches(4.8))
    tf_d = tb_d.text_frame
    tf_d.word_wrap = True

    p_d_h = tf_d.paragraphs[0]
    p_d_h.text = "Multimodal Data Sources"
    p_d_h.font.bold = True
    p_d_h.font.size = Pt(14)
    p_d_h.font.color.rgb = COLOR_CYAN
    p_d_h.space_after = Pt(10)

    data_pts = [
        "Optical Satellite Imagery: High-resolution Copernicus Sentinel-2 (10m/px) and Landsat-8 RGB imagery.",
        "Near-Infrared (NIR) Support: Configurable 3-band (RGB) and 4-band (RGB+NIR) input pipelines.",
        "Thermal Infrared (LST): Landsat Collection-2 Level-2 Surface Temperature products (Band 10 / TIRS).",
        "Elevation (DEM): Digital Elevation Model topographical data capturing lapse-rate cooling.",
        "Atmospheric Forcing: Baseline relative humidity (%) and surface solar irradiance (W/m²).",
        "Spatial Partitioning: Region-based geographic splitting preventing spatial leakage between train/test."
    ]
    add_bullet_points(tf_d, data_pts, font_size=11)

    # Right: Technology Stack
    add_content_card(s4, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.3))
    tb_t = s4.shapes.add_textbox(Inches(7.0), Inches(1.7), Inches(5.3), Inches(4.8))
    tf_t = tb_t.text_frame
    tf_t.word_wrap = True

    p_t_h = tf_t.paragraphs[0]
    p_t_h.text = "Production Technology Stack"
    p_t_h.font.bold = True
    p_t_h.font.size = Pt(14)
    p_t_h.font.color.rgb = COLOR_EMERALD
    p_t_h.space_after = Pt(10)

    tech_pts = [
        "Deep Learning Engine: PyTorch 2.5 with CUDA 12.1 hardware acceleration on NVIDIA RTX 3050 GPU.",
        "Geospatial Processing: Rasterio, GeoPandas, Shapely, PyProj, and Affine coordinate transforms.",
        "Scientific Numerical Core: NumPy, SciPy (multidimensional filters), Pandas, Scikit-learn.",
        "Web Architecture: FastAPI asynchronous REST server and Streamlit interactive dashboard.",
        "Front-End Design: Modern HTML5, Vanilla CSS (Glassmorphism & Obsidian theme), and Chart.js.",
        "Quality & Testing: Automated PyTest unit and integration suite (12/12 verified tests)."
    ]
    add_bullet_points(tf_t, tech_pts, font_size=11)

    # =========================================================================
    # SLIDE 5: SYSTEM ARCHITECTURE
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    apply_dark_background(s5)
    add_slide_header(s5, "Proposed System Architecture", "End-to-End Pipeline")

    arch_img_path = config.VISUALIZATIONS_DIR / "slide_architecture_diagram.png"
    if arch_img_path.exists():
        s5.shapes.add_picture(str(arch_img_path), Inches(1.1), Inches(1.5), Inches(11.1), Inches(4.8))

    # Caption / Footnote
    tb_c = s5.shapes.add_textbox(Inches(1.1), Inches(6.45), Inches(11.1), Inches(0.6))
    p_c = tb_c.text_frame.paragraphs[0]
    p_c.text = "Figure 1: Two-Stage Integrated Architecture connecting U-Net Semantic Segmentation and Deep ANN LST Regression with Counterfactual Mitigation Simulation."
    p_c.font.size = Pt(10)
    p_c.font.italic = True
    p_c.font.color.rgb = COLOR_MUTED
    p_c.alignment = PP_ALIGN.CENTER

    # =========================================================================
    # SLIDE 6: PHASE 1 – SEMANTIC SEGMENTATION USING U-NET
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    apply_dark_background(s6)
    add_slide_header(s6, "Phase 1: Land-Cover Classification Using U-Net", "Deep Semantic Segmentation")

    # Left: Explanation & Architecture
    add_content_card(s6, Inches(0.8), Inches(1.5), Inches(5.8), Inches(5.3))
    tb_u = s6.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(5.4), Inches(4.8))
    tf_u = tb_u.text_frame
    tf_u.word_wrap = True

    p_u_h = tf_u.paragraphs[0]
    p_u_h.text = "Encoder-Decoder with Skip Connections"
    p_u_h.font.bold = True
    p_u_h.font.size = Pt(14)
    p_u_h.font.color.rgb = COLOR_WHITE
    p_u_h.space_after = Pt(10)

    unet_pts = [
        "Contracting Path: 4 encoder stages with DoubleConv blocks (3x3 conv, BatchNorm, ReLU) and 2x2 Max Pooling.",
        "Expanding Path: 4 decoder stages with ConvTranspose2d upsampling concatenated with skip connections.",
        "Fine Boundary Recovery: Skip connections restore high-resolution spatial details lost during downsampling.",
        "Classification Layer: 1x1 convolution mapping base feature channels to 4 output class logits.",
        "Inference Modes: Softmax normalization produces per-pixel probability maps and argmax class masks."
    ]
    add_bullet_points(tf_u, unet_pts, font_size=11)

    # Right: Real Imagery Output
    add_content_card(s6, Inches(6.9), Inches(1.5), Inches(5.6), Inches(5.3))
    seg_img_path = config.VISUALIZATIONS_DIR / "slide_segmentation_pair.png"
    if seg_img_path.exists():
        s6.shapes.add_picture(str(seg_img_path), Inches(7.1), Inches(1.7), Inches(5.2), Inches(2.6))

    # Legend & Description
    tb_leg = s6.shapes.add_textbox(Inches(7.1), Inches(4.45), Inches(5.2), Inches(2.1))
    tf_leg = tb_leg.text_frame
    tf_leg.word_wrap = True

    p_l_h = tf_leg.paragraphs[0]
    p_l_h.text = "Four-Class Scientific Taxonomy:"
    p_l_h.font.bold = True
    p_l_h.font.size = Pt(11)
    p_l_h.font.color.rgb = COLOR_CYAN
    p_l_h.space_after = Pt(6)

    leg_items = [
        "Class 0: Impervious Surfaces (#E63946 Red) - Roads, buildings",
        "Class 1: Vegetation Canopy (#10B981 Green) - Trees, parks",
        "Class 2: Bare Soil (#E9C46A Yellow) - Exposed open ground",
        "Class 3: Water Bodies (#38BDF8 Blue) - Rivers, lakes"
    ]
    add_bullet_points(tf_leg, leg_items, font_size=10, bold_title=False)

    # =========================================================================
    # SLIDE 7: PHASE 2 – LST PREDICTION USING ANN
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    apply_dark_background(s7)
    add_slide_header(s7, "Phase 2: Land Surface Temperature (LST) Regression", "Thermal Anomaly Prediction")

    # Left: 7-Dimensional Feature Vector
    add_content_card(s7, Inches(0.8), Inches(1.5), Inches(5.7), Inches(5.3))
    tb_f = s7.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(5.3), Inches(4.8))
    tf_f = tb_f.text_frame
    tf_f.word_wrap = True

    p_f_h = tf_f.paragraphs[0]
    p_f_h.text = "7-Dimensional Input Feature Space"
    p_f_h.font.bold = True
    p_f_h.font.size = Pt(14)
    p_f_h.font.color.rgb = COLOR_CYAN
    p_f_h.space_after = Pt(10)

    feat_pts = [
        "Impervious Surface %: Derived dynamically from U-Net segmentation. Major driver of surface thermal absorption (+).",
        "Vegetation Canopy %: Primary cooling agent via evapotranspiration and solar shading (-).",
        "Bare Soil %: Intermediate heat capacity and thermal inertia (+).",
        "Water Body %: Natural thermal heat sink with high specific heat capacity (-).",
        "Elevation (DEM): Captures atmospheric lapse-rate cooling (- ~5.5°C per 1,000m elevation).",
        "Baseline Humidity %: Modulates evaporative resistance and sensible heat exchange (-).",
        "Solar Irradiance: Direct downwelling solar radiative flux (W/m²) driving surface heating (+)."
    ]
    add_bullet_points(tf_f, feat_pts, font_size=10.5)

    # Right: ANN Architecture & Training
    add_content_card(s7, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.3))
    tb_ann = s7.shapes.add_textbox(Inches(7.0), Inches(1.7), Inches(5.3), Inches(4.8))
    tf_ann = tb_ann.text_frame
    tf_ann.word_wrap = True

    p_a_h = tf_ann.paragraphs[0]
    p_a_h.text = "Deep Feedforward ANN Architecture"
    p_a_h.font.bold = True
    p_a_h.font.size = Pt(14)
    p_a_h.font.color.rgb = COLOR_EMERALD
    p_a_h.space_after = Pt(10)

    ann_pts = [
        "Input Layer: 7 continuous input neurons corresponding to normalized land-cover & environmental features.",
        "Hidden Layer 1: 128 neurons with Batch Normalization, ReLU activation, and 15% Dropout.",
        "Hidden Layer 2: 64 neurons with Batch Normalization and ReLU non-linearity.",
        "Hidden Layer 3: 32 neurons refining non-linear thermal interactions.",
        "Output Layer: 1 continuous linear neuron predicting regional LST Anomaly Index (°C deviation from regional baseline).",
        "Optimization: Trained using AdamW optimizer with Mean Squared Error (MSE) loss and learning rate scheduling."
    ]
    add_bullet_points(tf_ann, ann_pts, font_size=10.5)

    # =========================================================================
    # SLIDE 8: INTEGRATED AI WORKFLOW
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    apply_dark_background(s8)
    add_slide_header(s8, "Integrated End-to-End AI Workflow", "Coupled Model Inference")

    workflow_steps = [
        ("Step 1", "Satellite Ingestion", "Optical scene loaded, radiometrically calibrated, and divided into 256x256 image patches."),
        ("Step 2", "Pixel Segmentation", "U-Net executes pixel-level classification, categorizing pixels into Urban, Veg, Soil, Water."),
        ("Step 3", "Percentage Extraction", "Land-cover fractions are dynamically computed, ensuring rigorous normalization to 100%."),
        ("Step 4", "Feature Assembly", "Four surface percentages are concatenated with elevation, humidity, and solar irradiance into a 7D vector."),
        ("Step 5", "Thermal Regression", "Trained ANN processes scaled feature vector to predict the Land Surface Temperature anomaly (°C)."),
        ("Step 6", "Decision Support", "Results feed the web platform for interactive greening policy simulation and heat risk assessment.")
    ]

    # Two columns of 3 cards
    col_w = 5.6
    for idx, (step_tag, step_title, step_desc) in enumerate(workflow_steps):
        c_left = 0.8 if idx < 3 else 6.9
        c_top = 1.5 + (idx % 3) * 1.75
        add_content_card(s8, Inches(c_left), Inches(c_top), Inches(col_w), Inches(1.55))

        tb = s8.shapes.add_textbox(Inches(c_left + 0.2), Inches(c_top + 0.1), Inches(col_w - 0.4), Inches(1.35))
        tf = tb.text_frame
        tf.word_wrap = True

        p1 = tf.paragraphs[0]
        run_st = p1.add_run()
        run_st.text = f"{step_tag.upper()} • "
        run_st.font.bold = True
        run_st.font.size = Pt(10)
        run_st.font.color.rgb = COLOR_CYAN

        run_tit = p1.add_run()
        run_tit.text = step_title
        run_tit.font.bold = True
        run_tit.font.size = Pt(13)
        run_tit.font.color.rgb = COLOR_WHITE

        p2 = tf.add_paragraph()
        p2.text = step_desc
        p2.font.size = Pt(10.5)
        p2.font.color.rgb = COLOR_LIGHT_GREY
        p2.space_before = Pt(4)

    # =========================================================================
    # SLIDE 9: UHI MITIGATION SIMULATION
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    apply_dark_background(s9)
    add_slide_header(s9, "Simulating Urban Greening Interventions", "Counterfactual Mitigation Engine")

    # Left: Explanation & Policy Formulation
    add_content_card(s9, Inches(0.8), Inches(1.5), Inches(5.8), Inches(5.3))
    tb_sim = s9.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(5.4), Inches(4.8))
    tf_sim = tb_sim.text_frame
    tf_sim.word_wrap = True

    p_s_h = tf_sim.paragraphs[0]
    p_s_h.text = "Targeted Greening Intervention Engine"
    p_s_h.font.bold = True
    p_s_h.font.size = Pt(14)
    p_s_h.font.color.rgb = COLOR_WHITE
    p_s_h.space_after = Pt(10)

    sim_pts = [
        "Conservation Principle: Converts a designated percentage (10% to 50%) of impervious pixels strictly into vegetation canopy while preserving existing soil and water.",
        "Spatially Contiguous Greening: Leverages Euclidean distance transforms to simulate realistic contiguous green corridors and urban parks rather than scattered noise pixels.",
        "Dynamic Recalculation: Modified surface distributions are assembled with environmental metadata and fed into the regression ANN.",
        "Quantitative Cooling Metric:\n   Predicted Temperature Reduction = Original LST Anomaly - Modified LST Anomaly.",
        "Model-Based Disclaimer: Forecasted reductions represent scenario estimations under specified environmental forcing, serving as data-driven planning guides."
    ]
    add_bullet_points(tf_sim, sim_pts, font_size=10.5)

    # Right: Before & After Visual Preview
    add_content_card(s9, Inches(6.9), Inches(1.5), Inches(5.6), Inches(5.3))
    mit_img_path = config.VISUALIZATIONS_DIR / "slide_mitigation_pair.png"
    if mit_img_path.exists():
        s9.shapes.add_picture(str(mit_img_path), Inches(7.1), Inches(1.7), Inches(5.2), Inches(2.6))

    tb_sc = s9.shapes.add_textbox(Inches(7.1), Inches(4.45), Inches(5.2), Inches(2.1))
    tf_sc = tb_sc.text_frame
    tf_sc.word_wrap = True

    p_sc_h = tf_sc.paragraphs[0]
    p_sc_h.text = "Discrete Policy Evaluation Scenarios:"
    p_sc_h.font.bold = True
    p_sc_h.font.size = Pt(11)
    p_sc_h.font.color.rgb = COLOR_EMERALD
    p_sc_h.space_after = Pt(6)

    scenarios = [
        "10% Conversion: Baseline urban greening, minor localized cooling (~0.4°C)",
        "20% Conversion: Medium green corridor development (~0.8°C cooling)",
        "30% Conversion: Aggressive park network & green roof deployment (~1.3°C cooling)",
        "50% Conversion: High-density sustainable eco-district redevelopment (~2.1°C cooling)"
    ]
    add_bullet_points(tf_sc, scenarios, font_size=10, bold_title=False)

    # =========================================================================
    # SLIDE 10: RESULTS AND MODEL EVALUATION
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    apply_dark_background(s10)
    add_slide_header(s10, "Model Performance and Empirical Evaluation", "Experimental Results")

    # Left: Section A - Segmentation Model
    add_content_card(s10, Inches(0.8), Inches(1.5), Inches(5.7), Inches(3.2))
    tb_m1 = s10.shapes.add_textbox(Inches(1.0), Inches(1.65), Inches(5.3), Inches(2.8))
    tf_m1 = tb_m1.text_frame
    tf_m1.word_wrap = True

    p_m1_h = tf_m1.paragraphs[0]
    p_m1_h.text = "Section A: U-Net Segmentation Performance"
    p_m1_h.font.bold = True
    p_m1_h.font.size = Pt(13)
    p_m1_h.font.color.rgb = COLOR_EMERALD
    p_m1_h.space_after = Pt(6)

    m1_pts = [
        "Synthetic Test Partition: Mean IoU = 0.9947 | Pixel Accuracy = 99.74%",
        "Real Sentinel-2 Test Partition: Mean IoU = 0.6899 | Pixel Accuracy = 91.98%",
        "Per-Class Test IoU: Impervious: 0.995, Vegetation: 0.994, Soil: 0.994, Water: 0.996",
        "Hardware Performance: 50.3 seconds total GPU training on NVIDIA RTX 3050"
    ]
    add_bullet_points(tf_m1, m1_pts, font_size=10)

    # Left Lower: Section B - Regression Model
    add_content_card(s10, Inches(0.8), Inches(4.85), Inches(5.7), Inches(1.95))
    tb_m2 = s10.shapes.add_textbox(Inches(1.0), Inches(4.95), Inches(5.3), Inches(1.75))
    tf_m2 = tb_m2.text_frame
    tf_m2.word_wrap = True

    p_m2_h = tf_m2.paragraphs[0]
    p_m2_h.text = "Section B: LST Regression ANN Performance"
    p_m2_h.font.bold = True
    p_m2_h.font.size = Pt(13)
    p_m2_h.font.color.rgb = COLOR_CYAN
    p_m2_h.space_after = Pt(6)

    m2_pts = [
        "Synthetic Test Partition: RMSE = 0.4596 °C | MAE = 0.3649 °C | R² = 0.9167",
        "Real Environmental Test Partition: RMSE = 1.1800 °C | MAE = 1.0903 °C",
        "End-to-End System Evaluation: Combined E2E RMSE = 0.4602 °C (R² = 0.9165)"
    ]
    add_bullet_points(tf_m2, m2_pts, font_size=10)

    # Right: Embedded Actual Convergence Charts
    add_content_card(s10, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.3))
    chart_img_path = config.VISUALIZATIONS_DIR / "slide_model_metrics_chart.png"
    if chart_img_path.exists():
        s10.shapes.add_picture(str(chart_img_path), Inches(7.0), Inches(1.7), Inches(5.3), Inches(2.25))

    tb_provenance = s10.shapes.add_textbox(Inches(7.0), Inches(4.1), Inches(5.3), Inches(2.5))
    tf_prov = tb_provenance.text_frame
    tf_prov.word_wrap = True

    p_p_h = tf_prov.paragraphs[0]
    p_p_h.text = "Data Provenance & Scientific Transparency:"
    p_p_h.font.bold = True
    p_p_h.font.size = Pt(11)
    p_p_h.font.color.rgb = COLOR_WHITE
    p_p_h.space_after = Pt(6)

    prov_items = [
        "Tested on Unseen Geographic Partitions: 70% Train, 15% Val, 15% Test block split.",
        "Zero Data Leakage: Scalers fit strictly on training splits; spatial tiling avoids overlap.",
        "Authentic Satellite Tested: Validated on 1001x1001 Copernicus Sentinel-2 GeoTIFF imagery."
    ]
    add_bullet_points(tf_prov, prov_items, font_size=10, bold_title=False)

    # =========================================================================
    # SLIDE 11: APPLICATIONS AND FUTURE SCOPE
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    apply_dark_background(s11)
    add_slide_header(s11, "Applications and Future Enhancements", "Impact & Roadmap")

    # Left: Practical Applications
    add_content_card(s11, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.3))
    tb_app = s11.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(5.2), Inches(4.8))
    tf_app = tb_app.text_frame
    tf_app.word_wrap = True

    p_ap_h = tf_app.paragraphs[0]
    p_ap_h.text = "Target Urban Planning Applications"
    p_ap_h.font.bold = True
    p_ap_h.font.size = Pt(14)
    p_ap_h.font.color.rgb = COLOR_EMERALD
    p_ap_h.space_after = Pt(10)

    app_pts = [
        "Smart City Master Planning: Prioritize municipal greening investments in high-heat impervious zones.",
        "Heat-Risk Vulnerability Mapping: Identify urban communities disproportionately exposed to surface thermal extremes.",
        "Cool Roof & Pavement Programs: Evaluate retrofitting asphalt and dark rooftops with high-albedo materials.",
        "Urban Green Space Development: Support ecological park and green corridor placement with quantitative cooling estimates.",
        "Climate Resilience Auditing: Measure municipal compliance with sustainable development heat mitigation targets."
    ]
    add_bullet_points(tf_app, app_pts, font_size=11)

    # Right: Future Roadmap
    add_content_card(s11, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.3))
    tb_fut = s11.shapes.add_textbox(Inches(7.0), Inches(1.7), Inches(5.3), Inches(4.8))
    tf_fut = tb_fut.text_frame
    tf_fut.word_wrap = True

    p_f_h2 = tf_fut.paragraphs[0]
    p_f_h2.text = "Future Research & Engineering Enhancements"
    p_f_h2.font.bold = True
    p_f_h2.font.size = Pt(14)
    p_f_h2.font.color.rgb = COLOR_CYAN
    p_f_h2.space_after = Pt(10)

    fut_pts = [
        "Real-Time Satellite Streaming: Direct integration with Microsoft Planetary Computer & Copernicus STAC APIs.",
        "3D Urban Canopy Geometry: Incorporate LiDAR building height data and sky-view factor (SVF) for 3D canyon modeling.",
        "Species-Specific Transpiration: Differentiate tree canopy species by evapotranspiration potential.",
        "Temporal Time-Series Analysis: Track seasonal and diurnal heat island variations across multi-year satellite archives.",
        "City-Scale Cloud Deployment: Containerized Kubernetes microservices for nationwide metropolitan coverage."
    ]
    add_bullet_points(tf_fut, fut_pts, font_size=11)

    # =========================================================================
    # SLIDE 12: CONCLUSION AND THANK YOU
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    apply_dark_background(s12)
    add_slide_header(s12, "Conclusion & Key Contributions", "Project Summary")

    # Top Card: Project Synthesis
    add_content_card(s12, Inches(0.8), Inches(1.5), Inches(11.7), Inches(2.7), border_color=COLOR_EMERALD)
    tb_con = s12.shapes.add_textbox(Inches(1.0), Inches(1.65), Inches(11.3), Inches(2.3))
    tf_con = tb_con.text_frame
    tf_con.word_wrap = True

    p_c_title = tf_con.paragraphs[0]
    p_c_title.text = "Research Project Synthesis"
    p_c_title.font.bold = True
    p_c_title.font.size = Pt(14)
    p_c_title.font.color.rgb = COLOR_WHITE
    p_c_title.space_after = Pt(8)

    concl_pts = [
        "Unified AI Architecture: Successfully paired U-Net deep convolutional semantic segmentation with an Artificial Neural Network regression model for urban microclimatic modeling.",
        "High Scientific Fidelity: Demonstrated 99.7% pixel accuracy on baseline benchmarks and 92.0% accuracy on real Copernicus Sentinel-2 multispectral scenes.",
        "Actionable Intervention Modeling: Delivered an interactive mitigation engine quantifying cooling yields for 10% to 50% impervious surface conversion scenarios.",
        "Full-Stack Platform Delivery: Produced a complete production platform including an interactive web dashboard and GIS export capabilities."
    ]
    add_bullet_points(tf_con, concl_pts, font_size=11)

    # Bottom Banner: Thank You & Q&A
    add_content_card(s12, Inches(0.8), Inches(4.5), Inches(11.7), Inches(2.3), border_color=COLOR_CYAN)
    tb_ty = s12.shapes.add_textbox(Inches(1.0), Inches(4.65), Inches(11.3), Inches(2.0))
    tf_ty = tb_ty.text_frame
    tf_ty.word_wrap = True

    p_ty = tf_ty.paragraphs[0]
    p_ty.text = "THANK YOU"
    p_ty.font.name = "Arial"
    p_ty.font.size = Pt(32)
    p_ty.font.bold = True
    p_ty.font.color.rgb = COLOR_EMERALD
    p_ty.alignment = PP_ALIGN.CENTER

    p_qa = tf_ty.add_paragraph()
    p_qa.text = "Questions & Answers"
    p_qa.font.name = "Arial"
    p_qa.font.size = Pt(16)
    p_qa.font.bold = True
    p_qa.font.color.rgb = COLOR_WHITE
    p_qa.space_before = Pt(4)
    p_qa.alignment = PP_ALIGN.CENTER

    p_inst = tf_ty.add_paragraph()
    p_inst.text = "Kunal (Roll No: 24215215)  |  Christ University Delhi NCR  |  Project Guide: Dr. Ramesh Chandra Poonia"
    p_inst.font.name = "Arial"
    p_inst.font.size = Pt(11)
    p_inst.font.color.rgb = COLOR_MUTED
    p_inst.space_before = Pt(8)
    p_inst.alignment = PP_ALIGN.CENTER

    # Save PowerPoint file
    out_pptx = config.BASE_DIR / "UHI_Mitigation_Project_Presentation.pptx"
    prs.save(str(out_pptx))
    print("PowerPoint presentation generated successfully:", out_pptx)
    return out_pptx


if __name__ == "__main__":
    build_powerpoint()
