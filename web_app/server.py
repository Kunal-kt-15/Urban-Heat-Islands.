"""
FastAPI Backend Server for AI-Powered Urban Heat Island (UHI) Mitigation Platform.
Exposes REST endpoints for U-Net segmentation, LST regression ANN inference,
counterfactual greening simulation, and geospatial grid aggregation.
"""

from pathlib import Path
import io
import base64
import json
from typing import Dict, Any, Optional, List
import numpy as np
from PIL import Image

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import config
from utils.device import get_device_info, get_device
from utils.preprocessing import compute_landcover_percentages, assemble_feature_vector
from inference.integrated_pipeline import IntegratedUHIPipeline
from simulation.mitigation_engine import simulate_mitigation
from simulation.scenario_analysis import run_multi_scenario_analysis
from visualization.segmentation_visualizer import colorize_mask, create_segmentation_overlay
from data_pipeline.geospatial_processor import GeospatialProcessor

app = FastAPI(
    title="UHI Mitigation AI Platform",
    description="Deep Learning Urban Heat Island Analysis & Mitigation API",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline = IntegratedUHIPipeline()
try:
    pipeline.load_models()
    print("UHI Pipeline loaded successfully.")
except Exception as e:
    print(f"Warning: Failed to pre-load pipeline: {e}")

STATIC_DIR = Path(__file__).resolve().parent / "static"


def array_to_base64_png(array: np.ndarray) -> str:
    """Converts (H, W, 3) or (H, W) uint8 array to base64 PNG string."""
    if array.ndim == 2:
        img = Image.fromarray(array)
    else:
        img = Image.fromarray(array.astype(np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def load_sample_raw(sample_name: str) -> Dict[str, Any]:
    """Loads sample npz file."""
    sample_file = config.SAMPLES_DIR / sample_name
    if not sample_file.exists():
        # Fallback to sample 1
        sample_file = config.SAMPLES_DIR / "sample_area_1.npz"
    data = np.load(sample_file)
    return {
        "image": data["image"],
        "mask": data["mask"],
        "metadata": data["metadata"],
        "lst": data["lst"]
    }


# -----------------------------------------------------------------------------
# REST API ENDPOINTS
# -----------------------------------------------------------------------------

@app.get("/api/health")
def health_check():
    """Returns system hardware status, CUDA availability, and pipeline state."""
    dev_info = get_device_info()
    return {
        "status": "online",
        "pipeline_loaded": pipeline.is_loaded,
        "hardware": dev_info,
        "classes": config.CLASS_NAMES,
        "class_colors": config.CLASS_HEX_COLORS
    }


@app.get("/api/samples")
def list_samples():
    """Lists available real Sentinel-2 satellite patches and demonstration sectors."""
    samples = []
    # 1. Real Copernicus Sentinel-2 Satellite Patches
    for i in range(1, 6):
        name = f"sample_real_{i}.npz"
        path = config.SAMPLES_DIR / name
        if path.exists():
            data = np.load(path)
            mask = data["mask"]
            pcts = compute_landcover_percentages(mask)
            samples.append({
                "id": f"sample_real_{i}",
                "name": f"🛰️ Real Sentinel-2 Zone #{i:02d}",
                "filename": name,
                "type": "Real Satellite (Copernicus Sentinel-2)",
                "impervious_pct": pcts["impervious_pct"],
                "vegetation_pct": pcts["vegetation_pct"],
                "elevation_m": round(float(data["metadata"][0]), 1),
                "baseline_lst": round(float(data["lst"][0]), 2)
            })

    # 2. Demonstration Urban Sectors
    for i in range(1, 6):
        name = f"sample_area_{i}.npz"
        path = config.SAMPLES_DIR / name
        if path.exists():
            data = np.load(path)
            mask = data["mask"]
            pcts = compute_landcover_percentages(mask)
            samples.append({
                "id": f"sample_area_{i}",
                "name": f"🏙️ Urban Sector #{i:02d}",
                "filename": name,
                "type": "Synthetic Demonstration",
                "impervious_pct": pcts["impervious_pct"],
                "vegetation_pct": pcts["vegetation_pct"],
                "elevation_m": round(float(data["metadata"][0]), 1),
                "baseline_lst": round(float(data["lst"][0]), 2)
            })
    return {"samples": samples}


@app.get("/api/sample/{sample_id}")
def get_sample_data(sample_id: str):
    """Loads and formats a specific sample area."""
    filename = f"{sample_id}.npz" if not sample_id.endswith(".npz") else sample_id
    data = load_sample_raw(filename)

    image = data["image"]  # (3, 256, 256)
    rgb_hwc = np.transpose(image, (1, 2, 0))
    rgb_uint8 = (np.clip(rgb_hwc, 0.0, 1.0) * 255.0).astype(np.uint8)

    mask = data["mask"]
    colored_mask = colorize_mask(mask)
    overlay = create_segmentation_overlay(rgb_hwc, mask, alpha=0.45)
    pcts = compute_landcover_percentages(mask)

    return {
        "sample_id": sample_id,
        "image_base64": array_to_base64_png(rgb_uint8),
        "mask_base64": array_to_base64_png(colored_mask),
        "overlay_base64": array_to_base64_png(overlay),
        "landcover_percentages": pcts,
        "elevation_m": round(float(data["metadata"][0]), 1),
        "humidity_pct": round(float(data["metadata"][1]), 1),
        "solar_irradiance_wm2": round(float(data["metadata"][2]), 1),
        "baseline_lst_anomaly": round(float(data["lst"][0]), 2)
    }


@app.post("/api/segment")
async def run_segmentation(
    sample_id: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    """Runs U-Net segmentation on uploaded file or sample area."""
    if not pipeline.is_loaded:
        pipeline.load_models()

    if file is not None:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB").resize((256, 256))
        img_np = np.array(pil_img, dtype=np.float32) / 255.0
        # (256, 256, 3) -> (3, 256, 256)
        img_chw = np.transpose(img_np, (2, 0, 1))
    elif sample_id:
        filename = f"{sample_id}.npz" if not sample_id.endswith(".npz") else sample_id
        data = load_sample_raw(filename)
        img_chw = data["image"]
    else:
        data = load_sample_raw("sample_area_1.npz")
        img_chw = data["image"]

    tensor_x = pipeline.preprocess_image(img_chw)
    pred_mask, probs = pipeline.segment_image(tensor_x)
    pcts = compute_landcover_percentages(pred_mask)

    rgb_hwc = np.transpose(img_chw, (1, 2, 0))
    rgb_uint8 = (np.clip(rgb_hwc, 0.0, 1.0) * 255.0).astype(np.uint8)
    colored_mask = colorize_mask(pred_mask)
    overlay = create_segmentation_overlay(rgb_hwc, pred_mask, alpha=0.45)

    return {
        "image_base64": array_to_base64_png(rgb_uint8),
        "mask_base64": array_to_base64_png(colored_mask),
        "overlay_base64": array_to_base64_png(overlay),
        "landcover_percentages": pcts,
        "mask_raw": pred_mask.tolist()
    }


@app.post("/api/predict-lst")
def predict_lst_anomaly(payload: Dict[str, Any]):
    """Predicts LST anomaly based on 7 features."""
    if not pipeline.is_loaded:
        pipeline.load_models()

    lc_dict = {
        "impervious_pct": float(payload.get("impervious_pct", 50.0)),
        "vegetation_pct": float(payload.get("vegetation_pct", 25.0)),
        "bare_soil_pct": float(payload.get("bare_soil_pct", 15.0)),
        "water_pct": float(payload.get("water_pct", 10.0))
    }
    elev = float(payload.get("elevation_m", 210.0))
    hum = float(payload.get("humidity_pct", 50.0))
    solar = float(payload.get("solar_irradiance_wm2", 650.0))

    raw_f, scaled_f = pipeline.prepare_regression_features(
        landcover_pcts=lc_dict,
        elevation_m=elev,
        humidity_pct=hum,
        solar_irradiance_wm2=solar
    )
    predicted_lst = pipeline.predict_lst(scaled_f)

    # Risk level classification
    if predicted_lst < 1.0:
        risk = "Low Heat Risk"
        risk_color = "#10b981"
    elif predicted_lst < 3.5:
        risk = "Moderate Heat"
        risk_color = "#38bdf8"
    elif predicted_lst < 6.0:
        risk = "Elevated Heat Island"
        risk_color = "#fbbf24"
    else:
        risk = "Critical Heat Hotspot"
        risk_color = "#f43f5e"

    return {
        "predicted_lst_anomaly": predicted_lst,
        "heat_risk_level": risk,
        "risk_color": risk_color,
        "features": {
            "impervious_pct": lc_dict["impervious_pct"],
            "vegetation_pct": lc_dict["vegetation_pct"],
            "bare_soil_pct": lc_dict["bare_soil_pct"],
            "water_pct": lc_dict["water_pct"],
            "elevation_m": elev,
            "humidity_pct": hum,
            "solar_irradiance_wm2": solar
        }
    }


@app.post("/api/simulate")
def run_simulation(payload: Dict[str, Any]):
    """Simulates urban greening intervention on a land cover mask."""
    if not pipeline.is_loaded:
        pipeline.load_models()

    sample_id = payload.get("sample_id", "sample_area_1")
    filename = f"{sample_id}.npz" if not sample_id.endswith(".npz") else sample_id
    data = load_sample_raw(filename)

    mask = np.array(data["mask"], dtype=np.int64)
    conversion_pct = float(payload.get("conversion_pct", 0.20))
    contiguous = bool(payload.get("contiguous", True))

    elev = float(payload.get("elevation_m", data["metadata"][0]))
    hum = float(payload.get("humidity_pct", data["metadata"][1]))
    solar = float(payload.get("solar_irradiance_wm2", data["metadata"][2]))

    metadata = {
        "elevation_m": elev,
        "humidity_pct": hum,
        "solar_irradiance_wm2": solar
    }

    res = simulate_mitigation(
        mask=mask,
        metadata=metadata,
        conversion_percentage=conversion_pct,
        regression_model=pipeline.regression_ann,
        scaler=pipeline.scaler,
        contiguous=contiguous
    )

    orig_colored = colorize_mask(res["original_mask"])
    mod_colored = colorize_mask(res["modified_mask"])

    return {
        "conversion_percentage": res["conversion_percentage"],
        "conversion_pct_display": res["conversion_pct_display"],
        "pixels_converted": res["pixels_converted"],
        "original_mask_base64": array_to_base64_png(orig_colored),
        "modified_mask_base64": array_to_base64_png(mod_colored),
        "original_landcover_pcts": res["original_landcover_pcts"],
        "modified_landcover_pcts": res["modified_landcover_pcts"],
        "original_lst_anomaly": res["original_lst_anomaly"],
        "modified_lst_anomaly": res["modified_lst_anomaly"],
        "predicted_temperature_reduction": res["predicted_temperature_reduction"],
        "contiguous_mode": res["contiguous_mode"]
    }


@app.post("/api/multi-scenarios")
def get_multi_scenarios(payload: Dict[str, Any]):
    """Returns 10%, 20%, 30%, 40%, 50% intervention comparison."""
    if not pipeline.is_loaded:
        pipeline.load_models()

    sample_id = payload.get("sample_id", "sample_area_1")
    filename = f"{sample_id}.npz" if not sample_id.endswith(".npz") else sample_id
    data = load_sample_raw(filename)

    mask = np.array(data["mask"], dtype=np.int64)
    elev = float(payload.get("elevation_m", data["metadata"][0]))
    hum = float(payload.get("humidity_pct", data["metadata"][1]))
    solar = float(payload.get("solar_irradiance_wm2", data["metadata"][2]))

    metadata = {"elevation_m": elev, "humidity_pct": hum, "solar_irradiance_wm2": solar}

    df, scenario_details = run_multi_scenario_analysis(
        mask=mask,
        metadata=metadata,
        regression_model=pipeline.regression_ann,
        scaler=pipeline.scaler
    )

    records = df.to_dict(orient="records")
    return {"scenarios": records}


@app.get("/api/metrics")
def get_metrics():
    """Returns trained model performance statistics."""
    unet_hist_file = config.UNET_CONFIG["history_path"]
    reg_hist_file = config.REGRESSION_CONFIG["history_path"]
    report_file = config.REPORTS_DIR / "integrated_training_report.json"
    real_summary_file = config.REPORTS_DIR / "real_training_summary.json"
    real_prov_file = config.REPORTS_DIR / "real_data_provenance_report.json"

    unet_hist = {}
    reg_hist = {}
    report = {}
    real_summary = {}
    real_prov = {}

    if unet_hist_file.exists():
        with open(unet_hist_file, "r") as f:
            unet_hist = json.load(f)

    if reg_hist_file.exists():
        with open(reg_hist_file, "r") as f:
            reg_hist = json.load(f)

    if report_file.exists():
        with open(report_file, "r") as f:
            report = json.load(f)

    if real_summary_file.exists():
        with open(real_summary_file, "r") as f:
            real_summary = json.load(f)

    if real_prov_file.exists():
        with open(real_prov_file, "r") as f:
            real_prov = json.load(f)

    return {
        "unet_history": unet_hist,
        "regression_history": reg_hist,
        "integrated_report": report,
        "real_training_summary": real_summary,
        "real_provenance": real_prov
    }


@app.get("/api/zonal-grid")
def get_zonal_grid(cell_size: float = 0.01):
    """Generates geographic grid cells and computes zonal statistics."""
    proc = GeospatialProcessor()
    bounds = (77.15, 28.55, 77.25, 28.65)
    grid_gdf = proc.create_grid_cells(bounds, cell_size_deg=cell_size)

    sample = load_sample_raw("sample_area_1.npz")
    mask_2d = sample["mask"]
    thermal_field = 1.5 + 0.085 * (mask_2d == 0) * 100 - 0.065 * (mask_2d == 1) * 100

    csv_out = config.OUTPUTS_DIR / "reports" / "zonal_grid_features.csv"
    geojson_out = config.OUTPUTS_DIR / "reports" / "zonal_grid_features.geojson"

    df = proc.aggregate_grid_features(
        grid_gdf=grid_gdf,
        landcover_mask=mask_2d,
        lst_map=thermal_field,
        output_csv=csv_out,
        output_geojson=geojson_out
    )

    records = df.to_dict(orient="records")
    return {
        "total_zones": len(records),
        "bounds": bounds,
        "cell_size_deg": cell_size,
        "records": records[:20],  # First 20 for preview
        "has_geojson": geojson_out.exists(),
        "has_csv": csv_out.exists()
    }


@app.get("/api/download/csv")
def download_csv():
    csv_path = config.OUTPUTS_DIR / "reports" / "zonal_grid_features.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="CSV report not generated yet.")
    return FileResponse(csv_path, media_type="text/csv", filename="zonal_uhi_features.csv")


@app.get("/api/download/geojson")
def download_geojson():
    geojson_path = config.OUTPUTS_DIR / "reports" / "zonal_grid_features.geojson"
    if not geojson_path.exists():
        raise HTTPException(status_code=404, detail="GeoJSON report not generated yet.")
    return FileResponse(geojson_path, media_type="application/geo+json", filename="zonal_uhi_features.geojson")


# -----------------------------------------------------------------------------
# STATIC FILES & ROOT WEB PAGE
# -----------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>UHI Platform Frontend Loading...</h1>")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


if __name__ == "__main__":
    uvicorn.run("web_app.server:app", host="0.0.0.0", port=8000, reload=True)
