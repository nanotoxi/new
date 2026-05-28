from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import pickle, numpy as np, pandas as pd, os, logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="NanoToxi RF v9", version="9.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "ml_models", "RandomForest_v9b_combined.pkl")

pipeline = None
feature_names = []
material_lookup = {}
global_median = {}
top_cells = []
cell_cols = []
THRESHOLD = 0.57

try:
    with open(MODEL_PATH, "rb") as fh:
        data = pickle.load(fh)
    pipeline        = data["pipeline"]
    feature_names   = data["feature_names"]
    material_lookup = data["material_lookup"]
    global_median   = data["global_median"]
    top_cells       = data["top_cells"]
    cell_cols       = data["cell_cols"]
    THRESHOLD       = float(data.get("best_threshold", 0.57))
    logger.info(f"RF v9 loaded. Features={len(feature_names)}, Materials={len(material_lookup)}, Threshold={THRESHOLD}")
except Exception as e:
    logger.error(f"Failed to load RF v9: {e}")


# ── Schema ────────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    np_type: str = Field(..., description="e.g. ZnO, CuO, TiO2, SiO2, Au, PLGA")
    primary_size_nm: float = Field(..., gt=0)
    hydrodynamic_size_nm: Optional[float] = None
    zeta_potential_mv: float = 0.0
    surface_area_m2g: Optional[float] = 0.0
    cell_type: str = "HeLa"
    dose_max_ugml: float = Field(..., gt=0)
    exposure_time_h: float = Field(..., gt=0)
    ph: float = 7.4
    nanoparticle_name: Optional[str] = "Unknown"


class PredictResponse(BaseModel):
    toxicity_label: str
    confidence: float
    risk_level: str
    model_version: str
    material_found_in_lookup: bool
    threshold_used: float


# ── Feature engineering ───────────────────────────────────────────────────────
def engineer_features(req: PredictRequest) -> pd.DataFrame:
    mat = req.np_type.strip()
    props = material_lookup.get(mat, global_median)
    found = mat in material_lookup

    size   = req.primary_size_nm
    hydro  = req.hydrodynamic_size_nm if req.hydrodynamic_size_nm else size
    dose   = req.dose_max_ugml
    time_h = req.exposure_time_h
    zeta   = req.zeta_potential_mv
    surf   = req.surface_area_m2g or 0.0
    f_enth = float(props["formation_enthalpy"])

    log_dose      = float(np.log1p(dose))
    log_time      = float(np.log1p(time_h))
    log_core_size = float(np.log1p(size))
    log_hydro     = float(np.log1p(hydro))
    log_surf      = float(np.log1p(surf))

    row = {
        "log_dose":           log_dose,
        "log_time":           log_time,
        "log_core_size":      log_core_size,
        "log_hydro_size":     log_hydro,
        "log_surf_area":      log_surf,
        "surface_charge":     zeta,
        "formation_enthalpy": f_enth,
        "conduction_band":    float(props["conduction_band"]),
        "valence_band":       float(props["valence_band"]),
        "electronegativity":  float(props["electronegativity"]),
        "dose_x_time":        log_dose * log_time,
        "size_x_dose":        log_core_size * log_dose,
        "charge_x_dose":      zeta * log_dose,
        "enthalpy_x_dose":    f_enth * log_dose,
    }
    for c in cell_cols:
        row[c] = 0
    col = f"cell_{req.cell_type}"
    row[col if col in cell_cols else "cell_Other"] = 1

    return pd.DataFrame([row])[feature_names], found


def risk_level(label, proba):
    if label != "Toxic":
        return "LOW RISK"
    if proba >= 0.85: return "HIGH RISK"
    if proba >= 0.70: return "MODERATE RISK"
    return "LOW-MODERATE RISK"


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "rf_v9",
        "loaded": pipeline is not None,
        "features": len(feature_names),
        "materials_in_lookup": len(material_lookup),
        "threshold": THRESHOLD,
    }


@app.get("/materials")
def list_materials():
    return {"materials": sorted(material_lookup.keys()), "count": len(material_lookup)}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        X_df, found = engineer_features(req)
        proba = float(pipeline.predict_proba(X_df.to_numpy())[0][1])
        label = "Toxic" if proba >= THRESHOLD else "Non-Toxic"
        return PredictResponse(
            toxicity_label=label,
            confidence=round(proba, 4),
            risk_level=risk_level(label, proba),
            model_version="rf_v9",
            material_found_in_lookup=found,
            threshold_used=THRESHOLD,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
