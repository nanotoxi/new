from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import pickle, joblib, numpy as np, pandas as pd, os, logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="NanoToxi RF v16", version="16.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "ml_models", "RandomForest_v16.pkl")

pipeline     = None
feature_names = []
material_lookup = {}
global_median   = {}
top_cells  = []
cell_cols  = []
assay_cats       = []
morph_cols       = []
morph_api_map    = {}
cell_line_species = {}
cell_line_cancer  = {}
THRESHOLD        = 0.5

try:
    data = joblib.load(MODEL_PATH)
    pipeline        = data["pipeline"]
    feature_names   = data["feature_names"]
    material_lookup = data["material_lookup"]
    global_median   = data["global_median"]
    top_cells       = data["top_cells"]
    cell_cols       = data["cell_cols"]
    assay_cats      = data.get("assay_cats", ["MTT","CCK8","LDH","WST1","Other"])
    morph_cols      = data.get("morph_cols", [])
    morph_api_map     = data.get("morph_api_map", {})
    cell_line_species = data.get("cell_line_species", {})
    cell_line_cancer  = data.get("cell_line_cancer",  {})
    THRESHOLD         = float(data.get("best_threshold", 0.5))
    logger.info(f"RF v16 loaded. Features={len(feature_names)}, Materials={len(material_lookup)}, Threshold={THRESHOLD}")
except Exception as e:
    logger.error(f"Failed to load RF v16: {e}")


# ── Schema ────────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    # Required
    np_type: str = Field(..., description="e.g. ZnO, CuO, TiO2, SiO2, Au, Ag, PLGA")
    primary_size_nm: float = Field(..., gt=0, description="Core/primary size in nm")
    dose_max_ugml: float = Field(..., gt=0, description="Max exposure dose in µg/mL")
    exposure_time_h: float = Field(..., gt=0, description="Exposure duration in hours")

    # Physicochemical — optional
    hydrodynamic_size_nm: Optional[float] = Field(None, description="Hydrodynamic size in nm; defaults to primary_size if omitted")
    zeta_potential_mv: float = Field(0.0, description="Surface charge in mV")
    surface_area_m2g: Optional[float] = Field(0.0, description="BET surface area in m²/g")

    # Experimental conditions — optional
    cell_type: str = Field("HeLa", description="Cell line name")
    ph: float = Field(7.4, ge=4.0, le=10.0, description="pH of exposure medium (physiological default: 7.4)")
    temperature_c: float = Field(37.0, ge=20.0, le=42.0, description="Incubation temperature °C (cell culture default: 37.0)")

    # Particle properties — optional
    is_coated: int = Field(0, ge=0, le=1, description="1 = surface-coated/functionalized, 0 = bare")
    morphology: str = Field("Unknown", description="Particle shape: Sphere, Rod, Tube, Wire, Sheet, Core-Shell, Cubic, Dendrimer, Fibrous, Hexagonal, Porous, Other, Unknown")
    assay_type: str = Field("Unknown", description="Viability assay used: MTT, CCK-8, LDH, WST-1, Other, Unknown")

    # Biological context — optional
    cell_type_cancer: Optional[float] = Field(None, ge=0.0, le=1.0, description="Is cell line cancer-derived? 1=Cancer, 0=Normal, 0.5=Unknown. Omit to auto-detect from cell_type name.")
    cell_species_human: Optional[float] = Field(None, ge=0.0, le=1.0, description="Is cell line human-derived? 1=Human, 0=non-human (mouse/rat), 0.5=Unknown. Omit to auto-detect.")
    is_therapeutic: int = Field(0, ge=0, le=1, description="1 = NP designed for therapeutic delivery (intentionally cytotoxic context), 0 = safety study")
    np_type_class: str = Field("Inorganic", description="Material class: Inorganic, Organic, Hybrid")

    # Label only — not used in prediction
    nanoparticle_name: Optional[str] = Field("Unknown", description="Descriptive label only, not used in model")


class PredictResponse(BaseModel):
    toxicity_label: str
    confidence: float
    risk_level: str
    model_version: str
    material_found_in_lookup: bool
    threshold_used: float


# ── Feature engineering ───────────────────────────────────────────────────────
def engineer_features(req: PredictRequest) -> tuple[pd.DataFrame, bool]:
    mat   = req.np_type.strip()
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
        "ph":                 float(np.clip(req.ph, 4.0, 10.0)),
        "temperature":        float(np.clip(req.temperature_c, 20.0, 42.0)),
        "is_coated":          float(req.is_coated),
    }

    # Assay one-hot
    assay_norm = req.assay_type.strip().upper().replace("-","").replace(" ","")
    assay_map  = {"MTT":"MTT","MTS":"MTT","WST1":"WST1","WST8":"WST1","CCK8":"CCK8","LDH":"LDH"}
    resolved   = assay_map.get(assay_norm, "Other" if req.assay_type.lower() not in ("unknown","") else None)
    for cat in assay_cats:
        row[f"assay_{cat}"] = 1 if resolved == cat else 0

    # Morphology one-hot
    morph_key = req.morphology.strip().lower()
    resolved_morph = morph_api_map.get(morph_key)
    for col in morph_cols:
        row[col] = 1 if col == resolved_morph else 0

    # NP type class
    np_class = req.np_type_class.strip().lower()
    row["np_type_inorganic"] = 1.0 if np_class == "inorganic" else 0.0
    row["np_type_organic"]   = 1.0 if np_class == "organic"   else 0.0
    row["np_type_hybrid"]    = 1.0 if np_class == "hybrid"    else 0.0

    # Therapeutic flag
    row["is_therapeutic"] = float(req.is_therapeutic)

    # Cell species and cancer type — use provided values or auto-detect from cell_line knowledge
    cell_col_key = f"cell_{req.cell_type}"
    if req.cell_species_human is not None:
        row["cell_species_human"] = float(req.cell_species_human)
    else:
        row["cell_species_human"] = cell_line_species.get(cell_col_key, 0.5)
    if req.cell_type_cancer is not None:
        row["cell_type_cancer"] = float(req.cell_type_cancer)
    else:
        row["cell_type_cancer"] = cell_line_cancer.get(cell_col_key, 0.5)

    # Cell one-hot
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
        "model": "rf_v16",
        "loaded": pipeline is not None,
        "features": len(feature_names),
        "materials_in_lookup": len(material_lookup),
        "threshold": THRESHOLD,
    }


@app.get("/materials")
def list_materials():
    return {"materials": sorted(material_lookup.keys()), "count": len(material_lookup)}


@app.get("/morphologies")
def list_morphologies():
    return {"morphologies": list(morph_api_map.keys())}


@app.get("/assays")
def list_assays():
    return {"assay_types": assay_cats + ["Unknown"]}


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
            model_version="rf_v16",
            material_found_in_lookup=found,
            threshold_used=THRESHOLD,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
