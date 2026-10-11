import json
from pathlib import Path
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.config import STATIC_DIR, MODELS_DIR, VALUE_MAPPINGS
from app.schemas import ApplicantInput
from src.predictor import CreditRiskInferenceEngine

app = FastAPI(
    title="Credit Risk Profiling & Loan Default Prediction API",
    description="Dual Machine Learning Service using K-Means, Logistic Regression, Random Forest, and AdaBoost",
    version="1.0.0",
)

# Initialize Inference Engine
engine = CreditRiskInferenceEngine()

# Ensure static directory exists
STATIC_DIR.mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "css").mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "js").mkdir(parents=True, exist_ok=True)

# Mount static assets
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def serve_index():
    """Serves the main application dashboard."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(index_file)


@app.get("/api/health")
def health_check():
    """Health check verifying model artifacts."""
    models_ready = (
        (MODELS_DIR / "kmeans_model.joblib").exists()
        and (MODELS_DIR / "logistic_regression.joblib").exists()
        and (MODELS_DIR / "random_forest.joblib").exists()
        and (MODELS_DIR / "adaboost.joblib").exists()
    )
    return {
        "status": "online",
        "models_trained": models_ready,
        "models_available": ["logistic_regression", "random_forest", "adaboost"],
        "clustering": "kmeans",
    }


@app.get("/api/meta")
def get_metadata():
    """Returns dropdown options, field descriptions, and sample test profiles."""
    presets = {
        "prime_low_risk": {
            "title": "Prime Profile (Low Risk)",
            "description": "Mature applicant with established savings, home ownership, and clear credit record.",
            "data": {
                "checking_status": ">= 200 DM",
                "duration": 12,
                "credit_history": "all credits at this bank paid duly",
                "purpose": "car (new)",
                "credit_amount": 1800.0,
                "savings_status": ">= 1000 DM",
                "employment": ">= 7 years",
                "installment_rate": 2,
                "personal_status": "male : married/widowed",
                "other_parties": "none",
                "residence_since": 4,
                "property_magnitude": "real estate",
                "age": 45,
                "other_payment_plans": "none",
                "housing": "own",
                "existing_credits": 2,
                "job": "management / self-employed / qualified",
                "num_dependents": 1,
                "own_telephone": "yes",
                "foreign_worker": "yes",
            },
        },
        "moderate_standard_risk": {
            "title": "Moderate Profile (Standard Risk)",
            "description": "Mid-career borrower with moderate debt, average savings, and renting status.",
            "data": {
                "checking_status": "0 - 200 DM",
                "duration": 24,
                "credit_history": "existing credits paid duly till now",
                "purpose": "furniture/equipment",
                "credit_amount": 3800.0,
                "savings_status": "100 - 500 DM",
                "employment": "1 - 4 years",
                "installment_rate": 3,
                "personal_status": "female : single",
                "other_parties": "none",
                "residence_since": 2,
                "property_magnitude": "car or other",
                "age": 32,
                "other_payment_plans": "none",
                "housing": "rent",
                "existing_credits": 1,
                "job": "skilled employee / official",
                "num_dependents": 1,
                "own_telephone": "none",
                "foreign_worker": "yes",
            },
        },
        "high_exposure_risk": {
            "title": "Subprime Profile (High Exposure)",
            "description": "Long-term high credit request with negative checking balance and past payment delays.",
            "data": {
                "checking_status": "< 0 DM",
                "duration": 48,
                "credit_history": "delay in paying off in the past",
                "purpose": "business",
                "credit_amount": 9500.0,
                "savings_status": "< 100 DM",
                "employment": "< 1 year",
                "installment_rate": 4,
                "personal_status": "male : single",
                "other_parties": "none",
                "residence_since": 1,
                "property_magnitude": "unknown / no property",
                "age": 24,
                "other_payment_plans": "bank",
                "housing": "for free",
                "existing_credits": 3,
                "job": "unskilled resident",
                "num_dependents": 2,
                "own_telephone": "none",
                "foreign_worker": "yes",
            },
        },
    }

    categorical_choices = {col: list(mapping.values()) for col, mapping in VALUE_MAPPINGS.items()}

    return {
        "presets": presets,
        "choices": categorical_choices,
        "models": [
            {"key": "random_forest", "name": "Random Forest Classifier (Ensemble Bagging)"},
            {"key": "logistic_regression", "name": "Logistic Regression (Interpretable Linear)"},
            {"key": "adaboost", "name": "AdaBoost Classifier (Iterative Boosting)"},
        ],
    }


@app.post("/api/assess")
def assess_applicant(applicant: ApplicantInput):
    """Executes both K-Means risk profiling and supervised default risk prediction."""
    try:
        input_data = applicant.model_dump()
        model_name = input_data.pop("model_name", "random_forest")
        result = engine.assess_borrower(input_data, model_name=model_name)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Assessment failed: {str(e)}")


@app.get("/api/models/comparison")
def get_model_comparison():
    """Returns test-set comparative metrics across all 3 supervised models."""
    metrics_path = MODELS_DIR / "model_metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="Model metrics not generated. Please run training pipeline first.")

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    return metrics


@app.get("/api/kmeans/clusters")
def get_kmeans_clusters():
    """Returns K-Means clustering breakdown and persona centroids."""
    km_meta_path = MODELS_DIR / "kmeans_metadata.json"
    if not km_meta_path.exists():
        raise HTTPException(status_code=404, detail="K-Means metadata not found. Please run clustering pipeline first.")

    with open(km_meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return meta
