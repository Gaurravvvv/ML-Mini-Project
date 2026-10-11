import json
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import joblib

from src.config import MODELS_DIR
from src.preprocessor import CreditDataPreprocessor
from src.profiling_kmeans import KMeansCreditProfiler


class CreditRiskInferenceEngine:
    """Unified service for Credit Risk Profiling (K-Means) & Loan Default Prediction (Supervised)."""

    def __init__(self):
        self.preprocessor = CreditDataPreprocessor()
        self.kmeans_profiler = KMeansCreditProfiler()
        self.models: Dict[str, Any] = {}
        self.feature_importance: Dict[str, Any] = {}
        self.is_loaded = False

    def load_artifacts(self):
        """Loads all serialized models, preprocessors, and metadata."""
        if self.is_loaded:
            return

        model_names = ["logistic_regression", "random_forest", "adaboost"]
        for name in model_names:
            path = MODELS_DIR / f"{name}.joblib"
            if path.exists():
                self.models[name] = joblib.load(path)
            else:
                raise FileNotFoundError(f"Missing model artifact: {path}. Run training first.")

        fi_path = MODELS_DIR / "feature_importance.json"
        if fi_path.exists():
            with open(fi_path, "r", encoding="utf-8") as f:
                self.feature_importance = json.load(f)

        self.is_loaded = True

    def assess_borrower(
        self, applicant_data: Dict[str, Any], model_name: str = "random_forest"
    ) -> Dict[str, Any]:
        """Runs both unsupervised profiling and supervised default risk assessment."""
        self.load_artifacts()

        # 1. Unsupervised Risk Profiling (K-Means)
        profile_res = self.kmeans_profiler.profile_applicant(applicant_data)

        # 2. Transform input for supervised models
        X_vec = self.preprocessor.transform_single(applicant_data)

        # Run selected model
        if model_name not in self.models:
            model_name = "random_forest"

        selected_model = self.models[model_name]
        prob = float(selected_model.predict_proba(X_vec)[0, 1])
        prediction = int(selected_model.predict(X_vec)[0])

        # Underwriting decision logic
        if prob < 0.35:
            recommendation = "Approved (Low Default Risk)"
            recommendation_badge = "success"
        elif prob < 0.60:
            recommendation = "Underwriting Review (Moderate Risk)"
            recommendation_badge = "warning"
        else:
            recommendation = "Flagged / High Risk (High Default Probability)"
            recommendation_badge = "danger"

        # Side-by-side probabilities across all 3 models for transparency
        comparison_probs = {}
        for m_key, m_obj in self.models.items():
            p = float(m_obj.predict_proba(X_vec)[0, 1])
            comparison_probs[m_key] = {
                "default_probability": round(p * 100, 1),
                "is_default": int(p >= 0.5),
            }

        top_drivers = self.feature_importance.get(model_name, [])[:5]

        return {
            "model_used": model_name,
            "default_probability": round(prob * 100, 1),
            "will_default": bool(prediction == 1),
            "recommendation": recommendation,
            "recommendation_badge": recommendation_badge,
            "all_models_probabilities": comparison_probs,
            "top_drivers": top_drivers,
            "risk_profile": profile_res,
        }


if __name__ == "__main__":
    from src.data_loader import get_data

    df = get_data()
    sample = df.iloc[0].to_dict()
    del sample["default"]

    engine = CreditRiskInferenceEngine()
    assessment = engine.assess_borrower(sample, model_name="random_forest")
    print("Assessment Result Preview:")
    print(json.dumps(assessment, indent=2))
