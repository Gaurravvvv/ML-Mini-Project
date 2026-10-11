import json
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib

from src.config import MODELS_DIR, RANDOM_STATE
from src.data_loader import get_data
from src.preprocessor import CreditDataPreprocessor


def train_and_save_models():
    """Trains Logistic Regression, Random Forest, and AdaBoost models and saves artifacts."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("[TrainModels] Loading and preprocessing dataset...")
    df = get_data()
    preprocessor = CreditDataPreprocessor()
    X_train, X_test, _, _, y_train, y_test = preprocessor.fit_transform(df)

    feature_names = preprocessor.feature_names

    models: Dict[str, Any] = {
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=0.8,
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_split=4,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "adaboost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2, class_weight="balanced"),
            n_estimators=100,
            learning_rate=0.7,
            random_state=RANDOM_STATE,
        ),
    }

    feature_importance_dict = {}

    for model_name, model in models.items():
        print(f"[TrainModels] Training {model_name}...")
        model.fit(X_train, y_train)

        # Save model artifact
        model_file = MODELS_DIR / f"{model_name}.joblib"
        joblib.dump(model, model_file)
        print(f"[TrainModels] Saved {model_name} to {model_file}")

        # Extract top feature drivers
        if hasattr(model, "coef_"):
            # Logistic Regression: Absolute coefficient weight
            coefs = model.coef_[0]
            top_indices = np.argsort(np.abs(coefs))[::-1][:10]
            feature_importance_dict[model_name] = [
                {
                    "feature": feature_names[i],
                    "weight": round(float(coefs[i]), 4),
                    "impact": "Increases Risk" if coefs[i] > 0 else "Decreases Risk",
                }
                for i in top_indices
            ]
        elif hasattr(model, "feature_importances_"):
            # Tree Ensembles (Random Forest & AdaBoost)
            importances = model.feature_importances_
            top_indices = np.argsort(importances)[::-1][:10]
            feature_importance_dict[model_name] = [
                {
                    "feature": feature_names[i],
                    "weight": round(float(importances[i]), 4),
                    "impact": "Relative Importance",
                }
                for i in top_indices
            ]

    # Save feature drivers to JSON
    with open(MODELS_DIR / "feature_importance.json", "w", encoding="utf-8") as f:
        json.dump(feature_importance_dict, f, indent=2)

    print("[TrainModels] All 3 supervised models trained and exported successfully.")
    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    train_and_save_models()
