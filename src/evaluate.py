import json
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
import joblib

from src.config import MODELS_DIR
from src.data_loader import get_data
from src.preprocessor import CreditDataPreprocessor


def evaluate_models() -> Dict[str, Any]:
    """Evaluates all trained models on the test set and exports comparison metrics."""
    df = get_data()
    preprocessor = CreditDataPreprocessor()
    _, X_test_transformed, _, _, _, y_test = preprocessor.fit_transform(df)

    model_names = ["logistic_regression", "random_forest", "adaboost"]
    results = {}

    for name in model_names:
        model_path = MODELS_DIR / f"{name}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Trained model not found at {model_path}. Train models first.")

        model = joblib.load(model_path)
        y_pred = model.predict(X_test_transformed)
        y_prob = model.predict_proba(X_test_transformed)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred).tolist()  # [[TN, FP], [FN, TP]]

        # Downsample ROC points for clean JSON transmission to web charts
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        step = max(1, len(fpr) // 25)
        roc_points = [
            {"x": round(float(fpr[i]), 3), "y": round(float(tpr[i]), 3)}
            for i in range(0, len(fpr), step)
        ]
        if roc_points[-1] != {"x": 1.0, "y": 1.0}:
            roc_points.append({"x": 1.0, "y": 1.0})

        display_name = {
            "logistic_regression": "Logistic Regression",
            "random_forest": "Random Forest Classifier",
            "adaboost": "AdaBoost Classifier",
        }.get(name, name)

        results[name] = {
            "model_key": name,
            "display_name": display_name,
            "accuracy": round(float(acc) * 100, 2),
            "precision": round(float(prec) * 100, 2),
            "recall": round(float(rec) * 100, 2),
            "f1_score": round(float(f1) * 100, 2),
            "roc_auc": round(float(roc_auc) * 100, 2),
            "confusion_matrix": {
                "tn": int(cm[0][0]),
                "fp": int(cm[0][1]),
                "fn": int(cm[1][0]),
                "tp": int(cm[1][1]),
            },
            "roc_curve": roc_points,
        }

    with open(MODELS_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("[Evaluate] Evaluation metrics saved to models/model_metrics.json")
    return results


if __name__ == "__main__":
    metrics = evaluate_models()
    print(json.dumps(metrics, indent=2))
