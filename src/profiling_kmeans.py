import json
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import joblib

from src.config import (
    KMEANS_FEATURES,
    TARGET_COLUMN,
    MODELS_DIR,
    RANDOM_STATE,
)


class KMeansCreditProfiler:
    """Unsupervised K-Means clustering engine for Credit Risk Profiling."""

    def __init__(self, n_clusters: int = 3):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=RANDOM_STATE,
            n_init=10,
        )
        self.cluster_metadata: Dict[str, Any] = {}
        self.is_fitted: bool = False

    def find_optimal_k(self, df: pd.DataFrame, k_range: range = range(2, 7)) -> Dict[str, List[float]]:
        """Computes Inertia and Silhouette scores to evaluate optimal cluster count."""
        X_kmeans = self.scaler.fit_transform(df[KMEANS_FEATURES])
        inertias = []
        silhouettes = []

        for k in k_range:
            km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
            labels = km.fit_predict(X_kmeans)
            inertias.append(float(km.inertia_))
            silhouettes.append(float(silhouette_score(X_kmeans, labels)))

        return {
            "k_values": list(k_range),
            "inertias": inertias,
            "silhouette_scores": silhouettes,
        }

    def fit(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Fit K-Means clustering, assign risk personas, and save artifacts."""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        X_scaled = self.scaler.fit_transform(df[KMEANS_FEATURES])
        labels = self.kmeans.fit_predict(X_scaled)

        df_clustered = df.copy()
        df_clustered["cluster"] = labels

        # Analyze each cluster's financial exposure and historical default rate
        profiles = {}
        cluster_default_rates = {}

        for c in range(self.n_clusters):
            c_df = df_clustered[df_clustered["cluster"] == c]
            size = len(c_df)
            default_rate = float(c_df[TARGET_COLUMN].mean()) if TARGET_COLUMN in c_df.columns else 0.0
            cluster_default_rates[c] = default_rate

            means = {feat: float(c_df[feat].mean()) for feat in KMEANS_FEATURES}
            profiles[c] = {
                "cluster_id": c,
                "size": size,
                "percentage": round((size / len(df)) * 100, 1),
                "historical_default_rate": round(default_rate * 100, 1),
                "feature_means": means,
            }

        # Sort clusters by default rate to consistently assign risk tier labels
        sorted_by_risk = sorted(cluster_default_rates.items(), key=lambda x: x[1])
        tier_names = [
            ("Tier 1: Prime / Conservative Risk", "Low default probability. Established profile with manageable exposure.", "success"),
            ("Tier 2: Moderate / Standard Risk", "Moderate leverage. Typical balanced demographic profile.", "warning"),
            ("Tier 3: Elevated / High-Exposure Risk", "Significant exposure with extended loan durations or high debt burden.", "danger"),
        ]

        assigned_metadata = {}
        for idx, (cluster_id, def_rate) in enumerate(sorted_by_risk):
            name, desc, badge = tier_names[min(idx, len(tier_names) - 1)]
            info = profiles[cluster_id]
            info["persona_title"] = name
            info["description"] = desc
            info["badge_type"] = badge
            assigned_metadata[str(cluster_id)] = info

        # Global feature min-max for radar normalization
        feature_bounds = {}
        for feat in KMEANS_FEATURES:
            feature_bounds[feat] = {
                "min": float(df[feat].min()),
                "max": float(df[feat].max()),
            }

        self.cluster_metadata = {
            "n_clusters": self.n_clusters,
            "features": KMEANS_FEATURES,
            "feature_bounds": feature_bounds,
            "clusters": assigned_metadata,
            "cluster_centers_scaled": self.kmeans.cluster_centers_.tolist(),
        }
        self.is_fitted = True

        # Save artifacts
        joblib.dump(self.kmeans, MODELS_DIR / "kmeans_model.joblib")
        joblib.dump(self.scaler, MODELS_DIR / "kmeans_scaler.joblib")
        with open(MODELS_DIR / "kmeans_metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.cluster_metadata, f, indent=2)

        print(f"[KMeansProfiler] K-Means clustering completed with {self.n_clusters} clusters.")
        return self.cluster_metadata

    def profile_applicant(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Profile an individual applicant into their risk cluster."""
        if not self.is_fitted:
            model_path = MODELS_DIR / "kmeans_model.joblib"
            scaler_path = MODELS_DIR / "kmeans_scaler.joblib"
            meta_path = MODELS_DIR / "kmeans_metadata.json"
            if model_path.exists() and meta_path.exists():
                self.kmeans = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.cluster_metadata = json.load(f)
                self.is_fitted = True
            else:
                raise RuntimeError("KMeans model must be fitted or loaded before profiling.")

        # Extract features as DataFrame with correct feature names
        df_row = pd.DataFrame([{feat: float(input_dict.get(feat, 0.0)) for feat in KMEANS_FEATURES}])
        row_scaled = self.scaler.transform(df_row)

        cluster_id = int(self.kmeans.predict(row_scaled)[0])
        cluster_info = self.cluster_metadata["clusters"][str(cluster_id)]

        # Normalized radar chart points (0 to 100)
        applicant_radar = {}
        cluster_radar = {}
        for feat in KMEANS_FEATURES:
            f_min = self.cluster_metadata["feature_bounds"][feat]["min"]
            f_max = self.cluster_metadata["feature_bounds"][feat]["max"]
            val = float(input_dict.get(feat, f_min))
            c_val = float(cluster_info["feature_means"][feat])

            norm_val = 0.0 if f_max == f_min else ((val - f_min) / (f_max - f_min)) * 100
            norm_c_val = 0.0 if f_max == f_min else ((c_val - f_min) / (f_max - f_min)) * 100

            applicant_radar[feat] = round(max(0, min(100, norm_val)), 1)
            cluster_radar[feat] = round(max(0, min(100, norm_c_val)), 1)

        return {
            "cluster_id": cluster_id,
            "persona_title": cluster_info["persona_title"],
            "description": cluster_info["description"],
            "badge_type": cluster_info["badge_type"],
            "historical_default_rate": cluster_info["historical_default_rate"],
            "cluster_size_percentage": cluster_info["percentage"],
            "cluster_means": cluster_info["feature_means"],
            "applicant_values": {feat: float(input_dict.get(feat, 0.0)) for feat in KMEANS_FEATURES},
            "radar_data": {
                "labels": [f.replace("_", " ").title() for f in KMEANS_FEATURES],
                "applicant_scores": [applicant_radar[f] for f in KMEANS_FEATURES],
                "cluster_scores": [cluster_radar[f] for f in KMEANS_FEATURES],
            },
        }


if __name__ == "__main__":
    from src.data_loader import get_data

    df = get_data()
    profiler = KMeansCreditProfiler(n_clusters=3)
    meta = profiler.fit(df)
    print("Clusters Metadata Summary:")
    print(json.dumps(meta["clusters"], indent=2))
