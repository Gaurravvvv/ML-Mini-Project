from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

from src.config import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    MODELS_DIR,
    TEST_SIZE,
    RANDOM_STATE,
)


class CreditDataPreprocessor:
    """Handles feature preprocessing, scaling, encoding, and serialization."""

    def __init__(self):
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), NUMERICAL_FEATURES),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ]
        )
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    def fit_transform(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Series, pd.Series]:
        """Fit preprocessor on train data, transform train & test splits."""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        # Fit preprocessor
        X_train_transformed = self.preprocessor.fit_transform(X_train)
        X_test_transformed = self.preprocessor.transform(X_test)

        # Extract transformed feature names
        num_names = NUMERICAL_FEATURES
        cat_encoder = self.preprocessor.named_transformers_["cat"]
        cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
        self.feature_names = num_names + cat_names
        self.is_fitted = True

        # Save preprocessor artifact
        joblib.dump(self.preprocessor, MODELS_DIR / "preprocessor.joblib")
        joblib.dump(self.feature_names, MODELS_DIR / "feature_names.joblib")
        print(f"[Preprocessor] Preprocessor fitted and saved ({len(self.feature_names)} features).")

        return X_train_transformed, X_test_transformed, X_train, X_test, y_train, y_test

    def transform_single(self, input_dict: Dict[str, Any]) -> np.ndarray:
        """Transforms a single applicant dictionary into the model-ready feature vector."""
        if not self.is_fitted:
            prep_file = MODELS_DIR / "preprocessor.joblib"
            names_file = MODELS_DIR / "feature_names.joblib"
            if prep_file.exists():
                self.preprocessor = joblib.load(prep_file)
                self.feature_names = joblib.load(names_file)
                self.is_fitted = True
            else:
                raise RuntimeError("Preprocessor must be fitted before transforming single records.")

        expected_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
        filtered_dict = {k: input_dict.get(k) for k in expected_features}
        df_single = pd.DataFrame([filtered_dict])
        return self.preprocessor.transform(df_single)


if __name__ == "__main__":
    from src.data_loader import get_data

    df = get_data()
    preprocessor = CreditDataPreprocessor()
    X_train_t, X_test_t, X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)
    print(f"Train features shape: {X_train_t.shape}")
    print(f"Test features shape: {X_test_t.shape}")
