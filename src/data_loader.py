import urllib.request
from pathlib import Path
import pandas as pd

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_FILE,
    PROCESSED_DATA_FILE,
    UCI_DATA_URL,
    FALLBACK_DATA_URL,
    RAW_COLUMNS,
    VALUE_MAPPINGS,
    TARGET_COLUMN,
)


def ensure_directories():
    """Ensure data and models directories exist."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_dataset() -> Path:
    """Download the raw UCI German Credit dataset if not already present."""
    ensure_directories()
    if RAW_DATA_FILE.exists() and RAW_DATA_FILE.stat().st_size > 0:
        print(f"[DataLoader] Raw dataset already exists at: {RAW_DATA_FILE}")
        return RAW_DATA_FILE

    print(f"[DataLoader] Downloading German Credit dataset from: {UCI_DATA_URL}")
    try:
        urllib.request.urlretrieve(UCI_DATA_URL, RAW_DATA_FILE)
        print(f"[DataLoader] Successfully downloaded to {RAW_DATA_FILE}")
    except Exception as e:
        print(f"[DataLoader] UCI direct download failed ({e}), trying fallback URL...")
        try:
            # If fallback is CSV
            df_fallback = pd.read_csv(FALLBACK_DATA_URL)
            # Save as CSV in raw directory
            csv_path = RAW_DATA_DIR / "german_fallback.csv"
            df_fallback.to_csv(csv_path, index=False)
            return csv_path
        except Exception as e2:
            raise RuntimeError(f"Failed to download dataset from both sources: {e}, {e2}")

    return RAW_DATA_FILE


def load_raw_data() -> pd.DataFrame:
    """Loads and decodes the raw German Credit dataset into a clean DataFrame."""
    file_path = download_dataset()

    if file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
        # Normalize column names if fallback CSV
        df.columns = [c.lower().replace(".", "_").replace(" ", "_") for c in df.columns]
        if "default" in df.columns:
            return df
        return df

    # Raw space-separated format from UCI
    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=RAW_COLUMNS,
        engine="python",
    )

    # Decode categorical variables
    for col, mapping in VALUE_MAPPINGS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(df[col])

    # Harmonize target variable: 1 (Good) -> 0 (Non-default), 2 (Bad) -> 1 (Default)
    df[TARGET_COLUMN] = df["class"].apply(lambda x: 1 if x == 2 else 0)
    df.drop(columns=["class"], inplace=True)

    # Save processed CSV for fast access
    df.to_csv(PROCESSED_DATA_FILE, index=False)
    print(f"[DataLoader] Cleaned dataset saved to: {PROCESSED_DATA_FILE} (Shape: {df.shape})")

    return df


def get_data(reload_raw: bool = False) -> pd.DataFrame:
    """Get the processed DataFrame, generating it if necessary."""
    ensure_directories()
    if not reload_raw and PROCESSED_DATA_FILE.exists():
        return pd.read_csv(PROCESSED_DATA_FILE)
    return load_raw_data()


if __name__ == "__main__":
    df = get_data(reload_raw=True)
    print("Dataset Preview:")
    print(df.head())
    print("\nTarget Distribution:")
    print(df[TARGET_COLUMN].value_counts(normalize=True))
