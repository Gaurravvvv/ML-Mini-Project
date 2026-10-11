# Credit Risk Profiling & Loan Default Prediction

An end-to-end Machine Learning web application combining **Unsupervised Customer Segmentation** and **Supervised Default Prediction** trained on the **UCI German Credit Data (Statlog)**.

Built with **FastAPI**, **scikit-learn**, **Chart.js**, and an enterprise B2B dashboard theme inspired by **Retail-Eye**.

---

## 🚀 Key Highlights & Machine Learning Techniques

### 1. Unsupervised Credit Risk Profiling (K-Means Clustering)
- **Objective**: Identify behavioral risk segments/personas among loan applicants using financial exposure metrics (`duration`, `credit_amount`, `installment_rate`, `age`, `existing_credits`).
- **Optimal Clusters**: Evaluated via **Elbow Method (Inertia)** and **Silhouette Score Analysis** ($K = 3$).
- **Risk Personas**:
  - **Tier 1: Prime / Conservative Risk** (Low historical default rate, established borrowers with low leverage).
  - **Tier 2: Moderate / Standard Risk** (Average demographic leverage, balanced risk profile).
  - **Tier 3: Elevated / High-Exposure Risk** (Long loan terms, higher loan amounts, stretched disposable income).
- **Visualization**: Multi-axis **Radar Chart** comparing individual applicant exposure to cluster centroids.

### 2. Supervised Loan Default Prediction
- **Class Imbalance Resolution**: UCI German Credit dataset is imbalanced (~70% Good vs 30% Bad loans). Models use `class_weight='balanced'` to prevent minority-class misclassification.
- **Model 1: Logistic Regression**:
  - Transparent linear baseline with L2 regularization.
  - Generates odds ratios and calibrated default probabilities.
- **Model 2: Random Forest Classifier**:
  - Ensemble bagging model capturing non-linear interactions across 20 financial/demographic features.
  - Outputs Gini-based feature importances.
- **Model 3: AdaBoost Classifier**:
  - Iterative boosting algorithm dynamically focusing on difficult-to-classify edge borrowers.
- **Performance Evaluation**:
  - Full benchmarking table: **Accuracy, Precision, Recall, F1-Score, and ROC-AUC**.
  - Interactive **ROC Curve** comparison chart.
  - 2x2 **Confusion Matrices** (Type I and Type II error analysis).

---

## 🛠️ Tech Stack

- **ML & Data Engine**: Python 3.13, `scikit-learn`, `pandas`, `numpy`, `joblib`
- **Backend API**: `FastAPI`, `uvicorn`, `pydantic`
- **Frontend Dashboard**: HTML5, Modern CSS (Slate & Indigo B2B styling), Vanilla JS
- **Visualizations**: `Chart.js` (Risk Gauge, Cluster Radar Chart, ROC Curves)
- **Dataset**: UCI German Credit (Statlog) — 1,000 instances, 20 attributes

---

## 📁 Project Structure

```text
Mini Project/
├── data/
│   ├── raw/                      # Raw UCI German Credit dataset
│   └── processed/                # Cleaned CSV with decoded features
├── models/                       # Exported .joblib models & metadata
│   ├── kmeans_model.joblib
│   ├── kmeans_scaler.joblib
│   ├── kmeans_metadata.json
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   ├── adaboost.joblib
│   ├── preprocessor.joblib
│   ├── feature_names.joblib
│   ├── feature_importance.json
│   └── model_metrics.json
├── src/
│   ├── config.py                 # Paths, column maps & hyperparameters
│   ├── data_loader.py            # Dataset acquisition & categorical decoding
│   ├── preprocessor.py           # Feature transformations & scaling
│   ├── profiling_kmeans.py       # K-Means clustering & persona generation
│   ├── train_models.py           # Supervised model training
│   ├── evaluate.py               # Comparative metrics & ROC curve calculation
│   └── predictor.py              # Unified inference engine
├── app/
│   ├── main.py                   # FastAPI REST API endpoints
│   ├── schemas.py                # Pydantic input schemas
│   └── static/
│       ├── css/style.css         # Retail-Eye inspired B2B CSS theme
│       ├── js/app.js             # Interactive client & Chart.js rendering
│       └── index.html            # Loan officer web application
├── train_pipeline.py             # One-click end-to-end training pipeline
├── requirements.txt              # Pinned Python package dependencies
├── run.bat                       # Windows double-click launcher
└── README.md
```

---

## ⚡ Quick Start

### 1. Setup Virtual Environment & Install Dependencies
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the Models
```powershell
python train_pipeline.py
```

### 3. Launch the Application
```powershell
python -m uvicorn app.main:app --reload
```
Open your browser at: **`http://127.0.0.1:8000`**

Or simply run **`run.bat`** on Windows.

---

## 🌐 REST API Endpoints

- `GET /api/health` — Verifies system and model readiness.
- `GET /api/meta` — Returns dropdown options and sample borrower presets.
- `POST /api/assess` — Runs dual inference (K-Means persona + supervised default prediction).
- `GET /api/kmeans/clusters` — Returns cluster distribution and centroid metrics.
- `GET /api/models/comparison` — Returns comparative metrics and ROC curve coordinates for all 3 models.
