@echo off
echo ========================================================
echo   CREDIT RISK PROFILING & LOAN DEFAULT PREDICTION
echo   Models: K-Means, Logistic Regression, Random Forest, AdaBoost
echo ========================================================

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment 'venv' not found.
    echo Please create it using: python -m venv venv
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

REM Check if models are trained
if not exist "models\random_forest.joblib" (
    echo [INFO] Training ML models and clustering engine...
    python train_pipeline.py
)

echo [INFO] Starting FastAPI Web Dashboard on http://127.0.0.1:8000 ...
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
pause
