import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

from src.data_loader import get_data
from src.profiling_kmeans import KMeansCreditProfiler
from src.train_models import train_and_save_models
from src.evaluate import evaluate_models


def run_full_pipeline():
    print("=" * 70)
    print("STEP 1: Data Acquisition & Preprocessing (UCI German Credit)")
    print("=" * 70)
    df = get_data(reload_raw=True)
    print(f"Data shape: {df.shape}")
    print(f"Default class breakdown:\n{df['default'].value_counts()}")

    print("\n" + "=" * 70)
    print("STEP 2: Unsupervised Credit Risk Profiling (K-Means)")
    print("=" * 70)
    profiler = KMeansCreditProfiler(n_clusters=3)
    optimal_k_results = profiler.find_optimal_k(df)
    print(f"Elbow inertias (k=2..6): {optimal_k_results['inertias']}")
    print(f"Silhouette scores (k=2..6): {optimal_k_results['silhouette_scores']}")
    km_meta = profiler.fit(df)
    for c_id, info in km_meta["clusters"].items():
        print(f"-> Cluster {c_id}: {info['persona_title']} | Historical Default Rate: {info['historical_default_rate']}%")

    print("\n" + "=" * 70)
    print("STEP 3: Supervised Loan Default Prediction Training")
    print("Models: Logistic Regression, Random Forest, AdaBoost")
    print("=" * 70)
    train_and_save_models()

    print("\n" + "=" * 70)
    print("STEP 4: Model Benchmarking & Metric Extraction")
    print("=" * 70)
    metrics = evaluate_models()
    for name, m in metrics.items():
        print(f"-> {m['display_name']}: Accuracy={m['accuracy']}%, Precision={m['precision']}%, Recall={m['recall']}%, F1={m['f1_score']}%, ROC-AUC={m['roc_auc']}%")

    print("\n" + "=" * 70)
    print("ALL PIPELINE STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_full_pipeline()
