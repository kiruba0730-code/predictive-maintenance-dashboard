"""
Standalone script to train the predictive maintenance model
and save artifacts to models/predictive_maintenance_rf.joblib.
"""

from data_loader import load_dataset
from preprocessing import engineer_features
from model import train_and_evaluate

def main():
    print("Loading AI4I 2020 dataset...")
    df = load_dataset()
    print("Engineering features...")
    df_feat = engineer_features(df)
    print("Training model pipeline with anti-leakage guards & stratified split...")
    results = train_and_evaluate(df_feat, test_size=0.2, save_artifact=True)

    metrics = results["model_metrics"]
    baseline = results["baseline"]
    cm = metrics["confusion_matrix"]

    print("\n=== MODEL TRAINING COMPLETED ===")
    print(f"Train samples: {results['n_train']} | Test samples: {results['n_test']}")
    print(f"Features used ({len(results['feature_names'])}): {results['feature_names']}")
    print("\nBaseline (Dummy Majority):")
    print(f"  Accuracy: {baseline['accuracy']}% | F1: {baseline['f1_score']}")
    print("\nTrained Random Forest Classifier (Held-out Test Set):")
    print(f"  Accuracy:  {metrics['accuracy']}%")
    print(f"  Precision: {metrics['precision']}")
    print(f"  Recall:    {metrics['recall']}")
    print(f"  F1-Score:  {metrics['f1_score']}")
    print(f"  ROC-AUC:   {metrics['roc_auc']}")
    print("\nConfusion Matrix (Test Fold):")
    print(f"  True Negatives:  {cm['true_negatives']}")
    print(f"  False Positives: {cm['false_positives']}")
    print(f"  False Negatives: {cm['false_negatives']}")
    print(f"  True Positives:  {cm['true_positives']}")
    print("\nModel saved to models/predictive_maintenance_rf.joblib")

if __name__ == "__main__":
    main()
