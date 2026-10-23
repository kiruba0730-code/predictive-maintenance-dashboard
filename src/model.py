"""
Machine Learning module for industrial failure classification.
Ensures zero target leakage, stratified evaluation, baseline comparison,
and transparent evaluation metrics on held-out test data.
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_SEED = 42

def get_model_path() -> Path:
    """Returns standard path to serialized model artifact."""
    current_dir = Path(__file__).resolve().parent
    return current_dir.parent / "models" / "predictive_maintenance_rf.joblib"

def get_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extracts predictor features and binary target.
    
    STRICT ANTI-LEAKAGE POLICY:
    - Excludes 'Machine failure' (Ground truth target)
    - Excludes specific failure modes: 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'
      (These directly define the target mechanism and cannot be known prior to failure)
    - Excludes 'UDI' and 'Product ID' (Non-generalizable index identifiers)
    - Excludes 'Failure Status', 'Failure Mode', and 'Heuristic Risk Score'
    """
    # Exclude leakage and identifiers
    leakage_cols = [
        "UDI",
        "Product ID",
        "Machine failure",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF",
        "Failure Status",
        "Failure Mode",
        "Heuristic Risk Score",
    ]
    
    # Base and domain-engineered features
    feature_cols = [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]
    
    # Check if domain features are available
    if "Temp difference [K]" in df.columns:
        feature_cols.append("Temp difference [K]")
    if "Power [W]" in df.columns:
        feature_cols.append("Power [W]")
        
    X = df[feature_cols].copy()
    y = df["Machine failure"].astype(int).copy()
    
    return X, y

def build_model_pipeline() -> Pipeline:
    """Builds a scikit-learn pipeline with preprocessing and RandomForest."""
    categorical_features = ["Type"]
    numerical_features = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]
    # Optional domain numerical features
    optional_numericals = ["Temp difference [K]", "Power [W]"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
            ("num", StandardScaler(), numerical_features),
        ],
        remainder="passthrough",
    )

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight="balanced",  # Critical for 3.39% imbalanced positive class
        random_state=RANDOM_SEED,
        n_jobs=1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ]
    )
    return pipeline

def train_and_evaluate(
    df: pd.DataFrame,
    test_size: float = 0.2,
    save_artifact: bool = True,
) -> Dict[str, Any]:
    """
    Trains classification pipeline with stratified train/test split,
    computes genuine metrics against baseline, and saves model artifact.
    """
    X, y = get_features_and_target(df)

    # Stratified split to ensure identical 3.39% class proportion in both folds
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    # 1. Baseline Model (Majority class predictor)
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_pred = dummy.predict(X_test)
    dummy_accuracy = float(np.mean(dummy_pred == y_test))

    # 2. Random Forest Model
    pipeline = build_model_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Calculate metrics exclusively on held-out test data
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_prob))
    accuracy = float(np.mean(y_pred == y_test))

    results = {
        "pipeline": pipeline,
        "feature_names": list(X.columns),
        "random_seed": RANDOM_SEED,
        "test_size": test_size,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "baseline": {
            "model_type": "DummyClassifier (Majority Class)",
            "accuracy": round(dummy_accuracy * 100, 2),
            "f1_score": 0.0,
            "note": "A naive baseline that always predicts 'No Failure' achieves ~96.6% accuracy but catches 0% of failures.",
        },
        "model_metrics": {
            "accuracy": round(accuracy * 100, 2),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "confusion_matrix": {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            },
        },
        "limitations": (
            "Rare Event Challenge: Due to the 3.39% minority failure rate, standard accuracy is misleading. "
            "Maintenance operations prioritize Recall (minimizing False Negatives / missed failures) while "
            "monitoring Precision to avoid false-alarm fatigue. A model prediction is a statistical probability, "
            "not a physical certainty."
        ),
    }

    if save_artifact:
        model_path = get_model_path()
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(results, model_path)

    return results

def load_trained_model() -> Optional[Dict[str, Any]]:
    """Loads serialized model bundle if present."""
    model_path = get_model_path()
    if model_path.exists():
        try:
            return joblib.load(model_path)
        except Exception:
            return None
    return None

def predict_single_instance(
    model_bundle: Dict[str, Any],
    instance_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Predicts failure probability and classification for a single machine reading."""
    pipeline = model_bundle["pipeline"]
    feature_names = model_bundle["feature_names"]

    # Calculate derived features if not present
    inst = instance_dict.copy()
    if "Temp difference [K]" not in inst:
        inst["Temp difference [K]"] = inst["Process temperature [K]"] - inst["Air temperature [K]"]
    if "Power [W]" not in inst:
        angular_vel = inst["Rotational speed [rpm]"] * (2 * np.pi / 60.0)
        inst["Power [W]"] = inst["Torque [Nm]"] * angular_vel

    df_inst = pd.DataFrame([{col: inst.get(col, 0) for col in feature_names}])
    pred = int(pipeline.predict(df_inst)[0])
    prob = float(pipeline.predict_proba(df_inst)[0, 1])

    return {
        "prediction": pred,
        "prediction_label": "High Risk: Predicted Failure" if pred == 1 else "Nominal: Predicted Safe",
        "failure_probability": round(prob * 100, 2),
    }
