"""
Unit test suite for predictive maintenance data processing,
schema validation, feature engineering, and model inference.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Ensure src/ is importable
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader import load_dataset, validate_schema, EXPECTED_COLUMNS, get_data_path
from preprocessing import engineer_features, compute_heuristic_risk, get_executive_kpis
from model import (
    get_features_and_target,
    load_trained_model,
    predict_single_instance,
    build_model_pipeline,
)

@pytest.fixture
def raw_dataset():
    """Loads actual AI4I 2020 dataset."""
    return load_dataset()

@pytest.fixture
def engineered_dataset(raw_dataset):
    """Loads dataset with calculated engineering features."""
    return engineer_features(raw_dataset)

# ------------------------------------------------------------------------------
# 1. Schema and Provenance Tests
# ------------------------------------------------------------------------------

def test_dataset_file_exists():
    path = get_data_path()
    assert path.exists(), f"Dataset not found at expected path: {path}"
    assert path.stat().st_size > 10000, "Dataset file appears truncated or empty."

def test_schema_columns(raw_dataset):
    is_valid, err = validate_schema(raw_dataset)
    assert is_valid, f"Schema validation error: {err}"
    for col in EXPECTED_COLUMNS:
        assert col in raw_dataset.columns, f"Missing required column: {col}"

def test_dataset_dimensions(raw_dataset):
    # Official AI4I 2020 has exactly 10,000 instances
    assert len(raw_dataset) == 10000
    assert raw_dataset["Machine failure"].sum() == 339

def test_no_missing_values(raw_dataset):
    assert raw_dataset.isnull().sum().sum() == 0, "Dataset contains unexpected missing values."

# ------------------------------------------------------------------------------
# 2. Feature Engineering Tests
# ------------------------------------------------------------------------------

def test_temperature_difference(engineered_dataset):
    expected_diff = (
        engineered_dataset["Process temperature [K]"]
        - engineered_dataset["Air temperature [K]"]
    ).round(2)
    pd.testing.assert_series_equal(
        engineered_dataset["Temp difference [K]"], expected_diff, check_names=False
    )

def test_mechanical_power_calculation(engineered_dataset):
    angular_velocity = engineered_dataset["Rotational speed [rpm]"] * (2 * np.pi / 60.0)
    expected_power = (engineered_dataset["Torque [Nm]"] * angular_velocity).round(2)
    pd.testing.assert_series_equal(
        engineered_dataset["Power [W]"], expected_power, check_names=False
    )

def test_heuristic_risk_bounds(engineered_dataset):
    risk = engineered_dataset["Heuristic Risk Score"]
    assert risk.min() >= 0.0
    assert risk.max() <= 100.0

def test_heuristic_risk_single_row_series():
    single_df = pd.DataFrame([{
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 308.0,
        "Rotational speed [rpm]": 1350,
        "Torque [Nm]": 65.0,
        "Tool wear [min]": 210,
    }])
    risk = compute_heuristic_risk(single_df)
    assert isinstance(risk, pd.Series)
    assert hasattr(risk, "iloc")
    assert 0.0 <= float(risk.iloc[0]) <= 100.0

def test_failure_mode_resolution(engineered_dataset):
    # Rows with Machine failure == 0 should have Failure Mode == "No Failure"
    non_failures = engineered_dataset[engineered_dataset["Machine failure"] == 0]
    assert (non_failures["Failure Mode"] == "No Failure").all()

    # Rows with Machine failure == 1 must not be "No Failure"
    failures = engineered_dataset[engineered_dataset["Machine failure"] == 1]
    assert (failures["Failure Mode"] != "No Failure").all()

# ------------------------------------------------------------------------------
# 3. Dynamic KPI Calculation Tests
# ------------------------------------------------------------------------------

def test_executive_kpis(engineered_dataset):
    kpis = get_executive_kpis(engineered_dataset)
    assert kpis["total_records"] == 10000
    assert kpis["total_failures"] == 339
    assert kpis["failure_rate_pct"] == 3.39
    assert kpis["normal_count"] == 9661
    assert kpis["failure_modes"]["TWF (Tool Wear)"] == 46
    assert kpis["failure_modes"]["HDF (Heat Dissipation)"] == 115
    assert kpis["failure_modes"]["PWF (Power Failure)"] == 95
    assert kpis["failure_modes"]["OSF (Overstrain)"] == 98
    assert kpis["failure_modes"]["RNF (Random Failure)"] == 19

# ------------------------------------------------------------------------------
# 4. Anti-Leakage & Machine Learning Pipeline Tests
# ------------------------------------------------------------------------------

def test_anti_leakage_guarantee(engineered_dataset):
    X, y = get_features_and_target(engineered_dataset)
    forbidden_features = [
        "Machine failure",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF",
        "UDI",
        "Product ID",
        "Failure Status",
        "Failure Mode",
        "Heuristic Risk Score",
    ]
    for forbidden in forbidden_features:
        assert (
            forbidden not in X.columns
        ), f"TARGET LEAKAGE DETECTED! Forbidden column '{forbidden}' found in model predictors."

def test_model_artifact_and_inference():
    bundle = load_trained_model()
    if bundle is None:
        pytest.skip("Model artifact not yet serialized.")

    sample_instance = {
        "Type": "L",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 45.0,
        "Tool wear [min]": 120,
    }

    result = predict_single_instance(bundle, sample_instance)
    assert "prediction" in result
    assert result["prediction"] in [0, 1]
    assert 0.0 <= result["failure_probability"] <= 100.0

# ------------------------------------------------------------------------------
# 5. Filtering and Data Export Tests
# ------------------------------------------------------------------------------

def test_data_filtering_and_csv_generation(engineered_dataset):
    # Filter for Product Type 'L' and high torque
    filtered = engineered_dataset[
        (engineered_dataset["Type"] == "L") & (engineered_dataset["Torque [Nm]"] > 50)
    ]
    assert len(filtered) > 0
    assert len(filtered) < len(engineered_dataset)

    # Validate CSV serialization
    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    assert len(csv_bytes) > 0
    reloaded = pd.read_csv(pd.io.common.BytesIO(csv_bytes))
    assert len(reloaded) == len(filtered)
    assert (reloaded["Type"] == "L").all()
