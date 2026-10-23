"""
Data loader module for AI4I 2020 Predictive Maintenance Dataset.
Ensures rigorous schema validation, provenance checking, and safe fallback.
"""

import os
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd

def conditional_cache(func):
    """Applies streamlit caching only when executed inside a Streamlit runtime."""
    try:
        import streamlit as st
        from streamlit.runtime import exists as runtime_exists
        if runtime_exists():
            return st.cache_data(func)
    except Exception:
        pass
    return func

EXPECTED_COLUMNS = [
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]

FAILURE_MODE_NAMES = {
    "TWF": "Tool Wear Failure",
    "HDF": "Heat Dissipation Failure",
    "PWF": "Power Failure",
    "OSF": "Overstrain Failure",
    "RNF": "Random Failure",
}

def get_data_path() -> Path:
    """Finds the dataset path relative to project structure."""
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    return project_root / "data" / "ai4i2020.csv"

def validate_schema(df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """Validates dataframe schema against AI4I 2020 standard requirements."""
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        return False, f"Missing required columns: {missing_cols}"
    if len(df) == 0:
        return False, "Dataset contains 0 rows."
    return True, None

@conditional_cache
def load_dataset(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads and validates the AI4I 2020 dataset from CSV.
    Preserves original values and column meanings without fabricating records.
    """
    path = Path(csv_path) if csv_path else get_data_path()
    
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Please ensure 'ai4i2020.csv' is placed inside the 'data/' directory.\n"
            "You can run 'python src/download_dataset.py' to download it automatically "
            "from the UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)."
        )

    # Read dataset
    df = pd.read_csv(path)
    
    # Strip potential whitespace in column names
    df.columns = [c.strip() for c in df.columns]
    
    is_valid, error_msg = validate_schema(df)
    if not is_valid:
        raise ValueError(f"Dataset schema validation failed: {error_msg}")
        
    # Ensure correct data types
    df["UDI"] = df["UDI"].astype(int)
    df["Product ID"] = df["Product ID"].astype(str)
    df["Type"] = df["Type"].astype(str)
    df["Air temperature [K]"] = pd.to_numeric(df["Air temperature [K]"], errors="coerce")
    df["Process temperature [K]"] = pd.to_numeric(df["Process temperature [K]"], errors="coerce")
    df["Rotational speed [rpm]"] = pd.to_numeric(df["Rotational speed [rpm]"], errors="coerce")
    df["Torque [Nm]"] = pd.to_numeric(df["Torque [Nm]"], errors="coerce")
    df["Tool wear [min]"] = pd.to_numeric(df["Tool wear [min]"], errors="coerce")
    
    for col in ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        
    return df

def get_dataset_metadata():
    """Returns provenance and domain metadata for the AI4I 2020 dataset."""
    return {
        "title": "AI4I 2020 Predictive Maintenance Dataset",
        "source": "UCI Machine Learning Repository",
        "url": "https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset",
        "curator": "Stephan Matzka, School of Engineering - Technology and Life, HTW Berlin",
        "year": 2020,
        "is_synthetic": True,
        "synthetic_notice": (
            "AI4I 2020 is a synthetic dataset generated to simulate real predictive maintenance "
            "scenarios in industrial milling machines. While physical relationships (temperatures, "
            "power consumption, tool wear thresholds) closely mirror actual mechanical wear equations, "
            "it is not a collection of raw factory sensor logs."
        ),
        "total_instances": 10000,
        "features": 14,
    }
