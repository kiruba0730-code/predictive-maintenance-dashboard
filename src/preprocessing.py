"""
Preprocessing and feature engineering module for industrial predictive maintenance analytics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes domain-specific mechanical and thermal engineering features:
    - Temperature Difference [K] = Process Temp - Air Temp (Heat dissipation gradient)
    - Mechanical Power [W] = Torque [Nm] * (Rotational Speed [rpm] * 2 * pi / 60)
    - Overstrain Product [min*Nm] = Tool Wear [min] * Torque [Nm]
    - Primary Failure Mode: Resolved classification of failure category
    - Failure Status: Human-readable label ('Failure' vs 'No Failure')
    """
    df_feat = df.copy()

    # Temperature difference (thermal gradient across the milling cutter)
    df_feat["Temp difference [K]"] = (
        df_feat["Process temperature [K]"] - df_feat["Air temperature [K]"]
    ).round(2)

    # Mechanical power in Watts (Power = Torque * Angular Velocity)
    # Angular velocity omega = rpm * 2 * pi / 60
    angular_velocity = df_feat["Rotational speed [rpm]"] * (2 * np.pi / 60.0)
    df_feat["Power [W]"] = (df_feat["Torque [Nm]"] * angular_velocity).round(2)

    # Overstrain product (Wear * Torque)
    df_feat["Overstrain product [min*Nm]"] = (
        df_feat["Tool wear [min]"] * df_feat["Torque [Nm]"]
    ).round(2)

    # Failure Status label
    df_feat["Failure Status"] = df_feat["Machine failure"].map(
        {0: "No Failure", 1: "Failure"}
    )

    # Resolved Failure Mode Category
    def resolve_failure_mode(row) -> str:
        if row["Machine failure"] == 0:
            return "No Failure"
        modes = []
        if row["TWF"] == 1:
            modes.append("TWF (Tool Wear)")
        if row["HDF"] == 1:
            modes.append("HDF (Heat Dissipation)")
        if row["PWF"] == 1:
            modes.append("PWF (Power)")
        if row["OSF"] == 1:
            modes.append("OSF (Overstrain)")
        if row["RNF"] == 1:
            modes.append("RNF (Random)")
            
        if len(modes) == 0:
            return "Unclassified Failure"
        elif len(modes) == 1:
            return modes[0]
        else:
            return f"Multiple ({', '.join(modes)})"

    df_feat["Failure Mode"] = df_feat.apply(resolve_failure_mode, axis=1)

    # Compute Heuristic Risk Score (0-100)
    df_feat["Heuristic Risk Score"] = compute_heuristic_risk(df_feat)

    return df_feat

def compute_heuristic_risk(df: pd.DataFrame) -> pd.Series:
    """
    Calculates a heuristic risk index (0 to 100%) based on proximity to empirical
    operating stress thresholds derived from milling physics:
    - Thermal Stress: Temp diff < 8.6 K with low spindle speed (< 1380 rpm)
    - Power Extremes: Power < 3500 W or > 9000 W
    - Overstrain Stress: High wear combined with high torque
    - Tool Wear Fatigue: Tool wear > 200 min

    NOTE ON LIMITATIONS:
    This score is a rule-based engineering risk index representing operational stress.
    A high score reflects elevated proximity to known stress boundaries, but DOES NOT
    guarantee that a mechanical failure has occurred or will immediately occur.
    """
    risk_scores = np.zeros(len(df), dtype=float)

    # Factor 1: Heat dissipation risk (up to 30 points)
    # HDF threshold: temp diff < 8.6 K and speed < 1380 rpm
    temp_diff = df["Process temperature [K]"] - df["Air temperature [K]"]
    thermal_stress = np.where(
        (temp_diff < 9.0) & (df["Rotational speed [rpm]"] < 1400),
        30.0 * np.clip((9.0 - temp_diff) / 2.0, 0.5, 1.0),
        np.where(temp_diff < 8.6, 15.0, 0.0)
    )
    risk_scores += thermal_stress

    # Factor 2: Power extremity risk (up to 30 points)
    # Normal operating power is ~3500W to ~9000W
    power = df["Torque [Nm]"] * (df["Rotational speed [rpm]"] * 2 * np.pi / 60.0)
    power_stress = np.where(
        power < 3500,
        30.0 * np.clip((3500 - power) / 1000.0, 0.3, 1.0),
        np.where(
            power > 9000,
            30.0 * np.clip((power - 9000) / 2000.0, 0.3, 1.0),
            0.0
        )
    )
    risk_scores += power_stress

    # Factor 3: Tool wear fatigue (up to 25 points)
    # Tool wear typically fails after 200-240 minutes
    tool_wear = df["Tool wear [min]"]
    wear_stress = np.where(
        tool_wear > 180,
        25.0 * np.clip((tool_wear - 180) / 60.0, 0.2, 1.0),
        np.where(tool_wear > 150, 10.0, 0.0)
    )
    risk_scores += wear_stress

    # Factor 4: Overstrain stress (up to 15 points)
    overstrain = tool_wear * df["Torque [Nm]"]
    overstrain_stress = np.where(
        overstrain > 10000,
        15.0 * np.clip((overstrain - 10000) / 3000.0, 0.3, 1.0),
        0.0
    )
    risk_scores += overstrain_stress

    # Clip between 0 and 100 and round into a pandas Series matching df index
    clipped = np.clip(risk_scores, 0.0, 100.0).round(1)
    return pd.Series(clipped, index=df.index, name="Heuristic Risk Score")

def get_executive_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculates all key performance indicators dynamically from data."""
    total_records = int(len(df))
    if total_records == 0:
        return {
            "total_records": 0,
            "total_failures": 0,
            "failure_rate_pct": 0.0,
            "normal_count": 0,
            "failure_modes": {},
            "product_types": {},
            "operating_averages": {},
        }

    total_failures = int(df["Machine failure"].sum())
    normal_count = total_records - total_failures
    failure_rate = round((total_failures / total_records) * 100, 2)

    failure_modes = {
        "TWF (Tool Wear)": int(df["TWF"].sum()),
        "HDF (Heat Dissipation)": int(df["HDF"].sum()),
        "PWF (Power Failure)": int(df["PWF"].sum()),
        "OSF (Overstrain)": int(df["OSF"].sum()),
        "RNF (Random Failure)": int(df["RNF"].sum()),
    }

    product_types = df["Type"].value_counts().to_dict()

    operating_averages = {
        "air_temp_k": round(float(df["Air temperature [K]"].mean()), 2),
        "process_temp_k": round(float(df["Process temperature [K]"].mean()), 2),
        "rotational_speed_rpm": round(float(df["Rotational speed [rpm]"].mean()), 1),
        "torque_nm": round(float(df["Torque [Nm]"].mean()), 2),
        "tool_wear_min": round(float(df["Tool wear [min]"].mean()), 1),
    }

    return {
        "total_records": total_records,
        "total_failures": total_failures,
        "failure_rate_pct": failure_rate,
        "normal_count": normal_count,
        "failure_modes": failure_modes,
        "product_types": product_types,
        "operating_averages": operating_averages,
    }
