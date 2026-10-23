"""
Exploratory Data Analysis (EDA) Page.
Rigorous analytical reasoning, chart selection justifications, and mechanical interpretations.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader import load_dataset
from preprocessing import engineer_features
from visualizations import (
    plot_tool_wear_histogram,
    plot_speed_vs_torque,
    plot_temperatures_scatter,
    plot_failure_modes_bar,
    plot_product_type_distribution,
    plot_correlation_heatmap,
)

st.set_page_config(
    page_title="EDA — Predictive Maintenance",
    page_icon="📈",
    layout="wide",
)

@st.cache_data
def get_data():
    raw_df = load_dataset()
    return engineer_features(raw_df)

df = get_data()

st.title("📈 Exploratory Data Analysis (EDA)")
st.markdown(
    """
    Exploratory visualization enables industrial reliability engineers to uncover physical failure boundaries, 
    diagnose sensor correlations, and isolate anomalous clusters before training predictive classification algorithms.
    """
)

st.markdown("---")

# Chart 1: Tool Wear Distribution
st.header("1. Tool Wear Distribution by Operational Condition")
st.markdown(
    """
    **Analytical Purpose:** Determine whether cumulative cutting wear alone serves as a deterministic 
    trigger for machine failure, or whether failures occur across varied tool ages.
    """
)

col_bins, col_empty = st.columns([2, 4])
with col_bins:
    nbins = st.slider("Select Histogram Bins:", min_value=10, max_value=60, value=30, step=5)

st.plotly_chart(plot_tool_wear_histogram(df, nbins=nbins), use_container_width=True)

with st.expander("🔍 Engineering Insight & Takeaway"):
    st.markdown(
        """
        - **Nominal Operation (Blue/Green):** Uniformly distributed across 0 to 253 minutes because production cycles continuously advance tool wear until planned tool changes occur.
        - **Failures (Red):** The concentration of failure events increases markedly as tool wear exceeds **180–200 minutes**. However, noticeable failures also occur at early tool wear (< 50 minutes).
        - **Actionable Decision:** High tool wear alone **does not guarantee** failure, nor does new tooling guarantee safety. Failures at low wear are triggered by thermal dissipation anomalies (`HDF`) or electrical power surges (`PWF`).
        """
    )

st.markdown("---")

# Chart 2: Spindle Dynamics (Speed vs. Torque)
st.header("2. Spindle Dynamics: Rotational Speed [rpm] vs. Cutting Torque [Nm]")
st.markdown(
    """
    **Analytical Purpose:** Characterize the spindle operating envelope and identify whether mechanical failures 
    occur within nominal cutting parameters or cluster at extreme operating combinations.
    """
)

st.plotly_chart(plot_speed_vs_torque(df), use_container_width=True)

with st.expander("🔍 Engineering Insight & Takeaway"):
    st.markdown(
        r"""
        - **Physical Relationship:** The spindle operates with an inverse hyperbolic torque-speed profile ($P = \tau \cdot \omega$). High rotational speeds yield lower allowable torque, and vice-versa.
        - **Failure Clusters:**
          1. **Low Speed / Ultra-High Torque Cluster (Top-Left):** Cutting torque > 60 Nm combined with speed < 1,350 rpm causes catastrophic tool overstrain (`OSF`) and heat dissipation failures (`HDF`).
          2. **High Speed / Low Torque Cluster (Bottom-Right):** Speeds > 2,500 rpm with torque < 15 Nm breach the lower power boundary ($P < 3.5\text{ kW}$), triggering power failure (`PWF`).
          3. **Nominal Band (Center):** The vast majority of safe machine operations sit inside the nominal 3.5 kW to 9.0 kW envelope.
        """
    )

st.markdown("---")

# Chart 3: Thermal Operating Envelope (Air vs. Process Temperature)
st.header("3. Thermal Operating Envelope: Air vs. Process Temperature [K]")
st.markdown(
    """
    **Analytical Purpose:** Evaluate heat dissipation behavior across varying ambient room temperatures 
    and identify the empirical thermal boundary that triggers heat dissipation failures (`HDF`).
    """
)

st.plotly_chart(plot_temperatures_scatter(df), use_container_width=True)

with st.expander("🔍 Engineering Insight & Takeaway"):
    st.markdown(
        r"""
        - **Linear Tracking:** Process temperature is tightly coupled to ambient air temperature with an average thermal elevation of ~10 K to 11 K under normal machining friction.
        - **Heat Dissipation Threshold (Dashed Red Line):** When the temperature gradient $\Delta T = T_{process} - T_{air} < 8.6\text{ K}$, heat generated at the cutting tool cannot be dissipated effectively by coolant systems.
        - **Actionable Decision:** The dashed red line delineates the physical boundary: machines operating below or close to this boundary when combined with low spindle speed (< 1,380 rpm) account for 100% of recorded `HDF` incidents.
        """
    )

st.markdown("---")

# Charts 4 & 5: Failure Breakdown by Variant and Mechanism
st.header("4. Product Quality Breakdown & Failure Mechanism Counts")
col_var, col_mode = st.columns(2)

with col_var:
    st.subheader("Failures by Product Quality Variant")
    st.plotly_chart(plot_product_type_distribution(df), use_container_width=True)
    st.markdown(
        """
        **Analytical Purpose:** Compare machine failure vulnerability across the three manufactured product grades (`L`, `M`, `H`).
        - Variant `L` exhibits a **3.92%** failure rate.
        - Variant `M` exhibits a **2.77%** failure rate.
        - Variant `H` exhibits a **2.99%** failure rate.
        - **Insight:** Low-variant parts operate under tighter strain margins in tool wear equations, resulting in elevated overstrain risk.
        """
    )

with col_mode:
    st.subheader("Recorded Incidents by Failure Mode")
    failure_mode_counts = {
        "HDF (Heat Dissipation)": int(df["HDF"].sum()),
        "OSF (Overstrain)": int(df["OSF"].sum()),
        "PWF (Power Failure)": int(df["PWF"].sum()),
        "TWF (Tool Wear)": int(df["TWF"].sum()),
        "RNF (Random Failure)": int(df["RNF"].sum()),
    }
    st.plotly_chart(plot_failure_modes_bar(failure_mode_counts), use_container_width=True)
    st.markdown(
        """
        **Analytical Purpose:** Quantify the predominant failure mechanisms to prioritize preventative maintenance capital expenditure.
        - **Thermal (HDF) & Mechanical Overstrain (OSF)** dominate, making spindle load management and coolant flow monitoring the highest ROI intervention points.
        """
    )

st.markdown("---")

# Chart 6: Correlation Heatmap
st.header("5. Linear Correlation Heatmap for Continuous Parameters")
st.markdown(
    """
    **Analytical Purpose:** Assess multi-collinearity among sensor variables to inform feature selection 
    and identify which continuous attributes demonstrate the strongest bivariate correlation with machine failures.
    """
)

st.plotly_chart(plot_correlation_heatmap(df), use_container_width=True)

with st.expander("🔍 Engineering Insight & Takeaway"):
    st.markdown(
        """
        - **Air Temp vs. Process Temp:** Very strong positive correlation ($r \approx 0.88$), confirming that process temperature tracks ambient ambient fluctuations.
        - **Rotational Speed vs. Torque:** Strong inverse correlation ($r \approx -0.88$), reflecting mechanical transmission conservation of power.
        - **Machine Failure Correlations:** `Torque [Nm]` ($r \approx +0.19$) and `Tool wear [min]` ($r \approx +0.11$) display the strongest positive linear correlations with machine failure. However, linear correlation alone underestimates multi-parameter nonlinear failures (such as the product of tool wear and torque).
        """
    )
