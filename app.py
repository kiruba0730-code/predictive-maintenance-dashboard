"""
Predictive Maintenance of Industrial Machines Using Advanced Data Visualization
Main Application Entry Point & Executive Overview.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd

# Add src to path
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader import load_dataset, get_dataset_metadata
from preprocessing import engineer_features, get_executive_kpis
from visualizations import (
    plot_failure_modes_bar,
    plot_product_type_distribution,
    plot_speed_vs_torque,
)
from model import load_trained_model

# Page configuration
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling for industrial telemetry feel
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #1E293B;
        border-radius: 8px;
        padding: 16px 20px;
        border: 1px solid #334155;
        margin-bottom: 12px;
    }
    .metric-title {
        color: #94A3B8;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #F8FAFC;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .metric-delta {
        font-size: 0.8rem;
        color: #38BDF8;
        margin-top: 2px;
    }
    .status-badge-ok {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10B981;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    .status-badge-warn {
        background-color: rgba(239, 68, 68, 0.2);
        color: #EF4444;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load data with caching
@st.cache_data
def get_processed_data():
    raw_df = load_dataset()
    return engineer_features(raw_df)

try:
    df = get_processed_data()
    kpis = get_executive_kpis(df)
    metadata = get_dataset_metadata()
    data_loaded = True
except Exception as e:
    data_loaded = False
    st.error(f"Failed to load dataset: {e}")

# Sidebar controls & system health
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/engine.png",
        width=64,
    )
    st.title("Industrial Analytics")
    st.caption("CSI3005 Case Study Portfolio")
    st.markdown("---")

    st.subheader("Data Source Status")
    if data_loaded:
        st.markdown('<span class="status-badge-ok">● UCI AI4I 2020 CONNECTED</span>', unsafe_allow_html=True)
        st.caption(f"**Records:** {kpis['total_records']:,} rows")
        st.caption(f"**Failure Rate:** {kpis['failure_rate_pct']}% ({kpis['total_failures']} events)")
    else:
        st.markdown('<span class="status-badge-warn">● DISCONNECTED</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Machine Learning Model")
    model_bundle = load_trained_model()
    if model_bundle is not None:
        metrics = model_bundle["model_metrics"]
        st.markdown('<span class="status-badge-ok">● MODEL DEPLOYED</span>', unsafe_allow_html=True)
        st.caption(f"**Recall (Sensitivity):** {metrics['recall']*100:.1f}%")
        st.caption(f"**Precision:** {metrics['precision']*100:.1f}%")
        st.caption(f"**ROC-AUC:** {metrics['roc_auc']:.3f}")
    else:
        st.markdown('<span class="status-badge-warn">● MODEL NOT FOUND</span>', unsafe_allow_html=True)
        st.caption("Run `python src/train_model.py` to train artifact.")

    st.markdown("---")
    st.info(
        "💡 **Navigation Tip:** Use the sidebar menu above to explore:\n"
        "- **1. Exploratory Analysis**\n"
        "- **2. Machine Risk Analysis**\n"
        "- **3. Continuous Monitoring**\n"
        "- **4. CSI3005 Design Evaluation**"
    )

# Header Section
st.title("⚙️ Predictive Maintenance of Industrial Machines")
st.subheader("Advanced Data Visualization & Condition-Based Monitoring")
st.markdown(
    """
    This enterprise dashboard provides maintenance engineers, reliability managers, and plant supervisors 
    with actionable condition monitoring, operational risk scoring, and interactive diagnostic visualizations 
    based on the **AI4I 2020 Predictive Maintenance Dataset** from the UCI Machine Learning Repository.
    """
)

# Notice on Synthetic Nature of Dataset
st.warning(
    f"⚠️ **Dataset Provenance & Fidelity Notice:** {metadata['synthetic_notice']}"
)

st.markdown("---")

# Section A: Executive Overview KPIs
st.header("📊 Section A: Executive Overview")
st.markdown("Key high-level metrics calculated directly from the verified 10,000 operational records:")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Total Records Analysed</div>
            <div class="metric-value">{kpis['total_records']:,}</div>
            <div class="metric-delta">100% Verified UCI AI4I Records</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Recorded Machine Failures</div>
            <div class="metric-value">{kpis['total_failures']}</div>
            <div class="metric-delta" style="color: #EF4444;">Ground-Truth Failure Events</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Failure Rate</div>
            <div class="metric-value">{kpis['failure_rate_pct']}%</div>
            <div class="metric-delta">Severely Imbalanced (339 / 10,000)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Normal Production Cycles</div>
            <div class="metric-value">{kpis['normal_count']:,}</div>
            <div class="metric-delta" style="color: #10B981;">96.61% Nominal Operations</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Operating Condition Baseline Summary
st.markdown("### 🎛️ Baseline Operating Conditions (Fleet Averages)")
ops = kpis["operating_averages"]
oc1, oc2, oc3, oc4, oc5 = st.columns(5)
oc1.metric("Air Temp [K]", f"{ops['air_temp_k']} K", "Mean Ambient")
oc2.metric("Process Temp [K]", f"{ops['process_temp_k']} K", "Mean Friction Temp")
oc3.metric("Spindle Speed", f"{ops['rotational_speed_rpm']:,} rpm", "Mean Angular Velocity")
oc4.metric("Spindle Torque", f"{ops['torque_nm']} Nm", "Mean Cutting Load")
oc5.metric("Tool Wear", f"{ops['tool_wear_min']} min", "Mean Contact Time")

st.markdown("---")

# Charts row
c_left, c_right = st.columns(2)

with c_left:
    st.plotly_chart(
        plot_failure_modes_bar(kpis["failure_modes"]),
        use_container_width=True,
    )
    st.caption(
        "**Failure Mode Distribution:** Heat Dissipation Failure (HDF) and Overstrain Failure (OSF) "
        "constitute the highest incident counts, followed by Power Failure (PWF) and Tool Wear Failure (TWF). "
        "Random Failures (RNF) represent rare stochastic incidents unlinked to mechanical wear equations."
    )

with c_right:
    st.plotly_chart(
        plot_product_type_distribution(df),
        use_container_width=True,
    )
    st.caption(
        "**Product Variant Distribution:** Low quality variant (L) represents 60% of production volume "
        "and incurs the highest absolute failures. Medium (M, 30%) and High (H, 10%) variants exhibit "
        "progressively stricter mechanical tolerances."
    )

st.markdown("---")

# Overview Scatter Plot: Spindle Operating Envelope
st.subheader("⚡ Operational Spindle Envelope: Speed vs. Cutting Torque")
st.markdown(
    """
    This scatter plot demonstrates the inverse hyperbolic relationship between spindle speed [rpm] 
    and torque [Nm]. The dashed orange curves represent the **3,500 W** and **9,000 W** spindle power 
    boundaries where power failures (`PWF`) physically cluster.
    """
)
st.plotly_chart(plot_speed_vs_torque(df), use_container_width=True)

# Methodology & Dataset Metadata Collapsible Section
with st.expander("ℹ️ Dataset Metadata, Attributes & Academic Citation"):
    st.markdown(
        f"""
        ### AI4I 2020 Predictive Maintenance Dataset
        - **Source:** [{metadata['source']}]({metadata['url']})
        - **Curator:** {metadata['curator']} ({metadata['year']})
        - **Target Attribute:** `Machine failure` (Binary 0/1)
        - **Mechanical Attributes Recorded:**
          1. **UDI:** Unique identifier ranging from 1 to 10,000.
          2. **Product ID:** Variant letter (L/M/H) followed by serial number.
          3. **Type:** Quality variant: Low (60%), Medium (30%), High (10%).
          4. **Air temperature [K]:** Ambient room temperature.
          5. **Process temperature [K]:** Friction-generated tool-workpiece temperature.
          6. **Rotational speed [rpm]:** Spindle velocity for a 2,860 W motor.
          7. **Torque [Nm]:** Cutting resistance on the tool tip.
          8. **Tool wear [min]:** Cumulative cutting time on the milling insert.
          9. **Specific Failure Modes:** `TWF` (Tool Wear), `HDF` (Heat Dissipation), `PWF` (Power), `OSF` (Overstrain), `RNF` (Random).
        
        **Citation:**
        > Matzka, S. (2020). *Explainable Artificial Intelligence for Predictive Maintenance Applications*. 
        > In Third International Conference on Artificial Intelligence for Industries (AI4I 2020), IEEE.
        """
    )
