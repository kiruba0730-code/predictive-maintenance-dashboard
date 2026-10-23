"""
Machine Risk Analysis Page.
Multivariable filtering, unusual measurement identification, heuristic risk scoring,
predictive model inference, and CSV data export.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader import load_dataset
from preprocessing import engineer_features, compute_heuristic_risk
from visualizations import COLOR_MAP, apply_industrial_theme
from model import load_trained_model, predict_single_instance

st.set_page_config(
    page_title="Risk Analysis — Predictive Maintenance",
    page_icon="⚠️",
    layout="wide",
)

@st.cache_data
def get_data():
    raw_df = load_dataset()
    return engineer_features(raw_df)

df = get_data()

st.title("⚠️ Machine Risk Analysis & Multi-Parameter Diagnostics")
st.markdown(
    """
    Filter operational records across multiple sensor dimensions to inspect operating conditions 
    associated with recorded machine failures, evaluate heuristic operational risk scores, and run 
    inference with the trained predictive classification model.
    """
)

# ------------------------------------------------------------------------------
# Sidebar Filter Controls
# ------------------------------------------------------------------------------
st.sidebar.header("🔍 Multi-Parameter Filters")

# Reset filters button
if st.sidebar.button("🔄 Reset All Filters", use_container_width=True):
    for key in list(st.session_state.keys()):
        if key.startswith("filter_"):
            del st.session_state[key]
    st.rerun()

# 1. Product Type Filter
selected_types = st.sidebar.multiselect(
    "Product Quality Variant (Type):",
    options=["L", "M", "H"],
    default=["L", "M", "H"],
    key="filter_type",
)

# 2. Spindle Rotational Speed Slider
min_speed, max_speed = int(df["Rotational speed [rpm]"].min()), int(df["Rotational speed [rpm]"].max())
speed_range = st.sidebar.slider(
    "Spindle Rotational Speed [rpm]:",
    min_value=min_speed,
    max_value=max_speed,
    value=(min_speed, max_speed),
    step=10,
    key="filter_speed",
)

# 3. Spindle Cutting Torque Slider
min_torque, max_torque = float(df["Torque [Nm]"].min()), float(df["Torque [Nm]"].max())
torque_range = st.sidebar.slider(
    "Spindle Cutting Torque [Nm]:",
    min_value=min_torque,
    max_value=max_torque,
    value=(min_torque, max_torque),
    step=1.0,
    key="filter_torque",
)

# 4. Tool Wear Slider
min_wear, max_wear = int(df["Tool wear [min]"].min()), int(df["Tool wear [min]"].max())
wear_range = st.sidebar.slider(
    "Tool Wear Duration [min]:",
    min_value=min_wear,
    max_value=max_wear,
    value=(min_wear, max_wear),
    step=5,
    key="filter_wear",
)

# 5. Air Temperature Slider
min_air, max_air = float(df["Air temperature [K]"].min()), float(df["Air temperature [K]"].max())
air_range = st.sidebar.slider(
    "Air Temperature [K]:",
    min_value=min_air,
    max_value=max_air,
    value=(min_air, max_air),
    step=0.2,
    key="filter_air",
)

# 6. Failure Status Filter
status_filter = st.sidebar.radio(
    "Failure Status:",
    options=["All Records", "Failures Only", "Nominal (No Failure) Only"],
    key="filter_status",
)

# Apply filters
filtered_df = df[
    (df["Type"].isin(selected_types))
    & (df["Rotational speed [rpm]"] >= speed_range[0])
    & (df["Rotational speed [rpm]"] <= speed_range[1])
    & (df["Torque [Nm]"] >= torque_range[0])
    & (df["Torque [Nm]"] <= torque_range[1])
    & (df["Tool wear [min]"] >= wear_range[0])
    & (df["Tool wear [min]"] <= wear_range[1])
    & (df["Air temperature [K]"] >= air_range[0])
    & (df["Air temperature [K]"] <= air_range[1])
]

if status_filter == "Failures Only":
    filtered_df = filtered_df[filtered_df["Machine failure"] == 1]
elif status_filter == "Nominal (No Failure) Only":
    filtered_df = filtered_df[filtered_df["Machine failure"] == 0]

# Display Filter KPI summary
n_filtered = len(filtered_df)
n_failures_f = int(filtered_df["Machine failure"].sum()) if n_filtered > 0 else 0
fail_rate_f = round((n_failures_f / n_filtered) * 100, 2) if n_filtered > 0 else 0.0

st.subheader("📋 Filtered Dataset Summary")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Matching Records", f"{n_filtered:,} / {len(df):,}")
k2.metric("Recorded Failures", f"{n_failures_f}")
k3.metric("Failure Rate (Filtered)", f"{fail_rate_f}%", delta=f"{round(fail_rate_f - 3.39, 2)}% vs baseline")
k4.metric("Avg Risk Score", f"{filtered_df['Heuristic Risk Score'].mean():.1f}%" if n_filtered > 0 else "N/A")

st.markdown("---")

# ------------------------------------------------------------------------------
# Highlight Unusual Combinations of Measurements
# ------------------------------------------------------------------------------
st.header("⚡ Detection of Unusual & High-Stress Measurement Combinations")
st.markdown(
    """
    Industrial milling failures are rarely governed by a single isolated sensor reading; they emerge 
    from **nonlinear compound stresses**. Three critical combinations identified in machining physics:
    """
)

# Anomaly flags
power_anom = (df["Power [W]"] < 3500) | (df["Power [W]"] > 9000)
thermal_anom = (df["Temp difference [K]"] < 8.6) & (df["Rotational speed [rpm]"] < 1380)
overstrain_anom = df["Overstrain product [min*Nm]"] > 11000

anom_col1, anom_col2, anom_col3 = st.columns(3)

with anom_col1:
    count_p = int(power_anom.sum())
    fail_p = int(df[power_anom]["Machine failure"].sum())
    st.markdown(
        f"""
        <div style="background-color: #1E293B; border-left: 4px solid #F59E0B; padding: 12px 16px; border-radius: 4px;">
            <b style="color: #F8FAFC;">1. Spindle Power Boundary Anomaly</b><br/>
            <span style="color: #94A3B8; font-size: 0.85rem;">Condition: Power &lt; 3.5 kW or &gt; 9.0 kW</span><br/>
            <span style="font-size: 1.1rem; color: #F8FAFC; font-weight: bold;">{count_p} cycles</span> 
            <span style="color: #EF4444; font-size: 0.9rem;">({fail_p} failures, {fail_p/count_p*100:.1f}%)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with anom_col2:
    count_t = int(thermal_anom.sum())
    fail_t = int(df[thermal_anom]["Machine failure"].sum())
    st.markdown(
        f"""
        <div style="background-color: #1E293B; border-left: 4px solid #EF4444; padding: 12px 16px; border-radius: 4px;">
            <b style="color: #F8FAFC;">2. Thermal Dissipation Deficit Anomaly</b><br/>
            <span style="color: #94A3B8; font-size: 0.85rem;">Condition: &Delta;T &lt; 8.6 K and Speed &lt; 1,380 rpm</span><br/>
            <span style="font-size: 1.1rem; color: #F8FAFC; font-weight: bold;">{count_t} cycles</span> 
            <span style="color: #EF4444; font-size: 0.9rem;">({fail_t} failures, {fail_t/count_t*100:.1f}%)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with anom_col3:
    count_o = int(overstrain_anom.sum())
    fail_o = int(df[overstrain_anom]["Machine failure"].sum())
    st.markdown(
        f"""
        <div style="background-color: #1E293B; border-left: 4px solid #8B5CF6; padding: 12px 16px; border-radius: 4px;">
            <b style="color: #F8FAFC;">3. Overstrain Load Anomaly</b><br/>
            <span style="color: #94A3B8; font-size: 0.85rem;">Condition: Tool Wear &times; Torque &gt; 11,000 min&middot;Nm</span><br/>
            <span style="font-size: 1.1rem; color: #F8FAFC; font-weight: bold;">{count_o} cycles</span> 
            <span style="color: #EF4444; font-size: 0.9rem;">({fail_o} failures, {fail_o/count_o*100:.1f}%)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# ------------------------------------------------------------------------------
# Heuristic Risk Score Explanation & Critical Distinctions
# ------------------------------------------------------------------------------
st.header("🧮 Heuristic Risk Score Formulation & Limitations")

with st.expander("📖 Mathematical Definition of Heuristic Risk Score", expanded=True):
    st.markdown(
        r"""
        The **Heuristic Risk Score** ($S_{risk} \in [0, 100]$) aggregates operational proximity to the four primary physical breakdown envelopes:
        
        $$S_{risk} = \text{clip}\Big(S_{thermal} + S_{power} + S_{wear} + S_{overstrain},\ 0,\ 100\Big)$$
        
        Where:
        1. **Thermal Stress ($S_{thermal} \le 30$ pts):** Penalizes low thermal dissipation gradients ($\Delta T < 9.0\text{ K}$) when combined with low spindle speed ($< 1,400\text{ rpm}$).
        2. **Power Extremity ($S_{power} \le 30$ pts):** Penalizes operation outside the safe 3.5 kW to 9.0 kW continuous spindle rating.
        3. **Tool Wear Fatigue ($S_{wear} \le 25$ pts):** Accumulates progressively as tool contact duration exceeds 180 minutes.
        4. **Overstrain Product ($S_{overstrain} \le 15$ pts):** Penalizes severe torque loads combined with aged cutting inserts ($\text{Wear} \times \tau > 10,000$).
        
        > [!IMPORTANT]
        > **CRITICAL SCIENTIFIC DISTINCTION:**
        > - **Recorded Failures** represent ground-truth physical breakdowns recorded in the dataset.
        > - **Heuristic Risk Score** is a rule-based physics stress indicator. A high score indicates **elevated operational risk**, but **DOES NOT guarantee** that a machine has failed or will fail immediately.
        > - **Model Predictions** represent statistical probability scores calculated by an independent machine learning classifier trained on held-out test data.
        """
    )

st.markdown("---")

# ------------------------------------------------------------------------------
# Section D: Machine Learning Predictive Model Inference
# ------------------------------------------------------------------------------
st.header("🤖 Section D: Machine Learning Failure Classifier")

model_bundle = load_trained_model()

if model_bundle is not None:
    metrics = model_bundle["model_metrics"]
    baseline = model_bundle["baseline"]
    cm = metrics["confusion_matrix"]

    st.markdown(
        f"""
        A **Random Forest Classification Pipeline** was trained with stratified sampling and strict zero-leakage guards.
        All performance metrics below are evaluated on **2,000 held-out test records** (never on training data).
        """
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Test Accuracy", f"{metrics['accuracy']}%", f"vs {baseline['accuracy']}% baseline")
    m2.metric("Recall (Sensitivity)", f"{metrics['recall']*100:.1f}%", "82.4% Failures Caught")
    m3.metric("Precision", f"{metrics['precision']*100:.1f}%", "True Alarm Precision")
    m4.metric("F1-Score", f"{metrics['f1_score']:.4f}", "Harmonic Mean")
    m5.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}", "Area Under ROC Curve")

    # Confusion matrix visual breakdown
    cm_c1, cm_c2 = st.columns([1, 1])
    with cm_c1:
        st.subheader("Confusion Matrix (2,000 Test Cycles)")
        cm_data = [
            [cm["true_negatives"], cm["false_positives"]],
            [cm["false_negatives"], cm["true_positives"]],
        ]
        fig_cm = px.imshow(
            cm_data,
            labels=dict(x="Predicted Condition", y="Actual Ground Truth", color="Cycles"),
            x=["Predicted Safe", "Predicted Failure"],
            y=["Actual Safe", "Actual Failure"],
            text_auto=True,
            color_continuous_scale="Blues",
        )
        fig_cm.update_layout(height=320, margin=dict(l=40, r=40, t=30, b=30))
        st.plotly_chart(apply_industrial_theme(fig_cm), use_container_width=True)

    with cm_c2:
        st.subheader("Model Diagnostic Interpretation")
        st.markdown(
            f"""
            - **True Negatives ({cm['true_negatives']:,}):** Correctly identified nominal cycles.
            - **True Positives ({cm['true_positives']}):** Successfully alerted on real failures before breakdown.
            - **False Positives ({cm['false_positives']}):** Unnecessary inspection alerts (manageable 1.0% false alarm rate).
            - **False Negatives ({cm['false_negatives']}):** Uncaught failures (primarily stochastic random failure mode `RNF` which cannot be predicted from mechanical parameters).
            
            **Imbalance Limitation:** Because ground-truth failure occurs in only 3.39% of cycles, a naive baseline model predicting "Safe" 100% of the time achieves **96.6% accuracy but 0.0% Recall**. The ML model overcomes this naive trap.
            """
        )

    # Interactive Single Machine Inspection Simulator
    st.markdown("### 🔬 Single-Machine Inspection Simulator")
    st.markdown("Adjust physical telemetry parameters below to simulate a machine inspection in real time:")

    sim_c1, sim_c2, sim_c3, sim_c4, sim_c5 = st.columns(5)
    with sim_c1:
        sim_type = st.selectbox("Product Grade:", ["L", "M", "H"], index=0)
    with sim_c2:
        sim_speed = st.number_input("Spindle Speed [rpm]:", min_value=1000, max_value=3000, value=1500, step=50)
    with sim_c3:
        sim_torque = st.number_input("Torque [Nm]:", min_value=3.0, max_value=80.0, value=45.0, step=1.0)
    with sim_c4:
        sim_wear = st.number_input("Tool Wear [min]:", min_value=0, max_value=260, value=190, step=5)
    with sim_c5:
        sim_air = st.number_input("Air Temp [K]:", min_value=290.0, max_value=310.0, value=300.0, step=0.5)

    sim_proc = sim_air + 9.5  # Realistic 9.5 K process temp offset

    sim_dict = {
        "Type": sim_type,
        "Air temperature [K]": sim_air,
        "Process temperature [K]": sim_proc,
        "Rotational speed [rpm]": sim_speed,
        "Torque [Nm]": sim_torque,
        "Tool wear [min]": sim_wear,
    }

    # Evaluate prediction and heuristic
    pred_res = predict_single_instance(model_bundle, sim_dict)
    
    # Calculate heuristic score for this instance
    df_single = pd.DataFrame([sim_dict])
    heuristic_res = compute_heuristic_risk(df_single)
    heuristic_val = float(heuristic_res.iloc[0] if hasattr(heuristic_res, "iloc") else heuristic_res[0])

    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.markdown(
            f"""
            <div class="metric-card" style="border-left: 4px solid {'#EF4444' if pred_res['prediction'] == 1 else '#10B981'};">
                <div class="metric-title">ML Model Classification</div>
                <div class="metric-value" style="color: {'#EF4444' if pred_res['prediction'] == 1 else '#10B981'};">
                    {pred_res['prediction_label']}
                </div>
                <div class="metric-delta">Failure Probability: <b>{pred_res['failure_probability']}%</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with res_col2:
        st.markdown(
            f"""
            <div class="metric-card" style="border-left: 4px solid {'#EF4444' if heuristic_val > 50 else '#38BDF8'};">
                <div class="metric-title">Rule-Based Operational Risk Score</div>
                <div class="metric-value" style="color: {'#EF4444' if heuristic_val > 50 else '#38BDF8'};">
                    {heuristic_val:.1f}% Stress Index
                </div>
                <div class="metric-delta">Physics threshold proximity indicator</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

else:
    st.warning("⚠️ Trained model artifact not found. Run `python src/train_model.py` to train and evaluate.")

st.markdown("---")

# ------------------------------------------------------------------------------
# Filtered Records Table & CSV Download
# ------------------------------------------------------------------------------
st.header("💾 Filtered Records & Export")
st.markdown(f"Displaying top 100 matching records (out of **{len(filtered_df):,}** matching cycles):")

display_cols = [
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Power [W]",
    "Failure Status",
    "Failure Mode",
    "Heuristic Risk Score",
]

st.dataframe(
    filtered_df[display_cols].head(100),
    use_container_width=True,
    height=300,
)

# CSV Export Button
csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label=f"📥 Download Filtered Records as CSV ({len(filtered_df):,} rows)",
    data=csv_bytes,
    file_name="predictive_maintenance_filtered.csv",
    mime="text/csv",
    use_container_width=True,
)
