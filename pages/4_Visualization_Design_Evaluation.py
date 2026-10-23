"""
Visualization Design Evaluation Page.
Implements the 4 core tasks of the CSI3005 Advanced Data Visualization assessment:
Task 1: Frame the Problem
Task 2: Choose and Defend
Task 3: Critique and Correct
Task 4: Respond to a Changed Requirement
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader import load_dataset
from preprocessing import engineer_features
from visualizations import (
    plot_task2_speed_vs_torque,
    plot_task2_toolwear_boxplot,
    plot_task3_flawed_chart,
    plot_task3_corrected_chart,
)

st.set_page_config(
    page_title="Design Evaluation — CSI3005",
    page_icon="🎓",
    layout="wide",
)

@st.cache_data
def get_data():
    raw_df = load_dataset()
    return engineer_features(raw_df)

df = get_data()

st.title("🎓 Visualization Design Evaluation (CSI3005 Case Study)")
st.markdown(
    """
    This evaluation module demonstrates the four rigorous tasks required in the **CSI3005 Advanced Data 
    Visualization Techniques** curriculum. It bridges domain engineering requirements with visual design theory, 
    evaluating visual encodings, perceptual ergonomics, graphical integrity, and stream adaptation.
    """
)

# Navigation tabs for the 4 assessment tasks
tab1, tab2, tab3, tab4 = st.tabs([
    "📌 Task 1: Frame the Problem",
    "🎯 Task 2: Choose and Defend",
    "🔍 Task 3: Critique and Correct",
    "⚡ Task 4: Continuous Data Adaptation",
])

# ==============================================================================
# TAB 1: FRAME THE PROBLEM
# ==============================================================================
with tab1:
    st.header("Task 1: Framing the Problem & Analytical Questions")
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.subheader("1. Problem Statement")
        st.markdown(
            """
            In high-throughput CNC milling and manufacturing operations, unplanned machine stoppages 
            lead to severe economic penalties, tool breakage, damaged workpieces, and occupational safety hazards. 
            Traditional maintenance relies on rigid periodic maintenance schedules (which replace healthy tools 
            prematurely) or reactive firefighting (waiting for catastrophic failure).
            
            **The Challenge:** Build an interactive data visualization framework that enables maintenance teams 
            to extract actionable signals from multi-parameter sensor measurements to shift from reactive repairs 
            to proactive condition-based interventions.
            """
        )

        st.subheader("2. Target Audience & Operational Roles")
        st.markdown(
            """
            - **Primary Audience:** Factory Maintenance Engineers & Reliability Specialists.
              - *Characteristics:* High domain fluency in mechanical units (Nm, rpm, K), but requiring immediate, 
                unambiguous visual cues to guide shop-floor physical inspections.
            - **Secondary Audience:** Plant Operations Managers & Production Supervisors.
              - *Characteristics:* Need macro-level visibility into failure distributions across product variants 
                to optimize tool purchasing and tooling life cycle budgets.
            """
        )

    with col_p2:
        st.subheader("3. Decision Objectives")
        st.markdown(
            """
            1. **Prioritize Machine Inspections:** Isolate milling machines operating in extreme torque-speed 
               envelopes or deficient thermal gradients before tool breakage occurs.
            2. **Optimize Tool Replacement Schedules:** Establish data-driven tool wear replacement thresholds 
               rather than relying solely on arbitrary time intervals.
            3. **Diagnose Failure Root Causes:** Differentiate whether an impending machine failure is mechanical 
               overstrain (`OSF`), heat dissipation failure (`HDF`), power failure (`PWF`), or tool fatigue (`TWF`).
            """
        )

        st.subheader("4. Dataset Attributes & Technical Scope")
        st.markdown(
            r"""
            - **Source:** AI4I 2020 Predictive Maintenance Dataset (UCI Repository, 10,000 instances).
            - **Key Measured Parameters:**
              - Ambient Room Temperature ($T_{air}$ [K])
              - Friction Process Temperature ($T_{process}$ [K])
              - Spindle Rotational Speed ($\omega$ [rpm])
              - Cutting Torque ($\tau$ [Nm])
              - Tool Contact Duration ($t_{wear}$ [min])
              - Ground-Truth Binary Failure Status (`Machine failure` $\in \{0, 1\}$)
            """
        )

    st.markdown("---")
    st.subheader("5. The Two Core Analytical Questions")
    
    q_col1, q_col2 = st.columns(2)
    with q_col1:
        st.markdown(
            """
            <div style="background-color: #1E293B; border-left: 4px solid #38BDF8; padding: 16px; border-radius: 6px;">
                <b style="font-size: 1.1rem; color: #38BDF8;">Analytical Question 1:</b><br/>
                <p style="font-size: 1.05rem; color: #F8FAFC; margin-top: 6px;">
                    <b>How do rotational speed and torque relate to recorded machine failures?</b>
                </p>
                <p style="color: #94A3B8; font-size: 0.9rem;">
                    <b>Why it matters to maintenance engineers:</b> Spindle motors must deliver cutting torque 
                    without overloading electrical windings or exceeding tool shank shear strengths. 
                    Understanding whether failures occur across the entire speed-torque curve or exclusively 
                    at extreme operating boundaries allows engineers to define safe CNC feed-rate programming limits.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with q_col2:
        st.markdown(
            """
            <div style="background-color: #1E293B; border-left: 4px solid #10B981; padding: 16px; border-radius: 6px;">
                <b style="font-size: 1.1rem; color: #10B981;">Analytical Question 2:</b><br/>
                <p style="font-size: 1.05rem; color: #F8FAFC; margin-top: 6px;">
                    <b>How does tool wear differ between records with and without machine failure?</b>
                </p>
                <p style="color: #94A3B8; font-size: 0.9rem;">
                    <b>Why it matters to maintenance engineers:</b> Tool inserts progressively dull with cutting contact time. 
                    Engineers must determine whether tool wear exhibits a distinct distributional shift when machines fail, 
                    identifying whether a strict wear cut-off (e.g., 200 min) is sufficient or whether premature failures occur.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ==============================================================================
# TAB 2: CHOOSE AND DEFEND
# ==============================================================================
with tab2:
    st.header("Task 2: Choose and Defend Visualizations")
    st.markdown(
        """
        Implementation and empirical justification of the two principal visualizations designed to resolve 
        Analytical Questions 1 and 2, including comparison with alternative chart choices.
        """
    )

    # Visualization 1: Scatter Plot Speed vs. Torque
    st.markdown("---")
    st.subheader("Principal Visualization 1: Spindle Speed vs. Torque Scatter Plot")
    st.plotly_chart(plot_task2_speed_vs_torque(df), use_container_width=True)

    with st.expander("🛡️ Design Defense & Chart Comparison: Speed vs. Torque Scatter Plot", expanded=True):
        st.markdown(
            """
            | Design Dimension | Implementation Detail | Analytical Rationale |
            | :--- | :--- | :--- |
            | **X-Axis Encoding** | Spindle Rotational Speed [rpm] (Continuous) | Independent variable representing spindle angular velocity. |
            | **Y-Axis Encoding** | Spindle Cutting Torque [Nm] (Continuous) | Dependent mechanical load measured at the cutting interface. |
            | **Colour & Shape** | Nominal: Blue Circle | Failure: Crimson Red Diamond | High perceptual contrast ($L^*a^*b^*$ space) with redundant shape encoding for color-blind accessibility. |
            | **Analytical Question Answered** | Question 1: Speed and torque relationship to machine failure. | Directly demonstrates that machine failures do **not** occur randomly across the space; they cluster at the two extreme hyperbolic boundaries: high-torque/low-speed (overstrain) and low-torque/high-speed (power boundary). |
            
            **Comparison with Alternative Chart (2D Binned Heatmap / Density Contour):**
            - *Alternative Considered:* A 2D rectangular binned heatmap of speed vs. torque counts.
            - *Why the Scatter Plot is Superior:* In our dataset, failures represent only **3.39%** of observations (339 / 10,000). A 2D density heatmap aggregates points into bins dominated by the 9,661 normal operating cycles, completely submerging and obscuring the sparse failure incidents. The scatter plot with an explicit foreground z-order for red diamond failure points ensures that rare catastrophic events remain immediately visible.
            """
        )

    # Visualization 2: Box Plot of Tool Wear by Failure Status
    st.markdown("---")
    st.subheader("Principal Visualization 2: Tool Wear [min] Distribution by Failure Status Box Plot")
    st.plotly_chart(plot_task2_toolwear_boxplot(df), use_container_width=True)

    with st.expander("🛡️ Design Defense & Chart Comparison: Tool Wear Box Plot", expanded=True):
        st.markdown(
            """
            | Design Dimension | Implementation Detail | Analytical Rationale |
            | :--- | :--- | :--- |
            | **X-Axis Encoding** | Machine Operational Condition (Categorical Groups) | Groups data into discrete operational cohorts: **'No Failure'** vs. **'Failure'**. |
            | **Y-Axis Encoding** | Tool Wear Duration [min] (Continuous Numerical) | Continuous mechanical wear duration measured in minutes of cutting contact. |
            | **Visual Marks** | Median line, Interquartile Range (IQR) box, Whiskers (1.5x IQR), Mean line (dashed). | Conveys the full five-number distributional summary without assuming normal distribution. |
            | **Analytical Question Answered** | Question 2: Tool wear difference between failure vs. non-failure. | Conclusively shows that machines experiencing failure have a **substantially higher median tool wear (~148 min vs. ~105 min)** and a markedly elevated upper quartile. |

            **Comparison with Alternative Chart (Grouped Bar Chart of Averages):**
            - *Alternative Considered:* A bar chart comparing the mean tool wear of normal vs. failed machines (107.7 min vs. 143.7 min).
            - *Why the Box Plot is Superior:* A bar chart of means collapses rich distributional spread into single summary numbers, hiding variance, skewness, and outliers. Maintenance engineers need to know the **minimum wear at which failures begin** (represented by the lower whisker and outliers), not merely the average. The notched box plot shows that while failures skew toward high wear, failures also occur down to 0 min, proving that tool wear alone is not the sole trigger.
            """
        )

# ==============================================================================
# TAB 3: CRITIQUE AND CORRECT
# ==============================================================================
with tab3:
    st.header("Task 3: Critique and Correct Misleading Visualizations")
    st.markdown(
        """
        Graphical integrity requires that visual representations do not distort empirical data. 
        Below, we critique a deliberately flawed chart exhibiting two severe visual design violations, 
        demonstrate how it misleads engineers, and present the corrected chart.
        """
    )

    crit_col1, crit_col2 = st.columns(2)

    with crit_col1:
        st.subheader("❌ Deliberately Flawed Visualization")
        st.plotly_chart(plot_task3_flawed_chart(df), use_container_width=True)
        st.error("🚨 VIOLATIONS DETECTED: Truncated Y-Axis + Misleading Inverted Colour Coding")

    with crit_col2:
        st.subheader("✅ Corrected Honest Visualization")
        st.plotly_chart(plot_task3_corrected_chart(df), use_container_width=True)
        st.success("✨ CORRECTIONS: Zero-Baseline Anchoring + Neutral Accessible Palette + 95% Confidence Intervals")

    st.markdown("---")
    st.subheader("Detailed Critique of Design Flaws and Engineering Impact")

    cf_a, cf_b = st.columns(2)

    with cf_a:
        st.markdown(
            """
            ### Specific Flaw 1: Truncated Y-Axis (Baseline Disanchored)
            - **The Flaw:** In the flawed chart, the Y-axis range is truncated to begin at **106.5 minutes** and end at **108.5 minutes**, omitting the true origin ($0\text{ min}$).
            - **How it Misleads the Audience:** 
              - The actual mean tool wear across product variants is virtually identical:
                - Type L: **107.9 min**
                - Type M: **107.7 min**
                - Type H: **108.6 min**
              - The difference between Type M and Type H is a trivial **0.9 minutes (< 0.8% variance)**.
              - By truncating the baseline, the visual bar height for Type H appears **nearly 400% taller** than Type M!
              - A factory reliability manager viewing this would falsely conclude that Type H cutting tools wear out four times faster than Type M tools, potentially leading to erroneous supplier contract penalties or unnecessary machine parameter downgrades.
            """
        )

    with cf_b:
        st.markdown(
            """
            ### Specific Flaw 2: Misleading / Non-Monotonic Colour Encoding
            - **The Flaw:** The flawed chart applies arbitrary traffic-light colors (Green, Yellow, Red) to nominal product categories (L, M, H) where no intrinsic ordered hierarchy exists.
            - **How it Misleads the Audience:**
              - In industrial monitoring, **Green** universally denotes "Safe/Healthy", **Yellow** denotes "Warning", and **Red** denotes "Critical Alarm/Failure".
              - Painting Type H red falsely implies that Type H is dangerously failing or defective, while painting Type L green falsely implies complete safety (despite Type L having the *highest* actual failure count in the plant!).
              - This violates semiotic convention and triggers false visual alarms.
            """
        )

    st.markdown("---")
    st.subheader("How the Correction Restores Graphical Integrity")
    st.markdown(
        """
        1. **Zero-Baseline Anchoring ($0\text{ to }130\text{ min}$):** By extending the scale to the physical zero origin, the heights of the three bars honestly reflect the fact that tool wear contact duration is uniform across all product lines.
        2. **Neutral Visual Semantics:** All three variants are encoded with a consistent, accessible technical blue (`#3B82F6`), eliminating false hazard associations.
        3. **Statistical Uncertainty (95% Confidence Intervals):** Adding standard error error bars demonstrates that the minor variances between L, M, and H are within normal sampling variation and not statistically significant.
        """
    )

# ==============================================================================
# TAB 4: RESPOND TO A CHANGED REQUIREMENT
# ==============================================================================
with tab4:
    st.header("Task 4: Adapting Visualizations to Continuous Data Arrival")
    st.markdown(
        """
        **Scenario:** Plant management upgrades the CNC milling facility with IoT telemetry, requiring 
        the static Speed vs. Torque and Tool Wear visualizations to transition into a continuous condition 
        monitoring dashboard.
        """
    )

    t4_col1, t4_col2 = st.columns(2)

    with t4_col1:
        st.subheader("1. Update Interval & Polling Rate")
        st.markdown(
            """
            - **Selected Frequency:** **1.0 second per dashboard update cycle**.
            - **Engineering Justification:** High-speed spindle encoders generate data at ~1 kHz. Attempting to 
              render raw 1 kHz signals in a web browser produces catastrophic UI freezing and human perceptual overload. 
              Edge PLCs aggregate raw high-frequency waveforms into 1 Hz peak-and-mean telemetry packets, matching the 
              operator's visual absorption threshold.
            """
        )

        st.subheader("2. Display Time Window & Memory Buffering")
        st.markdown(
            r"""
            - **Recommended Window:** Rolling **50 to 100 cycles** (approx. 2 to 5 minutes of real-time operation).
            - **Memory Architecture:** Fixed-size double-ended queue (`collections.deque(maxlen=100)`). As new 
              telemetry cycles arrive, the oldest records are dropped in $\mathcal{O}(1)$ constant time, preventing 
              browser memory leakage during multi-week continuous production shifts.
            """
        )

        st.subheader("3. Individual Readings vs. Aggregated Values")
        st.markdown(
            """
            - **Dual-Trace Architecture:**
              - *Point Marks (Raw):* Every discrete cycle reading is retained as a point to capture instant impulse shocks 
                (e.g., hard workpiece inclusion).
              - *Overlay Line (Smoothed):* A **5-point moving average** is dynamically rendered to highlight the underlying 
                thermal and torque trendline.
            """
        )

    with t4_col2:
        st.subheader("4. Sudden-Change Detection vs. Long-Term Drift")
        st.markdown(
            r"""
            - **Sudden Transients:** Monitored via absolute safety thresholds (e.g., Torque $> 60\text{ Nm}$) and 
              first-order derivative rate-of-change ($\frac{\Delta\tau}{\Delta t} > 15\text{ Nm/s}$) triggers that 
              fire immediate audio-visual alerts.
            - **Long-Term Drift:** Monitored via cumulative tool wear slope, forecasting remaining useful life (RUL) 
              hours before failure limits are reached.
            """
        )

        st.subheader("5. Limitations of Replaying Historical Records")
        st.markdown(
            """
            > [!WARNING]
            > **Streaming Simulation vs. True Factory Floor Telemetry:**
            > 1. **No Hardware Sensor Feedback Loop:** Replaying pre-recorded CSV rows is strictly passive. In a physical factory, an operator would throttle feed rates or abort machining upon receiving an alarm.
            > 2. **Artificial Regularity:** Historical CSV rows arrive with uniform timestamps. Real industrial networks experience packet dropouts, variable network jitter, and out-of-order PLC delivery.
            > 3. **Static Failure Outcomes:** Historical failure rows cannot be avoided by testing different tool change strategies.
            """
        )

    st.markdown("---")
    st.info(
        "💡 **Interactive Demonstration:** To experience this streaming adaptation in action with active controls, "
        "moving average overlays, and threshold alert triggers, navigate to **Page 3: Continuous Monitoring** in the sidebar!"
    )
