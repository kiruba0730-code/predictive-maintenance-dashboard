"""
Machine Monitoring Page.
Simulated rolling telemetry replay of historical records with configurable thresholds,
moving average overlays, visual alarm states, and stream adaptation methodology.
"""

import sys
import time
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader import load_dataset
from preprocessing import engineer_features
from visualizations import plot_monitoring_stream

st.set_page_config(
    page_title="Monitoring — Predictive Maintenance",
    page_icon="📡",
    layout="wide",
)

@st.cache_data
def get_data():
    raw_df = load_dataset()
    return engineer_features(raw_df)

df = get_data()

st.title("📡 Continuous Machine Monitoring (Simulated Telemetry Replay)")

# Mandatory provenance & simulation disclaimer
st.info(
    "ℹ️ **Simulation Notice:** This monitoring interface is a **clearly labelled replay of historical dataset records**, "
    "not a live industrial sensor feed. Genuine continuous condition monitoring requires physical hardware connectivity "
    "(such as OPC-UA, MQTT, or Modbus industrial gateways connected to physical spindle accelerometers and thermistors). "
    "Here, historical cycles are stepped through sequentially to demonstrate streaming visualization concepts."
)

st.markdown("---")

# ------------------------------------------------------------------------------
# Playback State Management
# ------------------------------------------------------------------------------
if "replay_index" not in st.session_state:
    st.session_state.replay_index = 60  # Start with 60 cycles of initial history
if "is_playing" not in st.session_state:
    st.session_state.is_playing = False

# Sidebar Controls for Streaming Replay
with st.sidebar:
    st.header("🎛️ Replay & Window Controls")

    parameter_options = [
        "Torque [Nm]",
        "Rotational speed [rpm]",
        "Power [W]",
        "Tool wear [min]",
        "Temp difference [K]",
    ]
    selected_param = st.selectbox(
        "Monitored Telemetry Parameter:",
        options=parameter_options,
        index=0,
    )

    # Dynamic default threshold based on selected parameter
    threshold_defaults = {
        "Torque [Nm]": 60.0,
        "Rotational speed [rpm]": 2400.0,
        "Power [W]": 8500.0,
        "Tool wear [min]": 200.0,
        "Temp difference [K]": 8.6,
    }

    default_thresh = threshold_defaults.get(selected_param, 50.0)
    thresh_val = st.number_input(
        f"Critical Alarm Threshold for {selected_param}:",
        value=float(default_thresh),
        step=1.0 if "rpm" in selected_param or "W" in selected_param else 0.5,
    )

    window_size = st.slider("Rolling Display Window (Cycles):", min_value=20, max_value=120, value=50, step=10)
    playback_speed = st.select_slider(
        "Playback Replay Interval (Simulated Delay):",
        options=[0.2, 0.5, 1.0, 2.0],
        value=0.5,
        format_func=lambda s: f"{s} sec / cycle",
    )

    st.markdown("---")
    st.subheader("Transport Controls")
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    
    with btn_col1:
        if st.button("▶ Step +1", use_container_width=True):
            st.session_state.replay_index = min(st.session_state.replay_index + 1, len(df) - 1)
    
    with btn_col2:
        if st.button("⏩ Step +10", use_container_width=True):
            st.session_state.replay_index = min(st.session_state.replay_index + 10, len(df) - 1)
            
    with btn_col3:
        if st.button("⏮ Reset", use_container_width=True):
            st.session_state.replay_index = window_size

    # Fast jump to known historical failure incident
    st.markdown("---")
    if st.button("🚨 Jump to Next Recorded Failure"):
        future_failures = df.iloc[st.session_state.replay_index:][df["Machine failure"] == 1]
        if len(future_failures) > 0:
            next_idx = future_failures.index[0]
            st.session_state.replay_index = min(next_idx + 5, len(df) - 1)
        else:
            st.session_state.replay_index = 160  # Known early failure event

# Get current rolling slice
current_idx = st.session_state.replay_index
start_idx = max(0, current_idx - window_size)
window_data = df.iloc[start_idx : current_idx + 1].copy()
latest_reading = df.iloc[current_idx]
previous_reading = df.iloc[current_idx - 1] if current_idx > 0 else latest_reading

# Check threshold alarm condition
current_val = latest_reading[selected_param]
is_alarm = False
if "Temp diff" in selected_param:
    is_alarm = current_val < thresh_val  # Low temp diff is dangerous
else:
    is_alarm = current_val > thresh_val  # High value is dangerous

# Display Alarm Banner if triggered
if is_alarm:
    st.error(
        f"🚨 **CRITICAL ALARM TRIGGERED:** Current {selected_param} reading ({current_val}) "
        f"has crossed configured safety threshold ({thresh_val})! Immediate inspection recommended."
    )
elif latest_reading["Machine failure"] == 1:
    st.error(
        f"💥 **RECORDED MACHINE FAILURE:** Historical dataset record index #{current_idx} "
        f"experienced a ground-truth breakdown ({latest_reading['Failure Mode']})."
    )
else:
    st.success(
        f"✅ **NORMAL OPERATIONAL STATE:** Current {selected_param} reading ({current_val}) "
        f"is within nominal operating envelope."
    )

# Current Readings Live Cards
st.subheader(f"Current Replay Sequence: Cycle #{current_idx:,} / {len(df):,}")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(
    "Spindle Torque",
    f"{latest_reading['Torque [Nm]']} Nm",
    f"{round(latest_reading['Torque [Nm]'] - previous_reading['Torque [Nm]'], 1)} Nm",
)
c2.metric(
    "Spindle Speed",
    f"{latest_reading['Rotational speed [rpm]']} rpm",
    f"{int(latest_reading['Rotational speed [rpm]'] - previous_reading['Rotational speed [rpm]'])} rpm",
)
c3.metric(
    "Tool Wear",
    f"{latest_reading['Tool wear [min]']} min",
    f"+{int(latest_reading['Tool wear [min]'] - previous_reading['Tool wear [min]'])} min",
)
c4.metric(
    "Spindle Power",
    f"{latest_reading['Power [W]']:.0f} W",
    f"{int(latest_reading['Power [W]'] - previous_reading['Power [W]'])} W",
)
c5.metric(
    "Risk Score",
    f"{latest_reading['Heuristic Risk Score']}%",
    "Physics Stress Index",
)

st.markdown("---")

# Rolling Telemetry Chart
st.subheader(f"📈 Rolling Window Stream ({window_size} Historical Cycles)")
st.plotly_chart(
    plot_monitoring_stream(
        window_data,
        parameter=selected_param,
        threshold=thresh_val,
        window_size=window_size,
    ),
    use_container_width=True,
)

# ------------------------------------------------------------------------------
# Task 4 Companion Analysis: Adapting Visualizations to Continuous Data
# ------------------------------------------------------------------------------
st.markdown("---")
st.header("🔬 Engineering Analysis: Adapting Charts to Continuous Sensor Telemetry")
st.markdown(
    """
    Transitioning a static exploratory chart into an operational streaming monitoring system 
    requires addressing five fundamental real-time visualization engineering constraints:
    """
)

col_t4_a, col_t4_b = st.columns(2)

with col_t4_a:
    st.markdown(
        """
        ### 1. Recommended Update Interval & Polling Rate
        - **Spindle Vibration / Cutting Torque:** Physical cutting forces oscillate at spindle tooth passing frequencies (~50 Hz to 2 kHz). While high-speed edge hardware samples at kilohertz frequencies, sending raw waves to browser dashboards causes catastrophic DOM thrashing.
        - **Dashboard Refresh Rate:** The human visual reaction threshold is **1.0 to 2.0 seconds** for supervisor dashboards, with a local 500 ms sub-second refresh for acute condition monitoring.
        
        ### 2. Rolling Display Time Window
        - A fixed **50 to 100 cycle rolling window** (approx. 2–5 minutes of operational context) provides the optimal balance between high-frequency transient awareness and historical context.
        - Older records are aged out to fixed-size circular memory buffers (`collections.deque`), preventing memory exhaustion.
        """
    )

with col_t4_b:
    st.markdown(
        """
        ### 3. Individual Readings vs. Aggregated Values
        - **Dual-Layer Architecture:**
          - **Raw Telemetry:** Plotted as individual high-frequency points to catch sudden impulse spikes or instantaneous tool fractures.
          - **Moving Average (5-Point / 10-Point Rolling Mean):** Overlaid to reveal continuous underlying thermal and mechanical trends without sensor noise distraction.
        
        ### 4. Sudden-Change Detection vs. Long-Term Drift
        - **Sudden Changes:** Handled via threshold triggers and rate-of-change ($\frac{d\tau}{dt}$) interrupts that render immediate visual alert banners (e.g., Crimson alert shown above).
        - **Long-Term Drift:** Tracked via tool wear accumulation and baseline temperature shift over hundreds of cycles.
        """
    )

with st.expander("⚠️ Critical Operational Limitations of Replaying Historical Records"):
    st.markdown(
        """
        1. **Deterministic Pre-Recorded Sequence:** In historical replay, past machine failures are already fixed in time. In real production, operator intervention (such as reducing spindle feed rate or stopping the program) dynamically changes future telemetry.
        2. **Absence of Hardware Latency & Packet Drop:** Real IoT sensor gateways encounter network jitter, dropped UDP/MQTT packets, and asynchronous clock drift across disparate machine PLCs.
        3. **No Closed-Loop Control:** A dashboard replay is strictly an observational display. Real industrial systems require hard real-time safety interlocks connected to machine emergency-stop circuits.
        """
    )
