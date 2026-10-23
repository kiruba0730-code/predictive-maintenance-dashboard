"""
Visualization module providing interactive Plotly charts adhering to
industrial analytics standards, scientific readability, and CSI3005 assessment rubrics.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, List, Dict

# Standard color palette for industrial predictive maintenance
COLOR_MAP = {
    "No Failure": "#10B981",  # Emerald Green (Nominal state)
    "Failure": "#EF4444",     # Crimson Red (Alarm/Critical state)
    "0": "#10B981",
    "1": "#EF4444",
}

THEME_LAYOUT = dict(
    paper_bgcolor="rgba(15, 23, 42, 0)",
    plot_bgcolor="rgba(30, 41, 59, 0.4)",
    font=dict(color="#F8FAFC", family="Inter, system-ui, sans-serif"),
    xaxis=dict(
        gridcolor="#334155",
        zerolinecolor="#475569",
        linecolor="#475569",
        tickfont=dict(color="#94A3B8"),
        title_font=dict(color="#E2E8F0", size=13),
    ),
    yaxis=dict(
        gridcolor="#334155",
        zerolinecolor="#475569",
        linecolor="#475569",
        tickfont=dict(color="#94A3B8"),
        title_font=dict(color="#E2E8F0", size=13),
    ),
    legend=dict(
        bgcolor="rgba(15, 23, 42, 0.7)",
        bordercolor="#334155",
        borderwidth=1,
        font=dict(color="#E2E8F0"),
    ),
    margin=dict(l=50, r=30, t=50, b=50),
)

def apply_industrial_theme(fig: go.Figure, title: str = "") -> go.Figure:
    """Applies a consistent industrial dark-mode visual theme."""
    fig.update_layout(**THEME_LAYOUT)
    if title:
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=15, color="#F8FAFC", family="Inter, sans-serif"),
                x=0.02,
                y=0.96,
            )
        )
    return fig

# ------------------------------------------------------------------------------
# 1. Executive Overview Charts
# ------------------------------------------------------------------------------

def plot_failure_modes_bar(failure_mode_counts: Dict[str, int]) -> go.Figure:
    """Plots distribution of specific mechanical failure modes."""
    df_plot = pd.DataFrame(
        list(failure_mode_counts.items()), columns=["Failure Mode", "Count"]
    ).sort_values("Count", ascending=True)

    fig = px.bar(
        df_plot,
        x="Count",
        y="Failure Mode",
        orientation="h",
        color="Count",
        color_continuous_scale="Reds",
        text="Count",
    )
    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
        marker_line_color="#7F1D1D",
        marker_line_width=1.2,
    )
    fig.update_layout(
        coloraxis_showscale=False,
        xaxis_title="Number of Recorded Incidents",
        yaxis_title="Failure Mechanism",
    )
    return apply_industrial_theme(fig, "Machine Failures by Specific Mechanism")

def plot_product_type_distribution(df: pd.DataFrame) -> go.Figure:
    """Plots breakdown of product variant types (L, M, H) and failure rates."""
    agg = (
        df.groupby("Type")
        .agg(
            Total=("Machine failure", "count"),
            Failures=("Machine failure", "sum"),
        )
        .reset_index()
    )
    agg["Failure Rate (%)"] = (agg["Failures"] / agg["Total"] * 100).round(2)
    agg["Type Label"] = agg["Type"].map({
        "L": "Low (L) - 60% Target",
        "M": "Medium (M) - 30% Target",
        "H": "High (H) - 10% Target",
    })

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Normal Operations",
            x=agg["Type Label"],
            y=agg["Total"] - agg["Failures"],
            marker_color="#3B82F6",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Failures",
            x=agg["Type Label"],
            y=agg["Failures"],
            marker_color="#EF4444",
            text=agg["Failure Rate (%)"].apply(lambda r: f"{r}% fail"),
            textposition="outside",
        )
    )
    fig.update_layout(
        barmode="group",
        xaxis_title="Product Quality Variant",
        yaxis_title="Number of Production Cycles",
    )
    return apply_industrial_theme(fig, "Production Volume & Failure Rate by Product Variant")

# ------------------------------------------------------------------------------
# 2. Exploratory Data Analysis Charts
# ------------------------------------------------------------------------------

def plot_tool_wear_histogram(df: pd.DataFrame, nbins: int = 30) -> go.Figure:
    """Interactive histogram of tool wear partitioned by machine failure status."""
    fig = px.histogram(
        df,
        x="Tool wear [min]",
        color="Failure Status",
        color_discrete_map=COLOR_MAP,
        nbins=nbins,
        barmode="overlay",
        opacity=0.75,
        marginal="box",
    )
    fig.update_layout(
        xaxis_title="Tool Wear Duration [min]",
        yaxis_title="Count of Machine Cycles",
    )
    return apply_industrial_theme(fig, "Tool Wear Distribution by Machine Status")

def plot_speed_vs_torque(df: pd.DataFrame) -> go.Figure:
    """Rotational speed vs torque scatter plot colored by recorded machine failure."""
    fig = px.scatter(
        df,
        x="Rotational speed [rpm]",
        y="Torque [Nm]",
        color="Failure Status",
        color_discrete_map=COLOR_MAP,
        opacity=0.7,
        hover_data=["UDI", "Product ID", "Type", "Tool wear [min]", "Failure Mode"],
    )
    # Add hyperbolic power reference curves (Power = Torque * omega)
    # 3500 W lower limit and 9000 W upper limit (Power Failure boundaries)
    speeds = np.linspace(1100, 2900, 150)
    torques_low = 3500.0 / (speeds * 2 * np.pi / 60.0)
    torques_high = 9000.0 / (speeds * 2 * np.pi / 60.0)

    fig.add_trace(
        go.Scatter(
            x=speeds,
            y=torques_low,
            mode="lines",
            line=dict(color="#F59E0B", dash="dash", width=1.5),
            name="Power Failure Boundary (3.5 kW)",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=speeds,
            y=torques_high,
            mode="lines",
            line=dict(color="#F97316", dash="dot", width=1.5),
            name="Power Failure Boundary (9.0 kW)",
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        xaxis_title="Spindle Rotational Speed [rpm]",
        yaxis_title="Cutting Torque [Nm]",
    )
    return apply_industrial_theme(fig, "Spindle Rotational Speed vs. Cutting Torque")

def plot_temperatures_scatter(df: pd.DataFrame) -> go.Figure:
    """Air temperature vs process temperature scatter plot."""
    fig = px.scatter(
        df,
        x="Air temperature [K]",
        y="Process temperature [K]",
        color="Failure Status",
        color_discrete_map=COLOR_MAP,
        opacity=0.7,
        hover_data=["UDI", "Product ID", "Temp difference [K]", "Failure Mode"],
    )
    # Add Heat Dissipation boundary line: Process - Air = 8.6 K => Process = Air + 8.6 K
    air_vals = np.linspace(df["Air temperature [K]"].min() - 0.5, df["Air temperature [K]"].max() + 0.5, 50)
    proc_boundary = air_vals + 8.6
    fig.add_trace(
        go.Scatter(
            x=air_vals,
            y=proc_boundary,
            mode="lines",
            line=dict(color="#EF4444", dash="dash", width=1.8),
            name="Thermal Dissipation Threshold (ΔT = 8.6 K)",
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        xaxis_title="Ambient Air Temperature [K]",
        yaxis_title="Machining Process Temperature [K]",
    )
    return apply_industrial_theme(fig, "Thermal Operating Envelope: Air vs. Process Temperature")

def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Correlation matrix heatmap for all key continuous industrial parameters."""
    num_cols = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Temp difference [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Power [W]",
        "Tool wear [min]",
        "Machine failure",
    ]
    cols_present = [c for c in num_cols if c in df.columns]
    corr = df[cols_present].corr().round(3)

    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
    )
    fig.update_layout(
        coloraxis_colorbar=dict(title="Pearson Correlation (r)"),
    )
    return apply_industrial_theme(fig, "Linear Correlation Heatmap of Numerical Features")

# ------------------------------------------------------------------------------
# 3. CSI3005 Assessment Task 2: Choose and Defend Visualizations
# ------------------------------------------------------------------------------

def plot_task2_speed_vs_torque(df: pd.DataFrame) -> go.Figure:
    """
    Task 2 Chart 1: Rotational Speed [rpm] on X-axis, Torque [Nm] on Y-axis,
    colour representing recorded Machine Failure.
    Answers: How do rotational speed and torque relate to recorded machine failures?
    """
    fig = px.scatter(
        df,
        x="Rotational speed [rpm]",
        y="Torque [Nm]",
        color="Failure Status",
        color_discrete_map={"No Failure": "#38BDF8", "Failure": "#EF4444"},
        symbol="Failure Status",
        symbol_map={"No Failure": "circle", "Failure": "diamond"},
        opacity=0.75,
        hover_data=["UDI", "Product ID", "Type", "Tool wear [min]"],
    )
    fig.update_traces(marker=dict(size=7, line=dict(width=0.5, color="#0F172A")))
    fig.update_layout(
        xaxis_title="Spindle Rotational Speed [rpm]",
        yaxis_title="Spindle Cutting Torque [Nm]",
        legend_title="Recorded Status",
    )
    return apply_industrial_theme(fig, "Analytical Scatter: Rotational Speed [rpm] vs. Torque [Nm]")

def plot_task2_toolwear_boxplot(df: pd.DataFrame) -> go.Figure:
    """
    Task 2 Chart 2: Machine Failure status on X-axis (grouped categories),
    Tool Wear [min] on Y-axis.
    Answers: How does tool wear differ between records with and without machine failure?
    """
    fig = px.box(
        df,
        x="Failure Status",
        y="Tool wear [min]",
        color="Failure Status",
        color_discrete_map={"No Failure": "#38BDF8", "Failure": "#EF4444"},
        points="outliers",
        notched=True,
    )
    fig.update_traces(
        boxmean=True,  # displays both mean (dashed) and median (solid) line
        marker=dict(size=4, opacity=0.6),
    )
    fig.update_layout(
        xaxis_title="Machine Operational Condition",
        yaxis_title="Tool Wear Duration [min]",
        showlegend=False,
    )
    return apply_industrial_theme(fig, "Distributional Box Plot: Tool Wear [min] by Failure Status")

# ------------------------------------------------------------------------------
# 4. CSI3005 Assessment Task 3: Critique and Correct Visualizations
# ------------------------------------------------------------------------------

def plot_task3_flawed_chart(df: pd.DataFrame) -> go.Figure:
    """
    Deliberately flawed chart demonstrating 2 severe visual communication violations:
    1. Severely truncated Y-axis (exaggerating minuscule differences).
    2. Misleading / reversed colour encoding (green for dangerous failures, red for normal).
    """
    # Group by product type and calculate average tool wear
    grouped = df.groupby("Type")["Tool wear [min]"].mean().reset_index()
    # Average tool wear is ~107-108 min across all types (almost identical)
    
    fig = go.Figure()
    # Misleading color assignment: Red for high quality, Green for low quality
    misleading_colors = ["#22C55E", "#EAB308", "#EF4444"]
    
    fig.add_trace(
        go.Bar(
            x=grouped["Type"],
            y=grouped["Tool wear [min]"],
            marker_color=misleading_colors,
            text=grouped["Tool wear [min]"].round(2),
            textposition="outside",
        )
    )
    # Flaw 1: Truncating the Y-axis to start at 106 and end at 109!
    # This makes a 0.5 minute (0.4%) variance appear visually 500% larger!
    fig.update_layout(
        yaxis=dict(
            range=[106.5, 108.5],  # SEVERELY TRUNCATED Y-AXIS!
            title="Avg Tool Wear (Distorted Axis) [min]",
        ),
        xaxis_title="Product Type [L, M, H]",
    )
    return apply_industrial_theme(fig, "FLAWED: Average Tool Wear by Product Type (Truncated Axis & Unjustified Colour)")

def plot_task3_corrected_chart(df: pd.DataFrame) -> go.Figure:
    """
    Corrected version of Task 3 chart:
    1. Baseline anchored to zero (0 min), preserving true proportional magnitude.
    2. Neutral, uniform accessible color coding (removes false alarm/positive semantics).
    3. Clear confidence intervals / error bars or full distribution display.
    """
    grouped = (
        df.groupby("Type")["Tool wear [min]"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped["sem"] = grouped["std"] / np.sqrt(grouped["count"])
    grouped["Type Label"] = grouped["Type"].map({"L": "Type L (Low)", "M": "Type M (Medium)", "H": "Type H (High)"})

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=grouped["Type Label"],
            y=grouped["mean"],
            error_y=dict(type="data", array=grouped["sem"] * 1.96, visible=True),
            marker_color="#3B82F6",
            text=grouped["mean"].round(1).apply(lambda v: f"{v} min"),
            textposition="inside",
        )
    )
    # Zero baseline ensures honest representation of relative uniformity
    fig.update_layout(
        yaxis=dict(
            range=[0, 130],
            title="Mean Tool Wear [min] (Zero-Baseline Anchored)",
        ),
        xaxis_title="Product Variant",
    )
    return apply_industrial_theme(fig, "CORRECTED: Average Tool Wear by Product Type (Zero-Baseline & 95% CI)")

# ------------------------------------------------------------------------------
# 5. Continuous Monitoring Demonstration Chart
# ------------------------------------------------------------------------------

def plot_monitoring_stream(
    window_df: pd.DataFrame,
    parameter: str = "Torque [Nm]",
    threshold: Optional[float] = None,
    window_size: int = 50,
) -> go.Figure:
    """
    Renders rolling time-window telemetry stream with moving average and alert threshold.
    """
    fig = go.Figure()

    # Raw telemetry trace
    fig.add_trace(
        go.Scatter(
            x=window_df.index,
            y=window_df[parameter],
            mode="lines+markers",
            name=f"Replayed {parameter}",
            line=dict(color="#38BDF8", width=1.8),
            marker=dict(size=4),
        )
    )

    # 5-point rolling average if sufficient window
    if len(window_df) >= 5:
        rolling_mean = window_df[parameter].rolling(window=5, min_periods=1).mean()
        fig.add_trace(
            go.Scatter(
                x=window_df.index,
                y=rolling_mean,
                mode="lines",
                name="5-Point Rolling Mean",
                line=dict(color="#F59E0B", width=2.2, dash="dash"),
            )
        )

    # Configured threshold alert line
    if threshold is not None:
        fig.add_hline(
            y=threshold,
            line=dict(color="#EF4444", dash="dot", width=2),
            annotation_text=f"Alert Threshold ({threshold})",
            annotation_position="top right",
            annotation_font=dict(color="#EF4444", size=11),
        )

    # Highlight failure points if any in window
    failure_mask = window_df["Machine failure"] == 1
    if failure_mask.any():
        failures = window_df[failure_mask]
        fig.add_trace(
            go.Scatter(
                x=failures.index,
                y=failures[parameter],
                mode="markers",
                name="Failure Recorded",
                marker=dict(color="#EF4444", size=12, symbol="triangle-up", line=dict(width=2, color="#FFFFFF")),
            )
        )

    fig.update_layout(
        xaxis_title="Historical Cycle Sequence (Replay Index)",
        yaxis_title=parameter,
        showlegend=True,
    )
    return apply_industrial_theme(fig, f"Simulated Rolling Telemetry Replay — {parameter}")
