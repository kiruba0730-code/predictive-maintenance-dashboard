# ==============================================================================
# VELLORE INSTITUTE OF TECHNOLOGY (VIT)
# SCHOOL OF COMPUTER SCIENCE AND ENGINEERING
# Course: CSI3005 - Advanced Data Visualization
# Assessment: Case Study - Based Lab Assessment (Assessment 9 & 10)
#
# Project Title: Predictive Maintenance of Industrial Machines Using Advanced Data Visualization
# Dataset: AI4I 2020 Predictive Maintenance Dataset (UCI Repository, ID: 601)
# Author: Kirubakaran
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. Load Required Libraries
# ------------------------------------------------------------------------------
# Run install.packages(c("ggplot2", "dplyr", "tidyr", "corrplot", "gridExtra", "zoo", "scales")) if needed.
suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(scales)
  library(gridExtra)
})

# Optional packages for rolling averages and correlation matrices
has_zoo <- requireNamespace("zoo", quietly = TRUE)
has_corrplot <- requireNamespace("corrplot", quietly = TRUE)

# Set global ggplot theme
theme_industrial <- function() {
  theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 13, color = "#0F172A"),
      plot.subtitle = element_text(color = "#475569", size = 10, margin = margin(b = 10)),
      axis.title = element_text(face = "bold", size = 11, color = "#1E293B"),
      axis.text = element_text(color = "#334155"),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(color = "#E2E8F0"),
      legend.position = "top",
      legend.title = element_text(face = "bold", size = 10),
      legend.background = element_rect(fill = "white", color = "#CBD5E1")
    )
}

# ------------------------------------------------------------------------------
# 2. Data Loading & Preprocessing
# ------------------------------------------------------------------------------
dataset_path <- "data/ai4i2020.csv"
if (!file.exists(dataset_path)) {
  # Fallback check if script is run inside a subfolder
  dataset_path <- "../data/ai4i2020.csv"
}

df <- read.csv(dataset_path, stringsAsFactors = FALSE, check.names = FALSE)

# Clean column names
colnames(df) <- trimws(colnames(df))

# Domain Feature Engineering
df <- df %>%
  mutate(
    # Thermal gradient (Heat Dissipation)
    Temp_difference_K = `Process temperature [K]` - `Air temperature [K]`,
    # Mechanical spindle power in Watts: P = Torque * (rpm * 2 * pi / 60)
    Mechanical_Power_W = `Torque [Nm]` * (`Rotational speed [rpm]` * 2 * pi / 60),
    # Overstrain product: Wear * Torque
    Overstrain_Product = `Tool wear [min]` * `Torque [Nm]`,
    # Human-readable labels
    Failure_Status = factor(ifelse(`Machine failure` == 1, "Failure", "No Failure"), 
                            levels = c("No Failure", "Failure")),
    Product_Type = factor(Type, levels = c("L", "M", "H"), labels = c("Low (L)", "Medium (M)", "High (H)"))
  )

cat("Dataset loaded successfully:", nrow(df), "rows,", ncol(df), "columns.\n")
cat("Total recorded failures:", sum(df$`Machine failure`), "(", round(mean(df$`Machine failure`) * 100, 2), "%)\n")

# ------------------------------------------------------------------------------
# 3. Visualization 1 (Task 2, Chart 1): Spindle Speed vs. Torque Scatter Plot
# Answers: How do rotational speed and torque relate to recorded machine failures?
# ------------------------------------------------------------------------------
# Theoretical Power Failure boundary curves: 3,500 W and 9,000 W
speed_seq <- seq(1150, 2900, length.out = 200)
boundary_low <- data.frame(
  speed = speed_seq,
  torque = 3500 / (speed_seq * 2 * pi / 60),
  boundary = "Lower Power Bound (3.5 kW)"
)
boundary_high <- data.frame(
  speed = speed_seq,
  torque = 9000 / (speed_seq * 2 * pi / 60),
  boundary = "Upper Power Bound (9.0 kW)"
)

p1_speed_torque <- ggplot() +
  geom_point(data = df, aes(x = `Rotational speed [rpm]`, y = `Torque [Nm]`, 
                           color = Failure_Status, shape = Failure_Status), 
             alpha = 0.7, size = 2) +
  geom_line(data = boundary_low, aes(x = speed, y = torque), color = "#F59E0B", linetype = "dashed", size = 1) +
  geom_line(data = boundary_high, aes(x = speed, y = torque), color = "#EA580C", linetype = "dotted", size = 1) +
  scale_color_manual(values = c("No Failure" = "#0284C7", "Failure" = "#DC2626")) +
  scale_shape_manual(values = c("No Failure" = 16, "Failure" = 18)) +
  labs(
    title = "Analytical Scatter: Spindle Rotational Speed vs. Cutting Torque",
    subtitle = "Failures cluster at extreme boundaries: High Torque/Low Speed (Overstrain) and Low Torque/High Speed (Power limit)",
    x = "Spindle Rotational Speed [rpm]",
    y = "Cutting Torque [Nm]",
    color = "Machine Status",
    shape = "Machine Status"
  ) +
  theme_industrial()

# ------------------------------------------------------------------------------
# 4. Visualization 2 (Task 2, Chart 2): Tool Wear Distribution Box Plot
# Answers: How does tool wear differ between records with and without machine failure?
# ------------------------------------------------------------------------------
p2_tool_wear_box <- ggplot(df, aes(x = Failure_Status, y = `Tool wear [min]`, fill = Failure_Status)) +
  geom_boxplot(notch = TRUE, outlier.colour = "#DC2626", outlier.size = 2, alpha = 0.8, width = 0.45) +
  stat_summary(fun = mean, geom = "point", shape = 23, size = 3.5, fill = "yellow", color = "black") +
  scale_fill_manual(values = c("No Failure" = "#38BDF8", "Failure" = "#EF4444")) +
  labs(
    title = "Distributional Box Plot: Tool Wear Duration by Operational Status",
    subtitle = "Yellow diamond denotes mean; notch denotes 95% confidence interval of the median (~148 min vs. ~105 min)",
    x = "Machine Operational Status",
    y = "Tool Wear Contact Duration [min]"
  ) +
  theme_industrial() +
  theme(legend.position = "none")

# ------------------------------------------------------------------------------
# 5. Visualization 3 (Task 3): Critique and Correct
# Compares deliberately flawed chart (truncated Y-axis) with zero-anchored correction
# ------------------------------------------------------------------------------
summary_types <- df %>%
  group_by(Product_Type) %>%
  summarise(
    mean_wear = mean(`Tool wear [min]`),
    sd_wear = sd(`Tool wear [min]`),
    n = n(),
    se = sd_wear / sqrt(n),
    .groups = "drop"
  )

# Flawed: Truncated Y-axis (106.5 - 108.5 min) with misleading traffic-light colors
p3_flawed <- ggplot(summary_types, aes(x = Product_Type, y = mean_wear, fill = Product_Type)) +
  geom_bar(stat = "identity", width = 0.5) +
  coord_cartesian(ylim = c(106.5, 108.5)) +  # SEVERELY TRUNCATED AXIS
  scale_fill_manual(values = c("#22C55E", "#EAB308", "#EF4444")) + # Misleading colors
  labs(
    title = "[FLAWED] Mean Tool Wear by Product Grade",
    subtitle = "Flaw 1: Truncated Y-axis exaggerates 0.8% difference as 400% gap.\nFlaw 2: Inverted traffic-light colors create false alarm.",
    x = "Product Grade",
    y = "Mean Tool Wear [min] (Truncated Axis)"
  ) +
  theme_industrial() +
  theme(legend.position = "none")

# Corrected: Anchored at zero (0 - 130 min) with neutral technical palette and 95% CI error bars
p3_corrected <- ggplot(summary_types, aes(x = Product_Type, y = mean_wear)) +
  geom_bar(stat = "identity", fill = "#2563EB", width = 0.5, alpha = 0.85) +
  geom_errorbar(aes(ymin = mean_wear - 1.96 * se, ymax = mean_wear + 1.96 * se), width = 0.15, color = "#0F172A", size = 0.8) +
  coord_cartesian(ylim = c(0, 130)) +  # HONEST ZERO-BASELINE
  labs(
    title = "[CORRECTED] Mean Tool Wear by Product Grade",
    subtitle = "Correction: Zero-baseline demonstrates true uniformity across grades;\nError bars (95% CI) prove variance is statistically negligible.",
    x = "Product Grade",
    y = "Mean Tool Wear [min] (Zero-Anchored)"
  ) +
  theme_industrial()

# ------------------------------------------------------------------------------
# 6. Visualization 4 (Task 4 / Figure 5): Continuous Monitoring Adaptation
# Dual-layer: discrete raw observations + 5-point moving average + alert threshold
# ------------------------------------------------------------------------------
# Slice a 60-cycle window surrounding a recorded physical failure event (cycle #160)
window_start <- 105
window_end <- 165
stream_window <- df[window_start:window_end, ]
stream_window$cycle_index <- window_start:window_end

# Compute 5-point rolling average
if (has_zoo) {
  stream_window$roll_mean_torque <- zoo::rollmean(stream_window$`Torque [Nm]`, k = 5, fill = NA, align = "right")
} else {
  # Base R fallback for moving average
  stream_window$roll_mean_torque <- stats::filter(stream_window$`Torque [Nm]`, rep(1/5, 5), sides = 1)
}

p4_streaming_figure5 <- ggplot(stream_window, aes(x = cycle_index)) +
  geom_line(aes(y = `Torque [Nm]`), color = "#38BDF8", alpha = 0.6, size = 0.8) +
  geom_point(aes(y = `Torque [Nm]`), color = "#0284C7", size = 2) +
  geom_line(aes(y = roll_mean_torque), color = "#F59E0B", size = 1.2, linetype = "dashed", na.rm = TRUE) +
  geom_hline(yintercept = 60, color = "#DC2626", linetype = "dotted", size = 1) +
  annotate("text", x = window_end - 5, y = 62, label = "Safety Threshold (60 Nm)", color = "#DC2626", fontface = "bold", size = 3.5) +
  geom_point(data = filter(stream_window, `Machine failure` == 1), aes(x = cycle_index, y = `Torque [Nm]`), 
             color = "#DC2626", shape = 17, size = 5) +
  labs(
    title = "Figure 5: Continuous Monitoring Telemetry Showing Recent Readings and Moving-Average Overlay",
    subtitle = "Raw cycles (blue points), 5-point moving average trend (orange dashed), threshold interrupt (red dotted)",
    x = "Historical Cycle Sequence (Replay Index)",
    y = "Cutting Torque [Nm]"
  ) +
  theme_industrial()

# ------------------------------------------------------------------------------
# 7. Visualization 5: Specific Mechanical Failure Modes Distribution
# ------------------------------------------------------------------------------
failure_counts <- data.frame(
  Failure_Mode = c("Heat Dissipation (HDF)", "Overstrain (OSF)", "Power Failure (PWF)", "Tool Wear (TWF)", "Random Failure (RNF)"),
  Count = c(sum(df$HDF), sum(df$OSF), sum(df$PWF), sum(df$TWF), sum(df$RNF))
) %>% arrange(Count)

failure_counts$Failure_Mode <- factor(failure_counts$Failure_Mode, levels = failure_counts$Failure_Mode)

p5_failure_modes <- ggplot(failure_counts, aes(x = Count, y = Failure_Mode, fill = Count)) +
  geom_bar(stat = "identity", width = 0.6) +
  geom_text(aes(label = Count), hjust = -0.2, fontface = "bold", color = "#0F172A", size = 3.8) +
  scale_fill_gradient(low = "#FCA5A5", high = "#991B1B") +
  scale_x_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(
    title = "Mechanical Failure Incident Frequency by Specific Mechanism",
    subtitle = "Heat Dissipation (115) and Overstrain (98) constitute the predominant failure modes",
    x = "Number of Recorded Incidents",
    y = "Failure Mechanism"
  ) +
  theme_industrial() +
  theme(legend.position = "none")

# ------------------------------------------------------------------------------
# 8. Save or Display Plots
# ------------------------------------------------------------------------------
cat("\n=== VISUALIZATION GENERATION COMPLETED ===\n")
cat("Visualizations implemented:\n")
cat("1. p1_speed_torque: Rotational Speed vs. Cutting Torque Scatter Plot\n")
cat("2. p2_tool_wear_box: Tool Wear Box Plot by Operational Condition\n")
cat("3. p3_flawed & p3_corrected: Critique and Correct Misleading Charts\n")
cat("4. p4_streaming_figure5: Figure 5 Continuous Monitoring with Moving Average Overlay\n")
cat("5. p5_failure_modes: Failure Mode Breakdown Bar Chart\n")

# Uncomment below to save plots as PNG files:
# ggsave("figure1_speed_vs_torque.png", p1_speed_torque, width = 8, height = 5, dpi = 300)
# ggsave("figure2_toolwear_boxplot.png", p2_tool_wear_box, width = 8, height = 5, dpi = 300)
# ggsave("figure3_flawed_vs_corrected.png", grid.arrange(p3_flawed, p3_corrected, ncol = 2), width = 12, height = 5, dpi = 300)
# ggsave("figure5_continuous_monitoring.png", p4_streaming_figure5, width = 9, height = 5, dpi = 300)
