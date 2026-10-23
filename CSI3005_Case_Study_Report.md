# VELLORE INSTITUTE OF TECHNOLOGY (VIT)
## SCHOOL OF COMPUTER SCIENCE AND ENGINEERING
### Innovative Evaluation Practices for Laboratory Courses – Fall Semester 2026-27
**Case Study - Based Lab Assessment (Assessment 9 & 10)**  
**Course Code & Title:** CSI3005 – Advanced Data Visualization  
**Slot:** L27+28 & L35+36 | **Maximum Marks:** 20  

---

# Case Study Assessment Report: Predictive Maintenance of Industrial Machines Using Advanced Data Visualization

* **Student Name:** Kirubakaran
* **Degree / Programme:** B.Tech Computer Science and Engineering
* **Institution:** Vellore Institute of Technology (VIT), India
* **GitHub Repository:** [https://github.com/kiruba0730-code/predictive-maintenance-dashboard](https://github.com/kiruba0730-code/predictive-maintenance-dashboard)
* **Live Interactive Application:** [http://localhost:8501](http://localhost:8501) *(and Streamlit Community Cloud)*
* **Implementation Languages & Libraries:** 
  * **R:** `ggplot2`, `dplyr`, `tidyr`, `scales`, `zoo`, `gridExtra`, `corrplot` (`case_study_visualization.R`)
  * **Python:** `Streamlit`, `Plotly`, `Pandas`, `NumPy`, `Scikit-learn` (`app.py`, `src/`)

---

## Mapping with Evaluation Rubric (20 Marks Total)

| Evaluation Component | Maximum Marks | Section in This Report |
| :--- | :---: | :--- |
| **Component 1: Case Study, Dataset Selection & Data Preparation** | **5 Marks** | Section 1 (Problem & Objectives) & Section 2 (Dataset & Preprocessing) |
| **Component 2: Visualization Selection & Implementation** | **5 Marks** | Section 3 (Task 2 Principal Visualizations & R / Python Implementation) |
| **Component 3: Visualization Design & Presentation** | **5 Marks** | Section 4 (Task 3 Critique & Correction) & Section 5 (Task 4 / Figure 5 Streaming Adaptation) |
| **Component 4: Interpretation, Insights & Conclusion** | **5 Marks** | Section 6 (Analytical Insights), Section 7 (ML Pipeline), Section 8 (Engineering Conclusions) |

---

## 1. Title, Problem Description & Objectives (Component 1)

### 1.1 Title of the Case Study
**Predictive Maintenance of Industrial Machines Using Advanced Data Visualization**

### 1.2 Domain & Brief Description of the Real-World Problem
In modern automated manufacturing (such as high-speed computer numerical control milling and boring), cutting tools operate under intense mechanical torque loads and friction heat. Unplanned machine breakdowns lead to catastrophic equipment stoppages, scrapped workpieces, expensive secondary spindle damage, and factory worker hazards.

Traditional industrial maintenance operates at two inefficient extremes:
1. **Preventive / Periodic Maintenance:** Tools are replaced at fixed calendar or operating hour intervals. This discards millions of dollars worth of healthy cutting inserts prematurely.
2. **Corrective / Reactive Maintenance:** Machines are run until the cutter snaps or overheats in the workpiece, causing unscheduled downtime and costly repair cycles.

**The Case Study Problem:** How can factory reliability engineers and maintenance supervisors transform continuous industrial sensor telemetry into intuitive, actionable, and distortion-free visual representations to detect impending machine failure modes before catastrophic tool failure occurs?

### 1.3 Objective and Scope of the Analysis
The overarching goal is to establish an end-to-end data visualization framework that:
1. **Identifies Physical Operating Boundaries:** Characterize the safe cutting envelope across spindle rotational speeds, cutting torques, and thermal dissipation gradients.
2. **Prioritizes Physical Inspections:** Identify which machines and tool wear thresholds carry the highest probability of failure.
3. **Ensures Graphical Integrity:** Demonstrate ethical data visualization by critiquing and correcting misleading visualizations (truncated baselines and inverted hazard semantics).
4. **Adapts to Continuous Streaming Telemetry:** Model how static charts adapt to real-time IoT sensor telemetry streams with rolling memory windows, moving averages, and threshold alarms.
5. **Classifies Failure Modes:** Correlate sensor parameters with ground-truth failure mechanisms without data leakage.

---

## 2. Dataset Selection, Description & Preprocessing (Component 1)

### 2.1 Dataset Source and Provenance
* **Dataset Name:** AI4I 2020 Predictive Maintenance Dataset
* **Repository:** [UCI Machine Learning Repository (Dataset ID: 601)](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
* **Curator:** Stephan Matzka, School of Engineering - Technology and Life, HTW Berlin, Germany
* **Year of Publication:** 2020
* **Instances:** 10,000 recorded operational machine cycles
* **Target Feature:** `Machine failure` (Binary: 0 = Nominal, 1 = Failure)

> [!IMPORTANT]
> **Notice on Dataset Provenance & Synthetic Nature:**
> The AI4I 2020 dataset is a **synthetic dataset** designed to simulate real predictive maintenance scenarios in industrial milling machines. While its physical relationships (cutting power curves, thermal heat dissipation, overstrain products) strictly mirror empirical milling physics, it is not raw sensor logs collected from an active factory floor. Original values from the UCI repository are strictly preserved without fabricated records.

### 2.2 Important Attributes / Variables Considered

| Attribute Name | Variable Type | Engineering Unit | Analytical & Physical Meaning |
| :--- | :--- | :--- | :--- |
| `UDI` | Numerical (Integer) | None | Unique sequential cycle index from 1 to 10,000. |
| `Product ID` | Categorical (String) | None | Product identifier prefixed by quality grade (`L`, `M`, `H`). |
| `Type` | Categorical Factor | Dimensionless | Product variant: Low (60% volume), Medium (30%), High (10%). |
| `Air temperature [K]` | Numerical (Continuous) | Kelvin (K) | Ambient environment room temperature (Mean: 300.0 K, Std: 2.0 K). |
| `Process temperature [K]`| Numerical (Continuous) | Kelvin (K) | Cutting friction temperature generated at tool tip (Mean: 310.0 K). |
| `Rotational speed [rpm]` | Numerical (Integer) | Rev/min (rpm) | Spindle angular velocity for a 2,860 W motor (1,168 to 2,886 rpm). |
| `Torque [Nm]` | Numerical (Continuous) | Newton-meters | Cutting resistance load exerted on the tool tip (3.8 to 76.6 Nm). |
| `Tool wear [min]` | Numerical (Integer) | Minutes (min) | Cumulative active machining contact duration on insert (0 to 253 min). |
| `Machine failure` | Binary Indicator | $\{0, 1\}$ | Ground-truth machine breakdown event (339 failures, 9,661 nominal). |
| `TWF` | Binary Flag | $\{0, 1\}$ | **Tool Wear Failure:** Tool wear replacement limit exceeded (46 events). |
| `HDF` | Binary Flag | $\{0, 1\}$ | **Heat Dissipation Failure:** $\Delta T < 8.6\text{ K}$ and speed $< 1,380\text{ rpm}$ (115 events). |
| `PWF` | Binary Flag | $\{0, 1\}$ | **Power Failure:** Spindle power $< 3,500\text{ W}$ or $> 9,000\text{ W}$ (95 events). |
| `OSF` | Binary Flag | $\{0, 1\}$ | **Overstrain Failure:** $\text{Tool Wear} \times \text{Torque} > \text{Threshold}$ (98 events). |
| `RNF` | Binary Flag | $\{0, 1\}$ | **Random Failure:** Stochastic breakdown unlinked to wear ($p \approx 0.1\%$, 19 events). |

### 2.3 Data Cleaning & Preprocessing Performed
1. **Schema Validation & Missing Value Audit:**
   * Audited all 10,000 rows across 14 columns. Confirmed **0 missing values** (`NA`/`null`), preventing data corruption.
   * Stripped potential leading/trailing whitespace from header strings.
2. **Domain-Specific Feature Engineering:**
   * **Thermal Dissipation Gradient ($\Delta T$):**
     $$\text{Temp\_difference\_K} = \text{Process temperature [K]} - \text{Air temperature [K]}$$
     Measures the efficiency of coolant heat removal.
   * **Mechanical Spindle Power ($P$ in Watts):**
     $$\text{Power\_W} = \text{Torque [Nm]} \times \left(\text{Rotational speed [rpm]} \times \frac{2\pi}{60}\right)$$
     Derived from rotational angular velocity ($\omega$).
   * **Overstrain Mechanical Stress Product:**
     $$\text{Overstrain\_Product} = \text{Tool wear [min]} \times \text{Torque [Nm]}$$
     Quantifies the compound fatigue load on weakened cutting inserts.
3. **Factor Encoding:**
   * Converted `Machine failure` into an ordered factor: `"No Failure"` (96.61%) vs. `"Failure"` (3.39%).
   * Formatted product quality variants into descriptive factor levels: `"Low (L)"`, `"Medium (M)"`, and `"High (H)"`.

---

## 3. Visualization Selection & Implementation (Component 2 — 5 Marks)

### 3.1 Framing the Problem (Assessment Task 1)
* **Audience:** Factory maintenance engineers and reliability managers requiring unambiguous visual cues to schedule physical tool inspections.
* **Core Analytical Questions:**
  1. *How do rotational speed and cutting torque relate to recorded machine failures?*
  2. *How does tool wear differ between records with and without machine failure?*

---

### 3.2 Principal Visualization 1: Spindle Speed vs. Torque Scatter Plot (Task 2 Chart 1)

* **Analytical Question Answered:** How do rotational speed and torque relate to recorded machine failures?
* **Visual Encodings:**
  * **X-Axis:** Spindle Rotational Speed [rpm] (Continuous)
  * **Y-Axis:** Spindle Cutting Torque [Nm] (Continuous)
  * **Color:** Emerald Green / Sky Blue (`#0284C7`) for Nominal; Crimson Red (`#DC2626`) for Machine Failure.
  * **Shape:** Circle (Nominal) vs. Diamond (Failure) for color-blind accessibility.
  * **Boundary Lines:** Hyperbolic spindle power boundary curves ($P = 3,500\text{ W}$ and $P = 9,000\text{ W}$).

```r
# R Implementation (case_study_visualization.R)
p1_speed_torque <- ggplot() +
  geom_point(data = df, aes(x = `Rotational speed [rpm]`, y = `Torque [Nm]`, 
                           color = Failure_Status, shape = Failure_Status), 
             alpha = 0.7, size = 2) +
  geom_line(data = boundary_low, aes(x = speed, y = torque), color = "#F59E0B", linetype = "dashed", size = 1) +
  geom_line(data = boundary_high, aes(x = speed, y = torque), color = "#EA580C", linetype = "dotted", size = 1) +
  scale_color_manual(values = c("No Failure" = "#0284C7", "Failure" = "#DC2626")) +
  scale_shape_manual(values = c("No Failure" = 16, "Failure" = 18)) +
  labs(title = "Analytical Scatter: Rotational Speed vs. Cutting Torque",
       x = "Spindle Rotational Speed [rpm]", y = "Cutting Torque [Nm]") +
  theme_industrial()
```

*(Insert Screenshot: Principal Visualization 1 from Page 4 Tab 2)*

* **Defense vs. Alternative (2D Binned Heatmap / Density Contour):**
  * In this dataset, machine failures represent a severely imbalanced minority class (only 3.39% of cycles). A 2D density heatmap aggregates counts into bins dominated by the 9,661 normal operating cycles, completely submerging and concealing rare failure points. The scatter plot preserves isolated critical outliers and guarantees that every failure incident is visibly prominent.

---

### 3.3 Principal Visualization 2: Tool Wear Box Plot by Operational Status (Task 2 Chart 2)

* **Analytical Question Answered:** How does tool wear differ between records with and without machine failure?
* **Visual Encodings:**
  * **X-Axis:** Machine Operational Status (Categorical Groups: `"No Failure"` vs. `"Failure"`)
  * **Y-Axis:** Tool Wear Contact Duration [min] (Continuous Numerical)
  * **Marks:** Notched median bar (95% CI of median), Interquartile Range (IQR) box, 1.5× IQR whiskers, yellow mean diamond, outlier circles.

```r
# R Implementation (case_study_visualization.R)
p2_tool_wear_box <- ggplot(df, aes(x = Failure_Status, y = `Tool wear [min]`, fill = Failure_Status)) +
  geom_boxplot(notch = TRUE, outlier.colour = "#DC2626", outlier.size = 2, alpha = 0.8, width = 0.45) +
  stat_summary(fun = mean, geom = "point", shape = 23, size = 3.5, fill = "yellow", color = "black") +
  scale_fill_manual(values = c("No Failure" = "#38BDF8", "Failure" = "#EF4444")) +
  labs(title = "Distributional Box Plot: Tool Wear Duration by Operational Status",
       x = "Machine Operational Status", y = "Tool Wear Contact Duration [min]") +
  theme_industrial()
```

*(Insert Screenshot: Principal Visualization 2 from Page 4 Tab 2)*

* **Defense vs. Alternative (Grouped Bar Chart of Averages):**
  * A bar chart comparing means (107.7 min vs. 143.7 min) reduces rich variance to single numbers, concealing skewness, distribution spread, and minimum failure limits. Maintenance engineers need to know the minimum threshold at which failures begin (lower whisker and outliers), not just the average. The notched box plot preserves the five-number summary and proves that while failures skew toward higher wear, premature failures also occur down to 0 min.

---

## 4. Visualization Design, Presentation & Critique (Component 3 — 5 Marks)

### 4.1 Critique and Correction of Misleading Visualizations (Assessment Task 3)

#### The Deliberately Flawed Visualization
*(Insert Screenshot: Flawed Chart from Page 4 Tab 3)*

* **Specific Design Flaw 1 — Severely Truncated Y-Axis (Baseline Disanchored):**
  * *Implementation:* The Y-axis is constrained between **106.5 minutes** and **108.5 minutes**, omitting the true origin ($0\text{ min}$).
  * *How it Misleads:* Actual mean tool wear across variants is nearly identical: Type L = 107.9 min, Type M = 107.7 min, Type H = 108.6 min (a negligible 0.8% difference). Truncating the axis visually magnifies this 0.9-minute difference into an apparent **400% visual gap**, misleading plant leadership into falsely assuming that Type H cutting tools degrade four times faster.
* **Specific Design Flaw 2 — Misleading / Inverted Colour Semantics:**
  * *Implementation:* Applying arbitrary traffic-light colors (Green for Type L, Yellow for Type M, Red for Type H) to nominal product categories.
  * *How it Misleads:* In industrial engineering, green universally signifies "Safe" and red signifies "Hazard/Failure". Painting Type H red triggers false emergency alarms, while painting Type L green falsely communicates zero risk (even though Type L has the highest absolute breakdown count in the plant).

#### The Corrected Visualization
*(Insert Screenshot: Corrected Chart from Page 4 Tab 3)*

* **Correction 1 — Zero-Baseline Anchoring ($0\text{ to }130\text{ min}$):** Honoring the physical zero baseline truthfully represents that tool wear contact duration is uniform across all product lines.
* **Correction 2 — Neutral Technical Palette:** All bars use uniform technical blue (`#2563EB`), eliminating false hazard semantics.
* **Correction 3 — Statistical Error Bars (95% Confidence Intervals):** Adding standard error bars ($1.96 \times \text{SE}$) proves that the minor differences between variants are within expected sampling variation and not statistically significant.

```r
# R Implementation of Corrected Chart (case_study_visualization.R)
p3_corrected <- ggplot(summary_types, aes(x = Product_Type, y = mean_wear)) +
  geom_bar(stat = "identity", fill = "#2563EB", width = 0.5, alpha = 0.85) +
  geom_errorbar(aes(ymin = mean_wear - 1.96 * se, ymax = mean_wear + 1.96 * se), 
                width = 0.15, color = "#0F172A", size = 0.8) +
  coord_cartesian(ylim = c(0, 130)) +
  labs(title = "CORRECTED: Mean Tool Wear by Product Grade (Zero-Baseline & 95% CI)",
       x = "Product Grade", y = "Mean Tool Wear [min] (Zero-Anchored)") +
  theme_industrial()
```

---

## 5. Continuous Data Arrival Adaptation & Streaming Visualization (Task 4 / Figure 5)

When adapting static charts to continuous industrial IoT telemetry streams:

### 5.1 Update Interval & Polling Rate (1.0 Second)
Spindle vibration sensors and rotary encoders operate at ~1 kHz. Attempting to render raw 1 kHz waveforms inside a web browser causes severe UI freezing and cognitive overload. Edge PLCs downsample and aggregate raw signals into **1 Hz (1.0 second) telemetry packets**, perfectly matching the human visual absorption threshold.

### 5.2 Display Window & Memory Buffer (50–100 Cycles — Section 5.3)
The display window retains the **most recent 50–100 cycles** (approx. 2 to 5 minutes of real-time operation). A fixed-size circular buffer, such as Python's `collections.deque(maxlen=100)`, bounds memory consumption and prevents browser memory leakage by evicting older entries in $\mathcal{O}(1)$ constant time.

### 5.3 Individual Readings and Aggregated Values (Section 5.4)
The telemetry view combines two complementary visual layers:
* **Individual Readings (Point Marks):** Shows every raw cycle observation so sudden spikes, chip breakage shocks, or transient load spikes remain immediately visible.
* **Five-Point Moving Average (Smoothed Line):** Overlays a 5-point rolling mean to reveal the underlying thermal and torque trend without sensor noise distraction.

*(Insert Screenshot from Page 4 Tab 4)*  
**Figure 5: Continuous Monitoring page showing recent readings and the moving-average overlay**

```r
# R Implementation of Figure 5 (case_study_visualization.R)
p4_streaming_figure5 <- ggplot(stream_window, aes(x = cycle_index)) +
  geom_line(aes(y = `Torque [Nm]`), color = "#38BDF8", alpha = 0.6, size = 0.8) +
  geom_point(aes(y = `Torque [Nm]`), color = "#0284C7", size = 2) +
  geom_line(aes(y = roll_mean_torque), color = "#F59E0B", size = 1.2, linetype = "dashed") +
  geom_hline(yintercept = 60, color = "#DC2626", linetype = "dotted", size = 1) +
  geom_point(data = filter(stream_window, `Machine failure` == 1), 
             aes(x = cycle_index, y = `Torque [Nm]`), color = "#DC2626", shape = 17, size = 5) +
  labs(title = "Figure 5: Continuous Monitoring Telemetry Showing Recent Readings and Moving-Average Overlay",
       x = "Historical Cycle Sequence (Replay Index)", y = "Cutting Torque [Nm]") +
  theme_industrial()
```

### 5.4 Sudden-Change Detection vs. Long-Term Drift
* **Sudden Transients:** Monitored via absolute safety thresholds ($\tau > 60\text{ Nm}$) and rate-of-change ($\Delta\tau/\Delta t$) interrupts that trigger immediate audio-visual alarm banners.
* **Long-Term Drift:** Tracked via cumulative tool wear slope, forecasting Remaining Useful Life (RUL).

---

## 6. Interpretation, Insights and Discussion (Component 4 — 5 Marks)

### 6.1 Major Findings from Visual Analysis
1. **The Spindle Operating Envelope:** Rotational speed and cutting torque exhibit an inverse hyperbolic relationship. Safe operations occupy the 3.5 kW to 9.0 kW envelope. Machine failures do not occur randomly across the curve; they cluster exclusively at:
   * **Low-Speed / High-Torque Cluster ($\omega < 1,350\text{ rpm}, \tau > 60\text{ Nm}$):** Triggers mechanical overstrain (`OSF`) and heat dissipation failures (`HDF`).
   * **High-Speed / Low-Torque Cluster ($\omega > 2,500\text{ rpm}, \tau < 15\text{ Nm}$):** Triggers electrical power failures (`PWF`).
2. **Tool Wear Dynamics:** Median tool wear among failure records (**~148 minutes**) is significantly higher than nominal cycles (**~105 minutes**). However, noticeable failures occur at low wear (< 50 min), demonstrating that tool wear is an aggravating stress factor rather than an isolated deterministic cause.
3. **Failure Mechanism Distribution:**
   * Heat Dissipation Failure (`HDF`): 115 incidents (33.9% of all failures).
   * Overstrain Failure (`OSF`): 98 incidents (28.9%).
   * Power Failure (`PWF`): 95 incidents (28.0%).
   * Tool Wear Failure (`TWF`): 46 incidents (13.6%).
   * Random Failure (`RNF`): 19 incidents (5.6%).

---

## 7. Machine Learning Predictive Pipeline & Anti-Leakage Audit

To complement visual analytics, a machine learning classification model was trained on 8,000 samples and evaluated on **2,000 held-out test samples** (20% stratified split, `random_state=42`).

### 7.1 Anti-Target Leakage Guarantee
The model strictly excludes post-failure indicators: `Machine failure`, `TWF`, `HDF`, `PWF`, `OSF`, `RNF`, `UDI`, `Product ID`, and `Heuristic Risk Score`.

### 7.2 Model Performance on 2,000 Held-Out Test Records

| Metric | Naive Majority Baseline | Trained Random Forest Model | Operational Interpretation |
| :--- | :---: | :---: | :--- |
| **Accuracy** | 96.61% | **98.40%** | Baseline achieves 96.6% by predicting "Safe" 100% of the time. |
| **Recall (Sensitivity)** | 0.00% | **82.35%** | Successfully catches **56 of 68 real machine breakdowns**. |
| **Precision** | 0.00% | **73.68%** | Low false alarm rate (prevents operator alert fatigue). |
| **F1-Score** | 0.0000 | **0.7778** | Balanced harmonic score on imbalanced rare events. |
| **ROC-AUC** | 0.5000 | **0.9715** | Near-optimal discriminative power across all decision thresholds. |

#### Confusion Matrix (2,000 Test Records):
* **True Negatives:** 1,912 nominal cycles correctly verified
* **False Positives:** 20 false alarms (1.0% false alert rate)
* **False Negatives:** 12 uncaught failures (primarily stochastic random failure mode `RNF`)
* **True Positives:** 56 machine breakdowns successfully flagged before stoppage

---

## 8. Conclusions & Engineering Recommendations (Component 4)

1. **Implement Dual-Boundary Spindle Interlocks:** Restrict CNC operations above 60 Nm cutting torque when spindle speed drops below 1,350 rpm to eliminate Overstrain (`OSF`) and Heat Dissipation (`HDF`) failures.
2. **Dynamic Tool Replacement Threshold:** Establish predictive replacement alerts at **190–200 minutes** of contact wear, avoiding premature disposal while mitigating catastrophic in-cut fractures.
3. **Coolant Flow Monitoring:** Ensure the thermal dissipation gradient ($\Delta T = T_{process} - T_{air}$) remains strictly above **8.6 K** through proactive coolant pump maintenance.
4. **Graphical Integrity Mandate:** Enforce zero-baseline representations on ratio-scale industrial metrics to prevent visual distortion and misguided managerial decisions.

---

## Appendix: Complete R Implementation Script (`case_study_visualization.R`)

The complete R script is included in the project repository at `case_study_visualization.R`. It implements all data preprocessing, feature engineering, and ggplot2 visual encodings adhering to the VIT Advanced Data Visualization curriculum.
