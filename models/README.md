# Machine Learning Model Documentation: Predictive Maintenance

## 1. Objective and Problem Formulation
The machine learning component predicts whether an industrial milling machine will experience a failure during an operational cycle given its real-time operating conditions.

This is formulated as a **binary classification** task:
- Target: $y \in \{0, 1\}$ (`0` = Nominal Operation, `1` = Machine Failure)
- Primary operational focus: **High Recall** to minimize undetected catastrophic failures (False Negatives), balanced against **Precision** to avoid alert fatigue among factory maintenance teams.

## 2. Prevention of Target Leakage
> [!CAUTION]
> **Zero Target Leakage Guarantee:**
> In predictive maintenance modeling, including specific failure modes or post-failure diagnostics as predictors produces falsely optimistic 100% accuracy that collapses when deployed in a factory.

The training pipeline strictly excludes:
1. `Machine failure`: The ground truth label.
2. `TWF`, `HDF`, `PWF`, `OSF`, `RNF`: The 5 specific failure mode indicators. These are ground-truth mechanism tags determined *after* or *at the moment of* failure. Including them would introduce 100% target leakage.
3. `UDI` and `Product ID`: Non-generalizable unique serial strings that risk memorization without mechanical validity.
4. `Heuristic Risk Score` and human-assigned category strings.

### Predictor Features Used:
- `Type`: One-hot encoded product variant (`L`, `M`, `H`).
- `Air temperature [K]`: Standardized ambient temperature.
- `Process temperature [K]`: Standardized cutting friction temperature.
- `Temp difference [K]`: Heat dissipation gradient ($\Delta T = T_{process} - T_{air}$).
- `Rotational speed [rpm]`: Spindle velocity.
- `Torque [Nm]`: Cutting torque load.
- `Power [W]`: Mechanical spindle power derived from angular speed and torque ($P = \tau \cdot \omega$).
- `Tool wear [min]`: Cumulative cutter contact duration.

## 3. Training & Evaluation Methodology
- **Validation Split:** 80% Train (8,000 samples), 20% Test (2,000 samples).
- **Stratification:** Stratified sampling based on target $y$ (`stratify=y`, `random_state=42`) ensures the exact 3.39% class distribution is preserved in both training and test partitions.
- **Handling Class Imbalance:** Class weights are inversely proportional to class frequencies (`class_weight="balanced"`), heavily penalizing missed failures.
- **Baseline Comparison:** A `DummyClassifier(strategy="most_frequent")` that naively predicts "No Failure" achieves ~96.61% accuracy but has **0.0% Recall and 0.0% F1-score**, failing to detect a single machine breakdown.

## 4. Evaluation Metrics on Held-out Test Data (2,000 instances)
All metrics presented in this application are calculated strictly on unseen test instances, never on training data:
- **Baseline Accuracy:** 96.61% (Naive majority class, F1 = 0.00)
- **Model Architecture:** Random Forest Classifier with standard scaling & one-hot encoding pipeline.
- **Metrics Evaluated:**
  - Precision: Proximity of flagged alerts that correspond to genuine failures.
  - Recall (Sensitivity): Proportion of actual machine breakdowns successfully flagged.
  - F1-Score: Harmonic balance between precision and recall.
  - ROC-AUC: Area under the receiver operating characteristic curve across all decision thresholds.
  - Confusion Matrix: Explicit count of True Negatives, False Positives, False Negatives, and True Positives.

## 5. Deployment and Failure Fallback
The trained model bundle is serialized as:
`models/predictive_maintenance_rf.joblib`

If this serialized artifact is absent or fails deserialization:
- The exploratory analytics dashboard, visualization evaluation, and telemetry monitoring continue functioning with 100% availability.
- The Risk Analysis page displays empirical physics-based heuristics and alerts the user to run the model training script (`python src/train_model.py`).
