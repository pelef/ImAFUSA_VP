# Drone Density Analysis and Acceptance Modeling

This folder contains analysis and prediction tools used to explore the relationship between drone density and perceived acceptance in structured and unstructured drone paths.

---

## Contents

### 1. `density_correlation.py`

Trains regression models (Linear & XGBoost) per category to model how drone density affects user acceptance.

- **Inputs:**
  - `VISUAL_POLUTION_EXPERIMENT_DATA.2025.01.09.xlsx - DTV2.01 Data Spread.csv` (Structured)
  - `VISUAL_POLUTION_EXPERIMENT_DATA.2025.01.09.xlsx - DTV2.02 Data Spread.csv` (Unstructured)

- **Outputs:**
  - Trained models saved under `saved_models/{structured,unstructured}/`
  - Correlation and average bar plots
  - RMSE comparison plots per category

---

### 2. `plot_density_distributions.py`

Creates KDE and histogram plots of drone density across structured and unstructured paths.

- **Output:**
  - Comparative plot of density distributions

---

### 3. `predict_acceptance.py`

Command-line utility to predict user acceptance based on a given drone density value.

#### Example

```bash
python3 predict_acceptance.py 0.35 --model_dir saved_models/structured
```

#### Arguments

- `density`: Drone density value (float)
- `--model_dir`: Directory containing trained `.joblib` and/or `.json` models

#### Output

Table of predicted acceptance values per model.

---

## Data Format

Both input CSVs must include these columns:

| Environment | Participant | Acceptance Value | Density Value |
|-------------|-------------|------------------|---------------|

---

## Output Structure

```
saved_models/
├── structured/
│   ├── lin_model_*.joblib
│   └── xgb_model_*.json
├── unstructured/
│   ├── lin_model_*.joblib
│   └── xgb_model_*.json
```