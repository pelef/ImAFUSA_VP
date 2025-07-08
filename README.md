# ImAFUSA Comfort Modeling

This project models human comfort levels in response to drone activity across different urban and rural scenarios. It uses machine learning to learn from drone activity logs and predict comfort levels based on how frequently drones are observed over time.

---

## Project Structure

```
.
├── processed_comfort_data/              # Training CSVs per scenario
├── input_data.csv                       # Input for prediction script
├── comfort_prediction_models/
│   └── saved_models/                    # Trained model .joblib files
├── comfort_prediction_output/           # Prediction results
├── train_and_evaluate_comfort_vs_drones.py
├── load_and_predict_comfort.py
└── README.md
```

---

## Input Format

### Training Data (`processed_comfort_data/*.csv`)

Each CSV must have the following columns:

| comfort_level | time_video | yolo_count_sum |
|---------------|------------|----------------|
| "3,8"         | 67.831     | 39             |
| "3,7"         | 112.118    | 86             |
| ...           | ...        | ...            |

- The script computes `time_interval = time_video.diff()`
- Then it calculates: `drone_rate = yolo_count_sum / time_interval`
- `drone_rate` is used as the main input feature for training

---

### Prediction Input (`input_data.csv`)

Example:
```csv
time_video,yolo_count_sum
60,30
120,50
180,75
240,40
300,25
```

The prediction script will compute `time_interval` and `drone_rate` internally and output predicted comfort levels.

---

## Scripts

### `train_and_evaluate_comfort_vs_drones.py`

- Trains 6 models on each scenario:
  - Linear Regression
  - Ridge Regression
  - KNN
  - SVR
  - Random Forest
  - XGBoost
- Selects the best model based on geometric mean MAE
- Saves best models as `.joblib` files
- Exports MAE comparison plots as PDF

### `load_and_predict_comfort.py`

- Loads new data (`input_data.csv`)
- Computes `drone_rate`
- Uses all saved models to predict comfort level
- Outputs a CSV with:
  - `time_video`
  - `yolo_count_sum`
  - `time_interval`
  - `drone_rate`
  - Predicted comfort levels from each model

---

## Outputs

### `comfort_prediction_models/`

- `saved_models/*.joblib` — best models per category
- `*_model_mae.pdf` — individual model performance plots
- `grouped_model_mae_comparison.pdf` — comparison of all models

### `comfort_prediction_output/`

- `comfort_predictions.csv` — includes all predictions per input row

---

## Usage

### Train models

```bash
python3 train_and_evaluate_comfort_vs_drones.py
```

### Predict with new data

```bash
python3 load_and_predict_comfort.py
```

---

## Notes

- Datasets with fewer than N rows are skipped.
- You must include both `time_video` and `yolo_count_sum` columns for prediction.
- Models are trained separately per category (Urban Sensing, Rural Cargo, etc.)

---


