import os
import sys
import logging
import joblib
import pandas as pd
import numpy as np

# === Configure Logging ===
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# === Paths ===
model_dir = 'comfort_prediction_models/saved_models'
input_data_path = 'input_data.csv'  # Must contain time_video & yolo_count_sum
output_dir = 'comfort_prediction_output'
os.makedirs(output_dir, exist_ok=True)

# === Load and process input data ===
try:
    df = pd.read_csv(input_data_path)

    # Validate required columns
    required_cols = {'time_video', 'yolo_count_sum'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Input must contain columns: {required_cols}")

    # Compute time interval and drone_rate
    df = df.sort_values(by='time_video').reset_index(drop=True)
    df['time_interval'] = df['time_video'].diff()
    df = df.dropna(subset=['time_interval'])  # Drop first row (no interval)
    df['drone_rate'] = df['yolo_count_sum'] / df['time_interval']

    X = np.array(df['drone_rate']).reshape(-1, 1)
    logging.info(f"Loaded and processed input data with {len(df)} rows.")

except Exception as e:
    logging.error(f"Failed to load or process input data: {e}")
    sys.exit(1)

# === Load all models ===
model_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]
if not model_files:
    logging.error("No models found in directory.")
    sys.exit(1)

# === Prepare output DataFrame ===
results = df[['time_video', 'yolo_count_sum', 'time_interval', 'drone_rate']].copy()

# === Run predictions ===
for model_file in model_files:
    model_path = os.path.join(model_dir, model_file)
    try:
        model = joblib.load(model_path)
        preds = model.predict(X)
        model_name = model_file.replace('.joblib', '')
        results[model_name] = preds
        logging.info(f"Predictions made using {model_file}")
    except Exception as e:
        logging.error(f"Failed to predict with {model_file}: {e}")

# === Save results ===
output_path = os.path.join(output_dir, 'comfort_predictions.csv')
results.to_csv(output_path, index=False)
logging.info(f"Saved predictions to {output_path}")
