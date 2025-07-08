import os
import pandas as pd
import numpy as np
import matplotlib
import logging
import sys
import joblib
from xgboost import XGBRegressor

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from scipy.stats import gmean

# === Configure Logging ===
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info("Script started.")

# === PDF plot dimensions and styling ===
pdf_width, pdf_height = 7, 4.62
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 13,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--"
})

# Paths
data_dir = 'processed_comfort_data'
output_dir = 'comfort_prediction_models'
model_dir = os.path.join(output_dir, 'saved_models')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Scenario categories
categories = ["Urban Sensing", "Urban Cargo", "Urban Passenger", "Rural Sensing", "Rural Cargo", "Rural Passenger"]
category_data = {cat: [] for cat in categories}

# Helper to detect category from filename
def detect_category(filename):
    filename = filename.lower()
    if "urban" in filename:
        if "sensing" in filename:
            return "Urban Sensing"
        elif "cargo" in filename:
            return "Urban Cargo"
        elif "passenger" in filename:
            return "Urban Passenger"
    elif "rural" in filename:
        if "sensing" in filename:
            return "Rural Sensing"
        elif "cargo" in filename:
            return "Rural Cargo"
        elif "passenger" in filename:
            return "Rural Passenger"
    return None

# Load and group data
for filename in os.listdir(data_dir):
    if not filename.endswith(".csv"):
        continue

    category = detect_category(filename)
    if not category:
        logging.warning(f"Could not determine category for file: {filename}")
        continue

    filepath = os.path.join(data_dir, filename)
    try:
        df = pd.read_csv(filepath)

        # Normalize columns
        df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
        column_map = {}
        for col in df.columns:
            if "time" in col and ("view" in col or "video" in col):
                column_map[col] = "time_viewed"
            elif "yolo" in col and "count" in col:
                column_map[col] = "yolo_count_sum"
        df.rename(columns=column_map, inplace=True)

        # Validate
        if 'yolo_count_sum' not in df.columns or 'time_viewed' not in df.columns:
            raise ValueError(f"{filename} must contain 'yolo_count_sum' and 'time_viewed' columns. Found: {df.columns.tolist()}")

        # Compute time_interval and drone_rate
        df = df.sort_values(by='time_viewed').reset_index(drop=True)
        df['time_interval'] = df['time_viewed'].diff()
        df = df.dropna(subset=['time_interval'])
        df['comfort_level'] = df['comfort_level'].astype(str).str.replace(",", ".").astype(float)
        df['drone_rate'] = df['yolo_count_sum'] / df['time_interval']
        df = df.dropna(subset=['comfort_level', 'drone_rate'])

        category_data[category].extend(zip(df['drone_rate'], df['comfort_level']))
        logging.info(f"Loaded {filename} into category '{category}' with {len(df)} valid rows.")

    except Exception as e:
        logging.error(f"Failed to load {filename}: {e}")

# Regressors
models = {
    "Linear": LinearRegression(),
    "RandomForest": RandomForestRegressor(random_state=42),
    "SVR": SVR(),
    "KNN": KNeighborsRegressor(),
    "Ridge": Ridge(),
    "XGBoost": XGBRegressor(objective='reg:squarederror', random_state=17, verbosity=0)
}

# Store results
mae_scores = {cat: {} for cat in categories}
model_instances = {}

# Train and evaluate
for category, points in category_data.items():
    if len(points) < 10:
        logging.warning(f"Skipping {category}: not enough data ({len(points)} points).")
        continue

    X, y = zip(*points)
    X = np.array(X).reshape(-1, 1)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    for model_name, model in models.items():
        try:
            model_copy = model.__class__(**model.get_params())
            model_copy.fit(X_train, y_train)
            y_pred = model_copy.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            mae_scores[category][model_name] = mae
            model_instances[(category, model_name)] = model_copy
            logging.info(f"{model_name} trained on {category} — MAE: {mae:.3f}")
        except Exception as e:
            logging.error(f"{model_name} failed on {category}: {e}")

# Compute geometric mean MAE per model
geo_mae_per_model = {}
for model_name in models:
    maes = []
    for category in categories:
        if model_name in mae_scores[category]:
            maes.append(mae_scores[category][model_name])
    if maes:
        geo_mae = gmean(maes)
        geo_mae_per_model[model_name] = geo_mae
        logging.info(f"{model_name} — GeoMean MAE: {geo_mae:.3f}")

if not geo_mae_per_model:
    logging.error("No models trained — all datasets invalid or skipped.")
    sys.exit(1)

# Select and save best model(s)
best_model_name = min(geo_mae_per_model, key=geo_mae_per_model.get)
logging.info(f"Best model overall: {best_model_name} (GeoMean MAE: {geo_mae_per_model[best_model_name]:.3f})")

for (category, model_name), model_instance in model_instances.items():
    if model_name == best_model_name:
        model_filename = f"best_model__{model_name.lower()}__{category.replace(' ', '_').lower()}.joblib"
        model_path = os.path.join(model_dir, model_filename)
        joblib.dump(model_instance, model_path)
        logging.info(f"Saved best model: {model_path}")

# Plot MAE for all models
for category, model_maes in mae_scores.items():
    if not model_maes:
        logging.warning(f"No MAE scores for category: {category}")
        continue

    model_names = list(model_maes.keys())
    mae_values = [model_maes[m] for m in model_names]

    fig, ax = plt.subplots(figsize=(pdf_width, pdf_height), constrained_layout=True)
    ax.bar(model_names, mae_values, color='tab:orange', edgecolor='black')
    ax.set_ylabel("Mean Absolute Error")
    ax.set_title(f"Model MAE — {category}")
    ax.tick_params(axis='x', rotation=30)

    output_path = os.path.join(output_dir, f"{category.replace(' ', '_').lower()}_model_mae.pdf")
    fig.savefig(output_path, format='pdf')
    plt.close(fig)

    logging.info(f"Saved MAE plot for {category} to {output_path}")

# Grouped MAE plot
ordered_categories = [cat for cat in categories if mae_scores[cat]]
model_list = list(models.keys())
num_models = len(model_list)
bar_width = 0.11
x = np.arange(len(ordered_categories))

fig, ax = plt.subplots(figsize=(pdf_width * 1.4, pdf_height), constrained_layout=True)

for i, model_name in enumerate(model_list):
    maes = [mae_scores[cat].get(model_name, np.nan) for cat in ordered_categories]
    offset = (i - num_models / 2) * bar_width + bar_width / 2
    ax.bar(x + offset, maes, width=bar_width, label=model_name, edgecolor='black')

ax.set_xticks(x)
ax.set_xticklabels(ordered_categories, rotation=30)
ax.set_ylabel("Mean Absolute Error")
ax.set_title("Model MAE Comparison Across Scenarios")
ax.legend(title="Model", loc="upper right")

grouped_output_path = os.path.join(output_dir, "grouped_model_mae_comparison.pdf")
fig.savefig(grouped_output_path, format='pdf')
plt.close(fig)

logging.info(f"Saved grouped MAE comparison plot to {grouped_output_path}")
logging.info("Script completed.")
