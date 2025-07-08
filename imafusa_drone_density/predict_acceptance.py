import os
import sys
import joblib
import json
import argparse
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

def load_models(model_dir):
    models = {}
    for filename in os.listdir(model_dir):
        path = os.path.join(model_dir, filename)

        # Load .joblib models
        if filename.endswith(".joblib"):
            model_name = filename.replace(".joblib", "")
            models[model_name] = joblib.load(path)

        # Load XGBoost JSON models
        elif filename.endswith(".json"):
            model_name = filename.replace(".json", "")
            model = XGBRegressor()
            model.load_model(path)
            models[model_name] = model

    return models

def main():
    parser = argparse.ArgumentParser(description="Predict acceptance value from drone density using saved models.")
    parser.add_argument("density", type=float, help="Drone density value (float)")
    parser.add_argument("--model_dir", type=str, default="saved_models/structured", help="Path to directory containing models")

    args = parser.parse_args()
    drone_density = args.density
    model_dir = args.model_dir

    if not os.path.isdir(model_dir):
        print(f"Error: Model directory '{model_dir}' not found.")
        sys.exit(1)

    models = load_models(model_dir)
    if not models:
        print(f"No models found in '{model_dir}'.")
        sys.exit(1)

    X = np.array([[drone_density]])

    print(f"\nPredicted acceptance values for drone density = {drone_density:.2f}:\n")
    results = []

    for model_name, model in models.items():
        try:
            prediction = model.predict(X)[0]
            results.append((model_name, prediction))
        except Exception as e:
            results.append((model_name, f"Error: {e}"))

    df = pd.DataFrame(results, columns=["Model", "Predicted Acceptance"])
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
