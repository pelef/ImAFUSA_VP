import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# === Input Files for Both Types ===
data_files = {
    "structured": "VISUAL_POLUTION_EXPERIMENT_DATA.2025.01.09.xlsx - DTV2.01 Data Spread.csv",
    "unstructured": "VISUAL_POLUTION_EXPERIMENT_DATA.2025.01.09.xlsx - DTV2.02 Data Spread.csv"
}

base_model_dir = "saved_models"
os.makedirs(base_model_dir, exist_ok=True)

for path_type, file_path in data_files.items():
    print(f"\nProcessing {path_type.upper()} data from: {file_path}")

    df = pd.read_csv(file_path)
    df['Category'] = df['Environment']
    model_dir = os.path.join(base_model_dir, path_type)
    os.makedirs(model_dir, exist_ok=True)

    # === Correlation Plot ===
    correlations = df.groupby('Category').apply(lambda g: g['Acceptance Value'].corr(g['Density Value']))
    colors_corr = sns.color_palette("Blues", 2 * len(correlations))[6:6 + len(correlations)]
    correlations.plot(kind='bar', color=colors_corr, title=f'Correlation between Acceptance and Density ({path_type})')
    plt.ylabel('Correlation Coefficient')
    plt.xlabel('Experiment Setup')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
    plt.close()

    # === Averages Plot ===
    averages = df.groupby('Category')[['Acceptance Value', 'Density Value']].mean()
    averages.rename(columns={'Acceptance Value': 'Acceptance', 'Density Value': 'Density'}, inplace=True)
    averages.plot(kind='bar', color=['mediumseagreen', 'orange'],
                  title=f'Average Acceptance and Density Values ({path_type})')
    plt.ylabel('Average Value')
    plt.xlabel('Experiment Setup')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
    plt.close()

    # === Train & Save Models per Category ===
    lin_errors = {}
    xgb_errors = {}
    total_lin = 0
    total_xgb = 0

    for category, group in df.groupby('Category'):
        X = group[['Density Value']].values
        y = group['Acceptance Value'].values

        # Linear Regression
        lin_model = LinearRegression()
        lin_model.fit(X, y)
        y_pred_lin = lin_model.predict(X)
        lin_rmse = mean_squared_error(y, y_pred_lin, squared=False)
        lin_errors[category] = lin_rmse
        total_lin += lin_rmse

        lin_model_path = os.path.join(model_dir, f"lin_model_{category.replace(' ', '_')}.joblib")
        joblib.dump(lin_model, lin_model_path)

        # XGBoost
        xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, verbosity=0)
        xgb_model.fit(X, y)
        y_pred_xgb = xgb_model.predict(X)
        xgb_rmse = mean_squared_error(y, y_pred_xgb, squared=False)
        xgb_errors[category] = xgb_rmse
        total_xgb += xgb_rmse

        xgb_model_path = os.path.join(model_dir, f"xgb_model_{category.replace(' ', '_')}.json")
        xgb_model.save_model(xgb_model_path)

    # === RMSE Plot ===
    error_df = pd.DataFrame({
        'Linear Regression': lin_errors,
        'XGBoost Regression': xgb_errors
    }).T

    error_df.T.plot(kind='bar',
                    color=[sns.color_palette("Greens")[3], sns.color_palette("Oranges")[3]],
                    title=f'Model Errors per Experiment Setup ({path_type})')
    plt.ylabel('Root Mean Squared Error')
    plt.xlabel('Experiment Setup')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
    plt.close()

    # === Global Model: Density + Category ===
    X_cat = pd.get_dummies(df['Category'], prefix='Cat')
    X = pd.concat([df[['Density Value']], X_cat], axis=1)
    y = df['Acceptance Value']

    total_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, verbosity=0)
    total_model.fit(X, y)
    y_pred_total = total_model.predict(X)
    total_rmse = mean_squared_error(y, y_pred_total, squared=False)

    #global_model_path = os.path.join(model_dir, "xgb_model_global.joblib")
    #joblib.dump(total_model, global_model_path)

    # === Summary Output ===
    print(f"Total RMSE using Density + Category ({path_type}): {total_rmse:.4f}")
    print(f"Total RMSE Sum (XGBoost by category): {total_xgb:.2f}")
    print(f"Total RMSE Sum (Linear by category): {total_lin:.2f}")
    print(f"Models saved in: {model_dir}")
