import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Input Files ===
data_files = {
    "Structured": "VISUAL_POLUTION_EXPERIMENT_DATA.2025.01.09.xlsx - DTV2.01 Data Spread.csv",
    "Unstructured": "VISUAL_POLUTION_EXPERIMENT_DATA.2025.01.09.xlsx - DTV2.02 Data Spread.csv"
}

# === Prepare combined dataframe ===
all_data = []

for label, path in data_files.items():
    if not os.path.exists(path):
        print(f"Warning: File not found - {path}")
        continue

    df = pd.read_csv(path)
    if 'Density Value' not in df.columns:
        print(f"Warning: 'Density Value' column missing in {path}")
        continue

    df_filtered = df[['Density Value']].copy()
    df_filtered['Path Type'] = label
    all_data.append(df_filtered)

if not all_data:
    print("No valid data to plot.")
    exit(1)

df_combined = pd.concat(all_data, ignore_index=True)

# === Plot distribution ===
plt.figure(figsize=(10, 6))
sns.histplot(data=df_combined, x='Density Value', hue='Path Type', kde=True, element='step', stat='density', common_norm=False)
plt.title('Drone Density Value Distribution by Path Type')
plt.xlabel('Density Value (drones/km2)')
plt.ylabel('Density')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
