import os
import sys

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Ensure project root is on sys.path for local imports if needed
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def clean_and_normalize(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    # Separate metadata
    meta = df[["image", "label"]] if "image" in df.columns else df[["label"]]
    features = df.drop(columns=["image", "path", "label"], errors="ignore")

    # ---------- OUTLIER HANDLING (IQR) ----------
    for col in features.columns:
        Q1 = features[col].quantile(0.25)
        Q3 = features[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        features[col] = features[col].clip(lower, upper)

    # ---------- FEATURE-WISE NORMALIZATION ----------
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    features_scaled = pd.DataFrame(
        features_scaled,
        columns=features.columns
    )

    # ---------- FINAL DATASET ----------
    final_df = pd.concat([meta.reset_index(drop=True),
                           features_scaled.reset_index(drop=True)],
                          axis=1)

    final_df.to_csv(output_csv, index=False)

    print(f"Saved: {output_csv}")
    print("Feature range after normalization:",
          final_df.iloc[:, 2:].min().min(),
          "to",
          final_df.iloc[:, 2:].max().max())

if __name__ == "__main__":
    # HAM10000
    clean_and_normalize(
        "csv_outputs/ham10000_variant1.csv",
        "csv_outputs/ham10000_variant2.csv"
    )

    # ISIC2019
    clean_and_normalize(
        "csv_outputs/isic2019_variant1.csv",
        "csv_outputs/isic2019_variant2.csv"
    )
