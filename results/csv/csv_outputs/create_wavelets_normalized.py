import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ==============================
# CONFIGURATION
# ==============================

INPUT_DIR = r"C:\Users\umair\Videos\PhD\PhD Data\Week 8 Jannuary\Code\csv_outputs"
OUTPUT_DIR = INPUT_DIR  # save in same folder (change if needed)

FILES = [
    "isic2019_wavelet.csv",
    "ham10000_wavelet.csv"
]

LABEL_COL = "label"

# ==============================
# NORMALIZATION FUNCTION
# ==============================

def normalize_wavelet_csv(input_path, output_path):
    print(f"\nProcessing: {os.path.basename(input_path)}")

    # Load CSV
    df = pd.read_csv(input_path)

    # Sanity check
    if LABEL_COL not in df.columns:
        raise ValueError(f"'{LABEL_COL}' column not found in {input_path}")

    # Separate features and label
    # Drop both label and image columns (keep only numeric features)
    cols_to_drop = [LABEL_COL]
    if "image" in df.columns:
        cols_to_drop.append("image")
    
    X = df.drop(columns=cols_to_drop)
    y = df[LABEL_COL]

    # Column-wise normalization (z-score)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert back to DataFrame
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    # Combine label + features
    df_normalized = pd.concat([y.reset_index(drop=True),
                               X_scaled_df.reset_index(drop=True)], axis=1)

    # Save normalized CSV
    df_normalized.to_csv(output_path, index=False)

    # Quick sanity check
    print("  ✔ Saved:", os.path.basename(output_path))
    print("  ✔ Mean (first 5 features):",
          X_scaled_df.mean().head().round(4).tolist())
    print("  ✔ Std  (first 5 features):",
          X_scaled_df.std().head().round(4).tolist())


# ==============================
# MAIN LOOP
# ==============================

for filename in FILES:
    input_csv = os.path.join(INPUT_DIR, filename)
    output_csv = os.path.join(
        OUTPUT_DIR,
        filename.replace(".csv", "_normalized.csv")
    )

    normalize_wavelet_csv(input_csv, output_csv)

print("\n✅ All files normalized successfully.")
