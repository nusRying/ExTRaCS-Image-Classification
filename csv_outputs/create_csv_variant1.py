import os
import sys
import numpy as np
import pandas as pd
import cv2

# Ensure project root is on sys.path for local imports (e.g., features.*)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from features.extract_features import extract_features
from tqdm import tqdm

def create_csv(label_csv, output_csv):
    df = pd.read_csv(label_csv)

    rows = []
    missing_imgs = 0
    feat_len = None

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = row["path"]
        img = cv2.imread(img_path)
        if img is None:
            missing_imgs += 1
            continue

        try:
            feats = extract_features(img)
        except Exception as e:
            print(f"[ERROR] feature extraction failed for {img_path}: {e}")
            continue

        feats = np.asarray(feats).ravel()
        if feat_len is None:
            feat_len = feats.shape[0]

        rows.append([row["image"], row["path"], row["label"]] + feats.tolist())

    if not rows:
        print("No features extracted; nothing to save.")
        print(f"Missing/unreadable images: {missing_imgs}")
        return

    columns = ["image", "path", "label"] + [f"f{i}" for i in range(feat_len)]

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    out = pd.DataFrame(rows, columns=columns)
    out.to_csv(output_csv, index=False)

    print(f"Saved: {output_csv}")
    print("Feature dimension:", feat_len)
    print("Missing/unreadable images:", missing_imgs)

if __name__ == "__main__":
    # HAM10000
    create_csv(
        "CleanData/HAM10000/labels_binary.csv",
        "csv_outputs/ham10000_variant1.csv"
    )

    # ISIC2019
    create_csv(
        "CleanData/ISIC2019/labels_binary.csv",
        "csv_outputs/isic2019_variant1.csv"
    )
