import os
import sys
import cv2
import pandas as pd
from tqdm import tqdm

# Ensure project root on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from features.wavelet_features import extract_wavelet_features


# --------------------------------------------------
# CONFIG
# --------------------------------------------------
DATASETS = {
    "HAM10000": {
        "images": "CleanData/HAM10000/images",
        "labels": "CleanData/HAM10000/labels_binary.csv",
        "output": "csv_outputs/ham10000_wavelet.csv",
    },
    "ISIC2019": {
        "images": "CleanData/ISIC2019/images_train",
        "labels": "CleanData/ISIC2019/labels_binary.csv",
        "output": "csv_outputs/isic2019_wavelet.csv",
    },
}


def run_dataset(cfg):
    labels_df = pd.read_csv(cfg["labels"])
    rows = []

    print(f"\nProcessing {cfg['output']}")

    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df)):
        img_path = os.path.join(cfg["images"], row["image"])
        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        wavelet_feats = extract_wavelet_features(img)

        rows.append(
            {
                "image": row["image"],
                "label": row["label"],
                **{f"wav_{i}": wavelet_feats[i] for i in range(len(wavelet_feats))}
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(cfg["output"], index=False)

    print(f"Saved: {cfg['output']}")
    print(f"Samples: {len(df)} | Features: {len(df.columns) - 2}")


if __name__ == "__main__":
    for cfg in DATASETS.values():
        run_dataset(cfg)