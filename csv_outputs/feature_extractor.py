import os
import sys
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from skimage.feature import local_binary_pattern, graycomatrix, graycoprops


# =========================
# CONFIG
# =========================

# Ensure project root on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# CHANGE THIS
DATASET_NAME = "isic2019"

# Auto-detect paths based on dataset (use CleanData only)
if DATASET_NAME.lower() == "ham10000":
    IMAGE_ROOT = os.path.join(PROJECT_ROOT, "CleanData", "HAM10000", "images")
    LABEL_CSV = os.path.join(PROJECT_ROOT, "CleanData", "HAM10000", "labels_binary.csv")
elif DATASET_NAME.lower() == "isic2019":
    IMAGE_ROOT = os.path.join(PROJECT_ROOT, "CleanData", "ISIC2019", "images_train")
    LABEL_CSV = os.path.join(PROJECT_ROOT, "CleanData", "ISIC2019", "labels_binary.csv")
else:
    raise ValueError(f"Unknown dataset: {DATASET_NAME}")

OUT_DIR = os.path.join(PROJECT_ROOT, "csv_outputs")

# LBP parameters
LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS
LBP_METHOD = "uniform"

# GLCM parameters
GLCM_DISTANCES = [1, 2]
GLCM_ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]
GLCM_PROPS = [
    "contrast",
    "dissimilarity",
    "homogeneity",
    "energy",
    "correlation",
    "ASM",
]

os.makedirs(OUT_DIR, exist_ok=True)


# =========================
# FEATURE FUNCTIONS
# =========================

def extract_lbp(gray):
    lbp = local_binary_pattern(
        gray,
        P=LBP_POINTS,
        R=LBP_RADIUS,
        method=LBP_METHOD
    )
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    return hist


def extract_glcm(gray):
    glcm = graycomatrix(
        gray,
        distances=GLCM_DISTANCES,
        angles=GLCM_ANGLES,
        symmetric=True,
        normed=True
    )

    feats = []
    for prop in GLCM_PROPS:
        vals = graycoprops(glcm, prop)
        feats.extend(vals.flatten())

    return np.array(feats)


# =========================
# MAIN
# =========================

def main():
    # Validate paths exist
    if not os.path.isdir(IMAGE_ROOT):
        raise FileNotFoundError(f"IMAGE_ROOT not found: {IMAGE_ROOT}")
    if not os.path.isfile(LABEL_CSV):
        raise FileNotFoundError(f"LABEL_CSV not found: {LABEL_CSV}")
    
    print(f"Dataset: {DATASET_NAME}")
    print(f"Image root: {IMAGE_ROOT}")
    print(f"Label CSV: {LABEL_CSV}")
    
    labels_df = pd.read_csv(LABEL_CSV)

    lbp_rows = []
    glcm_rows = []

    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df)):
        image_id = row["image"]
        label = int(row["label"])

        img_path = os.path.join(IMAGE_ROOT, image_id)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Warning: could not read {img_path}")
            continue

        # LBP
        lbp_feat = extract_lbp(img)
        lbp_row = {"image": image_id, "label": label}
        for i, v in enumerate(lbp_feat):
            lbp_row[f"lbp_{i}"] = v
        lbp_rows.append(lbp_row)

        # GLCM
        glcm_feat = extract_glcm(img)
        glcm_row = {"image": image_id, "label": label}
        for i, v in enumerate(glcm_feat):
            glcm_row[f"glcm_{i}"] = v
        glcm_rows.append(glcm_row)

    # Save CSVs
    lbp_df = pd.DataFrame(lbp_rows)
    glcm_df = pd.DataFrame(glcm_rows)

    lbp_path = os.path.join(OUT_DIR, f"{DATASET_NAME}_lbp.csv")
    glcm_path = os.path.join(OUT_DIR, f"{DATASET_NAME}_glcm.csv")

    lbp_df.to_csv(lbp_path, index=False)
    glcm_df.to_csv(glcm_path, index=False)

    print(f"Saved: {lbp_path}")
    print(f"Saved: {glcm_path}")


if __name__ == "__main__":
    main()
