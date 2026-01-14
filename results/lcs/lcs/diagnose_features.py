import os
import sys
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def diagnose_dataset(csv_path, dataset_name):
    print(f"\n{'='*60}")
    print(f"DIAGNOSTIC REPORT: {dataset_name}")
    print(f"{'='*60}")

    data = pd.read_csv(csv_path)
    
    print(f"\n1. SHAPE & SIZE")
    print(f"   Total samples: {len(data)}")
    print(f"   Total features: {len(data.columns) - 2}")  # -image, -label
    
    print(f"\n2. LABEL DISTRIBUTION")
    label_counts = data["label"].value_counts().sort_index()
    for label, count in label_counts.items():
        pct = 100 * count / len(data)
        print(f"   Class {label}: {count} samples ({pct:.1f}%)")
    
    print(f"\n3. MISSING VALUES")
    missing = data.isna().sum().sum()
    print(f"   Total NaN: {missing}")
    
    print(f"\n4. FEATURE STATISTICS")
    X = data.drop(columns=["image", "label"]).values
    y = data["label"].values
    
    print(f"   Shape: {X.shape}")
    print(f"   Min value: {X.min():.6f}")
    print(f"   Max value: {X.max():.6f}")
    print(f"   Mean: {X.mean():.6f}")
    print(f"   Std: {X.std():.6f}")
    
    print(f"\n5. PER-FEATURE RANGES")
    for i in range(min(5, X.shape[1])):  # Show first 5
        col_min = X[:, i].min()
        col_max = X[:, i].max()
        col_mean = X[:, i].mean()
        print(f"   f{i}: [{col_min:.4f}, {col_max:.4f}], mean={col_mean:.4f}")
    if X.shape[1] > 5:
        print(f"   ... ({X.shape[1] - 5} more features)")
    
    print(f"\n6. SVM BASELINE (5-fold CV)")
    print(f"   Training SVM (this may take a minute)...")
    try:
        svm_scores = cross_val_score(SVC(kernel='rbf', C=1.0), X, y, cv=5, verbose=0)
        print(f"   Fold scores: {[f'{s:.4f}' for s in svm_scores]}")
        print(f"   Mean Accuracy: {svm_scores.mean():.4f} ± {svm_scores.std():.4f}")
        
        if svm_scores.mean() > 0.65:
            print(f"   ✅ SVM performs well → Features are good, ExSTraCS needs tuning")
        elif svm_scores.mean() > 0.55:
            print(f"   ⚠️  SVM moderate → Weak signal, might be feature engineering issue")
        else:
            print(f"   ❌ SVM poor (≈50%) → Features likely broken or data mislabeled")
    except Exception as e:
        print(f"   ❌ SVM failed: {e}")
    
    print(f"\n7. FEATURE VARIANCE")
    feature_vars = X.var(axis=0)
    low_var = np.sum(feature_vars < 0.01)
    print(f"   Features with variance < 0.01: {low_var}/{X.shape[1]}")
    if low_var > X.shape[1] * 0.5:
        print(f"   ⚠️  More than 50% of features have low variance!")


if __name__ == "__main__":
    diagnose_dataset("csv_outputs/ham10000_variant2.csv", "HAM10000")
    diagnose_dataset("csv_outputs/isic2019_variant2.csv", "ISIC2019")