import os
import sys
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import balanced_accuracy_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from skExSTraCS.ExSTraCS import ExSTraCS


def test_continuous_features():
    """Test 1: Current approach (continuous normalized features)"""
    print("\n" + "="*70)
    print("TEST 1: CONTINUOUS NORMALIZED FEATURES (Current Approach)")
    print("="*70)
    
    data = pd.read_csv("csv_outputs/ham10000_variant2.csv")
    X = data.drop(columns=["image", "label"]).values
    y = data["label"].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Feature range: [{X_train.min():.4f}, {X_train.max():.4f}]")
    print(f"Class dist (train): {np.bincount(y_train)}")
    print(f"Class dist (test): {np.bincount(y_test)}")
    
    model = ExSTraCS(
        learning_iterations=50000,
        N=1000,
        theta_GA=20,
        mu=0.04,
        random_state=42
    )
    
    t0 = time.time()
    model.fit(X_train, y_train)
    t1 = time.time()
    
    y_pred = model.predict(X_test)
    acc = balanced_accuracy_score(y_test, y_pred)
    
    print(f"\nTraining time: {t1-t0:.1f}s")
    print(f"Test balanced accuracy: {acc:.4f}")
    print(f"Rule population size: {len(model.population.popSet)}")
    print(f"✅ Result: {'GOOD' if acc > 0.65 else '❌ POOR'}")
    
    return acc


def test_discretized_features():
    """Test 2: Discretize continuous features into ordinal bins"""
    print("\n" + "="*70)
    print("TEST 2: DISCRETIZED FEATURES (5 bins per feature)")
    print("="*70)
    
    data = pd.read_csv("csv_outputs/ham10000_variant2.csv")
    X = data.drop(columns=["image", "label"]).values
    y = data["label"].values
    
    # Discretize to 5 bins
    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    X_disc = discretizer.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_disc, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"X_train shape: {X_train.shape}")
    print(f"Discretized feature range: [{X_train.min():.0f}, {X_train.max():.0f}]")
    print(f"Unique values per feature: {len(np.unique(X_train[:, 0]))}")
    print(f"Class dist (train): {np.bincount(y_train)}")
    
    model = ExSTraCS(
        learning_iterations=50000,
        N=1000,
        theta_GA=20,
        mu=0.04,
        random_state=42
    )
    
    t0 = time.time()
    model.fit(X_train, y_train)
    t1 = time.time()
    
    y_pred = model.predict(X_test)
    acc = balanced_accuracy_score(y_test, y_pred)
    
    print(f"\nTraining time: {t1-t0:.1f}s")
    print(f"Test balanced accuracy: {acc:.4f}")
    print(f"Rule population size: {len(model.population.popSet)}")
    print(f"{'✅ GOOD' if acc > 0.65 else '❌ POOR'}")
    
    return acc


def test_subset_high_variance_features():
    """Test 3: Select only high-variance features"""
    print("\n" + "="*70)
    print("TEST 3: HIGH-VARIANCE FEATURE SUBSET")
    print("="*70)
    
    data = pd.read_csv("csv_outputs/ham10000_variant2.csv")
    X = data.drop(columns=["image", "label"]).values
    y = data["label"].values
    
    # Select top 8 features by variance
    variances = X.var(axis=0)
    top_indices = np.argsort(variances)[-8:]
    X_subset = X[:, top_indices]
    
    print(f"Selected feature indices: {sorted(top_indices)}")
    print(f"Variances: {variances[top_indices]}")
    print(f"X_subset shape: {X_subset.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_subset, y, test_size=0.2, stratify=y, random_state=42
    )
    
    model = ExSTraCS(
        learning_iterations=50000,
        N=1000,
        theta_GA=20,
        mu=0.04,
        random_state=42
    )
    
    t0 = time.time()
    model.fit(X_train, y_train)
    t1 = time.time()
    
    y_pred = model.predict(X_test)
    acc = balanced_accuracy_score(y_test, y_pred)
    
    print(f"\nTraining time: {t1-t0:.1f}s")
    print(f"Test balanced accuracy: {acc:.4f}")
    print(f"Rule population size: {len(model.population.popSet)}")
    print(f"{'✅ GOOD' if acc > 0.65 else '❌ POOR'}")
    
    return acc


if __name__ == "__main__":
    results = {}
    
    try:
        results["continuous"] = test_continuous_features()
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        results["continuous"] = None
    
    try:
        results["discretized"] = test_discretized_features()
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
        results["discretized"] = None
    
    try:
        results["high_variance"] = test_subset_high_variance_features()
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
        results["high_variance"] = None
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for approach, acc in results.items():
        status = f"✅ {acc:.4f}" if acc and acc > 0.65 else f"❌ {acc:.4f}" if acc else "ERROR"
        print(f"{approach:20s}: {status}")
    
    best = max((k, v) for k, v in results.items() if v is not None)
    print(f"\n🏆 Best approach: {best[0]} ({best[1]:.4f})")