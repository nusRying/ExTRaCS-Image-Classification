import os
import sys
import pandas as pd

# --------------------------------------------------
# Ensure project root on Python path
# --------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from skExSTraCS.ExSTraCS import ExSTraCS


def run_exstracs(csv_path, dataset_name):
    print(f"\n=== Running ExSTraCS Experiment: {dataset_name} ===")

    # ---------------------------
    # Load data
    # ---------------------------
    data = pd.read_csv(csv_path)

    y = data["label"].values
    X = data.drop(columns=["image", "label"]).values

    # ---------------------------
    # Initialize model
    # ---------------------------
    model = ExSTraCS()

    # ---------------------------
    # CORE PARAMETERS
    # ---------------------------
    model.N = 1000                    # population size
    model.learningIterations = 50000  # allow rule maturation
    model.trackingFrequency = 2000

    # ---------------------------
    # GA PARAMETERS
    # ---------------------------
    model.theta_GA = 25               # GA activation threshold
    model.chi = 0.8                   # crossover probability
    model.mu = 0.04                   # mutation probability
    model.doSubsumption = True

    # ---------------------------
    # SELECTION & FITNESS (KEY CHANGE)
    # ---------------------------
    model.selectionMethod = "tournament"
    model.theta_sel = 0.5

    # ⭐ CRITICAL CHANGE ⭐
    # Handle class imbalance properly
    model.useBalancedAccuracy = True

    # ---------------------------
    # REPRODUCIBILITY
    # ---------------------------
    model.randomSeed = 42

    # ---------------------------
    # Train
    # ---------------------------
    model.fit(X, y)

    # ---------------------------
    # Evaluate (training sanity check)
    # ---------------------------
    acc = model.score(X, y)

    print(f"Training accuracy ({dataset_name}): {acc:.4f}")
    print(f"Final rule population size: {len(model.population.popSet)}")


if __name__ == "__main__":
    run_exstracs(
        "csv_outputs/ham10000_variant2.csv",
        "HAM10000"
    )

    run_exstracs(
        "csv_outputs/isic2019_variant2.csv",
        "ISIC2019"
    )
