import os
import sys
import time
import json
import datetime
import pandas as pd
import numpy as np
import traceback

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# ensure project root on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from skExSTraCS.ExSTraCS import ExSTraCS


# =========================
# METRICS
# =========================
def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp = cm[0]
    fn, tp = cm[1]

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    return bal_acc, tpr, tnr, int(tn), int(fp), int(fn), int(tp)


# =========================
# CROSS-VALIDATION
# =========================
def run_cv(csv_path, dataset_name, feature_name, param_grid, n_splits=5, out_dir="lcs"):
    print(f"\nDataset: {dataset_name} | Features: {feature_name}")

    csv_path = os.path.join(PROJECT_ROOT, csv_path)
    data = pd.read_csv(csv_path)

    feature_cols = [c for c in data.columns if c not in ("image", "label")]
    X = data[feature_cols].values.astype(float)
    y = data["label"].values.astype(int)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_results = []
    per_fold_records = []

    for params in param_grid:
        print("\nParams:", params)
        fold_scores = []

        for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
            seed = 42 + fold

            # =========================
            # FOLD-WISE NORMALIZATION
            # =========================
            X_tr_raw, X_te_raw = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr_raw)
            X_te = scaler.transform(X_te_raw)

            # =========================
            # MODEL SETUP
            # =========================
            model = ExSTraCS()
            model.N = params["N"]
            model.learningIterations = params["learningIterations"]
            model.theta_sel = params["theta_sel"]
            model.theta_GA = 15
            model.p_spec = 0.4
            model.nu = 3.0
            model.chi = 0.8
            model.mu = 0.04
            model.doSubsumption = True
            model.useBalancedAccuracy = True
            model.randomSeed = seed

            print(
                f"  Fold {fold} | seed={seed} "
                f"N={model.N} iters={model.learningIterations}"
            )

            start = time.time()
            fit_exception = None

            try:
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_te)
                bal_acc, tpr, tnr, tn, fp, fn, tp = compute_metrics(y_te, y_pred)
            except Exception:
                fit_exception = traceback.format_exc()
                bal_acc = tpr = tnr = tn = fp = fn = tp = None

            duration = time.time() - start

            pop_size = None
            try:
                pop_size = len(model.population.popSet)
            except Exception:
                pass

            print(
                f"    BA={bal_acc} | TPR={tpr} | TNR={tnr} "
                f"| time={duration:.1f}s | rules={pop_size}"
            )

            per_fold_records.append({
                "dataset": dataset_name,
                "features": feature_name,
                "params": params,
                "fold": fold,
                "bal_acc": bal_acc,
                "tpr": tpr,
                "tnr": tnr,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "rule_population": pop_size,
                "duration_seconds": round(duration, 2),
                "fit_exception": fit_exception
            })

            if bal_acc is not None:
                fold_scores.append(bal_acc)

        all_results.append({
            "dataset": dataset_name,
            "features": feature_name,
            "params": params,
            "mean_bal_acc": float(np.mean(fold_scores)),
            "std_bal_acc": float(np.std(fold_scores)),
            "timestamp": datetime.datetime.now().isoformat()
        })

    # =========================
    # SAVE RESULTS
    # =========================
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(out_dir, exist_ok=True)

    folds_path = os.path.join(
        out_dir, f"exstracs_cv_folds_{dataset_name}_{feature_name}_{ts}.jsonl"
    )
    summary_path = os.path.join(
        out_dir, f"exstracs_cv_summary_{dataset_name}_{feature_name}_{ts}.json"
    )

    with open(folds_path, "w") as f:
        for rec in per_fold_records:
            f.write(json.dumps(rec) + "\n")

    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Saved: {summary_path}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    param_grid = [
        {"N": 1500, "learningIterations": 100000, "theta_sel": 0.8},
        {"N": 2000, "learningIterations": 120000, "theta_sel": 0.8},
        {"N": 2000, "learningIterations": 120000, "theta_sel": 0.9},
    ]

    out_dir = os.path.join(PROJECT_ROOT, "lcs")

    experiments = [
        ("ISIC2019", "LBP", "csv_outputs/isic2019_lbp.csv"),
        ("ISIC2019", "GLCM", "csv_outputs/isic2019_glcm.csv"),
        ("ISIC2019", "Wavelet", "csv_outputs/isic2019_wavelet.csv"),
        ("HAM10000", "LBP", "csv_outputs/ham10000_lbp.csv"),
        ("HAM10000", "GLCM", "csv_outputs/ham10000_glcm.csv"),
        ("HAM10000", "Wavelet", "csv_outputs/ham10000_wavelet.csv"),
    ]

    for dataset, feature, csv_path in experiments:
        run_cv(
            csv_path=csv_path,
            dataset_name=dataset,
            feature_name=feature,
            param_grid=param_grid,
            n_splits=5,
            out_dir=out_dir
        )

    print("\nAll experiments completed successfully.")
