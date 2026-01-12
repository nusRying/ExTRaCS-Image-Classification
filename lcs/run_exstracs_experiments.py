import os
import sys
import time
import json
import datetime
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix,
)

# --------------------------------------------------
# Ensure project root on Python path
# --------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from skExSTraCS.ExSTraCS import ExSTraCS


# -----------------------------
# Utility
# -----------------------------
def majority_baseline_bal_acc(y):
    majority_class = int(pd.Series(y).value_counts().idxmax())
    y_majority = np.full_like(y, majority_class)
    return float(balanced_accuracy_score(y, y_majority)), majority_class


def compute_metrics_binary(y_true, y_pred):
    """
    Returns balanced accuracy, sensitivity (TPR), specificity (TNR), and confusion matrix parts.
    Assumes binary labels {0,1}.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    fn, tp = int(cm[1, 0]), int(cm[1, 1])

    # Avoid division-by-zero
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else None  # TPR
    specificity = tn / (tn + fp) if (tn + fp) > 0 else None  # TNR
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    return {
        "bal_acc": bal_acc,
        "sensitivity_tpr": sensitivity,
        "specificity_tnr": specificity,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def safe_auc_from_model(model, X_test, y_test):
    """
    Compute ROC-AUC only if predict_proba exists.
    If not available, returns None (correct and honest).
    """
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)
            # Expect shape (n,2) for binary
            if isinstance(proba, np.ndarray) and proba.ndim == 2 and proba.shape[1] == 2:
                return float(roc_auc_score(y_test, proba[:, 1]))
    except Exception:
        pass
    return None


def set_if_exists(obj, attr_name, value):
    if hasattr(obj, attr_name):
        setattr(obj, attr_name, value)
        return True
    return False


# -----------------------------
# Core CV runner
# -----------------------------
def run_experiment_cv(csv_path, dataset_name, param_grid, n_splits=5, seed=42):
    print(f"\n==============================")
    print(f"Dataset: {dataset_name}")
    print(f"CSV: {csv_path}")
    print(f"CV: StratifiedKFold(n_splits={n_splits}, seed={seed})")
    print(f"==============================")

    data = pd.read_csv(csv_path)

    if "label" not in data.columns:
        raise ValueError("CSV must contain a 'label' column.")

    drop_cols = ["label"]
    if "image" in data.columns:
        drop_cols.insert(0, "image")

    X = data.drop(columns=drop_cols).values
    y = data["label"].values.astype(int)

    maj_bal_acc, maj_class = majority_baseline_bal_acc(y)

    print(f"Samples: {len(y)} | Features: {X.shape[1]}")
    print(f"Class counts: {pd.Series(y).value_counts().to_dict()}")
    print(f"Majority class: {maj_class}")
    print(f"Majority baseline Balanced Acc: {maj_bal_acc:.4f}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Run metadata
    run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"{dataset_name}_{run_timestamp}"

    summary_rows = []
    fold_rows = []

    for params in param_grid:
        print("\nRunning params:", params)

        fold_metrics_list = []
        fold_rule_counts = []
        fold_durations = []
        fold_failures = 0

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = ExSTraCS()

            # ---------------------------------------
            # Set parameters (attribute-set style)
            # ---------------------------------------
            # Core
            model.N = int(params["N"])
            model.learningIterations = int(params["learningIterations"])
            model.trackingFrequency = int(params.get("trackingFrequency", 2000))

            # GA / pruning defaults (explicit)
            model.nu = float(params.get("nu", 1.0))
            model.chi = float(params.get("chi", 0.8))
            model.mu = float(params.get("mu", 0.04))
            model.theta_GA = float(params.get("theta_GA", 25))
            model.theta_del = int(params.get("theta_del", 20))
            model.theta_sub = int(params.get("theta_sub", 20))
            model.acc_sub = float(params.get("acc_sub", 0.99))

            # Subsumption flags (version dependent)
            set_if_exists(model, "doSubsumption", bool(params.get("doSubsumption", True)))
            set_if_exists(model, "do_correct_set_subsumption", bool(params.get("do_correct_set_subsumption", False)))
            set_if_exists(model, "do_GA_subsumption", bool(params.get("do_GA_subsumption", True)))

            # Selection
            # (package may use selectionMethod or selection_method)
            if hasattr(model, "selectionMethod"):
                model.selectionMethod = params.get("selection_method", "tournament")
            elif hasattr(model, "selection_method"):
                model.selection_method = params.get("selection_method", "tournament")

            model.theta_sel = float(params.get("theta_sel", 0.5))

            # Balanced accuracy fitness (key)
            set_if_exists(model, "useBalancedAccuracy", True)

            # Reproducibility (version dependent)
            set_if_exists(model, "randomSeed", seed)
            set_if_exists(model, "random_state", seed)

            # ---------------------------------------
            # Train (timed)
            # ---------------------------------------
            start_iso = datetime.datetime.now().isoformat()
            t0 = time.time()
            fit_exception = None

            try:
                model.fit(X_train, y_train)
            except Exception as e:
                fit_exception = str(e)

            t1 = time.time()
            end_iso = datetime.datetime.now().isoformat()
            duration_s = float(t1 - t0)

            fold_durations.append(duration_s)

            if fit_exception is not None:
                fold_failures += 1
                print(f"  Fold {fold_idx}/{n_splits} | duration={duration_s:.1f}s | FIT_FAIL: {fit_exception}")
                fold_rows.append({
                    "run_id": run_id,
                    "run_timestamp": run_timestamp,
                    "dataset": dataset_name,
                    "csv_path": csv_path,
                    "fold": fold_idx,
                    "N": params["N"],
                    "learningIterations": params["learningIterations"],
                    "theta_sel": params.get("theta_sel", 0.5),
                    "start_iso": start_iso,
                    "end_iso": end_iso,
                    "duration_s": duration_s,
                    "fit_exception": fit_exception,
                })
                continue

            # ---------------------------------------
            # Evaluate
            # ---------------------------------------
            y_pred = model.predict(X_test)
            metrics = compute_metrics_binary(y_test, y_pred)
            auc = safe_auc_from_model(model, X_test, y_test)

            # Rule population size
            try:
                rule_count = int(len(model.population.popSet))
            except Exception:
                rule_count = None

            fold_rule_counts.append(rule_count)

            fold_metrics_list.append({
                "fold": fold_idx,
                **metrics,
                "roc_auc": auc,
                "rule_population": rule_count,
                "start_iso": start_iso,
                "end_iso": end_iso,
                "duration_s": duration_s,
            })

            fold_rows.append({
                "run_id": run_id,
                "run_timestamp": run_timestamp,
                "dataset": dataset_name,
                "csv_path": csv_path,
                "fold": fold_idx,
                "N": params["N"],
                "learningIterations": params["learningIterations"],
                "theta_sel": params.get("theta_sel", 0.5),

                "test_bal_acc": metrics["bal_acc"],
                "test_sensitivity_tpr": metrics["sensitivity_tpr"],
                "test_specificity_tnr": metrics["specificity_tnr"],
                "tn": metrics["tn"], "fp": metrics["fp"], "fn": metrics["fn"], "tp": metrics["tp"],

                "roc_auc": auc,
                "rule_population": rule_count,

                "start_iso": start_iso,
                "end_iso": end_iso,
                "duration_s": duration_s,
                "fit_exception": None,
            })

            print(f"  Fold {fold_idx}/{n_splits} | duration={duration_s:.1f}s | "
                  f"BA={metrics['bal_acc']:.4f} | TPR={metrics['sensitivity_tpr'] if metrics['sensitivity_tpr'] is not None else 'NA'} "
                  f"| TNR={metrics['specificity_tnr'] if metrics['specificity_tnr'] is not None else 'NA'}")

        # ---------------------------------------
        # Aggregate across folds
        # ---------------------------------------
        bal_accs = [m["bal_acc"] for m in fold_metrics_list if m.get("bal_acc") is not None]
        tprs = [m["sensitivity_tpr"] for m in fold_metrics_list if m.get("sensitivity_tpr") is not None]
        tnrs = [m["specificity_tnr"] for m in fold_metrics_list if m.get("specificity_tnr") is not None]
        aucs = [m["roc_auc"] for m in fold_metrics_list if m.get("roc_auc") is not None]
        rules = [m["rule_population"] for m in fold_metrics_list if m.get("rule_population") is not None]

        summary = {
            "run_id": run_id,
            "run_timestamp": run_timestamp,
            "dataset": dataset_name,
            "csv_path": csv_path,
            "seed": seed,
            "n_splits": n_splits,

            "N": params["N"],
            "learningIterations": params["learningIterations"],
            "theta_sel": params.get("theta_sel", 0.5),

            "majority_class": maj_class,
            "majority_bal_acc": maj_bal_acc,

            "mean_test_bal_acc": float(np.mean(bal_accs)) if bal_accs else None,
            "std_test_bal_acc": float(np.std(bal_accs)) if bal_accs else None,

            "mean_test_sensitivity_tpr": float(np.mean(tprs)) if tprs else None,
            "mean_test_specificity_tnr": float(np.mean(tnrs)) if tnrs else None,

            "mean_test_roc_auc": float(np.mean(aucs)) if aucs else None,

            "mean_rule_population": float(np.mean(rules)) if rules else None,
            "mean_duration_s": float(np.mean(fold_durations)) if fold_durations else None,

            "fit_failures": fold_failures,

            # store full params
            "params_json": json.dumps(params, sort_keys=True),
        }

        summary_rows.append(summary)

        print("  => Mean BA:",
              f"{summary['mean_test_bal_acc']:.4f}" if summary["mean_test_bal_acc"] is not None else "None",
              "| Std:", f"{summary['std_test_bal_acc']:.4f}" if summary["std_test_bal_acc"] is not None else "None",
              "| Mean TPR:", f"{summary['mean_test_sensitivity_tpr']:.4f}" if summary["mean_test_sensitivity_tpr"] is not None else "None",
              "| Mean TNR:", f"{summary['mean_test_specificity_tnr']:.4f}" if summary["mean_test_specificity_tnr"] is not None else "None",
              "| Mean rules:", summary["mean_rule_population"])

    summary_df = pd.DataFrame(summary_rows)
    folds_df = pd.DataFrame(fold_rows)
    return summary_df, folds_df


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    # Small, hypothesis-driven grid (knee of curve)
    param_grid = [
        {"N": 1500, "learningIterations": 100000, "theta_sel": 0.8},
        {"N": 2000, "learningIterations": 120000, "theta_sel": 0.8},
        {"N": 2000, "learningIterations": 120000, "theta_sel": 0.9},
        {"N": 2500, "learningIterations": 150000, "theta_sel": 0.9},
    ]

    out_dir = "lcs"
    os.makedirs(out_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_summary = os.path.join(out_dir, f"exstracs_cv_summary_{timestamp}.csv")
    out_folds = os.path.join(out_dir, f"exstracs_cv_folds_{timestamp}.csv")

    # Run CV on both datasets
    ham_summary, ham_folds = run_experiment_cv(
        "csv_outputs/ham10000_variant2.csv",
        "HAM10000",
        param_grid,
        n_splits=5,
        seed=42
    )

    isic_summary, isic_folds = run_experiment_cv(
        "csv_outputs/isic2019_variant2.csv",
        "ISIC2019",
        param_grid,
        n_splits=5,
        seed=42
    )

    # Combine and save
    summary_all = pd.concat([ham_summary, isic_summary], ignore_index=True)
    folds_all = pd.concat([ham_folds, isic_folds], ignore_index=True)

    summary_all.to_csv(out_summary, index=False)
    folds_all.to_csv(out_folds, index=False)

    print("\n✅ All CV experiments completed.")
    print(f"Summary saved to: {out_summary}")
    print(f"Fold details saved to: {out_folds}")
