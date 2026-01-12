import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from skExSTraCS.ExSTraCS import ExSTraCS


def inspect_model_internals():
    """Deep dive into ExSTraCS internals to see why it's not learning"""
    
    print("\n" + "="*70)
    print("DEEP INSPECTION: ExSTraCS Internals")
    print("="*70)
    
    # Load minimal subset for faster debugging
    data = pd.read_csv("csv_outputs/ham10000_variant2.csv")
    X = data.drop(columns=["image", "label"]).values[:1000]  # Only first 1000
    y = data["label"].values[:1000]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"\nDataset info:")
    print(f"  Train shape: {X_train.shape}")
    print(f"  Train classes: {np.bincount(y_train)}")
    print(f"  Feature range: [{X_train.min():.4f}, {X_train.max():.4f}]")
    
    # -----------------------------------------------
    # Train with verbose settings
    # -----------------------------------------------
    model = ExSTraCS(
        learning_iterations=5000,  # Small for debugging
        N=500,
        theta_GA=10,
        random_state=42,
        track_accuracy_while_fit=True  # Enable accuracy tracking
    )
    
    print("\nTraining...")
    model.fit(X_train, y_train)
    
    # -----------------------------------------------
    # Inspect trained model
    # -----------------------------------------------
    print("\n" + "-"*70)
    print("POST-TRAINING INSPECTION")
    print("-"*70)
    
    print(f"\nPopulation stats:")
    print(f"  Macro pop size (# rules): {len(model.population.popSet)}")
    micro_pop = sum(cl.numerosity for cl in model.population.popSet) if len(model.population.popSet) > 0 else 0
    print(f"  Micro pop size (total numerosity): {micro_pop}")
    
    if len(model.population.popSet) > 0:
        print(f"\nFirst 5 rules:")
        for i, cl in enumerate(model.population.popSet[:5]):
            print(f"  Rule {i}: phenotype={cl.phenotype}, fitness={cl.fitness:.4f}, numerosity={cl.numerosity}, accuracy={cl.accuracy:.4f}")
    
    # -----------------------------------------------
    # Check coverage on training set
    # -----------------------------------------------
    print(f"\nCoverage analysis on training set:")
    covered_samples = 0
    
    for X_sample in X_train:
        model.population.makeEvalMatchSet(model, X_sample)
        if len(model.population.matchSet) > 0:
            covered_samples += 1
        model.population.clearSets()
    
    coverage = covered_samples / len(X_train)
    print(f"  Samples with matching rules: {covered_samples}/{len(X_train)} ({coverage*100:.1f}%)")
    
    if coverage < 0.5:
        print(f"  ⚠️  PROBLEM: Less than 50% of samples match ANY rule!")
        print(f"     This means most predictions default to majority class.")
    
    # -----------------------------------------------
    # Test predictions
    # -----------------------------------------------
    print(f"\nPrediction check on test set:")
    y_pred = model.predict(X_test)
    
    pred_class_0 = np.sum(y_pred == 0)
    pred_class_1 = np.sum(y_pred == 1)
    true_class_0 = np.sum(y_test == 0)
    true_class_1 = np.sum(y_test == 1)
    
    print(f"  Predicted class 0: {pred_class_0}/{len(y_test)}")
    print(f"  Predicted class 1: {pred_class_1}/{len(y_test)}")
    print(f"  True class 0: {true_class_0}/{len(y_test)}")
    print(f"  True class 1: {true_class_1}/{len(y_test)}")
    
    if pred_class_0 == len(y_test) or pred_class_1 == len(y_test):
        print(f"  ⚠️  PROBLEM: Model predicts only ONE class!")
        print(f"     Indicates matching set is always empty or rules favor majority class.")
    
    # -----------------------------------------------
    # Try to overfit on training set
    # -----------------------------------------------
    print(f"\nOverfit check (training set predictions):")
    y_pred_train = model.predict(X_train)
    train_acc = np.mean(y_pred_train == y_train)
    print(f"  Training accuracy: {train_acc:.4f}")
    
    if train_acc < 0.65:
        print(f"  ⚠️  CRITICAL: Model cannot even overfit training data!")
        print(f"     This suggests a fundamental issue with rule matching/prediction logic.")
    
    # -----------------------------------------------
    # Check environment data format
    # -----------------------------------------------
    print(f"\nEnvironment/Data format inspection:")
    print(f"  Discrete phenotype: {model.env.formatData.discretePhenotype}")
    print(f"  Phenotype list: {model.env.formatData.phenotypeList}")
    print(f"  Num attributes: {model.env.formatData.numAttributes}")
    print(f"  Num train instances: {model.env.formatData.numTrainInstances}")
    
    # -----------------------------------------------
    # Check tracking data
    # -----------------------------------------------
    print(f"\nTracking data (iteration history):")
    if hasattr(model, 'record') and hasattr(model.record, 'recordList'):
        print(f"  Total iterations tracked: {len(model.record.recordList)}")
        if len(model.record.recordList) > 0:
            last_record = model.record.recordList[-1]
            print(f"  Last iteration macro pop: {last_record[3]}")
            print(f"  Last iteration match set size: {last_record[6]}")
            print(f"  Last iteration correct set size: {last_record[7]}")
    
    print("\n" + "="*70)
    print("END INSPECTION")
    print("="*70)


if __name__ == "__main__":
    inspect_model_internals()