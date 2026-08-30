import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from src_pipeline import load_data, build_complete_overhaul_features

def main():
    print("=== Training Dynamic Weighted Ensemble Model ===")
    train_raw, test_raw, sample_sub = load_data()
    _, y, _ = build_complete_overhaul_features(train_raw, test_raw)

    oof_dict = {}
    test_dict = {}
    
    models = ["lightgbm", "xgboost", "catboost"]
    for m in models:
        oof_path = f"outputs/oof_{m}.npy"
        test_path = f"outputs/test_{m}.npy"
        if os.path.exists(oof_path) and os.path.exists(test_path):
            oof_dict[m] = np.load(oof_path)
            test_dict[m] = np.load(test_path)
            rmse = np.sqrt(np.mean((y - oof_dict[m]) ** 2))
            print(f"Loaded {m} OOF RMSE: {rmse:.5f}")

    if not oof_dict:
        print("No trained model OOF files found in outputs/!")
        return

    model_names = list(oof_dict.keys())
    oof_matrix = np.column_stack([oof_dict[m] for m in model_names])
    test_matrix = np.column_stack([test_dict[m] for m in model_names])

    # Optimize weights to minimize RMSE
    def loss_func(weights):
        weights = weights / np.sum(weights)
        pred = oof_matrix @ weights
        return np.sqrt(np.mean((y - pred) ** 2))

    init_weights = np.ones(len(model_names)) / len(model_names)
    bounds = [(0, 1)] * len(model_names)
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    res = minimize(loss_func, init_weights, bounds=bounds, constraints=constraints)
    best_weights = res.x / np.sum(res.x)

    print("\nOptimal Model Weights:")
    for name, w in zip(model_names, best_weights):
        print(f"  {name}: {w:.4f}")

    ensemble_oof = oof_matrix @ best_weights
    ensemble_rmse = np.sqrt(np.mean((y - ensemble_oof) ** 2))
    print(f"\nEnsemble OOF RMSE: {ensemble_rmse:.5f}")

    ensemble_test = test_matrix @ best_weights
    
    submission = sample_sub.copy()
    target_col = [c for c in sample_sub.columns if c != "TransactionID"][0]
    submission[target_col] = np.maximum(0, np.expm1(ensemble_test))
    
    sub_path = "submissions/submission_ensemble.csv"
    submission.to_csv(sub_path, index=False)
    print(f"Saved ensemble submission to {sub_path}")

if __name__ == "__main__":
    main()
