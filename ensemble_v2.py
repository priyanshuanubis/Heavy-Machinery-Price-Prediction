"""
ensemble_v2.py - Meta-learner stacking ensemble
Combines CatBoost, LightGBM, XGBoost OOF predictions
"""

import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from pipeline_v2 import load_data, build_features


def main():
    print("=" * 60)
    print("Ensemble v2 - Meta-Learner Stacking")
    print("=" * 60)

    train_raw, test_raw, sample_sub = load_data()
    _, y, _, _ = build_features(train_raw, test_raw)

    models = {
        "catboost_v2": ("outputs/oof_catboost_v2.npy", "outputs/test_catboost_v2.npy"),
        "lgbm_v2": ("outputs/oof_lgbm_v2.npy", "outputs/test_lgbm_v2.npy"),
        "xgb_v2": ("outputs/oof_xgb_v2.npy", "outputs/test_xgb_v2.npy"),
    }

    oof_dict = {}
    test_dict = {}

    for name, (oof_path, test_path) in models.items():
        if os.path.exists(oof_path) and os.path.exists(test_path):
            oof_dict[name] = np.load(oof_path)
            test_dict[name] = np.load(test_path)
            rmse = np.sqrt(np.mean((y - oof_dict[name]) ** 2))
            print(f"  {name}: OOF RMSE = {rmse:.5f}")
        else:
            print(f"  {name}: NOT FOUND - skipping")

    if len(oof_dict) < 2:
        print("Need at least 2 models for ensemble. Run individual models first.")
        return

    model_names = list(oof_dict.keys())
    oof_matrix = np.column_stack([oof_dict[m] for m in model_names])
    test_matrix = np.column_stack([test_dict[m] for m in model_names])

    # Optimize weights
    def loss(weights):
        w = weights / weights.sum()
        pred = oof_matrix @ w
        return np.sqrt(np.mean((y - pred) ** 2))

    n = len(model_names)
    res = minimize(
        loss,
        x0=np.ones(n) / n,
        bounds=[(0, 1)] * n,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
        method="SLSQP",
    )
    best_w = res.x / res.x.sum()

    print("\nOptimal Weights:")
    for name, w in zip(model_names, best_w):
        print(f"  {name}: {w:.4f}")

    ensemble_oof = oof_matrix @ best_w
    ensemble_rmse = np.sqrt(np.mean((y - ensemble_oof) ** 2))
    print(f"\nEnsemble OOF RMSE: {ensemble_rmse:.5f}")

    ensemble_test = test_matrix @ best_w

    os.makedirs("submissions", exist_ok=True)
    sub = sample_sub.copy()
    target_col = [c for c in sub.columns if c != "TransactionID"][0]
    sub[target_col] = np.maximum(0, np.expm1(ensemble_test))
    sub.to_csv("submissions/submission_ensemble_v2.csv", index=False)
    print("Saved ensemble v2 submission.")


if __name__ == "__main__":
    main()
