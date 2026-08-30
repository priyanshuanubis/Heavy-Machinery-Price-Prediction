"""
model_catboost_v3.py - Tailored Ground-Up CatBoost Architecture
"""

import os
import time
import numpy as np
import pandas as pd
import catboost as cb
from sklearn.model_selection import KFold
from pipeline_v3_clean import load_raw_data, build_ground_up_features, compute_smoothed_oof_target_encoding


def main():
    print("=" * 60)
    print("CatBoost v3 - Tailored Ground-Up Model Architecture")
    print("=" * 60)

    tr_raw, te_raw, sample_sub = load_raw_data()
    X_tr, y, X_te, cat_cols, num_cols = build_ground_up_features(tr_raw, te_raw)
    X_tr, X_te = compute_smoothed_oof_target_encoding(X_tr, y, X_te)

    # Convert categoricals to string for CatBoost
    for c in cat_cols:
        X_tr[c] = X_tr[c].fillna("_Missing_").astype(str)
        X_te[c] = X_te[c].fillna("_Missing_").astype(str)

    feature_cols = [c for c in X_tr.columns if c not in ["_is_test"]]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(X_tr))
    test_preds = np.zeros(len(X_te))

    params = {
        "iterations": 2500,
        "learning_rate": 0.06,
        "depth": 7,
        "l2_leaf_reg": 4.0,
        "cat_features": cat_cols,
        "random_seed": 42,
        "verbose": 250,
        "early_stopping_rounds": 150,
        "thread_count": -1,
    }

    start = time.time()
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_tr), 1):
        f_start = time.time()

        tr_x, va_x = X_tr.iloc[tr_idx][feature_cols], X_tr.iloc[va_idx][feature_cols]
        tr_y, va_y = y[tr_idx], y[va_idx]

        model = cb.CatBoostRegressor(**params)
        model.fit(tr_x, tr_y, eval_set=(va_x, va_y), use_best_model=True)

        preds = model.predict(va_x)
        oof[va_idx] = preds
        test_preds += model.predict(X_te[feature_cols]) / kf.n_splits

        fold_rmse = np.sqrt(np.mean((va_y - preds) ** 2))
        print(f"Fold {fold} RMSE: {fold_rmse:.5f} | Best Iter: {model.best_iteration_} | Time: {time.time()-f_start:.1f}s")

    total_rmse = np.sqrt(np.mean((y - oof) ** 2))
    print(f"\n CatBoost v3 5-Fold OOF RMSE: {total_rmse:.5f} | Total Time: {time.time()-start:.1f}s")

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("submissions", exist_ok=True)
    np.save("outputs/oof_catboost_v3.npy", oof)
    np.save("outputs/test_catboost_v3.npy", test_preds)

    sub = sample_sub.copy()
    target_col = [c for c in sub.columns if c != "TransactionID"][0]
    sub[target_col] = np.maximum(0, np.expm1(test_preds))
    sub.to_csv("submissions/submission_catboost_v3.csv", index=False)
    print("Saved CatBoost v3 submission.")


if __name__ == "__main__":
    main()
