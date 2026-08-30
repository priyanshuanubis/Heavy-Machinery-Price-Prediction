"""
model_catboost_v2.py - CatBoost with native categorical handling
Priority: SPEED + native categoricals (CatBoost's key advantage)
"""

import os
import time
import numpy as np
import pandas as pd
import catboost as cb
from sklearn.model_selection import KFold
from pipeline_v2 import run_full_pipeline


def main():
    print("=" * 60)
    print("CatBoost v2 - Native Categoricals")
    print("=" * 60)

    X_train, y, X_test, cat_cols, sample_sub = run_full_pipeline()

    # CatBoost needs string categoricals
    for c in cat_cols:
        X_train[c] = X_train[c].fillna("_Missing_").astype(str)
        X_test[c] = X_test[c].fillna("_Missing_").astype(str)

    # All feature columns
    feature_cols = list(X_train.columns)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))

    params = {
        "iterations": 1200,
        "learning_rate": 0.10,
        "depth": 6,
        "l2_leaf_reg": 3,
        "cat_features": cat_cols,
        "random_seed": 42,
        "verbose": 300,
        "early_stopping_rounds": 100,
        "thread_count": -1,
    }

    start = time.time()
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train), 1):
        fold_start = time.time()

        tr_x, va_x = X_train.iloc[tr_idx][feature_cols], X_train.iloc[va_idx][feature_cols]
        tr_y, va_y = y.iloc[tr_idx], y.iloc[va_idx]

        model = cb.CatBoostRegressor(**params)
        model.fit(tr_x, tr_y, eval_set=(va_x, va_y), use_best_model=True)

        preds = model.predict(va_x)
        oof[va_idx] = preds
        test_preds += model.predict(X_test[feature_cols]) / kf.n_splits

        rmse = np.sqrt(np.mean((va_y - preds) ** 2))
        print(f"Fold {fold}: RMSE={rmse:.5f} | best_iter={model.best_iteration_} | {time.time()-fold_start:.1f}s")

    overall = np.sqrt(np.mean((y - oof) ** 2))
    print(f"\nCatBoost v2 5-Fold OOF RMSE: {overall:.5f}")
    print(f"Total time: {time.time()-start:.1f}s")

    # Save
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("submissions", exist_ok=True)
    np.save("outputs/oof_catboost_v2.npy", oof)
    np.save("outputs/test_catboost_v2.npy", test_preds)

    sub = sample_sub.copy()
    target_col = [c for c in sub.columns if c != "TransactionID"][0]
    sub[target_col] = np.maximum(0, np.expm1(test_preds))
    sub.to_csv("submissions/submission_catboost_v2.csv", index=False)
    print("Saved CatBoost v2 submission.")


if __name__ == "__main__":
    main()
