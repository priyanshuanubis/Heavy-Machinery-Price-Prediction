"""
model_xgb_v2.py - XGBoost on numeric features
"""

import os
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from pipeline_v2 import run_full_pipeline


def main():
    print("=" * 60)
    print("XGBoost v2 - Numeric Features")
    print("=" * 60)

    X_train, y, X_test, cat_cols, sample_sub = run_full_pipeline()

    # XGBoost: numeric only (strictly filter by numeric dtype)
    num_cols = [c for c in X_train.columns if c not in cat_cols and pd.api.types.is_numeric_dtype(X_train[c])]
    X_tr_num = X_train[num_cols].copy()
    X_te_num = X_test[num_cols].copy()

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))

    params = {
        "n_estimators": 1000,
        "learning_rate": 0.08,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    }

    start = time.time()
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_tr_num), 1):
        fold_start = time.time()

        model = xgb.XGBRegressor(**params, early_stopping_rounds=100)
        model.fit(
            X_tr_num.iloc[tr_idx], y.iloc[tr_idx],
            eval_set=[(X_tr_num.iloc[va_idx], y.iloc[va_idx])],
            verbose=False,
        )

        preds = model.predict(X_tr_num.iloc[va_idx])
        oof[va_idx] = preds
        test_preds += model.predict(X_te_num) / kf.n_splits

        rmse = np.sqrt(np.mean((y.iloc[va_idx] - preds) ** 2))
        print(f"Fold {fold}: RMSE={rmse:.5f} | best_iter={model.best_iteration} | {time.time()-fold_start:.1f}s")

    overall = np.sqrt(np.mean((y - oof) ** 2))
    print(f"\nXGBoost v2 5-Fold OOF RMSE: {overall:.5f}")
    print(f"Total time: {time.time()-start:.1f}s")

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("submissions", exist_ok=True)
    np.save("outputs/oof_xgb_v2.npy", oof)
    np.save("outputs/test_xgb_v2.npy", test_preds)

    sub = sample_sub.copy()
    target_col = [c for c in sub.columns if c != "TransactionID"][0]
    sub[target_col] = np.maximum(0, np.expm1(test_preds))
    sub.to_csv("submissions/submission_xgb_v2.csv", index=False)
    print("Saved XGBoost v2 submission.")


if __name__ == "__main__":
    main()
