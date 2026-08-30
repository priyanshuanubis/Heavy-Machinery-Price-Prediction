"""
model_lgbm_v2.py - LightGBM with target encodings replacing raw categoricals
"""

import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from pipeline_v2 import run_full_pipeline


def main():
    print("=" * 60)
    print("LightGBM v2 - Target Encoded Features")
    print("=" * 60)

    X_train, y, X_test, cat_cols, sample_sub = run_full_pipeline()

    # Convert categoricals for LightGBM
    for c in cat_cols:
        X_train[c] = X_train[c].astype("category")
        X_test[c] = pd.Categorical(X_test[c], categories=X_train[c].cat.categories)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))

    params = {
        "n_estimators": 3000,
        "learning_rate": 0.03,
        "num_leaves": 255,
        "max_depth": 12,
        "min_child_samples": 15,
        "feature_fraction": 0.70,
        "bagging_fraction": 0.80,
        "bagging_freq": 1,
        "cat_smooth": 15,
        "max_bin": 512,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }

    start = time.time()
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train), 1):
        fold_start = time.time()

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train.iloc[tr_idx], y.iloc[tr_idx],
            eval_set=[(X_train.iloc[va_idx], y.iloc[va_idx])],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        )

        preds = model.predict(X_train.iloc[va_idx])
        oof[va_idx] = preds
        test_preds += model.predict(X_test) / kf.n_splits

        rmse = np.sqrt(np.mean((y.iloc[va_idx] - preds) ** 2))
        print(f"Fold {fold}: RMSE={rmse:.5f} | best_iter={model.best_iteration_} | {time.time()-fold_start:.1f}s")

    overall = np.sqrt(np.mean((y - oof) ** 2))
    print(f"\nLightGBM v2 5-Fold OOF RMSE: {overall:.5f}")
    print(f"Total time: {time.time()-start:.1f}s")

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("submissions", exist_ok=True)
    np.save("outputs/oof_lgbm_v2.npy", oof)
    np.save("outputs/test_lgbm_v2.npy", test_preds)

    sub = sample_sub.copy()
    target_col = [c for c in sub.columns if c != "TransactionID"][0]
    sub[target_col] = np.maximum(0, np.expm1(test_preds))
    sub.to_csv("submissions/submission_lgbm_v2.csv", index=False)
    print("Saved LightGBM v2 submission.")


if __name__ == "__main__":
    main()
