"""
model_lgbm_v3.py - Tailored Ground-Up LightGBM Architecture
"""

import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from pipeline_v3_clean import load_raw_data, build_ground_up_features, compute_smoothed_oof_target_encoding


def main():
    print("=" * 60)
    print("LightGBM v3 - Tailored Ground-Up Model Architecture")
    print("=" * 60)

    tr_raw, te_raw, sample_sub = load_raw_data()
    X_tr, y, X_te, cat_cols, num_cols = build_ground_up_features(tr_raw, te_raw)
    X_tr, X_te = compute_smoothed_oof_target_encoding(X_tr, y, X_te)

    # Prepare categorical columns for LightGBM
    for c in cat_cols:
        X_tr[c] = X_tr[c].astype("category")
        X_te[c] = pd.Categorical(X_te[c], categories=X_tr[c].cat.categories)

    feature_cols = [c for c in X_tr.columns if c not in ["_is_test"]]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(X_tr))
    test_preds = np.zeros(len(X_te))

    params = {
        "n_estimators": 2500,
        "learning_rate": 0.05,
        "num_leaves": 127,
        "max_depth": 10,
        "min_child_samples": 20,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "cat_smooth": 20,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }

    start = time.time()
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_tr), 1):
        f_start = time.time()

        tr_x, va_x = X_tr.iloc[tr_idx][feature_cols], X_tr.iloc[va_idx][feature_cols]
        tr_y, va_y = y[tr_idx], y[va_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            tr_x, tr_y,
            eval_set=[(va_x, va_y)],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        )

        preds = model.predict(va_x)
        oof[va_idx] = preds
        test_preds += model.predict(X_te[feature_cols]) / kf.n_splits

        fold_rmse = np.sqrt(np.mean((va_y - preds) ** 2))
        print(f"Fold {fold} RMSE: {fold_rmse:.5f} | Best Iter: {model.best_iteration_} | Time: {time.time()-f_start:.1f}s")

    total_rmse = np.sqrt(np.mean((y - oof) ** 2))
    print(f"\n LightGBM v3 5-Fold OOF RMSE: {total_rmse:.5f} | Total Time: {time.time()-start:.1f}s")

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("submissions", exist_ok=True)
    np.save("outputs/oof_lgbm_v3.npy", oof)
    np.save("outputs/test_lgbm_v3.npy", test_preds)

    sub = sample_sub.copy()
    target_col = [c for c in sub.columns if c != "TransactionID"][0]
    sub[target_col] = np.maximum(0, np.expm1(test_preds))
    sub.to_csv("submissions/submission_lgbm_v3.csv", index=False)
    print("Saved LightGBM v3 submission.")


if __name__ == "__main__":
    main()
