import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from src_pipeline import load_data, build_complete_overhaul_features, add_smooth_oof_target_encoding

def main():
    print("=== Training Overhauled LightGBM Model ===")
    train_raw, test_raw, sample_sub = load_data()
    X, y, X_test = build_complete_overhaul_features(train_raw, test_raw)

    target_cols = ["ProductConfigID", "Spec_FullDescriptor", "Spec_BaseClass", "VendorPartnerID", "RegionCode"]
    X, X_test = add_smooth_oof_target_encoding(X, y, X_test, target_cols=target_cols)

    categorical_cols = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
    for col in categorical_cols:
        X[col] = X[col].astype("category")
        X_test[col] = pd.Categorical(X_test[col], categories=X[col].cat.categories)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    params = {
        "n_estimators": 2500,
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

    start_time = time.time()
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        fold_start = time.time()
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )

        val_pred = model.predict(X_va)
        oof_preds[val_idx] = val_pred
        rmse = np.sqrt(np.mean((y_va - val_pred) ** 2))
        cv_scores.append(rmse)
        print(f"Fold {fold}: RMSE={rmse:.5f} | {time.time()-fold_start:.2f}s")

        test_preds += model.predict(X_test) / kf.n_splits

    overall_rmse = np.sqrt(np.mean((y - oof_preds) ** 2))
    print(f"\nOverhauled LightGBM 5-Fold Overall OOF RMSE: {overall_rmse:.5f}")
    print(f"Total time: {time.time()-start_time:.2f}s")

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("submissions", exist_ok=True)

    np.save("outputs/oof_lightgbm.npy", oof_preds)
    np.save("outputs/test_lightgbm.npy", test_preds)

    submission = sample_sub.copy()
    target_col = [c for c in sample_sub.columns if c != "TransactionID"][0]
    submission[target_col] = np.maximum(0, np.expm1(test_preds))
    sub_path = "submissions/submission_lightgbm.csv"
    submission.to_csv(sub_path, index=False)
    print(f"Saved submission to {sub_path}")

if __name__ == "__main__":
    main()
