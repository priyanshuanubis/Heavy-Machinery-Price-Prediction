import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from src_pipeline import load_data, build_complete_overhaul_features, add_smooth_oof_target_encoding

def main():
    print("=== Training Fast Multi-Model Hybrid Stacking Architecture ===")
    train_raw, test_raw, sample_sub = load_data()
    X, y, X_test = build_complete_overhaul_features(train_raw, test_raw)

    target_cols = ["ProductConfigID", "Spec_FullDescriptor", "Spec_BaseClass", "VendorPartnerID", "RegionCode"]
    X, X_test = add_smooth_oof_target_encoding(X, y, X_test, target_cols=target_cols)

    # Ridge Features
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    X_num = X[num_cols].copy().fillna(0)
    X_test_num = X_test[num_cols].copy().fillna(0)

    # LightGBM Features
    categorical_cols = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
    X_lgb = X.copy()
    X_test_lgb = X_test.copy()
    for col in categorical_cols:
        X_lgb[col] = X_lgb[col].astype("category")
        X_test_lgb[col] = pd.Categorical(X_test_lgb[col], categories=X_lgb[col].cat.categories)

    # XGBoost Features
    X_xgb = X[num_cols].copy().fillna(0)
    X_test_xgb = X_test[num_cols].copy().fillna(0)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    oof_ridge = np.zeros(len(X))
    oof_lgb = np.zeros(len(X))
    oof_xgb = np.zeros(len(X))

    test_ridge = np.zeros(len(X_test))
    test_lgb = np.zeros(len(X_test))
    test_xgb = np.zeros(len(X_test))

    lgb_params = {
        "n_estimators": 1500,
        "learning_rate": 0.04,
        "num_leaves": 127,
        "max_depth": 10,
        "min_child_samples": 15,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }

    xgb_params = {
        "n_estimators": 1500,
        "learning_rate": 0.04,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "random_state": 42,
        "n_jobs": -1,
    }

    start_time = time.time()
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        fold_start = time.time()

        # 1. Ridge
        model_ridge = Ridge(alpha=10.0)
        model_ridge.fit(X_num.iloc[train_idx], y.iloc[train_idx])
        oof_ridge[val_idx] = model_ridge.predict(X_num.iloc[val_idx])
        test_ridge += model_ridge.predict(X_test_num) / kf.n_splits

        # 2. LightGBM
        model_lgb = lgb.LGBMRegressor(**lgb_params)
        model_lgb.fit(
            X_lgb.iloc[train_idx], y.iloc[train_idx],
            eval_set=[(X_lgb.iloc[val_idx], y.iloc[val_idx])],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        oof_lgb[val_idx] = model_lgb.predict(X_lgb.iloc[val_idx])
        test_lgb += model_lgb.predict(X_test_lgb) / kf.n_splits

        # 3. XGBoost
        model_xgb = xgb.XGBRegressor(**xgb_params)
        model_xgb.fit(
            X_xgb.iloc[train_idx], y.iloc[train_idx],
            eval_set=[(X_xgb.iloc[val_idx], y.iloc[val_idx])],
            verbose=False
        )
        oof_xgb[val_idx] = model_xgb.predict(X_xgb.iloc[val_idx])
        test_xgb += model_xgb.predict(X_test_xgb) / kf.n_splits

        print(f"Fold {fold} complete in {time.time()-fold_start:.2f}s | LGB RMSE={np.sqrt(np.mean((y.iloc[val_idx]-oof_lgb[val_idx])**2)):.5f}")

    # Meta Stacking
    X_meta = np.column_stack([oof_ridge, oof_lgb, oof_xgb])
    X_meta_test = np.column_stack([test_ridge, test_lgb, test_xgb])

    meta_model = Ridge(alpha=1.0)
    meta_model.fit(X_meta, y)
    
    stack_oof = meta_model.predict(X_meta)
    stack_rmse = np.sqrt(np.mean((y - stack_oof)**2))

    print(f"\n==========================================")
    print(f"⭐ Meta-Learner Stacked OOF RMSE: {stack_rmse:.5f}")
    print(f"==========================================")

    stack_test = meta_model.predict(X_meta_test)

    os.makedirs("submissions", exist_ok=True)
    submission = sample_sub.copy()
    target_col = [c for c in sample_sub.columns if c != "TransactionID"][0]
    submission[target_col] = np.maximum(0, np.expm1(stack_test))
    
    sub_path = "submissions/submission_stacked_hybrid.csv"
    submission.to_csv(sub_path, index=False)
    print(f"Saved submission to {sub_path} in {time.time()-start_time:.2f}s total.")

if __name__ == "__main__":
    main()
