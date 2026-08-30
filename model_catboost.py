import os
import time
import numpy as np
import pandas as pd
import catboost as cb
from sklearn.model_selection import KFold
from src_pipeline import load_data, build_complete_overhaul_features, add_smooth_oof_target_encoding

def main():
    print("=== Training Overhauled CatBoost Model ===")
    train_raw, test_raw, sample_sub = load_data()
    X, y, X_test = build_complete_overhaul_features(train_raw, test_raw)

    target_cols = [
        "ProductConfigID", "Spec_FullDescriptor", "Spec_BaseClass", "VendorPartnerID", "RegionCode",
        ["ProductConfigID", "ManufactureYear"], ["Spec_BaseClass", "ManufactureYear"]
    ]
    X, X_test = add_smooth_oof_target_encoding(X, y, X_test, target_cols=target_cols)

    categorical_cols = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
    for col in categorical_cols:
        X[col] = X[col].fillna("Missing").astype(str)
        X_test[col] = X_test[col].fillna("Missing").astype(str)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    params = {
        "iterations": 2000,
        "learning_rate": 0.05,
        "depth": 8,
        "cat_features": categorical_cols,
        "random_seed": 42,
        "verbose": 200,
        "early_stopping_rounds": 50,
        "thread_count": -1,
    }

    start_time = time.time()
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        fold_start = time.time()
        print(f"\n--- Fold {fold} ---")
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

        model = cb.CatBoostRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=(X_va, y_va),
            use_best_model=True
        )

        val_pred = model.predict(X_va)
        oof_preds[val_idx] = val_pred
        rmse = np.sqrt(np.mean((y_va - val_pred) ** 2))
        cv_scores.append(rmse)
        print(f"Fold {fold} finished: RMSE={rmse:.5f} | {time.time()-fold_start:.2f}s")

        test_preds += model.predict(X_test) / kf.n_splits

    overall_rmse = np.sqrt(np.mean((y - oof_preds) ** 2))
    print(f"\nOverhauled CatBoost 5-Fold Overall OOF RMSE: {overall_rmse:.5f}")
    print(f"Total time: {time.time()-start_time:.2f}s")

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("submissions", exist_ok=True)

    np.save("outputs/oof_catboost.npy", oof_preds)
    np.save("outputs/test_catboost.npy", test_preds)

    submission = sample_sub.copy()
    target_col = [c for c in sample_sub.columns if c != "TransactionID"][0]
    submission[target_col] = np.maximum(0, np.expm1(test_preds))
    sub_path = "submissions/submission_catboost.csv"
    submission.to_csv(sub_path, index=False)
    print(f"Saved submission to {sub_path}")

if __name__ == "__main__":
    main()
