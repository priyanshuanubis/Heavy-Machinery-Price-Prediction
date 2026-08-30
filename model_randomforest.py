import os
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from src_pipeline import load_data, build_complete_overhaul_features

def main():
    print("=== Training Random Forest Model ===")
    train_raw, test_raw, sample_sub = load_data()
    X, y, X_test = build_complete_overhaul_features(train_raw, test_raw)

    num_cols = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
    cat_cols = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]), cat_cols),
        ]
    )

    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=14,
        min_samples_leaf=4,
        n_jobs=-1,
        random_state=42
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", rf_model)
    ])

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    start_time = time.time()
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        fold_start = time.time()
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

        pipeline.fit(X_tr, y_tr)
        val_pred = pipeline.predict(X_va)
        oof_preds[val_idx] = val_pred
        rmse = np.sqrt(np.mean((y_va - val_pred) ** 2))
        cv_scores.append(rmse)
        print(f"Fold {fold}: RMSE={rmse:.5f} | {time.time()-fold_start:.2f}s")

        test_preds += pipeline.predict(X_test) / kf.n_splits

    overall_rmse = np.sqrt(np.mean((y - oof_preds) ** 2))
    print(f"\nRandom Forest 5-Fold Overall OOF RMSE: {overall_rmse:.5f}")
    print(f"Total time: {time.time()-start_time:.2f}s")

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("submissions", exist_ok=True)

    np.save("outputs/oof_rf.npy", oof_preds)
    np.save("outputs/test_rf.npy", test_preds)

    submission = sample_sub.copy()
    target_col = [c for c in sample_sub.columns if c != "TransactionID"][0]
    submission[target_col] = np.maximum(0, np.expm1(test_preds))
    sub_path = "submissions/submission_rf.csv"
    submission.to_csv(sub_path, index=False)
    print(f"Saved submission to {sub_path}")

if __name__ == "__main__":
    main()
