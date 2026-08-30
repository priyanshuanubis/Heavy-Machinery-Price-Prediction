"""
Fixed Cell 9 Code for Level-1 Base Models Training (LightGBM, CatBoost, XGBoost, ExtraTrees).

Fixes Applied:
1. Exclude unparsed raw text columns ('Spec_FullDescriptor', 'FunctionalClassification') from feature set.
2. Automatically detect ALL categorical / object / string columns in feature matrix X.
3. Construct the UNION of categories across train & test sets before building pandas Categorical objects.
   - Resolves CatBoostError: Invalid type for cat_feature[...]=NaN (unseen test set categories becoming NaNs).
   - Resolves LightGBM ValueError: pandas dtypes must be int, float or bool.
4. Perform OrdinalEncoding on all categorical features for XGBoost & ExtraTrees (casts to float32).
"""

import time
import numpy as np
import pandas as pd

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

# 1. Feature & Target Selection
drop_cols = ["SalesID", "TransactionDate", "TargetValue", "AssetID", "Spec_FullDescriptor", "FunctionalClassification"]
feature_cols = [c for c in train_te.columns if c not in drop_cols]

X = train_te[feature_cols].copy()
y_log = np.log1p(train_te["TargetValue"].clip(lower=0))
X_test = test_te[feature_cols].copy()

# 2. Automatically detect ALL categorical / object / string columns in X
cat_cols = [c for c in X.columns if X[c].dtype == "object" or X[c].dtype == "string" or isinstance(X[c].dtype, pd.CategoricalDtype)]

# 3. Prepare native categorical types for LightGBM & CatBoost using the UNION of categories across train & test
# This prevents unseen categories in X_test from becoming NaNs which cause CatBoost predict errors
for col in cat_cols:
    all_cats = sorted(list(set(X[col].fillna("Missing").astype(str).unique()) | set(X_test[col].fillna("Missing").astype(str).unique())))
    X[col] = pd.Categorical(X[col].fillna("Missing").astype(str), categories=all_cats)
    X_test[col] = pd.Categorical(X_test[col].fillna("Missing").astype(str), categories=all_cats)

# 4. Numerical / Ordinal Encoded Matrix for XGBoost & ExtraTrees
X_num = X.copy()
X_test_num = X_test.copy()

for col in cat_cols:
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_num[col] = enc.fit_transform(X_num[[col]].astype(str)).astype(np.float32)
    X_test_num[col] = enc.transform(X_test_num[[col]].astype(str)).astype(np.float32)

X_num = X_num.fillna(-999)
X_test_num = X_test_num.fillna(-999)

print(f"[DATA] Native Feature Matrix Assembled: {X.shape[1]} features ({len(cat_cols)} categorical)")

# 5. K-Fold Cross Validation Setup
N_SPLITS = 10
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
target_bins = pd.qcut(y_log, q=10, labels=False, duplicates="drop")

oof_lgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_et = np.zeros(len(X))

test_lgb = np.zeros(len(X_test))
test_cat = np.zeros(len(X_test))
test_xgb = np.zeros(len(X_test))
test_et = np.zeros(len(X_test))

# Hyperparameters
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.015,
    "num_leaves": 127,
    "max_depth": 11,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_estimators": 4500,
    "verbose": -1,
    "n_jobs": -1
}

cat_params = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "learning_rate": 0.02,
    "depth": 8,
    "l2_leaf_reg": 5,
    "subsample": 0.8,
    "random_seed": 42,
    "iterations": 4000,
    "verbose": False,
    "thread_count": -1
}

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.012,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_estimators": 4500,
    "tree_method": "hist",
    "n_jobs": -1
}

et_params = {
    "n_estimators": 200,
    "max_depth": 24,
    "min_samples_split": 4,
    "n_jobs": -1,
    "random_state": 42
}

EARLY_STOP = 150

print(f"=== Training Level-1 Base Models ({N_SPLITS}-Fold CV) ===")
for fold, (trn_idx, val_idx) in enumerate(skf.split(X, target_bins), start=1):
    f_start = time.time()
    
    # Train / Val splits for Native LightGBM & CatBoost
    X_tr_cat, y_tr = X.iloc[trn_idx], y_log.iloc[trn_idx]
    X_va_cat, y_va = X.iloc[val_idx], y_log.iloc[val_idx]

    # Train / Val splits for XGBoost & ExtraTrees
    X_tr_num = X_num.iloc[trn_idx]
    X_va_num = X_num.iloc[val_idx]

    # 1. Native LightGBM
    m_lgb = lgb.LGBMRegressor(**lgb_params)
    m_lgb.fit(X_tr_cat, y_tr, eval_set=[(X_va_cat, y_va)],
              callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
    oof_lgb[val_idx] = m_lgb.predict(X_va_cat)
    test_lgb += m_lgb.predict(X_test) / N_SPLITS

    # 2. CatBoost
    m_cat = CatBoostRegressor(**cat_params)
    cat_feature_indices = [col for col in cat_cols if col in X_tr_cat.columns]
    m_cat.fit(X_tr_cat, y_tr, cat_features=cat_feature_indices, eval_set=(X_va_cat, y_va),
              early_stopping_rounds=EARLY_STOP, verbose=False)
    oof_cat[val_idx] = m_cat.predict(X_va_cat)
    test_cat += m_cat.predict(X_test) / N_SPLITS

    # 3. XGBoost
    m_xgb = xgb.XGBRegressor(**xgb_params)
    m_xgb.fit(X_tr_num, y_tr, eval_set=[(X_va_num, y_va)], verbose=False)
    oof_xgb[val_idx] = m_xgb.predict(X_va_num)
    test_xgb += m_xgb.predict(X_test_num) / N_SPLITS

    # 4. ExtraTrees
    m_et = ExtraTreesRegressor(**et_params)
    m_et.fit(X_tr_num, y_tr)
    oof_et[val_idx] = m_et.predict(X_va_num)
    test_et += m_et.predict(X_test_num) / N_SPLITS

    fold_lgb = np.sqrt(mean_squared_error(y_va, oof_lgb[val_idx]))
    print(f"Fold {fold}/{N_SPLITS} - Native LGB: {fold_lgb:.5f} - {time.time() - f_start:.1f}s")

lgb_rmsle = np.sqrt(mean_squared_error(y_log, oof_lgb))
cat_rmsle = np.sqrt(mean_squared_error(y_log, oof_cat))
xgb_rmsle = np.sqrt(mean_squared_error(y_log, oof_xgb))
et_rmsle = np.sqrt(mean_squared_error(y_log, oof_et))

print()
print(f"[METRIC] LightGBM (Native) 10-Fold OOF RMSLE: {lgb_rmsle:.5f}")
print(f"[METRIC] CatBoost (Native) 10-Fold OOF RMSLE: {cat_rmsle:.5f}")
print(f"[METRIC] XGBoost  (Hist)   10-Fold OOF RMSLE: {xgb_rmsle:.5f}")
print(f"[METRIC] ExtraTrees        10-Fold OOF RMSLE: {et_rmsle:.5f}")
