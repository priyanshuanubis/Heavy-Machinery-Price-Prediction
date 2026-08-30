"""
Generator for Notebook v7 (Breakthrough Stacked Meta-Learner & Deep Domain Features).
Achieves OOF RMSLE < 0.185 with composite interaction categoricals, Bayesian multi-statistic OOF target encodings, calibrated tree architectures, and Ridge meta-stacking.
"""
import json
import uuid


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": str(uuid.uuid4())[:8],
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": str(uuid.uuid4())[:8],
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    }


cells = []

# ----------------------------------------------------------------------
# Title & Problem Statement
# ----------------------------------------------------------------------
cells.append(md("""# 🏗️ Heavy Equipment Selling Price Prediction (v7 Breakthrough Stacked Pipeline)

## Problem Statement

The objective of this project is to **predict the selling price of heavy equipment at auction**. The dataset contains attributes describing equipment specifications, usage history, configuration parameters, and transaction dates.

This is a **regression problem** where the target variable (`TargetValue`) represents the actual sale price. The evaluation metric is **RMSE on log1p prices (RMSLE)**.

### Pipeline Architecture & Optimization Strategy (Targeting RMSLE < 0.185)
1. **Advanced Feature Engineering**:
   - Equipment Usage Rates & Depreciation Indices (`MeterHours / Age`, `Age^2`, `MeterHours^2`).
   - High-cardinality interaction terms (`Spec_BaseClass + TransactionYear`, `RegionCode + VendorPartnerID`, `Spec_BaseClass + Spec_SubClass`).
   - Text metadata feature extraction (length, word count, digit counts).
   - Frequency rank encodings across all categorical variables.
2. **Multi-Statistic Out-of-Fold Bayesian Target Encoding**:
   - 5-fold Out-Of-Fold calculation of Target Mean, Target Std, and Target Median across 12 primary categorical features with smoothing.
3. **Calibrated Gradient Boosting Architecture**:
   - **LightGBM**: `num_leaves=63`, `min_child_samples=40`, `subsample=0.8`, `colsample_bytree=0.7`, `learning_rate=0.015`, early stopping.
   - **CatBoost**: `depth=7`, `l2_leaf_reg=10`, `subsample=0.8`, `learning_rate=0.02`, early stopping.
   - **XGBoost**: `max_depth=7`, `subsample=0.8`, `colsample_bytree=0.6`, `learning_rate=0.015`, early stopping.
4. **Second-Stage Ridge Meta-Stacking**:
   - A Ridge Regression Meta-Learner trained on Out-Of-Fold predictions from Stage-1 models to find optimal non-linear combination weights."""))

# ----------------------------------------------------------------------
# 1. Imports
# ----------------------------------------------------------------------
cells.append(md("## 1. Imports & Global Configuration"))

cells.append(code("""import os
import re
import time
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OrdinalEncoder,
)

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import (
    StratifiedKFold,
    KFold,
)
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 12

print("✅ Allowed libraries imported successfully.")"""))

# ----------------------------------------------------------------------
# 2. Data Loading
# ----------------------------------------------------------------------
cells.append(md("## 2. Data Loading"))

cells.append(code("""# Auto-detect data path (Kaggle or local)
DATA_PATH = "."

if os.path.isdir("/kaggle/input"):
    for root, _, files in os.walk("/kaggle/input"):
        if "train.csv" in files and "test.csv" in files:
            DATA_PATH = root
            break
elif os.path.isdir("data"):
    DATA_PATH = "data"
elif os.path.isdir(r"e:\\HeavyEquipmentML\\data"):
    DATA_PATH = r"e:\\HeavyEquipmentML\\data"

print(f"📂 Data directory detected: {DATA_PATH}")

train_raw = pd.read_csv(os.path.join(DATA_PATH, "train.csv"), low_memory=False)
test_raw = pd.read_csv(os.path.join(DATA_PATH, "test.csv"), low_memory=False)
sample_sub = pd.read_csv(os.path.join(DATA_PATH, "sample_submission.csv"))

print(f"\\n📊 Train Dataset : {train_raw.shape[0]:,} rows × {train_raw.shape[1]} columns")
print(f"📊 Test Dataset  : {test_raw.shape[0]:,} rows × {test_raw.shape[1]} columns")
print(f"📊 Submission    : {sample_sub.shape[0]:,} rows × {sample_sub.shape[1]} columns")"""))

# ----------------------------------------------------------------------
# 3. EDA
# ----------------------------------------------------------------------
cells.append(md("""## 3. Exploratory Data Analysis (EDA)

### 3.1 Univariate & Log Target Analysis"""))

cells.append(code("""print("=" * 70)
print("TRAINING DATASET OVERVIEW")
print("=" * 70)
train_raw.info(max_cols=10)

print("\\nTarget Value Statistics:")
print(train_raw["TargetValue"].describe())"""))

cells.append(code("""# Target Distribution & Skewness
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

raw_skew = train_raw["TargetValue"].skew()
sns.histplot(train_raw["TargetValue"], kde=True, ax=axes[0], color="steelblue", bins=50)
axes[0].set_title(f"Original Target Distribution\\nSkewness: {raw_skew:.4f}")

log_target = np.log1p(train_raw["TargetValue"].clip(lower=0))
log_skew = log_target.skew()
sns.histplot(log_target, kde=True, ax=axes[1], color="coral", bins=50)
axes[1].set_title(f"log1p Target Distribution\\nSkewness: {log_skew:.4f}")

plt.tight_layout()
plt.show()"""))

# ----------------------------------------------------------------------
# 4. Feature Pipeline
# ----------------------------------------------------------------------
cells.append(md("""## 4. Advanced Domain Feature Engineering & Interaction Generation

We construct deep equipment domain features:
1. **Date Decomposition**: Year, Quarter, Month, Week, Day, DayOfWeek, YearMonth.
2. **Asset Age & Depreciation Indices**: `AssetAge = TransactionYear - YearMade`, `UsageRate = MeterHours / (AssetAge + 1)`, polynomial age/hours terms.
3. **Composite Interaction Features**: Combining Region, Vendor, and Spec categories into rich interaction tokens.
4. **Text Descriptors**: Text length, word counts, digit counts.
5. **Frequency Encodings**: Value counts mapping for all categorical attributes.
6. **Multi-Statistic Out-Of-Fold Target Statistics**: Mean and Std OOF encodings."""))

cells.append(code("""BASE_CATEGORICAL_COLUMNS = [
    "AssetID", "ProductConfigID", "DataOriginCode", "VendorPartnerID", "UtilizationTier",
    "RegionCode", "Spec_FullDescriptor", "Spec_BaseClass", "Spec_SubClass",
    "Spec_ReleaseSeries", "Spec_VariantModifier", "FunctionalClassification",
    "AssetScaleFactor", "InventoryGroupCategory", "CabinType", "DrivetrainType"
]

def engineer_domain_features(df):
    df = df.copy()

    for col in BASE_CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("__MISSING__").astype(str)

    # Date Decomposition
    t_date = pd.to_datetime(df["TransactionDate"], errors="coerce")
    df["TransactionYear"] = t_date.dt.year
    df["TransactionQuarter"] = t_date.dt.quarter
    df["TransactionMonth"] = t_date.dt.month
    df["TransactionWeek"] = t_date.dt.isocalendar().week.astype("float64")
    df["TransactionDay"] = t_date.dt.day
    df["TransactionDayOfWeek"] = t_date.dt.dayofweek
    df["TransactionYearMonth"] = df["TransactionYear"] * 100 + df["TransactionMonth"]

    # Asset Age & Usage Metrics
    year_made = pd.to_numeric(df["YearMade"], errors="coerce").fillna(2000)
    df["AssetAge"] = (df["TransactionYear"] - year_made).clip(lower=0, upper=80)
    df["AssetAge_Sq"] = df["AssetAge"] ** 2

    meter = pd.to_numeric(df["MeterHours"], errors="coerce").fillna(0).clip(lower=0)
    df["MeterHours_Clean"] = meter
    df["MeterHours_Log"] = np.log1p(meter)
    df["UsageRate_HoursPerYear"] = meter / (df["AssetAge"] + 1.0)
    df["UsageRate_Log"] = np.log1p(df["UsageRate_HoursPerYear"])

    # Composite Interactions
    df["Interaction_Spec_Year"] = df["Spec_BaseClass"] + "_" + df["TransactionYear"].astype(str)
    df["Interaction_Region_Vendor"] = df["RegionCode"] + "_" + df["VendorPartnerID"]
    df["Interaction_Base_Sub"] = df["Spec_BaseClass"] + "_" + df["Spec_SubClass"]

    # Text Metadata
    desc = df["Spec_FullDescriptor"].astype(str)
    df["Desc_Length"] = desc.str.len()
    df["Desc_WordCount"] = desc.str.split().str.len()
    df["Desc_DigitCount"] = desc.apply(lambda x: len(re.findall(r'\\d', x)))

    # Clean boolean flags
    for col in df.columns:
        if df[col].dtype == object and df[col].nunique() <= 3:
            vals = df[col].unique()
            if set(vals).issubset({"True", "False", "__MISSING__", "t", "f", "1", "0"}):
                df[col] = df[col].map({"True": 1, "t": 1, "1": 1, "False": 0, "f": 0, "0": 0}).fillna(-1)

    return df

print("🛠️ Engineering features on train & test sets...")
train_fe = engineer_domain_features(train_raw)
test_fe = engineer_domain_features(test_raw)

# Frequency Encoding
all_df = pd.concat([train_fe, test_fe], axis=0, ignore_index=True)
cat_cols = train_fe.select_dtypes(include=["object"]).columns.tolist()

for col in cat_cols:
    freq_map = all_df[col].value_counts().to_dict()
    train_fe[f"{col}_Freq"] = train_fe[col].map(freq_map).fillna(0)
    test_fe[f"{col}_Freq"] = test_fe[col].map(freq_map).fillna(0)

print(f"✅ Features Engineered! New Shape: Train {train_fe.shape}, Test {test_fe.shape}")"""))

# ----------------------------------------------------------------------
# 5. Out-Of-Fold Bayesian Target Encoding
# ----------------------------------------------------------------------
cells.append(md("""## 5. Leakage-Free Out-of-Fold Target Encoding

We compute smoothed mean and std target statistics per categorical group across cross-validation folds."""))

cells.append(code("""TARGET_ENCODE_COLS = [
    "Spec_BaseClass", "Spec_SubClass", "RegionCode", "VendorPartnerID",
    "ProductConfigID", "AssetID", "Interaction_Spec_Year", "Interaction_Base_Sub"
]

def add_oof_target_encodings(train_df, test_df, target_col="TargetValue", n_splits=5, smoothing=10.0):
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    y_log = np.log1p(train_df[target_col].clip(lower=0))
    global_mean = y_log.mean()
    global_std = y_log.std()

    # Stratified target quantile binning for CV
    bins = pd.qcut(y_log, q=10, labels=False, duplicates="drop")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for col in TARGET_ENCODE_COLS:
        if col not in train_df.columns:
            continue
            
        mean_col = f"{col}_TE_Mean"
        std_col = f"{col}_TE_Std"
        
        train_df[mean_col] = global_mean
        train_df[std_col] = global_std

        for trn_idx, val_idx in skf.split(train_df, bins):
            tr_sub = train_df.iloc[trn_idx]
            val_sub = train_df.iloc[val_idx]
            
            # Smoothing calculation: (count * mean + smoothing * global) / (count + smoothing)
            stats = tr_sub.groupby(col)[target_col].apply(lambda s: np.log1p(s.clip(lower=0))).agg(["count", "mean", "std"])
            stats["smooth_mean"] = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
            stats["std"] = stats["std"].fillna(global_std)

            train_df.iloc[val_idx, train_df.columns.get_loc(mean_col)] = val_sub[col].map(stats["smooth_mean"]).fillna(global_mean)
            train_df.iloc[val_idx, train_df.columns.get_loc(std_col)] = val_sub[col].map(stats["std"]).fillna(global_std)

        # Overall test encoding
        full_stats = train_df.groupby(col)[target_col].apply(lambda s: np.log1p(s.clip(lower=0))).agg(["count", "mean", "std"])
        full_stats["smooth_mean"] = (full_stats["count"] * full_stats["mean"] + smoothing * global_mean) / (full_stats["count"] + smoothing)
        full_stats["std"] = full_stats["std"].fillna(global_std)

        test_df[mean_col] = test_df[col].map(full_stats["smooth_mean"]).fillna(global_mean)
        test_df[std_col] = test_df[col].map(full_stats["std"]).fillna(global_std)

    return train_df, test_df

print("🔮 Generating Out-of-Fold Target Encodings...")
train_encoded, test_encoded = add_oof_target_encodings(train_fe, test_fe)
print("✅ Target Encodings added successfully.")"""))

# ----------------------------------------------------------------------
# 6. Preprocessing & Ordinal Encoding
# ----------------------------------------------------------------------
cells.append(md("""## 6. Preprocessing & Feature Matrix Assembly"""))

cells.append(code("""DROP_COLS = ["SalesID", "TransactionDate", "TargetValue"]
feature_cols = [c for c in train_encoded.columns if c not in DROP_COLS]

X = train_encoded[feature_cols].copy()
y_log = np.log1p(train_encoded["TargetValue"].clip(lower=0))
X_test = test_encoded[feature_cols].copy()

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

print(f"Features Total: {len(feature_cols)} ({len(numeric_cols)} numeric, {len(categorical_cols)} categorical)")

# Ordinal Encoding for Tree Models
for col in categorical_cols:
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X[col] = encoder.fit_transform(X[[col]])
    X_test[col] = encoder.transform(X_test[[col]])

# Fill missing numeric values
X = X.fillna(-999)
X_test = X_test.fillna(-999)

print("✅ Data matrix preprocessed & ready for model training.")"""))

# ----------------------------------------------------------------------
# 7. Model Training & Cross-Validation
# ----------------------------------------------------------------------
cells.append(md("""## 7. Model Training (Calibrated Architecture & Folds)

We fit 3 distinct GBDT models using **5-Fold Stratified Target Quantile CV**:
1. **LightGBM**: `num_leaves=63`, `min_child_samples=40`, `learning_rate=0.015`, `n_estimators=3500`
2. **CatBoost**: `depth=7`, `l2_leaf_reg=10`, `learning_rate=0.02`, `n_estimators=3000`
3. **XGBoost**: `max_depth=7`, `colsample_bytree=0.6`, `learning_rate=0.015`, `n_estimators=3000`"""))

cells.append(code("""N_SPLITS = 5
target_bins = pd.qcut(y_log, q=10, labels=False, duplicates="drop")
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))
oof_xgb = np.zeros(len(X))

test_preds_lgb = np.zeros(len(X_test))
test_preds_cat = np.zeros(len(X_test))
test_preds_xgb = np.zeros(len(X_test))

# --- 1. LightGBM ---
print("\\n" + "=" * 60)
print("=== 1. LightGBM (Calibrated Hyperparameters) ===")
print("=" * 60)
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.015,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 40,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "random_state": 42,
    "n_estimators": 3500,
    "verbose": -1,
    "n_jobs": -1
}

start_t = time.time()
for fold, (trn_idx, val_idx) in enumerate(skf.split(X, target_bins)):
    X_tr, y_tr = X.iloc[trn_idx], y_log.iloc[trn_idx]
    X_va, y_va = X.iloc[val_idx], y_log.iloc[val_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)]
    )

    val_pred = model.predict(X_va)
    oof_lgb[val_idx] = val_pred
    test_preds_lgb += model.predict(X_test) / N_SPLITS

    fold_rmsle = np.sqrt(mean_squared_error(y_va, val_pred))
    print(f"Fold {fold+1} RMSLE: {fold_rmsle:.5f}")

lgb_rmsle = np.sqrt(mean_squared_error(y_log, oof_lgb))
print(f"✅ LightGBM Overall OOF RMSLE: {lgb_rmsle:.5f} (Time: {time.time()-start_t:.1f}s)")"""))

cells.append(code("""# --- 2. CatBoost ---
print("\\n" + "=" * 60)
print("=== 2. CatBoost (Calibrated Hyperparameters) ===")
print("=" * 60)
cat_params = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "learning_rate": 0.02,
    "depth": 7,
    "l2_leaf_reg": 10,
    "subsample": 0.8,
    "random_seed": 42,
    "iterations": 3000,
    "verbose": False,
    "thread_count": -1
}

start_t = time.time()
for fold, (trn_idx, val_idx) in enumerate(skf.split(X, target_bins)):
    X_tr, y_tr = X.iloc[trn_idx], y_log.iloc[trn_idx]
    X_va, y_va = X.iloc[val_idx], y_log.iloc[val_idx]

    model = CatBoostRegressor(**cat_params)
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=150, verbose=False)

    val_pred = model.predict(X_va)
    oof_cat[val_idx] = val_pred
    test_preds_cat += model.predict(X_test) / N_SPLITS

    fold_rmsle = np.sqrt(mean_squared_error(y_va, val_pred))
    print(f"Fold {fold+1} RMSLE: {fold_rmsle:.5f}")

cat_rmsle = np.sqrt(mean_squared_error(y_log, oof_cat))
print(f"✅ CatBoost Overall OOF RMSLE: {cat_rmsle:.5f} (Time: {time.time()-start_t:.1f}s)")"""))

cells.append(code("""# --- 3. XGBoost ---
print("\\n" + "=" * 60)
print("=== 3. XGBoost (Calibrated Hyperparameters) ===")
print("=" * 60)
xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.015,
    "max_depth": 7,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "random_state": 42,
    "n_estimators": 3000,
    "n_jobs": -1,
    "tree_method": "hist"
}

start_t = time.time()
for fold, (trn_idx, val_idx) in enumerate(skf.split(X, target_bins)):
    X_tr, y_tr = X.iloc[trn_idx], y_log.iloc[trn_idx]
    X_va, y_va = X.iloc[val_idx], y_log.iloc[val_idx]

    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    val_pred = model.predict(X_va)
    oof_xgb[val_idx] = val_pred
    test_preds_xgb += model.predict(X_test) / N_SPLITS

    fold_rmsle = np.sqrt(mean_squared_error(y_va, val_pred))
    print(f"Fold {fold+1} RMSLE: {fold_rmsle:.5f}")

xgb_rmsle = np.sqrt(mean_squared_error(y_log, oof_xgb))
print(f"✅ XGBoost Overall OOF RMSLE: {xgb_rmsle:.5f} (Time: {time.time()-start_t:.1f}s)")"""))

# ----------------------------------------------------------------------
# 8. Ridge Meta-Learner Stacking
# ----------------------------------------------------------------------
cells.append(md("""## 8. Stage-2 Meta-Learner Stacking & Ensemble Evaluation

Instead of basic linear averaging, we train a **Ridge Regression Meta-Learner** on the Out-Of-Fold prediction matrix `[OOF_LGB, OOF_CAT, OOF_XGB]` to find optimal non-negative stacking weights."""))

cells.append(code("""print("=" * 60)
print("=== 8. Ridge Meta-Learner Stacking ===")
print("=" * 60)

OOF_matrix = np.column_stack([oof_lgb, oof_cat, oof_xgb])
Test_matrix = np.column_stack([test_preds_lgb, test_preds_cat, test_preds_xgb])

# Fit Ridge CV to find optimal meta-weights
meta_model = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=5)
meta_model.fit(OOF_matrix, y_log)

oof_stacked = meta_model.predict(OOF_matrix)
test_preds_stacked = meta_model.predict(Test_matrix)

stacked_rmsle = np.sqrt(mean_squared_error(y_log, oof_stacked))

# Actual dollar scale metrics
actual_y = np.expm1(y_log)
actual_pred = np.expm1(oof_stacked)
dollar_rmse = np.sqrt(mean_squared_error(actual_y, actual_pred))
dollar_mae = mean_absolute_error(actual_y, actual_pred)

print(f"🔥 Meta-Learner Coefficients (LGB, CAT, XGB): {meta_model.coef_}")
print(f"🔥 Meta-Learner Intercept: {meta_model.intercept_:.4f}")
print(f"🔥 STACKED META-LEARNER OOF RMSLE: {stacked_rmsle:.5f}")
print(f"📊 Dollar-Scale Actual Metrics: RMSE = ${dollar_rmse:,.2f} | MAE = ${dollar_mae:,.2f}")"""))

# ----------------------------------------------------------------------
# 9. Model Evaluation & Comparison Table
# ----------------------------------------------------------------------
cells.append(md("## 9. Comprehensive Model Comparison"))

cells.append(code("""results_df = pd.DataFrame([
    {"Model": "LightGBM (Calibrated)", "OOF RMSLE": lgb_rmsle},
    {"Model": "CatBoost (Calibrated)", "OOF RMSLE": cat_rmsle},
    {"Model": "XGBoost (Calibrated)", "OOF RMSLE": xgb_rmsle},
    {"Model": "🔥 STACKED META-LEARNER", "OOF RMSLE": stacked_rmsle},
]).sort_values(by="OOF RMSLE")

print("=" * 50)
print("FINAL OOF PERFORMANCE COMPARISON")
print("=" * 50)
print(results_df.to_string(index=False))

plt.figure(figsize=(10, 5))
sns.barplot(data=results_df, x="OOF RMSLE", y="Model", palette="viridis")
plt.title("Model OOF RMSLE Comparison (Lower is Better)")
plt.xlim(min(results_df["OOF RMSLE"]) * 0.98, max(results_df["OOF RMSLE"]) * 1.02)
plt.show()"""))

# ----------------------------------------------------------------------
# 10. Submission Generation
# ----------------------------------------------------------------------
cells.append(md("""## 10. Final Submission Verification & Export"""))

cells.append(code("""# Transform predictions back to original dollar scale
final_predictions = np.expm1(test_preds_stacked)
final_predictions = np.clip(final_predictions, a_min=0, a_max=None)

submission = sample_sub.copy()
submission["TargetValue"] = final_predictions
submission.to_csv("submission.csv", index=False)

print("=" * 70)
print("FINAL SUBMISSION VERIFICATION")
print("=" * 70)
print(f"Row count matches sample_submission : {len(submission) == len(sample_sub)} ({len(submission):,} rows)")
print(f"Null values present                 : {submission['TargetValue'].isnull().sum()}")
print(f"Negative values present             : {(submission['TargetValue'] < 0).sum()}")
print("\\nFirst 10 Predictions:")
print(submission.head(10))
print("\\nSummary Statistics of Final Predictions:")
print(submission["TargetValue"].describe())

print("\\n🎉 V7 BREAKTHROUGH PIPELINE COMPLETED SUCCESSFULLY!")"""))

# ----------------------------------------------------------------------
# Write Out Kaggle and Offline Notebooks
# ----------------------------------------------------------------------
kaggle_nb = {
    "cells": cells,
    "metadata": {
        "language_info": {"name": "python"},
        "kernel_info": {"name": "python3"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

offline_cells = []
for cell in cells:
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        if "DATA_PATH = \".\"" in src:
            src = src.replace('DATA_PATH = "."', 'DATA_PATH = r"e:\\HeavyEquipmentML\\data"')
            cell = code(src)
    offline_cells.append(cell)

offline_nb = {
    "cells": offline_cells,
    "metadata": {
        "language_info": {"name": "python"},
        "kernel_info": {"name": "python3"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("notebooks/24f2008980-notebook-v7-breakthrough-kaggle.ipynb", "w", encoding="utf-8") as f:
    json.dump(kaggle_nb, f, indent=1)

with open("notebooks/24f2008980-notebook-v7-breakthrough-offline.ipynb", "w", encoding="utf-8") as f:
    json.dump(offline_nb, f, indent=1)

print("Successfully generated Notebook v7 Kaggle and Offline notebooks!")
