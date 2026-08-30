"""
Generator script to produce a high-performing Kaggle notebook that satisfies all 9 Level 1 Viva requirements.
Focuses on advanced domain feature engineering, Target Encoding, categorical handling, and robust tree ensembles.
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
cells.append(md("""# 🏗️ Heavy Equipment Selling Price Prediction (Advanced Pipeline)

## Problem Statement

The objective of this project is to **predict the selling price of heavy equipment at auction**. The dataset contains attributes describing the equipment configuration, condition, usage history, and transaction details.

This is a **regression problem** where the target variable (`TargetValue`) represents the actual sale price. The evaluation metric is **RMSE (Root Mean Squared Error)** on log-transformed prices.

### Approach & Pipeline Architecture
1. **Exploratory Data Analysis (EDA)** — Understand key statistics, distributions, missingness, and relationships.
2. **Systematic Preprocessing & Domain Feature Engineering** — Extract domain signals (machine age, operational intensity, capacity specs, series tokens, and multi-tier out-of-fold target encodings).
3. **Reusable Pipeline Transformation** — Clean preprocessing and column handling applied consistently to avoid leakage.
4. **Feature Importance Analysis** — Analyze model feature contributions to validate engineered signals.
5. **Multi-Family Model Exploration & CV Evaluation** — Train and evaluate Random Forest, XGBoost, LightGBM, and CatBoost models.
6. **Hyperparameter Tuning & Ensembling** — Tune model hyperparameters and combine top predictions for optimal RMSE performance.
7. **Final Predictions & Submission** — Validate predictions and generate output."""))

# ----------------------------------------------------------------------
# 1. Imports
# ----------------------------------------------------------------------
cells.append(md("## 1. Imports & Setup"))

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
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
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

print("✅ All allowed libraries imported successfully.")"""))

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

Understanding feature distributions, target skewness, missing values, and bivariate correlations.

### 3.1 Univariate Analysis
Examining target and numerical distributions."""))

cells.append(code("""print("=" * 70)
print("DATASET SUMMARY")
print("=" * 70)
train_raw.info(max_cols=10)

print("\\nTarget Value Statistics:")
print(train_raw["TargetValue"].describe())"""))

cells.append(code("""# Missing Values Analysis
missing_df = pd.DataFrame({
    "Missing Count": train_raw.isnull().sum(),
    "Missing %": (train_raw.isnull().sum() / len(train_raw) * 100).round(2)
}).sort_values("Missing %", ascending=False)

missing_with_values = missing_df[missing_df["Missing Count"] > 0]
print(f"Columns with missing values: {len(missing_with_values)} / {len(train_raw.columns)}")
print(missing_with_values.head(15))"""))

cells.append(code("""# Target Distribution Analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

raw_skew = train_raw["TargetValue"].skew()
sns.histplot(train_raw["TargetValue"], kde=True, ax=axes[0], color="steelblue", bins=50)
axes[0].set_title(f"Original Target Distribution\\nSkewness: {raw_skew:.4f}")
axes[0].set_xlabel("TargetValue")

log_target = np.log1p(train_raw["TargetValue"])
log_skew = log_target.skew()
sns.histplot(log_target, kde=True, ax=axes[1], color="coral", bins=50)
axes[1].set_title(f"Log1p Target Distribution\\nSkewness: {log_skew:.4f}")
axes[1].set_xlabel("log1p(TargetValue)")

plt.tight_layout()
plt.show()

print(f"📌 Log transformation effectively normalizes the target variable (skew reduced from {raw_skew:.4f} to {log_skew:.4f}).")"""))

cells.append(md("### 3.2 Bivariate Analysis"))

cells.append(code("""# Numeric Correlations
numeric_cols = train_raw.select_dtypes(include=[np.number]).columns.tolist()
corr = train_raw[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Numeric Feature Correlation Matrix")
plt.show()"""))

# ----------------------------------------------------------------------
# 4. Systematic Preprocessing & Advanced Feature Engineering
# ----------------------------------------------------------------------
cells.append(md("""## 4. Systematic Preprocessing & Feature Engineering

We perform comprehensive domain-specific feature extraction and multi-tier out-of-fold target encoding to maximize signal for gradient boosting models."""))

cells.append(code("""def extract_domain_features(train_df, test_df):
    \"\"\"
    Extract domain features for heavy machinery valuation.
    Applied consistently to train and test sets.
    \"\"\"
    print("Executing Domain Feature Engineering...")
    tr = train_df.copy()
    te = test_df.copy()

    y_train = np.log1p(tr["TargetValue"].values)

    tr["_is_test"] = 0
    te["_is_test"] = 1
    te["TargetValue"] = np.nan

    df = pd.concat([tr, te], ignore_index=True)

    # 1. Sentinel & Missing Cleanup
    df["ManufactureYear_missing"] = (df["ManufactureYear"] == 1001).astype(int)
    df.loc[df["ManufactureYear"] == 1001, "ManufactureYear"] = np.nan

    df["Hours_missing"] = (df["OperationalHoursMeter"].isna() | (df["OperationalHoursMeter"] == 0)).astype(int)
    df.loc[df["OperationalHoursMeter"] == 0, "OperationalHoursMeter"] = np.nan

    # 2. Date Components
    dt = pd.to_datetime(df["TransactionDate"])
    df["SaleYear"] = dt.dt.year
    df["SaleMonth"] = dt.dt.month
    df["SaleQuarter"] = dt.dt.quarter
    df["SaleWeek"] = dt.dt.isocalendar().week.astype(int)
    df["SaleDayOfWeek"] = dt.dt.dayofweek
    df["SaleDayOfYear"] = dt.dt.dayofyear
    df["DaysSince2000"] = (dt - pd.to_datetime("2000-01-01")).dt.days

    # 3. Machine Age & Depreciation
    df["MachineAge"] = df["SaleYear"] - df["ManufactureYear"]
    df.loc[df["MachineAge"] < 0, "MachineAge"] = np.nan
    df["MachineAge_Sq"] = df["MachineAge"] ** 2
    df["MachineAge_Log"] = np.log1p(df["MachineAge"].clip(lower=0))
    df["IsNewMachine"] = (df["MachineAge"] <= 2).astype(float)
    df["IsOldMachine"] = (df["MachineAge"] >= 20).astype(float)

    df["AgeBin"] = pd.cut(
        df["MachineAge"],
        bins=[-1, 2, 5, 9, 14, 20, 30, 100],
        labels=[0, 1, 2, 3, 4, 5, 6],
    ).astype(float)

    # 4. Hours & Operational Intensity
    df["Hours_Log"] = np.log1p(df["OperationalHoursMeter"].clip(lower=0))
    df["Hours_Per_Year"] = df["OperationalHoursMeter"] / (df["MachineAge"].clip(lower=0) + 1.0)
    df["Hours_Per_Year_Log"] = np.log1p(df["Hours_Per_Year"].clip(lower=0))
    df["Age_x_Hours"] = df["MachineAge"] * df["OperationalHoursMeter"]
    df["HighUsageFlag"] = (df["Hours_Per_Year"] > 1500).astype(float)
    df["LowUsageFlag"] = (df["Hours_Per_Year"] < 200).astype(float)

    # 5. Domain Regex Extraction
    spec_str = df["Spec_FullDescriptor"].fillna("").astype(str)
    df["ModelToken"] = spec_str.str.extract(r"^([A-Z0-9]+)")[0].fillna("_Missing_")
    df["ModelSeriesPrefix"] = spec_str.str.extract(r"^([A-Za-z]+)")[0].fillna("_Missing_")
    df["ModelSeriesNumber"] = spec_str.str.extract(r"(\\d+)")[0].astype(float)

    func_str = df["FunctionalClassification"].fillna("").astype(str)
    df["EquipmentCategory"] = func_str.str.extract(r"^([A-Za-z\\s,]+)")[0].str.strip().fillna("_Missing_")

    df["Capacity_Tons_Min"] = func_str.str.extract(r"(\\d+\\.?\\d*)\\s*to\\s*\\d+\\.?\\d*\\s*Metric Tons")[0].astype(float)
    df["Capacity_Tons_Max"] = func_str.str.extract(r"\\d+\\.?\\d*\\s*to\\s*(\\d+\\.?\\d*)\\s*Metric Tons")[0].astype(float)
    df["Capacity_Tons_Mid"] = (df["Capacity_Tons_Min"] + df["Capacity_Tons_Max"]) / 2.0

    df["Power_HP_Min"] = func_str.str.extract(r"(\\d+\\.?\\d*)\\s*to\\s*\\d+\\.?\\d*\\s*Horsepower")[0].astype(float)
    df["Power_HP_Max"] = func_str.str.extract(r"\\d+\\.?\\d*\\s*to\\s*(\\d+\\.?\\d*)\\s*Horsepower")[0].astype(float)
    df["Power_HP_Mid"] = (df["Power_HP_Min"] + df["Power_HP_Max"]) / 2.0

    df["Depth_Ft_Min"] = func_str.str.extract(r"(\\d+\\.?\\d*)\\s*to\\s*\\d+\\.?\\d*\\s*Ft")[0].astype(float)
    df["Depth_Ft_Max"] = func_str.str.extract(r"\\d+\\.?\\d*\\s*to\\s*(\\d+\\.?\\d*)\\s*Ft")[0].astype(float)
    df["Depth_Ft_Mid"] = (df["Depth_Ft_Min"] + df["Depth_Ft_Max"]) / 2.0

    # 6. Frequency Encodings
    freq_cols = [
        "ProductConfigID", "Spec_FullDescriptor", "Spec_BaseClass",
        "VendorPartnerID", "RegionCode", "EquipmentCategory",
        "ModelToken", "DataOriginCode", "CabinType"
    ]
    for col in freq_cols:
        if col in df.columns:
            freq_map = df[col].value_counts(normalize=True).to_dict()
            df[f"{col}_Freq"] = df[col].map(freq_map).fillna(0.0)

    # Clean unused columns
    drop_unneeded = [
        "TransactionID", "TargetValue", "TransactionDate",
        "InventoryGroupDescription", "col18", "col19", "_is_test"
    ]
    df.drop(columns=[c for c in drop_unneeded if c in df.columns], inplace=True)

    X_train_df = df.iloc[:len(train_df)].copy().reset_index(drop=True)
    X_test_df = df.iloc[len(train_df):].copy().reset_index(drop=True)

    return X_train_df, y_train, X_test_df


def add_out_of_fold_target_encodings(X_train, y_train, X_test, n_splits=5, seed=42):
    \"\"\"
    Compute Bayesian smoothed target encodings out-of-fold to prevent leakage.
    \"\"\"
    print("Computing Multi-Tier Bayesian Target Encodings...")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    global_mean = y_train.mean()
    global_std = y_train.std()

    specs = [
        (["ProductConfigID"], 20),
        (["Spec_FullDescriptor"], 20),
        (["ModelToken"], 15),
        (["Spec_BaseClass"], 10),
        (["EquipmentCategory"], 10),
        (["RegionCode"], 10),
        (["VendorPartnerID"], 10),
        (["ProductConfigID", "AgeBin"], 30),
        (["Spec_FullDescriptor", "AgeBin"], 30),
        (["ProductConfigID", "RegionCode"], 30),
    ]

    X_tr = X_train.copy()
    X_te = X_test.copy()

    for cols, smooth in specs:
        if len(cols) == 1:
            key_tr = X_tr[cols[0]].astype(str)
            key_te = X_te[cols[0]].astype(str)
            prefix = cols[0]
        else:
            prefix = "_x_".join(cols)
            key_tr = X_tr[cols[0]].astype(str)
            key_te = X_te[cols[0]].astype(str)
            for c in cols[1:]:
                key_tr = key_tr + "___" + X_tr[c].astype(str)
                key_te = key_te + "___" + X_te[c].astype(str)

        mean_col = f"TE_{prefix}_Mean"
        std_col = f"TE_{prefix}_Std"
        count_col = f"TE_{prefix}_Count"

        X_tr[mean_col] = global_mean
        X_tr[std_col] = global_std
        X_tr[count_col] = 0.0

        for tr_idx, va_idx in kf.split(X_tr):
            f_key = key_tr.iloc[tr_idx]
            f_y = y_train[tr_idx]

            stats = pd.DataFrame({"k": f_key, "y": f_y}).groupby("k")["y"].agg(["count", "mean", "std"])
            smoothed = (stats["count"] * stats["mean"] + smooth * global_mean) / (stats["count"] + smooth)

            v_key = key_tr.iloc[va_idx]
            X_tr.iloc[va_idx, X_tr.columns.get_loc(mean_col)] = v_key.map(smoothed).fillna(global_mean).values
            X_tr.iloc[va_idx, X_tr.columns.get_loc(std_col)] = v_key.map(stats["std"]).fillna(global_std).values
            X_tr.iloc[va_idx, X_tr.columns.get_loc(count_col)] = v_key.map(stats["count"]).fillna(0.0).values

        full_stats = pd.DataFrame({"k": key_tr, "y": y_train}).groupby("k")["y"].agg(["count", "mean", "std"])
        full_smoothed = (full_stats["count"] * full_stats["mean"] + smooth * global_mean) / (full_stats["count"] + smooth)

        X_te[mean_col] = key_te.map(full_smoothed).fillna(global_mean).values
        X_te[std_col] = key_te.map(full_stats["std"]).fillna(global_std).values
        X_te[count_col] = key_te.map(full_stats["count"]).fillna(0.0).values

    return X_tr, X_te


X_tr_raw, y, X_te_raw = extract_domain_features(train_raw, test_raw)
X_train_df, X_test_df = add_out_of_fold_target_encodings(X_tr_raw, y, X_te_raw)

print(f"Features created: {X_train_df.shape[1]}")"""))

cells.append(md("""### Reusable Sklearn Preprocessing Pipeline"""))

cells.append(code("""# Identify numeric and categorical columns
num_cols = [c for c in X_train_df.columns if pd.api.types.is_numeric_dtype(X_train_df[c])]
cat_cols = [c for c in X_train_df.columns if c not in num_cols]

for col in cat_cols:
    X_train_df[col] = X_train_df[col].fillna("_Missing_").astype(str)
    X_test_df[col] = X_test_df[col].fillna("_Missing_").astype(str)

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline([
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ]
)

X_train_prep = preprocessor.fit_transform(X_train_df)
X_test_prep = preprocessor.transform(X_test_df)

feature_names = num_cols + cat_cols
X = pd.DataFrame(X_train_prep, columns=feature_names)
X_test = pd.DataFrame(X_test_prep, columns=feature_names)

print("✅ Reusable pipeline fit and transform completed.")
print(f"Processed Train Shape: {X.shape}, Processed Test Shape: {X_test.shape}")"""))

# ----------------------------------------------------------------------
# 5. Feature Importance
# ----------------------------------------------------------------------
cells.append(md("## 5. Feature Importance Analysis"))

cells.append(code("""rf_imp = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
rf_imp.fit(X, y)

imp_series = pd.Series(rf_imp.feature_importances_, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(10, 8))
imp_series.head(20).plot(kind="barh", color="teal").invert_yaxis()
plt.title("Top 20 Feature Importances (Random Forest)")
plt.xlabel("Importance Score")
plt.show()

print("Top 10 Features:")
print(imp_series.head(10))"""))

# ----------------------------------------------------------------------
# 6. Model Building & Comparison
# ----------------------------------------------------------------------
cells.append(md("""## 6. Model Building & Comparison

We evaluate 4 distinct models across 5 folds:
- **Random Forest** (Bagging)
- **XGBoost** (Gradient Boosting)
- **LightGBM** (Leaf-wise Gradient Boosting)
- **CatBoost** (Ordered Boosting)"""))

cells.append(code("""def evaluate_model_cv(model_fn, model_name, X_data, y_data, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(X_data))
    test_preds = np.zeros(len(X_test))

    rmse_scores, mae_scores, r2_scores = [], [], []
    start_time = time.time()

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_data), 1):
        X_tr, X_va = X_data.iloc[train_idx], X_data.iloc[val_idx]
        y_tr, y_va = y_data[train_idx], y_data[val_idx]

        model = model_fn()
        model.fit(X_tr, y_tr)

        val_pred = model.predict(X_va)
        oof[val_idx] = val_pred

        rmse = np.sqrt(mean_squared_error(y_va, val_pred))
        mae = mean_absolute_error(y_va, val_pred)
        r2 = r2_score(y_va, val_pred)

        rmse_scores.append(rmse)
        mae_scores.append(mae)
        r2_scores.append(r2)

        test_preds += model.predict(X_test) / n_splits
        print(f"  Fold {fold} - RMSE: {rmse:.5f} | MAE: {mae:.5f} | R2: {r2:.5f}")

    total_time = time.time() - start_time
    mean_rmse = np.mean(rmse_scores)
    print(f"✅ {model_name} Overall OOF RMSE: {mean_rmse:.5f} (Time: {total_time:.1f}s)")

    return {
        "name": model_name,
        "mean_rmse": mean_rmse,
        "mean_mae": np.mean(mae_scores),
        "mean_r2": np.mean(r2_scores),
        "oof": oof,
        "test_preds": test_preds,
        "time": total_time,
    }

results = {}"""))

cells.append(code("""print("=== 1. Random Forest ===")
rf_fn = lambda: RandomForestRegressor(n_estimators=150, max_depth=14, min_samples_leaf=4, random_state=42, n_jobs=-1)
results["RandomForest"] = evaluate_model_cv(rf_fn, "RandomForest", X, y)"""))

cells.append(code("""print("=== 2. XGBoost ===")
xgb_fn = lambda: xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=5,
    random_state=42,
    n_jobs=-1
)
results["XGBoost"] = evaluate_model_cv(xgb_fn, "XGBoost", X, y)"""))

cells.append(code("""print("=== 3. LightGBM ===")
lgb_fn = lambda: lgb.LGBMRegressor(
    n_estimators=1200,
    learning_rate=0.03,
    num_leaves=127,
    max_depth=10,
    min_child_samples=15,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=1,
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)
results["LightGBM"] = evaluate_model_cv(lgb_fn, "LightGBM", X, y)"""))

cells.append(code("""print("=== 4. CatBoost ===")
cb_fn = lambda: CatBoostRegressor(
    iterations=1200,
    learning_rate=0.04,
    depth=8,
    random_seed=42,
    verbose=0,
    thread_count=-1
)
results["CatBoost"] = evaluate_model_cv(cb_fn, "CatBoost", X, y)"""))

# ----------------------------------------------------------------------
# 7. Model Comparison & Ensembling
# ----------------------------------------------------------------------
cells.append(md("""## 7. Model Comparison & Ensemble Evaluation"""))

cells.append(code("""comp_df = pd.DataFrame([
    {
        "Model": res["name"],
        "OOF RMSE": f"{res['mean_rmse']:.5f}",
        "OOF MAE": f"{res['mean_mae']:.5f}",
        "OOF R2": f"{res['mean_r2']:.5f}",
        "Time (s)": f"{res['time']:.1f}",
    }
    for res in results.values()
])

print("=" * 70)
print("MODEL EVALUATION METRICS COMPARISON")
print("=" * 70)
print(comp_df.to_string(index=False))"""))

cells.append(code("""# Ensemble predictions (Blend of top boosting models)
blend_oof = (
    0.40 * results["LightGBM"]["oof"] +
    0.35 * results["CatBoost"]["oof"] +
    0.25 * results["XGBoost"]["oof"]
)

blend_rmse = np.sqrt(mean_squared_error(y, blend_oof))
print(f"\\n🔥 Blended Ensemble (LightGBM + CatBoost + XGBoost) OOF RMSE: {blend_rmse:.5f}")"""))

# ----------------------------------------------------------------------
# 8. Hyperparameter Tuning
# ----------------------------------------------------------------------
cells.append(md("## 8. Hyperparameter Optimization"))

cells.append(code("""print("Optimizing LightGBM via RandomizedSearchCV...")

base_lgb = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbosity=-1)
param_grid = {
    "n_estimators": [800, 1200],
    "learning_rate": [0.02, 0.04],
    "num_leaves": [127, 255],
    "max_depth": [8, 10, 12],
    "min_child_samples": [10, 20],
    "feature_fraction": [0.65, 0.8],
    "bagging_fraction": [0.75, 0.9],
}

search = RandomizedSearchCV(
    estimator=base_lgb,
    param_distributions=param_grid,
    n_iter=6,
    scoring="neg_root_mean_squared_error",
    cv=KFold(n_splits=3, shuffle=True, random_state=42),
    random_state=42,
    n_jobs=-1,
    verbose=1
)

search.fit(X, y)

print(f"Best Parameters: {search.best_params_}")
print(f"Best Tuned Score: {-search.best_score_:.5f}")"""))

# ----------------------------------------------------------------------
# 9. Final Submission / Results
# ----------------------------------------------------------------------
cells.append(md("## 9. Final Predictions & Result Verification"))

cells.append(code("""# Create blended test prediction
final_blend_test = (
    0.40 * results["LightGBM"]["test_preds"] +
    0.35 * results["CatBoost"]["test_preds"] +
    0.25 * results["XGBoost"]["test_preds"]
)

final_preds_original_scale = np.maximum(0, np.expm1(final_blend_test))

submission = sample_sub.copy()
target_col = [c for c in sample_sub.columns if c != "TransactionID"][0]
submission[target_col] = final_preds_original_scale

# Save submission file
submission.to_csv("submission.csv", index=False)
print("✅ submission.csv created successfully.")

print("\\nPrediction Summary:")
print(submission[target_col].describe())

# Display sample output
print("\\nFirst 10 Predictions:")
print(submission.head(10))"""))

# ----------------------------------------------------------------------
# Assemble & Write Notebooks
# ----------------------------------------------------------------------
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

# 1. Write standard notebook (v2)
with open(r"notebooks/24f2008980-notebook-v2-kaggle.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

# 2. Write offline notebook (v2)
offline_nb = json.loads(json.dumps(notebook))

# Update offline cells
for c in offline_nb["cells"]:
    if c["cell_type"] == "code":
        src = "".join(c["source"])
        if "submission.to_csv(\"submission.csv\", index=False)" in src:
            c["source"] = [line + "\n" for line in src.replace(
                "submission.to_csv(\"submission.csv\", index=False)\nprint(\"✅ submission.csv created successfully.\")",
                "print(\"=== OFFLINE MODE: Prediction Results (Not saving submission.csv) ===\")"
            ).split("\n")]

with open(r"notebooks/24f2008980-notebook-v2-offline.ipynb", "w", encoding="utf-8") as f:
    json.dump(offline_nb, f, indent=1, ensure_ascii=False)

print("Both notebooks generated successfully.")
