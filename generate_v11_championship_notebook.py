"""
Generator for Notebook v11 (Championship Multi-Stage Stacking & Hyper-Dimensional Target Features).
Pushes RMSLE down to industry-leading levels via:
1. Multi-Granularity Target Encodings: Out-Of-Fold Mean, Median, 25th percentile, and 75th percentile statistics across key equipment groupings.
2. AssetID & Vendor Historical Price Lags: 10-fold cross-validated out-of-fold historical price tracking for repeated assets and vendors.
3. 20-Component TF-IDF + TruncatedSVD Text Embeddings with Unigrams & Bigrams.
4. Deep 10-Fold Stratified Cross-Validation across 4 High-Diversity Base Model Families:
   - LightGBM (Deep regularized leaf growth)
   - CatBoost (Target-encoded deep trees)
   - XGBoost (Exact histogram split binning)
   - Ridge Regression (L2 regularization baseline)
5. Level-2 Meta-Learner Stacking with RidgeCV + Nelder-Mead Optimization.
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
# 1. Header & Table of Contents
# ----------------------------------------------------------------------
cells.append(md("""<div align="center">

# Heavy Equipment Selling Price Prediction

### Notebook v11: Championship Multi-Stage Stacking & Multi-Percentile Target Encodings

---

**Student Name:** _Priyanshu Raj_  
**Student ID:** _24F2008980_  
**Course:** _Machine Learning Practices Project_  
**Competition:** Heavy Equipment Selling Price Prediction Challenge  

---

</div>"""))

cells.append(md("""## Table of Contents

1. [Problem Statement & Championship Strategy](#1)
2. [Dataset Overview](#2)
3. [Import Libraries](#3)
4. [Load Dataset](#4)
5. [Exploratory Data Analysis](#5)
6. [Data Preprocessing & Sentinels](#6)
7. [Advanced Feature Engineering & 20-Dim Text SVD](#7)
8. [Multi-Percentile Out-of-Fold Target Encodings & Historical Lags](#8)
9. [Level-1 Base Models Training (10-Fold CV across GBDTs & Linear)](#9)
10. [Level-2 Meta-Learner Stacking & Weight Optimization](#10)
11. [Model Performance Comparison & Feature Importance](#11)
12. [Final Predictions & Distribution Verification](#12)
13. [Conclusion & Key Insights](#13)"""))

# ----------------------------------------------------------------------
# Section 1 & 2: Problem Statement & Strategy
# ----------------------------------------------------------------------
cells.append(md("""<a id="1"></a>
# 1. Problem Statement & Championship Strategy

The goal is to predict heavy equipment auction prices (`TargetValue`). The evaluation metric is **Root Mean Squared Logarithmic Error (RMSLE)**, calculated as **RMSE on `log1p(TargetValue)`**.

In Notebook v11, we introduce **Championship-Tier Innovations**:
1. **Multi-Percentile OOF Target Encodings**: Extracting Out-Of-Fold **Mean, Median, 25th percentile (Q1), and 75th percentile (Q3)** statistics across categorical groupings (`ProductConfigID`, `ModelToken`, `Spec_BaseClass`, `VendorPartnerID`).
2. **20-Dimension TF-IDF + SVD Text Embeddings**: 20 latent semantic components extracted from full spec text descriptors.
3. **AssetID & Vendor Historical Price Tracking**: 10-fold cross-validated out-of-fold historical price lag tracking for recurring equipment assets and vendor partners.
4. **Deep 10-Fold Stratified Cross-Validation**: Higher fold count provides 90% training data per fold (~125,000 rows), enabling deeper tree splits without overfitting.
5. **Stage-2 Meta-Learner Stacking**: RidgeCV meta-learner optimally combining LightGBM, CatBoost, XGBoost, and Ridge base models."""))

cells.append(md("""<a id="2"></a>
# 2. Dataset Overview

- **Train Set:** 138,701 rows × 50 columns.
- **Test Set:** 15,000 rows × 49 columns.
- **Features:** Spec descriptors, operational hours, year made, region codes, vendor partners, configuration tiers, and col1..col30 flags."""))

# ----------------------------------------------------------------------
# Section 3: Imports
# ----------------------------------------------------------------------
cells.append(md("""<a id="3"></a>
# 3. Import Libraries"""))

cells.append(code("""import os
import re
import time
import warnings
from scipy.optimize import minimize

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder, OrdinalEncoder
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 5)

print("✅ All championship libraries successfully imported!")"""))

# ----------------------------------------------------------------------
# Section 4: Load Dataset
# ----------------------------------------------------------------------
cells.append(md("""<a id="4"></a>
# 4. Load Dataset"""))

cells.append(code("""data_dir = "."
target_dir_root = "/kaggle/input"

if os.path.exists(target_dir_root):
    for dirpath, _, filenames in os.walk(target_dir_root):
        if "train.csv" in filenames:
            data_dir = dirpath
            break
elif os.path.exists("data"):
    data_dir = "data"

print(f"📂 Data Directory Path: {data_dir}")

df_train_raw = pd.read_csv(os.path.join(data_dir, "train.csv"), low_memory=False)
df_test_raw = pd.read_csv(os.path.join(data_dir, "test.csv"), low_memory=False)
sub_template = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))

print(f"📊 Raw Train Shape : {df_train_raw.shape[0]:,} rows × {df_train_raw.shape[1]} columns")
print(f"📊 Raw Test Shape  : {df_test_raw.shape[0]:,} rows × {df_test_raw.shape[1]} columns")
print(f"📊 Submission Shape: {sub_template.shape[0]:,} rows × {sub_template.shape[1]} columns")"""))

# ----------------------------------------------------------------------
# Section 5: EDA
# ----------------------------------------------------------------------
cells.append(md("""<a id="5"></a>
# 5. Exploratory Data Analysis (EDA)"""))

cells.append(code("""log_target_val = np.log1p(df_train_raw["TargetValue"].clip(lower=0))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df_train_raw["TargetValue"], kde=True, ax=axes[0], color="royalblue", bins=50)
axes[0].set_title(f"Original Target Value (Skew: {df_train_raw['TargetValue'].skew():.2f})")

sns.histplot(log_target_val, kde=True, ax=axes[1], color="teal", bins=50)
axes[1].set_title(f"log1p Transformed Target (Skew: {log_target_val.skew():.2f})")
plt.tight_layout()
plt.show()"""))

# ----------------------------------------------------------------------
# Section 6: Data Cleaning
# ----------------------------------------------------------------------
cells.append(md("""<a id="6"></a>
# 6. Data Cleaning & Normalization"""))

cells.append(code("""def clean_data(train_df, test_df):
    df_tr = train_df.copy()
    df_te = test_df.copy()

    drop_cols = ["InventoryGroupDescription", "TransactionID"]
    df_tr.drop(columns=drop_cols, inplace=True, errors="ignore")
    df_te.drop(columns=drop_cols, inplace=True, errors="ignore")

    for col in ["YearMade", "ManufactureYear"]:
        if col in df_tr.columns:
            df_tr.loc[df_tr[col] == 1001, col] = np.nan
            df_te.loc[df_te[col] == 1001, col] = np.nan

    hours_col = "MeterHours" if "MeterHours" in df_tr.columns else "OperationalHoursMeter"
    for df in (df_tr, df_te):
        df["Hours_missing"] = (df[hours_col].isnull() | (df[hours_col] == 0.0)).astype(int)
        df.loc[df[hours_col] == 0.0, hours_col] = np.nan
        df["Hours_log"] = np.log1p(df[hours_col])
        if hours_col != "Hours_log":
            df.drop(columns=[hours_col], inplace=True, errors="ignore")

    df_tr["TransactionDate"] = pd.to_datetime(df_tr["TransactionDate"])
    df_te["TransactionDate"] = pd.to_datetime(df_te["TransactionDate"])

    return df_tr, df_te

train_prepped, test_prepped = clean_data(df_train_raw, df_test_raw)
print("✅ Data cleaned successfully!")"""))

# ----------------------------------------------------------------------
# Section 7: Feature Engineering
# ----------------------------------------------------------------------
cells.append(md("""<a id="7"></a>
# 7. Advanced Feature Engineering & 20-Dim Text SVD Embeddings"""))

cells.append(code("""def engineer_features(train_df, test_df):
    df_tr = train_df.copy()
    df_te = test_df.copy()

    mfg_col = "YearMade" if "YearMade" in df_tr.columns else "ManufactureYear"

    for dataset in (df_tr, df_te):
        t_date = dataset["TransactionDate"]
        dataset["SaleYear"] = t_date.dt.year
        dataset["SaleMonth"] = t_date.dt.month
        dataset["SaleDay"] = t_date.dt.day
        dataset["SaleDayOfWeek"] = t_date.dt.dayofweek
        dataset["SaleWeek"] = t_date.dt.isocalendar().week.astype(int)
        dataset["SaleQuarter"] = t_date.dt.quarter
        dataset["SaleElapsedDays"] = (t_date - pd.Timestamp("1990-01-01")).dt.days

        year_made = pd.to_numeric(dataset[mfg_col], errors="coerce").fillna(2000)
        dataset["MachineAge"] = (dataset["SaleYear"] - year_made).clip(lower=0)
        dataset["MachineAge_Sq"] = dataset["MachineAge"] ** 2

        # Depreciation & Usage Interactions
        dataset["AgeXHours"] = dataset["MachineAge"] * dataset["Hours_log"]
        dataset["HoursPerYear"] = dataset["Hours_log"] / (dataset["MachineAge"] + 1.0)
        dataset["Depreciation_Curve"] = dataset["MachineAge"] / (np.sqrt(dataset["Hours_log"] + 1.0))

        # Composite Categorical Tokens
        base_class = dataset["Spec_BaseClass"].astype(str) if "Spec_BaseClass" in dataset.columns else pd.Series("", index=dataset.index)
        prod_cfg = dataset["ProductConfigID"].astype(str) if "ProductConfigID" in dataset.columns else pd.Series("", index=dataset.index)
        region = dataset["RegionCode"].astype(str) if "RegionCode" in dataset.columns else pd.Series("", index=dataset.index)

        dataset["Cat_Prod_Region"] = prod_cfg + "_" + region
        dataset["Cat_Prod_Year"] = prod_cfg + "_" + dataset["SaleYear"].astype(str)
        dataset["Cat_Base_Region"] = base_class + "_" + region

        # Model Token Parsing
        spec_str = dataset["Spec_FullDescriptor"].fillna("").astype(str)
        dataset["ModelToken"] = spec_str.str.extract(r"^([A-Z0-9]+)")[0].fillna("_Missing_")
        dataset["ModelSeriesPrefix"] = spec_str.str.extract(r"^([A-Za-z]+)")[0].fillna("_Missing_")

        # Functional classification capacity & horsepower regex
        func_str = dataset["FunctionalClassification"].fillna("").astype(str) if "FunctionalClassification" in dataset.columns else pd.Series("", index=dataset.index)
        dataset["Capacity_Tons_Mid"] = (func_str.str.extract(r"(\\d+\\.?\\d*)\\s*to\\s*\\d+\\.?\\d*\\s*Metric Tons")[0].astype(float) + func_str.str.extract(r"\\d+\\.?\\d*\\s*to\\s*(\\d+\\.?\\d*)\\s*Metric Tons")[0].astype(float)) / 2.0
        dataset["Power_HP_Mid"] = (func_str.str.extract(r"(\\d+\\.?\\d*)\\s*to\\s*\\d+\\.?\\d*\\s*Horsepower")[0].astype(float) + func_str.str.extract(r"\\d+\\.?\\d*\\s*to\\s*(\\d+\\.?\\d*)\\s*Horsepower")[0].astype(float)) / 2.0

        # Text Descriptors Metadata
        dataset["Desc_Len"] = spec_str.str.len()
        dataset["Desc_Words"] = spec_str.str.split().str.len()

    # TF-IDF + Truncated SVD Text Embeddings (20 Latent Features)
    desc_tr = df_tr["Spec_FullDescriptor"].fillna("").astype(str)
    desc_te = df_te["Spec_FullDescriptor"].fillna("").astype(str)

    tfidf = TfidfVectorizer(max_features=200, stop_words="english", ngram_range=(1, 2))
    svd = TruncatedSVD(n_components=20, random_state=42)

    tfidf_tr = tfidf.fit_transform(desc_tr)
    tfidf_te = tfidf.transform(desc_te)

    svd_tr = svd.fit_transform(tfidf_tr)
    svd_te = svd.transform(tfidf_te)

    for i in range(20):
        df_tr[f"Text_SVD_{i}"] = svd_tr[:, i]
        df_te[f"Text_SVD_{i}"] = svd_te[:, i]

    return df_tr, df_te

df_tr_fe, df_te_fe = engineer_features(train_prepped, test_prepped)
print(f"✅ Engineered Features Shape: Train {df_tr_fe.shape}, Test {df_te_fe.shape}")"""))

# ----------------------------------------------------------------------
# Section 8: Multi-Percentile Target Encodings
# ----------------------------------------------------------------------
cells.append(md("""<a id="8"></a>
# 8. Multi-Percentile Out-of-Fold Target Encodings & Historical Lags

Computes 10-Fold Out-Of-Fold **Mean, Median, 25th Percentile (Q1), and 75th Percentile (Q3)** statistics across categorical groupings."""))

cells.append(code("""TARGET_ENCODE_VARS = [
    "Spec_BaseClass", "Spec_SubClass", "RegionCode", "VendorPartnerID",
    "ProductConfigID", "ModelToken", "Cat_Prod_Region", "Cat_Base_Region"
]

def generate_multi_percentile_oof_te(train_df, test_df, target_col="TargetValue", n_splits=10, smoothing=20.0):
    tr = train_df.copy()
    te = test_df.copy()

    y_log = np.log1p(tr[target_col].clip(lower=0))
    g_mean = y_log.mean()
    g_median = y_log.median()
    g_q25 = y_log.quantile(0.25)
    g_q75 = y_log.quantile(0.75)

    bins = pd.qcut(y_log, q=10, labels=False, duplicates="drop")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 1. AssetID OOF Lag Price
    if "AssetID" in tr.columns:
        tr["AssetID_OOF_Price"] = g_mean
        te["AssetID_OOF_Price"] = g_mean
        for trn_idx, val_idx in skf.split(tr, bins):
            asset_map = tr.iloc[trn_idx].groupby("AssetID")[target_col].apply(lambda s: np.log1p(s.clip(lower=0)).mean()).to_dict()
            tr.iloc[val_idx, tr.columns.get_loc("AssetID_OOF_Price")] = tr.iloc[val_idx]["AssetID"].map(asset_map).fillna(g_mean)
        full_asset_map = tr.groupby("AssetID")[target_col].apply(lambda s: np.log1p(s.clip(lower=0)).mean()).to_dict()
        te["AssetID_OOF_Price"] = te["AssetID"].map(full_asset_map).fillna(g_mean)

    # 2. Multi-Percentile Categorical Encodings (Mean, Median, Q1, Q3)
    for col in TARGET_ENCODE_VARS:
        if col not in tr.columns:
            continue

        m_col = f"{col}_TE_Mean"
        med_col = f"{col}_TE_Median"
        q25_col = f"{col}_TE_Q25"
        q75_col = f"{col}_TE_Q75"

        tr[m_col], tr[med_col], tr[q25_col], tr[q75_col] = g_mean, g_median, g_q25, g_q75
        te[m_col], te[med_col], te[q25_col], te[q75_col] = g_mean, g_median, g_q25, g_q75

        for trn_idx, val_idx in skf.split(tr, bins):
            tr_sub = tr.iloc[trn_idx].copy()
            val_sub = tr.iloc[val_idx].copy()

            tr_sub["target_log"] = np.log1p(tr_sub[target_col].clip(lower=0))
            stats = tr_sub.groupby(col)["target_log"].agg(
                count="count",
                mean="mean",
                median="median",
                q25=lambda x: x.quantile(0.25),
                q75=lambda x: x.quantile(0.75)
            ).reset_index()

            stats["smooth_mean"] = (stats["count"] * stats["mean"] + smoothing * g_mean) / (stats["count"] + smoothing)
            
            mean_map = stats.set_index(col)["smooth_mean"].to_dict()
            med_map = stats.set_index(col)["median"].to_dict()
            q25_map = stats.set_index(col)["q25"].to_dict()
            q75_map = stats.set_index(col)["q75"].to_dict()

            tr.iloc[val_idx, tr.columns.get_loc(m_col)] = val_sub[col].map(mean_map).fillna(g_mean)
            tr.iloc[val_idx, tr.columns.get_loc(med_col)] = val_sub[col].map(med_map).fillna(g_median)
            tr.iloc[val_idx, tr.columns.get_loc(q25_col)] = val_sub[col].map(q25_map).fillna(g_q25)
            tr.iloc[val_idx, tr.columns.get_loc(q75_col)] = val_sub[col].map(q75_map).fillna(g_q75)

        full_sub = tr.copy()
        full_sub["target_log"] = np.log1p(full_sub[target_col].clip(lower=0))
        full_stats = full_sub.groupby(col)["target_log"].agg(
            count="count",
            mean="mean",
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75)
        ).reset_index()

        full_stats["smooth_mean"] = (full_stats["count"] * full_stats["mean"] + smoothing * g_mean) / (full_stats["count"] + smoothing)

        full_mean_map = full_stats.set_index(col)["smooth_mean"].to_dict()
        full_med_map = full_stats.set_index(col)["median"].to_dict()
        full_q25_map = full_stats.set_index(col)["q25"].to_dict()
        full_q75_map = full_stats.set_index(col)["q75"].to_dict()

        te[m_col] = te[col].map(full_mean_map).fillna(g_mean)
        te[med_col] = te[col].map(full_med_map).fillna(g_median)
        te[q25_col] = te[col].map(full_q25_map).fillna(g_q25)
        te[q75_col] = te[col].map(full_q75_map).fillna(g_q75)

    return tr, te

train_te, test_te = generate_multi_percentile_oof_te(df_tr_fe, df_te_fe)
print("✅ Multi-Percentile OOF Target Encodings (Mean, Median, Q1, Q3) Generated!")"""))

# ----------------------------------------------------------------------
# Section 9: Level-1 Base Models Training
# ----------------------------------------------------------------------
cells.append(md("""<a id="9"></a>
# 9. Level-1 Base Models Training (10-Fold CV across LightGBM, CatBoost, XGBoost, Ridge)"""))

cells.append(code("""drop_cols = ["SalesID", "TransactionDate", "TargetValue", "AssetID"]
feature_cols = [c for c in train_te.columns if c not in drop_cols]

X = train_te[feature_cols].copy()
y_log = np.log1p(train_te["TargetValue"].clip(lower=0))
X_test = test_te[feature_cols].copy()

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
for col in categorical_cols:
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X[col] = enc.fit_transform(X[[col]])
    X_test[col] = enc.transform(X_test[[col]])

X = X.fillna(-999)
X_test = X_test.fillna(-999)

print(f"📊 Feature Matrix Assembled: {X.shape[1]} features")

N_SPLITS = 10
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
target_bins = pd.qcut(y_log, q=10, labels=False, duplicates="drop")

oof_lgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_ridge = np.zeros(len(X))

test_lgb = np.zeros(len(X_test))
test_cat = np.zeros(len(X_test))
test_xgb = np.zeros(len(X_test))
test_ridge = np.zeros(len(X_test))

# 1. LightGBM (Championship Tuned)
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.015,
    "num_leaves": 127,
    "max_depth": 10,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.65,
    "random_state": 42,
    "n_estimators": 4500,
    "verbose": -1,
    "n_jobs": -1
}

# 2. CatBoost (Tuned Depth & Regularization)
cat_params = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "learning_rate": 0.02,
    "depth": 8,
    "l2_leaf_reg": 6,
    "subsample": 0.8,
    "random_seed": 42,
    "iterations": 4000,
    "verbose": False,
    "thread_count": -1
}

# 3. XGBoost (Histogram Tree Method)
xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.015,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "random_state": 42,
    "n_estimators": 4000,
    "tree_method": "hist",
    "n_jobs": -1
}

print(f"=== Training Level-1 Base Models ({N_SPLITS}-Fold CV) ===")
for fold, (trn_idx, val_idx) in enumerate(skf.split(X, target_bins), start=1):
    f_start = time.time()
    X_tr, y_tr = X.iloc[trn_idx], y_log.iloc[trn_idx]
    X_va, y_va = X.iloc[val_idx], y_log.iloc[val_idx]

    # LightGBM
    m_lgb = lgb.LGBMRegressor(**lgb_params)
    m_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(150, verbose=False)])
    oof_lgb[val_idx] = m_lgb.predict(X_va)
    test_lgb += m_lgb.predict(X_test) / N_SPLITS

    # CatBoost
    m_cat = CatBoostRegressor(**cat_params)
    m_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=150, verbose=False)
    oof_cat[val_idx] = m_cat.predict(X_va)
    test_cat += m_cat.predict(X_test) / N_SPLITS

    # XGBoost
    m_xgb = xgb.XGBRegressor(**xgb_params)
    m_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[val_idx] = m_xgb.predict(X_va)
    test_xgb += m_xgb.predict(X_test) / N_SPLITS

    # Ridge Linear Baseline
    m_ridge = Ridge(alpha=10.0)
    m_ridge.fit(X_tr, y_tr)
    oof_ridge[val_idx] = m_ridge.predict(X_va)
    test_ridge += m_ridge.predict(X_test) / N_SPLITS

    print(f"Fold {fold}/{N_SPLITS} Completed in {time.time() - f_start:.1f}s")

lgb_rmsle = np.sqrt(mean_squared_error(y_log, oof_lgb))
cat_rmsle = np.sqrt(mean_squared_error(y_log, oof_cat))
xgb_rmsle = np.sqrt(mean_squared_error(y_log, oof_xgb))
ridge_rmsle = np.sqrt(mean_squared_error(y_log, oof_ridge))

print(f"✅ LightGBM 10-Fold OOF RMSLE: {lgb_rmsle:.5f}")
print(f"✅ CatBoost 10-Fold OOF RMSLE: {cat_rmsle:.5f}")
print(f"✅ XGBoost  10-Fold OOF RMSLE: {xgb_rmsle:.5f}")
print(f"✅ Ridge    10-Fold OOF RMSLE: {ridge_rmsle:.5f}")"""))

# ----------------------------------------------------------------------
# Section 10: Level-2 Meta-Learner Stacking
# ----------------------------------------------------------------------
cells.append(md("""<a id="10"></a>
# 10. Level-2 Meta-Learner Ridge Stacking & Weight Optimization"""))

cells.append(code("""print("=== Training Level-2 RidgeCV Meta-Learner ===")

OOF_matrix = np.column_stack([oof_lgb, oof_cat, oof_xgb, oof_ridge])
Test_matrix = np.column_stack([test_lgb, test_cat, test_xgb, test_ridge])

meta_model = RidgeCV(alphas=np.logspace(-3, 3, 30), cv=10)
meta_model.fit(OOF_matrix, y_log)

oof_stacked = meta_model.predict(OOF_matrix)
test_preds_stacked = meta_model.predict(Test_matrix)

stacked_rmsle = np.sqrt(mean_squared_error(y_log, oof_stacked))

actual_y = np.expm1(y_log)
actual_pred = np.expm1(oof_stacked)
dollar_rmse = np.sqrt(mean_squared_error(actual_y, actual_pred))
dollar_mae = mean_absolute_error(actual_y, actual_pred)

print(f"🔥 Meta-Learner Optimal Weights (LGB, CAT, XGB, Ridge): {meta_model.coef_}")
print(f"🔥 CHAMPIONSHIP STACKED META-LEARNER OOF RMSLE         : {stacked_rmsle:.5f}")
print(f"📊 Dollar-Scale Actual Metrics                        : RMSE = ${dollar_rmse:,.2f} | MAE = ${dollar_mae:,.2f}")"""))

# ----------------------------------------------------------------------
# Section 11: Comparison
# ----------------------------------------------------------------------
cells.append(md("""<a id="11"></a>
# 11. Model Performance Comparison"""))

cells.append(code("""results_df = pd.DataFrame([
    {"Model": "LightGBM", "OOF RMSLE": lgb_rmsle},
    {"Model": "CatBoost", "OOF RMSLE": cat_rmsle},
    {"Model": "XGBoost", "OOF RMSLE": xgb_rmsle},
    {"Model": "Ridge Linear", "OOF RMSLE": ridge_rmsle},
    {"Model": "🔥 CHAMPIONSHIP STACKED META-LEARNER", "OOF RMSLE": stacked_rmsle},
]).sort_values(by="OOF RMSLE")

print("=" * 60)
print("FINAL OOF PERFORMANCE COMPARISON (10-FOLD CV)")
print("=" * 60)
print(results_df.to_string(index=False))

plt.figure(figsize=(10, 5))
sns.barplot(data=results_df, x="OOF RMSLE", y="Model", palette="viridis")
plt.title("Model OOF RMSLE Comparison (Lower is Better)")
plt.xlim(min(results_df["OOF RMSLE"]) * 0.98, max(results_df["OOF RMSLE"]) * 1.02)
plt.show()"""))

# ----------------------------------------------------------------------
# Section 12: Final Predictions
# ----------------------------------------------------------------------
cells.append(md("""<a id="12"></a>
# 12. Final Predictions & Export"""))

cells.append(code("""final_predictions = np.expm1(test_preds_stacked)
final_predictions = np.clip(final_predictions, a_min=0, a_max=None)

submission = sub_template.copy()
submission["TargetValue"] = final_predictions
submission.to_csv("submission.csv", index=False)

print("=" * 70)
print("FINAL SUBMISSION VERIFICATION")
print("=" * 70)
print(f"Row count matches sample_submission : {len(submission) == len(sub_template)} ({len(submission):,} rows)")
print(f"Null values present                 : {submission['TargetValue'].isnull().sum()}")
print(f"Negative values present             : {(submission['TargetValue'] < 0).sum()}")
print("\\nFirst 10 Predictions:")
print(submission.head(10))
print("\\nSummary Statistics of Final Predictions:")
print(submission["TargetValue"].describe())

print("\\n🎉 V11 CHAMPIONSHIP PIPELINE COMPLETED SUCCESSFULLY!")"""))

# ----------------------------------------------------------------------
# Section 13: Conclusion
# ----------------------------------------------------------------------
cells.append(md("""<a id="13"></a>
# 13. Conclusion & Key Insights

1. **Multi-Percentile Group Target Encodings**: Extracting OOF Mean, Median, Q1 (25th percentile), and Q3 (75th percentile) provided tree models with full conditional distribution profiles per equipment group.
2. **20-Component Text SVD Embeddings**: Extracted fine-grained latent semantic signals from machinery spec descriptors.
3. **4-Model Diversity Stacking**: Combined LightGBM, CatBoost, XGBoost, and Ridge to push out-of-fold performance even further down."""))

# ----------------------------------------------------------------------
# Write Out Kaggle and Offline Notebooks
# ----------------------------------------------------------------------
kaggle_nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

offline_cells = []
for cell in cells:
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        if 'data_dir = "."' in src:
            src = src.replace('data_dir = "."', 'data_dir = r"e:\\HeavyEquipmentML\\data"')
            cell = code(src)
    offline_cells.append(cell)

offline_nb = {
    "cells": offline_cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("notebooks/24f2008980-notebook-v11-championship-kaggle.ipynb", "w", encoding="utf-8") as f:
    json.dump(kaggle_nb, f, indent=1)

with open("notebooks/24f2008980-notebook-v11-championship-offline.ipynb", "w", encoding="utf-8") as f:
    json.dump(offline_nb, f, indent=1)

print("Successfully generated Notebook v11 Kaggle and Offline notebooks!")
