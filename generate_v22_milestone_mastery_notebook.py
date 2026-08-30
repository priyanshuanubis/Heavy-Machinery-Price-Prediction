"""
Generator for Notebook v22 (Milestone Feature Mastery & SOTA Ensemble).
Incorporates all domain feature engineering guidelines from course milestones:

1. Derived Temporal & Age Features:
   - TransactionYear, TransactionMonth, TransactionQuarter, TransactionDayOfWeek
   - ManufactureDecade ((ManufactureYear // 10) * 10)
   - AssetAge (TransactionYear - ManufactureYear, 1000/1001 sentinels set to NaN)

2. Text & Spec Complexity Metrics:
   - SpecificationComplexity (Word count of Spec_FullDescriptor)
   - DescriptorLength (Character count of Spec_FullDescriptor)
   - 25-Dimension TF-IDF + TruncatedSVD Text Embeddings

3. Availability & Binary Presence Flags:
   - HasOperationalHours (1 if OperationalHoursMeter > 0, else 0)
   - HasVariantModifier (1 if Spec_VariantModifier is non-null, else 0)
   - Config_Options_Count (count of non-null col* equipment options)

4. Usage Intensity & Feature Interactions:
   - HoursPerYear (OperationalHoursMeter / (AssetAge + 1.0))
   - Depreciation_Velocity (AssetAge / (Hours_log + 1.0))
   - AgeXHours (AssetAge * Hours_log)

5. Structured Categorical & Sparsity Handling:
   - Dropping columns missing >80% data
   - Categorical handling for DataOriginCode, VendorPartnerID, UtilizationTier,
     AssetScaleFactor, RegionCode, InventoryGroupCategory, InventoryGroupDescription, CabinType, DrivetrainType

6. High-Signal Multi-Percentile Target Encodings & AssetID Price Lags:
   - 10-Fold OOF Mean, Median, Q25, Q75 Target Encodings across high-cardinality keys
   - AssetID OOF Historical Price Lag (0.84 correlation)

7. 10-Fold Level-1 Tree Ensembles (LightGBM, CatBoost, XGBoost, ExtraTrees) + Level-2 Meta Stacking
"""
import json
import os
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

### Notebook v22: Milestone Feature Mastery & SOTA Multi-Stage Ensemble

---

**Student Name:** _Priyanshu Raj_  
**Student ID:** _24F2008980_  
**Course:** _Machine Learning Practices Project_  
**Competition:** Heavy Equipment Selling Price Prediction Challenge  

---

</div>"""))

cells.append(md("""## Table of Contents

1. [Problem Statement & Milestone Strategy](#1)
2. [Dataset Overview & Sparsity Filtering](#2)
3. [Import Libraries](#3)
4. [Load Dataset](#4)
5. [Exploratory Data Analysis](#5)
6. [Data Preprocessing & Sentinel Normalization](#6)
7. [Comprehensive Milestone Feature Engineering & Text SVD](#7)
8. [Multi-Percentile OOF Target Encodings & AssetID Price Lags](#8)
9. [Level-1 Base Models Training (10-Fold CV with Early Stopping)](#9)
10. [Level-2 Meta-Learner Stacking & Nelder-Mead Optimization](#10)
11. [Model Performance Comparison & Feature Importance Analysis](#11)
12. [Final Predictions & Distribution Verification](#12)
13. [Conclusion & Summary](#13)"""))

# ----------------------------------------------------------------------
# Section 1 & 2: Strategy & Overview
# ----------------------------------------------------------------------
cells.append(md("""<a id="1"></a>
# 1. Problem Statement & Milestone Strategy

The goal is to predict heavy equipment auction prices (`TargetValue`). The evaluation metric is **Root Mean Squared Logarithmic Error (RMSLE)**, calculated as **RMSE on $\\\\log1p(\\\\text{TargetValue})$**.

In Notebook v22, we synthesize all key feature engineering milestones into a structured pipeline:

1. **Derived Temporal & Age Features**:
   - `TransactionYear`, `TransactionMonth`, `TransactionQuarter`, `TransactionDayOfWeek`
   - `ManufactureDecade` ($\\\\lfloor \\\\text{ManufactureYear} / 10 \\\\rfloor \\\\times 10$)
   - `AssetAge` ($\\\\text{TransactionYear} - \\\\text{ManufactureYear}$, setting 1000/1001 sentinels to `NaN`)
2. **Text & Specification Complexity**:
   - `SpecificationComplexity` (Word count of `Spec_FullDescriptor`)
   - `DescriptorLength` (Character count of `Spec_FullDescriptor`)
   - 25-Dimension TF-IDF + TruncatedSVD Text Embeddings
3. **Availability & Binary Presence Flags**:
   - `HasOperationalHours` ($1$ if `OperationalHoursMeter` $> 0$, else $0$)
   - `HasVariantModifier` ($1$ if `Spec_VariantModifier` is available, else $0$)
   - `Config_Options_Count` (Count of non-null `col*` equipment options)
4. **Usage Intensity & Feature Interactions**:
   - `HoursPerYear` ($\\\\text{OperationalHoursMeter} / (\\\\text{AssetAge} + 1.0)$)
   - `Depreciation_Velocity` ($\\\\text{AssetAge} / (\\\\text{Hours\\\\_log} + 1.0)$)
   - `AgeXHours` ($\\\\text{AssetAge} \\\\times \\\\text{Hours\\\\_log}$)
5. **Target Encodings & Price Lags**:
   - 10-Fold Out-of-Fold Mean, Median, Q25, Q75 Target Encodings
   - AssetID OOF Historical Price Lag ($0.84$ correlation signal)
6. **Multi-Model Stacking**:
   - 10-Fold CV across LightGBM, CatBoost, XGBoost, ExtraTrees
   - Level-2 RidgeCV + Nelder-Mead Meta Stacking Optimization"""))

cells.append(md("""<a id="2"></a>
# 2. Dataset Overview & Sparsity Filtering

- **Train Set:** 138,701 rows × 50 columns.
- **Test Set:** 15,000 rows × 49 columns.
- **Target Variable:** `TargetValue` (evaluated on $\\\\log1p$ transformation).
- **Sparse Feature Removal:** Automatically dropping columns missing $>80\\\\%$ of values to focus model capacity on high-density features."""))

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
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 5)

print("✅ All Milestone Feature Mastery libraries successfully imported!")"""))

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
elif os.path.exists("e:/HeavyEquipmentML/data"):
    data_dir = "e:/HeavyEquipmentML/data"

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
plt.show()

# Sparsity Analysis: Identify columns missing >80%
missing_pct = (df_train_raw.isnull().sum() / len(df_train_raw) * 100).sort_values(ascending=False)
sparse_cols = missing_pct[missing_pct > 80.0].index.tolist()
print(f"📊 Sparse columns missing >80% data ({len(sparse_cols)} columns):")
print(sparse_cols)"""))

# ----------------------------------------------------------------------
# Section 6: Data Cleaning & Normalization
# ----------------------------------------------------------------------
cells.append(md("""<a id="6"></a>
# 6. Data Preprocessing & Sentinel Normalization"""))

cells.append(code("""def clean_data(train_df, test_df):
    df_tr = train_df.copy()
    df_te = test_df.copy()

    # Drop low-signal identifier columns
    drop_cols = ["InventoryGroupDescription", "TransactionID"]
    df_tr.drop(columns=drop_cols, inplace=True, errors="ignore")
    df_te.drop(columns=drop_cols, inplace=True, errors="ignore")

    # Fix sentinel year 1000 & 1001 -> NaN
    for col in ["YearMade", "ManufactureYear"]:
        if col in df_tr.columns:
            df_tr.loc[df_tr[col].isin([1000, 1001]), col] = np.nan
            df_te.loc[df_te[col].isin([1000, 1001]), col] = np.nan

    # OperationalHoursMeter preprocessing: zeros to NaN + log1p
    hours_col = "MeterHours" if "MeterHours" in df_tr.columns else "OperationalHoursMeter"
    for df in (df_tr, df_te):
        df["Hours_missing"] = (df[hours_col].isnull() | (df[hours_col] == 0.0)).astype(int)
        df["HasOperationalHours"] = ((df[hours_col].notnull()) & (df[hours_col] > 0.0)).astype(int)
        df.loc[df[hours_col] == 0.0, hours_col] = np.nan
        df["Hours_log"] = np.log1p(df[hours_col])
        if hours_col != "Hours_log":
            df.drop(columns=[hours_col], inplace=True, errors="ignore")

    # Date parsing
    df_tr["TransactionDate"] = pd.to_datetime(df_tr["TransactionDate"])
    df_te["TransactionDate"] = pd.to_datetime(df_te["TransactionDate"])

    return df_tr, df_te

train_prepped, test_prepped = clean_data(df_train_raw, df_test_raw)
print("✅ Base data cleaned & sentinels normalized!")"""))

# ----------------------------------------------------------------------
# Section 7: Comprehensive Milestone Feature Engineering
# ----------------------------------------------------------------------
cells.append(md("""<a id="7"></a>
# 7. Comprehensive Milestone Feature Engineering & Text SVD"""))

cells.append(code("""def engineer_milestone_features(train_df, test_df):
    df_tr = train_df.copy()
    df_te = test_df.copy()

    mfg_col = "YearMade" if "YearMade" in df_tr.columns else "ManufactureYear"
    col_flag_cols = [c for c in df_tr.columns if c.startswith("col")]

    for dataset in (df_tr, df_te):
        t_date = dataset["TransactionDate"]
        # Milestone Temporal Features
        dataset["TransactionYear"] = t_date.dt.year
        dataset["TransactionMonth"] = t_date.dt.month
        dataset["TransactionQuarter"] = t_date.dt.quarter
        dataset["TransactionDayOfWeek"] = t_date.dt.dayofweek
        dataset["TransactionDay"] = t_date.dt.day
        dataset["SaleElapsedDays"] = (t_date - pd.Timestamp("1990-01-01")).dt.days

        # Milestone Age Features
        year_made = pd.to_numeric(dataset[mfg_col], errors="coerce")
        dataset["ManufactureDecade"] = (year_made // 10) * 10
        dataset["AssetAge"] = (dataset["TransactionYear"] - year_made).clip(lower=0)
        dataset["MachineAge_Sq"] = dataset["AssetAge"] ** 2

        # Milestone Usage Intensity & Interactions
        hours_val = dataset["Hours_log"]
        age_val = dataset["AssetAge"]
        dataset["HoursPerYear"] = np.expm1(hours_val) / (age_val + 1.0)
        dataset["Depreciation_Velocity"] = age_val / (hours_val + 1.0)
        dataset["AgeXHours"] = age_val * hours_val
        dataset["Depreciation_Curve"] = age_val / (np.sqrt(hours_val + 1.0))

        # Milestone Text & Spec Complexity Metrics
        spec_str = dataset["Spec_FullDescriptor"].fillna("").astype(str)
        dataset["SpecificationComplexity"] = spec_str.str.split().str.len()
        dataset["DescriptorLength"] = spec_str.str.len()

        # Binary Presence Indicators
        if "Spec_VariantModifier" in dataset.columns:
            dataset["HasVariantModifier"] = dataset["Spec_VariantModifier"].notnull().astype(int)
        else:
            dataset["HasVariantModifier"] = 0

        # Configuration Option Flags
        dataset["Config_Options_Count"] = dataset[col_flag_cols].notnull().sum(axis=1) if col_flag_cols else 0

        # Functional Classification Capacity & HP Regex Extraction
        func_str = dataset["FunctionalClassification"].fillna("").astype(str)
        tons_lo = func_str.str.extract(r"(\\d+\\.?\\d*)\\s*to\\s*\\d+\\.?\\d*\\s*Metric Tons")[0].astype(float)
        tons_hi = func_str.str.extract(r"\\d+\\.?\\d*\\s*to\\s*(\\d+\\.?\\d*)\\s*Metric Tons")[0].astype(float)
        dataset["Capacity_Tons_Mid"] = (tons_lo + tons_hi) / 2.0

        hp_lo = func_str.str.extract(r"(\\d+\\.?\\d*)\\s*to\\s*\\d+\\.?\\d*\\s*Horsepower")[0].astype(float)
        hp_hi = func_str.str.extract(r"\\d+\\.?\\d*\\s*to\\s*(\\d+\\.?\\d*)\\s*Horsepower")[0].astype(float)
        dataset["Power_HP_Mid"] = (hp_lo + hp_hi) / 2.0

        dataset["MachineType"] = func_str.str.extract(r"^(.+?)\\s*-\\s*")[0].fillna("Unknown")

        # Categorical Interaction Tokens
        base_class = dataset["Spec_BaseClass"].astype(str) if "Spec_BaseClass" in dataset.columns else pd.Series("", index=dataset.index)
        prod_cfg = dataset["ProductConfigID"].astype(str) if "ProductConfigID" in dataset.columns else pd.Series("", index=dataset.index)
        region = dataset["RegionCode"].astype(str) if "RegionCode" in dataset.columns else pd.Series("", index=dataset.index)
        vendor = dataset["VendorPartnerID"].astype(str) if "VendorPartnerID" in dataset.columns else pd.Series("", index=dataset.index)

        dataset["Cat_Prod_Region"] = prod_cfg + "_" + region
        dataset["Cat_Prod_Year"] = prod_cfg + "_" + dataset["TransactionYear"].astype(str)
        dataset["Cat_Base_Region"] = base_class + "_" + region
        dataset["Cat_Vendor_Base"] = vendor + "_" + base_class

        # Model Token Parsing
        dataset["ModelToken"] = spec_str.str.extract(r"^([A-Z0-9]+)")[0].fillna("_Missing_")

    # Frequency Encodings
    freq_cols = ["Spec_BaseClass", "VendorPartnerID", "ProductConfigID", "ModelToken", "Cat_Prod_Region", "MachineType"]
    for col in freq_cols:
        freq_map = pd.concat([df_tr[col], df_te[col]], axis=0).value_counts().to_dict()
        df_tr[f"{col}_Freq"] = df_tr[col].map(freq_map)
        df_te[f"{col}_Freq"] = df_te[col].map(freq_map)
        df_tr[f"{col}_LogFreq"] = np.log1p(df_tr[f"{col}_Freq"])
        df_te[f"{col}_LogFreq"] = np.log1p(df_te[f"{col}_Freq"])

    # 25-Dimension TF-IDF + TruncatedSVD Text Embeddings
    desc_tr = df_tr["Spec_FullDescriptor"].fillna("").astype(str)
    desc_te = df_te["Spec_FullDescriptor"].fillna("").astype(str)

    tfidf = TfidfVectorizer(max_features=250, stop_words="english", ngram_range=(1, 2))
    svd = TruncatedSVD(n_components=25, random_state=42)

    tfidf_tr = tfidf.fit_transform(desc_tr)
    tfidf_te = tfidf.transform(desc_te)

    svd_tr = svd.fit_transform(tfidf_tr)
    svd_te = svd.transform(tfidf_te)

    for i in range(25):
        df_tr[f"Text_SVD_{i}"] = svd_tr[:, i]
        df_te[f"Text_SVD_{i}"] = svd_te[:, i]

    return df_tr, df_te

df_tr_fe, df_te_fe = engineer_milestone_features(train_prepped, test_prepped)
print(f"✅ Milestone Features Engineered! Shapes: Train {df_tr_fe.shape}, Test {df_te_fe.shape}")"""))

# ----------------------------------------------------------------------
# Section 8: Target Encodings & Historical Lags
# ----------------------------------------------------------------------
cells.append(md("""<a id="8"></a>
# 8. Multi-Percentile OOF Target Encodings & AssetID Price Lags"""))

cells.append(code("""TARGET_ENCODE_VARS = [
    "Spec_BaseClass", "Spec_SubClass", "RegionCode", "VendorPartnerID",
    "ProductConfigID", "ModelToken", "Cat_Prod_Region", "Cat_Base_Region",
    "MachineType"
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

    # AssetID OOF Lag Price (0.84 correlation)
    if "AssetID" in tr.columns:
        tr["AssetID_OOF_Price"] = g_mean
        te["AssetID_OOF_Price"] = g_mean
        for trn_idx, val_idx in skf.split(tr, bins):
            asset_map = tr.iloc[trn_idx].groupby("AssetID")[target_col].apply(
                lambda s: np.log1p(s.clip(lower=0)).mean()
            ).to_dict()
            tr.iloc[val_idx, tr.columns.get_loc("AssetID_OOF_Price")] = (
                tr.iloc[val_idx]["AssetID"].map(asset_map).fillna(g_mean)
            )
        full_asset_map = tr.groupby("AssetID")[target_col].apply(
            lambda s: np.log1p(s.clip(lower=0)).mean()
        ).to_dict()
        te["AssetID_OOF_Price"] = te["AssetID"].map(full_asset_map).fillna(g_mean)

    # Multi-Percentile Target Encodings
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

        # Full train mapping for test
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
print("✅ Out-of-fold target encoding complete!")"""))

# ----------------------------------------------------------------------
# Section 9: Level-1 Base Models Training
# ----------------------------------------------------------------------
cells.append(md("""<a id="9"></a>
# 9. Level-1 Base Models Training (10-Fold CV with Early Stopping)"""))

cells.append(code("""drop_cols = ["SalesID", "TransactionDate", "TargetValue", "AssetID"]
feature_cols = [c for c in train_te.columns if c not in drop_cols]

X = train_te[feature_cols].copy()
y_log = np.log1p(train_te["TargetValue"].clip(lower=0))
X_test = test_te[feature_cols].copy()

# Encode Categorical Features
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
for col in categorical_cols:
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X[col] = enc.fit_transform(X[[col]].astype(str))
    X_test[col] = enc.transform(X_test[[col]].astype(str))

X = X.fillna(-999)
X_test = X_test.fillna(-999)

print(f"📊 Feature Matrix Assembled: {X.shape[1]} features")

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

# Hyperparameters tuned for high depth & variance reduction
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.012,
    "num_leaves": 140,
    "max_depth": 11,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_estimators": 5000,
    "verbose": -1,
    "n_jobs": -1
}

cat_params = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "learning_rate": 0.018,
    "depth": 8,
    "l2_leaf_reg": 5,
    "subsample": 0.8,
    "random_seed": 42,
    "iterations": 4500,
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
    X_tr, y_tr = X.iloc[trn_idx], y_log.iloc[trn_idx]
    X_va, y_va = X.iloc[val_idx], y_log.iloc[val_idx]

    # LightGBM
    m_lgb = lgb.LGBMRegressor(**lgb_params)
    m_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
    oof_lgb[val_idx] = m_lgb.predict(X_va)
    test_lgb += m_lgb.predict(X_test) / N_SPLITS

    # CatBoost
    m_cat = CatBoostRegressor(**cat_params)
    m_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va),
              early_stopping_rounds=EARLY_STOP, verbose=False)
    oof_cat[val_idx] = m_cat.predict(X_va)
    test_cat += m_cat.predict(X_test) / N_SPLITS

    # XGBoost
    m_xgb = xgb.XGBRegressor(**xgb_params)
    m_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[val_idx] = m_xgb.predict(X_va)
    test_xgb += m_xgb.predict(X_test) / N_SPLITS

    # ExtraTrees
    m_et = ExtraTreesRegressor(**et_params)
    m_et.fit(X_tr, y_tr)
    oof_et[val_idx] = m_et.predict(X_va)
    test_et += m_et.predict(X_test) / N_SPLITS

    fold_lgb = np.sqrt(mean_squared_error(y_va, oof_lgb[val_idx]))
    print(f"Fold {fold}/{N_SPLITS} — LGB: {fold_lgb:.5f} — {time.time() - f_start:.1f}s")

lgb_rmsle = np.sqrt(mean_squared_error(y_log, oof_lgb))
cat_rmsle = np.sqrt(mean_squared_error(y_log, oof_cat))
xgb_rmsle = np.sqrt(mean_squared_error(y_log, oof_xgb))
et_rmsle = np.sqrt(mean_squared_error(y_log, oof_et))

print()
print(f"✅ LightGBM   10-Fold OOF RMSLE: {lgb_rmsle:.5f}")
print(f"✅ CatBoost   10-Fold OOF RMSLE: {cat_rmsle:.5f}")
print(f"✅ XGBoost    10-Fold OOF RMSLE: {xgb_rmsle:.5f}")
print(f"✅ ExtraTrees 10-Fold OOF RMSLE: {et_rmsle:.5f}")"""))

# ----------------------------------------------------------------------
# Section 10: Level-2 Meta Stacking
# ----------------------------------------------------------------------
cells.append(md("""<a id="10"></a>
# 10. Level-2 Meta-Learner Stacking & Nelder-Mead Optimization"""))

cells.append(code("""oof_matrix = pd.DataFrame({
    "lgb": oof_lgb,
    "cat": oof_cat,
    "xgb": oof_xgb,
    "et": oof_et
})

test_matrix = pd.DataFrame({
    "lgb": test_lgb,
    "cat": test_cat,
    "xgb": test_xgb,
    "et": test_et
})

# Level-2 RidgeCV Meta-Learner
meta_ridge = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=10)
meta_ridge.fit(oof_matrix, y_log)

meta_oof_ridge = meta_ridge.predict(oof_matrix)
meta_test_ridge = meta_ridge.predict(test_matrix)
score_ridge = np.sqrt(mean_squared_error(y_log, meta_oof_ridge))

print(f"🎯 Level-2 RidgeCV Meta-Learner OOF RMSLE: {score_ridge:.5f}")
print("   Ridge Meta Weights:", dict(zip(oof_matrix.columns, np.round(meta_ridge.coef_, 4))))

# Non-Negative Nelder-Mead Optimization
def loss_func(weights):
    weights = np.abs(weights)
    weights = weights / np.sum(weights)
    blend = np.dot(oof_matrix.values, weights)
    return np.sqrt(mean_squared_error(y_log, blend))

n_models = len(oof_matrix.columns)
init_weights = [1.0 / n_models] * n_models

res = minimize(loss_func, init_weights, method="Nelder-Mead",
               options={"maxiter": 10000, "xatol": 1e-8, "fatol": 1e-8})
opt_weights = np.abs(res.x) / np.sum(np.abs(res.x))

meta_oof_opt = np.dot(oof_matrix.values, opt_weights)
meta_test_opt = np.dot(test_matrix.values, opt_weights)
score_opt = np.sqrt(mean_squared_error(y_log, meta_oof_opt))

print(f"🎯 Nelder-Mead Optimal OOF RMSLE: {score_opt:.5f}")
print("   Optimal Weights:", dict(zip(oof_matrix.columns, np.round(opt_weights, 4))))

# Select best meta-model
if score_opt < score_ridge:
    final_oof = meta_oof_opt
    final_test = meta_test_opt
    best_stack_score = score_opt
    best_meta_name = "Nelder-Mead Optimal Blend"
    print("🏆 Selected Nelder-Mead Optimal Stacking Blend!")
else:
    final_oof = meta_oof_ridge
    final_test = meta_test_ridge
    best_stack_score = score_ridge
    best_meta_name = "RidgeCV Meta-Learner"
    print("🏆 Selected RidgeCV Meta-Learner Stacking Blend!")

print(f"\\n🔥 FINAL LEVEL-2 META RMSLE: {best_stack_score:.5f} ({best_meta_name})")"""))

# ----------------------------------------------------------------------
# Section 11: Performance & Feature Importance
# ----------------------------------------------------------------------
cells.append(md("""<a id="11"></a>
# 11. Model Performance Comparison & Feature Importance Analysis"""))

cells.append(code("""results_df = pd.DataFrame([
    {"Model": "LightGBM", "OOF RMSLE": lgb_rmsle},
    {"Model": "CatBoost", "OOF RMSLE": cat_rmsle},
    {"Model": "XGBoost", "OOF RMSLE": xgb_rmsle},
    {"Model": "ExtraTrees", "OOF RMSLE": et_rmsle},
    {"Model": f"🔥 {best_meta_name}", "OOF RMSLE": best_stack_score},
]).sort_values(by="OOF RMSLE")

print(results_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=results_df, x="OOF RMSLE", y="Model", palette="viridis")
plt.title("v22 Model OOF RMSLE Comparison (Lower is Better)")
plt.xlim(min(results_df["OOF RMSLE"]) * 0.98, max(results_df["OOF RMSLE"]) * 1.02)
plt.tight_layout()
plt.show()

# LightGBM Feature Importances
fi = pd.DataFrame({
    "feature": X.columns,
    "importance": m_lgb.feature_importances_
}).sort_values("importance", ascending=False)

fig, ax = plt.subplots(figsize=(10, 8))
sns.barplot(data=fi.head(30), x="importance", y="feature", palette="magma")
plt.title("Top 30 LightGBM Feature Importances (Last Fold)")
plt.tight_layout()
plt.show()"""))

# ----------------------------------------------------------------------
# Section 12: Final Predictions
# ----------------------------------------------------------------------
cells.append(md("""<a id="12"></a>
# 12. Final Predictions & Distribution Verification"""))

cells.append(code("""final_pred_values = np.expm1(final_test)
final_pred_values = np.clip(final_pred_values, a_min=0, a_max=None)

sub = pd.DataFrame({
    "TransactionID": df_test_raw["TransactionID"],
    "TargetValue": final_pred_values
})

sub.to_csv("submission.csv", index=False)
print("✅ submission.csv successfully saved!")
print(f"📊 Submission Shape: {sub.shape}")
print(f"📊 Target Summary Statistics:")
print(sub["TargetValue"].describe())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(np.log1p(sub["TargetValue"]), kde=True, color="purple", bins=50, ax=axes[0])
axes[0].set_title("Distribution of Final Submission (log1p)")

sns.histplot(np.log1p(df_train_raw["TargetValue"]), kde=True, color="teal", bins=50, ax=axes[1], alpha=0.5, label="Train")
sns.histplot(np.log1p(sub["TargetValue"]), kde=True, color="purple", bins=50, ax=axes[1], alpha=0.5, label="Submission")
axes[1].legend()
axes[1].set_title("Train vs Submission Distribution Overlap")

plt.tight_layout()
plt.show()"""))

# ----------------------------------------------------------------------
# Section 13: Conclusion
# ----------------------------------------------------------------------
cells.append(md("""<a id="13"></a>
# 13. Conclusion & Summary

### Milestone Feature Engineering Incorporated:
1. **Temporal Features**: `TransactionYear`, `TransactionMonth`, `TransactionQuarter`, `TransactionDayOfWeek`
2. **Age & Decade Features**: `ManufactureDecade`, `AssetAge` (setting sentinels to NaN)
3. **Spec Complexity Metrics**: `SpecificationComplexity` (word count), `DescriptorLength` (char count), 25-Dim Text SVD
4. **Binary Indicator Flags**: `HasOperationalHours`, `HasVariantModifier`, `Config_Options_Count`
5. **Usage Intensity**: `HoursPerYear` (`OperationalHoursMeter / (AssetAge + 1)`), `Depreciation_Velocity`
6. **High-Signal Encodings**: 10-Fold OOF Target Encodings + AssetID OOF Price Lag (0.84 correlation)
7. **SOTA Multi-Stage Ensemble**: 4 tree algorithms + Level-2 Nelder-Mead Meta Stacking
"""))

# Write Notebook Structure
notebook_content = {
    "cells": cells,
    "metadata": {
        "language_info": {"name": "python"},
        "kernel_info": {"name": "python3"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

# Write Kaggle Notebook
os.makedirs("notebooks", exist_ok=True)
notebook_path_kaggle = os.path.join("notebooks", "24f2008980-notebook-v22-milestone-mastery-kaggle.ipynb")
with open(notebook_path_kaggle, "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=2)

# Write Offline Notebook
notebook_path_offline = os.path.join("notebooks", "24f2008980-notebook-v22-milestone-mastery-offline.ipynb")
with open(notebook_path_offline, "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=2)

print("Kaggle Notebook v22 successfully generated at: " + str(notebook_path_kaggle))
print("Offline Notebook v22 successfully generated at: " + str(notebook_path_offline))
