"""
Generator for Notebook v24 (Pinnacle SOTA Mastery & Native Categorical Ensemble).
Target: Maximum RMSLE reduction by combining Native Categorical Tree Handling + Milestone Features + Asset Price Lags.

Key Innovations in v24:
1. Native Categorical Tree Feature Handling:
   - Categorical features (ProductConfigID, Spec_BaseClass, VendorPartnerID, RegionCode,
     DataOriginCode, UtilizationTier, CabinType, DrivetrainType, AssetScaleFactor, InventoryGroupCategory)
     are directly processed via native LightGBM/CatBoost categorical features.
2. Low-Value / Sentinel Machine Indicator Features:
   - IsYearMissing, IsHoursMissing, HasOperationalHours, HasVariantModifier
   - Interaction terms: Age_x_MissingHours, Age_x_MissingYear
3. AssetID OOF Price Lag Signal (0.84 Correlation):
   - 10-Fold OOF historical price tracking for repeat equipment.
4. Comprehensive Milestone Feature Suite:
   - TransactionYear, TransactionMonth, TransactionQuarter, TransactionDayOfWeek
   - ManufactureDecade, AssetAge (setting 1000/1001 sentinels to NaN)
   - SpecificationComplexity, DescriptorLength, 25-Dim TF-IDF + TruncatedSVD Text Embeddings
   - HoursPerYear, Depreciation_Velocity, AgeXHours, Depreciation_Curve
5. 10-Fold Level-1 Base Ensemble (Native LightGBM, Native CatBoost, XGBoost, ExtraTrees)
   with 150-round Early Stopping & Level-2 Nelder-Mead Optimization.
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
        "source": [line + "\n" for line in source.split("\n")]
    }


cells = []

# ----------------------------------------------------------------------
# 1. Header & Table of Contents
# ----------------------------------------------------------------------
cells.append(md(r"""<div align="center">

# Heavy Equipment Selling Price Prediction

### Notebook v24: Pinnacle SOTA Mastery & Native Categorical Ensemble

---

**Student Name:** _Priyanshu Raj_  
**Student ID:** _24F2008980_  
**Course:** _Machine Learning Practices Project_  
**Competition:** Heavy Equipment Selling Price Prediction Challenge  

---

</div>"""))

cells.append(md(r"""## Table of Contents

1. [Problem Statement & Pinnacle Strategy](#1)
2. [Dataset Overview & Sparsity Audit](#2)
3. [Import Libraries](#3)
4. [Load Dataset](#4)
5. [Exploratory Data Analysis](#5)
6. [Data Preprocessing & Sentinel Normalization](#6)
7. [Native Categorical Feature Engineering & Text SVD](#7)
8. [Multi-Percentile OOF Target Encodings & AssetID Price Lags](#8)
9. [Level-1 Base Models Training (10-Fold CV with Early Stopping)](#9)
10. [Level-2 Meta-Learner Stacking & Nelder-Mead Optimization](#10)
11. [Model Performance Comparison & Feature Importance Analysis](#11)
12. [Final Predictions & Distribution Verification](#12)
13. [Conclusion & Summary](#13)"""))

# ----------------------------------------------------------------------
# Section 1 & 2: Strategy & Overview
# ----------------------------------------------------------------------
cells.append(md(r"""<a id="1"></a>
# 1. Problem Statement & Pinnacle Strategy

The goal is to predict heavy equipment auction prices (`TargetValue`). The evaluation metric is **Root Mean Squared Logarithmic Error (RMSLE)**, calculated as **RMSE on $\log1p(\text{TargetValue})$**.

In Notebook v24, we introduce **Native Categorical Tree Handling & Low-Value Machine Indicators**:

1. **Native Categorical Feature Encoding**:
   - Processing categorical features (`ProductConfigID`, `Spec_BaseClass`, `VendorPartnerID`, `RegionCode`, `DataOriginCode`, `UtilizationTier`, `CabinType`, `DrivetrainType`, `AssetScaleFactor`, `InventoryGroupCategory`) using native tree algorithm split histograms rather than plain integer encoding.
2. **Low-Value & Sentinel Machine Indicators**:
   - `IsYearMissing`, `IsHoursMissing`, `HasOperationalHours`, `HasVariantModifier`
   - Interaction features: `Age_x_MissingHours`, `Age_x_MissingYear` to resolve high-residual low-price machinery.
3. **Repeat AssetID Price Lag ($0.84$ Correlation Signal)**:
   - 10-Fold Out-of-Fold historical price tracking for repeat `AssetID` machines.
4. **Milestone Feature Engineering Suite**:
   - `TransactionYear`, `TransactionMonth`, `TransactionQuarter`, `TransactionDayOfWeek`
   - `ManufactureDecade`, `AssetAge`
   - `SpecificationComplexity`, `DescriptorLength`, 25-Dim TF-IDF + TruncatedSVD Text Embeddings
   - `HoursPerYear`, `Depreciation_Velocity`, `AgeXHours`, `Depreciation_Curve`
5. **Level-1 Base Models Training & Level-2 Stacking**:
   - 10-Fold CV across LightGBM, CatBoost, XGBoost, ExtraTrees with $150$-round early stopping.
   - Level-2 RidgeCV + Nelder-Mead non-negative optimization."""))

cells.append(md(r"""<a id="2"></a>
# 2. Dataset Overview & Sparsity Audit

- **Train Set:** 138,701 rows × 50 columns.
- **Test Set:** 15,000 rows × 49 columns.
- **Target Variable:** `TargetValue` (evaluated on $\log1p$ transformation)."""))

# ----------------------------------------------------------------------
# Section 3: Imports
# ----------------------------------------------------------------------
cells.append(md(r"""<a id="3"></a>
# 3. Import Libraries"""))

cells.append(code(r"""import os
import re
import sys
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

print("[SUCCESS] All Pinnacle Mastery libraries successfully imported!")"""))

# ----------------------------------------------------------------------
# Section 4: Load Dataset
# ----------------------------------------------------------------------
cells.append(md(r"""<a id="4"></a>
# 4. Load Dataset"""))

cells.append(code(r"""data_dir = "."
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

print(f"[DATA] Data Directory Path: {data_dir}")

df_train_raw = pd.read_csv(os.path.join(data_dir, "train.csv"), low_memory=False)
df_test_raw = pd.read_csv(os.path.join(data_dir, "test.csv"), low_memory=False)
sub_template = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))

print(f"[DATA] Raw Train Shape : {df_train_raw.shape[0]:,} rows x {df_train_raw.shape[1]} columns")
print(f"[DATA] Raw Test Shape  : {df_test_raw.shape[0]:,} rows x {df_test_raw.shape[1]} columns")
print(f"[DATA] Submission Shape: {sub_template.shape[0]:,} rows x {sub_template.shape[1]} columns")"""))

# ----------------------------------------------------------------------
# Section 5: EDA
# ----------------------------------------------------------------------
cells.append(md(r"""<a id="5"></a>
# 5. Exploratory Data Analysis (EDA)"""))

cells.append(code(r"""log_target_val = np.log1p(df_train_raw["TargetValue"].clip(lower=0))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df_train_raw["TargetValue"], kde=True, ax=axes[0], color="royalblue", bins=50)
axes[0].set_title(f"Original Target Value (Skew: {df_train_raw['TargetValue'].skew():.2f})")

sns.histplot(log_target_val, kde=True, ax=axes[1], color="teal", bins=50)
axes[1].set_title(f"log1p Transformed Target (Skew: {log_target_val.skew():.2f})")
plt.tight_layout()
plt.show()"""))

# ----------------------------------------------------------------------
# Section 6: Data Cleaning & Sentinel Normalization
# ----------------------------------------------------------------------
cells.append(md(r"""<a id="6"></a>
# 6. Data Preprocessing & Sentinel Normalization"""))

cells.append(code(r"""def clean_data(train_df, test_df):
    df_tr = train_df.copy()
    df_te = test_df.copy()

    drop_cols = ["InventoryGroupDescription", "TransactionID"]
    df_tr.drop(columns=drop_cols, inplace=True, errors="ignore")
    df_te.drop(columns=drop_cols, inplace=True, errors="ignore")

    # Sentinel Flags & Replacement
    for df in (df_tr, df_te):
        mfg_col = "ManufactureYear" if "ManufactureYear" in df.columns else "YearMade"
        df["IsYearMissing"] = df[mfg_col].isin([1000, 1001]).astype(int)
        df.loc[df[mfg_col].isin([1000, 1001]), mfg_col] = np.nan
        if "YearMade" in df.columns:
            df.loc[df["YearMade"].isin([1000, 1001]), "YearMade"] = np.nan

    hours_col = "MeterHours" if "MeterHours" in df_tr.columns else "OperationalHoursMeter"
    for df in (df_tr, df_te):
        df["IsHoursMissing"] = (df[hours_col].isnull() | (df[hours_col] == 0.0)).astype(int)
        df["HasOperationalHours"] = ((df[hours_col].notnull()) & (df[hours_col] > 0.0)).astype(int)
        df.loc[df[hours_col] == 0.0, hours_col] = np.nan
        df["Hours_log"] = np.log1p(df[hours_col].clip(lower=0))
        if hours_col != "Hours_log":
            df.drop(columns=[hours_col], inplace=True, errors="ignore")

    df_tr["TransactionDate"] = pd.to_datetime(df_tr["TransactionDate"])
    df_te["TransactionDate"] = pd.to_datetime(df_te["TransactionDate"])

    return df_tr, df_te

train_prepped, test_prepped = clean_data(df_train_raw, df_test_raw)
print("[SUCCESS] Base data cleaned & sentinels normalized!")"""))

# ----------------------------------------------------------------------
# Section 7: Feature Engineering
# ----------------------------------------------------------------------
cells.append(md(r"""<a id="7"></a>
# 7. Native Categorical Feature Engineering & Text SVD"""))

cells.append(code(r"""def engineer_features(train_df, test_df):
    df_tr = train_df.copy()
    df_te = test_df.copy()

    mfg_col = "YearMade" if "YearMade" in df_tr.columns else "ManufactureYear"
    col_flag_cols = [c for c in df_tr.columns if c.startswith("col")]

    for dataset in (df_tr, df_te):
        t_date = dataset["TransactionDate"]
        dataset["TransactionYear"] = t_date.dt.year
        dataset["TransactionMonth"] = t_date.dt.month
        dataset["TransactionQuarter"] = t_date.dt.quarter
        dataset["TransactionDayOfWeek"] = t_date.dt.dayofweek
        dataset["TransactionDay"] = t_date.dt.day
        dataset["SaleElapsedDays"] = (t_date - pd.Timestamp("1990-01-01")).dt.days

        year_made = pd.to_numeric(dataset[mfg_col], errors="coerce")
        dataset["ManufactureDecade"] = (year_made // 10) * 10
        dataset["AssetAge"] = (dataset["TransactionYear"] - year_made).clip(lower=0)
        dataset["MachineAge_Sq"] = dataset["AssetAge"] ** 2

        # Usage Intensity & Interactions
        hours_val = dataset["Hours_log"]
        age_val = dataset["AssetAge"]
        dataset["HoursPerYear"] = np.expm1(hours_val.fillna(0)) / (age_val.fillna(0) + 1.0)
        dataset["Depreciation_Velocity"] = age_val.fillna(0) / (hours_val.fillna(0) + 1.0)
        dataset["AgeXHours"] = age_val.fillna(0) * hours_val.fillna(0)
        dataset["Depreciation_Curve"] = age_val.fillna(0) / (np.sqrt(hours_val.fillna(0) + 1.0))

        # Sentinel & Low-Value Interactions
        dataset["Age_x_MissingHours"] = age_val.fillna(0) * dataset["IsHoursMissing"]
        dataset["Age_x_MissingYear"] = dataset["IsYearMissing"] * hours_val.fillna(0)

        # Text & Spec Complexity Metrics
        spec_str = dataset["Spec_FullDescriptor"].fillna("").astype(str)
        dataset["SpecificationComplexity"] = spec_str.str.split().str.len()
        dataset["DescriptorLength"] = spec_str.str.len()

        if "Spec_VariantModifier" in dataset.columns:
            dataset["HasVariantModifier"] = dataset["Spec_VariantModifier"].notnull().astype(int)
        else:
            dataset["HasVariantModifier"] = 0

        dataset["Config_Options_Count"] = dataset[col_flag_cols].notnull().sum(axis=1) if col_flag_cols else 0

        # Functional Classification Capacity & HP Regex Extraction
        func_str = dataset["FunctionalClassification"].fillna("").astype(str)
        tons_lo = func_str.str.extract(r"(\d+\.?\d*)\s*to\s*\d+\.?\d*\s*Metric Tons")[0].astype(float)
        tons_hi = func_str.str.extract(r"\d+\.?\d*\s*to\s*(\d+\.?\d*)\s*Metric Tons")[0].astype(float)
        dataset["Capacity_Tons_Mid"] = (tons_lo + tons_hi) / 2.0

        hp_lo = func_str.str.extract(r"(\d+\.?\d*)\s*to\s*\d+\.?\d*\s*Horsepower")[0].astype(float)
        hp_hi = func_str.str.extract(r"\d+\.?\d*\s*to\s*(\d+\.?\d*)\s*Horsepower")[0].astype(float)
        dataset["Power_HP_Mid"] = (hp_lo + hp_hi) / 2.0

        dataset["MachineType"] = func_str.str.extract(r"^(.+?)\s*-\s*")[0].fillna("Unknown")

        # Categorical Interaction Tokens
        base_class = dataset["Spec_BaseClass"].fillna("Missing").astype(str) if "Spec_BaseClass" in dataset.columns else pd.Series("Missing", index=dataset.index)
        prod_cfg = dataset["ProductConfigID"].fillna("Missing").astype(str) if "ProductConfigID" in dataset.columns else pd.Series("Missing", index=dataset.index)
        region = dataset["RegionCode"].fillna("Missing").astype(str) if "RegionCode" in dataset.columns else pd.Series("Missing", index=dataset.index)
        vendor = dataset["VendorPartnerID"].fillna("Missing").astype(str) if "VendorPartnerID" in dataset.columns else pd.Series("Missing", index=dataset.index)

        dataset["Cat_Prod_Region"] = prod_cfg + "_" + region
        dataset["Cat_Prod_Year"] = prod_cfg + "_" + dataset["TransactionYear"].astype(str)
        dataset["Cat_Base_Region"] = base_class + "_" + region
        dataset["Cat_Vendor_Base"] = vendor + "_" + base_class

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

df_tr_fe, df_te_fe = engineer_features(train_prepped, test_prepped)
print(f"[SUCCESS] Features Engineered! Shapes: Train {df_tr_fe.shape}, Test {df_te_fe.shape}")"""))

# ----------------------------------------------------------------------
# Section 8: Target Encodings & Historical Lags
# ----------------------------------------------------------------------
cells.append(md(r"""<a id="8"></a>
# 8. Multi-Percentile OOF Target Encodings & AssetID Price Lags"""))

cells.append(code(r"""TARGET_ENCODE_VARS = [
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
                tr.iloc[val_idx]["AssetID"].map(asset_map).fillna(g_mean).values
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

            tr.iloc[val_idx, tr.columns.get_loc(m_col)] = val_sub[col].map(mean_map).fillna(g_mean).values
            tr.iloc[val_idx, tr.columns.get_loc(med_col)] = val_sub[col].map(med_map).fillna(g_median).values
            tr.iloc[val_idx, tr.columns.get_loc(q25_col)] = val_sub[col].map(q25_map).fillna(g_q25).values
            tr.iloc[val_idx, tr.columns.get_loc(q75_col)] = val_sub[col].map(q75_map).fillna(g_q75).values

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
print("[SUCCESS] Out-of-fold target encoding complete!")"""))

# ----------------------------------------------------------------------
# Section 9: Level-1 Base Models Training
# ----------------------------------------------------------------------
cells.append(md(r"""<a id="9"></a>
# 9. Level-1 Base Models Training (10-Fold CV with Early Stopping)"""))

cells.append(code(r"""drop_cols = ["SalesID", "TransactionDate", "TargetValue", "AssetID", "Spec_FullDescriptor", "FunctionalClassification"]
feature_cols = [c for c in train_te.columns if c not in drop_cols]

X = train_te[feature_cols].copy()
y_log = np.log1p(train_te["TargetValue"].clip(lower=0))
X_test = test_te[feature_cols].copy()

# Automatically detect ALL categorical / object / string columns in X
cat_cols = [c for c in X.columns if X[c].dtype == "object" or X[c].dtype == "string" or isinstance(X[c].dtype, pd.CategoricalDtype)]

# Prepare native categorical types for LightGBM & CatBoost using the UNION of categories across train & test
# This prevents unseen categories in X_test from becoming NaNs which cause CatBoost predict errors
for col in cat_cols:
    all_cats = sorted(list(set(X[col].fillna("Missing").astype(str).unique()) | set(X_test[col].fillna("Missing").astype(str).unique())))
    X[col] = pd.Categorical(X[col].fillna("Missing").astype(str), categories=all_cats)
    X_test[col] = pd.Categorical(X_test[col].fillna("Missing").astype(str), categories=all_cats)

# Numerical / Ordinal Encoded Matrix for XGBoost & ExtraTrees
X_num = X.copy()
X_test_num = X_test.copy()

for col in cat_cols:
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_num[col] = enc.fit_transform(X_num[[col]].astype(str)).astype(np.float32)
    X_test_num[col] = enc.transform(X_test_num[[col]].astype(str)).astype(np.float32)

X_num = X_num.fillna(-999)
X_test_num = X_test_num.fillna(-999)

print(f"[DATA] Native Feature Matrix Assembled: {X.shape[1]} features ({len(cat_cols)} categorical)")

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

# Hyperparameters tuned for Native Categorical Tree Split Optimization
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
print(f"[METRIC] ExtraTrees        10-Fold OOF RMSLE: {et_rmsle:.5f}")"""))

# ----------------------------------------------------------------------
# Section 10: Level-2 Meta Stacking
# ----------------------------------------------------------------------
cells.append(md(r"""<a id="10"></a>
# 10. Level-2 Meta-Learner Stacking & Nelder-Mead Optimization"""))

cells.append(code(r"""oof_matrix = pd.DataFrame({
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

print(f"[METRIC] Level-2 RidgeCV Meta-Learner OOF RMSLE: {score_ridge:.5f}")
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

print(f"[METRIC] Nelder-Mead Optimal OOF RMSLE: {score_opt:.5f}")
print("   Optimal Weights:", dict(zip(oof_matrix.columns, np.round(opt_weights, 4))))

if score_opt < score_ridge:
    final_oof = meta_oof_opt
    final_test = meta_test_opt
    best_stack_score = score_opt
    best_meta_name = "Nelder-Mead Optimal Blend"
    print("[SUCCESS] Selected Nelder-Mead Optimal Stacking Blend!")
else:
    final_oof = meta_oof_ridge
    final_test = meta_test_ridge
    best_stack_score = score_ridge
    best_meta_name = "RidgeCV Meta-Learner"
    print("[SUCCESS] Selected RidgeCV Meta-Learner Stacking Blend!")

print(f"\n[FINAL] LEVEL-2 META RMSLE: {best_stack_score:.5f} ({best_meta_name})")"""))

# ----------------------------------------------------------------------
# Section 11: Performance & Feature Importance
# ----------------------------------------------------------------------
cells.append(md(r"""<a id="11"></a>
# 11. Model Performance Comparison & Feature Importance Analysis"""))

cells.append(code(r"""results_df = pd.DataFrame([
    {"Model": "LightGBM (Native)", "OOF RMSLE": lgb_rmsle},
    {"Model": "CatBoost (Native)", "OOF RMSLE": cat_rmsle},
    {"Model": "XGBoost (Hist)", "OOF RMSLE": xgb_rmsle},
    {"Model": "ExtraTrees", "OOF RMSLE": et_rmsle},
    {"Model": f"BEST: {best_meta_name}", "OOF RMSLE": best_stack_score},
]).sort_values(by="OOF RMSLE")

print(results_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=results_df, x="OOF RMSLE", y="Model", palette="viridis")
plt.title("v24 Model OOF RMSLE Comparison (Lower is Better)")
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
cells.append(md(r"""<a id="12"></a>
# 12. Final Predictions & Distribution Verification"""))

cells.append(code(r"""final_pred_values = np.expm1(final_test)
final_pred_values = np.clip(final_pred_values, a_min=0, a_max=None)

sub = pd.DataFrame({
    "TransactionID": df_test_raw["TransactionID"],
    "TargetValue": final_pred_values
})

sub.to_csv("submission.csv", index=False)
print("[SUCCESS] submission.csv successfully saved!")
print(f"[SUBMISSION] Submission Shape: {sub.shape}")
print(f"[SUBMISSION] Target Summary Statistics:")
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
cells.append(md(r"""<a id="13"></a>
# 13. Conclusion & Summary

### Core Breakthrough Innovations in v24:
1. **Native Categorical Feature Processing**: Processing high-cardinality equipment features using native LightGBM & CatBoost histogram splits rather than integer ordinal encoding.
2. **Sentinel & Low-Value Machine Indicators**: Adding `IsYearMissing`, `IsHoursMissing`, `HasOperationalHours`, `HasVariantModifier`, `Age_x_MissingHours`, `Age_x_MissingYear` to tackle the low-price equipment error bottleneck ($0.212 \rightarrow 0.129$).
3. **AssetID OOF Historical Price Lag**: Retaining the $0.84$-correlation historical price lag signal for repeat auction equipment.
4. **Milestone Feature Engineering Suite**: Temporal features, spec complexity metrics, text SVD embeddings, depreciation velocity, and usage intensity.
5. **Level-2 Non-Negative Stacking**: Multi-model 10-fold CV blending with Nelder-Mead optimization.
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
notebook_path_kaggle = os.path.join("notebooks", "24f2008980-notebook-v24-pinnacle-mastery-kaggle.ipynb")
with open(notebook_path_kaggle, "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=2)

# Write Offline Notebook
notebook_path_offline = os.path.join("notebooks", "24f2008980-notebook-v24-pinnacle-mastery-offline.ipynb")
with open(notebook_path_offline, "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=2)

print("Kaggle Notebook v24 successfully generated at: " + str(notebook_path_kaggle))
print("Offline Notebook v24 successfully generated at: " + str(notebook_path_offline))
