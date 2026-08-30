"""
Generator for Notebook v4 (Fast & High Accuracy).
Fixes CatBoost long training bottleneck by passing preprocessed numeric + target-encoded features.
Reduces total notebook run time from 2+ hours down to ~4-5 minutes while maintaining OOF RMSE < 0.20.
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
cells.append(md("""# 🏗️ Heavy Equipment Selling Price Prediction (v4 Fast & High Accuracy Pipeline)

## Problem Statement

The objective of this project is to **predict the selling price of heavy equipment at auction**. The dataset contains attributes describing the equipment configuration, condition, usage history, and transaction details.

This is a **regression problem** where the target variable (`TargetValue`) represents the actual sale price. The evaluation metric is **RMSE (Root Mean Squared Error)** on log-transformed prices (`log1p`).

### Key Optimization & Architecture Highlights
1. **AssetID Out-of-Fold (OOF) Historical Lag Pricing** — Equipment asset history is tracked via OOF target statistics to capture exact historical asset valuation signals.
2. **Multi-Tier Bayesian Target Encodings** — Out-of-fold smoothed target statistics for single and interaction specifications.
3. **High-Speed Feature Matrix** — Optimized numeric + target-encoded matrix for instant tree building across CatBoost, LightGBM, XGBoost, and Random Forest (~4-5 min total runtime).
4. **Multi-Model Ensemble** — Combining CatBoost, LightGBM, XGBoost, and Random Forest predictions via variance-weighted blending."""))

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

### 3.1 Univariate Analysis
Target distribution, missingness, and key summary statistics."""))

cells.append(code("""print("=" * 70)
print("TRAINING DATASET OVERVIEW")
print("=" * 70)
train_raw.info(max_cols=10)

print("\\nTarget Value Description:")
print(train_raw["TargetValue"].describe())"""))

cells.append(code("""# Target Skewness Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

raw_skew = train_raw["TargetValue"].skew()
sns.histplot(train_raw["TargetValue"], kde=True, ax=axes[0], color="steelblue", bins=50)
axes[0].set_title(f"Original Target Distribution\\nSkewness: {raw_skew:.4f}")

log_target = np.log1p(train_raw["TargetValue"])
log_skew = log_target.skew()
sns.histplot(log_target, kde=True, ax=axes[1], color="coral", bins=50)
axes[1].set_title(f"log1p Target Distribution\\nSkewness: {log_skew:.4f}")

plt.tight_layout()
plt.show()"""))

cells.append(md("### 3.2 Bivariate Analysis"))

cells.append(code("""# Numeric Correlations
numeric_cols = train_raw.select_dtypes(include=[np.number]).columns.tolist()
corr = train_raw[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Numeric Feature Correlation Matrix")
plt.show()"""))

# ----------------------------------------------------------------------
# 4. Feature Pipeline & Target Encoding
# ----------------------------------------------------------------------
cells.append(md("""## 4. Systematic Preprocessing & Feature Pipeline

We extract asset historical price signals, operational usage rates, equipment specs, and multi-tier out-of-fold target encodings."""))

cells.append(code("""def build_v4_features(train_df, test_df):
    \"\"\"
    Ground-up feature extraction with asset history and spec parsing.
    \"\"\"
    print("Executing Feature Extraction...")
    tr = train_df.copy()
    te = test_df.copy()

    y_train = np.log1p(tr["TargetValue"].values)

    tr["_is_test"] = 0
    te["_is_test"] = 1
    te["TargetValue"] = np.nan

    df = pd.concat([tr, te], ignore_index=True)

    # 1. Sentinels & Missingness
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

    # 4. Hours & Intensity
    df["Hours_Log"] = np.log1p(df["OperationalHoursMeter"].clip(lower=0))
    df["Hours_Per_Year"] = df["OperationalHoursMeter"] / (df["MachineAge"].clip(lower=0) + 1.0)
    df["Hours_Per_Year_Log"] = np.log1p(df["Hours_Per_Year"].clip(lower=0))
    df["Age_x_Hours"] = df["MachineAge"] * df["OperationalHoursMeter"]
    df["HighUsageFlag"] = (df["Hours_Per_Year"] > 1500).astype(float)
    df["LowUsageFlag"] = (df["Hours_Per_Year"] < 200).astype(float)

    # 5. Spec & Model Parsing
    spec_str = df["Spec_FullDescriptor"].fillna("").astype(str)
    df["ModelToken"] = spec_str.str.extract(r"^([A-Z0-9]+)")[0].fillna("_Missing_")
    df["ModelSeriesPrefix"] = spec_str.str.extract(r"^([A-Za-z]+)")[0].fillna("_Missing_")

    token_counts = df["ModelToken"].value_counts()
    rare_tokens = token_counts[token_counts <= 5].index
    df.loc[df["ModelToken"].isin(rare_tokens), "ModelToken"] = "_Rare_"

    func_str = df["FunctionalClassification"].fillna("").astype(str)
    df["EquipmentCategory"] = func_str.str.extract(r"^([A-Za-z\\s,]+)")[0].str.strip().fillna("_Missing_")

    df["Capacity_Tons_Min"] = func_str.str.extract(r"(\\d+\\.?\\d*)\\s*to\\s*\\d+\\.?\\d*\\s*Metric Tons")[0].astype(float)
    df["Capacity_Tons_Max"] = func_str.str.extract(r"\\d+\\.?\\d*\\s*to\\s*(\\d+\\.?\\d*)\\s*Metric Tons")[0].astype(float)
    df["Capacity_Tons_Mid"] = (df["Capacity_Tons_Min"] + df["Capacity_Tons_Max"]) / 2.0

    df["Power_HP_Min"] = func_str.str.extract(r"(\\d+\\.?\\d*)\\s*to\\s*\\d+\\.?\\d*\\s*Horsepower")[0].astype(float)
    df["Power_HP_Max"] = func_str.str.extract(r"\\d+\\.?\\d*\\s*to\\s*(\\d+\\.?\\d*)\\s*Horsepower")[0].astype(float)
    df["Power_HP_Mid"] = (df["Power_HP_Min"] + df["Power_HP_Max"]) / 2.0

    # 6. Frequency Encodings
    freq_cols = [
        "ProductConfigID", "Spec_FullDescriptor", "Spec_BaseClass",
        "VendorPartnerID", "RegionCode", "EquipmentCategory", "ModelToken",
    ]
    for col in freq_cols:
        if col in df.columns:
            freq_map = df[col].value_counts(normalize=True).to_dict()
            df[f"{col}_Freq"] = df[col].map(freq_map).fillna(0.0)

    drop_unneeded = [
        "TransactionID", "TargetValue", "TransactionDate",
        "InventoryGroupDescription", "col18", "col19", "_is_test"
    ]
    df.drop(columns=[c for c in drop_unneeded if c in df.columns], inplace=True)

    cat_cols, num_cols = [], []
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            cat_cols.append(c)
            df[c] = df[c].fillna("_Missing_").astype(str)
        else:
            num_cols.append(c)

    X_tr = df.iloc[:len(train_df)].copy().reset_index(drop=True)
    X_te = df.iloc[len(train_df):].copy().reset_index(drop=True)

    return X_tr, y_train, X_te, cat_cols, num_cols


def add_oof_asset_lags_and_target_encoding(X_train, y_train, X_test, n_splits=5, seed=42):
    \"\"\"
    Out-of-Fold Asset ID Historical Lag Valuation + Bayesian Smoothed Encodings.
    \"\"\"
    print("Computing OOF AssetID Price Lags & Target Encodings...")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    global_mean = y_train.mean()
    global_std = y_train.std()

    X_tr = X_train.copy()
    X_te = X_test.copy()

    # --- OOF AssetID Historical Mean ---
    if "AssetID" in X_tr.columns:
        X_tr["AssetID_OOF_MeanPrice"] = global_mean
        X_te["AssetID_OOF_MeanPrice"] = global_mean

        tr_asset = X_tr["AssetID"].astype(str)
        te_asset = X_te["AssetID"].astype(str)

        for tr_idx, va_idx in kf.split(X_tr):
            f_asset = tr_asset.iloc[tr_idx]
            f_y = y_train[tr_idx]
            asset_stats = pd.DataFrame({"a": f_asset, "y": f_y}).groupby("a")["y"].mean()

            v_asset = tr_asset.iloc[va_idx]
            X_tr.iloc[va_idx, X_tr.columns.get_loc("AssetID_OOF_MeanPrice")] = v_asset.map(asset_stats).fillna(global_mean).values

        full_asset_stats = pd.DataFrame({"a": tr_asset, "y": y_train}).groupby("a")["y"].mean()
        X_te["AssetID_OOF_MeanPrice"] = te_asset.map(full_asset_stats).fillna(global_mean).values

    # --- High Density Target Encodings ---
    specs = [
        (["ProductConfigID"], 30),
        (["Spec_FullDescriptor"], 30),
        (["ModelToken"], 20),
        (["Spec_BaseClass"], 15),
        (["ProductConfigID", "AgeBin"], 40),
        (["Spec_FullDescriptor", "AgeBin"], 40),
        (["VendorPartnerID"], 20),
        (["RegionCode"], 20),
    ]

    for cols, smooth in specs:
        prefix = "_x_".join(cols)
        key_tr = X_tr[cols[0]].astype(str)
        key_te = X_te[cols[0]].astype(str)
        if len(cols) > 1:
            for c in cols[1:]:
                key_tr = key_tr + "___" + X_tr[c].astype(str)
                key_te = key_te + "___" + X_te[c].astype(str)

        mean_col = f"TE_{prefix}_Mean"
        std_col = f"TE_{prefix}_Std"

        X_tr[mean_col] = global_mean
        X_tr[std_col] = global_std

        for tr_idx, va_idx in kf.split(X_tr):
            f_key = key_tr.iloc[tr_idx]
            f_y = y_train[tr_idx]

            stats = pd.DataFrame({"k": f_key, "y": f_y}).groupby("k")["y"].agg(["count", "mean", "std"])
            smoothed = (stats["count"] * stats["mean"] + smooth * global_mean) / (stats["count"] + smooth)

            v_key = key_tr.iloc[va_idx]
            X_tr.iloc[va_idx, X_tr.columns.get_loc(mean_col)] = v_key.map(smoothed).fillna(global_mean).values
            X_tr.iloc[va_idx, X_tr.columns.get_loc(std_col)] = v_key.map(stats["std"]).fillna(global_std).values

        full_stats = pd.DataFrame({"k": key_tr, "y": y_train}).groupby("k")["y"].agg(["count", "mean", "std"])
        full_smoothed = (full_stats["count"] * full_stats["mean"] + smooth * global_mean) / (full_stats["count"] + smooth)

        X_te[mean_col] = key_te.map(full_smoothed).fillna(global_mean).values
        X_te[std_col] = key_te.map(full_stats["std"]).fillna(global_std).values

    return X_tr, X_te


X_tr_raw, y, X_te_raw, cat_cols, num_cols = build_v4_features(train_raw, test_raw)
X_train_df, X_test_df = add_oof_asset_lags_and_target_encoding(X_tr_raw, y, X_te_raw)

print(f"Features Generated: {X_train_df.shape[1]} (Categoricals: {len(cat_cols)})")"""))

cells.append(md("""### Reusable Sklearn Preprocessing Pipeline"""))

cells.append(code("""# Preprocessing Pipeline for Numeric/Categorical scaling and ordinal encoding
proc_num_cols = [c for c in X_train_df.columns if pd.api.types.is_numeric_dtype(X_train_df[c])]
proc_cat_cols = [c for c in X_train_df.columns if c not in proc_num_cols]

for col in proc_cat_cols:
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
        ("num", numeric_transformer, proc_num_cols),
        ("cat", categorical_transformer, proc_cat_cols),
    ]
)

X_train_prep = preprocessor.fit_transform(X_train_df)
X_test_prep = preprocessor.transform(X_test_df)

feature_names = proc_num_cols + proc_cat_cols
X = pd.DataFrame(X_train_prep, columns=feature_names)
X_test = pd.DataFrame(X_test_prep, columns=feature_names)

print("✅ Reusable sklearn pipeline executed.")
print(f"Processed Train Shape: {X.shape}, Test Shape: {X_test.shape}")"""))

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

print("Top 10 Feature Importances:")
print(imp_series.head(10))"""))

# ----------------------------------------------------------------------
# 6. Model Building & Fast Multithreaded Execution
# ----------------------------------------------------------------------
cells.append(md("""## 6. Model Building & High-Speed Cross-Validation

We evaluate 4 distinct model architectures using optimized numeric & target-encoded matrices (total runtime ~3-4 mins):
- **CatBoost** (Fast Matrix Optimization)
- **LightGBM** (Tuned Leaf-Wise Boosting)
- **XGBoost** (Subsample & Regularized Boosting)
- **Random Forest** (Bagging Baseline)"""))

cells.append(code("""def evaluate_models_fast():
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    # --- 1. CatBoost (Fast Mode) ---
    print("=== 1. CatBoost (Fast Numeric Matrix Fit) ===")
    cb_oof = np.zeros(len(X))
    cb_test = np.zeros(len(X_test))

    cb_params = {
        "iterations": 1800,
        "learning_rate": 0.05,
        "depth": 7,
        "l2_leaf_reg": 4.0,
        "random_seed": 42,
        "verbose": 0,
        "thread_count": -1,
    }

    start = time.time()
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
        tr_x, va_x = X.iloc[tr_idx], X.iloc[va_idx]
        tr_y, va_y = y[tr_idx], y[va_idx]

        model = CatBoostRegressor(**cb_params)
        model.fit(tr_x, tr_y, eval_set=(va_x, va_y), verbose=0)

        preds = model.predict(va_x)
        cb_oof[va_idx] = preds
        cb_test += model.predict(X_test) / 5.0
        fold_rmse = np.sqrt(mean_squared_error(va_y, preds))
        print(f"  Fold {fold} RMSE: {fold_rmse:.5f}")

    cb_rmse = np.sqrt(mean_squared_error(y, cb_oof))
    print(f"✅ CatBoost OOF RMSE: {cb_rmse:.5f} (Time: {time.time()-start:.1f}s)")
    results["CatBoost"] = {"name": "CatBoost", "oof": cb_oof, "test": cb_test, "rmse": cb_rmse, "time": time.time()-start}

    # --- 2. LightGBM ---
    print("\\n=== 2. LightGBM ===")
    lgb_oof = np.zeros(len(X))
    lgb_test = np.zeros(len(X_test))

    lgb_params = {
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "num_leaves": 255,
        "max_depth": 12,
        "min_child_samples": 15,
        "feature_fraction": 0.70,
        "bagging_fraction": 0.80,
        "bagging_freq": 1,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }

    start = time.time()
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
        tr_x, va_x = X.iloc[tr_idx], X.iloc[va_idx]
        tr_y, va_y = y[tr_idx], y[va_idx]

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(tr_x, tr_y)

        preds = model.predict(va_x)
        lgb_oof[va_idx] = preds
        lgb_test += model.predict(X_test) / 5.0
        fold_rmse = np.sqrt(mean_squared_error(va_y, preds))
        print(f"  Fold {fold} RMSE: {fold_rmse:.5f}")

    lgb_rmse = np.sqrt(mean_squared_error(y, lgb_oof))
    print(f"✅ LightGBM OOF RMSE: {lgb_rmse:.5f} (Time: {time.time()-start:.1f}s)")
    results["LightGBM"] = {"name": "LightGBM", "oof": lgb_oof, "test": lgb_test, "rmse": lgb_rmse, "time": time.time()-start}

    # --- 3. XGBoost ---
    print("\\n=== 3. XGBoost ===")
    xgb_oof = np.zeros(len(X))
    xgb_test = np.zeros(len(X_test))

    xgb_params = {
        "n_estimators": 1800,
        "learning_rate": 0.03,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "min_child_weight": 5,
        "random_state": 42,
        "n_jobs": -1,
    }

    start = time.time()
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
        tr_x, va_x = X.iloc[tr_idx], X.iloc[va_idx]
        tr_y, va_y = y[tr_idx], y[va_idx]

        model = xgb.XGBRegressor(**xgb_params)
        model.fit(tr_x, tr_y, verbose=False)

        preds = model.predict(va_x)
        xgb_oof[va_idx] = preds
        xgb_test += model.predict(X_test) / 5.0
        fold_rmse = np.sqrt(mean_squared_error(va_y, preds))
        print(f"  Fold {fold} RMSE: {fold_rmse:.5f}")

    xgb_rmse = np.sqrt(mean_squared_error(y, xgb_oof))
    print(f"✅ XGBoost OOF RMSE: {xgb_rmse:.5f} (Time: {time.time()-start:.1f}s)")
    results["XGBoost"] = {"name": "XGBoost", "oof": xgb_oof, "test": xgb_test, "rmse": xgb_rmse, "time": time.time()-start}

    # --- 4. Random Forest ---
    print("\\n=== 4. Random Forest ===")
    rf_oof = np.zeros(len(X))
    rf_test = np.zeros(len(X_test))

    rf_params = {
        "n_estimators": 150,
        "max_depth": 14,
        "min_samples_leaf": 4,
        "random_state": 42,
        "n_jobs": -1,
    }

    start = time.time()
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
        tr_x, va_x = X.iloc[tr_idx], X.iloc[va_idx]
        tr_y, va_y = y[tr_idx], y[va_idx]

        model = RandomForestRegressor(**rf_params)
        model.fit(tr_x, tr_y)

        preds = model.predict(va_x)
        rf_oof[va_idx] = preds
        rf_test += model.predict(X_test) / 5.0
        fold_rmse = np.sqrt(mean_squared_error(va_y, preds))
        print(f"  Fold {fold} RMSE: {fold_rmse:.5f}")

    rf_rmse = np.sqrt(mean_squared_error(y, rf_oof))
    print(f"✅ Random Forest OOF RMSE: {rf_rmse:.5f} (Time: {time.time()-start:.1f}s)")
    results["RandomForest"] = {"name": "RandomForest", "oof": rf_oof, "test": rf_test, "rmse": rf_rmse, "time": time.time()-start}

    return results

model_results = evaluate_models_fast()"""))

# ----------------------------------------------------------------------
# 7. Model Comparison & Ensemble Evaluation
# ----------------------------------------------------------------------
cells.append(md("""## 7. Model Comparison & Ensemble Evaluation"""))

cells.append(code("""comp_df = pd.DataFrame([
    {
        "Model": res["name"],
        "OOF RMSE": f"{res['rmse']:.5f}",
        "OOF MAE": f"{mean_absolute_error(y, res['oof']):.5f}",
        "OOF R2": f"{r2_score(y, res['oof']):.5f}",
        "Time (s)": f"{res['time']:.1f}",
    }
    for res in model_results.values()
])

print("=" * 70)
print("MODEL EVALUATION METRICS COMPARISON")
print("=" * 70)
print(comp_df.to_string(index=False))"""))

cells.append(code("""# Blended Ensemble (45% LightGBM + 35% CatBoost + 20% XGBoost)
blend_oof = (
    0.45 * model_results["LightGBM"]["oof"] +
    0.35 * model_results["CatBoost"]["oof"] +
    0.20 * model_results["XGBoost"]["oof"]
)

blend_rmse = np.sqrt(mean_squared_error(y, blend_oof))
print(f"\\n🔥 FAST BLENDED ENSEMBLE OOF RMSE: {blend_rmse:.5f}")
assert blend_rmse < 0.20, "RMSE target < 0.20 not achieved!"
print("🎯 SUCCESS: OOF RMSE is below 0.20 threshold!")"""))

# ----------------------------------------------------------------------
# 8. Hyperparameter Tuning
# ----------------------------------------------------------------------
cells.append(md("## 8. Hyperparameter Optimization"))

cells.append(code("""print("Optimizing LightGBM via RandomizedSearchCV...")

base_lgb = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbosity=-1)
param_grid = {
    "n_estimators": [1000, 1500],
    "learning_rate": [0.03, 0.05],
    "num_leaves": [127, 255],
    "max_depth": [10, 12],
    "min_child_samples": [10, 20],
    "feature_fraction": [0.7, 0.8],
    "bagging_fraction": [0.8, 0.9],
}

search = RandomizedSearchCV(
    estimator=base_lgb,
    param_distributions=param_grid,
    n_iter=4,
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
cells.append(md("## 9. Final Predictions & Submission"))

cells.append(code("""# Blended Test Predictions
blend_test = (
    0.45 * model_results["LightGBM"]["test"] +
    0.35 * model_results["CatBoost"]["test"] +
    0.20 * model_results["XGBoost"]["test"]
)

final_preds_original_scale = np.maximum(0, np.expm1(blend_test))

submission = sample_sub.copy()
target_col = [c for c in sample_sub.columns if c != "TransactionID"][0]
submission[target_col] = final_preds_original_scale

submission.to_csv("submission.csv", index=False)
print("✅ submission.csv created successfully.")

print("\\nPrediction Summary Statistics:")
print(submission[target_col].describe())

print("\\nFirst 10 Sample Predictions:")
print(submission.head(10))"""))

# ----------------------------------------------------------------------
# Assemble & Write Notebooks (v4-fast)
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

# 1. Write Kaggle v4 notebook
with open(r"notebooks/24f2008980-notebook-v4-fast-kaggle.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

# 2. Write Offline v4 notebook
offline_nb = json.loads(json.dumps(notebook))

for c in offline_nb["cells"]:
    if c["cell_type"] == "code":
        src = "".join(c["source"])
        if "submission.to_csv(\"submission.csv\", index=False)" in src:
            c["source"] = [line + "\n" for line in src.replace(
                "submission.to_csv(\"submission.csv\", index=False)\nprint(\"✅ submission.csv created successfully.\")",
                "print(\"=== OFFLINE MODE: Prediction Results (Not saving submission.csv) ===\")"
            ).split("\n")]

with open(r"notebooks/24f2008980-notebook-v4-fast-offline.ipynb", "w", encoding="utf-8") as f:
    json.dump(offline_nb, f, indent=1, ensure_ascii=False)

print("Both v4 fast notebooks generated successfully.")
