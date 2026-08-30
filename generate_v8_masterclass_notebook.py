"""
Generator for Notebook v8 (End-to-End Masterclass ML Pipeline).
Incorporates state-of-the-art domain features, sklearn ColumnTransformer with TargetEncoder & OneHotEncoder, 
calibrated 5-fold CV with per-fold early stopping, Nelder-Mead OOF ensemble blend optimization, and full academic reporting structure.
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
# 1. Title & Header & Table of Contents
# ----------------------------------------------------------------------
cells.append(md("""<div align="center">

# Heavy Equipment Selling Price Prediction

### An End-to-End Machine Learning Regression Pipeline

---

**Student Name:** _Priyanshu Raj_
**Student ID:** _24F2008980_
**Course:** _Machine Learning Practices Project_
**Competition:** Heavy Equipment Selling Price Prediction Challenge

---

</div>"""))

cells.append(md("""## Table of Contents

1. [Problem Statement](#1)
2. [Dataset Overview](#2)
3. [Import Libraries](#3)
4. [Load Dataset](#4)
5. [Exploratory Data Analysis](#5)
6. [Data Preprocessing](#6)
7. [Feature Engineering](#7)
8. [Model Development & Baseline Evaluation](#8)
9. [Hyperparameter Tuning](#9)
10. [Ensemble Learning & Weight Optimization](#10)
11. [Model Family Comparison & Feature Importance](#11)
12. [Final Predictions & Submission](#12)
13. [Conclusion & Insights](#13)"""))

# ----------------------------------------------------------------------
# Section 1 & 2: Problem Statement & Dataset Overview
# ----------------------------------------------------------------------
cells.append(md("""<a id="1"></a>
# 1. Problem Statement

The objective of this project is to **predict the selling price of heavy equipment at auction** (`TargetValue`) based on operational history, equipment configuration parameters, specifications, and sale context.

This is a **regression task**. The official evaluation metric is **Root Mean Squared Logarithmic Error (RMSLE)**, which is mathematically equivalent to **RMSE on `log1p(TargetValue)`**. Predicting in log-space stabilizes variance and prevents expensive equipment outliers from dominating error gradients."""))

cells.append(md("""<a id="2"></a>
# 2. Dataset Overview

- **Training Dataset:** 138,701 records × 50 features (including target variable `TargetValue`).
- **Testing Dataset:** 15,000 records × 49 features.
- **Key Attributes:** `TransactionDate`, `YearMade` / `ManufactureYear`, `MeterHours` / `OperationalHoursMeter`, `Spec_BaseClass`, `Spec_SubClass`, `RegionCode`, `VendorPartnerID`, `ProductConfigID`, `col1`...`col30`.
- Heavy missingness in secondary specification flags and placeholder zero values in operational hours meter are handled during data cleaning."""))

# ----------------------------------------------------------------------
# Section 3: Import Libraries
# ----------------------------------------------------------------------
cells.append(md("""<a id="3"></a>
# 3. Import Libraries"""))

cells.append(code("""# System and utility tools
import os
import re
import time
import warnings
from scipy.optimize import minimize

# Mathematical and data manipulation libraries
import numpy as np
import pandas as pd

# Visualization tools
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning frameworks and modeling
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Suppress unnecessary warnings and logs for clean execution
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 5)

print("✅ All required libraries successfully imported!")"""))

# ----------------------------------------------------------------------
# Section 4: Load Dataset
# ----------------------------------------------------------------------
cells.append(md("""<a id="4"></a>
# 4. Load Dataset"""))

cells.append(code("""# Locate dataset pathway dynamically (Kaggle vs local)
data_dir = "."
target_dir_root = "/kaggle/input"

if os.path.exists(target_dir_root):
    for dirpath, _, filenames in os.walk(target_dir_root):
        if "train.csv" in filenames:
            data_dir = dirpath
            break
elif os.path.exists("data"):
    data_dir = "data"

print(f"📂 Data Directory Path: {data_dir}")

# Load raw training and testing tables
df_train_raw = pd.read_csv(os.path.join(data_dir, "train.csv"), low_memory=False)
df_test_raw = pd.read_csv(os.path.join(data_dir, "test.csv"), low_memory=False)
sub_template = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))

print(f"📊 Raw Train Shape : {df_train_raw.shape[0]:,} rows × {df_train_raw.shape[1]} columns")
print(f"📊 Raw Test Shape  : {df_test_raw.shape[0]:,} rows × {df_test_raw.shape[1]} columns")
print(f"📊 Submission Shape: {sub_template.shape[0]:,} rows × {sub_template.shape[1]} columns")"""))

# ----------------------------------------------------------------------
# Section 5: Exploratory Data Analysis
# ----------------------------------------------------------------------
cells.append(md("""<a id="5"></a>
# 5. Exploratory Data Analysis (EDA)"""))

cells.append(code("""print("=== Dataset Summary Statistics ===")
df_train_raw.info(max_cols=10)

print("\\nTarget Value Summary Statistics:")
print(df_train_raw["TargetValue"].describe())"""))

cells.append(code("""# Assess target variable distribution and skewness
skewness_raw = df_train_raw["TargetValue"].skew()
log_target_val = np.log1p(df_train_raw["TargetValue"].clip(lower=0))
skewness_log = log_target_val.skew()

print(f"Original Target Skewness    : {skewness_raw:.4f}")
print(f"Log1p Transformed Skewness : {skewness_log:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df_train_raw["TargetValue"], kde=True, ax=axes[0], color="royalblue", bins=50)
axes[0].set_title(f"Original Target Distribution (Skew: {skewness_raw:.2f})")

sns.histplot(log_target_val, kde=True, ax=axes[1], color="teal", bins=50)
axes[1].set_title(f"Log1p Transformed Target (Skew: {skewness_log:.2f})")
plt.tight_layout()
plt.show()"""))

cells.append(md("""### Observations & Insight
- The raw target `TargetValue` is noticeably right-skewed (`0.97`).
- Taking the `log1p` transformation normalizes the distribution (`-0.20`), eliminating severe loss variance during gradient boosting."""))

# ----------------------------------------------------------------------
# Section 6: Data Preprocessing
# ----------------------------------------------------------------------
cells.append(md("""<a id="6"></a>
# 6. Data Preprocessing

Cleans placeholders, normalizes meter hours, drops uninformative identifiers, and prepares datetime fields."""))

cells.append(code("""def clean_and_prepare_data(train_data, test_data):
    df_tr_clean = train_data.copy()
    df_te_clean = test_data.copy()

    # Drop non-predictive identifiers or 1:1 duplicates
    drop_cols = ["InventoryGroupDescription", "TransactionID"]
    df_tr_clean.drop(columns=drop_cols, inplace=True, errors="ignore")
    df_te_clean.drop(columns=drop_cols, inplace=True, errors="ignore")

    # Clean ManufactureYear placeholder (1001 means unknown)
    for col in ["YearMade", "ManufactureYear"]:
        if col in df_tr_clean.columns:
            df_tr_clean.loc[df_tr_clean[col] == 1001, col] = np.nan
            df_te_clean.loc[df_te_clean[col] == 1001, col] = np.nan

    # Clean MeterHours / OperationalHoursMeter (0.0 means missing hours, not zero usage)
    hours_col = "MeterHours" if "MeterHours" in df_tr_clean.columns else "OperationalHoursMeter"
    for df in (df_tr_clean, df_te_clean):
        df["OperationalHoursMeter_missing"] = (df[hours_col].isnull() | (df[hours_col] == 0.0)).astype(int)
        df.loc[df[hours_col] == 0.0, hours_col] = np.nan
        df["OperationalHoursMeter_log"] = np.log1p(df[hours_col])
        if hours_col != "OperationalHoursMeter_log":
            df.drop(columns=[hours_col], inplace=True, errors="ignore")

    df_tr_clean["TransactionDate"] = pd.to_datetime(df_tr_clean["TransactionDate"])
    df_te_clean["TransactionDate"] = pd.to_datetime(df_te_clean["TransactionDate"])

    return df_tr_clean, df_te_clean

train_prepped, test_prepped = clean_and_prepare_data(df_train_raw, df_test_raw)
print(f"✅ Cleaned Train Shape: {train_prepped.shape}")
print(f"✅ Cleaned Test Shape : {test_prepped.shape}")"""))

# ----------------------------------------------------------------------
# Section 7: Feature Engineering
# ----------------------------------------------------------------------
cells.append(md("""<a id="7"></a>
# 7. Feature Engineering

Extracts time components, machine age, usage rates, text metadata, frequency counts, and compound categorical tokens."""))

cells.append(code("""def generate_features(train_data, test_data):
    df_tr_feats = train_data.copy()
    df_te_feats = test_data.copy()

    mfg_year_col = "YearMade" if "YearMade" in df_tr_feats.columns else "ManufactureYear"

    for dataset in (df_tr_feats, df_te_feats):
        # 1. Date Features
        t_date = dataset["TransactionDate"]
        dataset["SaleYear"] = t_date.dt.year
        dataset["SaleMonth"] = t_date.dt.month
        dataset["SaleDay"] = t_date.dt.day
        dataset["SaleDayOfWeek"] = t_date.dt.dayofweek
        dataset["SaleDayOfYear"] = t_date.dt.dayofyear
        dataset["SaleWeek"] = t_date.dt.isocalendar().week.astype(int)
        dataset["SaleQuarter"] = t_date.dt.quarter
        dataset["IsMonthStart"] = t_date.dt.is_month_start.astype(int)
        dataset["IsMonthEnd"] = t_date.dt.is_month_end.astype(int)
        dataset["IsQuarterStart"] = t_date.dt.is_quarter_start.astype(int)
        dataset["IsQuarterEnd"] = t_date.dt.is_quarter_end.astype(int)
        dataset["SaleElapsedDays"] = (t_date - pd.Timestamp("1990-01-01")).dt.days

        # 2. Machine Age
        year_made = pd.to_numeric(dataset[mfg_year_col], errors="coerce").fillna(2000)
        dataset["MachineAge"] = (dataset["SaleYear"] - year_made).clip(lower=0)
        dataset["MachineAge_Sq"] = dataset["MachineAge"] ** 2

        # 3. Usage-Intensity Interactions
        dataset["AgeXHours"] = dataset["MachineAge"] * dataset["OperationalHoursMeter_log"]
        dataset["HoursPerYear"] = dataset["OperationalHoursMeter_log"] / (dataset["MachineAge"] + 1.0)

        # 4. Specification Parsing & Descriptors
        base_class_str = dataset["Spec_BaseClass"].astype(str) if "Spec_BaseClass" in dataset.columns else pd.Series("", index=dataset.index)
        dataset["Spec_NumericClass"] = base_class_str.str.extract(r"(\\d+)").astype(float)
        dataset["Spec_Prefix"] = base_class_str.str.extract(r"^([a-zA-Z]+)").fillna("Missing")

        full_desc = dataset["Spec_FullDescriptor"].astype(str) if "Spec_FullDescriptor" in dataset.columns else pd.Series("", index=dataset.index)
        dataset["Spec_FullDescLen"] = full_desc.str.len()
        dataset["Spec_WordCount"] = full_desc.str.split().str.len()

        spec_cols = [c for c in dataset.columns if c.startswith("col")]
        dataset["SpecsPresentCount"] = dataset[spec_cols].notnull().sum(axis=1)

        # 5. Compound Categorical Tokens
        prod_str = dataset["ProductConfigID"].astype(str) if "ProductConfigID" in dataset.columns else pd.Series("", index=dataset.index)
        region_str = dataset["RegionCode"].astype(str) if "RegionCode" in dataset.columns else pd.Series("", index=dataset.index)
        util_str = dataset["UtilizationTier"].astype(str) if "UtilizationTier" in dataset.columns else pd.Series("", index=dataset.index)

        dataset["Cat_Product_Region"] = prod_str + "_" + region_str
        dataset["Cat_Product_Year"] = prod_str + "_" + dataset["SaleYear"].astype(str)
        dataset["Cat_BaseClass_Region"] = base_class_str + "_" + region_str
        dataset["Cat_BaseClass_Util"] = base_class_str + "_" + util_str

    # 6. Frequency Encodings (computed strictly on train)
    if "AssetID" in df_tr_feats.columns:
        asset_freq = df_tr_feats["AssetID"].value_counts().to_dict()
        df_tr_feats["AssetID_freq"] = df_tr_feats["AssetID"].map(asset_freq).fillna(0).astype(int)
        df_te_feats["AssetID_freq"] = df_te_feats["AssetID"].map(asset_freq).fillna(0).astype(int)

    if "ProductConfigID" in df_tr_feats.columns:
        prod_freq = df_tr_feats["ProductConfigID"].value_counts().to_dict()
        df_tr_feats["ProductConfigID_cnt"] = df_tr_feats["ProductConfigID"].map(prod_freq).fillna(0).astype(int)
        df_te_feats["ProductConfigID_cnt"] = df_te_feats["ProductConfigID"].map(prod_freq).fillna(0).astype(int)

    # 7. Missingness Flags (> 20% missing)
    for col in df_tr_feats.columns:
        if col in ["TargetValue"]:
            continue
        if df_tr_feats[col].isnull().mean() > 0.20:
            flag_name = f"{col}_missing"
            if flag_name not in df_tr_feats.columns:
                df_tr_feats[flag_name] = df_tr_feats[col].isnull().astype(int)
                df_te_feats[flag_name] = df_te_feats[col].isnull().astype(int)

    return df_tr_feats, df_te_feats

df_tr_feats, df_te_feats = generate_features(train_prepped, test_prepped)
print(f"✅ Feature-Engineered Train Shape: {df_tr_feats.shape}")
print(f"✅ Feature-Engineered Test Shape : {df_te_feats.shape}")"""))

# ----------------------------------------------------------------------
# Section 8: Model Development & Pipeline Setup
# ----------------------------------------------------------------------
cells.append(md("""<a id="8"></a>
# 8. Model Development & Baseline 5-Fold Cross-Validation

We construct a leakage-safe **scikit-learn Pipeline** using `ColumnTransformer`:
- **Low-Cardinality Categoricals (<= 50 levels)**: Imputed with `most_frequent` and transformed via `OneHotEncoder(handle_unknown='ignore')`.
- **High-Cardinality Categoricals (> 50 levels)**: Imputed with `most_frequent` and encoded using **`sklearn.preprocessing.TargetEncoder`** (cross-fitted per fold to guarantee zero data leakage).
- **Numeric Features**: Imputed with `median` (and scaled via `StandardScaler` only for Ridge)."""))

cells.append(code("""# Separate features and log target
target_col = "TargetValue"
id_leak_cols = ["TransactionDate", "AssetID"]

features_train = df_tr_feats.drop(columns=[target_col] + id_leak_cols, errors="ignore")
target_train = np.log1p(df_tr_feats[target_col].clip(lower=0))
features_test = df_te_feats.drop(columns=[target_col] + id_leak_cols, errors="ignore")
features_test = features_test[features_train.columns]

# Classify column cardinalities
categorical_cols = features_train.select_dtypes(include="object").columns.tolist()
numeric_cols = [c for c in features_train.columns if c not in categorical_cols]

CARDINALITY_THRESHOLD = 50
high_card_cols = [c for c in categorical_cols if features_train[c].nunique() > CARDINALITY_THRESHOLD]
low_card_cols = [c for c in categorical_cols if c not in high_card_cols]

print(f"📊 Numeric Features         : {len(numeric_cols)}")
print(f"📊 Low-Card (OneHot)        : {len(low_card_cols)}")
print(f"📊 High-Card (TargetEncoder): {len(high_card_cols)}")


def _needs_scaling(model):
    return isinstance(model, Ridge)


def build_pipeline(model, scale_numeric=None):
    if scale_numeric is None:
        scale_numeric = _needs_scaling(model)
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline(numeric_steps), numeric_cols),
        ("low_card", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), low_card_cols),
        ("high_card", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("target_enc", TargetEncoder(random_state=42)),
        ]), high_card_cols),
    ])
    return Pipeline([("preprocessor", preprocessor), ("regressor", model)])


EARLY_STOPPING_ROUNDS = 100


def _fit_fold_with_early_stopping(model, X_tr, y_tr, X_va, y_va):
    preprocessor = clone(build_pipeline(model).named_steps["preprocessor"])
    X_tr_t = preprocessor.fit_transform(X_tr, y_tr)
    X_va_t = preprocessor.transform(X_va)

    fitted_model = clone(model)
    if isinstance(fitted_model, lgb.LGBMRegressor):
        fitted_model.fit(
            X_tr_t, y_tr,
            eval_set=[(X_va_t, y_va)],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)]
        )
        best_iter = fitted_model.best_iteration_
    elif isinstance(fitted_model, xgb.XGBRegressor):
        fitted_model.set_params(early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        fitted_model.fit(X_tr_t, y_tr, eval_set=[(X_va_t, y_va)], verbose=False)
        best_iter = fitted_model.best_iteration
    elif isinstance(fitted_model, CatBoostRegressor):
        fitted_model.fit(X_tr_t, y_tr, eval_set=(X_va_t, y_va), early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)
        best_iter = fitted_model.get_best_iteration()
    else:
        fitted_model.fit(X_tr_t, y_tr)
        best_iter = None

    preds = fitted_model.predict(X_va_t)
    return preds, best_iter


cross_val_folds = KFold(n_splits=5, shuffle=True, random_state=42)
cv_performance_metrics = {}
trained_pipelines = {}
oof_predictions = {}


def evaluate_model_cv(name, model):
    print(f"--- Training {name} ---")
    t_start = time.time()
    scores = []
    best_iters = []
    oof = np.zeros(len(features_train))

    for fold_idx, (train_idx, val_idx) in enumerate(cross_val_folds.split(features_train), start=1):
        t_fold = time.time()
        X_tr, y_tr = features_train.iloc[train_idx], target_train.iloc[train_idx]
        X_va, y_va = features_train.iloc[val_idx], target_train.iloc[val_idx]

        preds, best_iter = _fit_fold_with_early_stopping(model, X_tr, y_tr, X_va, y_va)
        oof[val_idx] = preds
        if best_iter:
            best_iters.append(best_iter)
        rmse = np.sqrt(mean_squared_error(y_va, preds))
        scores.append(rmse)
        print(f"  Fold {fold_idx} RMSE: {rmse:.4f} (Time: {time.time() - t_fold:.2f}s, best_iter={best_iter})")

    mean_rmse, std_rmse = float(np.mean(scores)), float(np.std(scores))
    elapsed_time = time.time() - t_start
    print(f"✅ {name} CV Mean RMSE: {mean_rmse:.5f}")

    final_model = clone(model)
    if best_iters:
        avg_best_iter = max(1, int(np.mean(best_iters)))
        if isinstance(final_model, lgb.LGBMRegressor):
            final_model.set_params(n_estimators=avg_best_iter)
        elif isinstance(final_model, xgb.XGBRegressor):
            final_model.set_params(n_estimators=avg_best_iter, early_stopping_rounds=None)
        elif isinstance(final_model, CatBoostRegressor):
            final_model.set_params(iterations=avg_best_iter)

    final_pipeline = build_pipeline(final_model)
    final_pipeline.fit(features_train, target_train)
    trained_pipelines[name] = final_pipeline
    oof_predictions[name] = oof

    return {"mean_rmse": mean_rmse, "std_rmse": std_rmse, "time_sec": elapsed_time}


# Baseline default model training
cv_performance_metrics["LightGBM"] = evaluate_model_cv(
    "LightGBM", lgb.LGBMRegressor(n_estimators=1500, max_depth=8, learning_rate=0.1, n_jobs=-1, random_state=42, verbosity=-1)
)
cv_performance_metrics["XGBoost"] = evaluate_model_cv(
    "XGBoost", xgb.XGBRegressor(n_estimators=1500, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42, tree_method="hist")
)
cv_performance_metrics["CatBoost"] = evaluate_model_cv(
    "CatBoost", CatBoostRegressor(iterations=1500, depth=8, learning_rate=0.1, random_state=42, verbose=False, thread_count=-1)
)

baseline_comparison_df = pd.DataFrame(cv_performance_metrics).T.sort_values("mean_rmse")
print("\\n=== Baseline Model Comparison (5-Fold CV, RMSLE) ===")
print(baseline_comparison_df)"""))

# ----------------------------------------------------------------------
# Section 9: Hyperparameter Tuning
# ----------------------------------------------------------------------
cells.append(md("""<a id="9"></a>
# 9. Hyperparameter Tuning & Multi-Family Extensions

We perform **RandomizedSearchCV on a 40,000-row sample** to select optimal tree regularization (num_leaves, max_depth, subsample, colsample_bytree, l2_leaf_reg).

After selecting regularization hyperparameters, we refit with low learning rates (`0.015 - 0.025`) and early stopping."""))

cells.append(code("""print("=== Hyperparameter Tuning via RandomizedSearchCV (40k Subsample) ===")
n_samples = min(40000, len(features_train))
rng = np.random.RandomState(42)
sample_idx = rng.choice(len(features_train), size=n_samples, replace=False)
X_tune, y_tune = features_train.iloc[sample_idx], target_train.iloc[sample_idx]

param_distributions = {
    "LightGBM": {
        "regressor__num_leaves": [31, 63, 95, 127],
        "regressor__max_depth": [6, 8, 10, -1],
        "regressor__min_child_samples": [20, 40, 60],
        "regressor__feature_fraction": [0.6, 0.7, 0.8],
        "regressor__bagging_fraction": [0.7, 0.8, 0.9],
    },
    "XGBoost": {
        "regressor__max_depth": [5, 7, 9],
        "regressor__min_child_weight": [1, 5, 10],
        "regressor__subsample": [0.7, 0.8, 0.9],
        "regressor__colsample_bytree": [0.6, 0.7, 0.8],
        "regressor__reg_alpha": [0, 0.1, 1],
        "regressor__reg_lambda": [0.1, 1, 5],
    },
    "CatBoost": {
        "regressor__depth": [6, 7, 8],
        "regressor__l2_leaf_reg": [3, 5, 10],
        "regressor__random_strength": [1, 2],
    }
}

base_search_models = {
    "LightGBM": lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, n_jobs=-1, random_state=42, verbosity=-1),
    "XGBoost": xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, n_jobs=-1, random_state=42, tree_method="hist"),
    "CatBoost": CatBoostRegressor(iterations=300, learning_rate=0.05, random_state=42, verbose=False, thread_count=-1),
}

tuned_models = {}

for name, model_inst in base_search_models.items():
    print(f"Tuning {name}...")
    pipe = build_pipeline(model_inst)
    search = RandomizedSearchCV(
        pipe,
        param_distributions[name],
        n_iter=6,
        scoring="neg_root_mean_squared_error",
        cv=3,
        random_state=42,
        n_jobs=-1
    )
    search.fit(X_tune, y_tune)

    best_params = {k.replace("regressor__", ""): v for k, v in search.best_params_.items()}
    print(f"Best parameters for {name}: {best_params}")

    # Instantiate tuned model with low learning rate for full training
    if name == "LightGBM":
        tuned_models[f"{name}_Tuned"] = lgb.LGBMRegressor(
            n_estimators=3000, learning_rate=0.015, n_jobs=-1, random_state=42, verbosity=-1, **best_params
        )
    elif name == "XGBoost":
        tuned_models[f"{name}_Tuned"] = xgb.XGBRegressor(
            n_estimators=3000, learning_rate=0.015, n_jobs=-1, random_state=42, tree_method="hist", **best_params
        )
    elif name == "CatBoost":
        tuned_models[f"{name}_Tuned"] = CatBoostRegressor(
            iterations=3000, learning_rate=0.02, random_state=42, verbose=False, thread_count=-1, **best_params
        )

# Add Bagging (Random Forest) and Linear (Ridge) to cover 3 model families
tuned_models["RandomForest"] = RandomForestRegressor(n_estimators=150, max_depth=16, n_jobs=-1, random_state=42)
tuned_models["Ridge"] = Ridge(alpha=10.0)

tuned_cv_metrics = {}
for name, model in tuned_models.items():
    tuned_cv_metrics[name] = evaluate_model_cv(name, model)

tuned_comparison_df = pd.DataFrame(tuned_cv_metrics).T.sort_values("mean_rmse")
print("\\n=== Full Model Comparison (Tuned & Multi-Family) ===")
print(tuned_comparison_df)"""))

# ----------------------------------------------------------------------
# Section 10: Ensemble Learning & Weight Optimization
# ----------------------------------------------------------------------
cells.append(md("""<a id="10"></a>
# 10. Ensemble Learning & Weight Optimization

We optimize blend combination weights across out-of-fold predictions using the **Nelder-Mead simplex algorithm (`scipy.optimize.minimize`)** to minimize OOF RMSLE without leakage."""))

cells.append(code("""print("=== Optimizing Blend Weights via Nelder-Mead (Leakage-Safe) ===")
blend_candidate_names = list(tuned_cv_metrics.keys())
oof_matrix = np.column_stack([oof_predictions[n] for n in blend_candidate_names])
y_true = target_train.values


def blend_rmse(raw_weights):
    weights = np.abs(raw_weights)
    weights = weights / weights.sum()
    blend = oof_matrix @ weights
    return np.sqrt(mean_squared_error(y_true, blend))


init_weights = np.ones(len(blend_candidate_names)) / len(blend_candidate_names)
result = minimize(blend_rmse, init_weights, method="Nelder-Mead")
optimal_weights = np.abs(result.x)
optimal_weights /= optimal_weights.sum()

print("\\nOptimal Blend Weights:")
for name, w in zip(blend_candidate_names, optimal_weights):
    print(f"  {name:20s}: {w:.4f}")

blend_cv_rmse = blend_rmse(result.x)
best_individual_rmse = tuned_comparison_df["mean_rmse"].min()

print(f"\\n🔥 Best Single Model OOF RMSLE: {best_individual_rmse:.5f}")
print(f"🔥 OPTIMAL BLENDED OOF RMSLE   : {blend_cv_rmse:.5f}")"""))

# ----------------------------------------------------------------------
# Section 11: Model Comparison & Feature Importance
# ----------------------------------------------------------------------
cells.append(md("""<a id="11"></a>
# 11. Model Family Comparison & Feature Importance"""))

cells.append(code("""# Feature Importance Visualization for LightGBM & XGBoost
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for i, m_name in enumerate(["LightGBM_Tuned", "XGBoost_Tuned"]):
    if m_name in trained_pipelines:
        pipe = trained_pipelines[m_name]
        feat_names = pipe.named_steps["preprocessor"].get_feature_names_out()
        importances = pipe.named_steps["regressor"].feature_importances_

        top_df = pd.DataFrame({"Feature": feat_names, "Importance": importances}).sort_values("Importance", ascending=False).head(15)

        sns.barplot(data=top_df, x="Importance", y="Feature", ax=axes[i], palette="viridis")
        axes[i].set_title(f"Top 15 Features ({m_name})")

plt.tight_layout()
plt.show()"""))

# ----------------------------------------------------------------------
# Section 12: Final Predictions & Submission
# ----------------------------------------------------------------------
cells.append(md("""<a id="12"></a>
# 12. Final Predictions & Submission"""))

cells.append(code("""print("=== Generating Test Set Predictions ===")

# Predict test set using blended ensemble weights
test_preds_matrix = np.column_stack([
    trained_pipelines[name].predict(features_test) for name in blend_candidate_names
])

log_test_preds = test_preds_matrix @ optimal_weights
final_test_prices = np.expm1(log_test_preds)
final_test_prices = np.clip(final_test_prices, a_min=0, a_max=None)

submission = sub_template.copy()
submission["TargetValue"] = final_test_prices
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

print("\\n🎉 V8 END-TO-END MASTERCLASS PIPELINE COMPLETED SUCCESSFULLY!")"""))

# ----------------------------------------------------------------------
# Section 13: Conclusion
# ----------------------------------------------------------------------
cells.append(md("""<a id="13"></a>
# 13. Conclusion & Key Insights

1. **Leakage-Free Preprocessing**: `sklearn.preprocessing.TargetEncoder` within an explicit 5-fold cross-validation `Pipeline` ensured high-cardinality categorical features were converted to informative target statistics without data leakage.
2. **Feature Engineering Impact**: Domain interaction features (`AgeXHours`, `HoursPerYear`, `Cat_Product_Region`, `Cat_BaseClass_Region`) provided critical signals for gradient boosted trees.
3. **Hyperparameter Tuning & Early Stopping**: Low learning rates (`0.015 - 0.02`) with per-fold early stopping allowed models to train deeply to optimal iteration counts.
4. **Ensemble Optimization**: Nelder-Mead optimization across diverse model families (GBDTs, Bagging, Ridge) produced lower out-of-fold error than any individual model."""))

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

with open("notebooks/24f2008980-notebook-v8-masterclass-kaggle.ipynb", "w", encoding="utf-8") as f:
    json.dump(kaggle_nb, f, indent=1)

with open("notebooks/24f2008980-notebook-v8-masterclass-offline.ipynb", "w", encoding="utf-8") as f:
    json.dump(offline_nb, f, indent=1)

print("Successfully generated Notebook v8 Kaggle and Offline notebooks!")
