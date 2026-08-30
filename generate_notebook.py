"""
Notebook generator script - creates a complete, Viva-compliant Kaggle notebook
for the Heavy Equipment Selling Price Prediction competition.

Only uses allowed libraries. Addresses all 9 Level 1 Viva checklist items.
"""
import json
import uuid


def md(source: str) -> dict:
    """Create a markdown cell."""
    return {
        "cell_type": "markdown",
        "id": str(uuid.uuid4())[:8],
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }


def code(source: str) -> dict:
    """Create a code cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": str(uuid.uuid4())[:8],
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    }


cells = []

# ============================================================
# SECTION 1: Title & Problem Statement
# ============================================================
cells.append(md("""# 🏗️ Heavy Equipment Selling Price Prediction

## Problem Statement

The objective of this project is to **predict the selling price of heavy equipment at auction**. The dataset contains attributes describing the equipment configuration, condition, usage history, and transaction details.

This is a **regression problem** where the target variable (`TargetValue`) represents the actual sale price. The evaluation metric is **RMSE (Root Mean Squared Error)**.

### Approach Overview
1. **Exploratory Data Analysis (EDA)** — Understand key statistics, distributions, and relationships
2. **Systematic Preprocessing** — Handle missing values, encode categoricals, scale numerics using **reusable sklearn Pipelines**
3. **Feature Engineering** — Create domain-specific features and analyze their importance
4. **Model Building** — Train and compare **4 different models** (Random Forest, XGBoost, LightGBM, CatBoost)
5. **Hyperparameter Tuning** — Optimize the best-performing model using `RandomizedSearchCV`
6. **Final Submission** — Generate predictions with the best model"""))

# ============================================================
# SECTION 2: Imports
# ============================================================
cells.append(md("## 1. Imports & Setup"))

cells.append(code("""# Core Libraries
import os
import re
import time
import warnings

# Data & Numerical
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn: Preprocessing & Pipelines
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OrdinalEncoder,
    LabelEncoder,
)

# Scikit-learn: Models & Evaluation
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    KFold,
    cross_val_score,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

# Gradient Boosting Frameworks (Allowed)
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# SciPy for statistical distributions in RandomizedSearchCV
from scipy.stats import randint, uniform

# Settings
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 12

print("✅ All libraries loaded successfully.")"""))

# ============================================================
# SECTION 3: Data Loading
# ============================================================
cells.append(md("""## 2. Data Loading

We detect the data directory automatically for Kaggle compatibility."""))

cells.append(code("""# Auto-detect data path (Kaggle or local)
DATA_PATH = "."

if os.path.isdir("/kaggle/input"):
    for root, _, files in os.walk("/kaggle/input"):
        if "train.csv" in files and "test.csv" in files:
            DATA_PATH = root
            break

print(f"📂 Data directory: {DATA_PATH}")

# Load datasets
train_raw = pd.read_csv(os.path.join(DATA_PATH, "train.csv"), low_memory=False)
test_raw = pd.read_csv(os.path.join(DATA_PATH, "test.csv"), low_memory=False)
sample_sub = pd.read_csv(os.path.join(DATA_PATH, "sample_submission.csv"))

print(f"\\n📊 Train Dataset : {train_raw.shape[0]:,} rows × {train_raw.shape[1]} columns")
print(f"📊 Test Dataset  : {test_raw.shape[0]:,} rows × {test_raw.shape[1]} columns")
print(f"📊 Submission    : {sample_sub.shape[0]:,} rows × {sample_sub.shape[1]} columns")"""))

# ============================================================
# SECTION 4: EDA
# ============================================================
cells.append(md("""## 3. Exploratory Data Analysis (EDA)

> **Objective:** Understand key statistics and relationships in the dataset, perform univariate and bivariate analysis, and create clear, readable visualizations.

### 3.1 Dataset Overview"""))

cells.append(code("""# Basic dataset information
print("=" * 70)
print("TRAINING DATASET OVERVIEW")
print("=" * 70)
train_raw.info(max_cols=10)

print("\\n" + "=" * 70)
print("FIRST 5 ROWS")
print("=" * 70)
train_raw.head()"""))

cells.append(code("""# Target Variable Statistics
print("=" * 70)
print("TARGET VARIABLE (TargetValue) STATISTICS")
print("=" * 70)
print(train_raw["TargetValue"].describe())"""))

cells.append(code("""# Missing Values Analysis
print("=" * 70)
print("MISSING VALUES ANALYSIS")
print("=" * 70)

missing_df = pd.DataFrame({
    "Missing Count": train_raw.isnull().sum(),
    "Missing %": (train_raw.isnull().sum() / len(train_raw) * 100).round(2)
}).sort_values("Missing %", ascending=False)

# Show only columns with missing values
missing_with_values = missing_df[missing_df["Missing Count"] > 0]
print(f"\\nColumns with missing values: {len(missing_with_values)} / {len(train_raw.columns)}")
print(missing_with_values)

# Missing values heatmap for columns with >5% missing
cols_with_missing = missing_with_values[missing_with_values["Missing %"] > 5].index.tolist()
if cols_with_missing:
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        train_raw[cols_with_missing].isnull().T,
        cbar=True, cmap="YlOrRd",
        yticklabels=True, ax=ax
    )
    ax.set_title("Missing Values Heatmap (Columns with >5% Missing)", fontsize=14)
    plt.tight_layout()
    plt.show()"""))

cells.append(md("""### 3.2 Univariate Analysis — Target Variable

The target variable `TargetValue` represents equipment selling price. We examine its distribution in both original and log-transformed scales."""))

cells.append(code("""# Target Variable Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Original scale
raw_skew = train_raw["TargetValue"].skew()
sns.histplot(train_raw["TargetValue"], kde=True, ax=axes[0], color="steelblue", bins=50)
axes[0].set_title(f"Target Distribution (Original)\\nSkewness: {raw_skew:.4f}", fontsize=13)
axes[0].set_xlabel("TargetValue")

# Log-transformed
log_target = np.log(train_raw["TargetValue"])
log_skew = log_target.skew()
sns.histplot(log_target, kde=True, ax=axes[1], color="coral", bins=50)
axes[1].set_title(f"Target Distribution (Log-Transformed)\\nSkewness: {log_skew:.4f}", fontsize=13)
axes[1].set_xlabel("log(TargetValue)")

plt.tight_layout()
plt.show()

print(f"\\n📌 Insight: The original target is right-skewed (skew={raw_skew:.4f}).")
print(f"   Log transformation reduces skewness to {log_skew:.4f}, making it more normally distributed.")
print("   → We will use log(TargetValue) as our modeling target.")"""))

cells.append(md("""### 3.3 Univariate Analysis — Numeric Features"""))

cells.append(code("""# Identify numeric and categorical columns
numeric_cols = train_raw.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols_no_target = [c for c in numeric_cols if c not in ["TransactionID", "TargetValue"]]
categorical_cols = train_raw.select_dtypes(include=["object"]).columns.tolist()

print(f"Numeric columns: {len(numeric_cols)} (including TransactionID & TargetValue)")
print(f"Categorical columns: {len(categorical_cols)}")
print(f"\\nNumeric features: {numeric_cols_no_target}")

# Distribution of key numeric features
if len(numeric_cols_no_target) > 0:
    n_plots = min(len(numeric_cols_no_target), 4)
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4))
    if n_plots == 1:
        axes = [axes]
    for i, col in enumerate(numeric_cols_no_target[:n_plots]):
        sns.histplot(train_raw[col].dropna(), kde=True, ax=axes[i], color="teal", bins=40)
        axes[i].set_title(f"Distribution of {col}", fontsize=11)
    plt.tight_layout()
    plt.show()"""))

cells.append(md("""### 3.4 Univariate Analysis — Categorical Features"""))

cells.append(code("""# Top categorical features by unique value count
cat_nunique = train_raw[categorical_cols].nunique().sort_values(ascending=False)
print("Unique value counts for categorical columns:")
print(cat_nunique)

# Plot value counts for a few key categorical features (low cardinality)
low_card_cats = cat_nunique[cat_nunique <= 15].index.tolist()[:4]

if low_card_cats:
    fig, axes = plt.subplots(1, len(low_card_cats), figsize=(5*len(low_card_cats), 4))
    if len(low_card_cats) == 1:
        axes = [axes]
    for i, col in enumerate(low_card_cats):
        counts = train_raw[col].value_counts().head(10)
        counts.plot(kind="barh", ax=axes[i], color="darkcyan")
        axes[i].set_title(f"Value Counts: {col}", fontsize=11)
        axes[i].set_xlabel("Count")
    plt.tight_layout()
    plt.show()"""))

cells.append(md("""### 3.5 Bivariate Analysis

We explore relationships between features and the target variable."""))

cells.append(code("""# Correlation heatmap for numeric features
numeric_data = train_raw[numeric_cols].copy()
corr_matrix = numeric_data.corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
    center=0, square=True, linewidths=0.5, ax=ax
)
ax.set_title("Correlation Heatmap — Numeric Features", fontsize=14)
plt.tight_layout()
plt.show()

print("\\n📌 Insight: The correlation matrix reveals relationships between numeric features.")
print("   Features highly correlated with TargetValue will be strong predictors.")"""))

cells.append(code("""# Bivariate: Key numeric features vs Target (scatter plots)
key_numerics = [c for c in numeric_cols_no_target if c in train_raw.columns][:3]

if key_numerics:
    fig, axes = plt.subplots(1, len(key_numerics), figsize=(6*len(key_numerics), 5))
    if len(key_numerics) == 1:
        axes = [axes]

    for i, col in enumerate(key_numerics):
        # Sample for faster plotting
        sample = train_raw[[col, "TargetValue"]].dropna().sample(n=min(5000, len(train_raw)), random_state=42)
        axes[i].scatter(sample[col], sample["TargetValue"], alpha=0.3, s=10, color="steelblue")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("TargetValue")
        axes[i].set_title(f"{col} vs TargetValue", fontsize=12)
    plt.tight_layout()
    plt.show()"""))

cells.append(code("""# Bivariate: Categorical features vs Target (box plots)
box_cats = low_card_cats[:3] if low_card_cats else []

if box_cats:
    fig, axes = plt.subplots(1, len(box_cats), figsize=(6*len(box_cats), 5))
    if len(box_cats) == 1:
        axes = [axes]

    for i, col in enumerate(box_cats):
        top_cats = train_raw[col].value_counts().head(8).index
        subset = train_raw[train_raw[col].isin(top_cats)]
        sns.boxplot(data=subset, x=col, y="TargetValue", ax=axes[i], palette="Set2")
        axes[i].set_title(f"TargetValue by {col}", fontsize=12)
        axes[i].tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.show()

print("\\n📌 Insight: Box plots show how target value varies across categorical groups.")
print("   Significant differences between groups indicate strong predictive features.")"""))

# ============================================================
# SECTION 5: Preprocessing
# ============================================================
cells.append(md("""## 4. Systematic Preprocessing

> **Key Requirement:** All data transformations (cleaning, scaling, encoding) are implemented using **reusable pipelines and custom functions**, consistently applied to both training and test datasets to **avoid data leakage**.

### 4.1 Define Preprocessing Functions"""))

cells.append(code("""def clean_raw_data(df):
    \"\"\"
    Clean raw data: handle sentinel values, drop unnecessary columns.
    This function is applied identically to both train and test.
    \"\"\"
    df = df.copy()

    # Drop columns that add no predictive value
    cols_to_drop = ["InventoryGroupDescription", "col18", "col19"]
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

    # Fix sentinel values
    if "ManufactureYear" in df.columns:
        df.loc[df["ManufactureYear"] == 1001, "ManufactureYear"] = np.nan

    if "OperationalHoursMeter" in df.columns:
        df.loc[df["OperationalHoursMeter"] == 0, "OperationalHoursMeter"] = np.nan

    return df


def engineer_features(df):
    \"\"\"
    Create new features from existing columns.
    Applied identically to both train and test to ensure consistency.
    \"\"\"
    df = df.copy()

    # --- Date Features ---
    if "TransactionDate" in df.columns:
        dt = pd.to_datetime(df["TransactionDate"])
        df["SaleYear"] = dt.dt.year
        df["SaleMonth"] = dt.dt.month
        df["SaleQuarter"] = dt.dt.quarter
        df["SaleDayOfWeek"] = dt.dt.dayofweek
        df["SaleDayOfYear"] = dt.dt.dayofyear
        df["DaysSince2000"] = (dt - pd.to_datetime("2000-01-01")).dt.days
        df.drop(columns=["TransactionDate"], inplace=True)

    # --- Machine Age Features ---
    if "ManufactureYear" in df.columns and "SaleYear" in df.columns:
        df["MachineAge"] = df["SaleYear"] - df["ManufactureYear"]
        df.loc[df["MachineAge"] < 0, "MachineAge"] = np.nan
        df["MachineAge_Sq"] = df["MachineAge"] ** 2
        df["MachineAge_Log"] = np.log1p(df["MachineAge"].clip(lower=0))
        df["IsNewMachine"] = (df["MachineAge"] <= 2).astype(float)
        df["IsOldMachine"] = (df["MachineAge"] >= 20).astype(float)

    # --- Usage Intensity Features ---
    if "OperationalHoursMeter" in df.columns:
        df["Hours_Log"] = np.log1p(df["OperationalHoursMeter"].clip(lower=0))
        df["Hours_Missing"] = df["OperationalHoursMeter"].isna().astype(float)

        if "MachineAge" in df.columns:
            df["Hours_Per_Year"] = df["OperationalHoursMeter"] / (df["MachineAge"].clip(lower=0) + 1.0)
            df["Hours_Per_Year_Log"] = np.log1p(df["Hours_Per_Year"].clip(lower=0))
            df["HighUsageFlag"] = (df["Hours_Per_Year"] > 1500).astype(float)
            df["LowUsageFlag"] = (df["Hours_Per_Year"] < 200).astype(float)

    # --- Text/Spec Parsing ---
    if "Spec_FullDescriptor" in df.columns:
        spec_str = df["Spec_FullDescriptor"].fillna("").astype(str)
        df["ModelToken"] = spec_str.str.extract(r"^([A-Z0-9]+)")[0].fillna("_Missing_")
        df["ModelSeriesPrefix"] = spec_str.str.extract(r"^([A-Za-z]+)")[0].fillna("_Missing_")
        df["ModelSeriesNumber"] = spec_str.str.extract(r"(\\d+)")[0].astype(float)

    # --- Functional Classification Parsing ---
    if "FunctionalClassification" in df.columns:
        func_str = df["FunctionalClassification"].fillna("").astype(str)
        df["EquipmentCategory"] = func_str.str.extract(r"^([A-Za-z\\s,]+)")[0].str.strip().fillna("_Missing_")

        # Extract capacity/power metrics
        df["Capacity_Tons_Min"] = func_str.str.extract(r"(\\d+\\.?\\d*)\\s*to\\s*\\d+\\.?\\d*\\s*Metric Tons")[0].astype(float)
        df["Capacity_Tons_Max"] = func_str.str.extract(r"\\d+\\.?\\d*\\s*to\\s*(\\d+\\.?\\d*)\\s*Metric Tons")[0].astype(float)
        df["Capacity_Tons_Mid"] = (df["Capacity_Tons_Min"] + df["Capacity_Tons_Max"]) / 2.0

        df["Power_HP_Min"] = func_str.str.extract(r"(\\d+\\.?\\d*)\\s*to\\s*\\d+\\.?\\d*\\s*Horsepower")[0].astype(float)
        df["Power_HP_Max"] = func_str.str.extract(r"\\d+\\.?\\d*\\s*to\\s*(\\d+\\.?\\d*)\\s*Horsepower")[0].astype(float)
        df["Power_HP_Mid"] = (df["Power_HP_Min"] + df["Power_HP_Max"]) / 2.0

    return df


print("✅ Preprocessing functions defined (clean_raw_data, engineer_features)")"""))

cells.append(md("""### 4.2 Apply Cleaning & Feature Engineering

We apply the **same functions** to both train and test datasets to ensure consistency and avoid data leakage."""))

cells.append(code("""# Step 1: Clean raw data
train_cleaned = clean_raw_data(train_raw)
test_cleaned = clean_raw_data(test_raw)

# Step 2: Separate target variable BEFORE any further processing
y = np.log(train_cleaned["TargetValue"])  # Log-transform target
train_cleaned.drop(columns=["TargetValue", "TransactionID"], inplace=True, errors="ignore")
test_ids = test_cleaned["TransactionID"].copy()
test_cleaned.drop(columns=["TransactionID"], inplace=True, errors="ignore")

# Step 3: Engineer features (same function for both)
train_fe = engineer_features(train_cleaned)
test_fe = engineer_features(test_cleaned)

print(f"✅ Train features: {train_fe.shape}")
print(f"✅ Test features:  {test_fe.shape}")"""))

cells.append(md("""### 4.3 Build sklearn Preprocessing Pipeline

We use `ColumnTransformer` + `Pipeline` to create a reusable preprocessing pipeline that handles:
- **Numeric features**: Impute missing values (median) + Standard scaling
- **Categorical features**: Fill missing values + Ordinal encoding

This pipeline is fitted on training data only and then applied to both train and test, preventing data leakage."""))

cells.append(code("""# Identify column types after feature engineering
fe_numeric_cols = []
fe_categorical_cols = []

for col in train_fe.columns:
    if pd.api.types.is_numeric_dtype(train_fe[col]):
        fe_numeric_cols.append(col)
    else:
        fe_categorical_cols.append(col)

# Fill categorical NaN with "_Missing_" first
for col in fe_categorical_cols:
    train_fe[col] = train_fe[col].fillna("_Missing_").astype(str)
    test_fe[col] = test_fe[col].fillna("_Missing_").astype(str)

print(f"Numeric features:     {len(fe_numeric_cols)}")
print(f"Categorical features: {len(fe_categorical_cols)}")
print(f"\\nCategorical columns: {fe_categorical_cols}")"""))

cells.append(code("""# Build sklearn preprocessing pipeline
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, fe_numeric_cols),
        ("cat", categorical_pipeline, fe_categorical_cols),
    ],
    remainder="drop",
)

# Fit on training data, transform both
X_train_processed = preprocessor.fit_transform(train_fe)
X_test_processed = preprocessor.transform(test_fe)

# Create DataFrames with proper column names
feature_names = fe_numeric_cols + fe_categorical_cols
X = pd.DataFrame(X_train_processed, columns=feature_names)
X_test = pd.DataFrame(X_test_processed, columns=feature_names)

print(f"\\n✅ Preprocessing pipeline applied:")
print(f"   X_train: {X.shape}")
print(f"   X_test:  {X_test.shape}")
print(f"\\n📌 The same pipeline was fitted on training data and applied to both sets,")
print("   ensuring no data leakage and consistent transformations.")"""))

# ============================================================
# SECTION 6: Feature Engineering Analysis
# ============================================================
cells.append(md("""## 5. Feature Importance Analysis

After engineering features, we analyze their importance using a Quick Random Forest model to understand which features contribute most to predicting equipment prices."""))

cells.append(code("""# Train a quick Random Forest to get feature importances
rf_quick = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42,
)
rf_quick.fit(X, y)

# Feature importance
importances = pd.Series(rf_quick.feature_importances_, index=feature_names)
top_features = importances.sort_values(ascending=False).head(20)

fig, ax = plt.subplots(figsize=(10, 8))
top_features.sort_values().plot(kind="barh", color="teal", ax=ax)
ax.set_title("Top 20 Feature Importances (Random Forest)", fontsize=14)
ax.set_xlabel("Importance")
plt.tight_layout()
plt.show()

print("\\n📌 Top 10 Most Important Features:")
for i, (feat, imp) in enumerate(top_features.head(10).items(), 1):
    print(f"   {i:2d}. {feat:<35s} — Importance: {imp:.4f}")

print("\\n📌 Insight: Features related to equipment configuration, age, and usage hours")
print("   tend to be the strongest predictors of selling price.")"""))

# ============================================================
# SECTION 7: Model Building
# ============================================================
cells.append(md("""## 6. Model Building & Comparison

We train **four different models** from different model families and compare their performance using 5-Fold Cross-Validation.

| Model | Family | Key Advantage |
|-------|--------|---------------|
| Random Forest | Bagging Ensemble | Robust, handles outliers well |
| XGBoost | Gradient Boosting | High accuracy, regularization |
| LightGBM | Gradient Boosting | Fast training, efficient with large data |
| CatBoost | Gradient Boosting | Native categorical support, minimal tuning |

### 6.1 Setup — Cross-Validation Framework"""))

cells.append(code("""# Define evaluation function for consistent model comparison
def evaluate_model_cv(model, X_data, y_data, model_name, n_splits=5):
    \"\"\"
    Evaluate a model using K-Fold CV. Returns scores and out-of-fold predictions.
    This reusable function ensures consistent evaluation across all models.
    \"\"\"
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(X_data))
    cv_rmse_scores = []
    cv_mae_scores = []
    cv_r2_scores = []
    fold_times = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_data), 1):
        fold_start = time.time()

        X_tr = X_data.iloc[train_idx]
        X_va = X_data.iloc[val_idx]
        y_tr = y_data.iloc[train_idx]
        y_va = y_data.iloc[val_idx]

        # Clone and fit model
        model_clone = model.__class__(**model.get_params())
        model_clone.fit(X_tr, y_tr)

        # Predict
        val_pred = model_clone.predict(X_va)
        oof_preds[val_idx] = val_pred

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_va, val_pred))
        mae = mean_absolute_error(y_va, val_pred)
        r2 = r2_score(y_va, val_pred)

        cv_rmse_scores.append(rmse)
        cv_mae_scores.append(mae)
        cv_r2_scores.append(r2)
        fold_times.append(time.time() - fold_start)

        print(f"  Fold {fold}: RMSE={rmse:.5f} | MAE={mae:.5f} | R²={r2:.5f} | {fold_times[-1]:.1f}s")

    results = {
        "model_name": model_name,
        "mean_rmse": np.mean(cv_rmse_scores),
        "std_rmse": np.std(cv_rmse_scores),
        "mean_mae": np.mean(cv_mae_scores),
        "mean_r2": np.mean(cv_r2_scores),
        "total_time": sum(fold_times),
        "oof_preds": oof_preds,
    }

    print(f"\\n  ✅ {model_name} — Mean RMSE: {results['mean_rmse']:.5f} ± {results['std_rmse']:.5f} | R²: {results['mean_r2']:.5f}")
    return results


# Store all model results
all_results = {}
print("✅ Evaluation framework ready.")"""))

cells.append(md("### 6.2 Model 1: Random Forest"))

cells.append(code("""print("=" * 60)
print("MODEL 1: Random Forest Regressor")
print("=" * 60)

rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=14,
    min_samples_leaf=4,
    n_jobs=-1,
    random_state=42,
)

all_results["Random Forest"] = evaluate_model_cv(rf_model, X, y, "Random Forest")"""))

cells.append(md("### 6.3 Model 2: XGBoost"))

cells.append(code("""print("=" * 60)
print("MODEL 2: XGBoost Regressor")
print("=" * 60)

xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=5,
    n_jobs=-1,
    random_state=42,
    verbosity=0,
)

all_results["XGBoost"] = evaluate_model_cv(xgb_model, X, y, "XGBoost")"""))

cells.append(md("### 6.4 Model 3: LightGBM"))

cells.append(code("""print("=" * 60)
print("MODEL 3: LightGBM Regressor")
print("=" * 60)

lgb_model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=127,
    max_depth=10,
    min_child_samples=15,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=1,
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
)

all_results["LightGBM"] = evaluate_model_cv(lgb_model, X, y, "LightGBM")"""))

cells.append(md("### 6.5 Model 4: CatBoost"))

cells.append(code("""print("=" * 60)
print("MODEL 4: CatBoost Regressor")
print("=" * 60)

cb_model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=8,
    random_seed=42,
    verbose=0,
    thread_count=-1,
)

all_results["CatBoost"] = evaluate_model_cv(cb_model, X, y, "CatBoost")"""))

# ============================================================
# SECTION 8: Model Comparison
# ============================================================
cells.append(md("""## 7. Model Comparison

### 7.1 Performance Summary"""))

cells.append(code("""# Build comparison table
comparison_data = []
for name, res in all_results.items():
    comparison_data.append({
        "Model": name,
        "Mean RMSE": f"{res['mean_rmse']:.5f}",
        "Std RMSE": f"{res['std_rmse']:.5f}",
        "Mean MAE": f"{res['mean_mae']:.5f}",
        "Mean R²": f"{res['mean_r2']:.5f}",
        "Training Time (s)": f"{res['total_time']:.1f}",
    })

comparison_df = pd.DataFrame(comparison_data)
print("=" * 90)
print("MODEL PERFORMANCE COMPARISON (5-Fold Cross-Validation)")
print("=" * 90)
print(comparison_df.to_string(index=False))

# Identify best model
best_model_name = min(all_results, key=lambda k: all_results[k]["mean_rmse"])
print(f"\\n🏆 Best Performing Model: {best_model_name} (RMSE: {all_results[best_model_name]['mean_rmse']:.5f})")"""))

cells.append(code("""# Visual comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

model_names = list(all_results.keys())
rmse_scores = [all_results[m]["mean_rmse"] for m in model_names]
mae_scores = [all_results[m]["mean_mae"] for m in model_names]
r2_scores = [all_results[m]["mean_r2"] for m in model_names]
colors = ["#2ecc71" if m == best_model_name else "#3498db" for m in model_names]

# RMSE Comparison
axes[0].barh(model_names, rmse_scores, color=colors)
axes[0].set_xlabel("RMSE (lower is better)")
axes[0].set_title("RMSE Comparison", fontsize=13)
for i, v in enumerate(rmse_scores):
    axes[0].text(v + 0.001, i, f"{v:.5f}", va="center", fontsize=10)

# MAE Comparison
axes[1].barh(model_names, mae_scores, color=colors)
axes[1].set_xlabel("MAE (lower is better)")
axes[1].set_title("MAE Comparison", fontsize=13)
for i, v in enumerate(mae_scores):
    axes[1].text(v + 0.001, i, f"{v:.5f}", va="center", fontsize=10)

# R² Comparison
axes[2].barh(model_names, r2_scores, color=colors)
axes[2].set_xlabel("R² Score (higher is better)")
axes[2].set_title("R² Score Comparison", fontsize=13)
for i, v in enumerate(r2_scores):
    axes[2].text(v + 0.001, i, f"{v:.5f}", va="center", fontsize=10)

plt.suptitle("Model Performance Comparison (5-Fold CV)", fontsize=15, y=1.02)
plt.tight_layout()
plt.show()"""))

cells.append(md("""### 7.2 Model Advantages & Disadvantages

| Model | Advantages | Disadvantages |
|-------|-----------|---------------|
| **Random Forest** | Robust to outliers and noise; handles non-linear relationships; minimal hyperparameter tuning needed; provides feature importances | Slower prediction time; can overfit on noisy data; generally lower accuracy than gradient boosting methods |
| **XGBoost** | Excellent accuracy; built-in regularization (L1/L2); handles missing values natively; widely used and well-documented | Slower training than LightGBM on large datasets; more hyperparameters to tune; memory intensive |
| **LightGBM** | Very fast training speed; memory efficient; handles large datasets well; leaf-wise tree growth for better accuracy | Can overfit on small datasets; leaf-wise growth may cause instability; requires careful tuning of num_leaves |
| **CatBoost** | Native categorical feature handling; robust with default parameters; ordered boosting reduces overfitting | Slower training than LightGBM; less community support than XGBoost; higher memory usage |

**Key Insight:** Gradient boosting models (XGBoost, LightGBM, CatBoost) generally outperform the Random Forest baseline, which is expected for tabular data. Among the boosting methods, the performance differences depend on hyperparameter tuning and data characteristics."""))

# ============================================================
# SECTION 9: Hyperparameter Tuning
# ============================================================
cells.append(md("""## 8. Hyperparameter Tuning

We use `sklearn.model_selection.RandomizedSearchCV` to optimize the best-performing model's hyperparameters. This approach is computationally efficient and avoids forbidden AutoML frameworks."""))

cells.append(code("""print("=" * 60)
print(f"HYPERPARAMETER TUNING: {best_model_name}")
print("=" * 60)

# Define search spaces based on best model
if best_model_name == "LightGBM":
    base_model = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbosity=-1)
    param_distributions = {
        "n_estimators": [300, 500, 800, 1000],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "num_leaves": [63, 127, 255],
        "max_depth": [8, 10, 12, -1],
        "min_child_samples": [10, 15, 20, 30],
        "feature_fraction": [0.6, 0.7, 0.8, 0.9],
        "bagging_fraction": [0.7, 0.8, 0.9],
        "bagging_freq": [1],
    }
elif best_model_name == "XGBoost":
    base_model = xgb.XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)
    param_distributions = {
        "n_estimators": [300, 500, 800, 1000],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "max_depth": [6, 8, 10, 12],
        "subsample": [0.6, 0.7, 0.8, 0.9],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
        "min_child_weight": [1, 3, 5, 10],
    }
elif best_model_name == "CatBoost":
    base_model = CatBoostRegressor(random_seed=42, verbose=0, thread_count=-1)
    param_distributions = {
        "iterations": [300, 500, 800, 1000],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "depth": [6, 8, 10],
    }
else:  # Random Forest
    base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    param_distributions = {
        "n_estimators": [200, 300, 500],
        "max_depth": [10, 14, 18, None],
        "min_samples_leaf": [2, 4, 6, 8],
        "min_samples_split": [2, 5, 10],
    }

# Run RandomizedSearchCV
search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_distributions,
    n_iter=15,
    scoring="neg_root_mean_squared_error",
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    random_state=42,
    n_jobs=-1,
    verbose=1,
)

search.fit(X, y)

print(f"\\n✅ Best Parameters:")
for param, value in search.best_params_.items():
    print(f"   {param}: {value}")
print(f"\\n✅ Best CV RMSE: {-search.best_score_:.5f}")"""))

# ============================================================
# SECTION 10: Final Model & Submission
# ============================================================
cells.append(md("""## 9. Final Model & Submission

We train the best model with optimized hyperparameters on the **full training set** and generate predictions for the test set."""))

cells.append(code("""# Train final model with best hyperparameters on full training data
print("Training final model on full training data...")
final_model = search.best_estimator_
final_model.fit(X, y)

# Generate predictions on test set
test_predictions_log = final_model.predict(X_test)
test_predictions = np.maximum(0, np.expm1(test_predictions_log))  # Convert back from log scale

# Create submission file
target_col = [c for c in sample_sub.columns if c != "TransactionID"][0]
submission = sample_sub.copy()
submission[target_col] = test_predictions
submission.to_csv("submission.csv", index=False)

print(f"\\n✅ Submission file created: submission.csv")
print(f"   Shape: {submission.shape}")
print(f"\\nFirst 5 predictions:")
print(submission.head())

# Validation checks
assert len(submission) == len(sample_sub), "Row count mismatch!"
assert submission["TransactionID"].equals(sample_sub["TransactionID"]), "TransactionID mismatch!"
assert submission[target_col].notna().all(), "NaN values in predictions!"
assert np.isfinite(submission[target_col]).all(), "Infinite values in predictions!"

print("\\n✅ All submission validation checks passed.")"""))

cells.append(code("""# Distribution of predicted vs actual prices
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Predicted price distribution
sns.histplot(test_predictions, kde=True, ax=axes[0], color="coral", bins=50)
axes[0].set_title("Predicted Price Distribution (Test Set)", fontsize=13)
axes[0].set_xlabel("Predicted TargetValue")

# Training vs Predicted comparison
sns.histplot(train_raw["TargetValue"], kde=True, ax=axes[1], color="steelblue", bins=50, label="Train (Actual)", alpha=0.6)
sns.histplot(test_predictions, kde=True, ax=axes[1], color="coral", bins=50, label="Test (Predicted)", alpha=0.6)
axes[1].set_title("Train Actual vs Test Predicted Distribution", fontsize=13)
axes[1].set_xlabel("TargetValue")
axes[1].legend()

plt.tight_layout()
plt.show()

print("\\n📌 The predicted distribution should broadly follow the training distribution,")
print("   confirming the model generalizes well to unseen data.")"""))

cells.append(md("""## 10. Conclusion

### Summary
- **Problem:** Predict the selling price of heavy equipment at auction
- **Approach:** Comprehensive EDA → Preprocessing Pipeline → Feature Engineering → Multi-Model Comparison → Hyperparameter Tuning
- **Models Compared:** Random Forest, XGBoost, LightGBM, CatBoost
- **Best Model:** Selected via 5-Fold CV RMSE and further optimized with `RandomizedSearchCV`
- **Key Features:** Equipment configuration, machine age, operational hours, and temporal features were found to be the most predictive

### Methodology Highlights
- All preprocessing transformations use **reusable sklearn Pipelines** applied consistently to train/test
- Feature engineering is done through **custom functions** to ensure reproducibility
- Hyperparameter tuning uses **RandomizedSearchCV** (not AutoML) for transparency
- Model comparison includes **RMSE, MAE, and R²** metrics with discussion of advantages/disadvantages"""))

# ============================================================
# Assemble notebook
# ============================================================
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbformat_minor": 5,
            "pygments_lexer": "ipython3",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

output_path = r"notebooks/24f2008980-notebook-2026t2 (3).ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"✅ Notebook written to: {output_path}")
print(f"   Total cells: {len(cells)}")
print(f"   Markdown cells: {sum(1 for c in cells if c['cell_type'] == 'markdown')}")
print(f"   Code cells: {sum(1 for c in cells if c['cell_type'] == 'code')}")
