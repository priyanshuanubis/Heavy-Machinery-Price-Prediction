# %%

import os
import re
import time
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna

import lightgbm as lgb
import xgboost as xgb

from sklearn.model_selection import KFold, train_test_split

import warnings

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("All Set.")




# %%

from pathlib import Path

# ---------------------------------------------------
# Locate dataset automatically (VS Code + Kaggle)
# ---------------------------------------------------

if os.path.isdir("/kaggle/input"):

    DATA_PATH = None

    for root, _, files in os.walk("/kaggle/input"):
        if (
            "train.csv" in files
            and "test.csv" in files
            and "sample_submission.csv" in files
        ):
            DATA_PATH = root
            break

else:

    CURRENT_DIR = Path(__file__).resolve().parent

    PROJECT_ROOT = CURRENT_DIR.parent

    DATA_PATH = PROJECT_ROOT / "data"

print(f"\nDataset Path : {DATA_PATH}\n")

train_raw = pd.read_csv(
    Path(DATA_PATH) / "train.csv",
    low_memory=False,
)

test_raw = pd.read_csv(
    Path(DATA_PATH) / "test.csv",
    low_memory=False,
)

sample_sub = pd.read_csv(
    Path(DATA_PATH) / "sample_submission.csv"
)

print(f"Train : {train_raw.shape}")
print(f"Test  : {test_raw.shape}")



# %%

print("=" * 60)
print("Training Dataset Overview")
print("=" * 60)
train_raw.info(max_cols=10)
print("\nTarget Variable Statistics")
print("-" * 60)
print(
    train_raw["TargetValue"].describe()
)



# %%


raw_skew = train_raw["TargetValue"].skew()
log_skew = np.log(train_raw["TargetValue"]).skew()
print(f"Skewness (Original): {raw_skew:.4f}")
print(f"Skewness (Log Scale): {log_skew:.4f}")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(
    train_raw["TargetValue"],
    kde=True,
    ax=axes[0],
    color="steelblue"
)
axes[0].set_title(
    f"Original Target Distribution\nSkew = {raw_skew:.2f}"
)
sns.histplot(
    np.log(train_raw["TargetValue"]),
    kde=True,
    ax=axes[1],
    color="darkgreen"
)
axes[1].set_title(
    f"Log Target Distribution\nSkew = {log_skew:.2f}"
)
plt.tight_layout()
plt.show()
missing_pct = (
    train_raw
    .isna()
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)
print("\nColumns with Highest Missing Percentage")
print("-" * 45)
print(missing_pct.head(10))



# %%


# modelling_v5_preprocessing_patch.py
#
# Replace the preprocess_data() function in your modelling script
# with the version below.

def preprocess_data(train_df, test_df):

    print("Starting preprocessing...")

    train_cleaned = train_df.copy(deep=True)
    test_cleaned = test_df.copy(deep=True)

    cols_to_remove = [
        "InventoryGroupDescription",
        "TransactionID",
    ]

    sparse_cols = [
        c for c in train_cleaned.columns
        if train_cleaned[c].isna().mean() > 0.99
    ]
    cols_to_remove.extend(sparse_cols)

    train_cleaned.drop(columns=cols_to_remove, inplace=True, errors="ignore")
    test_cleaned.drop(columns=cols_to_remove, inplace=True, errors="ignore")

    constant_cols = [
        c for c in train_cleaned.columns
        if train_cleaned[c].nunique(dropna=False) <= 1
    ]

    train_cleaned.drop(columns=constant_cols, inplace=True, errors="ignore")
    test_cleaned.drop(columns=constant_cols, inplace=True, errors="ignore")

    for df in (train_cleaned, test_cleaned):

        df.loc[
            (df["ManufactureYear"] < 1900)
            | (df["ManufactureYear"] > 2035),
            "ManufactureYear",
        ] = np.nan

        df["OperationalHoursMeter_missing"] = (
            df["OperationalHoursMeter"].isna()
            | (df["OperationalHoursMeter"] <= 0)
        )

        df.loc[
            df["OperationalHoursMeter"] <= 0,
            "OperationalHoursMeter",
        ] = np.nan

    if "AssetID" in train_cleaned.columns:
        asset_freq = train_cleaned["AssetID"].value_counts()
        train_cleaned["AssetID_freq"] = train_cleaned["AssetID"].map(asset_freq).fillna(1)
        test_cleaned["AssetID_freq"] = test_cleaned["AssetID"].map(asset_freq).fillna(1)

    if "VendorPartnerID" in train_cleaned.columns:
        vendor_freq = train_cleaned["VendorPartnerID"].value_counts()
        train_cleaned["VendorPartnerID_freq"] = train_cleaned["VendorPartnerID"].map(vendor_freq).fillna(1)
        test_cleaned["VendorPartnerID_freq"] = test_cleaned["VendorPartnerID"].map(vendor_freq).fillna(1)

    train_cleaned["TransactionDate"] = pd.to_datetime(
        train_cleaned["TransactionDate"],
        errors="coerce",
    )
    test_cleaned["TransactionDate"] = pd.to_datetime(
        test_cleaned["TransactionDate"],
        errors="coerce",
    )

    for df in (train_cleaned, test_cleaned):
        for c in df.select_dtypes(include=["int64"]).columns:
            df[c] = pd.to_numeric(df[c], downcast="integer")
        for c in df.select_dtypes(include=["float64"]).columns:
            df[c] = pd.to_numeric(df[c], downcast="float")

    excluded = {
        "TransactionDate",
        "TargetValue",
        "ManufactureYear",
        "OperationalHoursMeter",
        "OperationalHoursMeter_missing",
        "AssetID_freq",
        "VendorPartnerID_freq",
    }

    for c in train_cleaned.columns:
        if c in excluded:
            continue
        if train_cleaned[c].dtype == "object":
            train_cleaned[c] = train_cleaned[c].fillna("Missing").astype("category")
            if c in test_cleaned.columns:
                test_cleaned[c] = test_cleaned[c].fillna("Missing").astype("category")

    for c in train_cleaned.columns:
        if c in test_cleaned.columns and str(train_cleaned[c].dtype) == "category":
            test_cleaned[c] = pd.Categorical(
                test_cleaned[c],
                categories=train_cleaned[c].cat.categories,
            )

    train_cleaned["OperationalHoursMeter_missing"] = (
        train_cleaned["OperationalHoursMeter_missing"].astype(bool)
    )
    test_cleaned["OperationalHoursMeter_missing"] = (
        test_cleaned["OperationalHoursMeter_missing"].astype(bool)
    )

    return train_cleaned, test_cleaned




# %%


import numpy as np
import pandas as pd

def feature_engineering(train_df, test_df):
    """
    Advanced feature engineering compatible with newv3.py
    """

    print("Generating advanced engineered features...")

    train_fe = train_df.copy(deep=True)
    test_fe = test_df.copy(deep=True)

    high_card_cols = [
        "AssetID",
        "VendorPartnerID",
        "ProductConfigID",
        "Spec_BaseClass",
        "Spec_SubClass",
        "Spec_ReleaseSeries",
        "Spec_VariantModifier",
        "Spec_FullDescriptor",
    ]

    for df in (train_fe, test_fe):

        sale = df["TransactionDate"]

        df["SaleYear"] = sale.dt.year
        df["SaleQuarter"] = sale.dt.quarter
        df["SaleMonth"] = sale.dt.month
        df["SaleWeek"] = sale.dt.isocalendar().week.astype("int16")
        df["SaleDay"] = sale.dt.day
        df["SaleDayOfWeek"] = sale.dt.dayofweek
        df["SaleDayOfYear"] = sale.dt.dayofyear

        df["Weekend"] = (df["SaleDayOfWeek"] >= 5).astype("int8")
        df["MonthStart"] = sale.dt.is_month_start.astype("int8")
        df["MonthEnd"] = sale.dt.is_month_end.astype("int8")
        df["QuarterStart"] = sale.dt.is_quarter_start.astype("int8")
        df["QuarterEnd"] = sale.dt.is_quarter_end.astype("int8")
        df["YearStart"] = sale.dt.is_year_start.astype("int8")
        df["YearEnd"] = sale.dt.is_year_end.astype("int8")

        df["SaleMonthSin"] = np.sin(2*np.pi*df["SaleMonth"]/12)
        df["SaleMonthCos"] = np.cos(2*np.pi*df["SaleMonth"]/12)

        df["MachineAge"] = df["SaleYear"] - df["ManufactureYear"]
        df.loc[df["MachineAge"] < 0, "MachineAge"] = np.nan

        df["MachineAge2"] = df["MachineAge"]**2
        df["MachineAge3"] = df["MachineAge"]**3
        df["MachineAgeLog"] = np.log1p(df["MachineAge"])
        df["MachineAgeSqrt"] = np.sqrt(df["MachineAge"])

        df["LogHours"] = np.log1p(df["OperationalHoursMeter"])
        df["HoursSquared"] = df["OperationalHoursMeter"]**2
        df["HoursSqrt"] = np.sqrt(df["OperationalHoursMeter"])

        df["HoursPerYear"] = (
            df["OperationalHoursMeter"] /
            (df["MachineAge"] + 1)
        )

        df["Age_x_Hours"] = (
            df["MachineAge"] *
            df["OperationalHoursMeter"]
        )

        df["Age_x_Month"] = (
            df["MachineAge"] *
            df["SaleMonth"]
        )

        df["Age_x_Quarter"] = (
            df["MachineAge"] *
            df["SaleQuarter"]
        )

        spec = df["Spec_BaseClass"].fillna("Missing").astype(str)

        df["Spec_NumericClass"] = (
            spec.str.extract(r"(\d+)").astype(float)
        )

        df["Spec_Prefix"] = (
            spec.str.extract(r"^([A-Za-z]+)").fillna("Missing")
        )

        df["Spec_Suffix"] = (
            spec.str.extract(r"([A-Za-z]+)$").fillna("Missing")
        )

        desc = df["Spec_FullDescriptor"].fillna("").astype(str)

        df["DescriptorLength"] = desc.str.len()
        df["DescriptorWords"] = desc.str.split().str.len()
        df["DescriptorDigits"] = desc.str.count(r"\d")
        df["DescriptorUpper"] = desc.str.count(r"[A-Z]")

        df["HasTurbo"] = desc.str.contains(
            "turbo", case=False, regex=False
        ).astype("int8")

        df["HasSeries"] = desc.str.contains(
            "series", case=False, regex=False
        ).astype("int8")

        df.drop(columns=["Age"], inplace=True, errors="ignore")

    for col in high_card_cols:
        if col in train_fe.columns:
            freq = train_fe[col].value_counts(normalize=True)
            train_fe[col + "_freq"] = train_fe[col].map(freq).fillna(0)
            test_fe[col + "_freq"] = test_fe[col].map(freq).fillna(0)

    important_missing = [
        "ManufactureYear",
        "OperationalHoursMeter",
        "Spec_BaseClass",
        "Spec_SubClass",
        "Spec_FullDescriptor",
        "VendorPartnerID",
        "ProductConfigID",
    ]

    for col in important_missing:
        if col in train_fe.columns:
            train_fe[col + "_missing"] = train_fe[col].isna().astype("int8")
            test_fe[col + "_missing"] = test_fe[col].isna().astype("int8")

    for df in (train_fe, test_fe):
        df["RowMissingCount"] = df.isna().sum(axis=1)
        df["RowMissingRatio"] = df.isna().mean(axis=1)
        df["Hour_per_Age"] = (
            df["OperationalHoursMeter"] /
            (df["MachineAge"] + 1)
        )

    print(f"Engineered Train Shape : {train_fe.shape}")
    print(f"Engineered Test Shape  : {test_fe.shape}")

    return train_fe, test_fe




# =====================================================
# Execute preprocessing
# =====================================================

train_cl, test_cl = preprocess_data(
    train_raw,
    test_raw,
)

print(f"Processed Train Shape : {train_cl.shape}")
print(f"Processed Test Shape  : {test_cl.shape}")

# =====================================================
# Execute feature engineering
# =====================================================

train_fe, test_fe = feature_engineering(
    train_cl,
    test_cl,
)

# =====================================================
# Create modelling datasets
# =====================================================

X = train_fe.drop(
    columns=["TargetValue", "TransactionDate"],
    errors="ignore",
)

y = np.log(train_fe["TargetValue"])

X_test = test_fe.drop(
    columns=["TargetValue", "TransactionDate"],
    errors="ignore",
)

X_test = X_test.reindex(columns=X.columns)

print("\nFeature Matrix")
print(X.shape)
print(X_test.shape)

assert list(X.columns) == list(X_test.columns)

categorical_cols = [
    c
    for c in X.columns
    if not pd.api.types.is_numeric_dtype(X[c])
]

for col in categorical_cols:

    X[col] = X[col].astype("category")

    X_test[col] = pd.Categorical(
        X_test[col],
        categories=X[col].cat.categories,
    )

print("Feature preparation complete.")



from sklearn.model_selection import RepeatedKFold
import numpy as np
import time
import lightgbm as lgb
import xgboost as xgb

# Assumes X, y already exist

cv = RepeatedKFold(
    n_splits=5,
    n_repeats=2,
    random_state=42,
)

results = {}
saved_baselines = {}

def run_cv(model_name, model, X_data, y_data):
    print(f"\n===== {model_name} =====")
    scores = []
    start = time.time()

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_data), start=1):
        X_tr = X_data.iloc[tr_idx]
        X_va = X_data.iloc[va_idx]
        y_tr = y_data.iloc[tr_idx]
        y_va = y_data.iloc[va_idx]

        model.fit(X_tr, y_tr)

        pred = model.predict(X_va)

        rmse = np.sqrt(np.mean((y_va - pred) ** 2))
        scores.append(rmse)

        print(
            f"Fold {fold:02d} | RMSE={rmse:.5f}"
        )

    model.fit(X_data, y_data)

    saved_baselines[model_name] = model

    results[model_name] = (
        np.mean(scores),
        np.std(scores),
        time.time() - start,
    )

    print(
        f"Mean RMSE : {results[model_name][0]:.5f}"
    )

xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    tree_method="hist",
    enable_categorical=True,
    n_estimators=600,
    learning_rate=0.05,
    max_depth=8,
    min_child_weight=5,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.30,
    reg_lambda=1.50,
    random_state=42,
    n_jobs=-1,
)

lgb_model = lgb.LGBMRegressor(
    objective="regression",
    boosting_type="gbdt",
    n_estimators=600,
    learning_rate=0.05,
    num_leaves=127,
    max_depth=-1,
    min_child_samples=25,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.30,
    reg_lambda=1.50,
    random_state=42,
    verbosity=-1,
    n_jobs=-1,
)

run_cv("XGBoost", xgb_model,X,y)
run_cv("LightGBM", lgb_model,X,y)

print("\nSummary")
for k,v in results.items():
    print(f"{k:<10} RMSE={v[0]:.5f} STD={v[1]:.5f}")





# %%


# %%


"""
hyper.py

Drop-in replacement for the hyperparameter tuning section of newv3.py.

Assumes X, y and kf already exist.
"""

import pickle
import numpy as np
import optuna
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split

print("=== Preparing Sample for Hyperparameter Optimization ===")

sample_size = min(25000, len(X))
rng = np.random.default_rng(42)
sample_idx = rng.choice(len(X), size=sample_size, replace=False)

X_sample = X.iloc[sample_idx].copy()
y_sample = y.iloc[sample_idx].copy()

X_tr_s, X_va_s, y_tr_s, y_va_s = train_test_split(
    X_sample,
    y_sample,
    test_size=0.20,
    random_state=42,
)

# ---------------- LightGBM ---------------- #

def lgb_objective(trial):

    params = {
        "objective": "regression",
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "max_depth": trial.suggest_int("max_depth", -1, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.65, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.65, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-6, 5.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-6, 5.0, log=True),
        "verbosity": -1,
        "random_state": 42,
        "n_jobs": -1,
        "n_estimators": 600,
    }

    model = lgb.LGBMRegressor(**params)

    model.fit(
        X_tr_s,
        y_tr_s,
        eval_set=[(X_va_s, y_va_s)],
        callbacks=[lgb.early_stopping(15, verbose=False)],
    )

    pred = model.predict(X_va_s)
    return np.sqrt(np.mean((y_va_s - pred) ** 2))


lgb_study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42),
)

lgb_study.optimize(lgb_objective, n_trials=8)

best_lgb_params = lgb_study.best_params
print("Best LightGBM:", best_lgb_params)

# ---------------- XGBoost ---------------- #

def xgb_objective(trial):

    params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "enable_categorical": True,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 12),
        "subsample": trial.suggest_float("subsample", 0.65, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 3.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 5.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
        "n_estimators": 600,
        "early_stopping_rounds": 15,
    }

    model = xgb.XGBRegressor(**params)

    model.fit(
        X_tr_s,
        y_tr_s,
        eval_set=[(X_va_s, y_va_s)],
        verbose=False,
    )

    pred = model.predict(X_va_s)
    return np.sqrt(np.mean((y_va_s - pred) ** 2))


xgb_study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42),
)

xgb_study.optimize(xgb_objective, n_trials=8)

best_xgb_params = xgb_study.best_params
print("Best XGBoost:", best_xgb_params)

# -------- Final Training -------- #

lgb_final = lgb.LGBMRegressor(
    n_estimators=600,
    random_state=42,
    verbosity=-1,
    n_jobs=-1,
    **best_lgb_params,
)

lgb_final.fit(X, y)

xgb_final = xgb.XGBRegressor(
    n_estimators=600,
    objective="reg:squarederror",
    tree_method="hist",
    enable_categorical=True,
    random_state=42,
    n_jobs=-1,
    **best_xgb_params,
)

xgb_final.fit(X, y)

with open("lightgbm_tuned.pkl", "wb") as f:
    pickle.dump(lgb_final, f)

with open("xgboost_tuned.pkl", "wb") as f:
    pickle.dump(xgb_final, f)

print("Optimized models saved.")


print("\n===== Tuned LightGBM CV =====")

lgb_scores = []

for fold, (tr_idx, va_idx) in enumerate(cv.split(X), start=1):

    X_tr = X.iloc[tr_idx]
    X_va = X.iloc[va_idx]

    y_tr = y.iloc[tr_idx]
    y_va = y.iloc[va_idx]

    model = lgb.LGBMRegressor(
        n_estimators=600,
        random_state=42,
        verbosity=-1,
        n_jobs=-1,
        **best_lgb_params,
    )

    model.fit(X_tr, y_tr)

    pred = model.predict(X_va)

    rmse = np.sqrt(np.mean((y_va - pred) ** 2))

    lgb_scores.append(rmse)

    print(f"Fold {fold:02d} | RMSE={rmse:.5f}")

print(f"LightGBM Tuned Mean RMSE : {np.mean(lgb_scores):.5f}")


print("\n===== Tuned LightGBM CV =====")

lgb_scores = []

for fold, (tr_idx, va_idx) in enumerate(cv.split(X), start=1):

    X_tr = X.iloc[tr_idx]
    X_va = X.iloc[va_idx]

    y_tr = y.iloc[tr_idx]
    y_va = y.iloc[va_idx]

    model = lgb.LGBMRegressor(
        n_estimators=600,
        random_state=42,
        verbosity=-1,
        n_jobs=-1,
        **best_lgb_params,
    )

    model.fit(X_tr, y_tr)

    pred = model.predict(X_va)

    rmse = np.sqrt(np.mean((y_va - pred) ** 2))

    lgb_scores.append(rmse)

    print(f"Fold {fold:02d} | RMSE={rmse:.5f}")

print(f"LightGBM Tuned Mean RMSE : {np.mean(lgb_scores):.5f}")# %%


# %%

import os
import pickle
import numpy as np
import pandas as pd


