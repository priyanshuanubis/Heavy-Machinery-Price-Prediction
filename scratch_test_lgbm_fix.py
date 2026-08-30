import os
import time
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

df_tr_raw = pd.read_csv("data/train.csv", low_memory=False)
df_te_raw = pd.read_csv("data/test.csv", low_memory=False)

import generate_v24_pinnacle_mastery_notebook as gen

# Clean data
df_tr, df_te = df_tr_raw.copy(), df_te_raw.copy()

mfg_col = "ManufactureYear" if "ManufactureYear" in df_tr.columns else "YearMade"
df_tr["IsYearMissing"] = df_tr[mfg_col].isin([1000, 1001]).astype(int)
df_tr.loc[df_tr[mfg_col].isin([1000, 1001]), mfg_col] = np.nan
df_te["IsYearMissing"] = df_te[mfg_col].isin([1000, 1001]).astype(int)
df_te.loc[df_te[mfg_col].isin([1000, 1001]), mfg_col] = np.nan

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

# Feature engineering
for dataset in (df_tr, df_te):
    t_date = dataset["TransactionDate"]
    dataset["TransactionYear"] = t_date.dt.year
    dataset["TransactionMonth"] = t_date.dt.month
    dataset["TransactionQuarter"] = t_date.dt.quarter
    dataset["TransactionDayOfWeek"] = t_date.dt.dayofweek
    dataset["TransactionDay"] = t_date.dt.day

    year_made = pd.to_numeric(dataset[mfg_col], errors="coerce")
    dataset["AssetAge"] = (dataset["TransactionYear"] - year_made).clip(lower=0)

    hours_val = dataset["Hours_log"]
    age_val = dataset["AssetAge"]
    dataset["HoursPerYear"] = np.expm1(hours_val.fillna(0)) / (age_val.fillna(0) + 1.0)

    spec_str = dataset["Spec_FullDescriptor"].fillna("").astype(str)
    dataset["SpecificationComplexity"] = spec_str.str.split().str.len()
    dataset["DescriptorLength"] = spec_str.str.len()

    func_str = dataset["FunctionalClassification"].fillna("").astype(str)
    dataset["MachineType"] = func_str.str.extract(r"^(.+?)\s*-\s*")[0].fillna("Unknown")

    base_class = dataset["Spec_BaseClass"].fillna("Missing").astype(str) if "Spec_BaseClass" in dataset.columns else pd.Series("Missing", index=dataset.index)
    prod_cfg = dataset["ProductConfigID"].fillna("Missing").astype(str) if "ProductConfigID" in dataset.columns else pd.Series("Missing", index=dataset.index)
    region = dataset["RegionCode"].fillna("Missing").astype(str) if "RegionCode" in dataset.columns else pd.Series("Missing", index=dataset.index)
    vendor = dataset["VendorPartnerID"].fillna("Missing").astype(str) if "VendorPartnerID" in dataset.columns else pd.Series("Missing", index=dataset.index)

    dataset["Cat_Prod_Region"] = prod_cfg + "_" + region
    dataset["Cat_Prod_Year"] = prod_cfg + "_" + dataset["TransactionYear"].astype(str)
    dataset["Cat_Base_Region"] = base_class + "_" + region
    dataset["Cat_Vendor_Base"] = vendor + "_" + base_class
    dataset["ModelToken"] = spec_str.str.extract(r"^([A-Z0-9]+)")[0].fillna("_Missing_")

# Target Encodings
y_log = np.log1p(df_tr["TargetValue"].clip(lower=0))

# FEATURE SELECTION & DATATYPE NORMALIZATION (THE FIX)
drop_cols = ["SalesID", "TransactionDate", "TargetValue", "AssetID", "Spec_FullDescriptor", "FunctionalClassification"]
feature_cols = [c for c in df_tr.columns if c not in drop_cols]

X = df_tr[feature_cols].copy()
X_test = df_te[feature_cols].copy()

# Automatically identify ALL categorical/object/string columns
cat_cols = [c for c in X.columns if X[c].dtype == "object" or X[c].dtype == "string" or isinstance(X[c].dtype, pd.CategoricalDtype)]

print(f"Total Features: {X.shape[1]}")
print(f"Categorical Features Detected: {len(cat_cols)}")
print(f"Categorical Feature Names: {cat_cols}")

# Convert all categorical features to pandas Category dtype
for col in cat_cols:
    X[col] = X[col].fillna("Missing").astype(str).astype("category")
    X_test[col] = pd.Categorical(X_test[col].fillna("Missing").astype(str), categories=X[col].cat.categories)

bad_dtypes = [f"{col}: {dtype}" for col, dtype in X.dtypes.items() if dtype == "object" or dtype == "string"]
print(f"Bad object/string dtypes remaining in X: {bad_dtypes}")

# Verify LightGBM fit on Fold 1
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
target_bins = pd.qcut(y_log, q=10, labels=False, duplicates="drop")

for trn_idx, val_idx in skf.split(X, target_bins):
    X_tr_cat, y_tr = X.iloc[trn_idx], y_log.iloc[trn_idx]
    X_va_cat, y_va = X.iloc[val_idx], y_log.iloc[val_idx]
    
    m_lgb = lgb.LGBMRegressor(n_estimators=50, learning_rate=0.03, random_state=42, verbose=-1, n_jobs=-1)
    m_lgb.fit(X_tr_cat, y_tr, eval_set=[(X_va_cat, y_va)], callbacks=[lgb.early_stopping(10, verbose=False)])
    preds = m_lgb.predict(X_va_cat)
    score = np.sqrt(mean_squared_error(y_va, preds))
    print(f"[TEST SUCCESS] LightGBM Fold 1 RMSLE: {score:.5f}")
    break
