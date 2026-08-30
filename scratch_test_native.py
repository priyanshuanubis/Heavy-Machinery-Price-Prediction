import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

df = pd.read_csv("data/train.csv", low_memory=False)
y_log = np.log1p(df["TargetValue"].clip(lower=0))

df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])
df["SaleYear"] = df["TransactionDate"].dt.year
df["SaleMonth"] = df["TransactionDate"].dt.month
df["SaleQuarter"] = df["TransactionDate"].dt.quarter
df["SaleDayOfWeek"] = df["TransactionDate"].dt.dayofweek

mfg = df["ManufactureYear"].replace([1000, 1001], np.nan)
df["MachineAge"] = (df["SaleYear"] - mfg).clip(lower=0)
df["Hours_log"] = np.log1p(df["OperationalHoursMeter"].replace(0, np.nan))
df["HoursPerYear"] = np.expm1(df["Hours_log"]) / (df["MachineAge"] + 1.0)
df["Depreciation_Velocity"] = df["MachineAge"] / (df["Hours_log"] + 1.0)

# Native LightGBM Test
cat_cols = ["ProductConfigID", "Spec_BaseClass", "VendorPartnerID", "RegionCode", "DataOriginCode", "UtilizationTier", "CabinType", "DrivetrainType", "AssetScaleFactor", "InventoryGroupCategory"]
for col in cat_cols:
    df[col] = df[col].astype(str).astype("category")

features = cat_cols + ["SaleYear", "SaleMonth", "SaleQuarter", "SaleDayOfWeek", "MachineAge", "Hours_log", "HoursPerYear", "Depreciation_Velocity"]
X = df[features]

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_lgb = np.zeros(len(df))

for tr_idx, val_idx in kf.split(X):
    X_tr, y_tr = X.iloc[tr_idx], y_log.iloc[tr_idx]
    X_va, y_va = X.iloc[val_idx], y_log.iloc[val_idx]

    model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=95,
        min_child_samples=25,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    oof_lgb[val_idx] = model.predict(X_va)

score_lgb = np.sqrt(mean_squared_error(y_log, oof_lgb))
print(f"Native Categorical LightGBM (No Target Encoding): {score_lgb:.5f}")
