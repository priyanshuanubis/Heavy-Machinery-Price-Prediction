import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

df = pd.read_csv("data/train.csv", low_memory=False)
y_log = np.log1p(df["TargetValue"].clip(lower=0))

df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])
df["SaleYear"] = df["TransactionDate"].dt.year
df["SaleMonth"] = df["TransactionDate"].dt.month
df["SaleQuarter"] = df["TransactionDate"].dt.quarter
df["SaleDayOfWeek"] = df["TransactionDate"].dt.dayofweek

mfg_raw = df["ManufactureYear"]
df["IsYearMissing"] = mfg_raw.isin([1000, 1001]).astype(int)
mfg = mfg_raw.replace([1000, 1001], np.nan)

df["MachineAge"] = (df["SaleYear"] - mfg).clip(lower=0)
df["MachineAge_Sq"] = df["MachineAge"] ** 2

hours_raw = df["OperationalHoursMeter"]
df["IsHoursMissing"] = (hours_raw.isnull() | (hours_raw == 0.0)).astype(int)
df["Hours_log"] = np.log1p(hours_raw.replace(0, np.nan))

df["HoursPerYear"] = np.expm1(df["Hours_log"]) / (df["MachineAge"] + 1.0)
df["Depreciation_Velocity"] = df["MachineAge"] / (df["Hours_log"] + 1.0)
df["AgeXHours"] = df["MachineAge"] * df["Hours_log"]

# Low price indicator features
df["Age_x_MissingHours"] = df["MachineAge"] * df["IsHoursMissing"]
df["Age_x_MissingYear"] = df["IsYearMissing"] * df["Hours_log"]

# OOF AssetID Price Lag
df["AssetID_OOF_Price"] = y_log.mean()
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for tr_idx, val_idx in kf.split(df):
    asset_map = df.iloc[tr_idx].groupby("AssetID")["TargetValue"].apply(
        lambda s: np.log1p(s.clip(lower=0)).mean()
    ).to_dict()
    df.iloc[val_idx, df.columns.get_loc("AssetID_OOF_Price")] = (
        df.iloc[val_idx]["AssetID"].map(asset_map).fillna(y_log.mean())
    )

cat_cols = ["ProductConfigID", "Spec_BaseClass", "VendorPartnerID", "RegionCode", "DataOriginCode", "UtilizationTier", "CabinType", "DrivetrainType", "AssetScaleFactor", "InventoryGroupCategory"]
for col in cat_cols:
    df[col] = df[col].astype(str).astype("category")

features = cat_cols + [
    "SaleYear", "SaleMonth", "SaleQuarter", "SaleDayOfWeek",
    "MachineAge", "MachineAge_Sq", "Hours_log", "HoursPerYear",
    "Depreciation_Velocity", "AgeXHours", "IsYearMissing", "IsHoursMissing",
    "Age_x_MissingHours", "Age_x_MissingYear", "AssetID_OOF_Price"
]

X = df[features]
oof_lgb = np.zeros(len(df))

for tr_idx, val_idx in kf.split(X):
    X_tr, y_tr = X.iloc[tr_idx], y_log.iloc[tr_idx]
    X_va, y_va = X.iloc[val_idx], y_log.iloc[val_idx]

    model = lgb.LGBMRegressor(
        n_estimators=3500,
        learning_rate=0.015,
        num_leaves=127,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.6,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(150, verbose=False)]
    )
    oof_lgb[val_idx] = model.predict(X_va)

score_lgb = np.sqrt(mean_squared_error(y_log, oof_lgb))
print(f"Native Categorical LightGBM + LowPrice & Asset Features 5-Fold OOF RMSLE: {score_lgb:.5f}")
