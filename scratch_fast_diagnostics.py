import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

df = pd.read_csv("data/train.csv", low_memory=False, nrows=30000)
y_log = np.log1p(df["TargetValue"].clip(lower=0))

df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])
df["SaleYear"] = df["TransactionDate"].dt.year
df["SaleMonth"] = df["TransactionDate"].dt.month

mfg = df["ManufactureYear"].replace([1000, 1001], np.nan)
df["MachineAge"] = (df["SaleYear"] - mfg).clip(lower=0)
df["Hours_log"] = np.log1p(df["OperationalHoursMeter"].replace(0, np.nan))

cat_cols = ["ProductConfigID", "Spec_BaseClass", "VendorPartnerID", "RegionCode", "DataOriginCode", "UtilizationTier", "CabinType", "DrivetrainType", "AssetScaleFactor"]
for col in cat_cols:
    df[col] = df[col].astype(str).astype("category")

features = cat_cols + ["SaleYear", "SaleMonth", "MachineAge", "Hours_log"]
X = df[features]

kf = KFold(n_splits=3, shuffle=True, random_state=42)
oof = np.zeros(len(df))

for tr_idx, val_idx in kf.split(X):
    X_tr, y_tr = X.iloc[tr_idx], y_log.iloc[tr_idx]
    X_va, y_va = X.iloc[val_idx], y_log.iloc[val_idx]

    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(30, verbose=False)])
    oof[val_idx] = model.predict(X_va)

rmsle = np.sqrt(mean_squared_error(y_log, oof))
print(f"Fast LightGBM Native Categorical Sample 30k RMSLE: {rmsle:.5f}")

# Target summary by target value bands
df["pred"] = oof
df["residual"] = np.abs(y_log - oof)

print("\nMean residual by target quantiles:")
df["qband"] = pd.qcut(df["TargetValue"], 5)
print(df.groupby("qband")["residual"].mean())
