import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load train data
df = pd.read_csv("data/train.csv", low_memory=False)
y_log = np.log1p(df["TargetValue"].clip(lower=0))

print(f"Total rows: {len(df)}")
print(f"Target log1p range: min={y_log.min():.4f}, max={y_log.max():.4f}, mean={y_log.mean():.4f}, std={y_log.std():.4f}")

# Check key columns
print("\nUnique counts:")
for col in ["ProductConfigID", "Spec_BaseClass", "VendorPartnerID", "RegionCode", "AssetID", "Spec_FullDescriptor"]:
    print(f"  {col}: {df[col].nunique()} unique, {df[col].isnull().sum()} nulls")

# Let's run a quick 5-fold LightGBM baseline with NATIVE categorical features
df_clean = df.copy()
df_clean["TransactionDate"] = pd.to_datetime(df_clean["TransactionDate"])
df_clean["SaleYear"] = df_clean["TransactionDate"].dt.year
df_clean["SaleMonth"] = df_clean["TransactionDate"].dt.month
df_clean["SaleQuarter"] = df_clean["TransactionDate"].dt.quarter

mfg = df_clean["ManufactureYear"].replace([1000, 1001], np.nan)
df_clean["MachineAge"] = (df_clean["SaleYear"] - mfg).clip(lower=0)
df_clean["Hours_log"] = np.log1p(df_clean["OperationalHoursMeter"].replace(0, np.nan))

cat_cols = ["ProductConfigID", "Spec_BaseClass", "VendorPartnerID", "RegionCode", "DataOriginCode", "UtilizationTier", "CabinType", "DrivetrainType", "AssetScaleFactor"]
for col in cat_cols:
    df_clean[col] = df_clean[col].astype(str).astype("category")

features = cat_cols + ["SaleYear", "SaleMonth", "SaleQuarter", "MachineAge", "Hours_log"]
X = df_clean[features]

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(df))

for tr_idx, val_idx in kf.split(X):
    X_tr, y_tr = X.iloc[tr_idx], y_log.iloc[tr_idx]
    X_va, y_va = X.iloc[val_idx], y_log.iloc[val_idx]

    model = lgb.LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=127,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    oof[val_idx] = model.predict(X_va)

rmsle = np.sqrt(mean_squared_error(y_log, oof))
print(f"\nSimple Native LightGBM 5-Fold OOF RMSLE: {rmsle:.5f}")

# Residual Analysis
residuals = np.abs(y_log - oof)
df["oof_pred"] = oof
df["residual"] = residuals
df["abs_pct_error"] = np.abs(df["TargetValue"] - np.expm1(oof)) / df["TargetValue"]

print("\nWorst 20 Residual Rows:")
print(df[["TargetValue", "oof_pred", "residual", "abs_pct_error", "ManufactureYear", "OperationalHoursMeter", "Spec_BaseClass", "ProductConfigID"]].sort_values("residual", ascending=False).head(20).to_string())

print("\nError grouped by TargetValue Quantiles:")
df["target_quantile"] = pd.qcut(df["TargetValue"], q=5)
print(df.groupby("target_quantile")["residual"].agg(["mean", "std", "count"]))
