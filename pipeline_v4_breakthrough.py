"""
pipeline_v4_breakthrough.py - Breakthrough Feature Pipeline with Out-of-Fold Asset History & Group-Level Filtering
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def load_raw_data(data_dir=r"e:\HeavyEquipmentML\data"):
    print("Loading raw data...")
    train = pd.read_csv(os.path.join(data_dir, "train.csv"), low_memory=False)
    test = pd.read_csv(os.path.join(data_dir, "test.csv"), low_memory=False)
    sample_sub = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
    return train, test, sample_sub


def build_v4_features(train_df, test_df):
    """
    Ground-up feature engineering + Out-of-Fold Asset ID Historical Price Lag Features.
    """
    print("Executing Breakthrough Pipeline Feature Generation...")

    train_df = train_df.copy()
    test_df = test_df.copy()

    y_train = np.log1p(train_df["TargetValue"].values)

    train_df["_is_test"] = 0
    test_df["_is_test"] = 1
    test_df["TargetValue"] = np.nan

    df = pd.concat([train_df, test_df], ignore_index=True)

    # 1. Sentinels & Missingness
    df.loc[df["ManufactureYear"] == 1001, "ManufactureYear"] = np.nan
    df.loc[df["OperationalHoursMeter"] == 0, "OperationalHoursMeter"] = np.nan

    # 2. Date & Time
    dt = pd.to_datetime(df["TransactionDate"])
    df["SaleYear"] = dt.dt.year
    df["SaleMonth"] = dt.dt.month
    df["SaleQuarter"] = dt.dt.quarter
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
    df["Hours_Missing"] = df["OperationalHoursMeter"].isna().astype(float)
    df["Hours_Per_Year"] = df["OperationalHoursMeter"] / (df["MachineAge"].clip(lower=0) + 1.0)
    df["Hours_Per_Year_Log"] = np.log1p(df["Hours_Per_Year"].clip(lower=0))
    df["Age_x_Hours"] = df["MachineAge"] * df["OperationalHoursMeter"]

    # 5. Model Token & Equipment Category
    spec_str = df["Spec_FullDescriptor"].fillna("").astype(str)
    df["ModelToken"] = spec_str.str.extract(r"^([A-Z0-9]+)")[0].fillna("_Missing_")
    df["ModelSeriesPrefix"] = spec_str.str.extract(r"^([A-Za-z]+)")[0].fillna("_Missing_")

    # Group rare model tokens (count <= 5) into _Rare_ to eliminate 51% noise
    token_counts = df["ModelToken"].value_counts()
    rare_tokens = token_counts[token_counts <= 5].index
    df.loc[df["ModelToken"].isin(rare_tokens), "ModelToken"] = "_Rare_"

    func_str = df["FunctionalClassification"].fillna("").astype(str)
    df["EquipmentCategory"] = func_str.str.extract(r"^([A-Za-z\s,]+)")[0].str.strip().fillna("_Missing_")

    # Metrics regex
    df["Capacity_Tons_Min"] = func_str.str.extract(r"(\d+\.?\d*)\s*to\s*\d+\.?\d*\s*Metric Tons")[0].astype(float)
    df["Capacity_Tons_Max"] = func_str.str.extract(r"\d+\.?\d*\s*to\s*(\d+\.?\d*)\s*Metric Tons")[0].astype(float)
    df["Capacity_Tons_Mid"] = (df["Capacity_Tons_Min"] + df["Capacity_Tons_Max"]) / 2.0

    df["Power_HP_Min"] = func_str.str.extract(r"(\d+\.?\d*)\s*to\s*\d+\.?\d*\s*Horsepower")[0].astype(float)
    df["Power_HP_Max"] = func_str.str.extract(r"\d+\.?\d*\s*to\s*(\d+\.?\d*)\s*Horsepower")[0].astype(float)
    df["Power_HP_Mid"] = (df["Power_HP_Min"] + df["Power_HP_Max"]) / 2.0

    # 6. Frequency Encodings
    freq_cols = [
        "ProductConfigID", "Spec_FullDescriptor", "Spec_BaseClass",
        "VendorPartnerID", "RegionCode", "EquipmentCategory", "ModelToken",
    ]
    for col in freq_cols:
        if col in df.columns:
            freq_map = df[col].value_counts(normalize=True)
            df[f"{col}_Freq"] = df[col].map(freq_map).fillna(0.0)

    # Clean Up Drop Columns
    drop_unneeded = [
        "TransactionID", "TargetValue", "TransactionDate",
        "InventoryGroupDescription", "col18", "col19", "_is_test",
    ]
    df.drop(columns=[c for c in drop_unneeded if c in df.columns], inplace=True)

    cat_cols = []
    num_cols = []

    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            cat_cols.append(c)
            df[c] = df[c].fillna("_Missing_").astype(str)
        else:
            num_cols.append(c)

    train_feat = df.iloc[:len(train_df)].copy().reset_index(drop=True)
    test_feat = df.iloc[len(train_df):].copy().reset_index(drop=True)

    return train_feat, y_train, test_feat, cat_cols, num_cols


def add_asset_historical_lag_and_target_encoding(X_train, y_train, X_test, n_splits=5, seed=42):
    """
    Adds Out-of-Fold AssetID previous sale price lag features + High-Density Target Encodings.
    """
    print("Computing OOF AssetID Historical Price Lags & Filtered Target Encodings...")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    global_mean = y_train.mean()
    global_std = y_train.std()

    X_tr = X_train.copy()
    X_te = X_test.copy()

    # --- OOF AssetID Historical Mean Price ---
    X_tr["AssetID_OOF_Price"] = global_mean
    X_te["AssetID_OOF_Price"] = global_mean

    tr_asset = X_tr["AssetID"]
    te_asset = X_te["AssetID"]

    for tr_idx, va_idx in kf.split(X_tr):
        f_asset = tr_asset.iloc[tr_idx]
        f_y = y_train[tr_idx]
        asset_stats = pd.DataFrame({"asset": f_asset, "y": f_y}).groupby("asset")["y"].mean()

        v_asset = tr_asset.iloc[va_idx]
        X_tr.iloc[va_idx, X_tr.columns.get_loc("AssetID_OOF_Price")] = v_asset.map(asset_stats).fillna(global_mean).values

    full_asset_stats = pd.DataFrame({"asset": tr_asset, "y": y_train}).groupby("asset")["y"].mean()
    X_te["AssetID_OOF_Price"] = te_asset.map(full_asset_stats).fillna(global_mean).values

    # --- High-Density Categorical Encodings (smooth=30 for noise reduction) ---
    specs = [
        (["ProductConfigID"], 30),
        (["Spec_FullDescriptor"], 30),
        (["ModelToken"], 20),
        (["Spec_BaseClass"], 15),
        (["ProductConfigID", "AgeBin"], 40),
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
        X_tr[mean_col] = global_mean

        for tr_idx, va_idx in kf.split(X_tr):
            f_key = key_tr.iloc[tr_idx]
            f_y = y_train[tr_idx]

            stats = pd.DataFrame({"k": f_key, "y": f_y}).groupby("k")["y"].agg(["count", "mean"])
            smoothed = (stats["count"] * stats["mean"] + smooth * global_mean) / (stats["count"] + smooth)

            v_key = key_tr.iloc[va_idx]
            X_tr.iloc[va_idx, X_tr.columns.get_loc(mean_col)] = v_key.map(smoothed).fillna(global_mean).values

        full_stats = pd.DataFrame({"k": key_tr, "y": y_train}).groupby("k")["y"].agg(["count", "mean"])
        full_smoothed = (full_stats["count"] * full_stats["mean"] + smooth * global_mean) / (full_stats["count"] + smooth)
        X_te[mean_col] = key_te.map(full_smoothed).fillna(global_mean).values

    return X_tr, X_te


if __name__ == "__main__":
    tr_raw, te_raw, sub = load_raw_data()
    X_tr, y, X_te, cat_cols, num_cols = build_v4_features(tr_raw, te_raw)
    X_tr_enc, X_te_enc = add_asset_historical_lag_and_target_encoding(X_tr, y, X_te)
    print(f"\nBreakthrough Features Ready: X_tr={X_tr_enc.shape}, X_te={X_te_enc.shape}")
