"""
pipeline_v3_clean.py - Complete Ground-Up Redesign of Pipeline & Feature Engineering
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


def build_ground_up_features(train_df, test_df):
    """
    Complete ground-up feature engineering tailored to heavy machinery valuation.
    """
    print("Executing Ground-Up Feature Engineering...")

    train_df = train_df.copy()
    test_df = test_df.copy()

    # Extract target from train
    y_train = np.log1p(train_df["TargetValue"].values)

    train_df["_is_test"] = 0
    test_df["_is_test"] = 1
    test_df["TargetValue"] = np.nan

    df = pd.concat([train_df, test_df], ignore_index=True)

    # 1. Clean Sentinel / Missing Values
    df.loc[df["ManufactureYear"] == 1001, "ManufactureYear"] = np.nan
    df.loc[df["OperationalHoursMeter"] == 0, "OperationalHoursMeter"] = np.nan

    # 2. Date & Time Decomposition
    dt = pd.to_datetime(df["TransactionDate"])
    df["SaleYear"] = dt.dt.year
    df["SaleMonth"] = dt.dt.month
    df["SaleQuarter"] = dt.dt.quarter
    df["SaleDayOfWeek"] = dt.dt.dayofweek
    df["SaleDayOfYear"] = dt.dt.dayofyear
    df["DaysSince2000"] = (dt - pd.to_datetime("2000-01-01")).dt.days

    # 3. Machine Age & Depreciation Features
    df["MachineAge"] = df["SaleYear"] - df["ManufactureYear"]
    df.loc[df["MachineAge"] < 0, "MachineAge"] = np.nan

    df["MachineAge_Sq"] = df["MachineAge"] ** 2
    df["MachineAge_Log"] = np.log1p(df["MachineAge"].clip(lower=0))
    df["IsNewMachine"] = (df["MachineAge"] <= 2).astype(float)
    df["IsOldMachine"] = (df["MachineAge"] >= 20).astype(float)

    # Age Decades / Bins
    df["AgeBin"] = pd.cut(
        df["MachineAge"],
        bins=[-1, 2, 5, 9, 14, 20, 30, 100],
        labels=[0, 1, 2, 3, 4, 5, 6],
    ).astype(float)

    # 4. Operational Hours & Intensity Features
    df["Hours_Log"] = np.log1p(df["OperationalHoursMeter"].clip(lower=0))
    df["Hours_Missing"] = df["OperationalHoursMeter"].isna().astype(float)
    df["Hours_Per_Year"] = df["OperationalHoursMeter"] / (df["MachineAge"].clip(lower=0) + 1.0)
    df["Hours_Per_Year_Log"] = np.log1p(df["Hours_Per_Year"].clip(lower=0))
    df["Age_x_Hours"] = df["MachineAge"] * df["OperationalHoursMeter"]

    # Heavy vs Light Usage Flag (relative to machine age)
    df["HighUsageFlag"] = (df["Hours_Per_Year"] > 1500).astype(float)
    df["LowUsageFlag"] = (df["Hours_Per_Year"] < 200).astype(float)

    # 5. Text Token & Model Family Parsing from Spec_FullDescriptor
    spec_str = df["Spec_FullDescriptor"].fillna("").astype(str)

    # Extract primary Model Series Token (e.g., 140G, 320C, 310G, 950F)
    df["ModelToken"] = spec_str.str.extract(r"^([A-Z0-9]+)")[0].fillna("_Missing_")

    # Extract Brand/Series Code Prefix (e.g. Caterpillar vs Komatsu vs Deere naming rules)
    df["ModelSeriesPrefix"] = spec_str.str.extract(r"^([A-Za-z]+)")[0].fillna("_Missing_")
    df["ModelSeriesNumber"] = spec_str.str.extract(r"(\d+)")[0].astype(float)

    # 6. FunctionalClassification Domain Extraction
    func_str = df["FunctionalClassification"].fillna("").astype(str)

    df["EquipmentCategory"] = func_str.str.extract(r"^([A-Za-z\s,]+)")[0].str.strip().fillna("_Missing_")

    # Extract Ton / Horsepower / Dig Depth Metrics via Regex
    df["Capacity_Tons_Min"] = func_str.str.extract(r"(\d+\.?\d*)\s*to\s*\d+\.?\d*\s*Metric Tons")[0].astype(float)
    df["Capacity_Tons_Max"] = func_str.str.extract(r"\d+\.?\d*\s*to\s*(\d+\.?\d*)\s*Metric Tons")[0].astype(float)
    df["Capacity_Tons_Mid"] = (df["Capacity_Tons_Min"] + df["Capacity_Tons_Max"]) / 2.0

    df["Power_HP_Min"] = func_str.str.extract(r"(\d+\.?\d*)\s*to\s*\d+\.?\d*\s*Horsepower")[0].astype(float)
    df["Power_HP_Max"] = func_str.str.extract(r"\d+\.?\d*\s*to\s*(\d+\.?\d*)\s*Horsepower")[0].astype(float)
    df["Power_HP_Mid"] = (df["Power_HP_Min"] + df["Power_HP_Max"]) / 2.0

    df["Depth_Ft_Min"] = func_str.str.extract(r"(\d+\.?\d*)\s*to\s*\d+\.?\d*\s*Ft")[0].astype(float)
    df["Depth_Ft_Max"] = func_str.str.extract(r"\d+\.?\d*\s*to\s*(\d+\.?\d*)\s*Ft")[0].astype(float)
    df["Depth_Ft_Mid"] = (df["Depth_Ft_Min"] + df["Depth_Ft_Max"]) / 2.0

    # 7. Frequency Encodings for Categorical Grids
    freq_cols = [
        "ProductConfigID", "Spec_FullDescriptor", "Spec_BaseClass",
        "VendorPartnerID", "RegionCode", "EquipmentCategory",
        "ModelToken", "DataOriginCode", "CabinType",
    ]

    for col in freq_cols:
        if col in df.columns:
            freq_map = df[col].value_counts(normalize=True)
            df[f"{col}_Freq"] = df[col].map(freq_map).fillna(0.0)

    # 8. Clean Up Columns & Define Column Sets
    drop_unneeded = [
        "TransactionID", "TargetValue", "TransactionDate",
        "InventoryGroupDescription", "col18", "col19", "_is_test",
    ]
    df.drop(columns=[c for c in drop_unneeded if c in df.columns], inplace=True)

    # Identify Categoricals vs Numerics
    cat_cols = []
    num_cols = []

    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            cat_cols.append(c)
            df[c] = df[c].fillna("_Missing_").astype(str)
        else:
            num_cols.append(c)

    # Separate Train and Test
    train_feat = df.iloc[:len(train_df)].copy().reset_index(drop=True)
    test_feat = df.iloc[len(train_df):].copy().reset_index(drop=True)

    print(f"Features Generated: {train_feat.shape[1]} total (Numerics: {len(num_cols)}, Categoricals: {len(cat_cols)})")
    return train_feat, y_train, test_feat, cat_cols, num_cols


def compute_smoothed_oof_target_encoding(X_train, y_train, X_test, n_splits=5, seed=42):
    """
    Computes strict out-of-fold Bayesian smoothed target encodings.
    """
    print("Computing Multi-Tier Bayesian Target Encodings...")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    global_mean = y_train.mean()
    global_std = y_train.std()

    # Spec hierarchy: (column_list, smoothing_weight)
    specs = [
        (["ProductConfigID"], 20),
        (["Spec_FullDescriptor"], 20),
        (["ModelToken"], 15),
        (["Spec_BaseClass"], 10),
        (["EquipmentCategory"], 10),
        (["RegionCode"], 10),
        (["VendorPartnerID"], 10),
        # Interaction Encodings
        (["ProductConfigID", "AgeBin"], 30),
        (["Spec_FullDescriptor", "AgeBin"], 30),
        (["ProductConfigID", "RegionCode"], 30),
    ]

    X_tr = X_train.copy()
    X_te = X_test.copy()

    for cols, smooth in specs:
        # Construct key
        if len(cols) == 1:
            key_tr = X_tr[cols[0]].astype(str)
            key_te = X_te[cols[0]].astype(str)
            prefix = cols[0]
        else:
            prefix = "_x_".join(cols)
            key_tr = X_tr[cols[0]].astype(str)
            key_te = X_te[cols[0]].astype(str)
            for c in cols[1:]:
                key_tr = key_tr + "___" + X_tr[c].astype(str)
                key_te = key_te + "___" + X_te[c].astype(str)

        mean_col = f"TE_{prefix}_Mean"
        std_col = f"TE_{prefix}_Std"
        count_col = f"TE_{prefix}_Count"

        X_tr[mean_col] = global_mean
        X_tr[std_col] = global_std
        X_tr[count_col] = 0.0

        # OOF Loop for Train
        for tr_idx, va_idx in kf.split(X_tr):
            f_key = key_tr.iloc[tr_idx]
            f_y = y_train[tr_idx]

            stats = pd.DataFrame({"k": f_key, "y": f_y}).groupby("k")["y"].agg(["count", "mean", "std"])
            smoothed = (stats["count"] * stats["mean"] + smooth * global_mean) / (stats["count"] + smooth)

            v_key = key_tr.iloc[va_idx]
            X_tr.iloc[va_idx, X_tr.columns.get_loc(mean_col)] = v_key.map(smoothed).fillna(global_mean).values
            X_tr.iloc[va_idx, X_tr.columns.get_loc(std_col)] = v_key.map(stats["std"]).fillna(global_std).values
            X_tr.iloc[va_idx, X_tr.columns.get_loc(count_col)] = v_key.map(stats["count"]).fillna(0.0).values

        # Full Fit for Test
        full_stats = pd.DataFrame({"k": key_tr, "y": y_train}).groupby("k")["y"].agg(["count", "mean", "std"])
        full_smoothed = (full_stats["count"] * full_stats["mean"] + smooth * global_mean) / (full_stats["count"] + smooth)

        X_te[mean_col] = key_te.map(full_smoothed).fillna(global_mean).values
        X_te[std_col] = key_te.map(full_stats["std"]).fillna(global_std).values
        X_te[count_col] = key_te.map(full_stats["count"]).fillna(0.0).values

    return X_tr, X_te


if __name__ == "__main__":
    tr_raw, te_raw, sub = load_raw_data()
    X_tr, y, X_te, cat_cols, num_cols = build_ground_up_features(tr_raw, te_raw)
    X_tr_enc, X_te_enc = compute_smoothed_oof_target_encoding(X_tr, y, X_te)
    print(f"\nFinal Ready Features: X_tr={X_tr_enc.shape}, X_te={X_te_enc.shape}")
