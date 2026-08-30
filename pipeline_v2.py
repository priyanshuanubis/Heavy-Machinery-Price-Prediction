"""
pipeline_v2.py - Complete data pipeline for Heavy Equipment Price Prediction
Target: 0.15 RMSE on log1p(TargetValue)
Allowed libraries: NumPy, Pandas, Scikit-learn, XGBoost, LightGBM, CatBoost, SciPy, regex, Python built-ins
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


# ============================================================
# 1. DATA LOADING
# ============================================================
def load_data(data_dir=r"e:\HeavyEquipmentML\data"):
    print("Loading data...")
    train = pd.read_csv(os.path.join(data_dir, "train.csv"), low_memory=False)
    test = pd.read_csv(os.path.join(data_dir, "test.csv"), low_memory=False)
    sample_sub = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
    print(f"  Train: {train.shape}, Test: {test.shape}")
    return train, test, sample_sub


# ============================================================
# 2. DATA CLEANING
# ============================================================
def clean_data(df):
    """Clean raw data - handle sentinels, outliers, normalize categories."""
    df = df.copy()

    # ManufactureYear: 1001 is sentinel for missing
    df.loc[df["ManufactureYear"] == 1001, "ManufactureYear"] = np.nan

    # OperationalHoursMeter: cap outliers at 99.5th percentile
    hours = df["OperationalHoursMeter"]
    valid_hours = hours[(hours > 0) & hours.notna()]
    if len(valid_hours) > 0:
        cap = valid_hours.quantile(0.995)
        df.loc[df["OperationalHoursMeter"] > cap, "OperationalHoursMeter"] = cap

    # DrivetrainType normalization
    if "DrivetrainType" in df.columns:
        dt_map = {
            "Autoshift": "AutoShift",
            "None or Unspecified": np.nan,
        }
        df["DrivetrainType"] = df["DrivetrainType"].replace(dt_map)

    # Drop near-empty columns (col18, col19 are 99.95% missing)
    drop_cols = ["col18", "col19"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    return df


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
def build_features(train_df, test_df):
    """Build all features for train and test."""
    print("Building features...")

    # Combine for consistent feature engineering
    train_df = clean_data(train_df)
    test_df = clean_data(test_df)

    y = np.log1p(train_df["TargetValue"])

    train_df["_is_test"] = 0
    test_df["_is_test"] = 1
    test_df["TargetValue"] = np.nan
    full = pd.concat([train_df, test_df], ignore_index=True)

    # --- Temporal features ---
    dt = pd.to_datetime(full["TransactionDate"])
    full["SaleYear"] = dt.dt.year
    full["SaleMonth"] = dt.dt.month
    full["SaleQuarter"] = dt.dt.quarter
    full["SaleDayOfWeek"] = dt.dt.dayofweek
    full["SaleWeekOfYear"] = dt.dt.isocalendar().week.astype(int)

    # --- Machine age features ---
    full["MachineAge"] = full["SaleYear"] - full["ManufactureYear"]
    full.loc[full["MachineAge"] < 0, "MachineAge"] = np.nan
    full["MachineAge_sq"] = full["MachineAge"] ** 2
    full["MachineAge_log"] = np.log1p(full["MachineAge"].clip(lower=0))
    full["MachineAge_bin"] = pd.cut(
        full["MachineAge"],
        bins=[-1, 2, 5, 8, 12, 16, 20, 30, 100],
        labels=[0, 1, 2, 3, 4, 5, 6, 7],
    ).astype(float)

    # --- Operational hours features ---
    full["Hours_missing"] = full["OperationalHoursMeter"].isna().astype(int)
    full["Hours_zero"] = (full["OperationalHoursMeter"] == 0).astype(int)
    full.loc[full["OperationalHoursMeter"] == 0, "OperationalHoursMeter"] = np.nan
    full["Hours_per_year"] = full["OperationalHoursMeter"] / (full["MachineAge"].clip(lower=0) + 1)
    full["Age_x_Hours"] = full["MachineAge"] * full["OperationalHoursMeter"]
    full["Hours_log"] = np.log1p(full["OperationalHoursMeter"].clip(lower=0))

    # --- UtilizationTier ordinal ---
    util_map = {"Low": 0, "Medium": 1, "High": 2}
    full["UtilizationTier_ord"] = full["UtilizationTier"].map(util_map)

    # --- FunctionalClassification parsing ---
    func_str = full["FunctionalClassification"].astype(str)

    # Extract machine type (everything before the first dash/number pattern)
    full["MachineType"] = func_str.str.extract(r"^([A-Za-z\s,]+)")[0].str.strip().str.rstrip(" -")

    # Extract numeric ranges
    full["Weight_Min"] = func_str.str.extract(r"(\d+\.?\d*)\s*to\s*\d+\.?\d*\s*Metric Tons").astype(float)
    full["Weight_Max"] = func_str.str.extract(r"\d+\.?\d*\s*to\s*(\d+\.?\d*)\s*Metric Tons").astype(float)
    full["Weight_Mid"] = (full["Weight_Min"] + full["Weight_Max"]) / 2

    full["HP_Min"] = func_str.str.extract(r"(\d+\.?\d*)\s*to\s*\d+\.?\d*\s*Horsepower").astype(float)
    full["HP_Max"] = func_str.str.extract(r"\d+\.?\d*\s*to\s*(\d+\.?\d*)\s*Horsepower").astype(float)
    full["HP_Mid"] = (full["HP_Min"] + full["HP_Max"]) / 2

    full["DigDepth_Min"] = func_str.str.extract(r"(\d+\.?\d*)\s*to\s*\d+\.?\d*\s*Ft").astype(float)
    full["DigDepth_Max"] = func_str.str.extract(r"\d+\.?\d*\s*to\s*(\d+\.?\d*)\s*Ft").astype(float)
    full["DigDepth_Mid"] = (full["DigDepth_Min"] + full["DigDepth_Max"]) / 2

    # --- Spec numeric extraction ---
    full["Spec_NumericClass"] = full["Spec_BaseClass"].astype(str).str.extract(r"(\d+)").astype(float)

    # --- ManufactureYear missing flag ---
    full["MfgYear_missing"] = full["ManufactureYear"].isna().astype(int)

    # --- Frequency encodings ---
    freq_cols = [
        "ProductConfigID", "Spec_FullDescriptor", "Spec_BaseClass",
        "VendorPartnerID", "RegionCode", "MachineType",
        "DataOriginCode", "AssetScaleFactor",
    ]
    for col in freq_cols:
        if col in full.columns:
            freq = full[col].value_counts(normalize=True)
            full[f"{col}_freq"] = full[col].map(freq).fillna(0)

    # --- Drop columns not needed for modeling ---
    drop_cols = [
        "TransactionID", "TargetValue", "TransactionDate",
        "InventoryGroupDescription", "_is_test",
    ]

    # Identify categorical columns BEFORE dropping
    cat_cols = []
    for c in full.columns:
        if c in drop_cols:
            continue
        if not pd.api.types.is_numeric_dtype(full[c]):
            cat_cols.append(c)

    # Fill missing categoricals
    for c in cat_cols:
        full[c] = full[c].fillna("_Missing_").astype(str)

    # Split back
    train_out = full[full.index < len(train_df)].drop(columns=drop_cols, errors="ignore").reset_index(drop=True)
    test_out = full[full.index >= len(train_df)].drop(columns=drop_cols, errors="ignore").reset_index(drop=True)

    # Final cat_cols list (remove any that were dropped)
    cat_cols = [c for c in cat_cols if c in train_out.columns]

    print(f"  Features: {train_out.shape[1]}, Categoricals: {len(cat_cols)}")
    return train_out, y, test_out, cat_cols


# ============================================================
# 4. HIERARCHICAL TARGET ENCODING (OOF, Bayesian smoothed)
# ============================================================
def add_target_encodings(X_train, y_train, X_test, n_splits=5, seed=42):
    """Add multi-level smoothed target encodings computed strictly out-of-fold."""
    print("Computing hierarchical target encodings...")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    global_mean = y_train.mean()
    global_std = y_train.std()

    # Define encoding groups: (columns, smoothing_factor)
    encoding_specs = [
        (["ProductConfigID"], 20),
        (["Spec_FullDescriptor"], 20),
        (["Spec_BaseClass"], 10),
        (["VendorPartnerID"], 10),
        (["RegionCode"], 10),
        (["FunctionalClassification"], 10),
        (["DataOriginCode"], 5),
        (["CabinType"], 5),
        (["MachineType"], 10),
    ]

    # Add interaction encodings - these are the KEY to getting below 0.17
    # Need to create binned MachineAge as string for interaction keys
    X_train["_MachineAge_bin_str"] = X_train["MachineAge_bin"].fillna(-1).astype(int).astype(str)
    X_test["_MachineAge_bin_str"] = X_test["MachineAge_bin"].fillna(-1).astype(int).astype(str)

    interaction_specs = [
        (["ProductConfigID", "_MachineAge_bin_str"], 30),
        (["Spec_FullDescriptor", "_MachineAge_bin_str"], 30),
        (["ProductConfigID", "RegionCode"], 30),
    ]
    encoding_specs.extend(interaction_specs)

    X_tr = X_train.copy()
    X_te = X_test.copy()

    for cols, smooth in encoding_specs:
        # Skip if column doesn't exist
        if not all(c in X_tr.columns for c in cols):
            continue

        if len(cols) == 1:
            key_col = cols[0]
            prefix = key_col
        else:
            # Create composite key
            key_col = "_x_".join(cols)
            prefix = key_col
            X_tr[key_col] = X_tr[cols[0]].astype(str)
            X_te[key_col] = X_te[cols[0]].astype(str)
            for c in cols[1:]:
                X_tr[key_col] = X_tr[key_col] + "___" + X_tr[c].astype(str)
                X_te[key_col] = X_te[key_col] + "___" + X_te[c].astype(str)

        mean_col = f"{prefix}_te_mean"
        std_col = f"{prefix}_te_std"
        count_col = f"{prefix}_te_count"

        X_tr[mean_col] = global_mean
        X_tr[std_col] = global_std
        X_tr[count_col] = 0.0

        # OOF encoding
        tr_key = X_tr[key_col].astype(str)
        te_key = X_te[key_col].astype(str)

        for train_idx, val_idx in kf.split(X_tr):
            fold_y = y_train.iloc[train_idx]
            fold_key = tr_key.iloc[train_idx]

            stats = pd.DataFrame({"k": fold_key, "y": fold_y}).groupby("k")["y"].agg(["count", "mean", "std"])
            smoothed_mean = (stats["count"] * stats["mean"] + smooth * global_mean) / (stats["count"] + smooth)

            val_key = tr_key.iloc[val_idx]
            X_tr.iloc[val_idx, X_tr.columns.get_loc(mean_col)] = val_key.map(smoothed_mean).fillna(global_mean).values
            X_tr.iloc[val_idx, X_tr.columns.get_loc(std_col)] = val_key.map(stats["std"]).fillna(global_std).values
            X_tr.iloc[val_idx, X_tr.columns.get_loc(count_col)] = val_key.map(stats["count"]).fillna(0).values

        # Full data encoding for test
        full_key = tr_key
        full_y = y_train
        full_stats = pd.DataFrame({"k": full_key, "y": full_y}).groupby("k")["y"].agg(["count", "mean", "std"])
        full_smoothed = (full_stats["count"] * full_stats["mean"] + smooth * global_mean) / (full_stats["count"] + smooth)

        X_te[mean_col] = te_key.map(full_smoothed).fillna(global_mean).values
        X_te[std_col] = te_key.map(full_stats["std"]).fillna(global_std).values
        X_te[count_col] = te_key.map(full_stats["count"]).fillna(0).values

        # Drop composite key columns if created
        if len(cols) > 1:
            X_tr.drop(columns=[key_col], inplace=True)
            X_te.drop(columns=[key_col], inplace=True)

    return X_tr, X_te


# ============================================================
# 5. CONVENIENCE: FULL PIPELINE
# ============================================================
def run_full_pipeline():
    """Run the complete pipeline and return processed data."""
    train_raw, test_raw, sample_sub = load_data()
    X_train, y, X_test, cat_cols = build_features(train_raw, test_raw)
    X_train, X_test = add_target_encodings(X_train, y, X_test)
    return X_train, y, X_test, cat_cols, sample_sub


if __name__ == "__main__":
    X_train, y, X_test, cat_cols, sample_sub = run_full_pipeline()
    print(f"\nFinal shapes: X_train={X_train.shape}, X_test={X_test.shape}")
    print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")
    print(f"Numeric columns: {X_train.shape[1] - len(cat_cols)}")
    print("Pipeline complete.")
