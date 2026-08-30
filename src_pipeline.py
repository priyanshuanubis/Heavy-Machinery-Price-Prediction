import os
import re
import numpy as np
import pandas as pd

def load_data(data_dir=r"e:\HeavyEquipmentML\data"):
    print(f"Loading data from: {data_dir}")
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    sample_sub_path = os.path.join(data_dir, "sample_submission.csv")

    train_raw = pd.read_csv(train_path, low_memory=False)
    test_raw = pd.read_csv(test_path, low_memory=False)
    sample_sub = pd.read_csv(sample_sub_path)
    return train_raw, test_raw, sample_sub

def build_complete_overhaul_features(train_raw, test_raw):
    print("Building clean overhauled feature pipeline...")
    train_df = train_raw.copy()
    test_df = test_raw.copy()

    y = np.log1p(train_df["TargetValue"])

    train_df["is_test_flag"] = 0
    test_df["is_test_flag"] = 1
    test_df["TargetValue"] = np.nan

    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_df["TransactionDate_dt"] = pd.to_datetime(full_df["TransactionDate"])

    # Clean manufacture year (1001 signifies missing)
    full_df["ManufactureYear_missing"] = (full_df["ManufactureYear"] == 1001).astype(int)
    full_df.loc[full_df["ManufactureYear"] == 1001, "ManufactureYear"] = np.nan

    # OperationalHoursMeter cleanup
    full_df["OperationalHoursMeter_missing"] = full_df["OperationalHoursMeter"].isna() | (full_df["OperationalHoursMeter"] == 0)
    full_df.loc[full_df["OperationalHoursMeter"] == 0, "OperationalHoursMeter"] = np.nan

    # Dates
    sale_date = full_df["TransactionDate_dt"]
    full_df["SaleYear"] = sale_date.dt.year
    full_df["SaleQuarter"] = sale_date.dt.quarter
    full_df["SaleMonth"] = sale_date.dt.month
    full_df["SaleWeek"] = sale_date.dt.isocalendar().week.astype(int)
    full_df["SaleDay"] = sale_date.dt.day
    full_df["SaleDayOfWeek"] = sale_date.dt.dayofweek
    full_df["SaleDayOfYear"] = sale_date.dt.dayofyear

    # Machine Age
    full_df["MachineAge"] = full_df["SaleYear"] - full_df["ManufactureYear"]
    full_df.loc[full_df["MachineAge"] < 0, "MachineAge"] = np.nan
    full_df["Hours_Per_Year"] = full_df["OperationalHoursMeter"] / (full_df["MachineAge"] + 1.0)
    full_df["Age_x_Hours"] = full_df["MachineAge"] * full_df["OperationalHoursMeter"]

    # Functional specs regex
    func_str = full_df["FunctionalClassification"].astype(str)
    full_df["Weight_Min_Tons"] = func_str.str.extract(r"(\d+\.?\d*)\s*to\s*\d+\.?\d*\s*Metric Tons").astype(float)
    full_df["Weight_Max_Tons"] = func_str.str.extract(r"\d+\.?\d*\s*to\s*(\d+\.?\d*)\s*Metric Tons").astype(float)
    full_df["HP_Min"] = func_str.str.extract(r"(\d+\.?\d*)\s*to\s*\d+\.?\d*\s*Horsepower").astype(float)
    full_df["HP_Max"] = func_str.str.extract(r"\d+\.?\d*\s*to\s*(\d+\.?\d*)\s*Horsepower").astype(float)
    full_df["Dig_Depth_Min_Ft"] = func_str.str.extract(r"(\d+\.?\d*)\s*to\s*\d+\.?\d*\s*Ft").astype(float)
    full_df["Dig_Depth_Max_Ft"] = func_str.str.extract(r"\d+\.?\d*\s*to\s*(\d+\.?\d*)\s*Ft").astype(float)

    spec = full_df["Spec_BaseClass"].astype(str)
    full_df["Spec_NumericClass"] = spec.str.extract(r"(\d+)").astype(float)

    # Frequency encodings
    high_card_cols = [
        "ProductConfigID", "VendorPartnerID", "RegionCode",
        "Spec_FullDescriptor", "Spec_BaseClass", "Spec_SubClass", "Spec_ReleaseSeries"
    ]
    for col in high_card_cols:
        freq_map = full_df[col].value_counts(normalize=True).to_dict()
        full_df[f"{col}_freq"] = full_df[col].map(freq_map).fillna(0.0)

    cols_to_remove = ["InventoryGroupDescription", "TransactionID", "TransactionDate", "TransactionDate_dt", "TargetValue"]
    full_df.drop(columns=cols_to_remove, inplace=True, errors="ignore")

    train_proc = full_df[full_df["is_test_flag"] == 0].drop(columns=["is_test_flag"])
    test_proc = full_df[full_df["is_test_flag"] == 1].drop(columns=["is_test_flag"])

    return train_proc, y, test_proc

def add_smooth_oof_target_encoding(X_train, y_train, X_test, target_cols, n_splits=5, smoothing=20):
    print("Computing Multi-Tier Target Encodings...")
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    global_mean = y_train.mean()
    X_tr_enc = X_train.copy()
    X_te_enc = X_test.copy()

    for col in target_cols:
        col_name = str(col)
        tr_cat = X_train[col].astype(str)
        te_cat = X_test[col].astype(str)

        col_enc_mean = f"{col_name}_target_mean"
        col_enc_std = f"{col_name}_target_std"

        X_tr_enc[col_enc_mean] = global_mean
        X_tr_enc[col_enc_std] = y_train.std()
        X_te_enc[col_enc_mean] = global_mean
        X_te_enc[col_enc_std] = y_train.std()

        for train_idx, val_idx in kf.split(X_train):
            tr_c = tr_cat.iloc[train_idx]
            val_c = tr_cat.iloc[val_idx]
            tr_y = y_train.iloc[train_idx]
            
            stats = pd.DataFrame({"cat": tr_c, "target": tr_y}).groupby("cat")["target"].agg(["count", "mean", "std"])
            smooth = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
            
            X_tr_enc.iloc[val_idx, X_tr_enc.columns.get_loc(col_enc_mean)] = val_c.map(smooth).fillna(global_mean).values
            X_tr_enc.iloc[val_idx, X_tr_enc.columns.get_loc(col_enc_std)] = val_c.map(stats["std"]).fillna(y_train.std()).values

        stats_full = pd.DataFrame({"cat": tr_cat, "target": y_train}).groupby("cat")["target"].agg(["count", "mean", "std"])
        smooth_full = (stats_full["count"] * stats_full["mean"] + smoothing * global_mean) / (stats_full["count"] + smoothing)
        
        X_te_enc[col_enc_mean] = te_cat.map(smooth_full).fillna(global_mean).values
        X_te_enc[col_enc_std] = te_cat.map(stats_full["std"]).fillna(y_train.std()).values

    return X_tr_enc, X_te_enc
