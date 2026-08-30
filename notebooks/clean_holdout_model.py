"""Terminal-first cleaned 80/20 RMSLE baseline."""
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
TARGET, SEED = "TargetValue", 42

CODED_CATEGORIES = {
    "ProductConfigID", "DataOriginCode", "VendorPartnerID", "UtilizationTier", "RegionCode",
    "Spec_FullDescriptor", "Spec_BaseClass", "Spec_SubClass", "Spec_ReleaseSeries",
    "Spec_VariantModifier", "AssetScaleFactor", "FunctionalClassification",
    "InventoryGroupCategory", "InventoryGroupDescription", "CabinType", "DrivetrainType", "Forks",
    *[f"col{i}" for i in range(1, 31)],
}

def log(message): print(f"[clean-holdout] {message}", flush=True)

def prepare_features(df):
    x = df.copy()
    date = pd.to_datetime(x.pop("TransactionDate"), errors="coerce")
    x["sale_year"] = date.dt.year; x["sale_month"] = date.dt.month; x["sale_quarter"] = date.dt.quarter
    x["sale_week"] = date.dt.isocalendar().week.astype(float)
    year = pd.to_numeric(x.get("ManufactureYear"), errors="coerce")
    hours = pd.to_numeric(x.get("OperationalHoursMeter"), errors="coerce")
    age = (date.dt.year - year).clip(0, 80)
    x["asset_age"] = age; x["asset_age_sq"] = age ** 2
    x["log_hours"] = np.log1p(hours.clip(lower=0)); x["has_hours"] = hours.notna().astype("int8")
    descriptor = x["Spec_FullDescriptor"].fillna("").astype(str)
    x["descriptor_length"] = descriptor.str.len(); x["descriptor_digits"] = descriptor.str.count(r"\d")
    # Keys do not generalise. AssetID is deliberately removed rather than treated as a number.
    x = x.drop(columns=["TransactionID", "AssetID"], errors="ignore")
    return x

def main():
    train = pd.read_csv(ROOT / "data" / "train.csv", low_memory=False)
    log(f"Raw rows: {len(train):,}; columns: {train.shape[1]}")
    train = train.drop_duplicates().dropna(subset=[TARGET])
    train = train[train[TARGET] > 0].copy()
    q1, q3 = train[TARGET].quantile([.25, .75]); iqr = q3 - q1
    lower, upper = max(0, q1 - 1.5 * iqr), q3 + 1.5 * iqr
    before = len(train); train = train[train[TARGET].between(lower, upper)].reset_index(drop=True)
    log(f"Removed {before-len(train):,} IQR target outliers; retained {len(train):,} rows ({lower:.0f} to {upper:.0f})")
    x = prepare_features(train.drop(columns=TARGET)); y = np.log1p(train[TARGET])
    empty = [c for c in x if x[c].isna().all()]
    x = x.drop(columns=empty)
    category_columns = [c for c in x if c in CODED_CATEGORIES or x[c].dtype == "object"]
    for c in category_columns: x[c] = x[c].fillna("__MISSING__").astype(str)
    numeric_columns = [c for c in x if c not in category_columns]
    x[numeric_columns] = x[numeric_columns].replace([np.inf, -np.inf], np.nan)
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=.20, random_state=SEED)
    # Medians are learned from the training partition only.
    medians = x_train[numeric_columns].median(); x_train[numeric_columns] = x_train[numeric_columns].fillna(medians); x_valid[numeric_columns] = x_valid[numeric_columns].fillna(medians)
    log(f"80/20 split: {len(x_train):,} train / {len(x_valid):,} validation; features: {x.shape[1]}")
    model = CatBoostRegressor(loss_function="RMSE", iterations=1400, learning_rate=.05, depth=8, l2_leaf_reg=6, random_seed=SEED, verbose=100, allow_writing_files=False)
    model.fit(x_train, y_train, cat_features=category_columns, eval_set=(x_valid, y_valid), early_stopping_rounds=120, use_best_model=True)
    pred = model.predict(x_valid); score = mean_squared_error(y_valid, pred) ** .5
    out = ROOT / "submissions"; out.mkdir(exist_ok=True)
    pd.DataFrame({"actual_log": y_valid, "prediction_log": pred}).to_csv(out / "clean_holdout_predictions.csv", index=False)
    (out / "clean_holdout_report.txt").write_text(f"RMSLE: {score:.6f}\nrows: {len(train)}\noutlier bounds: {lower:.2f}, {upper:.2f}\n", encoding="utf-8")
    log(f"80/20 RMSLE: {score:.6f}")
    log(f"Saved: {out / 'clean_holdout_report.txt'}")

if __name__ == "__main__": main()
