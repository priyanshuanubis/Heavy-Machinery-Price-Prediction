"""Leakage-free CatBoost pipeline for the supplied heavy-equipment data only."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
TARGET = "TargetValue"
SEED = 42
CODED_CATEGORIES = {
    "AssetID", "ProductConfigID", "DataOriginCode", "VendorPartnerID", "UtilizationTier",
    "RegionCode", "Spec_FullDescriptor", "Spec_BaseClass", "Spec_SubClass", "Spec_ReleaseSeries",
    "Spec_VariantModifier", "AssetScaleFactor", "FunctionalClassification",
    "InventoryGroupCategory", "InventoryGroupDescription", "CabinType", "DrivetrainType", "Forks",
    *[f"col{i}" for i in range(1, 31)],
}


def log(message: str) -> None:
    print(f"[leakage-free] {message}", flush=True)


def features(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Use only row-local information: no learned mappings or target aggregates."""
    x = raw.drop(columns=["TransactionID"], errors="ignore").copy()
    date = pd.to_datetime(x.pop("TransactionDate"), errors="coerce")
    x["SaleYear"] = date.dt.year
    x["SaleMonth"] = date.dt.month
    x["SaleQuarter"] = date.dt.quarter
    x["SaleWeek"] = date.dt.isocalendar().week.astype("float64")
    x["SaleDayOfWeek"] = date.dt.dayofweek
    x["SaleDayOfYear"] = date.dt.dayofyear

    made_year = pd.to_numeric(x.get("ManufactureYear"), errors="coerce")
    hours = pd.to_numeric(x.get("OperationalHoursMeter"), errors="coerce")
    age = (date.dt.year - made_year).clip(lower=0, upper=100)
    x["AssetAge"] = age
    x["AssetAgeSquared"] = age ** 2
    x["LogOperationalHours"] = np.log1p(hours.clip(lower=0))
    x["HasOperationalHours"] = hours.notna().astype("int8")
    x["HoursPerYear"] = hours / age.replace(0, np.nan)

    descriptor = x["Spec_FullDescriptor"].fillna("").astype(str)
    x["DescriptorLength"] = descriptor.str.len()
    x["DescriptorDigitCount"] = descriptor.str.count(r"\d")
    x["DescriptorFirstToken"] = descriptor.str.split().str[0].fillna("__MISSING__")

    # These interactions are raw row values, not target-derived aggregations.
    x["ConfigRegion"] = x["ProductConfigID"].astype(str) + "__" + x["RegionCode"].fillna("__MISSING__").astype(str)
    x["BaseSubClass"] = x["Spec_BaseClass"].fillna("__MISSING__").astype(str) + "__" + x["Spec_SubClass"].fillna("__MISSING__").astype(str)

    cat_columns = [c for c in x.columns if c in CODED_CATEGORIES or x[c].dtype == "object"]
    for c in cat_columns:
        x[c] = x[c].fillna("__MISSING__").astype(str)
    return x, cat_columns


def rmsle(log_y: np.ndarray, log_prediction: np.ndarray) -> float:
    return float(mean_squared_error(log_y, log_prediction) ** 0.5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", action="store_true", help="Train final model and save Kaggle submission after holdout validation.")
    args = parser.parse_args()

    train = pd.read_csv(ROOT / "data" / "train.csv", low_memory=False)
    test = pd.read_csv(ROOT / "data" / "test.csv", low_memory=False)
    # Invalid target rows only; do not discard valid expensive equipment as "outliers".
    train = train.loc[train[TARGET].notna() & train[TARGET].gt(0)].reset_index(drop=True)
    y = np.log1p(train[TARGET]).to_numpy()
    x, cat_columns = features(train.drop(columns=TARGET))
    x_test, test_cat_columns = features(test)
    assert x.columns.tolist() == x_test.columns.tolist() and cat_columns == test_cat_columns

    bins = pd.qcut(pd.Series(y).rank(method="first"), q=15, labels=False)
    fit_i, valid_i = train_test_split(np.arange(len(x)), test_size=.20, random_state=SEED, stratify=bins)
    log(f"Rows: {len(x):,}; 80/20 split: {len(fit_i):,}/{len(valid_i):,}; features: {x.shape[1]}")
    log("No external data, target encoding, frequency mapping, or validation/test-fitted preprocessing is used.")

    model = CatBoostRegressor(
        loss_function="RMSE", eval_metric="RMSE", iterations=3500, learning_rate=.035,
        depth=9, l2_leaf_reg=5, random_strength=.5, bagging_temperature=.5,
        random_seed=SEED, verbose=200, allow_writing_files=False,
    )
    model.fit(x.iloc[fit_i], y[fit_i], cat_features=cat_columns, eval_set=(x.iloc[valid_i], y[valid_i]), early_stopping_rounds=200, use_best_model=True)
    validation_score = rmsle(y[valid_i], model.predict(x.iloc[valid_i]))
    log(f"Leakage-free 80/20 RMSLE: {validation_score:.6f}")

    out = ROOT / "submissions"; out.mkdir(exist_ok=True)
    (out / "leakage_free_report.txt").write_text(f"holdout_rmsle={validation_score:.6f}\nbest_iteration={model.get_best_iteration()+1}\n", encoding="utf-8")
    if args.submission:
        iterations = max(500, model.get_best_iteration() + 1)
        final_model = CatBoostRegressor(loss_function="RMSE", iterations=iterations, learning_rate=.035, depth=9, l2_leaf_reg=5, random_strength=.5, bagging_temperature=.5, random_seed=SEED, verbose=200, allow_writing_files=False)
        final_model.fit(x, y, cat_features=cat_columns)
        prediction = np.maximum(0, np.expm1(final_model.predict(x_test)))
        pd.DataFrame({"TransactionID": test["TransactionID"], TARGET: prediction}).to_csv(out / "leakage_free_submission.csv", index=False)
        log(f"Saved: {out / 'leakage_free_submission.csv'}")


if __name__ == "__main__":
    main()
