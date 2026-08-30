"""Leakage-safe, asset-aware RMSLE model for Heavy Equipment ML."""
from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import StratifiedKFold


SEED = 2026
TARGET = "TargetValue"
FOLDS = 3


def paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[1]
    output = root / "submissions"
    output.mkdir(exist_ok=True)
    return root / "data", output


def feature_engineer(source: pd.DataFrame) -> pd.DataFrame:
    """Build only features available at prediction time; retain AssetID deliberately."""
    df = source.copy()
    date = pd.to_datetime(df.pop("TransactionDate"), errors="coerce")
    df["SaleYear"] = date.dt.year.astype("float32")
    df["SaleMonth"] = date.dt.month.astype("float32")
    df["SaleQuarter"] = date.dt.quarter.astype("float32")
    df["SaleWeek"] = date.dt.isocalendar().week.astype("float32")
    df["SaleDayOfWeek"] = date.dt.dayofweek.astype("float32")
    df["SaleYearMonth"] = (date.dt.year * 12 + date.dt.month).astype("float32")

    year = pd.to_numeric(df.get("ManufactureYear"), errors="coerce")
    hours = pd.to_numeric(df.get("OperationalHoursMeter"), errors="coerce")
    age = (date.dt.year - year).clip(0, 80)
    df["AssetAge"] = age.astype("float32")
    df["AssetAge2"] = (age ** 2).astype("float32")
    df["LogHours"] = np.log1p(hours.clip(lower=0)).astype("float32")
    df["HoursPerAge"] = (hours / age.replace(0, np.nan)).astype("float32")
    df["HoursAge"] = (np.log1p(hours.clip(lower=0)) * age).astype("float32")
    df["HasHours"] = hours.notna().astype("int8")
    df["HasYear"] = year.notna().astype("int8")

    descriptor = df["Spec_FullDescriptor"].fillna("").astype(str)
    df["DescriptorLength"] = descriptor.str.len().astype("float32")
    df["DescriptorDigits"] = descriptor.str.count(r"\d").astype("float32")
    df["DescriptorFirstToken"] = descriptor.str.split().str[0].fillna("__MISSING__")

    for left, right, name in [
        ("ProductConfigID", "RegionCode", "ConfigRegion"),
        ("ProductConfigID", "UtilizationTier", "ConfigUtilization"),
        ("Spec_BaseClass", "Spec_SubClass", "BaseSubClass"),
        ("Spec_FullDescriptor", "AssetAge", "DescriptorAgeBucket"),
    ]:
        if left in df and right in df:
            if right == "AssetAge":
                right_values = pd.cut(age, [-1, 3, 7, 15, 80], labels=False).fillna(-1).astype(str)
            else:
                right_values = df[right].fillna("__MISSING__").astype(str)
            df[name] = df[left].fillna("__MISSING__").astype(str) + "__" + right_values

    # TransactionID is a row key, while AssetID is a real repeat-sale signal.
    return df.drop(columns=["TransactionID"], errors="ignore")


def categoricals(df: pd.DataFrame) -> list[str]:
    cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    # Numeric identifiers encode labels rather than an ordered quantity.
    for col in ["AssetID", "ProductConfigID"]:
        if col in df.columns and col not in cols:
            cols.append(col)
    for col in cols:
        df[col] = df[col].fillna("__MISSING__").astype(str)
    return cols


def rmsle(y_log: np.ndarray, pred_log: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_log - pred_log) ** 2)))


def main() -> None:
    data_dir, output_dir = paths()
    train = pd.read_csv(data_dir / "train.csv", low_memory=False)
    test = pd.read_csv(data_dir / "test.csv", low_memory=False)
    y = np.log1p(train.pop(TARGET).clip(lower=0)).to_numpy()
    test_ids = test["TransactionID"].copy()

    X = feature_engineer(train)
    X_test = feature_engineer(test)
    cat_cols = categoricals(X)
    categoricals(X_test)
    assert X.columns.tolist() == X_test.columns.tolist()

    bins = pd.qcut(pd.Series(y).rank(method="first"), q=15, labels=False).to_numpy()
    cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X), dtype=np.float64)
    test_predictions = np.zeros((len(X_test), FOLDS), dtype=np.float64)
    fold_scores, best_iterations = [], []

    for fold, (fit_idx, val_idx) in enumerate(cv.split(X, bins), 1):
        print(f"Fold {fold}/{FOLDS}", flush=True)
        model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            iterations=1600,
            learning_rate=0.05,
            depth=8,
            l2_leaf_reg=6.0,
            random_strength=0.35,
            bagging_temperature=0.5,
            border_count=128,
            random_seed=SEED + fold,
            verbose=250,
            allow_writing_files=False,
            thread_count=-1,
        )
        model.fit(
            X.iloc[fit_idx], y[fit_idx], cat_features=cat_cols,
            eval_set=(X.iloc[val_idx], y[val_idx]),
            early_stopping_rounds=120, use_best_model=True,
        )
        oof[val_idx] = model.predict(X.iloc[val_idx])
        test_predictions[:, fold - 1] = model.predict(X_test)
        score = rmsle(y[val_idx], oof[val_idx])
        fold_scores.append(score)
        best_iterations.append(model.get_best_iteration() + 1)
        print(f"Fold {fold} RMSLE: {score:.6f}; iteration: {best_iterations[-1]}", flush=True)
        del model
        gc.collect()

    oof_score = rmsle(y, oof)
    report = pd.DataFrame({"fold": range(1, FOLDS + 1), "rmsle": fold_scores, "best_iteration": best_iterations})
    report.to_csv(output_dir / "asset_aware_cv_scores.csv", index=False)
    pd.DataFrame({"actual_log": y, "prediction_log": oof}).to_csv(output_dir / "asset_aware_oof.csv", index=False)

    predictions = np.maximum(0, np.expm1(test_predictions.mean(axis=1)))
    submission = pd.DataFrame({"TransactionID": test_ids, TARGET: predictions})
    submission.to_csv(output_dir / "asset_aware_catboost_submission.csv", index=False)
    print(f"OOF RMSLE: {oof_score:.6f} +/- {np.std(fold_scores):.6f}", flush=True)
    print(f"Saved: {output_dir / 'asset_aware_catboost_submission.csv'}", flush=True)


if __name__ == "__main__":
    main()
