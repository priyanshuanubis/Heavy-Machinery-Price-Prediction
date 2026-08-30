"""Fast, from-scratch RMSLE model with cross-fitted repeat-asset encodings."""
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.model_selection import StratifiedKFold

SEED, FOLDS, TARGET = 2026, 5, "TargetValue"

def make_features(df):
    x = df.drop(columns=["TransactionID"], errors="ignore").copy()
    d = pd.to_datetime(x.pop("TransactionDate"), errors="coerce")
    x["sale_year"] = d.dt.year; x["sale_month"] = d.dt.month; x["sale_week"] = d.dt.isocalendar().week.astype(float)
    year = pd.to_numeric(x.get("ManufactureYear"), errors="coerce")
    hours = pd.to_numeric(x.get("OperationalHoursMeter"), errors="coerce")
    age = (d.dt.year - year).clip(0, 80)
    x["age"] = age; x["age_squared"] = age ** 2; x["log_hours"] = np.log1p(hours.clip(lower=0))
    x["hours_per_age"] = hours / age.replace(0, np.nan); x["has_hours"] = hours.notna().astype("int8")
    desc = x["Spec_FullDescriptor"].fillna("").astype(str)
    x["descriptor_len"] = desc.str.len(); x["descriptor_digits"] = desc.str.count(r"\d")
    x["descriptor_age"] = desc + "_" + pd.cut(age, [-1, 3, 7, 15, 100], labels=False).fillna(-1).astype(str)
    x["config_region"] = x["ProductConfigID"].astype(str) + "_" + x["RegionCode"].fillna("missing").astype(str)
    return x

def add_crossfit_encodings(fit, apply, y, columns, bins):
    """OOF values for fit, full-fit mappings for apply: no target leakage."""
    a = pd.DataFrame(index=fit.index); b = pd.DataFrame(index=apply.index)
    cv = StratifiedKFold(4, shuffle=True, random_state=SEED)
    prior = float(np.mean(y))
    for col in columns:
        fit_key, apply_key = fit[col].fillna("__NA__").astype(str), apply[col].fillna("__NA__").astype(str)
        oof = np.zeros(len(fit))
        for train_i, val_i in cv.split(fit, bins):
            stats = pd.DataFrame({"k": fit_key.iloc[train_i].to_numpy(), "y": y[train_i]}).groupby("k").y.agg(["mean", "count"])
            mapping = (stats["mean"] * stats["count"] + 15 * prior) / (stats["count"] + 15)
            oof[val_i] = fit_key.iloc[val_i].map(mapping).fillna(prior)
        stats = pd.DataFrame({"k": fit_key.to_numpy(), "y": y}).groupby("k").y.agg(["mean", "count"])
        mapping = (stats["mean"] * stats["count"] + 15 * prior) / (stats["count"] + 15)
        a[col + "_te"] = oof.astype("float32")
        b[col + "_te"] = apply_key.map(mapping).fillna(prior).astype("float32")
    return a, b

def main():
    root = Path(__file__).resolve().parents[1]; out = root / "submissions"; out.mkdir(exist_ok=True)
    train, test = pd.read_csv(root / "data/train.csv", low_memory=False), pd.read_csv(root / "data/test.csv", low_memory=False)
    ids, y = test.TransactionID, np.log1p(train.pop(TARGET).clip(lower=0)).to_numpy()
    x, xt = make_features(train), make_features(test)
    cat = x.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    # AssetID and ProductConfigID are identifiers, not ordinal numbers.
    cat += [c for c in ["AssetID", "ProductConfigID"] if c not in cat]
    cat = list(dict.fromkeys(cat))
    numeric = [c for c in x.columns if c not in cat]
    bins = pd.qcut(pd.Series(y).rank(method="first"), 15, labels=False).to_numpy()
    cv = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
    oof, pred, scores = np.zeros(len(x)), np.zeros((len(xt), FOLDS)), []
    for fold, (fi, vi) in enumerate(cv.split(x, bins), 1):
        print(f"fold {fold}/{FOLDS}", flush=True)
        tr_te, va_te = add_crossfit_encodings(x.iloc[fi], x.iloc[vi], y[fi], cat, bins[fi])
        _, te_te = add_crossfit_encodings(x.iloc[fi], xt, y[fi], cat, bins[fi])
        tr_x = pd.concat([x.iloc[fi][numeric].reset_index(drop=True), tr_te.reset_index(drop=True)], axis=1).replace([np.inf, -np.inf], np.nan)
        va_x = pd.concat([x.iloc[vi][numeric].reset_index(drop=True), va_te.reset_index(drop=True)], axis=1).replace([np.inf, -np.inf], np.nan)
        test_x = pd.concat([xt[numeric].reset_index(drop=True), te_te.reset_index(drop=True)], axis=1).replace([np.inf, -np.inf], np.nan)
        model = LGBMRegressor(n_estimators=3000, learning_rate=.025, num_leaves=127, min_child_samples=25, colsample_bytree=.85, subsample=.85, reg_lambda=2, reg_alpha=.1, random_state=SEED+fold, n_jobs=-1)
        model.fit(tr_x, y[fi], eval_set=[(va_x, y[vi])], eval_metric="rmse", callbacks=[early_stopping(150), log_evaluation(200)])
        oof[vi] = model.predict(va_x); pred[:, fold-1] = model.predict(test_x)
        score = float(np.sqrt(np.mean((y[vi]-oof[vi])**2))); scores.append(score); print(f"fold RMSLE {score:.6f}", flush=True)
    score = float(np.sqrt(np.mean((y-oof)**2)))
    pd.DataFrame({"fold":range(1,FOLDS+1),"rmsle":scores}).to_csv(out / "asset_target_lgbm_cv.csv", index=False)
    pd.DataFrame({"TransactionID":ids, TARGET:np.maximum(0,np.expm1(pred.mean(1)))}).to_csv(out / "asset_target_lgbm_submission.csv", index=False)
    print(f"OOF RMSLE: {score:.6f}", flush=True)

if __name__ == "__main__": main()
