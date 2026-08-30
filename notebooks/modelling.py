from __future__ import annotations

import argparse
import gc
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


warnings.filterwarnings("ignore")
RANDOM_STATE = 42
DEBUG_SAMPLE_SIZE = 30_000
TUNING_SAMPLE_SIZE = 50_000
TARGET = "TargetValue"
REQUIRED_FEATURES = [
    "TransactionYear",
    "TransactionQuarter",
    "AssetAge",
    "DescriptorLength",
    "HasOperationalHours",
    "HasVariantModifier",
]
BASE_CATEGORICAL_COLUMNS = [
    "AssetID",
    "ProductConfigID",
    "DataOriginCode",
    "VendorPartnerID",
    "UtilizationTier",
    "RegionCode",
    "Spec_FullDescriptor",
    "Spec_BaseClass",
    "Spec_SubClass",
    "Spec_ReleaseSeries",
    "Spec_VariantModifier",
    "FunctionalClassification",
    "AssetScaleFactor",
    "InventoryGroupCategory",
    "InventoryGroupDescription",
    "CabinType",
    "DrivetrainType",
    "Forks",
    "col1",
    "col3",
    "col4",
    "col5",
    "col6",
    "col7",
    "col8",
    "col9",
    "col10",
    "col11",
    "col12",
    "col13",
    "col14",
    "col15",
    "col16",
    "col18",
    "col19",
    "col20",
    "col21",
    "col22",
    "col23",
    "col24",
    "col25",
    "col27",
    "col28",
    "col29",
    "col30",
]
TEXT_FEATURE_COLUMNS = [
    "Spec_FullDescriptor",
    "FunctionalClassification",
    "InventoryGroupDescription",
]
DROP_RAW_ID_COLUMNS = ["TransactionID", "AssetID"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Heavy Equipment Selling Price Prediction runner")
    parser.add_argument(
        "--mode",
        choices=["debug", "competition"],
        default="debug",
        help="Use debug for fast iteration, competition for the full pipeline.",
    )
    parser.add_argument(
        "--run-baselines",
        action="store_true",
        help="Run Linear Regression, Random Forest, and LightGBM baselines in competition mode.",
    )
    return parser.parse_args()


def resolve_paths() -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parent.parent
    kaggle_data = Path("/kaggle/input/competitions/heavy-equipment-selling-price-prediction-challenge")
    local_candidates = [repo_root / "data", Path.cwd() / "data"]
    data_dir = kaggle_data if kaggle_data.exists() else next(
        (path for path in local_candidates if path.exists()),
        repo_root / "data",
    )
    output_dir = Path("/kaggle/working") if kaggle_data.exists() else repo_root / "submissions"
    output_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, output_dir


def log(message: str) -> None:
    print(f"[modelling] {message}", flush=True)


def add_base_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    for column in BASE_CATEGORICAL_COLUMNS:
        if column in df.columns:
            df[column] = df[column].fillna("__MISSING__").astype(str)
    transaction_date = pd.to_datetime(df["TransactionDate"], errors="coerce")
    df["TransactionYear"] = transaction_date.dt.year
    df["TransactionQuarter"] = transaction_date.dt.quarter
    df["TransactionMonth"] = transaction_date.dt.month
    df["TransactionWeek"] = transaction_date.dt.isocalendar().week.astype("float64")
    df["TransactionDay"] = transaction_date.dt.day
    df["TransactionDayOfWeek"] = transaction_date.dt.dayofweek
    df["TransactionSeason"] = ((df["TransactionMonth"] % 12) // 3 + 1).astype("float64")
    df["TransactionYearMonth"] = df["TransactionYear"] * 100 + df["TransactionMonth"]
    df["IsMonthEnd"] = transaction_date.dt.is_month_end.astype("float64")
    df["IsQuarterEnd"] = transaction_date.dt.is_quarter_end.astype("float64")

    manufacture_year = pd.to_numeric(df.get("ManufactureYear"), errors="coerce")
    df["AssetAge"] = (df["TransactionYear"] - manufacture_year).clip(lower=0, upper=100)
    df["AssetAgeSquared"] = df["AssetAge"] ** 2
    df["AssetAgeCube"] = df["AssetAge"] ** 3
    df["HasManufactureYear"] = manufacture_year.notna().astype("int8")
    df["AgeBucket"] = pd.cut(
        df["AssetAge"],
        bins=[-1, 2, 5, 10, 20, 100],
        labels=["0_2", "3_5", "6_10", "11_20", "21_plus"],
    ).astype(str)

    descriptor = df.get("Spec_FullDescriptor", pd.Series("", index=df.index)).fillna("").astype(str)
    df["DescriptorLength"] = descriptor.str.len()
    df["DescriptorCharacterCount"] = df["DescriptorLength"]
    df["DescriptorWordCount"] = descriptor.str.split().str.len()
    df["DescriptorDigitCount"] = descriptor.str.count(r"\d")
    df["DescriptorHasNumber"] = descriptor.str.contains(r"\d", regex=True).astype("int8")
    df["DescriptorFirstToken"] = descriptor.str.split().str[0].fillna("__MISSING__")

    hours = pd.to_numeric(df.get("OperationalHoursMeter"), errors="coerce")
    df["HasOperationalHours"] = hours.notna().astype("int8")
    df["OperationalHoursMissing"] = hours.isna().astype("int8")
    df["LogOperationalHours"] = np.log1p(hours.clip(lower=0))
    df["OperationalHoursPerYear"] = hours / df["AssetAge"].replace(0, np.nan)
    df["HoursAgeInteraction"] = df["LogOperationalHours"] * df["AssetAge"]
    df["HoursPerDescriptorWord"] = df["LogOperationalHours"] / df["DescriptorWordCount"].replace(0, np.nan)
    df["HasVariantModifier"] = df.get("Spec_VariantModifier", pd.Series(index=df.index)).notna().astype("int8")
    df["MissingFeatureCount"] = df.isna().sum(axis=1)

    for left, right, name in [
        ("RegionCode", "Spec_BaseClass", "RegionBaseClass"),
        ("VendorPartnerID", "UtilizationTier", "VendorUtilization"),
        ("Spec_BaseClass", "Spec_SubClass", "BaseSubClass"),
        ("RegionCode", "InventoryGroupCategory", "RegionInventoryGroup"),
        ("AssetScaleFactor", "FunctionalClassification", "ScaleFunction"),
        ("ProductConfigID", "Spec_BaseClass", "ConfigBaseClass"),
    ]:
        if left in df and right in df:
            df[name] = (
                df[left].fillna("__MISSING__").astype(str)
                + "__"
                + df[right].fillna("__MISSING__").astype(str)
            )

    return df.drop(columns=["TransactionDate", *DROP_RAW_ID_COLUMNS], errors="ignore")


def fit_category_maps(
    frame: pd.DataFrame,
    categorical_columns: list[str],
    rare_min_count: int = 20,
) -> tuple[dict[str, dict[str, float]], dict[str, set[str]]]:
    frequency_maps: dict[str, dict[str, float]] = {}
    rare_values: dict[str, set[str]] = {}
    for column in categorical_columns:
        values = frame[column].fillna("__MISSING__").astype(str)
        counts = values.value_counts(dropna=False)
        frequency_maps[column] = (counts / len(frame)).to_dict()
        rare_values[column] = set(counts[counts < rare_min_count].index)
    return frequency_maps, rare_values


def apply_category_maps(
    frame: pd.DataFrame,
    categorical_columns: list[str],
    frequency_maps: dict[str, dict[str, float]],
    rare_values: dict[str, set[str]],
) -> pd.DataFrame:
    df = frame.copy()
    for column in categorical_columns:
        values = df[column].fillna("__MISSING__").astype(str)
        df[f"{column}_Frequency"] = values.map(frequency_maps[column]).fillna(0).astype("float32")
        df[column] = values.where(~values.isin(rare_values[column]), "__RARE__")
    return df


def regression_metrics(y_log_true: pd.Series, y_log_pred: np.ndarray) -> dict[str, float]:
    y_true = np.expm1(y_log_true)
    y_pred = np.maximum(0, np.expm1(y_log_pred))
    return {
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "RMSLE": mean_squared_error(np.log1p(y_true), np.log1p(y_pred)) ** 0.5,
    }


def make_stratification_bins(y_log: pd.Series, n_bins: int = 12) -> np.ndarray:
    ranked = y_log.rank(method="first")
    return pd.qcut(ranked, q=min(n_bins, len(y_log)), labels=False, duplicates="drop").astype(int)


def add_target_encoding_features(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    y_log: pd.Series,
    columns: list[str],
    strat_bins: np.ndarray,
    n_splits: int = 5,
    smoothing: float = 25.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_encoded = pd.DataFrame(index=train_frame.index)
    test_encoded = pd.DataFrame(index=test_frame.index)
    global_mean = float(y_log.mean())
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    for column in columns:
        train_col = train_frame[column].fillna("__MISSING__").astype(str)
        test_col = test_frame[column].fillna("__MISSING__").astype(str)
        oof_values = pd.Series(index=train_frame.index, dtype="float64")

        for fit_idx, val_idx in cv.split(train_frame, strat_bins):
            fit_values = train_col.iloc[fit_idx]
            fit_target = y_log.iloc[fit_idx]
            stats = (
                pd.DataFrame({"value": fit_values, "target": fit_target})
                .groupby("value")["target"]
                .agg(["mean", "count"])
            )
            smooth = (stats["mean"] * stats["count"] + smoothing * global_mean) / (stats["count"] + smoothing)
            oof_values.iloc[val_idx] = train_col.iloc[val_idx].map(smooth).fillna(global_mean)

        full_stats = (
            pd.DataFrame({"value": train_col, "target": y_log})
            .groupby("value")["target"]
            .agg(["mean", "count"])
        )
        full_smooth = (
            (full_stats["mean"] * full_stats["count"] + smoothing * global_mean)
            / (full_stats["count"] + smoothing)
        )
        train_encoded[f"{column}_TargetMean"] = oof_values.astype("float32")
        test_encoded[f"{column}_TargetMean"] = test_col.map(full_smooth).fillna(global_mean).astype("float32")

    return train_encoded, test_encoded


def save_feature_importance(importance: pd.DataFrame, output_dir: Path, filename: str, title: str) -> None:
    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance.head(25), x="importance", y="feature", color="teal")
    plt.title(title)
    plt.tight_layout()
    figure_path = output_dir / filename
    plt.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved feature importance plot: {figure_path.resolve()}")


def main() -> None:
    args = parse_args()
    mode = args.mode
    sns.set_theme(style="whitegrid", palette="deep")
    pd.set_option("display.max_columns", 100)

    data_dir, output_dir = resolve_paths()
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    meta_path = data_dir / "metadata.csv"

    assert train_path.exists() and test_path.exists(), f"Dataset not found in {data_dir.resolve()}"

    log(f"Running in {mode} mode")
    log(f"Loading data from {data_dir.resolve()}")
    train_raw = pd.read_csv(train_path, low_memory=False)
    test_raw = pd.read_csv(test_path, low_memory=False)
    metadata = pd.read_csv(meta_path) if meta_path.exists() else pd.DataFrame()

    if mode == "debug":
        train_raw = train_raw.sample(
            n=min(DEBUG_SAMPLE_SIZE, len(train_raw)),
            random_state=RANDOM_STATE,
        ).reset_index(drop=True)
        log(f"Using reproducible {len(train_raw):,}-row training sample")
    else:
        log("Using the complete training dataset")

    log(f"Train shape: {train_raw.shape}")
    log(f"Test shape: {test_raw.shape}")
    if not metadata.empty:
        log(f"Metadata shape: {metadata.shape}")

    log("Running feature engineering")
    train_base = add_base_features(train_raw.drop(columns=[TARGET]))
    test_base = add_base_features(test_raw)
    raw_categorical_columns = [
        column
        for column in train_base.columns
        if column in BASE_CATEGORICAL_COLUMNS or train_base[column].dtype == "object"
    ]
    frequency_maps, rare_values = fit_category_maps(train_base, raw_categorical_columns)
    X_all = apply_category_maps(train_base, raw_categorical_columns, frequency_maps, rare_values)
    X_test = apply_category_maps(test_base, raw_categorical_columns, frequency_maps, rare_values)
    y = np.log1p(train_raw[TARGET].clip(lower=0))

    assert all(feature in X_all.columns for feature in REQUIRED_FEATURES)
    assert list(X_all.columns) == list(X_test.columns)
    log(f"Features after engineering: {X_all.shape[1]}")

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_all,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )
    object_columns = X_all.select_dtypes(include=["object", "category"]).columns.tolist()
    text_columns = [column for column in TEXT_FEATURE_COLUMNS if column in X_all.columns]
    cat_columns = [column for column in object_columns if column not in text_columns]
    numeric_columns = [column for column in X_all.columns if column not in cat_columns + text_columns]

    for frame in (X_train, X_valid, X_all, X_test):
        frame[cat_columns + text_columns] = frame[cat_columns + text_columns].fillna("__MISSING__").astype(str)

    log(
        f"Numeric features: {len(numeric_columns)} | "
        f"Categorical features: {len(cat_columns)} | Text features: {len(text_columns)}"
    )

    results: list[dict[str, float | str]] = []

    if mode == "competition" and args.run_baselines:
        log("Training baseline models")
        baseline_cat_columns = cat_columns.copy()
        low_cardinality = [c for c in baseline_cat_columns if X_train[c].nunique() <= 30]
        linear_columns = numeric_columns + low_cardinality
        linear_preprocessor = ColumnTransformer(
            [
                ("num", SimpleImputer(strategy="median"), numeric_columns),
                ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), low_cardinality),
            ]
        )
        linear_model = Pipeline([("preprocess", linear_preprocessor), ("model", LinearRegression())])
        linear_model.fit(X_train[linear_columns], y_train)
        results.append(
            {
                "Model": "Linear Regression",
                **regression_metrics(y_valid, linear_model.predict(X_valid[linear_columns])),
            }
        )
        log("Linear Regression done")

        tree_preprocessor = ColumnTransformer(
            [
                ("num", SimpleImputer(strategy="median"), numeric_columns),
                ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), baseline_cat_columns + text_columns),
            ]
        )
        rf_model = Pipeline(
            [
                ("preprocess", tree_preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=80,
                        min_samples_leaf=2,
                        max_features=0.7,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
        rf_sample = X_train.sample(n=min(50_000, len(X_train)), random_state=RANDOM_STATE)
        rf_model.fit(rf_sample, y_train.loc[rf_sample.index])
        results.append(
            {
                "Model": "Random Forest (50k sample)",
                **regression_metrics(y_valid, rf_model.predict(X_valid)),
            }
        )
        log("Random Forest done")

        lgb_train = X_train.copy()
        lgb_valid = X_valid.copy()
        for column in baseline_cat_columns + text_columns:
            categories = pd.Index(sorted(X_all[column].astype(str).fillna("missing").unique()))
            lgb_train[column] = pd.Categorical(
                lgb_train[column].astype(str).fillna("missing"),
                categories=categories,
            ).codes
            lgb_valid[column] = pd.Categorical(
                lgb_valid[column].astype(str).fillna("missing"),
                categories=categories,
            ).codes
        lgb_model = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
        )
        lgb_model.fit(
            lgb_train,
            y_train,
            eval_set=[(lgb_valid, y_valid)],
            eval_metric="rmse",
            callbacks=[early_stopping(50), log_evaluation(100)],
        )
        results.append(
            {
                "Model": "LightGBM",
                **regression_metrics(y_valid, lgb_model.predict(lgb_valid)),
            }
        )
        log("LightGBM done")
    elif mode == "competition":
        log("Skipping baselines to spend runtime budget on CatBoost optimisation")
    else:
        log("Skipping Linear Regression, Random Forest, and LightGBM in debug mode")

    results_df = pd.DataFrame(results)

    log("Training CatBoost debug model" if mode == "debug" else "Skipping standalone CatBoost debug fit")
    if mode == "debug":
        catboost_model = CatBoostRegressor(
            loss_function="RMSE",
            iterations=700,
            learning_rate=0.05,
            depth=8,
            l2_leaf_reg=5,
            random_seed=RANDOM_STATE,
            verbose=100,
            allow_writing_files=False,
        )
        catboost_model.fit(
            X_train,
            y_train,
            cat_features=cat_columns,
            text_features=text_columns,
            eval_set=(X_valid, y_valid),
            early_stopping_rounds=80,
        )
        catboost_pred = catboost_model.predict(X_valid)
        results.append({"Model": "CatBoost (debug)", **regression_metrics(y_valid, catboost_pred)})
        results_df = pd.DataFrame(results)
        importance = pd.DataFrame(
            {
                "feature": X_all.columns,
                "importance": catboost_model.get_feature_importance(),
            }
        ).sort_values("importance", ascending=False).reset_index(drop=True)
        save_feature_importance(
            importance,
            output_dir,
            "catboost_feature_importance_debug.png",
            "CatBoost feature importance (debug holdout model)",
        )
    else:
        importance = pd.DataFrame({"feature": X_all.columns, "importance": np.nan})

    if not results_df.empty:
        results_path = output_dir / f"model_comparison_{mode}.csv"
        results_df.round(6).to_csv(results_path, index=False)
        log(f"Saved model comparison: {results_path.resolve()}")
        log(f"\n{results_df.round(5).to_string(index=False)}")

    importance["cumulative_importance"] = importance["importance"].cumsum() / importance["importance"].sum()
    selected_features = importance.loc[importance["cumulative_importance"] <= 0.98, "feature"].tolist()
    if not selected_features and mode == "debug":
        selected_features = [importance.iloc[0]["feature"]]
    selected_cat_columns = [column for column in cat_columns if column in selected_features]
    selected_text_columns = [column for column in text_columns if column in selected_features]

    if mode == "competition":
        selected_features = X_all.columns.tolist()
        selected_cat_columns = cat_columns
        selected_text_columns = text_columns

    log(f"Selected {len(selected_features)} of {X_all.shape[1]} features")

    tuning_candidates = []
    tuning_results: list[dict[str, float]] = []

    if mode == "competition":
        strat_bins = make_stratification_bins(y)
        tune_fit_idx, tune_valid_idx = train_test_split(
            np.arange(len(X_all)),
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=strat_bins,
        )
        tune_fit_frame = X_all.iloc[tune_fit_idx]
        tune_fit_target = y.iloc[tune_fit_idx]
        tune_valid_frame = X_all.iloc[tune_valid_idx]
        tune_valid_target = y.iloc[tune_valid_idx]
        tuning_fit = tune_fit_frame.sample(
            n=min(TUNING_SAMPLE_SIZE, len(tune_fit_frame)),
            random_state=RANDOM_STATE,
        )
        tuning_y = tune_fit_target.loc[tuning_fit.index]

        tuning_candidates = [
            {
                "depth": 8,
                "learning_rate": 0.035,
                "l2_leaf_reg": 4,
                "bagging_temperature": 0.5,
                "random_strength": 0.5,
            },
            {
                "depth": 9,
                "learning_rate": 0.03,
                "l2_leaf_reg": 5,
                "bagging_temperature": 1.0,
                "random_strength": 1.0,
            },
            {
                "depth": 10,
                "learning_rate": 0.028,
                "l2_leaf_reg": 6,
                "bagging_temperature": 1.5,
                "random_strength": 1.5,
            },
        ]

        log("Starting CatBoost tuning on a stratified holdout")
        for idx, parameters in enumerate(tuning_candidates, start=1):
            log(f"Tuning candidate {idx}/{len(tuning_candidates)}: {parameters}")
            candidate = CatBoostRegressor(
                loss_function="RMSE",
                eval_metric="RMSE",
                iterations=1800,
                random_seed=RANDOM_STATE,
                verbose=False,
                allow_writing_files=False,
                **parameters,
            )
            candidate.fit(
                tuning_fit[selected_features],
                tuning_y,
                cat_features=selected_cat_columns,
                text_features=selected_text_columns,
                eval_set=(tune_valid_frame[selected_features], tune_valid_target),
                early_stopping_rounds=150,
                use_best_model=True,
            )
            candidate_pred = candidate.predict(tune_valid_frame[selected_features])
            tuning_results.append(
                {
                    **parameters,
                    "best_iteration": candidate.get_best_iteration() + 1,
                    "RMSLE": regression_metrics(tune_valid_target, candidate_pred)["RMSLE"],
                }
            )
        tuning_results_df = pd.DataFrame(tuning_results).sort_values("RMSLE").reset_index(drop=True)
        best_parameters = tuning_results_df.iloc[0][
            ["depth", "learning_rate", "l2_leaf_reg", "bagging_temperature", "random_strength"]
        ].to_dict()
        best_parameters["depth"] = int(best_parameters["depth"])
        tuned_iteration_cap = max(1400, int(tuning_results_df.iloc[0]["best_iteration"]) + 250)
        log(f"Best tuning RMSLE: {tuning_results_df.iloc[0]['RMSLE']:.5f}")

        log("Starting 5-fold stratified CatBoost CV ensemble")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        catboost_oof_predictions = np.zeros(len(X_all))
        catboost_test_fold_predictions = np.zeros((len(X_test), 5))
        catboost_fold_scores: list[float] = []
        fold_iterations: list[int] = []
        for fold, (fit_idx, val_idx) in enumerate(cv.split(X_all, strat_bins), start=1):
            log(f"CV fold {fold}/5")
            model = CatBoostRegressor(
                loss_function="RMSE",
                eval_metric="RMSE",
                iterations=tuned_iteration_cap,
                random_seed=RANDOM_STATE + fold,
                verbose=250,
                allow_writing_files=False,
                **best_parameters,
            )
            model.fit(
                X_all.iloc[fit_idx][selected_features],
                y.iloc[fit_idx],
                cat_features=selected_cat_columns,
                text_features=selected_text_columns,
                eval_set=(X_all.iloc[val_idx][selected_features], y.iloc[val_idx]),
                early_stopping_rounds=150,
                use_best_model=True,
            )
            catboost_oof_predictions[val_idx] = model.predict(X_all.iloc[val_idx][selected_features])
            catboost_test_fold_predictions[:, fold - 1] = model.predict(X_test[selected_features])
            fold_rmsle = regression_metrics(y.iloc[val_idx], catboost_oof_predictions[val_idx])["RMSLE"]
            catboost_fold_scores.append(fold_rmsle)
            fold_iterations.append(model.get_best_iteration() + 1)
            log(f"CatBoost fold {fold} RMSLE = {fold_rmsle:.5f}")
            del model
            gc.collect()
        catboost_oof_rmsle = regression_metrics(y, catboost_oof_predictions)["RMSLE"]
        log(f"CatBoost OOF RMSLE: {catboost_oof_rmsle:.5f} +/- {np.std(catboost_fold_scores):.5f}")

        te_columns = [
            column
            for column in [
                "ProductConfigID",
                "Spec_FullDescriptor",
                "Spec_BaseClass",
                "VendorPartnerID",
                "RegionCode",
                "FunctionalClassification",
                "InventoryGroupCategory",
                "AssetScaleFactor",
                "DataOriginCode",
                "UtilizationTier",
                "RegionBaseClass",
                "ConfigBaseClass",
                "BaseSubClass",
                "AgeBucket",
            ]
            if column in X_all.columns
        ]
        log(f"Building leakage-safe target encodings for {len(te_columns)} key categorical features")
        te_train, te_test = add_target_encoding_features(
            X_all,
            X_test,
            y,
            te_columns,
            strat_bins,
            n_splits=5,
            smoothing=25.0,
        )
        lgb_train_frame = pd.concat([X_all[numeric_columns].reset_index(drop=True), te_train.reset_index(drop=True)], axis=1)
        lgb_test_frame = pd.concat([X_test[numeric_columns].reset_index(drop=True), te_test.reset_index(drop=True)], axis=1)
        lgb_train_frame = lgb_train_frame.replace([np.inf, -np.inf], np.nan)
        lgb_test_frame = lgb_test_frame.replace([np.inf, -np.inf], np.nan)

        log("Starting 5-fold LightGBM CV ensemble on target-encoded features")
        lgb_oof_predictions = np.zeros(len(X_all))
        lgb_test_fold_predictions = np.zeros((len(X_test), 5))
        lgb_fold_scores: list[float] = []
        for fold, (fit_idx, val_idx) in enumerate(cv.split(lgb_train_frame, strat_bins), start=1):
            log(f"LightGBM fold {fold}/5")
            lgb_model = LGBMRegressor(
                objective="regression",
                n_estimators=3000,
                learning_rate=0.02,
                num_leaves=127,
                max_depth=-1,
                min_child_samples=40,
                subsample=0.85,
                colsample_bytree=0.75,
                reg_alpha=0.2,
                reg_lambda=1.0,
                random_state=RANDOM_STATE + 100 + fold,
                n_jobs=-1,
            )
            lgb_model.fit(
                lgb_train_frame.iloc[fit_idx],
                y.iloc[fit_idx],
                eval_set=[(lgb_train_frame.iloc[val_idx], y.iloc[val_idx])],
                eval_metric="rmse",
                callbacks=[early_stopping(200), log_evaluation(250)],
            )
            lgb_oof_predictions[val_idx] = lgb_model.predict(lgb_train_frame.iloc[val_idx])
            lgb_test_fold_predictions[:, fold - 1] = lgb_model.predict(lgb_test_frame)
            lgb_fold_rmsle = regression_metrics(y.iloc[val_idx], lgb_oof_predictions[val_idx])["RMSLE"]
            lgb_fold_scores.append(lgb_fold_rmsle)
            log(f"LightGBM fold {fold} RMSLE = {lgb_fold_rmsle:.5f}")
            del lgb_model
            gc.collect()
        lgb_oof_rmsle = regression_metrics(y, lgb_oof_predictions)["RMSLE"]
        log(f"LightGBM OOF RMSLE: {lgb_oof_rmsle:.5f} +/- {np.std(lgb_fold_scores):.5f}")

        catboost_weight = 1 / max(catboost_oof_rmsle, 1e-6)
        lgb_weight = 1 / max(lgb_oof_rmsle, 1e-6)
        total_weight = catboost_weight + lgb_weight
        catboost_weight /= total_weight
        lgb_weight /= total_weight
        blended_oof_predictions = catboost_weight * catboost_oof_predictions + lgb_weight * lgb_oof_predictions
        blended_oof_rmsle = regression_metrics(y, blended_oof_predictions)["RMSLE"]
        log(
            f"Blend weights -> CatBoost: {catboost_weight:.3f}, LightGBM: {lgb_weight:.3f} | "
            f"Blended OOF RMSLE: {blended_oof_rmsle:.5f}"
        )

        log("Training final CatBoost model")
        best_iteration = max(1400, int(np.median(fold_iterations)))
        final_model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            iterations=best_iteration,
            random_seed=RANDOM_STATE,
            verbose=200,
            allow_writing_files=False,
            **best_parameters,
        )
        final_model.fit(
            X_all[selected_features],
            y,
            cat_features=selected_cat_columns,
            text_features=selected_text_columns,
        )
        final_importance = pd.DataFrame(
            {
                "feature": selected_features,
                "importance": final_model.get_feature_importance(),
            }
        ).sort_values("importance", ascending=False).reset_index(drop=True)
        save_feature_importance(
            final_importance,
            output_dir,
            "catboost_feature_importance_competition.png",
            "CatBoost feature importance (final competition model)",
        )

        log("Generating submission")
        catboost_test_log_predictions_cv = catboost_test_fold_predictions.mean(axis=1)
        catboost_test_log_predictions_full = final_model.predict(X_test[selected_features])
        catboost_test_log_predictions = 0.8 * catboost_test_log_predictions_cv + 0.2 * catboost_test_log_predictions_full
        lgb_test_log_predictions = lgb_test_fold_predictions.mean(axis=1)
        test_log_predictions = (
            catboost_weight * catboost_test_log_predictions
            + lgb_weight * lgb_test_log_predictions
        )
        test_predictions = np.maximum(0, np.expm1(test_log_predictions))
        submission = pd.DataFrame(
            {
                "TransactionID": test_raw["TransactionID"],
                TARGET: test_predictions,
            }
        )
        submission_path = output_dir / "catboost_lgb_blend_submission.csv"
        submission.to_csv(submission_path, index=False)
        assert len(submission) == len(test_raw)
        assert submission[TARGET].ge(0).all()
        log(f"Saved submission: {submission_path.resolve()}")

        report_lines = [
            "# Heavy Equipment Selling Price Prediction - Run Report",
            "",
            f"- Training rows: {len(train_raw):,}; test rows: {len(test_raw):,}",
            f"- Input features after engineering: {X_all.shape[1]}",
            f"- Required engineered features present: {', '.join(REQUIRED_FEATURES)}",
            f"- OOF CatBoost RMSLE (5-fold): {catboost_oof_rmsle:.5f}",
            f"- OOF LightGBM RMSLE (5-fold): {lgb_oof_rmsle:.5f}",
            f"- OOF blended RMSLE (5-fold): {blended_oof_rmsle:.5f}",
            f"- Best tuned holdout RMSLE: {tuning_results_df.iloc[0]['RMSLE']:.5f}",
            f"- Final selected features: {len(selected_features)}",
            f"- Submission file: {submission_path.resolve()}",
            "",
            "## Holdout model comparison",
            results_df.round(5).to_csv(index=False),
            "",
            "## Modelling decisions",
            "- Target is modelled as log1p(TargetValue), optimizing the RMSLE objective.",
            "- IQR outliers are reported but retained because extreme equipment can be valid transactions.",
            "- Category frequency and rare-category handling are fitted on training data only.",
            "- CatBoost is the final model because it natively handles missing and high-cardinality categorical values.",
        ]
        report_path = output_dir / "model_report.md"
        report_path.write_text("\n".join(report_lines), encoding="utf-8")
        log(f"Saved report: {report_path.resolve()}")
    else:
        metrics_row = results_df.loc[results_df["Model"] == "CatBoost (debug)"]
        if not metrics_row.empty:
            debug_rmsle = float(metrics_row.iloc[0]["RMSLE"])
            log(f"Debug RMSLE: {debug_rmsle:.6f}")
        log("Debug mode complete")


if __name__ == "__main__":
    main()
