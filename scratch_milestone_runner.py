import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

# Load dataset
df = pd.read_csv("data/train.csv", low_memory=False)

# Columns specified in prompt:
# TargetValue, UtilizationTier, DrivetrainType, CabinType, AssetScaleFactor, ManufactureYear, OperationalHoursMeter
# Spec_FullDescriptor, RegionCode, FunctionalClassification

# -------------------------------------------------------------
# Q1 — Stratified vs. Non-Stratified Split
# -------------------------------------------------------------
# Step A:
# Fill missing values in UtilizationTier with the string "Unknown".
df['UtilizationTier'] = df['UtilizationTier'].fillna("Unknown")

# Create PriceBand cutting TargetValue into 5 quantile-based bins using pd.qcut(..., q=5, labels=False)
df['PriceBand'] = pd.qcut(df['TargetValue'], q=5, labels=False)

# Split the data using train_test_split into 70% train / 30% validation, with random_state=42, stratifying on PriceBand.
train_strat, valid_strat = train_test_split(
    df, test_size=0.3, random_state=42, stratify=df['PriceBand']
)

val_strat_counts = valid_strat['PriceBand'].value_counts().sort_index().to_numpy()

# Step B:
# Repeat exact same split (test_size=0.3, random_state=42) but with stratify=None.
train_nonstrat, valid_nonstrat = train_test_split(
    df, test_size=0.3, random_state=42, stratify=None
)

val_nonstrat_counts = valid_nonstrat['PriceBand'].value_counts().sort_index().to_numpy()

# Step C:
# Convert both validation count arrays into proportions (divide by total validation size).
val_strat_prop = val_strat_counts / len(valid_strat)
val_nonstrat_prop = val_nonstrat_counts / len(valid_nonstrat)

abs_diffs = np.abs(val_strat_prop - val_nonstrat_prop)
max_abs_diff = np.max(abs_diffs)

print(f"Q1 - Maximum Absolute Difference: {max_abs_diff:.6f} -> Rounded: {max_abs_diff:.4f}")


# -------------------------------------------------------------
# Q2 — Preprocessing Pipeline
# -------------------------------------------------------------
# Stratified split features and target from Q1:
# Categorical columns: ["UtilizationTier", "DrivetrainType", "CabinType", "AssetScaleFactor"]
# Numerical columns: ["ManufactureYear", "OperationalHoursMeter"]

cat_cols = ["UtilizationTier", "DrivetrainType", "CabinType", "AssetScaleFactor"]
num_cols = ["ManufactureYear", "OperationalHoursMeter"]

feature_cols = cat_cols + num_cols

X_train = train_strat[feature_cols].copy()
X_valid = valid_strat[feature_cols].copy()
y_train = train_strat['TargetValue'].copy()
y_valid = valid_strat['TargetValue'].copy()

# Pipeline definition:
# Categorical: SimpleImputer(strategy="most_frequent") then OneHotEncoder(handle_unknown="ignore", sparse_output=True)
# Numerical: SimpleImputer(strategy="mean") then StandardScaler()

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
])

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', cat_transformer, cat_cols),
        ('num', num_transformer, num_cols)
    ],
    remainder='drop'
)

# Fit on X_train, transform both X_train and X_valid
X_train_proc = preprocessor.fit_transform(X_train)
X_valid_proc = preprocessor.transform(X_valid)

# Sum of all values in X_train_proc
if hasattr(X_train_proc, "toarray"):
    sum_x_train_proc = X_train_proc.toarray().sum()
else:
    sum_x_train_proc = X_train_proc.sum()

print(f"Q2 - Sum of all values in X_train_proc: {sum_x_train_proc:.6f} -> Rounded: {sum_x_train_proc:.3f}")


# -------------------------------------------------------------
# Q3 — Hyperparameter Tuning with RandomizedSearchCV
# -------------------------------------------------------------
# Tune a RandomForestRegressor(random_state=42) using RandomizedSearchCV over:
# - n_estimators: [50, 100, 200]
# - max_depth: [5, 10, 15]
# Configuration: n_iter=5, cv=3, random_state=42, n_jobs=-1, default scoring.

param_dist = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15]
}

rf = RandomForestRegressor(random_state=42)
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=5,
    cv=3,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train_proc, y_train)
best_n_estimators = random_search.best_params_['n_estimators']

print(f"Q3 - Best params found: {random_search.best_params_}")
print(f"Q3 - Best n_estimators: {best_n_estimators}")


# -------------------------------------------------------------
# Q4 — AdaBoost Estimator Error Variance
# -------------------------------------------------------------
# Train an AdaBoostRegressor(n_estimators=50, random_state=42) (all other parameters default) on (X_train_proc, y_train).
# Access model.estimator_errors_ (length-50 array of each weak learner's error).
# Compute np.var(model.estimator_errors_) and report rounded to 4 decimal places.

ada = AdaBoostRegressor(n_estimators=50, random_state=42)
ada.fit(X_train_proc, y_train)

error_var = np.var(ada.estimator_errors_)
print(f"Q4 - AdaBoost Estimator Error Variance: {error_var:.6f} -> Rounded: {error_var:.4f}")


# -------------------------------------------------------------
# Q5 — Feature Importance
# -------------------------------------------------------------
# Train a RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42) on (X_train_proc, y_train).
# Extract feature_importances_ and report the index of the feature with the maximum importance.
# If tie, return smallest index.

rf_fi = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_fi.fit(X_train_proc, y_train)

importances = rf_fi.feature_importances_
max_idx = np.argmax(importances)
print(f"Q5 - Feature Index with max importance: {max_idx} (Importance: {importances[max_idx]:.6f})")


# -------------------------------------------------------------
# Q6 — Counting MLP Weights
# -------------------------------------------------------------
# Let N = X_train_proc.shape[1]
# hidden_layer_sizes = (128, 64, 32)
# Architecture:
# Input: N
# Hidden 1: 128
# Hidden 2: 64
# Hidden 3: 32
# Output: 1
# Connection weights (excluding biases): N*128 + 128*64 + 64*32 + 32*1

N = X_train_proc.shape[1]
weights_count = (N * 128) + (128 * 64) + (64 * 32) + (32 * 1)

print(f"Q6 - N (Number of features after preprocessing): {N}")
print(f"Q6 - Total Connection Weights (excluding biases): {weights_count}")
