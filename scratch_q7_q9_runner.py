import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error

# Load dataset
df = pd.read_csv("data/train.csv", low_memory=False)

# Preprocessing from previous questions:
df['UtilizationTier'] = df['UtilizationTier'].fillna("Unknown")
df['PriceBand'] = pd.qcut(df['TargetValue'], q=5, labels=False)

train_strat, valid_strat = train_test_split(
    df, test_size=0.3, random_state=42, stratify=df['PriceBand']
)

cat_cols = ["UtilizationTier", "DrivetrainType", "CabinType", "AssetScaleFactor"]
num_cols = ["ManufactureYear", "OperationalHoursMeter"]
feature_cols = cat_cols + num_cols

X_train = train_strat[feature_cols].copy()
X_valid = valid_strat[feature_cols].copy()
y_train = train_strat['TargetValue'].copy()
y_valid = valid_strat['TargetValue'].copy()

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

X_train_proc = preprocessor.fit_transform(X_train)
X_valid_proc = preprocessor.transform(X_valid)


# -------------------------------------------------------------
# Q7 — Training Loss After 5 Iterations
# -------------------------------------------------------------
# Train MLPRegressor: hidden_layer_sizes=(128, 64, 32), activation="relu", solver="adam", max_iter=5, batch_size=32, random_state=42
# Report model.loss_ rounded to 4 decimal places.

mlp_q7 = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),
    activation="relu",
    solver="adam",
    max_iter=5,
    batch_size=32,
    random_state=42
)

mlp_q7.fit(X_train_proc, y_train)
loss_q7 = mlp_q7.loss_
print(f"Q7 - Training Loss (model.loss_): {loss_q7:.6f} -> Rounded: {loss_q7:.4f}")


# -------------------------------------------------------------
# Q8 — Effect of Regularization (alpha)
# -------------------------------------------------------------
# Model A: alpha=0.0001, hidden_layer_sizes=(100,), max_iter=5, random_state=42
# Model B: alpha=1.0, hidden_layer_sizes=(100,), max_iter=5, random_state=42
# Compute MAE on training set, report absolute difference between two MAE values rounded to 4 decimal places.

mlp_a = MLPRegressor(hidden_layer_sizes=(100,), alpha=0.0001, max_iter=5, random_state=42)
mlp_b = MLPRegressor(hidden_layer_sizes=(100,), alpha=1.0, max_iter=5, random_state=42)

mlp_a.fit(X_train_proc, y_train)
mlp_b.fit(X_train_proc, y_train)

pred_a = mlp_a.predict(X_train_proc)
pred_b = mlp_b.predict(X_train_proc)

mae_a = mean_absolute_error(y_train, pred_a)
mae_b = mean_absolute_error(y_train, pred_b)

diff_q8 = abs(mae_a - mae_b)
print(f"Q8 - MAE A: {mae_a:.4f}, MAE B: {mae_b:.4f}, Abs Diff: {diff_q8:.6f} -> Rounded: {diff_q8:.4f}")


# -------------------------------------------------------------
# Q9 — Validation Performance
# -------------------------------------------------------------
# Predictions on X_valid_proc using model from Q7. Compute MAE on validation set rounded to 4 decimal places.

pred_val_q7 = mlp_q7.predict(X_valid_proc)
mae_q9 = mean_absolute_error(y_valid, pred_val_q7)
print(f"Q9 - Validation MAE: {mae_q9:.6f} -> Rounded: {mae_q9:.4f}")


# -------------------------------------------------------------
# Calculating Asset Age Question
# -------------------------------------------------------------
# Replace 1000 and 1001 with NaN in ManufactureYear
# AssetAge = SaleYear - ManufactureYear
# What is median AssetAge in training data?

df_raw = pd.read_csv("data/train.csv", low_memory=False)
df_raw["TransactionDate"] = pd.to_datetime(df_raw["TransactionDate"])
df_raw["SaleYear"] = df_raw["TransactionDate"].dt.year

mfg_year = df_raw["ManufactureYear"].copy()
mfg_year = mfg_year.replace([1000, 1001], np.nan)

asset_age = df_raw["SaleYear"] - mfg_year
median_asset_age = asset_age.median()
print(f"Calculating Asset Age - Median AssetAge: {median_asset_age}")


# -------------------------------------------------------------
# Feature Interaction (Intensity) Question
# -------------------------------------------------------------
# HoursPerYear = OperationalHoursMeter / AssetAge
# Median number of hours used per year (ignore infinite/NaN)?

hours = df_raw["OperationalHoursMeter"].copy()
# Replace 0 or keep as is? Let's check hours / asset_age ignoring inf and nan
h_per_year = hours / asset_age
h_per_year = h_per_year.replace([np.inf, -np.inf], np.nan).dropna()
median_h_per_year = h_per_year.median()
print(f"Feature Interaction - Median HoursPerYear: {median_h_per_year:.2f}")


# -------------------------------------------------------------
# Dropping Sparse Features Question
# -------------------------------------------------------------
# Set threshold to drop any column missing more than 80% of its data. Exactly how many columns will be removed?

missing_pct = df_raw.isnull().sum() / len(df_raw)
cols_dropped_80 = (missing_pct > 0.80).sum()
print(f"Dropping Sparse Features - Columns with >80% missing: {cols_dropped_80}")
