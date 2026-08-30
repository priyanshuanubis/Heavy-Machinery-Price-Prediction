import pandas as pd
import numpy as np

df = pd.read_csv("data/train.csv", low_memory=False)

# 1. Imputing Missing Categorical Data
# How many total rows will contain "Missing" if CabinType missing values filled with "Missing"?
cabin_missing_count = df["CabinType"].isnull().sum()
print(f"CabinType missing count: {cabin_missing_count}")

# 2. Train-Test Split Verification
# 80/20 train-test split on train dataset (138,701 rows total)
total_rows = len(df)
train_rows_80 = int(round(total_rows * 0.8)) # or 138701 * 0.8
from sklearn.model_selection import train_test_split
X_tr, X_te = train_test_split(df, test_size=0.2, random_state=42)
print(f"Total rows: {total_rows}, X_train rows (80% split): {len(X_tr)}")

# 3. Log Transformed Target Standard Deviation
log_target = np.log1p(df["TargetValue"].clip(lower=0))
std_log_target = log_target.std()
print(f"log1p(TargetValue) standard deviation: {std_log_target:.6f} -> Rounded: {std_log_target:.4f}")
