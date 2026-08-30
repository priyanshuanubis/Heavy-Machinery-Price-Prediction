import pandas as pd
import numpy as np

df = pd.read_csv("data/train.csv", low_memory=False)

df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])
df["TransactionYear"] = df["TransactionDate"].dt.year

mfg_year = df["ManufactureYear"].replace([1000, 1001], np.nan)
df["AssetAge"] = df["TransactionYear"] - mfg_year

mode_age = df["AssetAge"].mode()[0]
mode_count = (df["AssetAge"] == mode_age).sum()

print(f"Most frequent AssetAge: {mode_age} (Count: {mode_count})")
print("\nTop 10 AssetAge values:")
print(df["AssetAge"].value_counts().head(10))
