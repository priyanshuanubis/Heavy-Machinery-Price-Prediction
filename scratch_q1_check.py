import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("data/train.csv", low_memory=False)

# Derived features as specified in Image 1:
# TransactionYear (Extract from TransactionDate)
# TransactionMonth (Extract from TransactionDate)
# TransactionQuarter (Extract from TransactionDate)
# ManufactureDecade (Extract from ManufactureYear)
# SpecificationComplexity (Number of words present in Spec_FullDescriptor)

df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])
df["TransactionYear"] = df["TransactionDate"].dt.year
df["TransactionMonth"] = df["TransactionDate"].dt.month
df["TransactionQuarter"] = df["TransactionDate"].dt.quarter

# Handle ManufactureYear sentinel (1001) or invalid values if needed, but let's check raw floor decade first
df["ManufactureDecade"] = (df["ManufactureYear"] // 10) * 10

# SpecificationComplexity
df["SpecificationComplexity"] = df["Spec_FullDescriptor"].fillna("").astype(str).str.split().str.len()

print("--- Q1 Check: Average TargetValue by ManufactureDecade ---")
decade_stats = df.groupby("ManufactureDecade")["TargetValue"].agg(["mean", "count"])
print(decade_stats)

print("\n--- Valid Decades (excluding sentinel 1000) ---")
valid_decades = df[df["ManufactureYear"] != 1001].groupby("ManufactureDecade")["TargetValue"].agg(["mean", "count"])
print(valid_decades)

# Check options: 2010, 2000, 1990, 1980
options = [2010, 2000, 1990, 1980]
print("\nOptions mean TargetValue:")
for opt in options:
    val = df[df["ManufactureDecade"] == opt]["TargetValue"].mean()
    print(f"Decade {opt}: {val:.2f}")
