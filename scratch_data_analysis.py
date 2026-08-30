import pandas as pd
import numpy as np

df = pd.read_csv('data/train.csv', low_memory=False)

# Check the regex pattern issue - are there actual "Metric Tons" / "Horsepower" matches?
func = df['FunctionalClassification'].dropna()
print("Sample FunctionalClassification values:")
for v in func.unique()[:30]:
    print(f"  '{v}'")

print()

# Test regex extraction with correct Python regex (not double-escaped)
import re
# Try to extract tons
tons_pattern = r'(\d+\.?\d*)\s*to\s*(\d+\.?\d*)\s*Metric Tons'
hp_pattern = r'(\d+\.?\d*)\s*to\s*(\d+\.?\d*)\s*Horsepower'

tons_matches = func.str.extract(tons_pattern)
hp_matches = func.str.extract(hp_pattern)
print(f"Tons extracted: {tons_matches.dropna().shape[0]} out of {len(func)}")
print(f"HP extracted: {hp_matches.dropna().shape[0]} out of {len(func)}")
print()

# How about a simpler classification? Extract machine type
type_pattern = r'^(.+?)\s*-\s*'
machine_types = func.str.extract(type_pattern)
print("Machine types extracted:")
print(machine_types[0].value_counts().head(15))
print()

# Check Spec_BaseClass distribution (this is int64 but with 1249 unique)
print("Spec_BaseClass value_counts (top 20):")
print(df['Spec_BaseClass'].value_counts().head(20))
print()

# Check Spec_ReleaseSeries
print("Spec_ReleaseSeries value_counts (top 15):")
print(df['Spec_ReleaseSeries'].value_counts().head(15))
print(f"Nulls: {df['Spec_ReleaseSeries'].isnull().sum()}")
print()

# Check how well AssetID predicts price for repeat assets
repeat_assets = df.groupby('AssetID').filter(lambda x: len(x) > 1)
print(f"Repeat asset transactions: {len(repeat_assets)}")
print(f"Unique repeat assets: {repeat_assets['AssetID'].nunique()}")

# Check temporal leakage potential
df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
df = df.sort_values('TransactionDate')

# For repeat assets, calculate how well previous price predicts next
repeat_assets_sorted = repeat_assets.sort_values(['AssetID', 'TransactionDate'])
repeat_assets_sorted['prev_price'] = repeat_assets_sorted.groupby('AssetID')['TargetValue'].shift(1)
valid = repeat_assets_sorted.dropna(subset=['prev_price'])
if len(valid) > 0:
    corr = np.corrcoef(np.log1p(valid['prev_price']), np.log1p(valid['TargetValue']))[0, 1]
    print(f"Correlation of prev_price with current for repeat assets: {corr:.4f}")
print()

# Check test data distribution
te = pd.read_csv('data/test.csv', low_memory=False)
print(f"Test columns same as train (minus target): {set(te.columns) - set(df.columns)}")
print(f"Train columns not in test: {set(df.columns) - set(te.columns)}")
print()

# Check if test AssetIDs overlap with train
overlap = set(te['AssetID']) & set(df['AssetID'])
print(f"Test AssetIDs overlapping with train: {len(overlap)} out of {te['AssetID'].nunique()}")
