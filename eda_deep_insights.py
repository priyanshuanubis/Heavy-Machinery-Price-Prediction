"""
eda_deep_insights.py - Deep exploratory data analysis to uncover exact mathematical properties of the dataset
"""

import pandas as pd
import numpy as np

train = pd.read_csv('data/train.csv', low_memory=False)
test = pd.read_csv('data/test.csv', low_memory=False)

train['SaleYear'] = pd.to_datetime(train['TransactionDate']).dt.year
train.loc[train['ManufactureYear'] == 1001, 'ManufactureYear'] = np.nan
train['MachineAge'] = train['SaleYear'] - train['ManufactureYear']
train['y'] = np.log1p(train['TargetValue'])

test['SaleYear'] = pd.to_datetime(test['TransactionDate']).dt.year
test.loc[test['ManufactureYear'] == 1001, 'ManufactureYear'] = np.nan
test['MachineAge'] = test['SaleYear'] - test['ManufactureYear']

print("=== DEEP EDA & PATTERN RECOGNITION ===")
print("1. Target Distribution:")
print(f"   Min: {train['TargetValue'].min()}, Max: {train['TargetValue'].max()}")
print(f"   Log1p Min: {train['y'].min():.4f}, Max: {train['y'].max():.4f}, Mean: {train['y'].mean():.4f}, Std: {train['y'].std():.4f}")

print("\n2. High-Cardinality Categorical Granularity:")
for col in ['ProductConfigID', 'Spec_FullDescriptor', 'Spec_BaseClass', 'VendorPartnerID', 'RegionCode', 'FunctionalClassification']:
    tr_u = train[col].nunique()
    te_u = test[col].nunique()
    common = len(set(train[col].dropna()).intersection(set(test[col].dropna())))
    print(f"   {col:28s} | Train unique: {tr_u:5d} | Test unique: {te_u:5d} | Overlap: {common:5d} ({100*common/max(te_u,1):.1f}%)")

print("\n3. Asset Repeat Sales Analysis:")
asset_tr = set(train['AssetID'])
asset_te = set(test['AssetID'])
asset_overlap = asset_tr.intersection(asset_te)
print(f"   Total Train Assets: {len(asset_tr)}")
print(f"   Total Test Assets:  {len(asset_te)}")
print(f"   Overlap Assets:     {len(asset_overlap)} ({100*len(asset_overlap)/len(asset_te):.1f}% of test)")

# Target variance for repeated assets in train
asset_counts = train['AssetID'].value_counts()
multi_assets = asset_counts[asset_counts > 1].index
multi_df = train[train['AssetID'].isin(multi_assets)]
std_within_asset = multi_df.groupby('AssetID')['y'].std().dropna().mean()
print(f"   Mean target std within repeat assets: {std_within_asset:.4f}")

print("\n4. Missingness & Zero Structure:")
print(f"   OperationalHoursMeter missing in train: {train['OperationalHoursMeter'].isna().mean():.1%}")
print(f"   OperationalHoursMeter == 0 in train:    {(train['OperationalHoursMeter'] == 0).mean():.1%}")
print(f"   OperationalHoursMeter missing in test:  {test['OperationalHoursMeter'].isna().mean():.1%}")
print(f"   OperationalHoursMeter == 0 in test:     {(test['OperationalHoursMeter'] == 0).mean():.1%}")

print("\n5. Spec_FullDescriptor String Tokens:")
# Analyze tokens inside Spec_FullDescriptor
all_specs = pd.concat([train['Spec_FullDescriptor'], test['Spec_FullDescriptor']]).dropna().astype(str)
tokens = pd.Series([t for spec in all_specs for t in spec.replace('/', ' ').replace('-', ' ').split()]).value_counts()
print("   Top 15 tokens in Spec_FullDescriptor:")
print(tokens.head(15).to_string())
