import pandas as pd
import numpy as np

train = pd.read_csv('data/train.csv', low_memory=False)
test = pd.read_csv('data/test.csv', low_memory=False)

train['SaleYear'] = pd.to_datetime(train['TransactionDate']).dt.year
test['SaleYear'] = pd.to_datetime(test['TransactionDate']).dt.year

print('=== TRAIN/TEST YEAR RANGE ===')
print('Train:', train['SaleYear'].min(), 'to', train['SaleYear'].max())
print('Test:', test['SaleYear'].min(), 'to', test['SaleYear'].max())

test_years = set(test['SaleYear'].unique())
train_years = set(train['SaleYear'].unique())
print('Test years not in train:', test_years - train_years)

# Repeat assets in test
test_assets = set(test['AssetID'].unique())
train_assets = set(train['AssetID'].unique())
overlap_assets = test_assets & train_assets
print('Test assets also in train:', len(overlap_assets), 'out of', len(test_assets),
      '(' + str(round(100*len(overlap_assets)/len(test_assets), 1)) + '%)')

# For overlapping assets, what was their last sale price?
if overlap_assets:
    overlap_df = train[train['AssetID'].isin(overlap_assets)]
    last_sale = overlap_df.sort_values('TransactionDate').groupby('AssetID').last()
    print('Mean last sale log_price for overlap assets:', round(np.log1p(last_sale['TargetValue']).mean(), 4))

print()
print('=== METADATA CHECK ===')
meta = pd.read_csv('data/metadata.csv')
print(meta.to_string())
