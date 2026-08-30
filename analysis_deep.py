import pandas as pd
import numpy as np

train = pd.read_csv('data/train.csv', low_memory=False)
train['SaleYear'] = pd.to_datetime(train['TransactionDate']).dt.year
train.loc[train['ManufactureYear'] == 1001, 'ManufactureYear'] = np.nan
train['MachineAge'] = train['SaleYear'] - train['ManufactureYear']
train['y'] = np.log1p(train['TargetValue'])

print('=== OPERATIONAL HOURS ANALYSIS ===')
hours = train['OperationalHoursMeter']
print(f'Non-null hours: {hours.notna().sum()} ({100*hours.notna().mean():.1f}%)')
print(f'Zero hours: {(hours == 0).sum()}')
valid_hours = hours[(hours > 0) & (hours.notna())]
print(f'Valid hours (>0): {len(valid_hours)}')
print(f'Hours p5={valid_hours.quantile(0.05):.0f}, p50={valid_hours.quantile(0.50):.0f}, p95={valid_hours.quantile(0.95):.0f}')
print(f'Hours > 50000: {(valid_hours > 50000).sum()}')

mask = (hours > 0) & (hours.notna())
corr_hours = hours[mask].corr(train.loc[mask, 'y'])
print(f'Corr valid hours with y: {corr_hours:.4f}')

hpy = hours[mask] / (train.loc[mask, 'MachineAge'] + 1)
corr_hpy = hpy.corr(train.loc[mask, 'y'])
print(f'Corr hours/year with y: {corr_hpy:.4f}')

print()
print('=== FUNCTIONAL CLASSIFICATION ===')
print(train['FunctionalClassification'].value_counts().head(10))

print()
print('=== PRICE TREND OVER TIME ===')
yearly = train.groupby('SaleYear')['y'].agg(['mean', 'std', 'count'])
print(yearly)

print()
print('=== UTILIZATION TIER ===')
print(train['UtilizationTier'].value_counts(dropna=False))

print()
print('=== DRIVETRAIN TYPE ===')
print(train['DrivetrainType'].value_counts(dropna=False))

print()
print('=== COL10 (lowest missing among col columns) ===')
print(train['col10'].value_counts(dropna=False).head(15))

print()
print('=== CABIN TYPE ===')
print(train['CabinType'].value_counts(dropna=False))
print(f'CabinType corr with y: {train["CabinType"].astype("category").cat.codes.corr(train["y"]):.4f}')
