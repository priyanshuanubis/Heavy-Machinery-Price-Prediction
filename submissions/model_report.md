# Heavy Equipment Selling Price Prediction - Run Report

- Training rows: 138,701; test rows: 15,000
- Input features after engineering: 112
- Required engineered features present: TransactionYear, TransactionQuarter, AssetAge, DescriptorLength, HasOperationalHours, HasVariantModifier
- OOF CatBoost RMSLE (5-fold): 0.23208
- Best tuned holdout RMSLE: 0.23908
- Final selected features: 112
- Submission file: E:\HeavyEquipmentML\submissions\catboost_submission.csv

## Holdout model comparison
Model,RMSE,MAE,R2,RMSLE
Linear Regression,22338.38674,16690.72397,0.2718,0.55637
Random Forest (50k sample),9449.50043,6292.65819,0.86969,0.23176
LightGBM,8621.61379,5820.83513,0.89153,0.21416


## Modelling decisions
- Target is modelled as log1p(TargetValue), optimizing the RMSLE objective.
- IQR outliers are reported but retained because extreme equipment can be valid transactions.
- Category frequency and rare-category handling are fitted on training data only.
- CatBoost is the final model because it natively handles missing and high-cardinality categorical values.