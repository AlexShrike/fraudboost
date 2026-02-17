# Benchmark Results: FraudBoost vs Standard ML Models

**Dataset:** Kartik2112 Credit Card Fraud Detection  
**Training Set:** 207K transactions (~7K fraud, 200K legit)  
**Test Set:** 555K transactions (full test set)  
**Date:** 2026-02-16 20:49:10

## Performance Summary

| Model | AUC-ROC | Precision@80%Recall | Recall@0.5 | F1-Score | VDR | Net Savings | False Positives | Train Time (s) |
|-------|---------|-------------------|-----------|----------|-----|-------------|----------------|---------------|
| XGBoost | 0.9968 | 0.6595 | 0.9613 | 0.2943 | 0.9961 | $148,448 | 9,805 | 0.3 |
| LightGBM | 0.9966 | 0.6804 | 0.9543 | 0.2973 | 0.9899 | $164,126 | 9,577 | 0.5 |
| Random Forest | 0.9940 | 0.5743 | 0.9319 | 0.3384 | 0.9904 | $355,471 | 7,670 | 4.0 |
| Logistic Regression | 0.8753 | 0.0263 | 0.7385 | 0.1176 | 0.9812 | $-1,209,876 | 23,219 | 0.1 |

## Optimal Threshold Performance

| Model | Optimal Threshold | Net Savings | VDR |
|-------|------------------|-------------|-----|
| XGBoost | 0.960 | $980,625 | 0.9169 |
| LightGBM | 0.960 | $989,003 | 0.9264 |
| Random Forest | 0.870 | $986,941 | 0.9307 |
| Logistic Regression | 0.990 | $660,628 | 0.7954 |

## Key Insights

- **Value Detection Rate (VDR):** Percentage of total fraud dollar value detected
- **Net Savings:** Total fraud value detected minus false positive costs (FP cost = $100)
- **Optimal Threshold:** Threshold that maximizes net savings for each model

## Feature Engineering

Features used (no target leakage):
- Transaction amount (amt)
- Category (label encoded)
- Gender (binary encoded)  
- City population
- Customer/merchant lat/long coordinates
- Transaction hour and day of week
- Customer age (calculated from DOB)
- Distance between customer and merchant (haversine)
- Log-transformed amount

## Model Configurations

- **FraudBoost:** n_estimators=20, max_depth=2, lr=0.1, value_weighted loss (reduced for 16GB Mac)
- **XGBoost:** n_estimators=200, max_depth=4, lr=0.1, scale_pos_weight=~27
- **LightGBM:** n_estimators=200, max_depth=4, lr=0.1, is_unbalance=True
- **Random Forest:** n_estimators=200, max_depth=10, class_weight='balanced'
- **Logistic Regression:** C=1.0, max_iter=1000, class_weight='balanced', StandardScaler

