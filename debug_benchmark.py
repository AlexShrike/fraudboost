#!/usr/bin/env python3
"""Debug version to identify where the issue is."""

print("Starting debug...")

print("1. Basic imports...")
import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("2. Matplotlib setup...")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("3. Sklearn imports...")
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, recall_score, precision_score, f1_score,
    precision_recall_curve, roc_curve, confusion_matrix
)

print("4. XGBoost import...")
import xgboost as xgb

print("5. FraudBoost import...")
from fraudboost import FraudBoostClassifier

print("6. All imports successful!")

print("7. Testing basic functionality...")
# Quick test
X_test = np.random.randn(100, 5)
y_test = np.random.binomial(1, 0.1, 100)

print("8. Testing XGBoost...")
xgb_model = xgb.XGBClassifier(n_estimators=2, max_depth=2, random_state=42)
xgb_model.fit(X_test, y_test)

print("9. Testing FraudBoost...")
fb_model = FraudBoostClassifier(n_estimators=2, max_depth=2, random_state=42)
fb_model.fit(X_test, y_test)

print("10. All tests passed! Ready to run benchmark.")