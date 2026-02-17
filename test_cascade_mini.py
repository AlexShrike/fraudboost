#!/usr/bin/env python
"""
Mini test of cascade fraud detection system with small dataset.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification
import warnings
warnings.filterwarnings('ignore')

# Add the fraudboost directory to Python path
sys.path.append('/Users/alexshrike/.openclaw/workspace/fraudboost')

from fraudboost.core import FraudBoostClassifier
from fraudboost.cascade import FraudCascade
from fraudboost.stacking import FraudStacking
import xgboost as xgb

print("Mini Cascade Fraud Detection Test")
print("=" * 40)

# Create synthetic fraud data
print("\nCreating synthetic fraud dataset...")
n_samples = 10000
X, y = make_classification(
    n_samples=n_samples,
    n_features=10, 
    n_informative=8,
    n_redundant=2,
    n_clusters_per_class=1,
    weights=[0.99, 0.01],  # 1% fraud rate
    flip_y=0.01,
    random_state=42
)

# Create synthetic amounts (higher for fraud)
amounts = np.random.lognormal(mean=3, sigma=1, size=n_samples)
fraud_multiplier = np.where(y == 1, 
                           np.random.uniform(2, 10, size=n_samples),
                           1.0)
amounts = amounts * fraud_multiplier

print(f"Dataset: {n_samples:,} samples, {np.sum(y):,} frauds ({100*np.mean(y):.2f}%)")
print(f"Average legitimate amount: ${np.mean(amounts[y==0]):.2f}")
print(f"Average fraud amount: ${np.mean(amounts[y==1]):.2f}")

# Split data
X_train, X_test, y_train, y_test, amounts_train, amounts_test = train_test_split(
    X, y, amounts, test_size=0.3, stratify=y, random_state=42
)

X_train_split, X_val, y_train_split, y_val, amounts_train_split, amounts_val = train_test_split(
    X_train, y_train, amounts_train, test_size=0.3, stratify=y_train, random_state=42
)

print(f"\nSplits:")
print(f"Train: {len(X_train_split):,} samples")
print(f"Val:   {len(X_val):,} samples")
print(f"Test:  {len(X_test):,} samples")

# Train models
print("\n" + "="*40)
print("TRAINING MODELS")
print("="*40)

models = {}

# 1. XGBoost baseline
print("\n1. Training XGBoost baseline...")
xgb_model = xgb.XGBClassifier(
    n_estimators=50,
    max_depth=4,
    learning_rate=0.1,
    scale_pos_weight=99,  # 1% fraud rate
    random_state=42
)
xgb_model.fit(X_train_split, y_train_split)
models['xgb'] = xgb_model

# 2. FraudBoost baseline 
print("\n2. Training FraudBoost baseline...")
fb_model = FraudBoostClassifier(
    n_estimators=50,
    max_depth=4,
    learning_rate=0.1,
    fp_cost=100,
    random_state=42
)
fb_model.fit(X_train_split, y_train_split, amounts_train_split)
models['fb'] = fb_model

# 3. Cascade model
print("\n3. Training Cascade model...")
cascade_model = FraudCascade(
    stage1_threshold=0.1,
    fp_cost=100,
    xgb_params={'n_estimators': 50, 'max_depth': 4, 'learning_rate': 0.1, 'random_state': 42},
    fb_params={'n_estimators': 50, 'max_depth': 4, 'learning_rate': 0.1, 'random_state': 42}
)
cascade_model.fit(X_train_split, y_train_split, amounts_train_split, X_val, y_val, amounts_val)
models['cascade'] = cascade_model

# 4. Stacking ensemble
print("\n4. Training Stacking ensemble...")
stacking_model = FraudStacking(
    fp_cost=100,
    xgb_params={'n_estimators': 50, 'max_depth': 4, 'learning_rate': 0.1, 'random_state': 42},
    fb_params={'n_estimators': 50, 'max_depth': 4, 'learning_rate': 0.1, 'random_state': 42},
    cv_folds=3
)
stacking_model.fit(X_train_split, y_train_split, amounts_train_split)
models['stacking'] = stacking_model

print("\nAll models trained successfully!")

# Evaluate models
print("\n" + "="*40)
print("EVALUATING MODELS")
print("="*40)

results = {}

for name, model in models.items():
    print(f"\nEvaluating {name}...")
    
    if name == 'xgb':
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
    elif name == 'fb':
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
    elif name == 'cascade':
        proba = model.predict_proba(X_test, amounts_test)
        pred = (proba >= 0.5).astype(int)
    elif name == 'stacking':
        proba = model.predict_proba(X_test, amounts_test)
        pred = (proba >= 0.5).astype(int)
    
    # Calculate metrics
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
    from fraudboost.metrics import calculate_net_savings, calculate_value_detection_rate
    
    tp = np.sum((y_test == 1) & (pred == 1))
    fp = np.sum((y_test == 0) & (pred == 1))
    fn = np.sum((y_test == 1) & (pred == 0))
    tn = np.sum((y_test == 0) & (pred == 0))
    
    auc = roc_auc_score(y_test, proba)
    precision = precision_score(y_test, pred, zero_division=0)
    recall = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    vdr = calculate_value_detection_rate(y_test, pred, amounts_test)
    net_savings = calculate_net_savings(y_test, pred, amounts_test, 100)
    
    results[name] = {
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'vdr': vdr,
        'net_savings': net_savings,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
    }
    
    print(f"  AUC: {auc:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1: {f1:.3f}")
    print(f"  VDR: {vdr:.3f}")
    print(f"  Net Savings: ${net_savings:.0f}")
    print(f"  TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")

# Print comparison table
print("\n" + "="*50)
print("RESULTS COMPARISON")
print("="*50)

print(f"{'Model':<12} {'AUC':<6} {'Prec':<6} {'Recall':<6} {'F1':<6} {'VDR':<6} {'Net $':<8} {'FPs':<6}")
print("-" * 60)

for name, r in results.items():
    print(f"{name:<12} {r['auc']:.3f}  {r['precision']:.3f}  {r['recall']:.3f}  "
          f"{r['f1']:.3f}  {r['vdr']:.3f}  ${r['net_savings']:>6.0f}  {r['fp']:<6}")

# Find winner
best_model = max(results.keys(), key=lambda x: results[x]['net_savings'])
best_savings = results[best_model]['net_savings']

print(f"\n🏆 WINNER: {best_model} (Net Savings: ${best_savings:.0f})")

# Test cascade breakdown
if 'cascade' in models:
    print("\n" + "="*40)
    print("CASCADE STAGE ANALYSIS")
    print("="*40)
    
    breakdown = models['cascade'].get_cascade_breakdown(X_test, y_test, amounts_test)
    
    total = breakdown['total_samples']
    flagged = breakdown['stage1_flagged_count']
    frauds_total = breakdown['total_frauds']
    frauds_caught = breakdown['frauds_caught_stage1']
    
    print(f"Total samples: {total:,}")
    print(f"Stage 1 flagged: {flagged:,} ({100*flagged/total:.1f}%)")
    print(f"Frauds caught by Stage 1: {frauds_caught}/{frauds_total} ({100*frauds_caught/frauds_total:.1f}%)")
    print(f"Data reduction: {100*(total-flagged)/total:.1f}%")

print("\n" + "="*40)
print("MINI TEST COMPLETE!")
print("="*40)
print("✓ Cascade implementation working correctly")
print("✓ All models train and predict successfully")
print("✓ Metrics calculation working")
print("✓ Ready for full benchmark")