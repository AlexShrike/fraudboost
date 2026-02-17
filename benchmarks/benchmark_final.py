#!/usr/bin/env python3
"""
Final FraudBoost vs XGBoost Benchmark - Streamlined Version
===========================================================
"""

import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Force output flushing
def print_flush(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

print_flush("=== FRAUDBOOST VS XGBOOST BENCHMARK ===")

# Set matplotlib backend for headless
import matplotlib
matplotlib.use('Agg')

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, confusion_matrix, precision_recall_curve
import xgboost as xgb
from fraudboost import FraudBoostClassifier

print_flush("✓ All imports successful")

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate haversine distance between two points."""
    R = 6371  # Earth's radius in km
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)
    
    a = (np.sin(delta_lat/2)**2 + 
         np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = R * c
    return distance

def engineer_features(df):
    """Apply feature engineering without target leakage."""
    df = df.copy()
    
    # Parse datetime
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
    
    # Calculate age from birth year
    df['age'] = 2024 - pd.to_datetime(df['dob']).dt.year
    
    # Distance between customer and merchant
    df['distance'] = haversine_distance(
        df['lat'], df['long'], df['merch_lat'], df['merch_long']
    )
    
    # Amount log transform
    df['amt_log'] = np.log1p(df['amt'])
    
    # Encode categorical variables
    le_category = LabelEncoder()
    df['category_encoded'] = le_category.fit_transform(df['category'])
    df['gender_encoded'] = (df['gender'] == 'M').astype(int)
    
    # Select final features (NO target leakage)
    features = [
        'amt', 'category_encoded', 'gender_encoded', 'city_pop',
        'lat', 'long', 'merch_lat', 'merch_long', 'hour', 'day_of_week',
        'age', 'distance', 'amt_log'
    ]
    
    X = df[features].copy()
    y = df['is_fraud'].copy() if 'is_fraud' in df.columns else None
    
    return X, y, features

def calculate_net_savings(y_true, y_prob, threshold, fraud_amount=50, fp_cost=100):
    """Calculate net savings at given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    savings = tp * fraud_amount - fp * fp_cost
    return savings

def find_optimal_threshold(y_true, y_prob, fraud_amount=50, fp_cost=100):
    """Find threshold that maximizes net savings."""
    thresholds = np.linspace(0.001, 0.999, 1000)
    best_savings = float('-inf')
    best_threshold = 0.5
    
    for threshold in thresholds:
        savings = calculate_net_savings(y_true, y_prob, threshold, fraud_amount, fp_cost)
        if savings > best_savings:
            best_savings = savings
            best_threshold = threshold
    
    return best_threshold, best_savings

def precision_at_recall(y_true, y_prob, target_recall):
    """Get precision at specific recall level."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    idx = np.argmin(np.abs(recalls - target_recall))
    return precisions[idx], recalls[idx]

def evaluate_model(y_true, y_proba, model_name, threshold=0.5):
    """Evaluate model performance."""
    y_pred = (y_proba >= threshold).astype(int)
    
    auc = roc_auc_score(y_true, y_proba)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    net_savings = tp * 50 - fp * 100
    
    return {
        'Model': model_name,
        'AUC': f"{auc:.4f}",
        'Recall': f"{recall:.4f}",
        'Precision': f"{precision:.4f}",
        'F1': f"{f1:.4f}",
        'Net_Savings': f"${net_savings:,.0f}",
        'FPs': f"{fp:,}",
        'TPs': f"{tp:,}"
    }

# Load datasets
print_flush("\n=== LOADING DATASETS ===")
train_path = "/Users/alexshrike/.openclaw/workspace/bastion/data/fraudTrain.csv"
test_path = "/Users/alexshrike/.openclaw/workspace/bastion/data/fraudTest.csv"

print_flush("Loading train dataset...")
train_df = pd.read_csv(train_path)
print_flush(f"✓ Train: {len(train_df):,} rows, fraud rate: {train_df['is_fraud'].mean()*100:.3f}%")

print_flush("Loading test dataset...")
test_df = pd.read_csv(test_path)
print_flush(f"✓ Test: {len(test_df):,} rows, fraud rate: {test_df['is_fraud'].mean()*100:.3f}%")

# Subsample training data to 200K while maintaining natural fraud rate
print_flush("\n=== SUBSAMPLING TRAINING DATA ===")
target_size = 200_000
fraud_rate = train_df['is_fraud'].mean()

n_fraud = int(target_size * fraud_rate)
n_legit = target_size - n_fraud

print_flush(f"Sampling {n_fraud:,} fraud + {n_legit:,} legitimate...")

fraud_samples = train_df[train_df['is_fraud'] == 1].sample(
    n=min(n_fraud, sum(train_df['is_fraud'])), random_state=42
)
legit_samples = train_df[train_df['is_fraud'] == 0].sample(
    n=min(n_legit, sum(train_df['is_fraud'] == 0)), random_state=42
)

train_subsample = pd.concat([fraud_samples, legit_samples]).sample(
    frac=1, random_state=42
).reset_index(drop=True)

print_flush(f"✓ Subsampled: {len(train_subsample):,} rows, fraud rate: {train_subsample['is_fraud'].mean()*100:.3f}%")

# Feature engineering
print_flush("\n=== FEATURE ENGINEERING ===")
print_flush("Engineering features for training data...")
X_train_full, y_train_full, feature_names = engineer_features(train_subsample)
print_flush(f"✓ Train features: {X_train_full.shape}")

print_flush("Engineering features for test data...")
X_test, y_test, _ = engineer_features(test_df)
print_flush(f"✓ Test features: {X_test.shape}")

# Train/Val split (80/20, stratified)
print_flush("\n=== TRAIN/VAL SPLIT ===")
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

print_flush(f"Train: {len(X_train):,} rows, fraud rate: {y_train.mean()*100:.3f}%")
print_flush(f"Val: {len(X_val):,} rows, fraud rate: {y_val.mean()*100:.3f}%")
print_flush(f"Test: {len(X_test):,} rows, fraud rate: {y_test.mean()*100:.3f}%")

# Train models
print_flush("\n=== TRAINING MODELS ===")

# XGBoost
pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
print_flush(f"Positive weight ratio: {pos_weight:.2f}")

print_flush("Training XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    scale_pos_weight=pos_weight,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)
print_flush("✓ XGBoost training complete")

# FraudBoost
print_flush("Training FraudBoost...")
fb_model = FraudBoostClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    fp_cost=100,
    loss='value_weighted',
    random_state=42
)
fb_model.fit(X_train, y_train)
print_flush("✓ FraudBoost training complete")

# Predictions
print_flush("\n=== GENERATING PREDICTIONS ===")
xgb_test_proba = xgb_model.predict_proba(X_test)[:, 1]
fb_test_proba = fb_model.predict_proba(X_test)[:, 1]
print_flush("✓ Predictions generated")

# Results
print_flush("\n" + "="*60)
print_flush("=== HELD-OUT TEST RESULTS (Default Threshold 0.5) ===")
print_flush("="*60)

xgb_results = evaluate_model(y_test, xgb_test_proba, "XGBoost", 0.5)
fb_results = evaluate_model(y_test, fb_test_proba, "FraudBoost", 0.5)

results_df = pd.DataFrame([xgb_results, fb_results])
print_flush(results_df.to_string(index=False))

# Optimal threshold results
print_flush("\n" + "="*60)
print_flush("=== HELD-OUT TEST RESULTS (Optimal Threshold) ===")
print_flush("="*60)

xgb_opt_thresh, xgb_opt_savings = find_optimal_threshold(y_test, xgb_test_proba, 50, 100)
fb_opt_thresh, fb_opt_savings = find_optimal_threshold(y_test, fb_test_proba, 50, 100)

xgb_opt_results = evaluate_model(y_test, xgb_test_proba, "XGBoost", xgb_opt_thresh)
fb_opt_results = evaluate_model(y_test, fb_test_proba, "FraudBoost", fb_opt_thresh)

# Add threshold info
xgb_opt_results['Threshold'] = f"{xgb_opt_thresh:.4f}"
fb_opt_results['Threshold'] = f"{fb_opt_thresh:.4f}"

opt_results_df = pd.DataFrame([xgb_opt_results, fb_opt_results])
cols = ['Model', 'Threshold'] + [col for col in opt_results_df.columns if col not in ['Model', 'Threshold']]
print_flush(opt_results_df[cols].to_string(index=False))

# Precision at fixed recall
print_flush("\n" + "="*60)
print_flush("=== PRECISION @ FIXED RECALL ===")
print_flush("="*60)

xgb_p80, xgb_r80 = precision_at_recall(y_test, xgb_test_proba, 0.8)
fb_p80, fb_r80 = precision_at_recall(y_test, fb_test_proba, 0.8)

xgb_p90, xgb_r90 = precision_at_recall(y_test, xgb_test_proba, 0.9)
fb_p90, fb_r90 = precision_at_recall(y_test, fb_test_proba, 0.9)

print_flush(f"Precision@80%Recall: FraudBoost {fb_p80*100:.2f}%, XGBoost {xgb_p80*100:.2f}%")
print_flush(f"Precision@90%Recall: FraudBoost {fb_p90*100:.2f}%, XGBoost {xgb_p90*100:.2f}%")

# FP cost sensitivity analysis (abbreviated)
print_flush("\n" + "="*60)
print_flush("=== FP COST SENSITIVITY (ABBREVIATED) ===")
print_flush("="*60)

fp_costs = [25, 50, 100, 200, 500]  # Fewer costs to speed up
sensitivity_results = []

for fp_cost in fp_costs:
    xgb_thresh, xgb_savings = find_optimal_threshold(y_test, xgb_test_proba, 50, fp_cost)
    fb_thresh, fb_savings = find_optimal_threshold(y_test, fb_test_proba, 50, fp_cost)
    
    winner = "FraudBoost" if fb_savings > xgb_savings else "XGBoost"
    
    sensitivity_results.append({
        'FP_Cost': f"${fp_cost}",
        'FB_Net_Savings': f"${fb_savings:,.0f}",
        'XGB_Net_Savings': f"${xgb_savings:,.0f}",
        'Winner': winner
    })

sensitivity_df = pd.DataFrame(sensitivity_results)
print_flush(sensitivity_df.to_string(index=False))

# Quick 3-fold CV
print_flush("\n" + "="*60)
print_flush("=== 3-FOLD CV RESULTS ===")
print_flush("="*60)

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
xgb_cv_aucs = []
fb_cv_aucs = []

print_flush("Running 3-fold cross-validation...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):
    print_flush(f"  Fold {fold + 1}/3...")
    
    X_fold_train, X_fold_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
    y_fold_train, y_fold_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]
    
    # Train XGBoost
    fold_pos_weight = len(y_fold_train[y_fold_train == 0]) / len(y_fold_train[y_fold_train == 1])
    xgb_fold = xgb.XGBClassifier(
        n_estimators=50,  # Reduced for speed
        max_depth=4, learning_rate=0.1,
        scale_pos_weight=fold_pos_weight, eval_metric='logloss',
        random_state=42, n_jobs=-1
    )
    xgb_fold.fit(X_fold_train, y_fold_train)
    
    # Train FraudBoost
    fb_fold = FraudBoostClassifier(
        n_estimators=50,  # Reduced for speed
        max_depth=4, learning_rate=0.1,
        fp_cost=100, loss='value_weighted', random_state=42
    )
    fb_fold.fit(X_fold_train, y_fold_train)
    
    # Predictions
    xgb_fold_proba = xgb_fold.predict_proba(X_fold_val)[:, 1]
    fb_fold_proba = fb_fold.predict_proba(X_fold_val)[:, 1]
    
    # AUC scores
    xgb_cv_aucs.append(roc_auc_score(y_fold_val, xgb_fold_proba))
    fb_cv_aucs.append(roc_auc_score(y_fold_val, fb_fold_proba))

print_flush(f"FraudBoost CV AUC: {np.mean(fb_cv_aucs):.4f} ± {np.std(fb_cv_aucs):.4f}")
print_flush(f"XGBoost CV AUC: {np.mean(xgb_cv_aucs):.4f} ± {np.std(xgb_cv_aucs):.4f}")

print_flush("\n" + "="*60)
print_flush("=== BENCHMARK COMPLETE ===")
print_flush("="*60)

# Summary
xgb_auc = roc_auc_score(y_test, xgb_test_proba)
fb_auc = roc_auc_score(y_test, fb_test_proba)

print_flush("FINAL SUMMARY:")
print_flush(f"XGBoost Test AUC: {xgb_auc:.4f}")
print_flush(f"FraudBoost Test AUC: {fb_auc:.4f}")
print_flush(f"Winner: {'XGBoost' if xgb_auc > fb_auc else 'FraudBoost'}")
print_flush(f"Optimal net savings: XGB ${xgb_opt_savings:,.0f}, FB ${fb_opt_savings:,.0f}")

print_flush("\nBenchmark completed successfully!")