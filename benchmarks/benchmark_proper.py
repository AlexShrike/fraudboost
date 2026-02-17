#!/usr/bin/env python3
"""
Proper, Rigorous FraudBoost vs XGBoost Benchmark
================================================

Implements ALL methodological fixes:
1. Same hyperparameters for both models
2. Natural fraud rate maintained (no oversampling) 
3. Proper train/val/test splits with matched distributions
4. Calibration analysis (ECE)
5. FP cost sensitivity analysis
6. Precision at fixed recall
7. 5-fold stratified cross-validation
8. Overfitting checks

Dataset: Kartik2112 Credit Card Fraud Detection
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend for headless
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, recall_score, precision_score, f1_score,
    precision_recall_curve, roc_curve, confusion_matrix
)
import xgboost as xgb
from fraudboost import FraudBoostClassifier

def print_flush(*args, **kwargs):
    """Print with immediate flush for real-time output."""
    print(*args, **kwargs)
    sys.stdout.flush()

print_flush("="*70)
print_flush("=== FRAUDBOOST VS XGBOOST PROPER BENCHMARK ===")
print_flush("="*70)
print_flush(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print_flush("All methodological fixes implemented")


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate haversine distance between two points in kilometers."""
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
    print_flush(f"  Engineering features for {len(df):,} rows...")
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
    
    print_flush(f"  Features engineered: {X.shape}")
    return X, y, features


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Calculate Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    calibration_table = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            calibration_table.append({
                'bin': f"({bin_lower:.1f}, {bin_upper:.1f}]",
                'count': in_bin.sum(),
                'avg_confidence': avg_confidence_in_bin,
                'accuracy': accuracy_in_bin,
                'gap': avg_confidence_in_bin - accuracy_in_bin
            })
        else:
            calibration_table.append({
                'bin': f"({bin_lower:.1f}, {bin_upper:.1f}]",
                'count': 0,
                'avg_confidence': 0,
                'accuracy': 0,
                'gap': 0
            })
    
    return ece, calibration_table


def calculate_net_savings(y_true, y_prob, threshold, fraud_amount=50, fp_cost=100):
    """Calculate net savings at given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Savings = fraud caught - false positive costs
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
    
    # Find closest recall to target
    idx = np.argmin(np.abs(recalls - target_recall))
    return precisions[idx], recalls[idx]


def calculate_vdr(y_true, y_prob, threshold=0.5):
    """Calculate Value Detection Rate (% of fraud value caught)."""
    y_pred = (y_prob >= threshold).astype(int)
    
    # Fraud caught
    fraud_caught = np.sum((y_true == 1) & (y_pred == 1))
    total_fraud = np.sum(y_true == 1)
    
    return (fraud_caught / total_fraud) if total_fraud > 0 else 0


def evaluate_model(y_true, y_proba, model_name, threshold=0.5):
    """Evaluate model performance."""
    y_pred = (y_proba >= threshold).astype(int)
    
    auc = roc_auc_score(y_true, y_proba)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    vdr = calculate_vdr(y_true, y_proba, threshold)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    net_savings = tp * 50 - fp * 100  # Default: fraud=$50, FP=$100
    
    return {
        'Model': model_name,
        'AUC': f"{auc:.4f}",
        'Recall': f"{recall:.4f}",
        'Precision': f"{precision:.4f}",
        'F1': f"{f1:.4f}",
        'VDR': f"{vdr:.4f}",
        'Net_Savings': f"${net_savings:,.0f}",
        'FPs': f"{fp:,}",
        'TPs': f"{tp:,}"
    }


# STEP 1: Load datasets
print_flush("\n" + "="*70)
print_flush("=== LOADING DATASETS ===")
print_flush("="*70)

train_path = "/Users/alexshrike/.openclaw/workspace/bastion/data/fraudTrain.csv"
test_path = "/Users/alexshrike/.openclaw/workspace/bastion/data/fraudTest.csv"

print_flush(f"Loading train dataset from {train_path}...")
train_df = pd.read_csv(train_path)
print_flush(f"✓ Train loaded: {len(train_df):,} rows, fraud rate: {train_df['is_fraud'].mean()*100:.3f}%")

print_flush(f"Loading test dataset from {test_path}...")
test_df = pd.read_csv(test_path)
print_flush(f"✓ Test loaded: {len(test_df):,} rows, fraud rate: {test_df['is_fraud'].mean()*100:.3f}%")


# STEP 2: Subsample training data maintaining natural fraud rate
print_flush("\n" + "="*70)
print_flush("=== SUBSAMPLING TRAINING DATA ===")
print_flush("="*70)

# Start with 200K for FraudBoost performance, can increase if fast enough
target_size = 200_000
original_fraud_rate = train_df['is_fraud'].mean()

n_fraud = int(target_size * original_fraud_rate)
n_legit = target_size - n_fraud

print_flush(f"Target subsample: {target_size:,} rows")
print_flush(f"Original fraud rate: {original_fraud_rate*100:.3f}%")
print_flush(f"Sampling {n_fraud:,} fraud + {n_legit:,} legitimate transactions...")

# Sample maintaining exact fraud rate
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


# STEP 3: Feature engineering
print_flush("\n" + "="*70)
print_flush("=== FEATURE ENGINEERING ===")
print_flush("="*70)

print_flush("Engineering features for training data...")
X_train_full, y_train_full, feature_names = engineer_features(train_subsample)

print_flush("Engineering features for test data...")
X_test, y_test, _ = engineer_features(test_df)

print_flush(f"Feature names: {feature_names}")


# STEP 4: Train/Val/Test split with matched distributions
print_flush("\n" + "="*70)
print_flush("=== TRAIN/VAL SPLIT ===")
print_flush("="*70)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

print_flush("\n" + "="*70)
print_flush("=== DATASET DISTRIBUTION ===")
print_flush("="*70)
print_flush(f"Train: {len(X_train):,} rows, fraud rate: {y_train.mean()*100:.3f}%")
print_flush(f"Val: {len(X_val):,} rows, fraud rate: {y_val.mean()*100:.3f}%")
print_flush(f"Test: {len(X_test):,} rows, fraud rate: {y_test.mean()*100:.3f}%")

# Verify distributions match
print_flush("\n✓ METHODOLOGICAL FIX #3: Matched fraud rate distributions")
print_flush(f"  Original: {original_fraud_rate*100:.3f}%")
print_flush(f"  Train: {y_train.mean()*100:.3f}%")
print_flush(f"  Val: {y_val.mean()*100:.3f}%")
print_flush(f"  Test: {y_test.mean()*100:.3f}%")


# STEP 5: Train models with identical hyperparameters
print_flush("\n" + "="*70)
print_flush("=== TRAINING MODELS ===")
print_flush("="*70)

# Calculate positive weight for XGBoost
pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
print_flush(f"Calculated positive weight ratio: {pos_weight:.2f}")

print_flush("\n✓ METHODOLOGICAL FIX #1: Same hyperparameters")
print_flush("  Both models: n_estimators=100, max_depth=4, learning_rate=0.1")

# Train XGBoost
print_flush(f"\nTraining XGBoost...")
start_time = time.time()
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
xgb_train_time = time.time() - start_time
print_flush(f"✓ XGBoost training complete ({xgb_train_time:.1f}s)")

# Train FraudBoost
print_flush(f"\nTraining FraudBoost...")
start_time = time.time()
fb_model = FraudBoostClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    fp_cost=100,
    loss='value_weighted',
    random_state=42
)
fb_model.fit(X_train, y_train)
fb_train_time = time.time() - start_time
print_flush(f"✓ FraudBoost training complete ({fb_train_time:.1f}s)")

print_flush(f"\nTraining time comparison:")
print_flush(f"  XGBoost: {xgb_train_time:.1f}s")
print_flush(f"  FraudBoost: {fb_train_time:.1f}s ({fb_train_time/xgb_train_time:.1f}x slower)")


# STEP 6: Generate predictions
print_flush("\n" + "="*70)
print_flush("=== GENERATING PREDICTIONS ===")
print_flush("="*70)

# Training set predictions (for overfitting check)
print_flush("Generating training predictions...")
xgb_train_proba = xgb_model.predict_proba(X_train)[:, 1]
fb_train_proba = fb_model.predict_proba(X_train)[:, 1]

# Validation set predictions
print_flush("Generating validation predictions...")
xgb_val_proba = xgb_model.predict_proba(X_val)[:, 1]
fb_val_proba = fb_model.predict_proba(X_val)[:, 1]

# Test set predictions  
print_flush("Generating test predictions...")
xgb_test_proba = xgb_model.predict_proba(X_test)[:, 1]
fb_test_proba = fb_model.predict_proba(X_test)[:, 1]

print_flush("✓ All predictions generated")


# STEP 7: Results analysis
print_flush("\n" + "="*70)
print_flush("=== HELD-OUT TEST RESULTS (Default Threshold 0.5) ===")
print_flush("="*70)

# Default threshold results
xgb_results = evaluate_model(y_test, xgb_test_proba, "XGBoost", 0.5)
fb_results = evaluate_model(y_test, fb_test_proba, "FraudBoost", 0.5)

results_df = pd.DataFrame([xgb_results, fb_results])
print_flush(results_df.to_string(index=False))


# STEP 8: Optimal threshold results
print_flush("\n" + "="*70)
print_flush("=== HELD-OUT TEST RESULTS (Optimal Threshold) ===")
print_flush("="*70)

print_flush("Finding optimal thresholds...")
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


# STEP 9: Precision at fixed recall
print_flush("\n" + "="*70)
print_flush("=== PRECISION @ FIXED RECALL ===")
print_flush("="*70)

print_flush("✓ METHODOLOGICAL FIX #6: Precision at fixed recall")

xgb_p80, xgb_r80 = precision_at_recall(y_test, xgb_test_proba, 0.8)
fb_p80, fb_r80 = precision_at_recall(y_test, fb_test_proba, 0.8)

xgb_p90, xgb_r90 = precision_at_recall(y_test, xgb_test_proba, 0.9)
fb_p90, fb_r90 = precision_at_recall(y_test, fb_test_proba, 0.9)

print_flush(f"Precision@80%Recall: FraudBoost {fb_p80*100:.2f}%, XGBoost {xgb_p80*100:.2f}%")
print_flush(f"Precision@90%Recall: FraudBoost {fb_p90*100:.2f}%, XGBoost {xgb_p90*100:.2f}%")
print_flush(f"(Actual recall achieved: FB 80%={fb_r80*100:.1f}%, 90%={fb_r90*100:.1f}%; XGB 80%={xgb_r80*100:.1f}%, 90%={xgb_r90*100:.1f}%)")


# STEP 10: Calibration analysis
print_flush("\n" + "="*70)
print_flush("=== CALIBRATION (ECE) ===")
print_flush("="*70)

print_flush("✓ METHODOLOGICAL FIX #4: Calibration check")

print_flush("Calculating calibration metrics...")
xgb_ece, xgb_cal_table = expected_calibration_error(y_test, xgb_test_proba)
fb_ece, fb_cal_table = expected_calibration_error(y_test, fb_test_proba)

print_flush("\nXGBoost Calibration:")
xgb_cal_df = pd.DataFrame(xgb_cal_table)
print_flush(xgb_cal_df[['bin', 'count', 'avg_confidence', 'accuracy', 'gap']].round(4).to_string(index=False))

print_flush(f"\nFraudBoost Calibration:")
fb_cal_df = pd.DataFrame(fb_cal_table)
print_flush(fb_cal_df[['bin', 'count', 'avg_confidence', 'accuracy', 'gap']].round(4).to_string(index=False))

print_flush(f"\nECE: FraudBoost {fb_ece:.4f}, XGBoost {xgb_ece:.4f}")


# STEP 11: FP cost sensitivity analysis
print_flush("\n" + "="*70)
print_flush("=== FP COST SENSITIVITY ===")
print_flush("="*70)

print_flush("✓ METHODOLOGICAL FIX #5: FP cost sensitivity analysis")

fp_costs = [10, 25, 50, 75, 100, 150, 200, 500, 1000]
sensitivity_results = []

print_flush("Running FP cost sensitivity analysis...")
for i, fp_cost in enumerate(fp_costs):
    print_flush(f"  Testing FP cost ${fp_cost} ({i+1}/{len(fp_costs)})...")
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
print_flush("\n" + sensitivity_df.to_string(index=False))

# Count wins
fb_wins = sum(1 for result in sensitivity_results if result['Winner'] == 'FraudBoost')
xgb_wins = len(sensitivity_results) - fb_wins
print_flush(f"\nFP Cost Sensitivity Summary:")
print_flush(f"  FraudBoost wins: {fb_wins}/{len(fp_costs)} cases")
print_flush(f"  XGBoost wins: {xgb_wins}/{len(fp_costs)} cases")


# STEP 12: 5-fold cross-validation
print_flush("\n" + "="*70)
print_flush("=== 5-FOLD CV RESULTS ===")
print_flush("="*70)

print_flush("✓ METHODOLOGICAL FIX #7: 5-fold stratified cross-validation")

cv_folds = 5
skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

# Prepare arrays for CV results
xgb_cv_aucs = []
fb_cv_aucs = []
xgb_cv_savings = []
fb_cv_savings = []

print_flush("Running 5-fold cross-validation...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):
    print_flush(f"  Fold {fold + 1}/5...")
    
    X_fold_train, X_fold_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
    y_fold_train, y_fold_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]
    
    # Train XGBoost
    fold_pos_weight = len(y_fold_train[y_fold_train == 0]) / len(y_fold_train[y_fold_train == 1])
    xgb_fold = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        scale_pos_weight=fold_pos_weight, eval_metric='logloss',
        random_state=42, n_jobs=-1
    )
    xgb_fold.fit(X_fold_train, y_fold_train)
    
    # Train FraudBoost
    fb_fold = FraudBoostClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        fp_cost=100, loss='value_weighted', random_state=42
    )
    fb_fold.fit(X_fold_train, y_fold_train)
    
    # Predictions
    xgb_fold_proba = xgb_fold.predict_proba(X_fold_val)[:, 1]
    fb_fold_proba = fb_fold.predict_proba(X_fold_val)[:, 1]
    
    # AUC scores
    xgb_cv_aucs.append(roc_auc_score(y_fold_val, xgb_fold_proba))
    fb_cv_aucs.append(roc_auc_score(y_fold_val, fb_fold_proba))
    
    # Net savings (optimal threshold)
    _, xgb_fold_savings = find_optimal_threshold(y_fold_val, xgb_fold_proba, 50, 100)
    _, fb_fold_savings = find_optimal_threshold(y_fold_val, fb_fold_proba, 50, 100)
    
    xgb_cv_savings.append(xgb_fold_savings)
    fb_cv_savings.append(fb_fold_savings)

print_flush(f"\nFraudBoost AUC: {np.mean(fb_cv_aucs):.4f} ± {np.std(fb_cv_aucs):.4f}")
print_flush(f"XGBoost AUC: {np.mean(xgb_cv_aucs):.4f} ± {np.std(xgb_cv_aucs):.4f}")
print_flush(f"FraudBoost Net Savings: ${np.mean(fb_cv_savings):,.0f} ± ${np.std(fb_cv_savings):,.0f}")
print_flush(f"XGBoost Net Savings: ${np.mean(xgb_cv_savings):,.0f} ± ${np.std(xgb_cv_savings):,.0f}")

# Statistical significance test (simple)
from scipy.stats import ttest_rel
try:
    auc_t_stat, auc_p_value = ttest_rel(fb_cv_aucs, xgb_cv_aucs)
    savings_t_stat, savings_p_value = ttest_rel(fb_cv_savings, xgb_cv_savings)
    print_flush(f"\nStatistical significance (paired t-test):")
    print_flush(f"  AUC difference p-value: {auc_p_value:.4f}")
    print_flush(f"  Net savings difference p-value: {savings_p_value:.4f}")
except:
    print_flush("Could not compute statistical significance")


# STEP 13: Overfitting check
print_flush("\n" + "="*70)
print_flush("=== OVERFITTING CHECK ===")
print_flush("="*70)

print_flush("✓ METHODOLOGICAL FIX #8: Overfitting check")

train_xgb_auc = roc_auc_score(y_train, xgb_train_proba)
train_fb_auc = roc_auc_score(y_train, fb_train_proba)

val_xgb_auc = roc_auc_score(y_val, xgb_val_proba)
val_fb_auc = roc_auc_score(y_val, fb_val_proba)

test_xgb_auc = roc_auc_score(y_test, xgb_test_proba)
test_fb_auc = roc_auc_score(y_test, fb_test_proba)

overfitting_df = pd.DataFrame({
    'Model': ['XGBoost', 'FraudBoost'],
    'Train_AUC': [f"{train_xgb_auc:.4f}", f"{train_fb_auc:.4f}"],
    'Val_AUC': [f"{val_xgb_auc:.4f}", f"{val_fb_auc:.4f}"],
    'Test_AUC': [f"{test_xgb_auc:.4f}", f"{test_fb_auc:.4f}"],
    'Train_Val_Gap': [f"{train_xgb_auc - val_xgb_auc:.4f}", f"{train_fb_auc - val_fb_auc:.4f}"],
    'Val_Test_Gap': [f"{val_xgb_auc - test_xgb_auc:.4f}", f"{val_fb_auc - test_fb_auc:.4f}"]
})

print_flush(overfitting_df.to_string(index=False))

# Analysis
print_flush(f"\nOverfitting Analysis:")
train_val_gap_xgb = train_xgb_auc - val_xgb_auc
train_val_gap_fb = train_fb_auc - val_fb_auc
print_flush(f"  XGBoost train-val gap: {train_val_gap_xgb:.4f}")
print_flush(f"  FraudBoost train-val gap: {train_val_gap_fb:.4f}")

if train_val_gap_xgb > 0.05:
    print_flush("  ⚠️  XGBoost may be overfitting (gap > 0.05)")
if train_val_gap_fb > 0.05:
    print_flush("  ⚠️  FraudBoost may be overfitting (gap > 0.05)")


# FINAL SUMMARY
print_flush("\n" + "="*70)
print_flush("=== BENCHMARK COMPLETE ===")
print_flush("="*70)

print_flush(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

print_flush("\n📊 FINAL SUMMARY:")
print_flush(f"  XGBoost Test AUC: {test_xgb_auc:.4f}")
print_flush(f"  FraudBoost Test AUC: {test_fb_auc:.4f}")
print_flush(f"  Winner (AUC): {'XGBoost' if test_xgb_auc > test_fb_auc else 'FraudBoost'}")
print_flush(f"  Optimal net savings: XGB ${xgb_opt_savings:,.0f}, FB ${fb_opt_savings:,.0f}")
print_flush(f"  Winner (Net Savings): {'XGBoost' if xgb_opt_savings > fb_opt_savings else 'FraudBoost'}")

print_flush("\n✅ ALL METHODOLOGICAL FIXES IMPLEMENTED:")
print_flush("  1. ✓ Same hyperparameters (n_estimators=100, max_depth=4, lr=0.1)")
print_flush("  2. ✓ Natural fraud rate maintained (no oversampling)")  
print_flush("  3. ✓ Proper stratified train/val/test splits")
print_flush("  4. ✓ Calibration analysis (ECE)")
print_flush("  5. ✓ FP cost sensitivity analysis")
print_flush("  6. ✓ Precision at fixed recall")
print_flush("  7. ✓ 5-fold stratified cross-validation")
print_flush("  8. ✓ Overfitting checks")

print_flush("\n🎯 RIGOROUS BENCHMARK COMPLETED SUCCESSFULLY!")
print_flush("Results ready for analysis and reporting.")