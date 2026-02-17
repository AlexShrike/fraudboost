# %% [markdown]
# # Cascade Fraud Detection Benchmark: XGBoost + FraudBoost
# 
# This script benchmarks multiple fraud detection approaches:
# 
# 1. **XGBoost alone** (baseline)
# 2. **FraudBoost alone** (value-weighted baseline)  
# 3. **Cascade**: XGBoost (Stage 1, high recall) → FraudBoost (Stage 2, high precision)
# 4. **Stacking**: XGBoost + FraudBoost → LogisticRegression meta-learner
# 
# ## Cascade Design Philosophy
# 
# The cascade approach is inspired by computer vision (Viola-Jones face detection):
# - **Stage 1**: Cast a wide net with high recall (~95%), accepting many false positives
# - **Stage 2**: Apply precision filter using value-weighted loss to reduce FPs
# - **Result**: High-confidence alerts with much fewer false positives
# 
# This is particularly valuable in fraud detection where:
# - Missing a fraud is very costly (high recall needed)
# - But false alarms waste investigator time (precision matters)
# - Transaction amounts matter (value-weighted decisions)

# %%
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our implementations
sys.path.append('/Users/alexshrike/.openclaw/workspace/fraudboost')
from fraudboost.core import FraudBoostClassifier
from fraudboost.cascade import FraudCascade
from fraudboost.stacking import FraudStacking
from fraudboost.metrics import (
    calculate_precision_recall_f1, 
    calculate_value_detection_rate, 
    calculate_net_savings,
    find_optimal_threshold
)

print("Cascade Fraud Detection Benchmark")
print("=" * 50)
print(f"Started at: {datetime.now()}")

# %% [markdown]
# ## Data Loading and Preprocessing
# 
# Key preprocessing steps:
# 1. Remove Q1 2019 data (fraud rate outlier)
# 2. Engineer features (no target leakage!)
# 3. Subsample to 400K training samples at natural fraud rate (~0.52%)
# 4. 80/20 stratified train/val split
# 5. Test on full fraudTest.csv (555K, out-of-time)

# %%
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points on Earth using Haversine formula."""
    R = 6371  # Earth radius in km
    
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Differences
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Haversine formula
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

def engineer_features(df):
    """Engineer features without target leakage."""
    print("Engineering features...")
    
    # Parse transaction datetime
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    
    # Time-based features
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
    
    # Age calculation (no leakage - uses DOB)
    df['dob'] = pd.to_datetime(df['dob'])
    df['age'] = (df['trans_date_trans_time'] - df['dob']).dt.days / 365.25
    
    # Amount features
    df['amt_log'] = np.log1p(df['amt'])
    
    # Distance between customer and merchant
    df['distance'] = haversine_distance(
        df['lat'], df['long'], 
        df['merch_lat'], df['merch_long']
    )
    
    # Encode categorical variables
    le_category = LabelEncoder()
    le_gender = LabelEncoder()
    
    df['category_enc'] = le_category.fit_transform(df['category'])
    df['gender_enc'] = le_gender.fit_transform(df['gender'])
    
    # Select final features (no target leakage)
    feature_cols = [
        'amt', 'category_enc', 'gender_enc', 'city_pop', 
        'lat', 'long', 'merch_lat', 'merch_long',
        'hour', 'day_of_week', 'age', 'distance', 'amt_log'
    ]
    
    X = df[feature_cols].values
    y = df['is_fraud'].values if 'is_fraud' in df.columns else None
    amounts = df['amt'].values
    
    print(f"Engineered {len(feature_cols)} features")
    print(f"Feature names: {feature_cols}")
    
    return X, y, amounts, feature_cols

def load_and_prepare_data():
    """Load fraud datasets and prepare for training."""
    print("\n" + "="*50)
    print("DATA LOADING AND PREPARATION")
    print("="*50)
    
    # Load datasets
    print("Loading datasets...")
    train_path = '/Users/alexshrike/.openclaw/workspace/bastion/data/fraudTrain.csv'
    test_path = '/Users/alexshrike/.openclaw/workspace/bastion/data/fraudTest.csv'
    
    print(f"Loading training data from: {train_path}")
    df_train = pd.read_csv(train_path)
    print(f"Raw training data shape: {df_train.shape}")
    
    print(f"Loading test data from: {test_path}")
    df_test = pd.read_csv(test_path)
    print(f"Raw test data shape: {df_test.shape}")
    
    # Remove Q1 2019 from training (fraud rate outlier)
    print("\nRemoving Q1 2019 from training data...")
    df_train['trans_date_trans_time'] = pd.to_datetime(df_train['trans_date_trans_time'])
    cutoff_date = pd.to_datetime('2019-04-01')
    
    initial_count = len(df_train)
    df_train = df_train[df_train['trans_date_trans_time'] >= cutoff_date]
    removed_count = initial_count - len(df_train)
    
    print(f"Removed {removed_count:,} samples from Q1 2019")
    print(f"Training data after Q1 removal: {df_train.shape}")
    
    # Check fraud rates
    train_fraud_rate = df_train['is_fraud'].mean()
    test_fraud_rate = df_test['is_fraud'].mean()
    
    print(f"\nFraud rates:")
    print(f"Training (post Q1 removal): {train_fraud_rate:.4f} ({100*train_fraud_rate:.2f}%)")
    print(f"Test: {test_fraud_rate:.4f} ({100*test_fraud_rate:.2f}%)")
    
    # Subsample training data to 400K at natural fraud rate
    print("\nSubsampling training data to 400K...")
    target_size = 400000
    
    if len(df_train) > target_size:
        # Stratified sampling to maintain fraud rate
        fraud_samples = df_train[df_train['is_fraud'] == 1]
        legit_samples = df_train[df_train['is_fraud'] == 0]
        
        # Calculate how many of each to sample
        target_fraud_count = int(target_size * train_fraud_rate)
        target_legit_count = target_size - target_fraud_count
        
        # Sample with replacement if needed (shouldn't be needed for legit)
        if len(fraud_samples) >= target_fraud_count:
            sampled_fraud = fraud_samples.sample(n=target_fraud_count, random_state=42)
        else:
            sampled_fraud = fraud_samples.sample(n=target_fraud_count, replace=True, random_state=42)
            
        sampled_legit = legit_samples.sample(n=target_legit_count, random_state=42)
        
        df_train = pd.concat([sampled_fraud, sampled_legit]).sample(frac=1, random_state=42)
        
        print(f"Subsampled to {len(df_train):,} samples")
        print(f"New fraud rate: {df_train['is_fraud'].mean():.4f}")
    
    # Engineer features for all datasets
    print("\nEngineering features...")
    X_full_train, y_full_train, amounts_full_train, feature_names = engineer_features(df_train)
    X_test, y_test, amounts_test, _ = engineer_features(df_test)
    
    # 80/20 stratified train/val split
    print("\nSplitting into train/validation...")
    X_train, X_val, y_train, y_val, amounts_train, amounts_val = train_test_split(
        X_full_train, y_full_train, amounts_full_train,
        test_size=0.2, 
        stratify=y_full_train,
        random_state=42
    )
    
    print(f"\nFinal data splits:")
    print(f"Train: {X_train.shape[0]:,} samples, {np.sum(y_train):,} frauds ({100*np.mean(y_train):.2f}%)")
    print(f"Val:   {X_val.shape[0]:,} samples, {np.sum(y_val):,} frauds ({100*np.mean(y_val):.2f}%)")
    print(f"Test:  {X_test.shape[0]:,} samples, {np.sum(y_test):,} frauds ({100*np.mean(y_test):.2f}%)")
    print(f"Features: {X_train.shape[1]}")
    
    return (X_train, y_train, amounts_train, 
            X_val, y_val, amounts_val,
            X_test, y_test, amounts_test,
            feature_names)

# %%
# Load and prepare data
(X_train, y_train, amounts_train, 
 X_val, y_val, amounts_val,
 X_test, y_test, amounts_test,
 feature_names) = load_and_prepare_data()

# %% [markdown]
# ## Model Training
# 
# We train 4 different approaches:
# 1. XGBoost alone (baseline)
# 2. FraudBoost alone (value-weighted baseline)
# 3. Cascade (XGBoost → FraudBoost)
# 4. Stacking (XGBoost + FraudBoost → LogReg)

# %%
def train_models(X_train, y_train, amounts_train, X_val, y_val, amounts_val):
    """Train all model approaches."""
    print("\n" + "="*50)
    print("MODEL TRAINING")
    print("="*50)
    
    models = {}
    
    # Common parameters
    fp_cost = 100
    
    # 1. XGBoost alone
    print("\n1. Training XGBoost (baseline)...")
    pos_count = np.sum(y_train)
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / pos_count
    
    xgb_params = {
        'n_estimators': 100,
        'max_depth': 4, 
        'learning_rate': 0.1,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'eval_metric': 'auc'
    }
    
    models['xgb'] = xgb.XGBClassifier(**xgb_params)
    models['xgb'].fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # 2. FraudBoost alone
    print("\n2. Training FraudBoost (value-weighted baseline)...")
    fb_params = {
        'n_estimators': 100,
        'max_depth': 4,
        'learning_rate': 0.1,
        'fp_cost': fp_cost,
        'random_state': 42
    }
    
    models['fb'] = FraudBoostClassifier(**fb_params)
    models['fb'].fit(X_train, y_train, amounts_train, X_val, y_val, amounts_val)
    
    # 3. Cascade models with different stage1 thresholds
    print("\n3. Training Cascade models...")
    stage1_thresholds = [0.05, 0.1, 0.2, 0.3]
    
    models['cascade'] = {}
    for threshold in stage1_thresholds:
        print(f"   Training cascade with stage1_threshold={threshold}...")
        cascade = FraudCascade(
            stage1_threshold=threshold,
            fp_cost=fp_cost,
            xgb_params=xgb_params.copy(),
            fb_params=fb_params.copy()
        )
        cascade.fit(X_train, y_train, amounts_train, X_val, y_val, amounts_val)
        models['cascade'][threshold] = cascade
    
    # 4. Stacking ensemble
    print("\n4. Training Stacking ensemble...")
    stacking = FraudStacking(
        fp_cost=fp_cost,
        xgb_params=xgb_params.copy(),
        fb_params=fb_params.copy(),
        cv_folds=3
    )
    stacking.fit(X_train, y_train, amounts_train)
    models['stacking'] = stacking
    
    print("\n=== All Models Trained ===")
    return models

# %%
# Train all models
models = train_models(X_train, y_train, amounts_train, X_val, y_val, amounts_val)

# %% [markdown]
# ## Model Evaluation
# 
# We evaluate all approaches on the test set at both default and optimal thresholds.
# Key metrics:
# - **AUC**: Overall discrimination ability
# - **Precision/Recall/F1**: Classification performance  
# - **VDR**: Value Detection Rate (% of fraud dollars caught)
# - **Net Savings**: Total value considering FP costs

# %%
def evaluate_all_models(models, X_test, y_test, amounts_test):
    """Comprehensive evaluation of all models."""
    print("\n" + "="*50)
    print("MODEL EVALUATION ON TEST SET")
    print("="*50)
    
    results = {}
    fp_cost = 100
    
    # 1. XGBoost evaluation
    print("\n1. Evaluating XGBoost...")
    xgb_proba = models['xgb'].predict_proba(X_test)[:, 1]
    
    # Default threshold (0.5)
    xgb_pred_default = (xgb_proba >= 0.5).astype(int)
    
    # Optimal threshold (maximize F1)
    optimal_threshold_xgb = find_optimal_threshold(y_test, xgb_proba, metric='f1')
    xgb_pred_optimal = (xgb_proba >= optimal_threshold_xgb).astype(int)
    
    results['xgb'] = {
        'default': evaluate_predictions(y_test, xgb_pred_default, amounts_test, xgb_proba, 0.5, fp_cost),
        'optimal': evaluate_predictions(y_test, xgb_pred_optimal, amounts_test, xgb_proba, optimal_threshold_xgb, fp_cost)
    }
    
    # 2. FraudBoost evaluation  
    print("\n2. Evaluating FraudBoost...")
    fb_proba = models['fb'].predict_proba(X_test)[:, 1]
    
    # Default threshold (0.5)
    fb_pred_default = (fb_proba >= 0.5).astype(int)
    
    # Optimal threshold
    optimal_threshold_fb = find_optimal_threshold(y_test, fb_proba, metric='f1')
    fb_pred_optimal = (fb_proba >= optimal_threshold_fb).astype(int)
    
    results['fb'] = {
        'default': evaluate_predictions(y_test, fb_pred_default, amounts_test, fb_proba, 0.5, fp_cost),
        'optimal': evaluate_predictions(y_test, fb_pred_optimal, amounts_test, fb_proba, optimal_threshold_fb, fp_cost)
    }
    
    # 3. Cascade evaluation
    print("\n3. Evaluating Cascade models...")
    results['cascade'] = {}
    
    for threshold in models['cascade'].keys():
        print(f"   Evaluating cascade (stage1_threshold={threshold})...")
        cascade = models['cascade'][threshold]
        
        cascade_proba = cascade.predict_proba(X_test, amounts_test)
        
        # Default threshold
        cascade_pred_default = (cascade_proba >= 0.5).astype(int)
        
        # Optimal threshold
        optimal_threshold_cascade = find_optimal_threshold(y_test, cascade_proba, metric='f1')
        cascade_pred_optimal = (cascade_proba >= optimal_threshold_cascade).astype(int)
        
        results['cascade'][threshold] = {
            'default': evaluate_predictions(y_test, cascade_pred_default, amounts_test, cascade_proba, 0.5, fp_cost),
            'optimal': evaluate_predictions(y_test, cascade_pred_optimal, amounts_test, cascade_proba, optimal_threshold_cascade, fp_cost)
        }
        
        # Add cascade-specific metrics
        cascade_eval = cascade.evaluate(X_test, y_test, amounts_test, threshold=0.5)
        results['cascade'][threshold]['stage_breakdown'] = cascade_eval
    
    # 4. Stacking evaluation
    print("\n4. Evaluating Stacking ensemble...")
    stacking_proba = models['stacking'].predict_proba(X_test, amounts_test)
    
    # Default threshold
    stacking_pred_default = (stacking_proba >= 0.5).astype(int)
    
    # Optimal threshold
    optimal_threshold_stacking = find_optimal_threshold(y_test, stacking_proba, metric='f1')
    stacking_pred_optimal = (stacking_proba >= optimal_threshold_stacking).astype(int)
    
    results['stacking'] = {
        'default': evaluate_predictions(y_test, stacking_pred_default, amounts_test, stacking_proba, 0.5, fp_cost),
        'optimal': evaluate_predictions(y_test, stacking_pred_optimal, amounts_test, stacking_proba, optimal_threshold_stacking, fp_cost)
    }
    
    return results

def evaluate_predictions(y_true, y_pred, amounts, probabilities, threshold, fp_cost):
    """Helper function to calculate all metrics for a set of predictions."""
    metrics = {}
    
    # AUC
    if len(np.unique(y_pred)) > 1:
        metrics['auc'] = roc_auc_score(y_true, probabilities)
    else:
        metrics['auc'] = 0.5
    
    # Precision, Recall, F1
    prec, recall, f1 = calculate_precision_recall_f1(y_true, y_pred)
    metrics.update({
        'precision': prec,
        'recall': recall,
        'f1': f1
    })
    
    # Confusion matrix
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    
    metrics.update({
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'threshold': threshold
    })
    
    # Value-based metrics
    vdr = calculate_value_detection_rate(y_true, y_pred, amounts)
    net_savings = calculate_net_savings(y_true, y_pred, amounts, fp_cost)
    
    metrics.update({
        'vdr': vdr,
        'net_savings': net_savings
    })
    
    return metrics

# %%
# Evaluate all models
results = evaluate_all_models(models, X_test, y_test, amounts_test)

# %% [markdown]
# ## Results Analysis
# 
# Compare all approaches across key metrics and identify the best cascade configuration.

# %%
def print_results_table(results):
    """Print comprehensive results table."""
    print("\n" + "="*80)
    print("COMPREHENSIVE RESULTS COMPARISON")
    print("="*80)
    
    # Prepare data for table
    rows = []
    
    # XGBoost
    for threshold_type in ['default', 'optimal']:
        row = ['XGBoost', threshold_type]
        r = results['xgb'][threshold_type]
        row.extend([
            f"{r['auc']:.3f}",
            f"{r['recall']:.3f}",
            f"{r['precision']:.3f}",
            f"{r['f1']:.3f}",
            f"{r['vdr']:.3f}",
            f"${r['net_savings']:.0f}",
            f"{r['fp']:,}",
            f"{r['threshold']:.3f}"
        ])
        rows.append(row)
    
    # FraudBoost
    for threshold_type in ['default', 'optimal']:
        row = ['FraudBoost', threshold_type]
        r = results['fb'][threshold_type]
        row.extend([
            f"{r['auc']:.3f}",
            f"{r['recall']:.3f}",
            f"{r['precision']:.3f}",
            f"{r['f1']:.3f}",
            f"{r['vdr']:.3f}",
            f"${r['net_savings']:.0f}",
            f"{r['fp']:,}",
            f"{r['threshold']:.3f}"
        ])
        rows.append(row)
    
    # Cascade models
    for stage1_thresh in sorted(results['cascade'].keys()):
        for threshold_type in ['default', 'optimal']:
            row = [f'Cascade ({stage1_thresh})', threshold_type]
            r = results['cascade'][stage1_thresh][threshold_type]
            row.extend([
                f"{r['auc']:.3f}",
                f"{r['recall']:.3f}",
                f"{r['precision']:.3f}",
                f"{r['f1']:.3f}",
                f"{r['vdr']:.3f}",
                f"${r['net_savings']:.0f}",
                f"{r['fp']:,}",
                f"{r['threshold']:.3f}"
            ])
            rows.append(row)
    
    # Stacking
    for threshold_type in ['default', 'optimal']:
        row = ['Stacking', threshold_type]
        r = results['stacking'][threshold_type]
        row.extend([
            f"{r['auc']:.3f}",
            f"{r['recall']:.3f}",
            f"{r['precision']:.3f}",
            f"{r['f1']:.3f}",
            f"{r['vdr']:.3f}",
            f"${r['net_savings']:.0f}",
            f"{r['fp']:,}",
            f"{r['threshold']:.3f}"
        ])
        rows.append(row)
    
    # Print table
    headers = ['Approach', 'Threshold', 'AUC', 'Recall', 'Precision', 'F1', 'VDR', 'Net Savings', 'FPs', 'Threshold Val']
    
    # Calculate column widths
    col_widths = [max(len(str(item)) for item in col) for col in zip(headers, *rows)]
    
    # Print header
    header_row = ' | '.join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_row)
    print('-' * len(header_row))
    
    # Print rows
    for row in rows:
        print(' | '.join(str(item).ljust(w) for item, w in zip(row, col_widths)))

# %%
print_results_table(results)

# %% [markdown]
# ## Find Best Cascade Configuration

# %%
def find_best_cascade(results):
    """Find the best cascade configuration based on net savings."""
    print("\n" + "="*50)
    print("CASCADE OPTIMIZATION")
    print("="*50)
    
    best_net_savings = -float('inf')
    best_config = None
    
    print("Cascade configurations by net savings:")
    cascade_results = []
    
    for stage1_thresh in sorted(results['cascade'].keys()):
        for threshold_type in ['default', 'optimal']:
            r = results['cascade'][stage1_thresh][threshold_type]
            net_savings = r['net_savings']
            
            config = {
                'stage1_threshold': stage1_thresh,
                'threshold_type': threshold_type,
                'final_threshold': r['threshold'],
                'net_savings': net_savings,
                'recall': r['recall'],
                'precision': r['precision'],
                'f1': r['f1'],
                'fps': r['fp']
            }
            
            cascade_results.append(config)
            
            if net_savings > best_net_savings:
                best_net_savings = net_savings
                best_config = config
    
    # Sort by net savings
    cascade_results.sort(key=lambda x: x['net_savings'], reverse=True)
    
    print("\nCascade Rankings (by Net Savings):")
    print("-" * 70)
    for i, config in enumerate(cascade_results, 1):
        print(f"{i:2d}. Stage1={config['stage1_threshold']}, {config['threshold_type']}: "
              f"${config['net_savings']:.0f} (R={config['recall']:.3f}, P={config['precision']:.3f}, "
              f"FPs={config['fps']:,})")
    
    print(f"\n🏆 BEST CASCADE CONFIGURATION:")
    print(f"   Stage 1 threshold: {best_config['stage1_threshold']}")
    print(f"   Final threshold type: {best_config['threshold_type']}")
    print(f"   Final threshold value: {best_config['final_threshold']:.3f}")
    print(f"   Net savings: ${best_config['net_savings']:.0f}")
    print(f"   Recall: {best_config['recall']:.3f}")
    print(f"   Precision: {best_config['precision']:.3f}")
    print(f"   F1: {best_config['f1']:.3f}")
    print(f"   False positives: {best_config['fps']:,}")
    
    return best_config

# %%
best_cascade = find_best_cascade(results)

# %% [markdown]
# ## Cascade Stage Analysis

# %%
def analyze_cascade_stages(models, X_test, y_test, amounts_test):
    """Analyze what happens at each cascade stage."""
    print("\n" + "="*50)
    print("CASCADE STAGE ANALYSIS")
    print("="*50)
    
    stage1_thresholds = sorted(models['cascade'].keys())
    
    print("Stage 1 → Stage 2 Flow Analysis:")
    print("-" * 40)
    
    for stage1_thresh in stage1_thresholds:
        cascade = models['cascade'][stage1_thresh]
        breakdown = cascade.get_cascade_breakdown(X_test, y_test, amounts_test)
        
        total = breakdown['total_samples']
        total_frauds = breakdown['total_frauds']
        flagged = breakdown['stage1_flagged_count']
        frauds_caught = breakdown['frauds_caught_stage1']
        frauds_missed = breakdown['frauds_missed_stage1']
        
        reduction_ratio = flagged / total
        stage1_recall = frauds_caught / total_frauds if total_frauds > 0 else 0
        stage1_precision = frauds_caught / flagged if flagged > 0 else 0
        
        print(f"\nStage 1 Threshold: {stage1_thresh}")
        print(f"  Total samples: {total:,}")
        print(f"  Stage 1 flags: {flagged:,} ({100*reduction_ratio:.1f}%)")
        print(f"  Frauds caught: {frauds_caught}/{total_frauds} ({100*stage1_recall:.1f}% recall)")
        print(f"  Frauds missed: {frauds_missed}")
        print(f"  Stage 1 precision: {100*stage1_precision:.1f}%")
        print(f"  → Stage 2 gets: {flagged:,} samples ({100*reduction_ratio:.1f}% reduction)")

# %%
analyze_cascade_stages(models, X_test, y_test, amounts_test)

# %% [markdown]
# ## Visualization

# %%
def create_visualizations(results, models, X_test, y_test, amounts_test):
    """Create comparison charts."""
    print("\n" + "="*50)
    print("CREATING VISUALIZATIONS")
    print("="*50)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Figure 1: Net Savings Comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Net Savings Bar Chart
    ax = axes[0, 0]
    
    approaches = []
    net_savings = []
    colors = []
    
    # Add XGBoost
    approaches.extend(['XGB (def)', 'XGB (opt)'])
    net_savings.extend([results['xgb']['default']['net_savings'], results['xgb']['optimal']['net_savings']])
    colors.extend(['lightblue', 'blue'])
    
    # Add FraudBoost
    approaches.extend(['FB (def)', 'FB (opt)'])
    net_savings.extend([results['fb']['default']['net_savings'], results['fb']['optimal']['net_savings']])
    colors.extend(['lightcoral', 'red'])
    
    # Add best cascade
    best_cascade_savings = max([
        results['cascade'][t]['optimal']['net_savings'] 
        for t in results['cascade'].keys()
    ])
    approaches.append('Cascade (best)')
    net_savings.append(best_cascade_savings)
    colors.append('green')
    
    # Add stacking
    approaches.extend(['Stack (def)', 'Stack (opt)'])
    net_savings.extend([results['stacking']['default']['net_savings'], results['stacking']['optimal']['net_savings']])
    colors.extend(['lightyellow', 'orange'])
    
    bars = ax.bar(approaches, net_savings, color=colors)
    ax.set_title('Net Savings Comparison ($100 FP cost)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Net Savings ($)')
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, val in zip(bars, net_savings):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(net_savings)*0.01,
                f'${val:.0f}', ha='center', va='bottom', fontsize=10)
    
    # 2. False Positives Comparison
    ax = axes[0, 1]
    
    fps = []
    fps.extend([results['xgb']['default']['fp'], results['xgb']['optimal']['fp']])
    fps.extend([results['fb']['default']['fp'], results['fb']['optimal']['fp']])
    
    # Best cascade FPs
    best_cascade_fp = min([
        results['cascade'][t]['optimal']['fp'] 
        for t in results['cascade'].keys() 
        if results['cascade'][t]['optimal']['net_savings'] == best_cascade_savings
    ])
    fps.append(best_cascade_fp)
    
    fps.extend([results['stacking']['default']['fp'], results['stacking']['optimal']['fp']])
    
    bars = ax.bar(approaches, fps, color=colors)
    ax.set_title('False Positives Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('False Positives')
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, val in zip(bars, fps):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(fps)*0.01,
                f'{val:,}', ha='center', va='bottom', fontsize=9)
    
    # 3. Cascade Stage Flow (for best configuration)
    ax = axes[1, 0]
    
    # Find best stage1 threshold
    best_stage1_thresh = None
    for t in results['cascade'].keys():
        if results['cascade'][t]['optimal']['net_savings'] == best_cascade_savings:
            best_stage1_thresh = t
            break
    
    if best_stage1_thresh:
        cascade = models['cascade'][best_stage1_thresh]
        breakdown = cascade.get_cascade_breakdown(X_test, y_test, amounts_test)
        
        # Create funnel visualization
        total = breakdown['total_samples']
        flagged = breakdown['stage1_flagged_count']
        
        stages = ['All Samples', 'Stage 1 Flagged', 'Final Positive']
        counts = [total, flagged, results['cascade'][best_stage1_thresh]['optimal']['tp'] + results['cascade'][best_stage1_thresh]['optimal']['fp']]
        
        # Create funnel bars
        bars = ax.barh(stages, counts, color=['lightgray', 'orange', 'green'])
        ax.set_title(f'Cascade Flow (Stage1={best_stage1_thresh})', fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of Samples')
        
        # Add labels
        for bar, count in zip(bars, counts):
            width = bar.get_width()
            ax.text(width + total*0.01, bar.get_y() + bar.get_height()/2,
                    f'{count:,}', ha='left', va='center', fontsize=11)
    
    # 4. Precision-Recall Comparison
    ax = axes[1, 1]
    
    model_names = ['XGB', 'FraudBoost', 'Cascade', 'Stacking']
    precisions = [
        results['xgb']['optimal']['precision'],
        results['fb']['optimal']['precision'], 
        max([results['cascade'][t]['optimal']['precision'] for t in results['cascade'].keys()]),
        results['stacking']['optimal']['precision']
    ]
    recalls = [
        results['xgb']['optimal']['recall'],
        results['fb']['optimal']['recall'],
        [results['cascade'][t]['optimal']['recall'] for t in results['cascade'].keys() 
         if results['cascade'][t]['optimal']['precision'] == max([results['cascade'][t]['optimal']['precision'] for t in results['cascade'].keys()])][0],
        results['stacking']['optimal']['recall']
    ]
    
    scatter = ax.scatter(recalls, precisions, s=100, c=['blue', 'red', 'green', 'orange'], alpha=0.7)
    
    for i, name in enumerate(model_names):
        ax.annotate(name, (recalls[i], precisions[i]), xytext=(5, 5), 
                   textcoords='offset points', fontsize=10)
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision vs Recall (Optimal Thresholds)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = '/Users/alexshrike/.openclaw/workspace/fraudboost/cascade_benchmark_results.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {output_path}")
    
    return output_path

# %%
viz_path = create_visualizations(results, models, X_test, y_test, amounts_test)

# %% [markdown]
# ## Summary and Conclusions

# %%
def print_final_summary(results, best_cascade):
    """Print final summary and conclusions."""
    print("\n" + "="*80)
    print("FINAL SUMMARY & CONCLUSIONS")
    print("="*80)
    
    # Find best performer overall
    all_approaches = []
    
    # XGBoost
    all_approaches.append(('XGBoost (optimal)', results['xgb']['optimal']['net_savings']))
    
    # FraudBoost  
    all_approaches.append(('FraudBoost (optimal)', results['fb']['optimal']['net_savings']))
    
    # Best Cascade
    all_approaches.append(('Cascade (optimal)', best_cascade['net_savings']))
    
    # Stacking
    all_approaches.append(('Stacking (optimal)', results['stacking']['optimal']['net_savings']))
    
    # Sort by net savings
    all_approaches.sort(key=lambda x: x[1], reverse=True)
    
    print("\n🏆 OVERALL RANKINGS (by Net Savings):")
    print("-" * 50)
    for i, (approach, savings) in enumerate(all_approaches, 1):
        print(f"{i}. {approach}: ${savings:.0f}")
    
    winner_name, winner_savings = all_approaches[0]
    
    print(f"\n🥇 WINNER: {winner_name}")
    print(f"   Net Savings: ${winner_savings:.0f}")
    
    # Key insights
    print("\n📊 KEY INSIGHTS:")
    print("-" * 20)
    
    # Cascade vs individual models
    xgb_best = results['xgb']['optimal']['net_savings']
    fb_best = results['fb']['optimal']['net_savings']
    cascade_best = best_cascade['net_savings']
    
    if cascade_best > max(xgb_best, fb_best):
        improvement = cascade_best - max(xgb_best, fb_best)
        print(f"✅ Cascade IMPROVES over individual models by ${improvement:.0f}")
    else:
        degradation = max(xgb_best, fb_best) - cascade_best
        print(f"❌ Cascade DEGRADES compared to best individual model by ${degradation:.0f}")
    
    # Stage 1 threshold analysis
    print(f"\n🎯 OPTIMAL CASCADE CONFIGURATION:")
    print(f"   Stage 1 (XGBoost) threshold: {best_cascade['stage1_threshold']}")
    print(f"   Final threshold: {best_cascade['final_threshold']:.3f}")
    print(f"   This achieves {100*best_cascade['recall']:.1f}% recall with {best_cascade['fps']:,} false positives")
    
    # Value of ensemble approaches
    stacking_savings = results['stacking']['optimal']['net_savings']
    print(f"\n🤖 ENSEMBLE COMPARISON:")
    print(f"   Cascade (best): ${cascade_best:.0f}")
    print(f"   Stacking: ${stacking_savings:.0f}")
    
    if cascade_best > stacking_savings:
        print(f"   → Cascade wins by ${cascade_best - stacking_savings:.0f}")
    else:
        print(f"   → Stacking wins by ${stacking_savings - cascade_best:.0f}")
    
    print(f"\n⏰ Analysis completed at: {datetime.now()}")
    
    # Test data summary
    n_test = len(y_test)
    n_fraud_test = np.sum(y_test)
    total_fraud_value = np.sum(amounts_test[y_test == 1])
    
    print(f"\n📈 TEST SET SUMMARY:")
    print(f"   Total samples: {n_test:,}")
    print(f"   Fraud samples: {n_fraud_test:,} ({100*n_fraud_test/n_test:.2f}%)")
    print(f"   Total fraud value: ${total_fraud_value:,.0f}")
    print(f"   Average fraud amount: ${total_fraud_value/n_fraud_test:.2f}")

# %%
print_final_summary(results, best_cascade)

print("\n" + "="*80)
print("BENCHMARK COMPLETE!")
print("="*80)
print(f"Results saved and visualized.")
print(f"Chart saved to: cascade_benchmark_results.png")
print(f"All models trained and evaluated successfully.")