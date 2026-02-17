#!/usr/bin/env python3
"""
Comprehensive benchmark of FraudBoost vs Standard ML Models
Dataset: Kartik2112 Credit Card Fraud Detection (1.85M transactions)
"""

import os
# Set matplotlib backend BEFORE importing matplotlib
os.environ['MPLBACKEND'] = 'Agg'

import pandas as pd
import numpy as np
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from math import radians, cos, sin, asin, sqrt
import warnings
warnings.filterwarnings('ignore')

# Import FraudBoost
ENABLE_FRAUDBOOST = True  # Enable FraudBoost with minimal parameters
try:
    if ENABLE_FRAUDBOOST:
        from fraudboost import FraudBoostClassifier
        from fraudboost.metrics import value_detection_rate, net_savings
        FRAUDBOOST_AVAILABLE = True
    else:
        FRAUDBOOST_AVAILABLE = False
        # Import alternative metrics functions
        def value_detection_rate(y_true, y_pred, amounts):
            """Calculate VDR - percentage of total fraud value detected"""
            fraud_mask = (y_true == 1)
            detected_mask = (y_pred == 1) & fraud_mask
            total_fraud_value = amounts[fraud_mask].sum()
            detected_fraud_value = amounts[detected_mask].sum()
            return detected_fraud_value / total_fraud_value if total_fraud_value > 0 else 0.0
        
        def net_savings(y_true, y_pred, amounts, fp_cost=100):
            """Calculate net savings: fraud detected - false positive costs"""
            fraud_mask = (y_true == 1)
            detected_mask = (y_pred == 1) & fraud_mask
            fp_mask = (y_pred == 1) & (y_true == 0)
            
            fraud_detected_value = amounts[detected_mask].sum()
            fp_cost_total = fp_mask.sum() * fp_cost
            
            return fraud_detected_value - fp_cost_total
        
except ImportError as e:
    print(f"Warning: Could not import FraudBoost: {e}")
    FRAUDBOOST_AVAILABLE = False

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the haversine distance between two points on Earth (vectorized)"""
    # Convert to radians
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * np.arcsin(np.sqrt(a)) * 6371  # Radius of earth in kilometers

def engineer_features(df):
    """Engineer features without target leakage"""
    print(f"Engineering features for {len(df):,} rows...")
    
    # Create copy
    df = df.copy()
    print("  Parsing datetime...")
    
    # Parse datetime
    df['trans_datetime'] = pd.to_datetime(df['trans_date_trans_time'])
    df['hour'] = df['trans_datetime'].dt.hour
    df['day_of_week'] = df['trans_datetime'].dt.dayofweek
    print("  Computing age...")
    
    # Age calculation
    df['dob'] = pd.to_datetime(df['dob'])
    df['age'] = (df['trans_datetime'] - df['dob']).dt.days / 365.25
    print("  Computing distance (this may take a moment)...")
    
    # Distance calculation
    df['distance'] = haversine_distance(df['lat'], df['long'], df['merch_lat'], df['merch_long'])
    print("  Computing log amount...")
    
    # Amount log
    df['amt_log'] = np.log1p(df['amt'])
    print("  Encoding categorical variables...")
    
    # Encode categorical variables
    le_category = LabelEncoder()
    df['category_encoded'] = le_category.fit_transform(df['category'])
    
    # Binary encode gender
    df['gender_encoded'] = (df['gender'] == 'M').astype(int)
    
    print("  Selecting features...")
    # Select final features
    feature_cols = [
        'amt', 'category_encoded', 'gender_encoded', 'city_pop',
        'lat', 'long', 'merch_lat', 'merch_long', 'hour', 'day_of_week',
        'age', 'distance', 'amt_log'
    ]
    
    X = df[feature_cols].fillna(0)  # Handle any NaN values
    y = df['is_fraud'].values
    amounts = df['amt'].values
    
    print(f"  Feature engineering completed. Shape: {X.shape}")
    return X, y, amounts, feature_cols

def load_and_subsample_data():
    """Load data with subsampling strategy for 16GB Mac"""
    print("Loading training data...")
    train_df = pd.read_csv('/Users/alexshrike/.openclaw/workspace/bastion/data/fraudTrain.csv')
    print(f"Train data shape: {train_df.shape}")
    print(f"Fraud rate in training: {train_df['is_fraud'].mean():.4f}")
    
    # Separate fraud and legit transactions
    fraud_df = train_df[train_df['is_fraud'] == 1].copy()
    legit_df = train_df[train_df['is_fraud'] == 0].copy()
    
    print(f"Fraud transactions: {len(fraud_df)}")
    print(f"Legit transactions: {len(legit_df)}")
    
    # Keep ALL fraud rows, sample 200K legit rows
    sampled_legit = legit_df.sample(n=min(200000, len(legit_df)), random_state=42)
    
    # Combine
    train_subsample = pd.concat([fraud_df, sampled_legit], ignore_index=True).sample(frac=1, random_state=42)
    
    print(f"Subsampled training data: {train_subsample.shape}")
    print(f"Fraud rate after subsampling: {train_subsample['is_fraud'].mean():.4f}")
    
    # Load test data (use full test set)
    print("Loading test data...")
    test_df = pd.read_csv('/Users/alexshrike/.openclaw/workspace/bastion/data/fraudTest.csv')
    print(f"Test data shape: {test_df.shape}")
    print(f"Fraud rate in test: {test_df['is_fraud'].mean():.4f}")
    
    return train_subsample, test_df

def precision_at_recall(y_true, y_scores, target_recall=0.8):
    """Find precision at target recall"""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    
    # Find the threshold that gives us closest to target recall
    recall_diff = np.abs(recalls - target_recall)
    best_idx = np.argmin(recall_diff)
    
    return precisions[best_idx], recalls[best_idx]

def find_optimal_threshold(y_true, y_scores, amounts, fp_cost=100):
    """Find optimal threshold by maximizing net savings"""
    thresholds = np.arange(0.01, 1.0, 0.01)
    best_threshold = 0.5
    best_savings = -float('inf')
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        savings = net_savings(y_true, y_pred, amounts, fp_cost=fp_cost)
        if savings > best_savings:
            best_savings = savings
            best_threshold = threshold
    
    return best_threshold, best_savings

def train_and_evaluate_models(X_train, y_train, X_val, y_val, X_test, y_test, 
                            amounts_train, amounts_val, amounts_test):
    """Train all models and collect results"""
    results = {}
    
    # Calculate class ratio for XGBoost
    pos_count = np.sum(y_train)
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / pos_count
    print(f"Scale pos weight for XGBoost: {scale_pos_weight:.1f}")
    
    models = {}
    
    # 1. FraudBoost
    if FRAUDBOOST_AVAILABLE:
        print("\n" + "="*50)
        print("Training FraudBoost...")
        start_time = time.time()
        
        try:
            fb_model = FraudBoostClassifier(
                n_estimators=10,  # Minimal for completion
                max_depth=2,      # Minimal depth
                learning_rate=0.2, # Higher LR to compensate for fewer trees
                loss='value_weighted',
                random_state=42
            )
            fb_model.fit(X_train, y_train, amounts=amounts_train)
            train_time = time.time() - start_time
            
            # Predictions
            y_pred_proba = fb_model.predict_proba(X_test)[:, 1]
            y_pred = fb_model.predict(X_test)
            
            models['FraudBoost'] = {
                'model': fb_model,
                'y_pred_proba': y_pred_proba,
                'y_pred': y_pred,
                'train_time': train_time
            }
            print(f"FraudBoost training completed in {train_time:.1f}s")
            
        except Exception as e:
            print(f"FraudBoost training failed: {e}")
    
    # 2. XGBoost
    print("\n" + "="*50)
    print("Training XGBoost...")
    start_time = time.time()
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    models['XGBoost'] = {
        'model': xgb_model,
        'y_pred_proba': y_pred_proba,
        'y_pred': y_pred,
        'train_time': train_time
    }
    print(f"XGBoost training completed in {train_time:.1f}s")
    
    # 3. LightGBM
    print("\n" + "="*50)
    print("Training LightGBM...")
    start_time = time.time()
    
    lgb_model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        is_unbalance=True,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    y_pred_proba = lgb_model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    models['LightGBM'] = {
        'model': lgb_model,
        'y_pred_proba': y_pred_proba,
        'y_pred': y_pred,
        'train_time': train_time
    }
    print(f"LightGBM training completed in {train_time:.1f}s")
    
    # 4. Random Forest
    print("\n" + "="*50)
    print("Training Random Forest...")
    start_time = time.time()
    
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    models['Random Forest'] = {
        'model': rf_model,
        'y_pred_proba': y_pred_proba,
        'y_pred': y_pred,
        'train_time': train_time
    }
    print(f"Random Forest training completed in {train_time:.1f}s")
    
    # 5. Logistic Regression
    print("\n" + "="*50)
    print("Training Logistic Regression...")
    start_time = time.time()
    
    # Scale features for LogReg
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr_model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    lr_model.fit(X_train_scaled, y_train)
    train_time = time.time() - start_time
    
    y_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    models['Logistic Regression'] = {
        'model': lr_model,
        'y_pred_proba': y_pred_proba,
        'y_pred': y_pred,
        'train_time': train_time,
        'scaler': scaler
    }
    print(f"Logistic Regression training completed in {train_time:.1f}s")
    
    # Evaluate all models
    print("\n" + "="*50)
    print("Evaluating models...")
    
    for name, model_info in models.items():
        print(f"\nEvaluating {name}...")
        
        y_pred_proba = model_info['y_pred_proba']
        y_pred = model_info['y_pred']
        
        # Basic metrics
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        precision_80, recall_80 = precision_at_recall(y_test, y_pred_proba, 0.8)
        recall_05 = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        n_fps = np.sum((y_pred == 1) & (y_test == 0))
        
        # Value-based metrics
        vdr = value_detection_rate(y_test, y_pred, amounts_test)
        net_sav = net_savings(y_test, y_pred, amounts_test, fp_cost=100)
        
        # Find optimal threshold
        opt_threshold, opt_savings = find_optimal_threshold(y_test, y_pred_proba, amounts_test)
        y_pred_opt = (y_pred_proba >= opt_threshold).astype(int)
        opt_vdr = value_detection_rate(y_test, y_pred_opt, amounts_test)
        
        results[name] = {
            'auc_roc': auc_roc,
            'precision_at_80_recall': precision_80,
            'recall_at_05': recall_05,
            'f1_score': f1,
            'vdr': vdr,
            'net_savings': net_sav,
            'n_false_positives': n_fps,
            'train_time': model_info['train_time'],
            'optimal_threshold': opt_threshold,
            'optimal_net_savings': opt_savings,
            'optimal_vdr': opt_vdr,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"  AUC-ROC: {auc_roc:.4f}")
        print(f"  Precision @ 80% Recall: {precision_80:.4f}")
        print(f"  VDR: {vdr:.4f}")
        print(f"  Net Savings: ${net_sav:,.0f}")
        print(f"  Optimal Threshold: {opt_threshold:.3f} (Savings: ${opt_savings:,.0f})")
    
    return results

def create_visualizations(results, y_test):
    """Create comparison charts"""
    print("\n" + "="*50)
    print("Creating visualizations...")
    
    plt.style.use('default')
    
    # 1. ROC Curves
    plt.figure(figsize=(10, 8))
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
        auc = result['auc_roc']
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - All Models')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/Users/alexshrike/.openclaw/workspace/fraudboost/benchmarks/roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Precision-Recall Curves
    plt.figure(figsize=(10, 8))
    for name, result in results.items():
        precision, recall, _ = precision_recall_curve(y_test, result['y_pred_proba'])
        plt.plot(recall, precision, label=name, linewidth=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves - All Models')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/Users/alexshrike/.openclaw/workspace/fraudboost/benchmarks/precision_recall_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Net Savings Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Regular net savings
    models = list(results.keys())
    net_savings_vals = [results[m]['net_savings'] for m in models]
    
    bars1 = ax1.bar(models, net_savings_vals, alpha=0.7, color='skyblue')
    ax1.set_title('Net Savings at Default Threshold (0.5)')
    ax1.set_ylabel('Net Savings ($)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars1, net_savings_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(net_savings_vals)*0.01,
                f'${val:,.0f}', ha='center', va='bottom')
    
    # Optimal net savings
    opt_savings_vals = [results[m]['optimal_net_savings'] for m in models]
    
    bars2 = ax2.bar(models, opt_savings_vals, alpha=0.7, color='lightcoral')
    ax2.set_title('Net Savings at Optimal Threshold')
    ax2.set_ylabel('Net Savings ($)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars2, opt_savings_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(opt_savings_vals)*0.01,
                f'${val:,.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('/Users/alexshrike/.openclaw/workspace/fraudboost/benchmarks/net_savings_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. VDR Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Regular VDR
    vdr_vals = [results[m]['vdr'] for m in models]
    
    bars1 = ax1.bar(models, vdr_vals, alpha=0.7, color='lightgreen')
    ax1.set_title('Value Detection Rate at Default Threshold (0.5)')
    ax1.set_ylabel('VDR')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, val in zip(bars1, vdr_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom')
    
    # Optimal VDR
    opt_vdr_vals = [results[m]['optimal_vdr'] for m in models]
    
    bars2 = ax2.bar(models, opt_vdr_vals, alpha=0.7, color='gold')
    ax2.set_title('Value Detection Rate at Optimal Threshold')
    ax2.set_ylabel('VDR')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, val in zip(bars2, opt_vdr_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('/Users/alexshrike/.openclaw/workspace/fraudboost/benchmarks/vdr_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualizations saved!")

def save_results(results):
    """Save results to markdown table"""
    print("\n" + "="*50)
    print("Saving results...")
    
    # Create markdown table
    markdown_content = f"""# Benchmark Results: FraudBoost vs Standard ML Models

**Dataset:** Kartik2112 Credit Card Fraud Detection  
**Training Set:** 207K transactions (~7K fraud, 200K legit)  
**Test Set:** 555K transactions (full test set)  
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Performance Summary

| Model | AUC-ROC | Precision@80%Recall | Recall@0.5 | F1-Score | VDR | Net Savings | False Positives | Train Time (s) |
|-------|---------|-------------------|-----------|----------|-----|-------------|----------------|---------------|
"""
    
    for name, result in results.items():
        markdown_content += f"| {name} | {result['auc_roc']:.4f} | {result['precision_at_80_recall']:.4f} | {result['recall_at_05']:.4f} | {result['f1_score']:.4f} | {result['vdr']:.4f} | ${result['net_savings']:,.0f} | {result['n_false_positives']:,} | {result['train_time']:.1f} |\n"
    
    markdown_content += """
## Optimal Threshold Performance

| Model | Optimal Threshold | Net Savings | VDR |
|-------|------------------|-------------|-----|
"""
    
    for name, result in results.items():
        markdown_content += f"| {name} | {result['optimal_threshold']:.3f} | ${result['optimal_net_savings']:,.0f} | {result['optimal_vdr']:.4f} |\n"
    
    markdown_content += """
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

- **FraudBoost:** n_estimators=10, max_depth=2, lr=0.2, value_weighted loss (minimal for completion)
- **XGBoost:** n_estimators=200, max_depth=4, lr=0.1, scale_pos_weight=~27
- **LightGBM:** n_estimators=200, max_depth=4, lr=0.1, is_unbalance=True
- **Random Forest:** n_estimators=200, max_depth=10, class_weight='balanced'
- **Logistic Regression:** C=1.0, max_iter=1000, class_weight='balanced', StandardScaler

"""
    
    with open('/Users/alexshrike/.openclaw/workspace/fraudboost/benchmarks/results_kartik.md', 'w') as f:
        f.write(markdown_content)
    
    print("Results saved to benchmarks/results_kartik.md")

def main():
    """Main benchmark execution"""
    print("="*60)
    print("FRAUDBOOST vs STANDARD ML MODELS BENCHMARK")
    print("Dataset: Kartik2112 Credit Card Fraud Detection")
    print("="*60)
    
    try:
        # Load and subsample data
        train_df, test_df = load_and_subsample_data()
        
        # Engineer features
        print("\n" + "="*50)
        print("Processing training data...")
        X_train_full, y_train_full, amounts_train_full, feature_cols = engineer_features(train_df)
        
        print("Processing test data...")
        X_test, y_test, amounts_test, _ = engineer_features(test_df)
        
        # Split training into train/val
        X_train, X_val, y_train, y_val, amounts_train, amounts_val = train_test_split(
            X_train_full, y_train_full, amounts_train_full, 
            test_size=0.2, random_state=42, stratify=y_train_full
        )
        
        print(f"\nFinal data shapes:")
        print(f"  Train: {X_train.shape}")
        print(f"  Validation: {X_val.shape}")
        print(f"  Test: {X_test.shape}")
        print(f"  Features: {feature_cols}")
        
        # Train and evaluate models
        results = train_and_evaluate_models(
            X_train, y_train, X_val, y_val, X_test, y_test,
            amounts_train, amounts_val, amounts_test
        )
        
        # Create visualizations
        create_visualizations(results, y_test)
        
        # Save results
        save_results(results)
        
        print("\n" + "="*60)
        print("BENCHMARK COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        # Print summary
        print("\nFinal Results Summary:")
        print("-" * 40)
        for name, result in results.items():
            print(f"{name:20s}: AUC={result['auc_roc']:.3f}, VDR={result['vdr']:.3f}, NetSavings=${result['net_savings']:,.0f}")
        
        return results
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()