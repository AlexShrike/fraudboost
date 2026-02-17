#!/usr/bin/env python3
"""
Mini benchmark test to verify the Rust backend works with realistic fraud data.
"""
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import FraudBoost and metrics
from fraudboost import FraudBoostClassifier
from fraudboost.metrics import value_detection_rate, net_savings
from fraudboost._rust_core import has_rust_backend

def create_realistic_fraud_dataset(n_samples=10000, fraud_rate=0.002):
    """Create a realistic fraud dataset similar to credit card fraud."""
    np.random.seed(42)
    
    # Generate features that mimic credit card transactions
    n_features = 15
    X = np.random.randn(n_samples, n_features)
    
    # Create fraud labels
    n_fraud = int(n_samples * fraud_rate)
    y = np.zeros(n_samples)
    fraud_indices = np.random.choice(n_samples, n_fraud, replace=False)
    y[fraud_indices] = 1
    
    # Add signal to features for fraud detection
    # Fraud transactions have different patterns
    X[fraud_indices, 0] += 2.0   # Higher values in feature 0
    X[fraud_indices, 1] -= 1.5   # Lower values in feature 1  
    X[fraud_indices, 2:5] += np.random.randn(n_fraud, 3) * 0.8  # More variance
    
    # Generate transaction amounts
    # Normal transactions: mostly small amounts
    amounts = np.random.lognormal(mean=3, sigma=1.2, size=n_samples)
    
    # Fraud transactions: tend to be higher amounts
    amounts[fraud_indices] *= np.random.uniform(2, 8, n_fraud)
    
    # Clip amounts to reasonable range
    amounts = np.clip(amounts, 1, 50000)
    
    return X, y, amounts

def run_benchmark():
    print("Creating realistic fraud dataset...")
    X, y, amounts = create_realistic_fraud_dataset(n_samples=50000, fraud_rate=0.002)
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Fraud rate: {np.mean(y):.3%}")
    print(f"Mean amount (fraud): ${np.mean(amounts[y==1]):.2f}")
    print(f"Mean amount (normal): ${np.mean(amounts[y==0]):.2f}")
    
    # Split the data
    X_train, X_test, y_train, y_test, amounts_train, amounts_test = train_test_split(
        X, y, amounts, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Test fraud rate: {np.mean(y_test):.3%}")
    
    # Test parameters
    params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'min_samples_leaf': 10,
        'min_samples_split': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_lambda': 1.0,
        'fp_cost': 100.0,
        'random_state': 42,
        'verbose': 1
    }
    
    results = {}
    
    # Test Python backend
    print("\n" + "="*60)
    print("FraudBoost (Python Backend)")
    print("="*60)
    
    t0 = time.time()
    model_python = FraudBoostClassifier(backend='python', **params)
    model_python.fit(X_train, y_train, amounts=amounts_train)
    python_time = time.time() - t0
    
    # Make predictions
    probs_python = model_python.predict_proba(X_test)[:, 1]
    
    # Use threshold that catches ~90% of fraud value
    thresholds = np.linspace(0.1, 0.9, 50)
    best_threshold = 0.5
    best_vdr = 0
    
    for threshold in thresholds:
        preds = (probs_python > threshold).astype(int)
        vdr = value_detection_rate(y_test, preds, amounts_test)
        if vdr > best_vdr:
            best_vdr = vdr
            best_threshold = threshold
    
    preds_python = (probs_python > best_threshold).astype(int)
    
    # Calculate metrics
    tp = np.sum((preds_python == 1) & (y_test == 1))
    fp = np.sum((preds_python == 1) & (y_test == 0))
    fn = np.sum((preds_python == 0) & (y_test == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    vdr = value_detection_rate(y_test, preds_python, amounts_test)
    net_save = net_savings(y_test, preds_python, amounts_test, fp_cost=100)
    
    results['python'] = {
        'time': python_time,
        'precision': precision,
        'recall': recall,
        'vdr': vdr,
        'net_savings': net_save,
        'threshold': best_threshold,
        'n_trees': len(model_python.estimators_)
    }
    
    print(f"Training time: {python_time:.2f}s")
    print(f"Trees built: {len(model_python.estimators_)}")
    print(f"Best threshold: {best_threshold:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"VDR (Value Detection Rate): {vdr:.3f}")
    print(f"Net Savings: ${net_save:,.2f}")
    
    # Test Rust backend
    if has_rust_backend():
        print("\n" + "="*60)
        print("FraudBoost (Rust Backend)")
        print("="*60)
        
        t0 = time.time()
        model_rust = FraudBoostClassifier(backend='rust', **params)
        model_rust.fit(X_train, y_train, amounts=amounts_train)
        rust_time = time.time() - t0
        
        # Make predictions
        probs_rust = model_rust.predict_proba(X_test)[:, 1]
        preds_rust = (probs_rust > best_threshold).astype(int)  # Use same threshold
        
        # Calculate metrics
        tp = np.sum((preds_rust == 1) & (y_test == 1))
        fp = np.sum((preds_rust == 1) & (y_test == 0))
        fn = np.sum((preds_rust == 0) & (y_test == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        vdr = value_detection_rate(y_test, preds_rust, amounts_test)
        net_save = net_savings(y_test, preds_rust, amounts_test, fp_cost=100)
        
        results['rust'] = {
            'time': rust_time,
            'precision': precision,
            'recall': recall,
            'vdr': vdr,
            'net_savings': net_save,
            'threshold': best_threshold,
            'n_trees': model_rust._rust_booster.n_estimators_built
        }
        
        print(f"Training time: {rust_time:.2f}s")
        print(f"Trees built: {model_rust._rust_booster.n_estimators_built}")
        print(f"Threshold used: {best_threshold:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"VDR (Value Detection Rate): {vdr:.3f}")
        print(f"Net Savings: ${net_save:,.2f}")
        
        # Compare
        print("\n" + "="*60)
        print("Performance Comparison")
        print("="*60)
        
        speedup = results['python']['time'] / results['rust']['time']
        print(f"🚀 Speedup: {speedup:.1f}x faster with Rust!")
        
        vdr_diff = abs(results['python']['vdr'] - results['rust']['vdr'])
        savings_diff = abs(results['python']['net_savings'] - results['rust']['net_savings'])
        
        print(f"VDR difference: {vdr_diff:.4f}")
        print(f"Net savings difference: ${savings_diff:,.2f}")
        
        if speedup > 2.0 and vdr_diff < 0.01:
            print("✅ Rust backend delivers significant speedup with similar accuracy!")
        else:
            print("⚠️  Results may need tuning")
    
    else:
        print("\nRust backend not available")
    
    return results

if __name__ == "__main__":
    try:
        print("Mini FraudBoost Benchmark with Rust Backend")
        print("=" * 60)
        print(f"Rust backend available: {has_rust_backend()}")
        
        results = run_benchmark()
        
        print("\n✅ Benchmark completed successfully!")
        
        if 'rust' in results:
            speedup = results['python']['time'] / results['rust']['time']
            print(f"\n🎯 Final Result: {speedup:.1f}x speedup with Rust backend!")
            
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()