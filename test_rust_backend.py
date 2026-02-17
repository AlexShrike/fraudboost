#!/usr/bin/env python3
"""
Test the Rust backend implementation.
"""
import sys
import numpy as np
import time
from fraudboost import FraudBoostClassifier
from fraudboost._rust_core import has_rust_backend

def test_basic_functionality():
    """Test basic functionality of both backends."""
    print(f"Rust backend available: {has_rust_backend()}")
    
    # Create test data
    np.random.seed(42)
    n_samples = 5000
    n_features = 10
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate labels with some signal
    y = np.zeros(n_samples)
    fraud_indices = np.random.choice(n_samples, size=100, replace=False)
    y[fraud_indices] = 1
    
    # Add signal to make the problem learnable
    X[fraud_indices, 0] += 2.0  # Positive signal for fraud
    X[fraud_indices, 1] -= 1.5  # Negative signal for fraud
    
    # Generate amounts (higher for fraud)
    amounts = np.random.uniform(10, 500, n_samples)
    amounts[fraud_indices] *= 5  # Higher amounts for fraud
    
    print(f"\nDataset: {n_samples} samples, {n_features} features")
    print(f"Fraud rate: {np.mean(y):.3f}")
    print(f"Mean amount (fraud): ${np.mean(amounts[y==1]):.2f}")
    print(f"Mean amount (normal): ${np.mean(amounts[y==0]):.2f}")
    
    # Test parameters
    test_params = {
        'n_estimators': 50,
        'max_depth': 4,
        'learning_rate': 0.1,
        'min_samples_leaf': 10,
        'min_samples_split': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_lambda': 1.0,
        'verbose': 1,
        'random_state': 42
    }
    
    results = {}
    
    # Test Python backend
    print("\n" + "="*50)
    print("Testing Python backend")
    print("="*50)
    
    t0 = time.time()
    model_python = FraudBoostClassifier(backend='python', **test_params)
    model_python.fit(X, y, amounts=amounts)
    python_time = time.time() - t0
    
    probs_python = model_python.predict_proba(X)[:, 1]
    preds_python = (probs_python > 0.5).astype(int)
    
    tp_python = ((preds_python == 1) & (y == 1)).sum()
    fp_python = ((preds_python == 1) & (y == 0)).sum()
    fn_python = ((preds_python == 0) & (y == 1)).sum()
    
    precision_python = tp_python / (tp_python + fp_python) if (tp_python + fp_python) > 0 else 0
    recall_python = tp_python / (tp_python + fn_python) if (tp_python + fn_python) > 0 else 0
    
    results['python'] = {
        'time': python_time,
        'tp': tp_python,
        'fp': fp_python, 
        'fn': fn_python,
        'precision': precision_python,
        'recall': recall_python,
        'base_score': model_python.base_score_,
        'n_trees': len(model_python.estimators_)
    }
    
    print(f"Training time: {python_time:.3f}s")
    print(f"Base score: {model_python.base_score_:.4f}")
    print(f"Trees built: {len(model_python.estimators_)}")
    print(f"TP: {tp_python}, FP: {fp_python}, FN: {fn_python}")
    print(f"Precision: {precision_python:.3f}, Recall: {recall_python:.3f}")
    
    # Test Rust backend (if available)
    if has_rust_backend():
        print("\n" + "="*50)
        print("Testing Rust backend")
        print("="*50)
        
        t0 = time.time()
        model_rust = FraudBoostClassifier(backend='rust', **test_params)
        model_rust.fit(X, y, amounts=amounts)
        rust_time = time.time() - t0
        
        probs_rust = model_rust.predict_proba(X)[:, 1]
        preds_rust = (probs_rust > 0.5).astype(int)
        
        tp_rust = ((preds_rust == 1) & (y == 1)).sum()
        fp_rust = ((preds_rust == 1) & (y == 0)).sum()
        fn_rust = ((preds_rust == 0) & (y == 1)).sum()
        
        precision_rust = tp_rust / (tp_rust + fp_rust) if (tp_rust + fp_rust) > 0 else 0
        recall_rust = tp_rust / (tp_rust + fn_rust) if (tp_rust + fn_rust) > 0 else 0
        
        results['rust'] = {
            'time': rust_time,
            'tp': tp_rust,
            'fp': fp_rust,
            'fn': fn_rust,
            'precision': precision_rust,
            'recall': recall_rust,
            'base_score': model_rust.base_score_,
            'n_trees': model_rust._rust_booster.n_estimators_built
        }
        
        print(f"Training time: {rust_time:.3f}s")
        print(f"Base score: {model_rust.base_score_:.4f}")
        print(f"Trees built: {model_rust._rust_booster.n_estimators_built}")
        print(f"TP: {tp_rust}, FP: {fp_rust}, FN: {fn_rust}")
        print(f"Precision: {precision_rust:.3f}, Recall: {recall_rust:.3f}")
        
        # Compare results
        print("\n" + "="*50)
        print("Comparison")
        print("="*50)
        
        speedup = python_time / rust_time if rust_time > 0 else float('inf')
        print(f"Speedup: {speedup:.2f}x")
        
        base_score_diff = abs(results['python']['base_score'] - results['rust']['base_score'])
        print(f"Base score difference: {base_score_diff:.6f}")
        
        prob_diff = np.mean(np.abs(probs_python - probs_rust))
        print(f"Mean probability difference: {prob_diff:.6f}")
        
        print("\nMetrics comparison:")
        for metric in ['precision', 'recall']:
            py_val = results['python'][metric]
            rust_val = results['rust'][metric] 
            diff = abs(py_val - rust_val)
            print(f"  {metric.capitalize()}: Python={py_val:.3f}, Rust={rust_val:.3f}, Diff={diff:.3f}")
    
    else:
        print("\nRust backend not available")
    
    # Test auto backend
    print("\n" + "="*50)
    print("Testing 'auto' backend")
    print("="*50)
    
    model_auto = FraudBoostClassifier(backend='auto', **test_params)
    model_auto.fit(X, y, amounts=amounts)
    backend_used = "Rust" if model_auto._use_rust else "Python"
    print(f"Auto backend chose: {backend_used}")
    
    return results

if __name__ == "__main__":
    try:
        results = test_basic_functionality()
        print("\n✅ Test completed successfully!")
        
        if 'rust' in results:
            speedup = results['python']['time'] / results['rust']['time']
            print(f"🚀 Rust backend is {speedup:.1f}x faster!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)