#!/usr/bin/env python3
"""
Demonstration of FraudBoost Rust backend performance vs Python backend.
This shows the speedup achieved by rewriting the core tree building in Rust.
"""
import numpy as np
import time
from fraudboost import FraudBoostClassifier
from fraudboost._rust_core import has_rust_backend

def create_demo_dataset(n_samples=25000, n_features=15, fraud_rate=0.002):
    """Create a realistic fraud detection dataset."""
    np.random.seed(42)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Create fraud labels
    n_fraud = int(n_samples * fraud_rate)
    y = np.zeros(n_samples)
    fraud_indices = np.random.choice(n_samples, n_fraud, replace=False)
    y[fraud_indices] = 1
    
    # Add signal for fraud detection
    X[fraud_indices, 0] += 2.5  # Strong signal in first feature
    X[fraud_indices, 1] -= 1.8  # Opposite signal in second feature
    X[fraud_indices, 2:5] += np.random.randn(n_fraud, 3) * 0.5  # Subtle signals
    
    # Generate realistic amounts
    amounts = np.random.lognormal(mean=3.5, sigma=1.0, size=n_samples)
    amounts[fraud_indices] *= np.random.uniform(3, 10, n_fraud)  # Higher fraud amounts
    amounts = np.clip(amounts, 10, 25000)
    
    return X, y, amounts

def benchmark_backends():
    """Benchmark both Python and Rust backends."""
    
    print("🚀 FraudBoost Rust Backend Performance Demo")
    print("=" * 60)
    print(f"Rust backend available: {has_rust_backend()}")
    
    if not has_rust_backend():
        print("❌ Rust backend not available. Please build with:")
        print("   maturin develop --release -m rust_core/Cargo.toml")
        return
    
    # Create dataset
    print("\n📊 Creating fraud detection dataset...")
    X, y, amounts = create_demo_dataset()
    
    print(f"Dataset size: {X.shape[0]:,} samples, {X.shape[1]} features")
    print(f"Fraud rate: {np.mean(y):.3%} ({np.sum(y)} fraud cases)")
    print(f"Mean amount (fraud): ${np.mean(amounts[y==1]):,.2f}")
    print(f"Mean amount (normal): ${np.mean(amounts[y==0]):,.2f}")
    
    # Test parameters
    params = {
        'n_estimators': 50,  # Reasonable size for demo
        'max_depth': 6,
        'learning_rate': 0.1,
        'min_samples_leaf': 10,
        'min_samples_split': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_lambda': 1.0,
        'fp_cost': 100.0,
        'random_state': 42,
        'verbose': 0  # Keep output clean
    }
    
    # Benchmark Python backend
    print(f"\n⚙️  Training with Python backend...")
    model_python = FraudBoostClassifier(backend='python', **params)
    
    t0 = time.time()
    model_python.fit(X, y, amounts=amounts)
    python_time = time.time() - t0
    
    # Get predictions and basic metrics
    probs_py = model_python.predict_proba(X)[:, 1]
    preds_py = (probs_py > 0.5).astype(int)
    
    tp_py = np.sum((preds_py == 1) & (y == 1))
    fp_py = np.sum((preds_py == 1) & (y == 0))
    fn_py = np.sum((preds_py == 0) & (y == 1))
    
    precision_py = tp_py / (tp_py + fp_py) if (tp_py + fp_py) > 0 else 0
    recall_py = tp_py / (tp_py + fn_py) if (tp_py + fn_py) > 0 else 0
    
    print(f"   Time: {python_time:.2f}s")
    print(f"   Trees: {len(model_python.estimators_)}")
    print(f"   Precision: {precision_py:.3f}, Recall: {recall_py:.3f}")
    
    # Benchmark Rust backend
    print(f"\n🦀 Training with Rust backend...")
    model_rust = FraudBoostClassifier(backend='rust', **params)
    
    t0 = time.time()
    model_rust.fit(X, y, amounts=amounts)
    rust_time = time.time() - t0
    
    # Get predictions and basic metrics
    probs_rust = model_rust.predict_proba(X)[:, 1]
    preds_rust = (probs_rust > 0.5).astype(int)
    
    tp_rust = np.sum((preds_rust == 1) & (y == 1))
    fp_rust = np.sum((preds_rust == 1) & (y == 0))
    fn_rust = np.sum((preds_rust == 0) & (y == 1))
    
    precision_rust = tp_rust / (tp_rust + fp_rust) if (tp_rust + fp_rust) > 0 else 0
    recall_rust = tp_rust / (tp_rust + fn_rust) if (tp_rust + fn_rust) > 0 else 0
    
    print(f"   Time: {rust_time:.2f}s")
    print(f"   Trees: {model_rust._rust_booster.n_estimators_built}")
    print(f"   Precision: {precision_rust:.3f}, Recall: {recall_rust:.3f}")
    
    # Performance comparison
    print("\n📈 Performance Comparison:")
    print("=" * 40)
    
    speedup = python_time / rust_time if rust_time > 0 else float('inf')
    print(f"🚀 Speedup: {speedup:.1f}x faster with Rust!")
    print(f"📊 Python: {python_time:.2f}s → Rust: {rust_time:.2f}s")
    
    # Accuracy comparison
    prob_diff = np.mean(np.abs(probs_py - probs_rust))
    base_diff = abs(model_python.base_score_ - model_rust.base_score_)
    
    print(f"\n🎯 Accuracy Comparison:")
    print(f"   Mean probability difference: {prob_diff:.6f}")
    print(f"   Base score difference: {base_diff:.6f}")
    print(f"   Precision difference: {abs(precision_py - precision_rust):.4f}")
    print(f"   Recall difference: {abs(recall_py - recall_rust):.4f}")
    
    if prob_diff < 0.01 and speedup > 2.0:
        print("\n✅ SUCCESS: Rust backend delivers significant speedup with nearly identical results!")
    elif speedup > 2.0:
        print(f"\n⚠️  Speedup achieved but with some accuracy difference (may need tuning)")
    else:
        print(f"\n❓ Unexpected results - investigate further")
    
    # Memory usage estimation
    python_trees = len(model_python.estimators_)
    rust_trees = model_rust._rust_booster.n_estimators_built
    
    print(f"\n💾 Memory & Model Info:")
    print(f"   Python trees created: {python_trees}")
    print(f"   Rust trees created: {rust_trees}")
    print(f"   Feature importance available: {model_python.feature_importances_ is not None}")
    print(f"   Rust feature importance available: {model_rust.feature_importances_ is not None}")
    
    return {
        'python_time': python_time,
        'rust_time': rust_time,
        'speedup': speedup,
        'accuracy_similar': prob_diff < 0.01
    }

def demonstrate_api_compatibility():
    """Show that the API is identical for both backends."""
    print(f"\n🔄 API Compatibility Demo")
    print("=" * 30)
    
    # Small dataset for quick demo
    np.random.seed(123)
    X_small = np.random.randn(1000, 5)
    y_small = np.zeros(1000)
    y_small[:20] = 1  # 2% fraud
    amounts_small = np.random.uniform(50, 1000, 1000)
    
    print("Testing identical API usage...")
    
    # Both backends use identical API
    for backend_name in ['python', 'rust']:
        if backend_name == 'rust' and not has_rust_backend():
            continue
            
        print(f"\n{backend_name.capitalize()} backend:")
        
        model = FraudBoostClassifier(
            n_estimators=10,
            max_depth=3,
            learning_rate=0.2,
            backend=backend_name,
            random_state=456,
            verbose=0
        )
        
        # Fit
        model.fit(X_small, y_small, amounts=amounts_small)
        
        # Predict
        probs = model.predict_proba(X_small)
        preds = model.predict(X_small)
        
        print(f"   Model fitted: ✓")
        print(f"   Predictions shape: {probs.shape}")
        print(f"   Mean fraud probability: {np.mean(probs[:, 1]):.4f}")
        print(f"   Predicted fraud cases: {np.sum(preds)}")
    
    print("\n✅ API is identical across backends!")

if __name__ == "__main__":
    try:
        # Run the main benchmark
        results = benchmark_backends()
        
        # Show API compatibility
        demonstrate_api_compatibility()
        
        print(f"\n🎉 Demo completed successfully!")
        
        if results and results['speedup'] > 1.0:
            print(f"🏆 Key Achievement: {results['speedup']:.1f}x speedup with Rust backend!")
            
            if results['accuracy_similar']:
                print("🎯 Accuracy is nearly identical between backends.")
            
            print("\n📝 Next Steps:")
            print("   1. Run full benchmarks with: python benchmarks/benchmark_kartik.py")  
            print("   2. Try different hyperparameters")
            print("   3. Test on your own fraud datasets")
            print("   4. Use backend='auto' for automatic selection")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()