#!/usr/bin/env python3
"""
Simple test to debug the Rust backend.
"""
import numpy as np
import time

print("Testing imports...")
try:
    from fraudboost import FraudBoostClassifier
    print("✅ FraudBoostClassifier imported successfully")
except Exception as e:
    print(f"❌ Failed to import FraudBoostClassifier: {e}")
    exit(1)

try:
    from fraudboost._rust_core import has_rust_backend, RustBooster
    print(f"✅ Rust backend available: {has_rust_backend()}")
    if has_rust_backend():
        print("✅ RustBooster imported successfully")
except Exception as e:
    print(f"❌ Failed to import Rust backend: {e}")

# Test very small dataset
print("\nTesting with small dataset...")
np.random.seed(42)
X = np.random.randn(100, 5)
y = np.zeros(100)
y[:10] = 1  # 10% fraud
amounts = np.random.uniform(10, 500, 100)

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

# Test Python backend first
print("\nTesting Python backend...")
try:
    model = FraudBoostClassifier(
        n_estimators=5,
        max_depth=3,
        backend='python',
        verbose=1,
        random_state=42
    )
    print("Model created")
    
    t0 = time.time()
    model.fit(X, y, amounts=amounts)
    python_time = time.time() - t0
    print(f"Python backend fit completed in {python_time:.3f}s")
    
    probs = model.predict_proba(X)
    print(f"Predictions shape: {probs.shape}")
    print(f"Mean prediction: {np.mean(probs[:, 1]):.4f}")
    
except Exception as e:
    print(f"❌ Python backend failed: {e}")
    import traceback
    traceback.print_exc()

# Test Rust backend if available
if has_rust_backend():
    print("\nTesting Rust backend...")
    try:
        model_rust = FraudBoostClassifier(
            n_estimators=5,
            max_depth=3,
            backend='rust',
            verbose=1,
            random_state=42
        )
        print("Rust model created")
        
        t0 = time.time()
        model_rust.fit(X, y, amounts=amounts)
        rust_time = time.time() - t0
        print(f"Rust backend fit completed in {rust_time:.3f}s")
        
        probs_rust = model_rust.predict_proba(X)
        print(f"Rust predictions shape: {probs_rust.shape}")
        print(f"Rust mean prediction: {np.mean(probs_rust[:, 1]):.4f}")
        
        speedup = python_time / rust_time
        print(f"Speedup: {speedup:.2f}x")
        
    except Exception as e:
        print(f"❌ Rust backend failed: {e}")
        import traceback
        traceback.print_exc()

print("\nTest completed!")