#!/usr/bin/env python3
"""Quick test to verify imports and basic setup."""

print("Testing imports...")

try:
    import numpy as np
    print("✓ numpy")
except Exception as e:
    print(f"✗ numpy: {e}")

try:
    import pandas as pd
    print("✓ pandas")
except Exception as e:
    print(f"✗ pandas: {e}")

try:
    import xgboost as xgb
    print("✓ xgboost")
except Exception as e:
    print(f"✗ xgboost: {e}")

try:
    from fraudboost import FraudBoostClassifier
    print("✓ fraudboost")
except Exception as e:
    print(f"✗ fraudboost: {e}")

try:
    from sklearn.model_selection import train_test_split
    print("✓ sklearn")
except Exception as e:
    print(f"✗ sklearn: {e}")

# Test dataset loading
print("\nTesting dataset loading...")
try:
    train_path = "/Users/alexshrike/.openclaw/workspace/bastion/data/fraudTrain.csv"
    test_path = "/Users/alexshrike/.openclaw/workspace/bastion/data/fraudTest.csv"
    
    print(f"Checking train path: {train_path}")
    import os
    if os.path.exists(train_path):
        train_df = pd.read_csv(train_path, nrows=5)  # Just first 5 rows
        print(f"✓ Train dataset loaded, shape: {train_df.shape}")
        print(f"Columns: {list(train_df.columns)}")
    else:
        print(f"✗ Train dataset not found at {train_path}")
        
    print(f"Checking test path: {test_path}")
    if os.path.exists(test_path):
        test_df = pd.read_csv(test_path, nrows=5)  # Just first 5 rows
        print(f"✓ Test dataset loaded, shape: {test_df.shape}")
    else:
        print(f"✗ Test dataset not found at {test_path}")
        
except Exception as e:
    print(f"✗ Dataset loading failed: {e}")

print("\nTest complete.")