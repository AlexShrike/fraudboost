#!/usr/bin/env python
"""Quick test to verify cascade imports work."""

import sys
import os

# Add the fraudboost directory to Python path
sys.path.append('/Users/alexshrike/.openclaw/workspace/fraudboost')

print("Testing imports...")

try:
    from fraudboost.core import FraudBoostClassifier
    print("✓ FraudBoostClassifier imported successfully")
except ImportError as e:
    print(f"✗ FraudBoostClassifier import failed: {e}")

try:
    from fraudboost.cascade import FraudCascade  
    print("✓ FraudCascade imported successfully")
except ImportError as e:
    print(f"✗ FraudCascade import failed: {e}")

try:
    from fraudboost.stacking import FraudStacking
    print("✓ FraudStacking imported successfully")  
except ImportError as e:
    print(f"✗ FraudStacking import failed: {e}")

try:
    import xgboost as xgb
    print(f"✓ XGBoost {xgb.__version__} imported successfully")
except ImportError as e:
    print(f"✗ XGBoost import failed: {e}")

try:
    import pandas as pd
    print(f"✓ Pandas {pd.__version__} imported successfully")
except ImportError as e:
    print(f"✗ Pandas import failed: {e}")

try:
    # Test basic instantiation
    cascade = FraudCascade(stage1_threshold=0.1)
    print("✓ FraudCascade instantiation successful")
except Exception as e:
    print(f"✗ FraudCascade instantiation failed: {e}")

try:
    stacking = FraudStacking(fp_cost=100)
    print("✓ FraudStacking instantiation successful")
except Exception as e:
    print(f"✗ FraudStacking instantiation failed: {e}")

print("\nImport test complete!")