#!/usr/bin/env python3
"""Mini benchmark test to verify logic works before running full benchmark."""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend for headless
import matplotlib
matplotlib.use('Agg')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from fraudboost import FraudBoostClassifier

print("=== MINI BENCHMARK TEST ===")

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
    
    # Select final features
    features = [
        'amt', 'category_encoded', 'gender_encoded', 'city_pop',
        'lat', 'long', 'merch_lat', 'merch_long', 'hour', 'day_of_week',
        'age', 'distance', 'amt_log'
    ]
    
    X = df[features].copy()
    y = df['is_fraud'].copy() if 'is_fraud' in df.columns else None
    
    return X, y, features

# Load small sample
print("Loading datasets...")
train_path = "/Users/alexshrike/.openclaw/workspace/bastion/data/fraudTrain.csv"
train_df = pd.read_csv(train_path, nrows=10000)  # Small sample

print(f"Loaded: {len(train_df):,} rows, fraud rate: {train_df['is_fraud'].mean()*100:.3f}%")

# Feature engineering
print("Feature engineering...")
X, y, feature_names = engineer_features(train_df)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train: {len(X_train):,} rows, fraud rate: {y_train.mean()*100:.3f}%")
print(f"Test: {len(X_test):,} rows, fraud rate: {y_test.mean()*100:.3f}%")

# Train models
print("Training XGBoost...")
pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
xgb_model = xgb.XGBClassifier(
    n_estimators=10,  # Small for test
    max_depth=4,
    learning_rate=0.1,
    scale_pos_weight=pos_weight,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)

print("Training FraudBoost...")
fb_model = FraudBoostClassifier(
    n_estimators=10,  # Small for test
    max_depth=4,
    learning_rate=0.1,
    fp_cost=100,
    loss='value_weighted',
    random_state=42
)
fb_model.fit(X_train, y_train)

# Test predictions
print("Making predictions...")
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
fb_proba = fb_model.predict_proba(X_test)[:, 1]

# Calculate AUCs
xgb_auc = roc_auc_score(y_test, xgb_proba)
fb_auc = roc_auc_score(y_test, fb_proba)

print("\n=== RESULTS ===")
print(f"XGBoost AUC: {xgb_auc:.4f}")
print(f"FraudBoost AUC: {fb_auc:.4f}")
print("Mini test complete! Full benchmark should work.")