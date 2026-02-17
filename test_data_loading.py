#!/usr/bin/env python3
"""Test just the data loading and preparation parts."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

print("Starting data loading test...")

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
    print(f"  Starting feature engineering on {len(df)} rows...")
    df = df.copy()
    
    # Parse datetime
    print("  Parsing datetime...")
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
    
    # Calculate age from birth year
    print("  Calculating age...")
    df['age'] = 2024 - pd.to_datetime(df['dob']).dt.year
    
    # Distance between customer and merchant
    print("  Calculating distances...")
    df['distance'] = haversine_distance(
        df['lat'], df['long'], df['merch_lat'], df['merch_long']
    )
    
    # Amount log transform
    print("  Log transforming amounts...")
    df['amt_log'] = np.log1p(df['amt'])
    
    # Encode categorical variables
    print("  Encoding categories...")
    le_category = LabelEncoder()
    df['category_encoded'] = le_category.fit_transform(df['category'])
    df['gender_encoded'] = (df['gender'] == 'M').astype(int)
    
    # Select final features (NO target leakage)
    features = [
        'amt', 'category_encoded', 'gender_encoded', 'city_pop',
        'lat', 'long', 'merch_lat', 'merch_long', 'hour', 'day_of_week',
        'age', 'distance', 'amt_log'
    ]
    
    print("  Selecting features...")
    X = df[features].copy()
    y = df['is_fraud'].copy() if 'is_fraud' in df.columns else None
    
    print(f"  Feature engineering complete: {X.shape}")
    return X, y, features

# Load datasets step by step
print("\n=== LOADING DATASETS ===")
train_path = "/Users/alexshrike/.openclaw/workspace/bastion/data/fraudTrain.csv"

print(f"Loading train dataset from {train_path}...")
train_df = pd.read_csv(train_path)
print(f"✓ Train loaded: {len(train_df):,} rows")

# Take a smaller subsample for testing
print("\n=== SUBSAMPLING FOR TEST ===")
target_size = 50_000  # Smaller sample for testing
fraud_rate = train_df['is_fraud'].mean()

n_fraud = int(target_size * fraud_rate)
n_legit = target_size - n_fraud

print(f"Sampling {n_fraud:,} fraud + {n_legit:,} legitimate transactions...")

fraud_samples = train_df[train_df['is_fraud'] == 1].sample(
    n=min(n_fraud, sum(train_df['is_fraud'])), random_state=42
)
legit_samples = train_df[train_df['is_fraud'] == 0].sample(
    n=min(n_legit, sum(train_df['is_fraud'] == 0)), random_state=42
)

train_subsample = pd.concat([fraud_samples, legit_samples]).sample(
    frac=1, random_state=42
).reset_index(drop=True)

print(f"✓ Subsampled: {len(train_subsample):,} rows, fraud rate: {train_subsample['is_fraud'].mean()*100:.3f}%")

# Feature engineering
print("\n=== FEATURE ENGINEERING ===")
X, y, feature_names = engineer_features(train_subsample)

print(f"✓ Features engineered: {X.shape}")
print(f"Feature names: {feature_names}")

# Train/test split
print("\n=== TRAIN/TEST SPLIT ===")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(X_train):,} rows, fraud rate: {y_train.mean()*100:.3f}%")
print(f"Test: {len(X_test):,} rows, fraud rate: {y_test.mean()*100:.3f}%")

print("\n✓ Data loading and preparation test complete!")
print("Ready to proceed with model training.")