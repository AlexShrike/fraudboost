"""
Basic usage example for FraudBoost library.

This example demonstrates:
1. Creating synthetic fraud detection data
2. Training a FraudBoostClassifier 
3. Evaluating performance with fraud-specific metrics
4. Optimizing decision thresholds
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Import FraudBoost components
from fraudboost import (
    FraudBoostClassifier,
    FraudDetectionPipeline,
    ParetoOptimizer,
    classification_report_fraud,
    value_detection_rate,
    net_savings,
    roi
)


def create_synthetic_fraud_data(n_samples=10000, fraud_rate=0.005, random_state=42):
    """Create realistic synthetic fraud detection dataset."""
    
    # Generate imbalanced classification data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_clusters_per_class=1,
        weights=[1-fraud_rate, fraud_rate],
        flip_y=0.01,  # Add some noise
        random_state=random_state
    )
    
    # Generate transaction amounts
    # Legitimate transactions: smaller amounts
    # Fraudulent transactions: tend to be higher value
    np.random.seed(random_state)
    amounts = np.random.lognormal(4, 1, n_samples)  # Base amounts: ~$50-500
    
    # Fraud transactions are typically higher value
    fraud_mask = (y == 1)
    amounts[fraud_mask] *= np.random.uniform(3, 8, np.sum(fraud_mask))
    
    # Create feature names
    feature_names = [f'feature_{i:02d}' for i in range(X.shape[1])]
    
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(X, columns=feature_names)
    df['amount'] = amounts
    df['is_fraud'] = y
    
    return df


def demonstrate_basic_classifier():
    """Demonstrate basic FraudBoostClassifier usage."""
    
    print("=" * 60)
    print("FRAUDBOOST BASIC CLASSIFIER EXAMPLE")
    print("=" * 60)
    
    # Create synthetic data
    print("1. Creating synthetic fraud detection dataset...")
    df = create_synthetic_fraud_data(n_samples=5000, fraud_rate=0.01)  # 1% fraud rate
    
    print(f"   Dataset shape: {df.shape}")
    print(f"   Fraud rate: {df['is_fraud'].mean():.4f} ({df['is_fraud'].mean()*100:.2f}%)")
    print(f"   Average fraud amount: ${df[df['is_fraud']==1]['amount'].mean():,.2f}")
    print(f"   Average legit amount: ${df[df['is_fraud']==0]['amount'].mean():,.2f}")
    
    # Split features and targets
    X = df.drop(['amount', 'is_fraud'], axis=1).values
    y = df['is_fraud'].values
    amounts = df['amount'].values
    
    # Train/test split
    X_train, X_test, y_train, y_test, amounts_train, amounts_test = train_test_split(
        X, y, amounts, test_size=0.3, stratify=y, random_state=42
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Train FraudBoost model
    print("\n2. Training FraudBoost classifier...")
    model = FraudBoostClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        fp_cost=150,  # $150 cost per false positive investigation
        loss='value_weighted',  # Use value-weighted loss
        early_stopping_rounds=10,
        verbose=1
    )
    
    # Fit with transaction amounts for value weighting
    model.fit(X_train, y_train, amounts=amounts_train)
    
    print(f"   Model trained with {len(model.estimators_)} estimators")
    if model.best_iteration_ is not None:
        print(f"   Early stopped at iteration {model.best_iteration_}")
    
    # Make predictions
    print("\n3. Making predictions...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test, threshold=0.5)
    
    print(f"   Average prediction probability: {np.mean(y_pred_proba):.4f}")
    print(f"   Predictions flagged as fraud: {np.sum(y_pred)} ({np.sum(y_pred)/len(y_pred)*100:.2f}%)")
    
    # Evaluate performance
    print("\n4. Evaluating performance...")
    
    # Traditional metrics
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"   Traditional metrics:")
    print(f"     Precision: {precision:.4f}")
    print(f"     Recall: {recall:.4f}")
    print(f"     F1-Score: {f1:.4f}")
    print(f"     AUC: {auc:.4f}")
    
    # Fraud-specific metrics
    vdr = value_detection_rate(y_test, y_pred, amounts_test)
    savings = net_savings(y_test, y_pred, amounts_test, fp_cost=150)
    roi_value = roi(y_test, y_pred, amounts_test, fp_cost=150)
    
    print(f"\n   Fraud-specific metrics:")
    print(f"     Value Detection Rate: {vdr:.4f} ({vdr*100:.1f}% of fraud $ caught)")
    print(f"     Net Savings: ${savings:,.2f}")
    print(f"     ROI: {roi_value:.2f}x")
    
    # Feature importance
    print(f"\n5. Top 10 most important features:")
    feature_names = [f'feature_{i:02d}' for i in range(X.shape[1])]
    importance = model.feature_importances_
    
    # Sort by importance
    sorted_idx = np.argsort(importance)[::-1]
    for i in range(min(10, len(feature_names))):
        idx = sorted_idx[i]
        print(f"     {feature_names[idx]}: {importance[idx]:.4f}")
    
    return model, X_test, y_test, amounts_test, y_pred_proba


def demonstrate_threshold_optimization(model, X_test, y_test, amounts_test, y_pred_proba):
    """Demonstrate Pareto-optimal threshold selection."""
    
    print("\n" + "=" * 60)
    print("THRESHOLD OPTIMIZATION WITH PARETO FRONTIER")
    print("=" * 60)
    
    # Initialize Pareto optimizer
    print("1. Finding Pareto-optimal thresholds...")
    optimizer = ParetoOptimizer(fp_cost=150)
    optimizer.fit(y_test, y_pred_proba, amounts_test)
    
    print(f"   Evaluated {len(optimizer.results_['thresholds'])} thresholds")
    print(f"   Found {len(optimizer.pareto_indices_)} Pareto-optimal points")
    
    # Show different threshold strategies
    print("\n2. Recommended thresholds for different business objectives:")
    
    strategies = ['max_vdr', 'max_savings', 'min_fps', 'balanced', 'max_roi']
    strategy_names = {
        'max_vdr': 'Maximize Value Detection',
        'max_savings': 'Maximize Net Savings', 
        'min_fps': 'Minimize False Positives',
        'balanced': 'Balanced Precision/Recall',
        'max_roi': 'Maximize ROI'
    }
    
    for strategy in strategies:
        if strategy in optimizer.recommended_thresholds_:
            rec = optimizer.recommend_threshold(strategy)
            threshold = rec['threshold']
            metrics = rec['metrics']
            
            print(f"\n   {strategy_names[strategy]}:")
            print(f"     Optimal threshold: {threshold:.3f}")
            print(f"     Precision: {metrics['precision']:.3f}")
            print(f"     Recall: {metrics['recall']:.3f}")
            print(f"     VDR: {metrics['vdr']:.3f}")
            print(f"     Net Savings: ${metrics['net_savings']:,.0f}")
            print(f"     ROI: {metrics['roi']:.2f}x")
            print(f"     Investigations per day: {metrics['tp_count'] + metrics['fp_count']:,}")
    
    # Generate comprehensive report
    print("\n3. Comprehensive threshold optimization report:")
    print(optimizer.generate_report())
    
    return optimizer


def demonstrate_full_pipeline():
    """Demonstrate end-to-end FraudDetectionPipeline."""
    
    print("\n" + "=" * 60)
    print("END-TO-END FRAUD DETECTION PIPELINE")
    print("=" * 60)
    
    # Create dataset with entity relationships for spectral features
    print("1. Creating dataset with entity relationships...")
    
    # Generate base fraud data
    df = create_synthetic_fraud_data(n_samples=3000, fraud_rate=0.01)
    
    # Add synthetic entity relationships (customer, merchant, device)
    np.random.seed(42)
    n_customers = 500
    n_merchants = 100
    n_devices = 200
    
    df['customer_id'] = np.random.randint(1, n_customers + 1, len(df))
    df['merchant_id'] = np.random.randint(1, n_merchants + 1, len(df))
    df['device_id'] = np.random.randint(1, n_devices + 1, len(df))
    
    # Split data
    train_df = df.iloc[:2000].copy()
    test_df = df.iloc[2000:].copy()
    
    print(f"   Training samples: {len(train_df)}")
    print(f"   Test samples: {len(test_df)}")
    
    # Prepare data
    feature_cols = [col for col in train_df.columns if col.startswith('feature_')]
    entity_cols = ['customer_id', 'merchant_id', 'device_id']
    
    X_train = train_df[feature_cols].values
    y_train = train_df['is_fraud'].values  
    amounts_train = train_df['amount'].values
    
    X_test = test_df[feature_cols].values
    y_test = test_df['is_fraud'].values
    amounts_test = test_df['amount'].values
    
    # Initialize full pipeline
    print("\n2. Training complete pipeline with spectral features...")
    pipeline = FraudDetectionPipeline(
        n_estimators=50,
        learning_rate=0.1,
        fp_cost=150,
        use_spectral_features=True,  # Enable graph features
        spectral_components=8,
        optimize_threshold=True,
        enable_drift_detection=True,
        verbose=1
    )
    
    # Fit pipeline (pass DataFrames for spectral feature extraction)
    pipeline.fit(
        train_df[feature_cols + entity_cols], 
        y_train, 
        amounts_train,
        entity_columns=entity_cols
    )
    
    print(f"   Pipeline trained successfully!")
    print(f"   Total features used: {len(pipeline.feature_names_)}")
    print(f"   Optimal threshold: {pipeline.optimal_threshold_:.3f}")
    
    # Evaluate on test set
    print("\n3. Evaluating complete pipeline...")
    
    evaluation = pipeline.evaluate(
        test_df[feature_cols + entity_cols], 
        y_test, 
        amounts_test,
        entity_columns=entity_cols
    )
    
    # Print results
    metrics = evaluation['classification_metrics']
    print(f"\n   Final Pipeline Performance:")
    print(f"     Precision: {metrics['precision']:.4f}")
    print(f"     Recall: {metrics['recall']:.4f}")
    print(f"     F1-Score: {metrics['f1_score']:.4f}")
    print(f"     Value Detection Rate: {metrics['value_detection_rate']:.4f}")
    print(f"     Net Savings: ${metrics['net_savings']:,.2f}")
    print(f"     ROI: {metrics['roi']:.2f}x")
    
    cost_benefit = evaluation['cost_benefit_analysis']
    print(f"\n   Cost-Benefit Analysis:")
    print(f"     Fraud Prevented: ${cost_benefit['fraud_prevented']:,.2f}")
    print(f"     Investigation Cost: ${cost_benefit['investigation_cost']:,.2f}")
    print(f"     Net Benefit: ${cost_benefit['net_benefit']:,.2f}")
    
    return pipeline


if __name__ == "__main__":
    print("FraudBoost Library - Basic Usage Examples")
    print("=========================================")
    
    # Run examples
    try:
        # Basic classifier example
        model, X_test, y_test, amounts_test, y_pred_proba = demonstrate_basic_classifier()
        
        # Threshold optimization example
        optimizer = demonstrate_threshold_optimization(model, X_test, y_test, amounts_test, y_pred_proba)
        
        # Full pipeline example
        pipeline = demonstrate_full_pipeline()
        
        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()