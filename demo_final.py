#!/usr/bin/env python3
"""
Final demonstration of FraudBoost library capabilities.
Shows the complete workflow from data generation to business impact evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from fraudboost import (
    FraudBoostClassifier,
    ParetoOptimizer,
    classification_report_fraud,
    value_detection_rate,
    net_savings,
    roi
)


def main():
    print("🎯 FraudBoost - Fraud Detection Made for Business Impact")
    print("=" * 60)
    
    # Create realistic fraud dataset
    np.random.seed(42)
    print("📊 Creating synthetic fraud detection dataset...")
    
    # Highly imbalanced dataset (realistic fraud rate)
    X, y = make_classification(
        n_samples=5000,
        n_features=15,
        n_informative=12,
        n_clusters_per_class=2,
        weights=[0.99, 0.01],  # 1% fraud rate
        flip_y=0.005,  # Small amount of noise
        random_state=42
    )
    
    # Generate realistic transaction amounts
    # Legitimate: $10-$500 (log-normal distribution)
    amounts = np.random.lognormal(mean=3.5, sigma=1.0, size=len(y))
    
    # Fraud: $100-$10,000 (much higher value)
    fraud_mask = (y == 1)
    amounts[fraud_mask] = np.random.lognormal(mean=5.5, sigma=1.2, size=np.sum(fraud_mask))
    
    print(f"   Dataset: {len(X):,} transactions")
    print(f"   Fraud rate: {np.mean(y):.2%} ({np.sum(y)} fraud cases)")
    print(f"   Average legit amount: ${np.mean(amounts[~fraud_mask]):,.0f}")
    print(f"   Average fraud amount: ${np.mean(amounts[fraud_mask]):,.0f}")
    print(f"   Total fraud value: ${np.sum(amounts[fraud_mask]):,.0f}")
    
    # Train/test split
    X_train, X_test, y_train, y_test, amounts_train, amounts_test = train_test_split(
        X, y, amounts, test_size=0.3, stratify=y, random_state=42
    )
    
    print(f"\n🤖 Training FraudBoost classifier...")
    print("   Key features:")
    print("   • Value-weighted loss (FN cost ∝ transaction amount)")
    print("   • Tree splits maximize net savings")  
    print("   • Built-in early stopping")
    
    # Train FraudBoost with value-weighted loss
    model = FraudBoostClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        fp_cost=200,  # $200 cost per investigation
        loss='value_weighted',  # Key innovation!
        early_stopping_rounds=10,
        verbose=0
    )
    
    model.fit(X_train, y_train, amounts=amounts_train)
    
    print(f"   ✓ Model trained with {len(model.estimators_)} trees")
    if model.best_iteration_:
        print(f"   ✓ Early stopped at iteration {model.best_iteration_ + 1}")
    
    # Get predictions 
    print(f"\n🎯 Finding optimal decision threshold...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Use Pareto optimization to find best threshold
    optimizer = ParetoOptimizer(fp_cost=200)
    optimizer.fit(y_test, y_pred_proba, amounts_test)
    
    # Get recommendation for maximizing savings
    recommendation = optimizer.recommend_threshold('max_savings')
    optimal_threshold = recommendation['threshold']
    
    print(f"   ✓ Optimal threshold: {optimal_threshold:.3f}")
    print(f"   ✓ Expected net savings: ${recommendation['metrics']['net_savings']:,.0f}")
    
    # Make predictions with optimal threshold
    y_pred = model.predict(X_test, threshold=optimal_threshold)
    
    print(f"\n📈 Business Impact Results:")
    print("-" * 40)
    
    # Traditional metrics
    from sklearn.metrics import precision_score, recall_score, f1_score
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"Traditional Metrics:")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1-Score: {f1:.3f}")
    
    # Fraud-specific business metrics
    vdr = value_detection_rate(y_test, y_pred, amounts_test)
    savings = net_savings(y_test, y_pred, amounts_test, fp_cost=200)
    roi_value = roi(y_test, y_pred, amounts_test, fp_cost=200)
    
    print(f"\n💰 Business Impact Metrics:")
    print(f"  Value Detection Rate: {vdr:.1%} of fraud $ caught")
    print(f"  Net Savings: ${savings:,.0f}")
    print(f"  ROI: {roi_value:.1f}x return on investigation")
    print(f"  Investigations: {np.sum(y_pred):,} cases to review")
    
    # Operational details
    tp = np.sum((y_test == 1) & (y_pred == 1))
    fp = np.sum((y_test == 0) & (y_pred == 1)) 
    fn = np.sum((y_test == 1) & (y_pred == 0))
    
    fraud_caught = np.sum(amounts_test[(y_test == 1) & (y_pred == 1)])
    fraud_missed = np.sum(amounts_test[(y_test == 1) & (y_pred == 0)])
    
    print(f"\n🎪 Operational Impact:")
    print(f"  Fraud Caught: {tp} cases worth ${fraud_caught:,.0f}")
    print(f"  Fraud Missed: {fn} cases worth ${fraud_missed:,.0f}")
    print(f"  False Alarms: {fp} cases costing ${fp * 200:,.0f}")
    print(f"  Daily Workload: ~{np.sum(y_pred)} investigations")
    
    # Show feature importance
    print(f"\n🔍 Top 5 Most Important Features:")
    feature_names = [f'Feature_{i:02d}' for i in range(X.shape[1])]
    importance = model.feature_importances_
    top_features = np.argsort(importance)[-5:][::-1]
    
    for i, feat_idx in enumerate(top_features):
        print(f"  {i+1}. {feature_names[feat_idx]}: {importance[feat_idx]:.3f}")
    
    # Compare thresholds
    print(f"\n⚖️  Threshold Strategy Comparison:")
    strategies = ['max_vdr', 'max_savings', 'min_fps', 'balanced']
    strategy_names = {
        'max_vdr': 'Max Value Detection',
        'max_savings': 'Max Net Savings', 
        'min_fps': 'Min False Positives',
        'balanced': 'Balanced F1'
    }
    
    for strategy in strategies:
        if strategy in optimizer.recommended_thresholds_:
            rec = optimizer.recommended_thresholds_[strategy]
            metrics = rec['metrics']
            print(f"  {strategy_names[strategy]:<20}: "
                  f"thresh={rec['threshold']:.3f}, "
                  f"VDR={metrics['vdr']:.2f}, "
                  f"savings=${metrics['net_savings']:,.0f}")
    
    print(f"\n🎉 FraudBoost Demo Complete!")
    print(f"   The algorithm successfully optimizes for financial impact,")
    print(f"   not just statistical accuracy!")
    
    print(f"\n💡 Key Innovations:")
    print(f"   ✓ False negatives weighted by transaction amount")
    print(f"   ✓ Tree splits maximize net business savings")
    print(f"   ✓ Automatic threshold optimization")
    print(f"   ✓ Business-focused evaluation metrics")


if __name__ == "__main__":
    main()