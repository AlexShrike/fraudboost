"""
FraudBoost: Gradient boosting framework purpose-built for fraud detection.

A novel gradient boosting implementation optimized for fraud detection with:
- Value-weighted asymmetric loss functions
- Spectral graph features from transaction networks  
- Pareto-optimal threshold selection
- Temporal drift monitoring
- Financial impact optimization
"""

__version__ = "0.1.0"
__author__ = "Alex Shrike"

# Core classifier
from .core import FraudBoostClassifier

# Loss functions
from .losses import (
    ValueWeightedLogLoss, 
    FocalLoss, 
    LogLoss,
    BaseLoss
)

# Decision tree
from .tree import ValueWeightedDecisionTree

# Spectral features
from .spectral import SpectralFeatureExtractor

# Pareto optimization
from .pareto import ParetoOptimizer  

# Drift detection
from .drift import TemporalDriftDetector

# End-to-end pipeline
from .pipeline import FraudDetectionPipeline

# Fraud-specific metrics
from .metrics import (
    value_detection_rate,
    net_savings,
    roi,
    precision_at_dollar_threshold,
    classification_report_fraud,
    evaluate_at_thresholds,
    cost_benefit_analysis
)

# Main classes for easy import
__all__ = [
    # Core components
    'FraudBoostClassifier',
    'FraudDetectionPipeline',
    
    # Loss functions
    'ValueWeightedLogLoss',
    'FocalLoss', 
    'LogLoss',
    'BaseLoss',
    
    # Feature extraction
    'SpectralFeatureExtractor',
    
    # Optimization and monitoring
    'ParetoOptimizer',
    'TemporalDriftDetector',
    
    # Tree implementation
    'ValueWeightedDecisionTree',
    
    # Metrics
    'value_detection_rate',
    'net_savings',
    'roi',
    'precision_at_dollar_threshold', 
    'classification_report_fraud',
    'evaluate_at_thresholds',
    'cost_benefit_analysis'
]

# Package metadata
__doc__ = """
FraudBoost: Purpose-built gradient boosting for fraud detection

Key features:
- Value-weighted loss functions that cost FN by transaction amount
- Spectral graph features from transaction relationships  
- Pareto-optimal threshold selection across business objectives
- Built-in temporal drift monitoring and retraining alerts
- Financial impact metrics (net savings, ROI, value detection rate)

Quick start:
    >>> from fraudboost import FraudBoostClassifier
    >>> model = FraudBoostClassifier(fp_cost=100)
    >>> model.fit(X, y, amounts=transaction_amounts)
    >>> predictions = model.predict_proba(X_new)

For full pipeline with spectral features and optimization:
    >>> from fraudboost import FraudDetectionPipeline
    >>> pipeline = FraudDetectionPipeline(use_spectral_features=True)
    >>> pipeline.fit(X, y, amounts, entity_columns=['customer', 'merchant'])
    >>> evaluation = pipeline.evaluate(X_test, y_test, amounts_test)
"""