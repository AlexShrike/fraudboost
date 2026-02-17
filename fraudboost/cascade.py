"""
Cascade Fraud Detection: XGBoost (Stage 1, High Recall) → FraudBoost (Stage 2, High Precision)

The cascade design works as follows:
1. Stage 1 (XGBoost): Cast a wide net with high recall (~95%), accepting high FPs
2. Stage 2 (FraudBoost): Apply precision filter to Stage 1 positives using value-weighted loss
3. Final output: High-confidence fraud alerts with much fewer FPs

This approach is inspired by classic computer vision cascade classifiers (Viola-Jones face detection)
but adapted for fraud detection where value-weighted precision is crucial.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split
import xgboost as xgb
from fraudboost.core import FraudBoostClassifier
from fraudboost.metrics import (
    calculate_precision_recall_f1, 
    calculate_value_detection_rate, 
    calculate_net_savings
)


class FraudCascade:
    """
    Two-stage cascade fraud detection system.
    
    Stage 1: XGBoost optimized for high recall (catches most frauds, many FPs)
    Stage 2: FraudBoost optimized for precision on Stage 1 positives (filters FPs)
    
    Parameters:
    -----------
    stage1_threshold : float, default=0.1
        Threshold for Stage 1 (XGBoost). Lower values = higher recall, more FPs to Stage 2
    fp_cost : float, default=100
        False positive cost for FraudBoost (Stage 2)
    xgb_params : dict, optional
        XGBoost parameters. If None, uses sensible defaults for high recall
    fb_params : dict, optional
        FraudBoost parameters. If None, uses sensible defaults for high precision
    """
    
    def __init__(self, stage1_threshold=0.1, fp_cost=100, xgb_params=None, fb_params=None):
        self.stage1_threshold = stage1_threshold
        self.fp_cost = fp_cost
        
        # Default XGBoost params optimized for recall
        if xgb_params is None:
            self.xgb_params = {
                'n_estimators': 100,
                'max_depth': 4,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'eval_metric': 'auc',
                'early_stopping_rounds': 10
            }
        else:
            self.xgb_params = xgb_params
            
        # Default FraudBoost params optimized for precision
        if fb_params is None:
            self.fb_params = {
                'n_estimators': 100,
                'max_depth': 4,
                'learning_rate': 0.1,
                'fp_cost': fp_cost,
                'random_state': 42
            }
        else:
            self.fb_params = fb_params
            self.fb_params['fp_cost'] = fp_cost  # Always use the cascade's fp_cost
        
        self.xgb_model = None
        self.fb_model = None
        self.is_fitted = False
        
        # Stage statistics for evaluation
        self.stage1_stats = {}
        self.stage2_stats = {}
    
    def fit(self, X_train, y_train, amounts_train, X_val=None, y_val=None, amounts_val=None):
        """
        Fit the cascade model.
        
        Training strategy:
        1. Train XGBoost on full training data with scale_pos_weight for class imbalance
        2. Train FraudBoost on full training data (it needs to see legitimate transactions)
        3. During prediction, Stage 2 only processes Stage 1 positives
        
        Parameters:
        -----------
        X_train, X_val : array-like
            Feature matrices
        y_train, y_val : array-like  
            Target labels (0=legit, 1=fraud)
        amounts_train, amounts_val : array-like
            Transaction amounts for value-weighted loss
        """
        print("Training Cascade Fraud Detection System...")
        print(f"Stage 1 threshold: {self.stage1_threshold}")
        print(f"FP cost: ${self.fp_cost}")
        
        # Calculate class weights for XGBoost
        pos_count = np.sum(y_train)
        neg_count = len(y_train) - pos_count
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        
        print(f"Training data: {len(y_train):,} samples, {pos_count:,} frauds ({100*pos_count/len(y_train):.3f}%)")
        print(f"Scale pos weight: {scale_pos_weight:.2f}")
        
        # Stage 1: Train XGBoost for high recall
        print("\n=== Stage 1: Training XGBoost (High Recall) ===")
        self.xgb_params['scale_pos_weight'] = scale_pos_weight
        
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            eval_names = ['train', 'val']
        else:
            eval_set = [(X_train, y_train)]
            eval_names = ['train']
        
        self.xgb_model = xgb.XGBClassifier(**self.xgb_params)
        self.xgb_model.fit(
            X_train, y_train, 
            eval_set=eval_set,
            verbose=False
        )
        
        # Stage 2: Train FraudBoost for high precision
        print("\n=== Stage 2: Training FraudBoost (High Precision) ===")
        self.fb_model = FraudBoostClassifier(**self.fb_params)
        
        # FraudBoost needs amounts, so we pass them
        if X_val is not None:
            self.fb_model.fit(X_train, y_train, amounts_train, X_val, y_val, amounts_val)
        else:
            self.fb_model.fit(X_train, y_train, amounts_train)
        
        self.is_fitted = True
        
        # Calculate stage statistics on training data for analysis
        self._calculate_stage_stats(X_train, y_train, amounts_train, "train")
        if X_val is not None:
            self._calculate_stage_stats(X_val, y_val, amounts_val, "val")
        
        print(f"\n=== Cascade Training Complete ===")
        return self
    
    def _calculate_stage_stats(self, X, y, amounts, split_name):
        """Calculate stage-by-stage statistics for analysis."""
        # Stage 1 predictions
        xgb_proba = self.xgb_model.predict_proba(X)[:, 1]
        stage1_flags = xgb_proba >= self.stage1_threshold
        
        n_total = len(y)
        n_stage1_pos = np.sum(stage1_flags)
        n_fraud = np.sum(y)
        n_fraud_caught_stage1 = np.sum(y[stage1_flags]) if n_stage1_pos > 0 else 0
        
        stage1_recall = n_fraud_caught_stage1 / n_fraud if n_fraud > 0 else 0
        stage1_precision = n_fraud_caught_stage1 / n_stage1_pos if n_stage1_pos > 0 else 0
        
        stats = {
            'total_samples': n_total,
            'total_frauds': n_fraud,
            'stage1_flagged': n_stage1_pos,
            'stage1_fraud_caught': n_fraud_caught_stage1,
            'stage1_recall': stage1_recall,
            'stage1_precision': stage1_precision,
            'reduction_ratio': n_stage1_pos / n_total if n_total > 0 else 0
        }
        
        if split_name == "train":
            self.stage1_stats = stats
        else:
            self.stage2_stats = stats
            
        print(f"\nStage 1 Stats ({split_name}):")
        print(f"  Flagged: {n_stage1_pos:,}/{n_total:,} ({100*stats['reduction_ratio']:.1f}%)")
        print(f"  Recall: {100*stage1_recall:.1f}% ({n_fraud_caught_stage1}/{n_fraud})")
        print(f"  Precision: {100*stage1_precision:.1f}%")
    
    def predict_proba(self, X, amounts=None):
        """
        Predict fraud probabilities using the cascade.
        
        Process:
        1. All samples go through Stage 1 (XGBoost)
        2. Only Stage 1 positives go through Stage 2 (FraudBoost) 
        3. Final probability = Stage 2 probability for flagged samples, 0 for others
        
        Returns:
        --------
        probabilities : array
            Final fraud probabilities after cascade filtering
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        n_samples = len(X)
        
        # Stage 1: XGBoost predictions for all samples
        xgb_proba = self.xgb_model.predict_proba(X)[:, 1]
        stage1_flags = xgb_proba >= self.stage1_threshold
        n_stage1_pos = np.sum(stage1_flags)
        
        # Initialize final probabilities (start with zeros)
        final_probabilities = np.zeros(n_samples)
        
        if n_stage1_pos > 0:
            # Stage 2: FraudBoost predictions only for Stage 1 positives
            X_stage2 = X[stage1_flags]
            
            stage2_proba = self.fb_model.predict_proba(X_stage2)[:, 1]
            
            # Assign Stage 2 probabilities to flagged samples
            final_probabilities[stage1_flags] = stage2_proba
        
        return final_probabilities
    
    def predict(self, X, amounts=None, threshold=0.5):
        """
        Predict fraud labels using the cascade.
        
        Parameters:
        -----------
        X : array-like
            Features
        amounts : array-like, optional
            Transaction amounts 
        threshold : float, default=0.5
            Final threshold for cascade probabilities
            
        Returns:
        --------
        predictions : array
            Binary fraud predictions (0=legit, 1=fraud)
        """
        probas = self.predict_proba(X, amounts)
        return (probas >= threshold).astype(int)
    
    def evaluate(self, X, y, amounts, threshold=0.5):
        """
        Comprehensive evaluation of the cascade system.
        
        Returns metrics for both individual stages and the final cascade,
        plus stage-by-stage breakdown for analysis.
        
        Parameters:
        -----------
        X, y, amounts : array-like
            Test features, labels, and amounts
        threshold : float, default=0.5
            Final threshold for cascade predictions
            
        Returns:
        --------
        results : dict
            Comprehensive metrics including stage breakdown
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        n_samples = len(y)
        n_fraud = np.sum(y)
        
        # Stage 1 evaluation
        xgb_proba = self.xgb_model.predict_proba(X)[:, 1]
        stage1_flags = xgb_proba >= self.stage1_threshold
        n_stage1_pos = np.sum(stage1_flags)
        
        # Stage 2 evaluation (cascade)
        cascade_proba = self.predict_proba(X, amounts)
        cascade_pred = (cascade_proba >= threshold).astype(int)
        
        # Stage 1 metrics
        stage1_metrics = {}
        if n_fraud > 0:
            stage1_fraud_caught = np.sum(y[stage1_flags]) if n_stage1_pos > 0 else 0
            stage1_metrics = {
                'flagged': n_stage1_pos,
                'fraud_caught': stage1_fraud_caught,
                'recall': stage1_fraud_caught / n_fraud,
                'precision': stage1_fraud_caught / n_stage1_pos if n_stage1_pos > 0 else 0,
                'reduction_ratio': n_stage1_pos / n_samples
            }
        
        # Cascade metrics
        cascade_metrics = {}
        if len(np.unique(cascade_pred)) > 1:  # Avoid AUC issues with single class
            cascade_metrics['auc'] = roc_auc_score(y, cascade_proba)
        else:
            cascade_metrics['auc'] = 0.5
            
        prec, recall, f1 = calculate_precision_recall_f1(y, cascade_pred)
        cascade_metrics.update({
            'precision': prec,
            'recall': recall, 
            'f1': f1,
            'true_positives': np.sum((y == 1) & (cascade_pred == 1)),
            'false_positives': np.sum((y == 0) & (cascade_pred == 1)),
            'false_negatives': np.sum((y == 1) & (cascade_pred == 0)),
            'true_negatives': np.sum((y == 0) & (cascade_pred == 0))
        })
        
        # Value-based metrics
        vdr = calculate_value_detection_rate(y, cascade_pred, amounts)
        net_savings = calculate_net_savings(y, cascade_pred, amounts, self.fp_cost)
        
        cascade_metrics.update({
            'vdr': vdr,
            'net_savings': net_savings
        })
        
        return {
            'stage1': stage1_metrics,
            'cascade': cascade_metrics,
            'threshold': threshold,
            'stage1_threshold': self.stage1_threshold,
            'fp_cost': self.fp_cost
        }
    
    def get_cascade_breakdown(self, X, y, amounts):
        """
        Detailed breakdown of what happens at each cascade stage.
        Useful for understanding and visualizing the cascade flow.
        """
        xgb_proba = self.xgb_model.predict_proba(X)[:, 1]
        stage1_flags = xgb_proba >= self.stage1_threshold
        cascade_proba = self.predict_proba(X, amounts)
        
        breakdown = {
            'total_samples': len(X),
            'total_frauds': np.sum(y),
            'stage1_probabilities': xgb_proba,
            'stage1_flags': stage1_flags,
            'stage1_flagged_count': np.sum(stage1_flags),
            'cascade_probabilities': cascade_proba,
            'frauds_caught_stage1': np.sum(y[stage1_flags]) if np.sum(stage1_flags) > 0 else 0,
            'frauds_missed_stage1': np.sum(y) - (np.sum(y[stage1_flags]) if np.sum(stage1_flags) > 0 else 0)
        }
        
        return breakdown