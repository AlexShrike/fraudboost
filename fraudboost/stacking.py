"""
Stacking Ensemble: XGBoost + FraudBoost → Logistic Regression Meta-Learner

This approach combines predictions from both XGBoost and FraudBoost using a 
meta-learner that considers:
- XGBoost probability 
- FraudBoost probability
- Transaction amount (raw and log-transformed)

The meta-learner learns how to optimally weight the base models' predictions
along with transaction amount information.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from fraudboost.core import FraudBoostClassifier
from fraudboost.metrics import (
    calculate_precision_recall_f1, 
    calculate_value_detection_rate, 
    calculate_net_savings
)


class FraudStacking:
    """
    Stacking ensemble for fraud detection.
    
    Architecture:
    - Base Model 1: XGBoost 
    - Base Model 2: FraudBoost (value-weighted)
    - Meta-learner: Logistic Regression on [xgb_proba, fb_proba, amount, amount_log]
    
    The meta-learner learns how to combine base model predictions optimally,
    potentially discovering that certain amount ranges favor one model over another.
    
    Parameters:
    -----------
    fp_cost : float, default=100
        False positive cost for FraudBoost
    xgb_params : dict, optional
        XGBoost parameters
    fb_params : dict, optional  
        FraudBoost parameters
    meta_params : dict, optional
        Meta-learner (LogisticRegression) parameters
    cv_folds : int, default=3
        Cross-validation folds for generating out-of-fold predictions
    """
    
    def __init__(self, fp_cost=100, xgb_params=None, fb_params=None, meta_params=None, cv_folds=3):
        self.fp_cost = fp_cost
        self.cv_folds = cv_folds
        
        # Base model 1: XGBoost
        if xgb_params is None:
            self.xgb_params = {
                'n_estimators': 100,
                'max_depth': 4,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'eval_metric': 'auc'
            }
        else:
            self.xgb_params = xgb_params
            
        # Base model 2: FraudBoost
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
            self.fb_params['fp_cost'] = fp_cost
            
        # Meta-learner: Logistic Regression
        if meta_params is None:
            self.meta_params = {
                'C': 1.0,
                'random_state': 42,
                'max_iter': 1000
            }
        else:
            self.meta_params = meta_params
            
        self.xgb_model = None
        self.fb_model = None
        self.meta_model = None
        self.is_fitted = False
        
        # Store feature names for meta-learner
        self.meta_feature_names = ['xgb_proba', 'fb_proba', 'amount', 'amount_log']
    
    def fit(self, X_train, y_train, amounts_train):
        """
        Fit the stacking ensemble.
        
        Process:
        1. Generate out-of-fold (OOF) predictions from base models using CV
        2. Create meta-features: [xgb_proba, fb_proba, amount, amount_log]
        3. Train meta-learner on meta-features → y_train
        4. Refit base models on full training data for final inference
        
        Parameters:
        -----------
        X_train : array-like
            Training features  
        y_train : array-like
            Training labels (0=legit, 1=fraud)
        amounts_train : array-like
            Training transaction amounts
        """
        print("Training Stacking Ensemble...")
        print(f"FP cost: ${self.fp_cost}")
        print(f"CV folds: {self.cv_folds}")
        
        n_samples = len(y_train)
        pos_count = np.sum(y_train)
        neg_count = n_samples - pos_count
        
        print(f"Training data: {n_samples:,} samples, {pos_count:,} frauds ({100*pos_count/n_samples:.3f}%)")
        
        # Calculate class weights for XGBoost
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        self.xgb_params['scale_pos_weight'] = scale_pos_weight
        
        print(f"Scale pos weight: {scale_pos_weight:.2f}")
        
        # Step 1: Generate out-of-fold predictions from base models
        print("\n=== Generating OOF predictions ===")
        
        # XGBoost OOF predictions
        print("XGBoost OOF predictions...")
        xgb_temp = xgb.XGBClassifier(**self.xgb_params)
        xgb_oof = cross_val_predict(
            xgb_temp, X_train, y_train, 
            cv=self.cv_folds, 
            method='predict_proba',
            n_jobs=-1
        )[:, 1]  # Probability of fraud class
        
        # FraudBoost OOF predictions - more complex due to amounts
        print("FraudBoost OOF predictions...")
        fb_oof = self._get_fraudboost_oof(X_train, y_train, amounts_train)
        
        # Step 2: Create meta-features
        print("Creating meta-features...")
        meta_X = self._create_meta_features(xgb_oof, fb_oof, amounts_train)
        
        print(f"Meta-features shape: {meta_X.shape}")
        print(f"Meta-features: {self.meta_feature_names}")
        
        # Step 3: Train meta-learner
        print("Training meta-learner...")
        self.meta_model = LogisticRegression(**self.meta_params)
        self.meta_model.fit(meta_X, y_train)
        
        # Print meta-learner coefficients for insight
        coeffs = self.meta_model.coef_[0]
        print("Meta-learner coefficients:")
        for name, coeff in zip(self.meta_feature_names, coeffs):
            print(f"  {name}: {coeff:.4f}")
        print(f"  intercept: {self.meta_model.intercept_[0]:.4f}")
        
        # Step 4: Refit base models on full training data
        print("\n=== Training final base models ===")
        
        # Final XGBoost model
        print("Training final XGBoost...")
        self.xgb_model = xgb.XGBClassifier(**self.xgb_params)
        self.xgb_model.fit(X_train, y_train, verbose=False)
        
        # Final FraudBoost model  
        print("Training final FraudBoost...")
        self.fb_model = FraudBoostClassifier(**self.fb_params)
        self.fb_model.fit(X_train, y_train, amounts_train)
        
        self.is_fitted = True
        print("\n=== Stacking Training Complete ===")
        return self
    
    def _get_fraudboost_oof(self, X, y, amounts):
        """
        Generate out-of-fold predictions for FraudBoost using manual CV.
        We need manual CV because FraudBoost requires amounts parameter.
        """
        from sklearn.model_selection import StratifiedKFold
        
        n_samples = len(X)
        oof_predictions = np.zeros(n_samples)
        
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"  Fold {fold + 1}/{self.cv_folds}")
            
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]  
            amounts_fold_train, amounts_fold_val = amounts[train_idx], amounts[val_idx]
            
            # Train FraudBoost on fold training data
            fb_fold = FraudBoostClassifier(**self.fb_params)
            fb_fold.fit(X_fold_train, y_fold_train, amounts_fold_train)
            
            # Predict on fold validation data
            fold_pred = fb_fold.predict_proba(X_fold_val)[:, 1]
            oof_predictions[val_idx] = fold_pred
            
        return oof_predictions
    
    def _create_meta_features(self, xgb_proba, fb_proba, amounts):
        """Create meta-features for the meta-learner."""
        amounts_log = np.log1p(amounts)  # log(1 + amount) to handle 0s
        
        meta_features = np.column_stack([
            xgb_proba,
            fb_proba, 
            amounts,
            amounts_log
        ])
        
        return meta_features
    
    def predict_proba(self, X, amounts=None):
        """
        Predict fraud probabilities using stacking ensemble.
        
        Process:
        1. Get predictions from both base models
        2. Create meta-features
        3. Use meta-learner to combine them
        
        Parameters:
        -----------
        X : array-like
            Features
        amounts : array-like
            Transaction amounts (required for FraudBoost)
            
        Returns:
        --------
        probabilities : array 
            Stacking ensemble fraud probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        if amounts is None:
            raise ValueError("amounts parameter is required for stacking ensemble")
        
        # Get base model predictions
        xgb_proba = self.xgb_model.predict_proba(X)[:, 1]
        fb_proba = self.fb_model.predict_proba(X)[:, 1]
        
        # Create meta-features
        meta_X = self._create_meta_features(xgb_proba, fb_proba, amounts)
        
        # Meta-learner prediction
        stacking_proba = self.meta_model.predict_proba(meta_X)[:, 1]
        
        return stacking_proba
    
    def predict(self, X, amounts=None, threshold=0.5):
        """
        Predict fraud labels using stacking ensemble.
        
        Parameters:
        -----------
        X : array-like
            Features
        amounts : array-like
            Transaction amounts
        threshold : float, default=0.5
            Classification threshold
            
        Returns:
        --------
        predictions : array
            Binary fraud predictions
        """
        probas = self.predict_proba(X, amounts)
        return (probas >= threshold).astype(int)
    
    def evaluate(self, X, y, amounts, threshold=0.5):
        """
        Comprehensive evaluation of the stacking ensemble.
        
        Parameters:
        -----------
        X, y, amounts : array-like
            Test features, labels, and amounts
        threshold : float, default=0.5
            Classification threshold
            
        Returns:
        --------
        results : dict
            Comprehensive metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        # Predictions
        stacking_proba = self.predict_proba(X, amounts)
        stacking_pred = (stacking_proba >= threshold).astype(int)
        
        # Base model predictions for comparison
        xgb_proba = self.xgb_model.predict_proba(X)[:, 1]
        fb_proba = self.fb_model.predict_proba(X)[:, 1]
        
        # Metrics
        results = {}
        
        # AUC
        if len(np.unique(stacking_pred)) > 1:
            results['auc'] = roc_auc_score(y, stacking_proba)
        else:
            results['auc'] = 0.5
            
        # Precision, Recall, F1
        prec, recall, f1 = calculate_precision_recall_f1(y, stacking_pred)
        results.update({
            'precision': prec,
            'recall': recall,
            'f1': f1
        })
        
        # Confusion matrix components
        results.update({
            'true_positives': np.sum((y == 1) & (stacking_pred == 1)),
            'false_positives': np.sum((y == 0) & (stacking_pred == 1)),
            'false_negatives': np.sum((y == 1) & (stacking_pred == 0)),
            'true_negatives': np.sum((y == 0) & (stacking_pred == 0))
        })
        
        # Value-based metrics
        vdr = calculate_value_detection_rate(y, stacking_pred, amounts)
        net_savings = calculate_net_savings(y, stacking_pred, amounts, self.fp_cost)
        
        results.update({
            'vdr': vdr,
            'net_savings': net_savings
        })
        
        # Base model AUCs for comparison
        results['base_aucs'] = {
            'xgb_auc': roc_auc_score(y, xgb_proba),
            'fb_auc': roc_auc_score(y, fb_proba)
        }
        
        return results
    
    def get_feature_importance(self):
        """
        Get meta-learner coefficients as feature importance.
        Positive coefficients increase fraud probability.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")
            
        coeffs = self.meta_model.coef_[0]
        importance = {}
        
        for name, coeff in zip(self.meta_feature_names, coeffs):
            importance[name] = coeff
            
        importance['intercept'] = self.meta_model.intercept_[0]
        
        return importance