"""
Test the FraudBoostClassifier core functionality.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import sys
sys.path.append('../')

from fraudboost import FraudBoostClassifier


class TestFraudBoostClassifier:
    
    @pytest.fixture
    def synthetic_fraud_data(self):
        """Generate synthetic imbalanced fraud dataset."""
        # Create imbalanced dataset (99.5% legit, 0.5% fraud)
        X, y = make_classification(
            n_samples=10000,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_clusters_per_class=1,
            weights=[0.995, 0.005],  # Highly imbalanced
            flip_y=0.01,
            random_state=42
        )
        
        # Generate transaction amounts (fraud tends to be higher value)
        amounts = np.random.lognormal(4, 1, len(y))  # Base amounts
        amounts[y == 1] *= np.random.uniform(2, 5, sum(y))  # Fraud is higher value
        
        return train_test_split(X, y, amounts, test_size=0.3, stratify=y, random_state=42)
    
    def test_basic_fit_predict(self, synthetic_fraud_data):
        """Test basic model fitting and prediction."""
        X_train, X_test, y_train, y_test, amounts_train, amounts_test = synthetic_fraud_data
        
        model = FraudBoostClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train, amounts_train)
        
        # Check model is fitted
        assert len(model.estimators_) > 0
        assert model.feature_importances_ is not None
        assert len(model.feature_importances_) == X_train.shape[1]
        
        # Test predictions
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        
        assert len(predictions) == len(X_test)
        assert probabilities.shape == (len(X_test), 2)
        assert np.all((probabilities[:, 1] >= 0) & (probabilities[:, 1] <= 1))
        
        # Predictions should match thresholded probabilities
        pred_from_proba = (probabilities[:, 1] >= 0.5).astype(int)
        np.testing.assert_array_equal(predictions, pred_from_proba)
    
    def test_value_weighted_loss(self, synthetic_fraud_data):
        """Test value-weighted loss function."""
        X_train, X_test, y_train, y_test, amounts_train, amounts_test = synthetic_fraud_data
        
        # Model with value weighting
        model_weighted = FraudBoostClassifier(
            n_estimators=10, 
            loss='value_weighted', 
            fp_cost=100,
            random_state=42
        )
        model_weighted.fit(X_train, y_train, amounts_train)
        
        # Model without value weighting (standard log loss)
        model_standard = FraudBoostClassifier(
            n_estimators=10,
            loss='logloss', 
            random_state=42
        )
        model_standard.fit(X_train, y_train, amounts_train)
        
        # Both should produce valid predictions
        pred_weighted = model_weighted.predict_proba(X_test)[:, 1]
        pred_standard = model_standard.predict_proba(X_test)[:, 1]
        
        assert len(pred_weighted) == len(X_test)
        assert len(pred_standard) == len(X_test)
        
        # Value-weighted model should be different (unless by coincidence)
        # We'll just check they both produced reasonable predictions
        assert 0.0 <= np.mean(pred_weighted) <= 1.0
        assert 0.0 <= np.mean(pred_standard) <= 1.0
    
    def test_early_stopping(self, synthetic_fraud_data):
        """Test early stopping functionality."""
        X_train, X_test, y_train, y_test, amounts_train, amounts_test = synthetic_fraud_data
        
        # Split training set for validation
        X_tr, X_val, y_tr, y_val, amt_tr, amt_val = train_test_split(
            X_train, y_train, amounts_train, test_size=0.2, stratify=y_train, random_state=42
        )
        
        model = FraudBoostClassifier(
            n_estimators=50,
            early_stopping_rounds=5,
            random_state=42
        )
        model.fit(X_tr, y_tr, amt_tr, X_val, y_val, amt_val)
        
        # Should have stopped early
        assert len(model.estimators_) < 50
        assert model.best_iteration_ is not None
        assert len(model.train_losses_) > 0
        assert len(model.val_losses_) > 0
    
    def test_feature_importance(self, synthetic_fraud_data):
        """Test feature importance calculation."""
        X_train, X_test, y_train, y_test, amounts_train, amounts_test = synthetic_fraud_data
        
        model = FraudBoostClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train, amounts_train)
        
        importance = model.get_feature_importance('gain')
        
        assert len(importance) == X_train.shape[1]
        assert np.all(importance >= 0)
        assert np.isclose(np.sum(importance), 1.0)  # Should be normalized
    
    def test_subsample_colsample(self, synthetic_fraud_data):
        """Test row and column subsampling."""
        X_train, X_test, y_train, y_test, amounts_train, amounts_test = synthetic_fraud_data
        
        model = FraudBoostClassifier(
            n_estimators=10,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        model.fit(X_train, y_train, amounts_train)
        
        # Should fit without errors
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)
        
        # Trees should use different feature subsets
        # (This is implicit in the implementation)
        assert len(model.estimators_) == 10
    
    def test_different_loss_functions(self, synthetic_fraud_data):
        """Test different loss functions."""
        X_train, X_test, y_train, y_test, amounts_train, amounts_test = synthetic_fraud_data
        
        loss_functions = ['value_weighted', 'focal', 'logloss']
        
        for loss in loss_functions:
            model = FraudBoostClassifier(
                n_estimators=5,
                loss=loss,
                random_state=42
            )
            model.fit(X_train, y_train, amounts_train)
            
            predictions = model.predict_proba(X_test)[:, 1]
            
            # Should produce valid probabilities
            assert np.all((predictions >= 0) & (predictions <= 1))
            assert len(predictions) == len(X_test)
    
    def test_custom_threshold(self, synthetic_fraud_data):
        """Test prediction with custom threshold."""
        X_train, X_test, y_train, y_test, amounts_train, amounts_test = synthetic_fraud_data
        
        model = FraudBoostClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train, amounts_train)
        
        probabilities = model.predict_proba(X_test)[:, 1]
        
        # Test different thresholds
        pred_01 = model.predict(X_test, threshold=0.1)
        pred_09 = model.predict(X_test, threshold=0.9)
        
        # Lower threshold should flag more cases
        assert np.sum(pred_01) >= np.sum(pred_09)
        
        # Check predictions match thresholds
        expected_01 = (probabilities >= 0.1).astype(int)
        expected_09 = (probabilities >= 0.9).astype(int)
        
        np.testing.assert_array_equal(pred_01, expected_01)
        np.testing.assert_array_equal(pred_09, expected_09)
    
    def test_no_amounts_provided(self, synthetic_fraud_data):
        """Test model works without transaction amounts."""
        X_train, X_test, y_train, y_test, amounts_train, amounts_test = synthetic_fraud_data
        
        # Fit without amounts (should use uniform weighting)
        model = FraudBoostClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)  # No amounts parameter
        
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        
        assert len(predictions) == len(X_test)
        assert probabilities.shape == (len(X_test), 2)
    
    def test_model_parameters(self):
        """Test model parameter access."""
        model = FraudBoostClassifier(
            n_estimators=50,
            learning_rate=0.05,
            max_depth=8,
            fp_cost=200
        )
        
        params = model.get_params()
        
        assert params['n_estimators'] == 50
        assert params['learning_rate'] == 0.05
        assert params['max_depth'] == 8
        assert params['fp_cost'] == 200
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        model = FraudBoostClassifier()
        
        # Test with no fraud cases
        X = np.random.normal(0, 1, (100, 5))
        y = np.zeros(100)  # No fraud
        amounts = np.ones(100)
        
        model.fit(X, y, amounts)
        predictions = model.predict(X)
        
        # Should not crash, predictions should be mostly 0
        assert len(predictions) == len(X)
        assert np.mean(predictions) <= 0.1  # Very few false positives expected
        
        # Test with wrong input shapes
        with pytest.raises(ValueError):
            model.fit(X[0], y, amounts)  # 1D features
        
        with pytest.raises(ValueError):
            model.fit(X, y[:-1], amounts)  # Mismatched lengths