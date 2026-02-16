"""
Test loss functions for gradient boosting.
"""

import numpy as np
import pytest

import sys
sys.path.append('../')

from fraudboost.losses import (
    ValueWeightedLogLoss,
    FocalLoss,
    LogLoss,
    BaseLoss
)


class TestLossFunctions:
    
    @pytest.fixture
    def sample_binary_data(self):
        """Generate sample binary classification data."""
        np.random.seed(42)
        n_samples = 100
        
        y_true = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        y_pred = np.random.uniform(0.01, 0.99, n_samples)  # Avoid exact 0/1
        amounts = np.random.lognormal(4, 1, n_samples)
        
        return y_true, y_pred, amounts
    
    def test_value_weighted_log_loss_basic(self, sample_binary_data):
        """Test basic ValueWeightedLogLoss functionality."""
        y_true, y_pred, amounts = sample_binary_data
        
        loss_fn = ValueWeightedLogLoss(fp_cost=100.0)
        
        # Test loss computation
        loss_value = loss_fn(y_true, y_pred, amounts)
        assert np.isfinite(loss_value)
        assert loss_value > 0  # Log loss should be positive
        
        # Test gradient computation
        gradients = loss_fn.gradient(y_true, y_pred, amounts)
        assert len(gradients) == len(y_true)
        assert np.all(np.isfinite(gradients))
        
        # Test hessian computation
        hessians = loss_fn.hessian(y_true, y_pred, amounts)
        assert len(hessians) == len(y_true)
        assert np.all(np.isfinite(hessians))
        assert np.all(hessians > 0)  # Hessians should be positive for log loss
    
    def test_value_weighted_loss_without_amounts(self, sample_binary_data):
        """Test ValueWeightedLogLoss without transaction amounts."""
        y_true, y_pred, amounts = sample_binary_data
        
        loss_fn = ValueWeightedLogLoss(fp_cost=100.0)
        
        # Should work without amounts (uniform weighting)
        loss_value = loss_fn(y_true, y_pred)  # No amounts
        gradients = loss_fn.gradient(y_true, y_pred)
        hessians = loss_fn.hessian(y_true, y_pred)
        
        assert np.isfinite(loss_value)
        assert np.all(np.isfinite(gradients))
        assert np.all(np.isfinite(hessians))
    
    def test_focal_loss_basic(self, sample_binary_data):
        """Test basic FocalLoss functionality."""
        y_true, y_pred, amounts = sample_binary_data
        
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        
        loss_value = loss_fn(y_true, y_pred, amounts)
        gradients = loss_fn.gradient(y_true, y_pred, amounts)
        hessians = loss_fn.hessian(y_true, y_pred, amounts)
        
        assert np.isfinite(loss_value)
        assert loss_value > 0
        assert len(gradients) == len(y_true)
        assert np.all(np.isfinite(gradients))
        assert len(hessians) == len(y_true)
        assert np.all(np.isfinite(hessians))
    
    def test_standard_log_loss(self, sample_binary_data):
        """Test standard LogLoss implementation."""
        y_true, y_pred, amounts = sample_binary_data
        
        loss_fn = LogLoss()
        
        loss_value = loss_fn(y_true, y_pred)
        gradients = loss_fn.gradient(y_true, y_pred)
        hessians = loss_fn.hessian(y_true, y_pred)
        
        assert np.isfinite(loss_value)
        assert loss_value > 0
        assert np.all(np.isfinite(gradients))
        assert np.all(np.isfinite(hessians))
        assert np.all(hessians > 0)
    
    def test_gradient_hessian_consistency(self, sample_binary_data):
        """Test that gradients and hessians are mathematically consistent."""
        y_true, y_pred, amounts = sample_binary_data
        
        loss_functions = [
            ValueWeightedLogLoss(fp_cost=100),
            FocalLoss(alpha=0.25, gamma=2.0),
            LogLoss()
        ]
        
        for loss_fn in loss_functions:
            # Test gradient approximation using finite differences
            epsilon = 1e-7
            
            # Pick a few random indices to test
            test_indices = np.random.choice(len(y_pred), 5, replace=False)
            
            for i in test_indices:
                y_pred_eps = y_pred.copy()
                
                # Forward difference
                y_pred_eps[i] = y_pred[i] + epsilon
                if hasattr(loss_fn, '__call__'):
                    if isinstance(loss_fn, LogLoss):
                        loss_plus = loss_fn(y_true, y_pred_eps)
                    else:
                        loss_plus = loss_fn(y_true, y_pred_eps, amounts)
                else:
                    continue
                
                # Backward difference
                y_pred_eps[i] = y_pred[i] - epsilon
                if isinstance(loss_fn, LogLoss):
                    loss_minus = loss_fn(y_true, y_pred_eps)
                else:
                    loss_minus = loss_fn(y_true, y_pred_eps, amounts)
                
                # Numerical gradient
                numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)
                
                # Analytical gradient
                if isinstance(loss_fn, LogLoss):
                    analytical_grad = loss_fn.gradient(y_true, y_pred)[i]
                else:
                    analytical_grad = loss_fn.gradient(y_true, y_pred, amounts)[i]
                
                # Should be approximately equal (allowing for numerical errors)
                assert np.abs(numerical_grad - analytical_grad) < 1e-4, \
                    f"Gradient mismatch for {type(loss_fn).__name__} at index {i}"
    
    def test_edge_cases_extreme_predictions(self):
        """Test loss functions with extreme prediction values."""
        y_true = np.array([0, 1, 0, 1])
        amounts = np.array([100, 200, 300, 400])
        
        loss_functions = [
            ValueWeightedLogLoss(),
            FocalLoss(),
            LogLoss()
        ]
        
        # Test with predictions very close to 0 and 1
        extreme_predictions = [
            np.array([1e-10, 1-1e-10, 1e-10, 1-1e-10]),  # Very confident
            np.array([0.5, 0.5, 0.5, 0.5]),              # Very uncertain
        ]
        
        for loss_fn in loss_functions:
            for y_pred in extreme_predictions:
                try:
                    if isinstance(loss_fn, LogLoss):
                        loss_value = loss_fn(y_true, y_pred)
                        gradients = loss_fn.gradient(y_true, y_pred)
                        hessians = loss_fn.hessian(y_true, y_pred)
                    else:
                        loss_value = loss_fn(y_true, y_pred, amounts)
                        gradients = loss_fn.gradient(y_true, y_pred, amounts)
                        hessians = loss_fn.hessian(y_true, y_pred, amounts)
                    
                    # Should not produce NaN or infinite values
                    assert np.isfinite(loss_value), f"{type(loss_fn).__name__} produced non-finite loss"
                    assert np.all(np.isfinite(gradients)), f"{type(loss_fn).__name__} produced non-finite gradients"
                    assert np.all(np.isfinite(hessians)), f"{type(loss_fn).__name__} produced non-finite hessians"
                    
                except Exception as e:
                    pytest.fail(f"{type(loss_fn).__name__} failed with extreme predictions: {e}")
    
    def test_value_weighting_effect(self):
        """Test that value weighting actually affects loss calculation."""
        y_true = np.array([1, 1])  # Two fraud cases
        y_pred = np.array([0.5, 0.5])  # Same predictions
        
        # Different amounts
        amounts_small = np.array([100, 200])
        amounts_large = np.array([10000, 20000])
        
        loss_fn = ValueWeightedLogLoss(fp_cost=100)
        
        # Compute losses
        loss_small = loss_fn(y_true, y_pred, amounts_small)
        loss_large = loss_fn(y_true, y_pred, amounts_large)
        
        # Compute gradients
        grad_small = loss_fn.gradient(y_true, y_pred, amounts_small)
        grad_large = loss_fn.gradient(y_true, y_pred, amounts_large)
        
        # Larger amounts should generally lead to different (typically larger) gradients
        # since missing high-value fraud is more costly
        assert not np.allclose(grad_small, grad_large), \
            "Value weighting should affect gradients"
    
    def test_focal_loss_parameters(self):
        """Test FocalLoss with different parameter values."""
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
        amounts = np.array([100, 200, 300, 400, 500])
        
        # Test different alpha values
        for alpha in [0.1, 0.25, 0.5, 0.75]:
            loss_fn = FocalLoss(alpha=alpha, gamma=2.0)
            loss_value = loss_fn(y_true, y_pred, amounts)
            
            assert np.isfinite(loss_value)
            assert loss_value > 0
        
        # Test different gamma values
        for gamma in [0.5, 1.0, 2.0, 5.0]:
            loss_fn = FocalLoss(alpha=0.25, gamma=gamma)
            loss_value = loss_fn(y_true, y_pred, amounts)
            
            assert np.isfinite(loss_value)
            assert loss_value > 0
    
    def test_loss_function_with_sample_weights(self):
        """Test loss functions with sample weights."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0.2, 0.8, 0.3, 0.7])
        sample_weights = np.array([1.0, 2.0, 1.5, 0.5])
        
        # Standard log loss supports sample weights
        loss_fn = LogLoss()
        
        loss_unweighted = loss_fn(y_true, y_pred)
        loss_weighted = loss_fn(y_true, y_pred, sample_weights)
        
        # Weighted loss should be different
        assert not np.isclose(loss_unweighted, loss_weighted)
        
        grad_unweighted = loss_fn.gradient(y_true, y_pred)
        grad_weighted = loss_fn.gradient(y_true, y_pred, sample_weights)
        
        # Weighted gradients should be different
        assert not np.allclose(grad_unweighted, grad_weighted)
    
    def test_base_loss_abstract_class(self):
        """Test that BaseLoss is properly abstract."""
        # Should not be able to instantiate BaseLoss directly
        with pytest.raises(TypeError):
            BaseLoss()
    
    def test_loss_consistency_with_perfect_predictions(self):
        """Test loss behavior with perfect predictions."""
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred_perfect = y_true.astype(float)  # Perfect predictions
        
        # Adjust for numerical stability (avoid exact 0/1)
        y_pred_perfect[y_pred_perfect == 0] = 1e-15
        y_pred_perfect[y_pred_perfect == 1] = 1 - 1e-15
        
        amounts = np.array([100, 200, 300, 400, 500])
        
        loss_functions = [
            ValueWeightedLogLoss(),
            FocalLoss(),
            LogLoss()
        ]
        
        for loss_fn in loss_functions:
            if isinstance(loss_fn, LogLoss):
                loss_value = loss_fn(y_true, y_pred_perfect)
            else:
                loss_value = loss_fn(y_true, y_pred_perfect, amounts)
            
            # Perfect predictions should give very small loss
            assert loss_value < 0.1, f"{type(loss_fn).__name__} loss too high for perfect predictions"
    
    def test_loss_increases_with_wrong_predictions(self):
        """Test that loss increases as predictions get worse."""
        y_true = np.array([1, 1, 1, 1, 1])  # All positive
        amounts = np.array([100, 200, 300, 400, 500])
        
        # Predictions from good to bad
        prediction_sets = [
            np.array([0.9, 0.9, 0.9, 0.9, 0.9]),  # Good predictions
            np.array([0.7, 0.7, 0.7, 0.7, 0.7]),  # OK predictions
            np.array([0.3, 0.3, 0.3, 0.3, 0.3]),  # Bad predictions
            np.array([0.1, 0.1, 0.1, 0.1, 0.1]),  # Very bad predictions
        ]
        
        loss_functions = [
            ValueWeightedLogLoss(),
            LogLoss()
        ]
        
        for loss_fn in loss_functions:
            previous_loss = 0
            
            for y_pred in prediction_sets:
                if isinstance(loss_fn, LogLoss):
                    current_loss = loss_fn(y_true, y_pred)
                else:
                    current_loss = loss_fn(y_true, y_pred, amounts)
                
                # Loss should generally increase as predictions get worse
                assert current_loss >= previous_loss - 1e-10  # Allow small numerical errors
                previous_loss = current_loss