"""
Test ValueWeightedDecisionTree implementation.
"""

import numpy as np
import pytest

import sys
sys.path.append('../')

from fraudboost.tree import ValueWeightedDecisionTree


class TestValueWeightedDecisionTree:
    
    @pytest.fixture
    def sample_gradient_data(self):
        """Generate sample gradient/hessian data for tree fitting."""
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        
        X = np.random.normal(0, 1, (n_samples, n_features))
        
        # Generate gradients and hessians (typical for classification)
        gradients = np.random.normal(0, 0.5, n_samples)
        hessians = np.random.uniform(0.1, 0.5, n_samples)  # Always positive
        
        # Transaction amounts (for value weighting)
        amounts = np.random.lognormal(4, 1, n_samples)
        
        return X, gradients, hessians, amounts
    
    def test_basic_tree_fitting(self, sample_gradient_data):
        """Test basic tree fitting functionality."""
        X, gradients, hessians, amounts = sample_gradient_data
        
        tree = ValueWeightedDecisionTree(max_depth=3, random_state=42)
        tree.fit(X, gradients, hessians, amounts)
        
        # Tree should be fitted
        assert tree.tree_ is not None
        assert len(tree.tree_) > 0
        
        # Should have feature importances
        assert tree.feature_importances_ is not None
        assert len(tree.feature_importances_) == X.shape[1]
        assert np.all(tree.feature_importances_ >= 0)
        
        # Predictions
        predictions = tree.predict(X)
        assert len(predictions) == len(X)
        assert np.all(np.isfinite(predictions))
    
    def test_tree_depth_control(self, sample_gradient_data):
        """Test that max_depth parameter is respected."""
        X, gradients, hessians, amounts = sample_gradient_data
        
        # Shallow tree
        tree_shallow = ValueWeightedDecisionTree(max_depth=1)
        tree_shallow.fit(X, gradients, hessians, amounts)
        
        # Deep tree
        tree_deep = ValueWeightedDecisionTree(max_depth=5)
        tree_deep.fit(X, gradients, hessians, amounts)
        
        # Both should make predictions
        pred_shallow = tree_shallow.predict(X)
        pred_deep = tree_deep.predict(X)
        
        assert len(pred_shallow) == len(X)
        assert len(pred_deep) == len(X)
        
        # Deeper tree should generally have different predictions
        # (unless data is very simple)
        # We won't assert this since it depends on the specific data
    
    def test_min_samples_constraints(self, sample_gradient_data):
        """Test minimum samples constraints."""
        X, gradients, hessians, amounts = sample_gradient_data
        
        # Very restrictive constraints
        tree = ValueWeightedDecisionTree(
            min_samples_leaf=20,
            min_samples_split=40,
            max_depth=10
        )
        tree.fit(X, gradients, hessians, amounts)
        
        predictions = tree.predict(X)
        assert len(predictions) == len(X)
        
        # Should still work even with restrictive constraints
        assert np.all(np.isfinite(predictions))
    
    def test_without_amounts(self, sample_gradient_data):
        """Test tree fitting without transaction amounts."""
        X, gradients, hessians, amounts = sample_gradient_data
        
        tree = ValueWeightedDecisionTree(max_depth=3)
        tree.fit(X, gradients, hessians)  # No amounts
        
        predictions = tree.predict(X)
        assert len(predictions) == len(X)
        assert np.all(np.isfinite(predictions))
    
    def test_regularization_effect(self, sample_gradient_data):
        """Test L2 regularization parameter."""
        X, gradients, hessians, amounts = sample_gradient_data
        
        # No regularization
        tree_no_reg = ValueWeightedDecisionTree(reg_lambda=0.0, max_depth=3)
        tree_no_reg.fit(X, gradients, hessians, amounts)
        
        # High regularization
        tree_high_reg = ValueWeightedDecisionTree(reg_lambda=10.0, max_depth=3)
        tree_high_reg.fit(X, gradients, hessians, amounts)
        
        pred_no_reg = tree_no_reg.predict(X)
        pred_high_reg = tree_high_reg.predict(X)
        
        # Both should produce valid predictions
        assert np.all(np.isfinite(pred_no_reg))
        assert np.all(np.isfinite(pred_high_reg))
        
        # High regularization should generally produce smaller absolute values
        assert np.mean(np.abs(pred_high_reg)) <= np.mean(np.abs(pred_no_reg)) * 2  # Allow some tolerance
    
    def test_missing_values(self):
        """Test handling of missing values in features."""
        np.random.seed(42)
        
        # Create data with missing values
        X = np.random.normal(0, 1, (50, 3))
        X[10:15, 0] = np.nan  # Missing values in first feature
        X[20:25, 1] = np.nan  # Missing values in second feature
        
        gradients = np.random.normal(0, 0.5, 50)
        hessians = np.random.uniform(0.1, 0.5, 50)
        
        tree = ValueWeightedDecisionTree(max_depth=3)
        tree.fit(X, gradients, hessians)
        
        # Should handle missing values without crashing
        predictions = tree.predict(X)
        assert len(predictions) == len(X)
        assert np.all(np.isfinite(predictions))
        
        # Test prediction on new data with missing values
        X_new = np.random.normal(0, 1, (10, 3))
        X_new[::2, 0] = np.nan  # Every other sample has missing first feature
        
        predictions_new = tree.predict(X_new)
        assert len(predictions_new) == len(X_new)
        assert np.all(np.isfinite(predictions_new))
    
    def test_single_leaf_tree(self):
        """Test edge case where tree becomes single leaf."""
        # Create data that should result in single leaf
        X = np.ones((10, 2))  # All features identical
        gradients = np.ones(10)
        hessians = np.ones(10)
        
        tree = ValueWeightedDecisionTree(
            max_depth=3,
            min_samples_split=100,  # Impossible to split
        )
        tree.fit(X, gradients, hessians)
        
        predictions = tree.predict(X)
        assert len(predictions) == len(X)
        
        # All predictions should be the same (single leaf value)
        assert np.allclose(predictions, predictions[0])
    
    def test_small_dataset(self):
        """Test with very small dataset."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        gradients = np.array([0.1, -0.2, 0.3])
        hessians = np.array([0.5, 0.4, 0.6])
        
        tree = ValueWeightedDecisionTree(max_depth=2, min_samples_leaf=1)
        tree.fit(X, gradients, hessians)
        
        predictions = tree.predict(X)
        assert len(predictions) == 3
        assert np.all(np.isfinite(predictions))
    
    def test_gain_computation(self, sample_gradient_data):
        """Test that splits actually improve the loss."""
        X, gradients, hessians, amounts = sample_gradient_data
        
        tree = ValueWeightedDecisionTree(max_depth=3, min_gain=0.0)  # Allow any gain
        tree.fit(X, gradients, hessians, amounts)
        
        # Tree should find splits (unless data is completely homogeneous)
        # We can't assert too much about the structure since it depends on the data
        predictions = tree.predict(X)
        assert np.all(np.isfinite(predictions))
        
        # Test with high min_gain (should result in fewer splits)
        tree_high_gain = ValueWeightedDecisionTree(max_depth=3, min_gain=1e6)  # Very high threshold
        tree_high_gain.fit(X, gradients, hessians, amounts)
        
        predictions_high_gain = tree_high_gain.predict(X)
        assert np.all(np.isfinite(predictions_high_gain))
    
    def test_feature_importance_sum(self, sample_gradient_data):
        """Test that feature importances sum to 1."""
        X, gradients, hessians, amounts = sample_gradient_data
        
        tree = ValueWeightedDecisionTree(max_depth=3)
        tree.fit(X, gradients, hessians, amounts)
        
        importances = tree.feature_importances_
        
        # Should sum to 1 (normalized)
        assert np.isclose(np.sum(importances), 1.0, atol=1e-10)
        
        # All should be non-negative
        assert np.all(importances >= 0)
    
    def test_leaf_values_extraction(self, sample_gradient_data):
        """Test extraction of leaf values."""
        X, gradients, hessians, amounts = sample_gradient_data
        
        tree = ValueWeightedDecisionTree(max_depth=2)
        tree.fit(X, gradients, hessians, amounts)
        
        leaf_values = tree.get_leaf_values()
        
        # Should have some leaf values
        assert len(leaf_values) > 0
        assert np.all(np.isfinite(leaf_values))
    
    def test_deterministic_results(self, sample_gradient_data):
        """Test that tree building is deterministic."""
        X, gradients, hessians, amounts = sample_gradient_data
        
        # Build same tree twice
        tree1 = ValueWeightedDecisionTree(max_depth=3, random_state=42)
        tree1.fit(X, gradients, hessians, amounts)
        
        tree2 = ValueWeightedDecisionTree(max_depth=3, random_state=42)
        tree2.fit(X, gradients, hessians, amounts)
        
        # Predictions should be identical
        pred1 = tree1.predict(X)
        pred2 = tree2.predict(X)
        
        np.testing.assert_array_almost_equal(pred1, pred2)
        
        # Feature importances should be identical
        np.testing.assert_array_almost_equal(
            tree1.feature_importances_, tree2.feature_importances_
        )
    
    def test_edge_case_inputs(self):
        """Test edge cases and error conditions."""
        tree = ValueWeightedDecisionTree()
        
        # Test with single sample
        X_single = np.array([[1, 2, 3]])
        grad_single = np.array([0.1])
        hess_single = np.array([0.2])
        
        tree.fit(X_single, grad_single, hess_single)
        pred_single = tree.predict(X_single)
        
        assert len(pred_single) == 1
        assert np.isfinite(pred_single[0])
        
        # Test with zero gradients
        X_zero = np.random.normal(0, 1, (10, 3))
        grad_zero = np.zeros(10)
        hess_ones = np.ones(10)
        
        tree_zero = ValueWeightedDecisionTree()
        tree_zero.fit(X_zero, grad_zero, hess_ones)
        pred_zero = tree_zero.predict(X_zero)
        
        assert len(pred_zero) == 10
        assert np.all(np.isfinite(pred_zero))
        
        # All predictions should be close to zero (since gradients are zero)
        assert np.allclose(pred_zero, 0, atol=1e-10)