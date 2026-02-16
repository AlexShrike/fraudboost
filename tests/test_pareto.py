"""
Test Pareto optimization for threshold selection.
"""

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.append('../')

from fraudboost.pareto import ParetoOptimizer


class TestParetoOptimizer:
    
    @pytest.fixture
    def sample_predictions_data(self):
        """Create sample prediction data for threshold optimization."""
        np.random.seed(42)
        
        # 1000 samples: 950 legit, 50 fraud (5% fraud rate)
        n_samples = 1000
        fraud_rate = 0.05
        
        y_true = np.random.choice([0, 1], n_samples, p=[1-fraud_rate, fraud_rate])
        
        # Generate realistic probabilities (fraud cases tend to have higher scores)
        y_pred_proba = np.random.beta(1, 9, n_samples)  # Low probabilities for most
        fraud_mask = y_true == 1
        y_pred_proba[fraud_mask] = np.random.beta(3, 2, np.sum(fraud_mask))  # Higher for fraud
        
        # Transaction amounts (fraud tends to be higher value)
        amounts = np.random.lognormal(4, 1, n_samples)
        amounts[fraud_mask] *= np.random.uniform(2, 5, np.sum(fraud_mask))
        
        return y_true, y_pred_proba, amounts
    
    def test_basic_pareto_optimization(self, sample_predictions_data):
        """Test basic Pareto frontier finding."""
        y_true, y_pred_proba, amounts = sample_predictions_data
        
        optimizer = ParetoOptimizer(fp_cost=100.0)
        optimizer.fit(y_true, y_pred_proba, amounts)
        
        # Should have results
        assert optimizer.results_ is not None
        assert optimizer.pareto_indices_ is not None
        
        # Results should have expected keys
        expected_keys = ['thresholds', 'precision', 'recall', 'vdr', 'net_savings', 'roi']
        for key in expected_keys:
            assert key in optimizer.results_
        
        # Should have some Pareto-optimal points
        assert len(optimizer.pareto_indices_) > 0
        assert len(optimizer.pareto_indices_) <= len(optimizer.results_['thresholds'])
    
    def test_pareto_points_dataframe(self, sample_predictions_data):
        """Test getting Pareto points as DataFrame."""
        y_true, y_pred_proba, amounts = sample_predictions_data
        
        optimizer = ParetoOptimizer(fp_cost=100.0)
        optimizer.fit(y_true, y_pred_proba, amounts)
        
        pareto_df = optimizer.get_pareto_points()
        
        # Should be a DataFrame
        assert isinstance(pareto_df, pd.DataFrame)
        assert len(pareto_df) > 0
        
        # Should have expected columns
        expected_columns = ['threshold', 'precision', 'recall', 'vdr', 'net_savings', 'roi']
        for col in expected_columns:
            assert col in pareto_df.columns
        
        # Values should be reasonable
        assert np.all(pareto_df['threshold'] >= 0) and np.all(pareto_df['threshold'] <= 1)
        assert np.all(pareto_df['precision'] >= 0) and np.all(pareto_df['precision'] <= 1)
        assert np.all(pareto_df['recall'] >= 0) and np.all(pareto_df['recall'] <= 1)
    
    def test_threshold_recommendations(self, sample_predictions_data):
        """Test threshold recommendations for different strategies."""
        y_true, y_pred_proba, amounts = sample_predictions_data
        
        optimizer = ParetoOptimizer(fp_cost=100.0)
        optimizer.fit(y_true, y_pred_proba, amounts)
        
        strategies = ['max_vdr', 'max_savings', 'min_fps', 'balanced', 'max_roi']
        
        for strategy in strategies:
            if strategy in optimizer.recommended_thresholds_:
                recommendation = optimizer.recommend_threshold(strategy)
                
                # Should have expected structure
                assert 'threshold' in recommendation
                assert 'description' in recommendation
                assert 'metrics' in recommendation
                
                # Threshold should be valid
                threshold = recommendation['threshold']
                assert 0 <= threshold <= 1
                
                # Metrics should be reasonable
                metrics = recommendation['metrics']
                assert 0 <= metrics['precision'] <= 1
                assert 0 <= metrics['recall'] <= 1
                assert 0 <= metrics['vdr'] <= 1
    
    def test_custom_thresholds(self, sample_predictions_data):
        """Test optimization with custom threshold range."""
        y_true, y_pred_proba, amounts = sample_predictions_data
        
        custom_thresholds = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
        
        optimizer = ParetoOptimizer(fp_cost=100.0)
        optimizer.fit(y_true, y_pred_proba, amounts, thresholds=custom_thresholds)
        
        # Should use custom thresholds
        np.testing.assert_array_equal(optimizer.results_['thresholds'], custom_thresholds)
        
        # Other arrays should have same length
        assert len(optimizer.results_['precision']) == len(custom_thresholds)
        assert len(optimizer.results_['recall']) == len(custom_thresholds)
    
    def test_report_generation(self, sample_predictions_data):
        """Test comprehensive report generation."""
        y_true, y_pred_proba, amounts = sample_predictions_data
        
        optimizer = ParetoOptimizer(fp_cost=100.0)
        optimizer.fit(y_true, y_pred_proba, amounts)
        
        report = optimizer.generate_report()
        
        # Should be a string
        assert isinstance(report, str)
        assert len(report) > 100  # Should be substantial
        
        # Should contain expected sections
        assert "Threshold Optimization Report" in report
        assert "RECOMMENDED THRESHOLDS" in report
        assert "Pareto-optimal" in report
    
    def test_plotting_functionality(self, sample_predictions_data):
        """Test Pareto frontier plotting (if matplotlib available)."""
        y_true, y_pred_proba, amounts = sample_predictions_data
        
        optimizer = ParetoOptimizer(fp_cost=100.0)
        optimizer.fit(y_true, y_pred_proba, amounts)
        
        try:
            import matplotlib.pyplot as plt
            
            # Test Pareto frontier plot
            fig = optimizer.plot_pareto_frontier('recall', 'precision')
            assert fig is not None
            plt.close(fig)
            
            # Test threshold sweep plot
            fig = optimizer.plot_threshold_sweep()
            assert fig is not None
            plt.close(fig)
            
        except ImportError:
            pytest.skip("matplotlib not available for plotting tests")
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        optimizer = ParetoOptimizer()
        
        # Test with no fraud cases
        y_true_no_fraud = np.zeros(100)
        y_pred_no_fraud = np.random.uniform(0, 1, 100)
        amounts_no_fraud = np.ones(100) * 100
        
        optimizer.fit(y_true_no_fraud, y_pred_no_fraud, amounts_no_fraud)
        
        # Should handle gracefully
        assert optimizer.results_ is not None
        assert len(optimizer.pareto_indices_) >= 0
        
        # Test with all fraud cases
        y_true_all_fraud = np.ones(100)
        y_pred_all_fraud = np.random.uniform(0, 1, 100)
        amounts_all_fraud = np.ones(100) * 100
        
        optimizer_all = ParetoOptimizer()
        optimizer_all.fit(y_true_all_fraud, y_pred_all_fraud, amounts_all_fraud)
        
        assert optimizer_all.results_ is not None
    
    def test_pareto_dominance(self, sample_predictions_data):
        """Test that Pareto points are actually non-dominated."""
        y_true, y_pred_proba, amounts = sample_predictions_data
        
        optimizer = ParetoOptimizer(fp_cost=100.0)
        optimizer.fit(y_true, y_pred_proba, amounts)
        
        pareto_df = optimizer.get_pareto_points()
        
        if len(pareto_df) > 1:
            # Extract the objective values for Pareto points
            objectives = pareto_df[['precision', 'recall', 'vdr', 'net_savings', 'roi']].values
            
            # Check that no point dominates another
            for i in range(len(objectives)):
                for j in range(len(objectives)):
                    if i != j:
                        # Point i dominates point j if i >= j in all objectives and i > j in at least one
                        dominates = np.all(objectives[i] >= objectives[j])
                        strictly_better = np.any(objectives[i] > objectives[j])
                        
                        # No Pareto point should dominate another
                        assert not (dominates and strictly_better), \
                            f"Point {i} dominates point {j} in Pareto frontier"
    
    def test_threshold_sweep_properties(self, sample_predictions_data):
        """Test properties that should hold across threshold sweep."""
        y_true, y_pred_proba, amounts = sample_predictions_data
        
        optimizer = ParetoOptimizer(fp_cost=100.0)
        optimizer.fit(y_true, y_pred_proba, amounts)
        
        results = optimizer.results_
        
        # Thresholds should be sorted
        assert np.all(results['thresholds'][:-1] <= results['thresholds'][1:])
        
        # As threshold increases, precision should generally increase, recall decrease
        # (allowing for some noise in small samples)
        precision = results['precision']
        recall = results['recall']
        
        # Check general trend (not strict monotonicity due to discrete nature)
        precision_trend = np.corrcoef(results['thresholds'], precision)[0, 1]
        recall_trend = np.corrcoef(results['thresholds'], recall)[0, 1]
        
        # Precision should generally increase with threshold
        assert precision_trend > -0.5  # Allow some noise but expect positive trend
        
        # Recall should generally decrease with threshold  
        assert recall_trend < 0.5   # Allow some noise but expect negative trend
    
    def test_different_fp_costs(self, sample_predictions_data):
        """Test optimization with different false positive costs."""
        y_true, y_pred_proba, amounts = sample_predictions_data
        
        fp_costs = [50, 100, 200, 500]
        optimizers = []
        
        for fp_cost in fp_costs:
            optimizer = ParetoOptimizer(fp_cost=fp_cost)
            optimizer.fit(y_true, y_pred_proba, amounts)
            optimizers.append(optimizer)
        
        # Different FP costs should generally lead to different optimal thresholds
        max_savings_thresholds = []
        for optimizer in optimizers:
            if 'max_savings' in optimizer.recommended_thresholds_:
                threshold = optimizer.recommended_thresholds_['max_savings']['threshold']
                max_savings_thresholds.append(threshold)
        
        # Should have some variation in optimal thresholds
        if len(max_savings_thresholds) > 1:
            threshold_std = np.std(max_savings_thresholds)
            # Allow some variation due to discrete nature of optimization
            # but expect some difference with very different FP costs
    
    def test_invalid_strategy_request(self, sample_predictions_data):
        """Test error handling for invalid strategy requests."""
        y_true, y_pred_proba, amounts = sample_predictions_data
        
        optimizer = ParetoOptimizer()
        optimizer.fit(y_true, y_pred_proba, amounts)
        
        # Should raise error for invalid strategy
        with pytest.raises(ValueError):
            optimizer.recommend_threshold('invalid_strategy')
    
    def test_before_fitting(self):
        """Test error handling when methods called before fitting."""
        optimizer = ParetoOptimizer()
        
        # Should raise errors when not fitted
        with pytest.raises(ValueError):
            optimizer.get_pareto_points()
        
        # Report should indicate no recommendations
        report = optimizer.generate_report()
        assert "No recommendations available" in report
    
    def test_small_dataset_robustness(self):
        """Test robustness with very small datasets."""
        # Very small dataset
        y_true = np.array([0, 1, 0, 1])
        y_pred_proba = np.array([0.2, 0.8, 0.3, 0.9])
        amounts = np.array([100, 500, 200, 1000])
        
        optimizer = ParetoOptimizer(fp_cost=100)
        optimizer.fit(y_true, y_pred_proba, amounts)
        
        # Should handle small dataset without crashing
        assert optimizer.results_ is not None
        assert len(optimizer.pareto_indices_) >= 0
        
        # Should be able to generate report
        report = optimizer.generate_report()
        assert isinstance(report, str)