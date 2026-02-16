"""
Test fraud-specific metrics.
"""

import numpy as np
import pytest

import sys
sys.path.append('../')

from fraudboost.metrics import (
    value_detection_rate,
    net_savings,
    roi,
    precision_at_dollar_threshold,
    classification_report_fraud,
    evaluate_at_thresholds,
    cost_benefit_analysis
)


class TestFraudMetrics:
    
    @pytest.fixture
    def sample_fraud_data(self):
        """Create sample fraud detection results for testing."""
        # 10 transactions: 8 legit, 2 fraud
        y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
        
        # Predictions: catch 1 fraud, miss 1 fraud, 2 false positives  
        y_pred = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0])
        
        # Amounts: fraud cases are $1000 and $5000, legit cases are $10-100
        amounts = np.array([10, 20, 30, 40, 50, 60, 70, 80, 1000, 5000])
        
        return y_true, y_pred, amounts
    
    def test_value_detection_rate(self, sample_fraud_data):
        """Test Value Detection Rate calculation."""
        y_true, y_pred, amounts = sample_fraud_data
        
        vdr = value_detection_rate(y_true, y_pred, amounts)
        
        # Total fraud value: $1000 + $5000 = $6000
        # Detected fraud value: $1000 (caught the first fraud case)
        # VDR = $1000 / $6000 = 1/6 ≈ 0.167
        expected_vdr = 1000.0 / 6000.0
        assert np.isclose(vdr, expected_vdr)
        
        # Test edge case: no fraud
        y_true_no_fraud = np.array([0, 0, 0, 0])
        y_pred_no_fraud = np.array([0, 1, 0, 1])
        amounts_no_fraud = np.array([100, 200, 300, 400])
        
        vdr_no_fraud = value_detection_rate(y_true_no_fraud, y_pred_no_fraud, amounts_no_fraud)
        assert vdr_no_fraud == 0.0
        
        # Test perfect detection
        y_pred_perfect = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
        vdr_perfect = value_detection_rate(y_true, y_pred_perfect, amounts)
        assert vdr_perfect == 1.0
    
    def test_net_savings(self, sample_fraud_data):
        """Test net savings calculation."""
        y_true, y_pred, amounts = sample_fraud_data
        fp_cost = 100.0
        
        savings = net_savings(y_true, y_pred, amounts, fp_cost)
        
        # Fraud caught: $1000 (true positive)
        # False positives: 2 cases × $100 = $200
        # Net savings: $1000 - $200 = $800
        expected_savings = 1000.0 - (2 * 100.0)
        assert np.isclose(savings, expected_savings)
        
        # Test with different FP cost
        savings_high_fp = net_savings(y_true, y_pred, amounts, fp_cost=500.0)
        expected_high_fp = 1000.0 - (2 * 500.0)  # = -$500 (loss)
        assert np.isclose(savings_high_fp, expected_high_fp)
    
    def test_roi(self, sample_fraud_data):
        """Test ROI calculation."""
        y_true, y_pred, amounts = sample_fraud_data
        fp_cost = 100.0
        
        roi_value = roi(y_true, y_pred, amounts, fp_cost)
        
        # Total investigations: 3 (sum of y_pred)  
        # Investigation cost: 3 × $100 = $300
        # Fraud caught: $1000
        # Net savings: $1000 - $200 = $800 (2 FP × $100)
        # ROI = ($800 + $300) / $300 = $1100 / $300 ≈ 3.67
        total_investigations = np.sum(y_pred)
        investigation_cost = total_investigations * fp_cost
        fraud_caught = 1000.0  # Amount from the true positive
        net_savings_value = net_savings(y_true, y_pred, amounts, fp_cost)
        
        expected_roi = (net_savings_value + investigation_cost) / investigation_cost
        assert np.isclose(roi_value, expected_roi)
        
        # Test edge case: no investigations
        y_pred_none = np.zeros_like(y_pred)
        roi_none = roi(y_true, y_pred_none, amounts, fp_cost)
        assert roi_none == 0.0
    
    def test_precision_at_dollar_threshold(self):
        """Test precision calculation for high-value transactions only."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred_proba = np.array([0.2, 0.8, 0.3, 0.9, 0.1, 0.7])
        amounts = np.array([50, 100, 200, 500, 1000, 2000])
        
        # Only consider transactions >= $200
        precision_200 = precision_at_dollar_threshold(y_true, y_pred_proba, amounts, 200)
        
        # High-value transactions: indices 2, 3, 4, 5
        # y_true for these: [0, 1, 0, 1]
        # y_pred for these: [0, 1, 0, 1] (using 0.5 threshold)
        # Precision = 2 TP / (2 TP + 0 FP) = 1.0
        assert np.isclose(precision_200, 1.0)
        
        # Test with threshold higher than any transaction
        precision_high = precision_at_dollar_threshold(y_true, y_pred_proba, amounts, 10000)
        assert precision_high == 0.0
    
    def test_classification_report_fraud(self, sample_fraud_data):
        """Test comprehensive fraud classification report."""
        y_true, y_pred, amounts = sample_fraud_data
        fp_cost = 100.0
        
        # Test string output
        report_str = classification_report_fraud(y_true, y_pred, amounts, fp_cost)
        assert isinstance(report_str, str)
        assert "Value Detection Rate" in report_str
        assert "Net Savings" in report_str
        assert "ROI" in report_str
        
        # Test dictionary output
        report_dict = classification_report_fraud(y_true, y_pred, amounts, fp_cost, output_dict=True)
        assert isinstance(report_dict, dict)
        
        # Check required keys
        required_keys = [
            'precision', 'recall', 'f1_score', 'value_detection_rate',
            'net_savings', 'roi', 'true_positives', 'false_positives',
            'fraud_caught_dollars', 'fraud_missed_dollars'
        ]
        for key in required_keys:
            assert key in report_dict
        
        # Check values make sense
        assert 0 <= report_dict['precision'] <= 1
        assert 0 <= report_dict['recall'] <= 1
        assert 0 <= report_dict['value_detection_rate'] <= 1
        assert report_dict['true_positives'] == 1  # From our test data
        assert report_dict['false_positives'] == 2  # From our test data
    
    def test_evaluate_at_thresholds(self, sample_fraud_data):
        """Test threshold evaluation sweep."""
        y_true, y_pred, amounts = sample_fraud_data
        y_pred_proba = np.array([0.1, 0.2, 0.8, 0.3, 0.4, 0.9, 0.15, 0.25, 0.95, 0.05])
        
        thresholds = np.array([0.1, 0.5, 0.9])
        results = evaluate_at_thresholds(y_true, y_pred_proba, amounts, thresholds, fp_cost=100)
        
        # Check structure
        assert 'thresholds' in results
        assert 'precision' in results
        assert 'recall' in results
        assert 'vdr' in results
        assert 'net_savings' in results
        assert 'roi' in results
        
        # Check array lengths
        assert len(results['thresholds']) == len(thresholds)
        assert len(results['precision']) == len(thresholds)
        assert len(results['vdr']) == len(thresholds)
        
        # Lower thresholds should generally have higher recall
        assert results['recall'][0] >= results['recall'][1]  # 0.1 >= 0.5
        assert results['recall'][1] >= results['recall'][2]  # 0.5 >= 0.9
        
        # Higher thresholds should generally have higher precision
        assert results['precision'][2] >= results['precision'][0]  # 0.9 >= 0.1
    
    def test_cost_benefit_analysis(self, sample_fraud_data):
        """Test cost-benefit analysis."""
        y_true, y_pred, amounts = sample_fraud_data
        
        analysis = cost_benefit_analysis(y_true, y_pred, amounts, fp_cost=100, fn_cost_rate=1.0)
        
        # Check structure
        required_keys = [
            'fraud_prevented', 'investigation_cost', 'fraud_losses', 
            'net_benefit', 'benefit_cost_ratio', 'true_positives'
        ]
        for key in required_keys:
            assert key in analysis
        
        # Check values
        assert analysis['fraud_prevented'] == 1000.0  # TP amount
        assert analysis['investigation_cost'] == 200.0  # 2 FP × $100
        assert analysis['fraud_losses'] == 5000.0  # 1 FN × $5000
        
        expected_net_benefit = 1000.0 - 200.0 - 5000.0  # = -$4200 (net loss)
        assert np.isclose(analysis['net_benefit'], expected_net_benefit)
        
        # Test with different FN cost rate
        analysis_partial = cost_benefit_analysis(y_true, y_pred, amounts, 
                                               fp_cost=100, fn_cost_rate=0.5)
        assert analysis_partial['fraud_losses'] == 2500.0  # 50% of FN amount
    
    def test_edge_cases(self):
        """Test edge cases for all metrics."""
        # All zeros
        y_true_zeros = np.zeros(5)
        y_pred_zeros = np.zeros(5)
        amounts_zeros = np.ones(5) * 100
        
        assert value_detection_rate(y_true_zeros, y_pred_zeros, amounts_zeros) == 0.0
        assert net_savings(y_true_zeros, y_pred_zeros, amounts_zeros) == 0.0
        assert roi(y_true_zeros, y_pred_zeros, amounts_zeros) == 0.0
        
        # All ones (perfect detection of all-fraud dataset)
        y_true_ones = np.ones(5)
        y_pred_ones = np.ones(5)
        amounts_ones = np.ones(5) * 100
        
        assert value_detection_rate(y_true_ones, y_pred_ones, amounts_ones) == 1.0
        
        # Empty arrays
        empty = np.array([])
        assert value_detection_rate(empty, empty, empty) == 0.0
        assert net_savings(empty, empty, empty) == 0.0
        
    def test_metric_consistency(self, sample_fraud_data):
        """Test that metrics are consistent with each other."""
        y_true, y_pred, amounts = sample_fraud_data
        
        # Get report dictionary
        report = classification_report_fraud(y_true, y_pred, amounts, output_dict=True)
        
        # Individual metric calculations
        vdr = value_detection_rate(y_true, y_pred, amounts)
        savings = net_savings(y_true, y_pred, amounts)
        roi_val = roi(y_true, y_pred, amounts)
        
        # Should match report values
        assert np.isclose(report['value_detection_rate'], vdr)
        assert np.isclose(report['net_savings'], savings)
        assert np.isclose(report['roi'], roi_val)