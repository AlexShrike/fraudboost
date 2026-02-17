"""
Fraud-specific evaluation metrics that consider transaction values.

Traditional metrics like precision/recall treat all fraud equally.
These metrics weight fraud by transaction amount - catching $10K fraud
is more valuable than catching $10 fraud.
"""

import numpy as np
from typing import Optional, Dict, Any, Union
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import warnings


def value_detection_rate(y_true: np.ndarray, y_pred: np.ndarray, 
                        amounts: np.ndarray) -> float:
    """
    Value Detection Rate (VDR): Percentage of fraud value detected.
    
    VDR = (Sum of detected fraud amounts) / (Sum of all fraud amounts)
    
    This is more meaningful than traditional recall for fraud detection
    since catching high-value fraud is more important.
    
    Args:
        y_true: True binary labels (0=legit, 1=fraud)
        y_pred: Predicted binary labels  
        amounts: Transaction amounts
        
    Returns:
        VDR score between 0 and 1
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    amounts = np.asarray(amounts)
    
    # Find actual fraud cases
    fraud_mask = (y_true == 1)
    
    if not np.any(fraud_mask):
        warnings.warn("No fraud cases in y_true")
        return 0.0
    
    # Total fraud value
    total_fraud_value = np.sum(amounts[fraud_mask])
    
    if total_fraud_value == 0:
        return 0.0
    
    # Detected fraud value (true positives)
    detected_fraud_mask = fraud_mask & (y_pred == 1)
    detected_fraud_value = np.sum(amounts[detected_fraud_mask])
    
    return detected_fraud_value / total_fraud_value


def net_savings(y_true: np.ndarray, y_pred: np.ndarray, amounts: np.ndarray,
               fp_cost: float = 100.0) -> float:
    """
    Net savings from fraud detection system.
    
    Net Savings = (Value of detected fraud) - (Cost of investigations)
    
    Where:
    - Value of detected fraud = sum of amounts for true positives
    - Cost of investigations = number of false positives × fp_cost
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        amounts: Transaction amounts  
        fp_cost: Fixed cost per false positive investigation
        
    Returns:
        Net savings (can be negative if too many false positives)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    amounts = np.asarray(amounts)
    
    # True positives: correctly identified fraud
    tp_mask = (y_true == 1) & (y_pred == 1)
    fraud_caught_value = np.sum(amounts[tp_mask])
    
    # False positives: incorrectly flagged as fraud
    fp_mask = (y_true == 0) & (y_pred == 1)
    investigation_cost = np.sum(fp_mask) * fp_cost
    
    return fraud_caught_value - investigation_cost


def roi(y_true: np.ndarray, y_pred: np.ndarray, amounts: np.ndarray,
        fp_cost: float = 100.0) -> float:
    """
    Return on Investment for fraud detection system.
    
    ROI = (Net Savings) / (Total Investigation Cost)
    
    A ROI > 1.0 means the system is profitable.
    A ROI < 1.0 means investigation costs exceed recovered fraud.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels  
        amounts: Transaction amounts
        fp_cost: Fixed cost per false positive investigation
        
    Returns:
        ROI ratio (>1 is good, <1 is bad)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    amounts = np.asarray(amounts)
    
    # Total investigations (all positive predictions)
    total_investigations = np.sum(y_pred == 1)
    
    if total_investigations == 0:
        return 0.0  # No investigations, no ROI
    
    investigation_cost = total_investigations * fp_cost
    
    # Net savings
    savings = net_savings(y_true, y_pred, amounts, fp_cost)
    
    return (savings + investigation_cost) / investigation_cost


def precision_at_dollar_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray,
                                 amounts: np.ndarray, dollar_threshold: float) -> float:
    """
    Precision when only considering high-value transactions.
    
    Only evaluate precision on transactions above dollar_threshold.
    Useful for focusing on high-value fraud detection performance.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        amounts: Transaction amounts
        dollar_threshold: Minimum transaction amount to consider
        
    Returns:
        Precision score for high-value transactions
    """
    # Filter to high-value transactions
    high_value_mask = amounts >= dollar_threshold
    
    if not np.any(high_value_mask):
        warnings.warn(f"No transactions above ${dollar_threshold}")
        return 0.0
    
    y_true_filtered = y_true[high_value_mask]
    y_pred_filtered = (y_pred_proba[high_value_mask] >= 0.5).astype(int)
    
    return precision_score(y_true_filtered, y_pred_filtered, zero_division=0)


def classification_report_fraud(y_true: np.ndarray, y_pred: np.ndarray, 
                               amounts: np.ndarray, fp_cost: float = 100.0,
                               output_dict: bool = False) -> Union[str, Dict]:
    """
    Comprehensive fraud detection classification report.
    
    Includes traditional metrics plus fraud-specific value metrics:
    - Precision, Recall, F1 (traditional)
    - VDR (Value Detection Rate)  
    - Net Savings
    - ROI (Return on Investment)
    - Dollar amounts caught vs missed
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        amounts: Transaction amounts
        fp_cost: Cost per false positive investigation
        output_dict: If True, return dict instead of string
        
    Returns:
        Formatted report string or dictionary
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    amounts = np.asarray(amounts)
    
    # Traditional metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Value metrics
    vdr = value_detection_rate(y_true, y_pred, amounts)
    net_savings_value = net_savings(y_true, y_pred, amounts, fp_cost)
    roi_value = roi(y_true, y_pred, amounts, fp_cost)
    
    # Dollar amounts
    tp_mask = (y_true == 1) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)
    
    fraud_caught = np.sum(amounts[tp_mask])
    fraud_missed = np.sum(amounts[fn_mask])
    total_fraud = fraud_caught + fraud_missed
    
    investigation_cost = fp * fp_cost
    
    if output_dict:
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'value_detection_rate': vdr,
            'net_savings': net_savings_value,
            'roi': roi_value,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'fraud_caught_dollars': fraud_caught,
            'fraud_missed_dollars': fraud_missed,
            'total_fraud_dollars': total_fraud,
            'investigation_cost': investigation_cost
        }
    
    # Format string report
    report = f"""
Fraud Detection Classification Report
=====================================

Traditional Metrics:
  Precision:           {precision:.4f}
  Recall:              {recall:.4f}  
  F1-Score:            {f1:.4f}

Value Metrics:
  Value Detection Rate: {vdr:.4f} ({vdr*100:.1f}% of fraud $ caught)
  Net Savings:         ${net_savings_value:,.2f}
  ROI:                 {roi_value:.2f}x

Confusion Matrix:
  True Positives:      {tp:,} transactions
  False Positives:     {fp:,} transactions  
  True Negatives:      {tn:,} transactions
  False Negatives:     {fn:,} transactions

Dollar Impact:
  Fraud Caught:        ${fraud_caught:,.2f}
  Fraud Missed:        ${fraud_missed:,.2f}
  Total Fraud:         ${total_fraud:,.2f}
  Investigation Cost:  ${investigation_cost:,.2f}
"""
    
    return report.strip()


def evaluate_at_thresholds(y_true: np.ndarray, y_pred_proba: np.ndarray,
                          amounts: np.ndarray, thresholds: Optional[np.ndarray] = None,
                          fp_cost: float = 100.0) -> Dict[str, np.ndarray]:
    """
    Evaluate fraud detection performance across different decision thresholds.
    
    Useful for creating ROC curves, precision-recall curves, and finding
    optimal operating points for different business objectives.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities (0-1)
        amounts: Transaction amounts
        thresholds: Decision thresholds to evaluate (default: 0.01 to 0.99)
        fp_cost: Cost per false positive investigation
        
    Returns:
        Dictionary with arrays of metrics at each threshold
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)
    
    n_thresholds = len(thresholds)
    
    # Initialize result arrays
    results = {
        'thresholds': thresholds,
        'precision': np.zeros(n_thresholds),
        'recall': np.zeros(n_thresholds), 
        'f1': np.zeros(n_thresholds),
        'vdr': np.zeros(n_thresholds),
        'net_savings': np.zeros(n_thresholds),
        'roi': np.zeros(n_thresholds),
        'tp_count': np.zeros(n_thresholds, dtype=int),
        'fp_count': np.zeros(n_thresholds, dtype=int),
        'tn_count': np.zeros(n_thresholds, dtype=int),
        'fn_count': np.zeros(n_thresholds, dtype=int)
    }
    
    for i, threshold in enumerate(thresholds):
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Traditional metrics
        results['precision'][i] = precision_score(y_true, y_pred, zero_division=0)
        results['recall'][i] = recall_score(y_true, y_pred, zero_division=0)
        results['f1'][i] = f1_score(y_true, y_pred, zero_division=0)
        
        # Value metrics  
        results['vdr'][i] = value_detection_rate(y_true, y_pred, amounts)
        results['net_savings'][i] = net_savings(y_true, y_pred, amounts, fp_cost)
        results['roi'][i] = roi(y_true, y_pred, amounts, fp_cost)
        
        # Confusion matrix counts
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        results['tp_count'][i] = tp
        results['fp_count'][i] = fp  
        results['tn_count'][i] = tn
        results['fn_count'][i] = fn
    
    return results


def cost_benefit_analysis(y_true: np.ndarray, y_pred: np.ndarray, amounts: np.ndarray,
                         fp_cost: float = 100.0, fn_cost_rate: float = 1.0) -> Dict[str, float]:
    """
    Complete cost-benefit analysis of fraud detection system.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        amounts: Transaction amounts  
        fp_cost: Fixed cost per false positive
        fn_cost_rate: Cost rate for false negatives (0-1, where 1 = full amount lost)
        
    Returns:
        Dictionary with detailed cost breakdown
    """
    # Confusion matrix masks
    tp_mask = (y_true == 1) & (y_pred == 1)
    fp_mask = (y_true == 0) & (y_pred == 1) 
    fn_mask = (y_true == 1) & (y_pred == 0)
    tn_mask = (y_true == 0) & (y_pred == 0)
    
    # Benefits (fraud prevented)
    fraud_prevented = np.sum(amounts[tp_mask])
    
    # Costs
    investigation_cost = np.sum(fp_mask) * fp_cost
    fraud_losses = np.sum(amounts[fn_mask]) * fn_cost_rate
    
    # Net benefit
    net_benefit = fraud_prevented - investigation_cost - fraud_losses
    
    return {
        'fraud_prevented': fraud_prevented,
        'investigation_cost': investigation_cost,  
        'fraud_losses': fraud_losses,
        'net_benefit': net_benefit,
        'benefit_cost_ratio': fraud_prevented / (investigation_cost + fraud_losses) if (investigation_cost + fraud_losses) > 0 else np.inf,
        'true_positives': int(np.sum(tp_mask)),
        'false_positives': int(np.sum(fp_mask)),
        'false_negatives': int(np.sum(fn_mask)),
        'true_negatives': int(np.sum(tn_mask))
    }


# Additional helper functions for compatibility
def calculate_precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """
    Calculate precision, recall, and F1 score.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        
    Returns:
        Tuple of (precision, recall, f1)
    """
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return precision, recall, f1


def calculate_value_detection_rate(y_true: np.ndarray, y_pred: np.ndarray, amounts: np.ndarray) -> float:
    """Alias for value_detection_rate function."""
    return value_detection_rate(y_true, y_pred, amounts)


def calculate_net_savings(y_true: np.ndarray, y_pred: np.ndarray, amounts: np.ndarray, fp_cost: float = 100.0) -> float:
    """Alias for net_savings function."""
    return net_savings(y_true, y_pred, amounts, fp_cost)


def find_optimal_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray, metric: str = 'f1') -> float:
    """
    Find optimal classification threshold for a given metric.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities  
        metric: Metric to optimize ('f1', 'precision', 'recall')
        
    Returns:
        Optimal threshold between 0 and 1
    """
    from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score
    
    thresholds = np.linspace(0.01, 0.99, 100)
    best_score = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
            
    return best_threshold