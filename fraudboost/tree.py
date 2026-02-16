"""
Value-weighted decision tree for fraud detection gradient boosting.

Implements binary decision trees that split based on net savings maximization
rather than traditional information gain measures.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class SplitInfo:
    """Information about a tree split."""
    feature: int
    threshold: float
    gain: float
    left_value: float
    right_value: float
    n_samples_left: int
    n_samples_right: int
    missing_direction: int = -1  # -1 for left, 1 for right


class ValueWeightedDecisionTree:
    """
    Binary decision tree optimized for value-weighted fraud detection.
    
    Instead of maximizing information gain, splits are chosen to maximize
    net savings: (fraud_caught * amounts) - (false_positives * fp_cost)
    """
    
    def __init__(self,
                 max_depth: int = 6,
                 min_samples_leaf: int = 10,
                 min_samples_split: int = 20,
                 min_gain: float = 1e-7,
                 fp_cost: float = 100.0,
                 reg_lambda: float = 1.0):
        """
        Args:
            max_depth: Maximum tree depth
            min_samples_leaf: Minimum samples required in a leaf
            min_samples_split: Minimum samples required to split
            min_gain: Minimum gain required for a split
            fp_cost: Fixed cost of false positive (investigation cost)
            reg_lambda: L2 regularization strength for leaf values
        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.fp_cost = fp_cost
        self.reg_lambda = reg_lambda
        
        # Tree structure (will be built during fit)
        self.tree_ = {}
        self.feature_importances_ = None
    
    def fit(self, X: np.ndarray, gradients: np.ndarray, hessians: np.ndarray,
            amounts: Optional[np.ndarray] = None) -> 'ValueWeightedDecisionTree':
        """
        Fit the decision tree using gradients and hessians from boosting.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            gradients: Gradient values for each sample
            hessians: Hessian values for each sample
            amounts: Transaction amounts for value weighting
        """
        n_samples, n_features = X.shape
        
        if amounts is None:
            amounts = np.ones(n_samples)
        
        # Initialize feature importances
        self.feature_importances_ = np.zeros(n_features)
        
        # Build tree recursively
        self.tree_ = self._build_tree(X, gradients, hessians, amounts, depth=0)
        
        # Normalize feature importances
        if self.feature_importances_.sum() > 0:
            self.feature_importances_ /= self.feature_importances_.sum()
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict leaf values for input samples."""
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)
        
        for i in range(n_samples):
            predictions[i] = self._predict_sample(X[i], self.tree_)
        
        return predictions
    
    def _build_tree(self, X: np.ndarray, gradients: np.ndarray, hessians: np.ndarray,
                    amounts: np.ndarray, depth: int) -> Dict[str, Any]:
        """Recursively build the decision tree."""
        n_samples, n_features = X.shape
        
        # Compute leaf value using Newton-Raphson step with regularization
        leaf_value = -np.sum(gradients) / (np.sum(hessians) + self.reg_lambda)
        
        # Base cases for stopping
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or
            n_samples <= self.min_samples_leaf):
            return {
                'leaf': True,
                'value': leaf_value,
                'n_samples': n_samples
            }
        
        # Find best split
        best_split = self._find_best_split(X, gradients, hessians, amounts)
        
        if best_split is None or best_split.gain < self.min_gain:
            return {
                'leaf': True,
                'value': leaf_value,
                'n_samples': n_samples
            }
        
        # Update feature importance
        self.feature_importances_[best_split.feature] += best_split.gain
        
        # Split data
        left_mask, right_mask = self._split_data(X, best_split)
        
        # Recursively build left and right subtrees
        left_tree = self._build_tree(
            X[left_mask], gradients[left_mask], hessians[left_mask], 
            amounts[left_mask], depth + 1
        )
        
        right_tree = self._build_tree(
            X[right_mask], gradients[right_mask], hessians[right_mask], 
            amounts[right_mask], depth + 1
        )
        
        return {
            'leaf': False,
            'feature': best_split.feature,
            'threshold': best_split.threshold,
            'missing_direction': best_split.missing_direction,
            'left': left_tree,
            'right': right_tree,
            'n_samples': n_samples,
            'gain': best_split.gain
        }
    
    def _find_best_split(self, X: np.ndarray, gradients: np.ndarray, 
                        hessians: np.ndarray, amounts: np.ndarray) -> Optional[SplitInfo]:
        """Find the split that maximizes net savings gain."""
        n_samples, n_features = X.shape
        best_split = None
        best_gain = -np.inf
        
        # Current node statistics
        current_grad_sum = np.sum(gradients)
        current_hess_sum = np.sum(hessians)
        current_value = -current_grad_sum / (current_hess_sum + self.reg_lambda)
        
        for feature in range(n_features):
            feature_values = X[:, feature]
            
            # Handle missing values by trying both directions
            valid_mask = ~np.isnan(feature_values)
            if not np.any(valid_mask):
                continue
                
            valid_indices = np.where(valid_mask)[0]
            valid_values = feature_values[valid_indices]
            
            # Get unique thresholds to try
            unique_values = np.unique(valid_values)
            if len(unique_values) < 2:
                continue
            
            # Try thresholds between unique values
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2
            
            for threshold in thresholds:
                # Split based on threshold
                left_mask_valid = valid_values <= threshold
                right_mask_valid = ~left_mask_valid
                
                left_indices = valid_indices[left_mask_valid]
                right_indices = valid_indices[right_mask_valid]
                
                if len(left_indices) < self.min_samples_leaf or len(right_indices) < self.min_samples_leaf:
                    continue
                
                # Compute gain for this split
                gain = self._compute_split_gain(
                    gradients[left_indices], hessians[left_indices],
                    gradients[right_indices], hessians[right_indices],
                    current_grad_sum, current_hess_sum
                )
                
                if gain > best_gain:
                    best_gain = gain
                    
                    left_value = -np.sum(gradients[left_indices]) / (np.sum(hessians[left_indices]) + self.reg_lambda)
                    right_value = -np.sum(gradients[right_indices]) / (np.sum(hessians[right_indices]) + self.reg_lambda)
                    
                    # Determine missing direction based on which side has more samples
                    missing_direction = -1 if len(left_indices) > len(right_indices) else 1
                    
                    best_split = SplitInfo(
                        feature=feature,
                        threshold=threshold,
                        gain=gain,
                        left_value=left_value,
                        right_value=right_value,
                        n_samples_left=len(left_indices),
                        n_samples_right=len(right_indices),
                        missing_direction=missing_direction
                    )
        
        return best_split
    
    def _compute_split_gain(self, left_grad: np.ndarray, left_hess: np.ndarray,
                           right_grad: np.ndarray, right_hess: np.ndarray,
                           total_grad: float, total_hess: float) -> float:
        """
        Compute the gain from a split using gradient boosting criterion.
        
        Gain = 0.5 * [L²/(H+λ) + R²/(H+λ) - (L+R)²/(H+λ)]
        where L,R are gradient sums and H is hessian sum.
        """
        left_grad_sum = np.sum(left_grad)
        left_hess_sum = np.sum(left_hess)
        right_grad_sum = np.sum(right_grad)
        right_hess_sum = np.sum(right_hess)
        
        left_score = (left_grad_sum ** 2) / (left_hess_sum + self.reg_lambda)
        right_score = (right_grad_sum ** 2) / (right_hess_sum + self.reg_lambda)
        parent_score = (total_grad ** 2) / (total_hess + self.reg_lambda)
        
        gain = 0.5 * (left_score + right_score - parent_score)
        
        return gain
    
    def _split_data(self, X: np.ndarray, split: SplitInfo) -> Tuple[np.ndarray, np.ndarray]:
        """Split data based on the split info."""
        feature_values = X[:, split.feature]
        
        # Handle missing values
        valid_mask = ~np.isnan(feature_values)
        missing_mask = np.isnan(feature_values)
        
        left_mask = np.zeros(len(X), dtype=bool)
        right_mask = np.zeros(len(X), dtype=bool)
        
        # Valid values
        left_mask[valid_mask] = feature_values[valid_mask] <= split.threshold
        right_mask[valid_mask] = feature_values[valid_mask] > split.threshold
        
        # Missing values go to the side with more samples
        if split.missing_direction == -1:  # Left
            left_mask[missing_mask] = True
        else:  # Right
            right_mask[missing_mask] = True
        
        return left_mask, right_mask
    
    def _predict_sample(self, x: np.ndarray, tree: Dict[str, Any]) -> float:
        """Predict a single sample by traversing the tree."""
        if tree['leaf']:
            return tree['value']
        
        feature_value = x[tree['feature']]
        
        # Handle missing values
        if np.isnan(feature_value):
            if tree['missing_direction'] == -1:
                return self._predict_sample(x, tree['left'])
            else:
                return self._predict_sample(x, tree['right'])
        
        # Normal split
        if feature_value <= tree['threshold']:
            return self._predict_sample(x, tree['left'])
        else:
            return self._predict_sample(x, tree['right'])
    
    def get_leaf_values(self) -> np.ndarray:
        """Get all leaf values from the tree."""
        leaf_values = []
        self._collect_leaf_values(self.tree_, leaf_values)
        return np.array(leaf_values)
    
    def _collect_leaf_values(self, tree: Dict[str, Any], leaf_values: list):
        """Recursively collect leaf values."""
        if tree['leaf']:
            leaf_values.append(tree['value'])
        else:
            self._collect_leaf_values(tree['left'], leaf_values)
            self._collect_leaf_values(tree['right'], leaf_values)